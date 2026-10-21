//! The capabilities section: [`CapabilitySet`] to bytes and back.
//!
//! # Layout
//!
//! ```text
//! u32   schema
//! str   backend name
//! u32   pattern count, then per pattern:
//!   str pattern id
//!   u32 element, u32 quant kind, u32 block size, u8 symmetric
//!   u32 shape class kind, u8 rank
//!   u32 support level
//!   u8  has_reason, then str if set
//! u32   compute unit count, then per unit:
//!   u32 index
//!   str name
//!   str vendor kind
//!   u8  has_memory, then u64 if set
//! u64   max_allocation_bytes, u64::MAX for unset
//! u64   max_graph_bytes,      u64::MAX for unset
//! u32   max_rank,             u32::MAX for unset
//! u64   max_dim,              u64::MAX for unset
//! u64   max_nodes,            u64::MAX for unset
//! u32   dtype mask
//! u8    accepts_unpartitioned
//! ```
//!
//! Limits are *not* length-prefixed because there are five of them and they are
//! the part of the blob a host reads on every load. Each is a fixed-width integer
//! with an explicit "unset" sentinel rather than an option tag: an unset limit
//! encodes as `u64::MAX`, which is not a limit any device has, so a host reading
//! an older schema still gets the right answer for the fields it knows.
//!
//! # One dtype decoder, shared
//!
//! The `(element, quant kind, block size, symmetric)` quadruple is decoded by
//! [`graph::dtype_from_parts`](super::graph::dtype_from_parts) and nothing else,
//! because the graph section encodes the same four fields and two decoders is two
//! chances for them to disagree — and that disagreement would be a silent
//! numeric bug rather than a parse error.

use super::graph::{dtype_from_parts, shape_class_to_raw};
use super::{Reader, WireError, Writer};
use crate::raw::{StrataQuantKind, StrataShapeClassKind, StrataSupportLevel};
use ferrite_strata::capabilities::{
    CapabilitySet, ComputeUnit, DTypeMask, DeclaredLimits, PatternSupport, SupportLevel,
};
use ferrite_strata::dtype::DType;
use ferrite_strata::shape::ShapeClass;

/// The capability schema this encoder writes.
pub const CAPS_SCHEMA: u32 = 1;

/// Smallest bytes one pattern occupies: an empty id(4) + dtype(17) + shape(5) +
/// level(4) + has_reason(1).
const MIN_PATTERN_BYTES: usize = 4 + 17 + 5 + 4 + 1;

/// Smallest bytes one compute unit occupies: index(4) + two empty strings(8).
const MIN_UNIT_BYTES: usize = 4 + 8;

/// The "no limit declared" sentinel.
const UNSET: u64 = u64::MAX;

/// Encode a capability set.
#[must_use]
pub fn encode(caps: &CapabilitySet) -> Vec<u8> {
    let mut w = Writer::new();
    w.u32(CAPS_SCHEMA);
    w.string(&caps.backend_name);

    w.u32(caps.patterns.len() as u32);
    for pattern in &caps.patterns {
        w.string(&pattern.pattern_id);
        let (element, quant, block_size, symmetric) = super::graph::dtype_to_parts(pattern.dtype);
        w.u32(element);
        w.u32(quant);
        w.u32(block_size);
        w.u8(symmetric);
        let class = shape_class_to_raw(pattern.shape_class);
        w.u32(class.kind as u32);
        w.u8(class.rank);
        w.u32(level_to_raw(pattern.level) as u32);
        match &pattern.reason {
            Some(reason) => {
                w.bool(true);
                w.string(reason);
            }
            None => w.bool(false),
        }
    }

    w.u32(caps.compute_units.len() as u32);
    for unit in &caps.compute_units {
        w.u32(unit.index);
        w.string(&unit.name);
        w.string(&unit.vendor_kind);
        match unit.memory_bytes {
            Some(bytes) => {
                w.bool(true);
                w.u64(bytes);
            }
            None => w.bool(false),
        }
    }

    w.u64(caps.limits.max_allocation_bytes.unwrap_or(UNSET));
    w.u64(caps.limits.max_graph_bytes.unwrap_or(UNSET));
    w.u32(caps.limits.max_rank.unwrap_or(u32::MAX));
    w.u64(caps.limits.max_dim.unwrap_or(UNSET));
    w.u64(caps.limits.max_nodes.unwrap_or(UNSET));
    w.u32(caps.dtype_mask.to_bits());
    w.bool(caps.accepts_unpartitioned);

    w.finish()
}

/// Decode a capability blob.
///
/// # Errors
///
/// [`WireError`] for a malformed blob. An unknown support level is an error
/// rather than a refusal, and the reason is the asymmetry that matters: "this
/// newer plugin says it can do something at a level this build has never heard
/// of" is not the same answer as "it refuses", and guessing `Fallback` would
/// make the host offer a subgraph it has no evidence the plugin can run.
pub fn decode(bytes: &[u8]) -> Result<CapabilitySet, WireError> {
    let mut r = Reader::new(bytes);
    let schema = r.u32()?;
    if schema != CAPS_SCHEMA {
        return Err(WireError::UnsupportedFormat {
            major: schema,
            minor: 0,
        });
    }
    let backend_name = r.string()?;

    let pattern_count = r.count(MIN_PATTERN_BYTES)?;
    let mut patterns = Vec::with_capacity(pattern_count);
    for _ in 0..pattern_count {
        let pattern_id = r.string()?;
        let element = r.u32()?;
        let quant_kind = r.u32()?;
        let block_size = r.u32()?;
        let symmetric = r.u8()?;
        let dtype = dtype_from_parts(element, quant_kind, block_size, symmetric)?;
        let shape_class = read_shape_class(&mut r)?;
        let level_raw = r.u32()?;
        let level = level_from_raw(level_raw).ok_or(WireError::UnknownTag {
            field: "support-level",
            tag: level_raw,
        })?;
        let reason = if r.u8()? != 0 {
            Some(r.string()?)
        } else {
            None
        };
        patterns.push(PatternSupport {
            pattern_id,
            dtype,
            shape_class,
            level,
            reason,
        });
    }

    let unit_count = r.count(MIN_UNIT_BYTES)?;
    let mut compute_units = Vec::with_capacity(unit_count);
    for _ in 0..unit_count {
        let index = r.u32()?;
        let name = r.string()?;
        let vendor_kind = r.string()?;
        let memory_bytes = if r.u8()? != 0 { Some(r.u64()?) } else { None };
        compute_units.push(ComputeUnit {
            index,
            name,
            vendor_kind,
            memory_bytes,
        });
    }

    let max_allocation_bytes = opt_u64(r.u64()?);
    let max_graph_bytes = opt_u64(r.u64()?);
    let max_rank_raw = r.u32()?;
    let max_rank = (max_rank_raw != u32::MAX).then_some(max_rank_raw);
    let max_dim = opt_u64(r.u64()?);
    let max_nodes = opt_u64(r.u64()?);
    let dtype_mask = DTypeMask::from_bits(r.u32()?);
    let accepts_unpartitioned = r.u8()? != 0;

    Ok(CapabilitySet {
        backend_name,
        patterns,
        compute_units,
        limits: DeclaredLimits {
            max_allocation_bytes,
            max_graph_bytes,
            max_rank,
            max_dim,
            max_nodes,
        },
        dtype_mask,
        accepts_unpartitioned,
    })
}

fn opt_u64(raw: u64) -> Option<u64> {
    (raw != UNSET).then_some(raw)
}

fn read_shape_class(r: &mut Reader<'_>) -> Result<ShapeClass, WireError> {
    let kind = r.u32()?;
    let rank = r.u8()?;
    let kind = match kind {
        k if k == StrataShapeClassKind::Scalar as u32 => ShapeClass::Scalar,
        k if k == StrataShapeClassKind::Empty as u32 => ShapeClass::Empty,
        k if k == StrataShapeClassKind::OuterProduct as u32 => ShapeClass::OuterProduct,
        k if k == StrataShapeClassKind::Vector as u32 => ShapeClass::Vector,
        k if k == StrataShapeClassKind::Matrix as u32 => ShapeClass::Matrix,
        k if k == StrataShapeClassKind::Batch3 as u32 => ShapeClass::Batch3,
        k if k == StrataShapeClassKind::Batch4 as u32 => ShapeClass::Batch4,
        k if k == StrataShapeClassKind::HigherRank as u32 => ShapeClass::HigherRank { rank },
        other => {
            return Err(WireError::UnknownTag {
                field: "shape-class",
                tag: other,
            });
        }
    };
    Ok(kind)
}

fn level_to_raw(level: SupportLevel) -> StrataSupportLevel {
    match level {
        SupportLevel::Refuse => StrataSupportLevel::Refuse,
        SupportLevel::Fallback => StrataSupportLevel::Fallback,
        SupportLevel::Fused => StrataSupportLevel::Fused,
    }
}

/// Decode a raw support-level tag, or `None` for a tag this build does not know.
///
/// Public because the pull query returns the raw tag: a plugin answering with a
/// level from a newer header is a case the host has to recognise and refuse,
/// not a case it can round-trip through a closed Rust enum.
#[must_use]
pub fn level_from_raw(raw: u32) -> Option<SupportLevel> {
    match raw {
        v if v == StrataSupportLevel::Refuse as u32 => Some(SupportLevel::Refuse),
        v if v == StrataSupportLevel::Fallback as u32 => Some(SupportLevel::Fallback),
        v if v == StrataSupportLevel::Fused as u32 => Some(SupportLevel::Fused),
        _ => None,
    }
}

/// The four ABI fields of a dtype, for a plugin building a capability blob by
/// hand.
#[must_use]
pub fn dtype_to_parts(dtype: DType) -> (u32, u32, u32, u8) {
    super::graph::dtype_to_parts(dtype)
}

/// The quant-kind tag for a plain dtype, for a caller writing a blob by hand.
pub const QUANT_NONE: u32 = StrataQuantKind::None as u32;
