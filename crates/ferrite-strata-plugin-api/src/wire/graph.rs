//! The graph section: [`Graph`] to bytes and back.
//!
//! # Layout
//!
//! ```text
//! u32   schema                 the graph schema version, not the wire format's
//! str   name
//! u32   value_count
//!   per value:
//!     u32   element
//!     u32   quant kind
//!     u32   block size
//!     u8    symmetric
//!     u32   rank
//!     u64 * rank              dims
//!     u64   producer           u64::MAX for a graph input, else index | PRODUCER_FLAG
//!     u32   output slot
//!     str   name
//! u32   node_count
//!   per node:
//!     u32   op kind
//!     ...   op payload        typed, see `write_op`
//!     u32   input count, then u32 each
//!     u32   output count, then u32 each
//!     u32   attr count
//!       per attr: str key, u8 tag, payload
//!     str   name
//! u32   output_count, then u32 each
//! ```
//!
//! Values come before nodes so a reader can validate every id against the value
//! count before it meets a node that uses one, and nodes are in the topological
//! order the graph already guarantees — which is what lets the reader reject a
//! use-before-definition rather than accepting a cycle.
//!
//! # Op payloads are typed, not attributes
//!
//! `Op` carries data for eight of its variants (`Softmax`'s axis,
//! `LayerNorm`'s gamma/beta/eps, `Linear`'s weight and optional bias, the target
//! shapes of `Reshape` and `Expand`, `Transpose`'s permutation, `Slice`'s four
//! integers). The obvious encoding is to shovel those into `Attributes` on the
//! way out and read them back on the way in, which keeps the op field fixed-width
//! — and is wrong, because the core crate's own contract is that attributes are
//! the *parameter* channel and an unknown key is skipped rather than a parse
//! failure. A payload in an attribute is a payload a newer Strata can silently
//! drop. So each op writes its own typed payload, and the tag stays the
//! discriminator.

use super::{Reader, WireError, Writer};
use crate::raw::{
    StrataDType, StrataElementType, StrataOpKind, StrataQuantKind, StrataQuantSpec,
    StrataShapeClass, StrataShapeClassKind,
};
use ferrite_strata::attrs::{AttrKey, AttrValue, Attributes};
use ferrite_strata::dtype::{DType, ElementType, QuantSpec};
use ferrite_strata::graph::{Graph, Node, Op, Value};
use ferrite_strata::id::{NodeId, StableName, ValueId};
use ferrite_strata::shape::{Shape, ShapeClass};

/// The graph schema this encoder writes.
///
/// Separate from the wire format version on purpose: the wire *framing* can gain
/// a section without the graph inside it changing, and a bump here means the
/// field list below moved.
pub const GRAPH_SCHEMA: u32 = 1;

/// Smallest number of bytes one encoded value can occupy: dtype(13) + rank(4) +
/// producer(8) + slot(4) + a one-byte name length. Used to bound a count from
/// an untrusted blob before allocating.
const MIN_VALUE_BYTES: usize = 13 + 4 + 8 + 4 + 1;

/// Smallest number of bytes one encoded node can occupy: op(4) + two empty count
/// fields(8) + an empty attr count(4) + a one-byte name.
const MIN_NODE_BYTES: usize = 4 + 8 + 4 + 1;

/// The high bit of the producer field, so a producer is distinguishable from a
/// node index of 0 without spending a byte on an enum.
///
/// Node indices are bounded by the arena and are nowhere near `2^63`, so the bit
/// is free.
const PRODUCER_FLAG: u64 = 1 << 63;

/// The sentinel for "this value is a graph input".
const NO_PRODUCER: u64 = u64::MAX;

/// Attribute value tags.
///
/// Private rather than public: the encoder and decoder here are the only things
/// that should be writing these numbers, because a plugin that hand-rolls one has
/// no way to learn that adding a variant changes the others.
mod attr_tag {
    pub(super) const INT: u8 = 0;
    pub(super) const FLOAT: u8 = 1;
    pub(super) const BOOL: u8 = 2;
    pub(super) const STR: u8 = 3;
    pub(super) const INTS: u8 = 4;
}

/// Encode a graph into the graph section's bytes.
#[must_use]
pub fn encode(graph: &Graph) -> Vec<u8> {
    let mut w = Writer::new();
    w.u32(GRAPH_SCHEMA);
    w.string(graph.name());

    w.u32(graph.value_count() as u32);
    for value in graph.values() {
        let dtype = dtype_to_raw(value.dtype);
        w.u32(dtype.element as u32);
        w.u32(dtype.quant.kind as u32);
        w.u32(dtype.quant.block_size);
        w.u8(dtype.quant.symmetric);
        w.u32(value.shape.rank() as u32);
        for dim in value.shape.dims() {
            w.u64(*dim as u64);
        }
        w.u64(match value.producer {
            Some(node) => node.0 as u64 | PRODUCER_FLAG,
            None => NO_PRODUCER,
        });
        w.u32(value.output_slot as u32);
        w.string(value.name.as_str());
    }

    w.u32(graph.node_count() as u32);
    for node in graph.nodes() {
        write_op(&mut w, &node.op);
        w.u32(node.inputs.len() as u32);
        for id in &node.inputs {
            w.u32(id.0);
        }
        w.u32(node.outputs.len() as u32);
        for id in &node.outputs {
            w.u32(id.0);
        }
        let entries = node.attrs.entries();
        w.u32(entries.len() as u32);
        for (key, value) in entries {
            w.string(key.as_str());
            match value {
                AttrValue::Int(v) => {
                    w.u8(attr_tag::INT);
                    w.u64(*v as u64);
                }
                AttrValue::Float(v) => {
                    w.u8(attr_tag::FLOAT);
                    w.u32(v.to_bits());
                }
                AttrValue::Bool(v) => {
                    w.u8(attr_tag::BOOL);
                    w.bool(*v);
                }
                AttrValue::Str(v) => {
                    w.u8(attr_tag::STR);
                    w.string(v);
                }
                AttrValue::Ints(v) => {
                    w.u8(attr_tag::INTS);
                    w.u32(v.len() as u32);
                    for item in v {
                        w.u64(*item as u64);
                    }
                }
            }
        }
        w.string(node.name.as_str());
    }

    w.u32(graph.outputs().len() as u32);
    for id in graph.outputs() {
        w.u32(id.0);
    }

    w.finish()
}

fn write_op(w: &mut Writer, op: &Op) {
    w.u32(op_to_raw(op) as u32);
    match op {
        Op::Input | Op::Add | Op::Mul | Op::MatMul | Op::Log | Op::Exp | Op::Gelu => {}
        Op::Softmax { axis } => w.u64(*axis as u64),
        Op::LayerNorm { gamma, beta, eps } => {
            w.u32(gamma.0);
            w.u32(beta.0);
            w.u32(eps.to_bits());
        }
        Op::Linear { weight, bias } => {
            w.u32(weight.0);
            match bias {
                Some(id) => {
                    w.bool(true);
                    w.u32(id.0);
                }
                None => w.bool(false),
            }
        }
        Op::Reshape { shape } | Op::Expand { shape } => {
            w.u32(shape.rank() as u32);
            for dim in shape.dims() {
                w.u64(*dim as u64);
            }
        }
        Op::Transpose { permutation } => {
            w.u32(permutation.len() as u32);
            for axis in permutation {
                w.u64(*axis as u64);
            }
        }
        Op::Slice {
            axis,
            start,
            end,
            stride,
        } => {
            w.u64(*axis as u64);
            w.u64(*start as u64);
            w.u64(*end as u64);
            w.u64(*stride as u64);
        }
        Op::Custom { pattern_id } => w.string(pattern_id),
    }
}

/// Decode the graph section.
///
/// # Errors
///
/// [`WireError`] for a malformed blob, or [`WireError::Graph`] if the arenas it
/// decodes to do not form a well-formed graph. The second is the important one:
/// every field can be individually well-typed and the graph still nonsense, and
/// handing that to a backend would be handing it something no backend could
/// have produced.
pub fn decode(bytes: &[u8]) -> Result<Graph, WireError> {
    let mut r = Reader::new(bytes);
    let schema = r.u32()?;
    if schema != GRAPH_SCHEMA {
        return Err(WireError::UnsupportedFormat {
            major: schema,
            minor: 0,
        });
    }
    let name = r.string()?;

    let value_count = r.count(MIN_VALUE_BYTES)?;
    let mut values = Vec::with_capacity(value_count);
    for _ in 0..value_count {
        let dtype = read_dtype(&mut r)?;
        let rank = usize::try_from(r.u32()?).unwrap_or(usize::MAX);
        if rank > ferrite_strata::shape::MAX_RANK {
            return Err(WireError::Shape(
                ferrite_strata::shape::ShapeError::RankTooHigh {
                    rank,
                    limit: ferrite_strata::shape::MAX_RANK,
                },
            ));
        }
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(usize::try_from(r.u64()?).unwrap_or(usize::MAX));
        }
        let shape = Shape::new(&dims).map_err(WireError::Shape)?;

        let raw_producer = r.u64()?;
        let producer = if raw_producer == NO_PRODUCER {
            None
        } else {
            Some(NodeId((raw_producer & !PRODUCER_FLAG) as u32))
        };
        let output_slot = r.u32()? as usize;
        let value_name = r.string()?;
        values.push(Value {
            dtype,
            shape,
            producer,
            output_slot,
            name: StableName::new(value_name),
        });
    }

    let node_count = r.count(MIN_NODE_BYTES)?;
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        let op = read_op(&mut r)?;
        let inputs = read_ids(&mut r)?;
        let outputs = read_ids(&mut r)?;

        let attr_count = r.count(2)?;
        let mut attrs = Attributes::new();
        for _ in 0..attr_count {
            let key = AttrKey::new(&r.string()?).map_err(WireError::Attr)?;
            let value = read_attr(&mut r)?;
            // A duplicate or rejected key is skipped rather than fatal, matching
            // the core rule that an unknown key is skipped rather than a parse
            // failure: a newer host must not be able to reject an older plugin's
            // blob over an attribute it does not recognise.
            let _ = attrs.set(key, value);
        }

        nodes.push(Node {
            op,
            inputs,
            outputs,
            attrs,
            name: StableName::new(r.string()?),
        });
    }

    let outputs = read_ids(&mut r)?;

    Graph::from_parts(nodes, values, outputs, name).map_err(WireError::Graph)
}

fn read_op(r: &mut Reader<'_>) -> Result<Op, WireError> {
    let raw = r.u32()?;
    let kind = StrataOpKind::from_u32(raw).ok_or(WireError::UnknownTag {
        field: "op",
        tag: raw,
    })?;
    let op = match kind {
        StrataOpKind::Input => Op::Input,
        StrataOpKind::Add => Op::Add,
        StrataOpKind::Mul => Op::Mul,
        StrataOpKind::MatMul => Op::MatMul,
        StrataOpKind::Log => Op::Log,
        StrataOpKind::Exp => Op::Exp,
        StrataOpKind::Gelu => Op::Gelu,
        StrataOpKind::Softmax => Op::Softmax {
            axis: r.u64()? as i64,
        },
        StrataOpKind::LayerNorm => Op::LayerNorm {
            gamma: ValueId(r.u32()?),
            beta: ValueId(r.u32()?),
            eps: f32::from_bits(r.u32()?),
        },
        StrataOpKind::Linear => {
            let weight = ValueId(r.u32()?);
            let has_bias = r.u8()? != 0;
            let bias = if has_bias {
                Some(ValueId(r.u32()?))
            } else {
                None
            };
            Op::Linear { weight, bias }
        }
        StrataOpKind::Reshape => Op::Reshape {
            shape: read_shape(r)?,
        },
        StrataOpKind::Expand => Op::Expand {
            shape: read_shape(r)?,
        },
        StrataOpKind::Transpose => {
            let len = r.count(8)?;
            let mut permutation = Vec::with_capacity(len);
            for _ in 0..len {
                permutation.push(r.u64()? as i64);
            }
            Op::Transpose { permutation }
        }
        StrataOpKind::Slice => Op::Slice {
            axis: r.u64()? as i64,
            start: r.u64()? as i64,
            end: r.u64()? as i64,
            stride: r.u64()? as i64,
        },
        StrataOpKind::Custom => Op::Custom {
            pattern_id: r.string()?,
        },
    };
    Ok(op)
}

fn read_shape(r: &mut Reader<'_>) -> Result<Shape, WireError> {
    let rank = usize::try_from(r.u32()?).unwrap_or(usize::MAX);
    if rank > ferrite_strata::shape::MAX_RANK {
        return Err(WireError::Shape(
            ferrite_strata::shape::ShapeError::RankTooHigh {
                rank,
                limit: ferrite_strata::shape::MAX_RANK,
            },
        ));
    }
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(usize::try_from(r.u64()?).unwrap_or(usize::MAX));
    }
    Shape::new(&dims).map_err(WireError::Shape)
}

fn read_ids(r: &mut Reader<'_>) -> Result<Vec<ValueId>, WireError> {
    let count = r.count(4)?;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(ValueId(r.u32()?));
    }
    Ok(out)
}

fn read_attr(r: &mut Reader<'_>) -> Result<AttrValue, WireError> {
    let tag = r.u8()?;
    let value = match tag {
        attr_tag::INT => AttrValue::Int(r.u64()? as i64),
        attr_tag::FLOAT => AttrValue::Float(f32::from_bits(r.u32()?)),
        attr_tag::BOOL => AttrValue::Bool(r.u8()? != 0),
        attr_tag::STR => AttrValue::Str(r.string()?),
        attr_tag::INTS => {
            let count = r.count(8)?;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(r.u64()? as i64);
            }
            AttrValue::Ints(items)
        }
        other => {
            return Err(WireError::UnknownTag {
                field: "attr",
                tag: other as u32,
            });
        }
    };
    Ok(value)
}

fn read_dtype(r: &mut Reader<'_>) -> Result<DType, WireError> {
    let element = r.u32()?;
    let quant_kind = r.u32()?;
    let block_size = r.u32()?;
    let symmetric = r.u8()?;
    dtype_from_parts(element, quant_kind, block_size, symmetric)
}

/// Decode the four ABI fields of a dtype.
///
/// Shared with the capability section, which encodes exactly the same four
/// fields. One decoder rather than two because two decoders is two chances for
/// them to disagree, and a disagreement here is a silently mis-scaled tensor
/// rather than a parse error.
///
/// The inverse of [`dtype_to_parts`], and public for the same reason: a vendor
/// writing a capability blob by hand needs both halves.
///
/// # Errors
///
/// [`WireError::UnknownTag`] for an element or quant-kind tag this build does
/// not know, or [`WireError::BareIntegerDtype`] for an integer tensor with no
/// scaling, which has no defined numeric meaning.
pub fn dtype_from_parts(
    element_raw: u32,
    quant_kind: u32,
    block_size: u32,
    symmetric: u8,
) -> Result<DType, WireError> {
    let element = element_from_u32(element_raw)?;
    let quant = match quant_kind {
        k if k == StrataQuantKind::None as u32 => QuantSpec::None,
        k if k == StrataQuantKind::Block as u32 => QuantSpec::Block {
            block_size,
            symmetric: symmetric != 0,
        },
        k if k == StrataQuantKind::Microscaling as u32 => QuantSpec::Microscaling { block_size },
        other => {
            return Err(WireError::UnknownTag {
                field: "quant-kind",
                tag: other,
            });
        }
    };
    // A raw-integer tensor with no scaling has no defined numeric meaning, so the
    // core crate's own constructor is not the right door. Rejecting it here means
    // a plugin cannot hand the host a dtype the rest of the system has no way to
    // interpret.
    if element.is_int() && quant == QuantSpec::None {
        return Err(WireError::BareIntegerDtype {
            element: element.name(),
        });
    }
    Ok(DType { element, quant })
}

/// The four ABI fields of a dtype, as a tuple, for a caller writing a blob by
/// hand.
#[must_use]
pub fn dtype_to_parts(dtype: DType) -> (u32, u32, u32, u8) {
    let raw = dtype_to_raw(dtype);
    (
        raw.element as u32,
        raw.quant.kind as u32,
        raw.quant.block_size,
        raw.quant.symmetric,
    )
}

/// Map a core dtype to its ABI form, for a caller that wants to hand one to a
/// plugin in a struct rather than in a blob.
#[must_use]
pub fn dtype_to_raw(dtype: DType) -> StrataDType {
    StrataDType {
        element: element_to_raw(dtype.element),
        quant: StrataQuantSpec {
            kind: match dtype.quant {
                QuantSpec::None => StrataQuantKind::None,
                QuantSpec::Block { .. } => StrataQuantKind::Block,
                QuantSpec::Microscaling { .. } => StrataQuantKind::Microscaling,
            },
            block_size: dtype.quant.block_size().unwrap_or(0),
            symmetric: u8::from(matches!(
                dtype.quant,
                QuantSpec::Block {
                    symmetric: true,
                    ..
                }
            )),
        },
    }
}

/// Map an ABI dtype back.
///
/// Lossy on an unknown tag, which is stated rather than hidden: this is for a
/// host that has already validated the blob, not for reading an untrusted one.
/// [`decode`] is the validating path.
#[must_use]
pub fn dtype_from_raw(raw: StrataDType) -> DType {
    DType {
        element: element_from_u32_lossy(raw.element as u32),
        quant: match raw.quant.kind {
            StrataQuantKind::Block => QuantSpec::Block {
                block_size: raw.quant.block_size,
                symmetric: raw.quant.symmetric != 0,
            },
            StrataQuantKind::Microscaling => QuantSpec::Microscaling {
                block_size: raw.quant.block_size,
            },
            StrataQuantKind::None => QuantSpec::None,
        },
    }
}

/// Map a core element type to its ABI tag.
#[must_use]
pub fn element_to_raw(element: ElementType) -> StrataElementType {
    match element {
        ElementType::F8E5M2 => StrataElementType::F8E5M2,
        ElementType::F8E4M3 => StrataElementType::F8E4M3,
        ElementType::F16 => StrataElementType::F16,
        ElementType::F32 => StrataElementType::F32,
        ElementType::F64 => StrataElementType::F64,
        ElementType::BF16 => StrataElementType::BF16,
        ElementType::I8 => StrataElementType::I8,
        ElementType::I16 => StrataElementType::I16,
        ElementType::I32 => StrataElementType::I32,
        ElementType::I64 => StrataElementType::I64,
        ElementType::U8 => StrataElementType::U8,
        ElementType::Bool => StrataElementType::Bool,
        ElementType::F8E8M0 => StrataElementType::F8E8M0,
        ElementType::F4E2M1 => StrataElementType::F4E2M1,
        // `ElementType` is `#[non_exhaustive]`, so this arm is reachable only by
        // a future core variant. F32 is the safe default because a float is
        // readable by every backend in the tree, and `element_mapping_covers_all`
        // fails the build's tests if a variant is ever added without a tag.
        _ => StrataElementType::F32,
    }
}

fn element_from_u32_lossy(raw: u32) -> ElementType {
    element_from_u32(raw).unwrap_or(ElementType::F32)
}

fn element_from_u32(raw: u32) -> Result<ElementType, WireError> {
    let element = match raw {
        v if v == StrataElementType::F8E5M2 as u32 => ElementType::F8E5M2,
        v if v == StrataElementType::F8E4M3 as u32 => ElementType::F8E4M3,
        v if v == StrataElementType::F16 as u32 => ElementType::F16,
        v if v == StrataElementType::F32 as u32 => ElementType::F32,
        v if v == StrataElementType::F64 as u32 => ElementType::F64,
        v if v == StrataElementType::BF16 as u32 => ElementType::BF16,
        v if v == StrataElementType::I8 as u32 => ElementType::I8,
        v if v == StrataElementType::I16 as u32 => ElementType::I16,
        v if v == StrataElementType::I32 as u32 => ElementType::I32,
        v if v == StrataElementType::I64 as u32 => ElementType::I64,
        v if v == StrataElementType::U8 as u32 => ElementType::U8,
        v if v == StrataElementType::Bool as u32 => ElementType::Bool,
        v if v == StrataElementType::F8E8M0 as u32 => ElementType::F8E8M0,
        v if v == StrataElementType::F4E2M1 as u32 => ElementType::F4E2M1,
        other => {
            return Err(WireError::UnknownTag {
                field: "element-type",
                tag: other,
            });
        }
    };
    Ok(element)
}

/// Map a core op to its ABI tag.
#[must_use]
pub fn op_to_raw(op: &Op) -> StrataOpKind {
    match op {
        Op::Input => StrataOpKind::Input,
        Op::Add => StrataOpKind::Add,
        Op::Mul => StrataOpKind::Mul,
        Op::MatMul => StrataOpKind::MatMul,
        Op::Log => StrataOpKind::Log,
        Op::Exp => StrataOpKind::Exp,
        Op::Softmax { .. } => StrataOpKind::Softmax,
        Op::LayerNorm { .. } => StrataOpKind::LayerNorm,
        Op::Gelu => StrataOpKind::Gelu,
        Op::Linear { .. } => StrataOpKind::Linear,
        Op::Reshape { .. } => StrataOpKind::Reshape,
        Op::Transpose { .. } => StrataOpKind::Transpose,
        Op::Expand { .. } => StrataOpKind::Expand,
        Op::Slice { .. } => StrataOpKind::Slice,
        Op::Custom { .. } => StrataOpKind::Custom,
    }
}

/// Map a core shape class to its ABI form.
#[must_use]
pub fn shape_class_to_raw(class: ShapeClass) -> StrataShapeClass {
    let (kind, rank) = match class {
        ShapeClass::Scalar => (StrataShapeClassKind::Scalar, 0),
        ShapeClass::Empty => (StrataShapeClassKind::Empty, 0),
        ShapeClass::OuterProduct => (StrataShapeClassKind::OuterProduct, 0),
        ShapeClass::Vector => (StrataShapeClassKind::Vector, 0),
        ShapeClass::Matrix => (StrataShapeClassKind::Matrix, 0),
        ShapeClass::Batch3 => (StrataShapeClassKind::Batch3, 0),
        ShapeClass::Batch4 => (StrataShapeClassKind::Batch4, 0),
        ShapeClass::HigherRank { rank } => (StrataShapeClassKind::HigherRank, rank),
    };
    StrataShapeClass {
        kind,
        rank,
        reserved: [0; 2],
    }
}

/// The graph schema version, for a plugin that wants to check before reading.
#[must_use]
pub const fn schema() -> u32 {
    GRAPH_SCHEMA
}
