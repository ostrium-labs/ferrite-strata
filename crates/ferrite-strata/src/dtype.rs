//! Element types, and the quantisation formats layered on top of them.
//!
//! # Why this is a closed enum and not a string
//!
//! PJRT's dtype enum is worth watching: it already carries `BF16` and the MX-style
//! `F8E8M0FNU` / `F4E2M1FN` formats, and a Strata vendor reading a PJRT header
//! would reasonably assume those are the formats on offer. They are in ours too.
//!
//! The difference is what PJRT has *no representation for at all*:
//!
//! > **int8 block-quant scales and zero-points.** Quantisation rides inside HLO
//! > rather than in the ABI.
//!
//! So a dtype here is not enough to describe a tensor on a real inference
//! accelerator. An `int8` weight tensor at 8-bit has a block size, a scale and a
//! zero-point per block, and two vendors will disagree about the block size while
//! agreeing about the dtype. [`DType`] therefore comes in two layers:
//! [`ElementType`] for the storage bits, and [`QuantSpec`] for the scaling
//! convention that gives them meaning.
//!
//! This is gap #3 in the gap list in
//! [`docs/design-notes.md`](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/design-notes.md)
//! and it is the one gap we have to close ourselves.
//!
//! # Inference only
//!
//! There are no grad, optimiser or sparse formats here
//! ([ADR-0015](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0015-inference-first-no-training-support.md)).
//! Adding one means extending [`ElementType`] and every exhaustive match over it,
//! which is the intended cost: a new format is a visible ABI change rather than an
//! attribute nobody notices.

use core::fmt;

/// The storage bits of a tensor, with no scaling convention attached.
///
/// A bare [`ElementType::I8`] is not a complete description of an inference
/// tensor; see [`QuantSpec`] and [`DType`] for why.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ElementType {
    /// 8-bit IEEE 754 binary floating point.
    F8E5M2,
    /// 8-bit IEEE 754 binary floating point with a bias and no sign.
    F8E4M3,
    /// 16-bit IEEE 754 binary floating point. Not inferred from `f32`.
    F16,
    /// 32-bit IEEE 754 binary floating point.
    F32,
    /// 64-bit IEEE 754 binary floating point.
    ///
    /// Not an inference type. It is here because the graph IR is also used to
    /// express geometry and index arithmetic, and because Loams summarization
    /// arithmetic is f64 by contract — see
    /// [ADR-0013](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0013-no-acceleration-claim-for-graph-text-or-analytical-loams-workloads.md).
    F64,
    /// "Brain float": the top 16 bits of an `f32`, so the exponent range is `f32`
    /// and the mantissa is truncated.
    ///
    /// Inferred from `f32`, so a vendor that silently produced `f16` here would be
    /// a correctness bug and not a performance trade.
    BF16,
    /// The 8-bit block-scaled microscaling format: one shared `E8M0` exponent per
    /// block of 32 elements, no per-element exponent.
    ///
    /// Carried by PJRT as `F8E8M0FNU`.
    F8E8M0,
    /// The 4-bit element microscaling format, paired with an `E8M0` block scale.
    ///
    /// Carried by PJRT as `F4E2M1FN`.
    F4E2M1,
    /// Signed 8-bit integer. Meaningless without a [`QuantSpec`]; a tensor with no
    /// spec attached is rejected rather than read as raw `i8`.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Boolean. One byte per element, `0` or `1`.
    Bool,
}

impl ElementType {
    /// The size of one element in bytes.
    ///
    /// `#[non_exhaustive]` means this cannot be a bare `match` without a
    /// fallback arm, which is deliberate: an unhandled variant should be a
    /// compile error at the declaration site, not a silently wrong stride.
    /// Invariant: every variant is 1, 2, 4 or 8 bytes, so a buffer's byte length
    /// is always a whole number of elements.
    #[must_use]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F8E5M2 | Self::F8E4M3 | Self::F8E8M0 | Self::I8 | Self::U8 | Self::Bool => 1,
            Self::F16 | Self::BF16 | Self::F4E2M1 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
        }
    }

    /// Whether this is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F8E5M2
                | Self::F8E4M3
                | Self::F16
                | Self::F32
                | Self::F64
                | Self::BF16
                | Self::F8E8M0
                | Self::F4E2M1
        )
    }

    /// Whether this is an integer or boolean type, i.e. exact arithmetic.
    #[must_use]
    pub const fn is_int(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::U8 | Self::Bool
        )
    }

    /// The type a cast to this type reads as, for diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F8E5M2 => "f8e5m2",
            Self::F8E4M3 => "f8e4m3",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::BF16 => "bf16",
            Self::F8E8M0 => "f8e8m0",
            Self::F4E2M1 => "f4e2m1",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::Bool => "bool",
        }
    }

    /// Every element type, for exhaustive iteration in tests and tools.
    ///
    /// A function rather than a `const` slice so that adding a variant cannot be
    /// silently omitted from a loop that assumes it is complete.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::F8E5M2,
            Self::F8E4M3,
            Self::F16,
            Self::F32,
            Self::F64,
            Self::BF16,
            Self::F8E8M0,
            Self::F4E2M1,
            Self::I8,
            Self::I16,
            Self::I32,
            Self::I64,
            Self::U8,
            Self::Bool,
        ]
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The block size the shipped microscaling formats define.
///
/// 32 for both [`ElementType::F8E8M0`] and [`ElementType::F4E2M1`]. Named rather
/// than inlined at the use sites so the invariant has one definition to change if a
/// future format differs.
pub const MICOSCALING_BLOCK: u32 = 32;

/// A complete numeric description of a tensor: storage bits plus how they scale.
///
/// The point of the two layers is that `int8` alone is not a format. Eight bits
/// plus a block size, a scale and a zero-point is a format, and two backends
/// agreeing on `i8` tells the planner nothing about whether their weights are
/// interchangeable.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DType {
    /// The storage bits.
    pub element: ElementType,
    /// The scaling convention, if any.
    pub quant: QuantSpec,
}

impl DType {
    /// An unquantised type: the bits are the value.
    ///
    /// Only valid for types that carry their own exponent, i.e. the IEEE formats.
    /// `DType::plain` is not the place to reject it; construction of a
    /// bare integer tensor without a spec goes through [`DType::quantised`].
    #[must_use]
    pub const fn plain(element: ElementType) -> Self {
        Self {
            element,
            quant: QuantSpec::None,
        }
    }

    /// An integer type plus its per-block affine scaling convention.
    ///
    /// # Panics
    ///
    /// If `element` is not an integer type. A `bf16` tensor with a per-block
    /// scale is a contradiction rather than a redundant description, and silently
    /// keeping both would produce a dtype that serialises to two conflicting
    /// answers. This is a programming error at graph-construction time, so it
    /// panics rather than returning a `Result` that every caller would ignore.
    ///
    /// Also panics if `quant` is [`QuantSpec::Microscaling`]: that convention
    /// belongs to the microscaling *float* formats and goes through
    /// [`DType::microscaled`], which checks the element type too.
    #[must_use]
    pub const fn quantised(element: ElementType, quant: QuantSpec) -> Self {
        assert!(
            element.is_int(),
            "a quantisation spec on a non-integer element type is a contradiction; \
             use DType::plain for the IEEE formats",
        );
        assert!(
            matches!(quant, QuantSpec::Block { .. }),
            "a per-block affine scale applies to integer storage; the microscaling \
             formats carry their scale implicitly and go through DType::microscaled",
        );
        Self { element, quant }
    }

    /// A microscaling float: `f8_e8m0` or `f4e2m1`, sharing one `E8M0` exponent
    /// per block of 32 elements.
    ///
    /// Separate from [`DType::quantised`] because these are the one case where a
    /// scaling convention rides on a float element type — the exponent is not in
    /// the element, it is in the block. So the pair `(F8E8M0, Microscaling)` is
    /// meaningful where `(F32, Block)` is a contradiction, and folding them
    /// together would make the constructor unable to say which is which.
    ///
    /// # Panics
    ///
    /// If `element` is not one of [`ElementType::F8E8M0`] or
    /// [`ElementType::F4E2M1`], or if `block_size` is not 32. Both are programming
    /// errors: the shipped microscaling formats define the block at 32, and a
    /// vendor-defined format with a different block size is not this constructor's
    /// business to accept under this name.
    #[must_use]
    pub const fn microscaled(element: ElementType, block_size: u32) -> Self {
        assert!(
            matches!(element, ElementType::F8E8M0 | ElementType::F4E2M1),
            "only the microscaling formats carry an implicit block exponent; \
             f32 does not have a block scale to carry",
        );
        assert!(
            block_size == MICOSCALING_BLOCK,
            "the shipped microscaling formats define the block at 32, not this",
        );
        Self {
            element,
            quant: QuantSpec::Microscaling { block_size },
        }
    }

    /// Whether this tensor carries a scaling convention beyond its own exponent.
    #[must_use]
    pub const fn is_quantised(self) -> bool {
        !matches!(self.quant, QuantSpec::None)
    }

    /// Bytes occupied by `elements` values of this type.
    ///
    /// Saturates rather than wrapping: a shape whose byte length exceeds
    /// `usize::MAX` is a rejected graph, and a wrapped length is a buffer
    /// allocated at the wrong size.
    #[must_use]
    pub const fn byte_len(self, elements: usize) -> usize {
        elements.saturating_mul(self.element.size_in_bytes())
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.quant {
            QuantSpec::None => write!(f, "{}", self.element),
            QuantSpec::Block { .. } | QuantSpec::Microscaling { .. } => {
                write!(f, "{}:{}", self.element, self.quant)
            }
        }
    }
}

/// How integer storage bits are scaled into the numeric domain.
///
/// `None` is not "the identity" for integers — it is "this type carries its own
/// exponent", which is true only of the IEEE formats. An integer tensor with
/// `None` is a type error rather than a raw-integer tensor, because a raw `i8`
/// weight has no defined numeric meaning and treating it as a raw integer is how
/// a quantised network ends up scaled twice.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QuantSpec {
    /// No separate scaling: the element type is the value.
    None,

    /// Symmetric per-block affine quantisation, the format int8 inference weights
    /// are usually stored in.
    ///
    /// Each block of `block_size` consecutive elements along the last dimension
    /// shares one `scale` and one `zero_point`. `zero_point` is zero for the
    /// symmetric case and non-zero for the asymmetric one; both are carried here
    /// rather than split into two formats so that the block size and the two
    /// parameters stay associated.
    Block {
        /// Elements sharing one scale and zero-point, counted along the last
        /// dimension.
        ///
        /// A power of two, because the formats that use this one pack the scale
        /// beside the block it applies to.
        block_size: u32,
        /// Whether the zero-point is fixed at zero.
        symmetric: bool,
    },

    /// One shared `E8M0` exponent per block of 32 elements, with no per-element
    /// exponent and no zero-point.
    ///
    /// The microscaling formats ([`ElementType::F8E8M0`] and
    /// [`ElementType::F4E2M1`]) carry their block scale implicitly at 32, so this
    /// records the block size rather than inventing a scale storage for them.
    Microscaling {
        /// Elements sharing one exponent. Always 32 for the shipped formats.
        block_size: u32,
    },
}

impl QuantSpec {
    /// The shared block size, if the format has one.
    #[must_use]
    pub const fn block_size(self) -> Option<u32> {
        match self {
            QuantSpec::None => None,
            QuantSpec::Block { block_size, .. } | QuantSpec::Microscaling { block_size } => {
                Some(block_size)
            }
        }
    }

    /// Whether the format stores a per-block zero-point.
    #[must_use]
    pub const fn has_zero_point(self) -> bool {
        match self {
            QuantSpec::None | QuantSpec::Microscaling { .. } => false,
            QuantSpec::Block { symmetric, .. } => !symmetric,
        }
    }
}

impl fmt::Display for QuantSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantSpec::None => f.write_str("none"),
            QuantSpec::Block {
                block_size,
                symmetric,
            } => write!(
                f,
                "block{}-{}-zp",
                block_size,
                if *symmetric { "sym" } else { "asym" }
            ),
            QuantSpec::Microscaling { block_size } => write!(f, "mx{block_size}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DType, ElementType, QuantSpec};

    #[test]
    fn every_element_type_has_a_byte_size_that_is_a_whole_element() {
        for element in ElementType::all() {
            assert!(
                [1, 2, 4, 8].contains(&element.size_in_bytes()),
                "{element} has size {}",
                element.size_in_bytes()
            );
        }
    }

    #[test]
    fn float_and_int_are_disjoint_and_exhaustive() {
        for element in ElementType::all() {
            assert!(
                element.is_float() ^ element.is_int(),
                "{element} claims to be both or neither"
            );
        }
    }

    #[test]
    fn all_covers_every_variant() {
        // `all` is a function precisely so that adding a variant cannot silently
        // drop it from iteration. A duplicate would let this pass while leaving a
        // variant uncovered, so check for that too.
        let all = ElementType::all();
        let mut sorted = all.to_vec();
        sorted.sort_unstable();
        let len = sorted.len();
        sorted.dedup();
        assert_eq!(len, sorted.len(), "ElementType::all has a duplicate");
        assert_eq!(sorted.len(), ElementType::all().len());
    }

    #[test]
    #[should_panic(expected = "contradiction")]
    fn a_quant_spec_on_a_float_is_rejected() {
        let _ = DType::quantised(
            ElementType::F32,
            QuantSpec::Block {
                block_size: 64,
                symmetric: true,
            },
        );
    }

    #[test]
    fn a_block_spec_carries_a_size_and_knows_about_its_zero_point() {
        let spec = QuantSpec::Block {
            block_size: 64,
            symmetric: false,
        };
        assert_eq!(spec.block_size(), Some(64));
        assert!(spec.has_zero_point());

        let sym = QuantSpec::Block {
            block_size: 64,
            symmetric: true,
        };
        assert!(!sym.has_zero_point());
    }

    #[test]
    fn microscaling_has_a_block_size_and_no_zero_point() {
        let spec = QuantSpec::Microscaling { block_size: 32 };
        assert_eq!(spec.block_size(), Some(32));
        assert!(!spec.has_zero_point());
    }

    #[test]
    fn byte_len_saturates_instead_of_wrapping() {
        let dtype = DType::quantised(
            ElementType::I8,
            QuantSpec::Block {
                block_size: 64,
                symmetric: true,
            },
        );
        assert_eq!(dtype.byte_len(10), 10);
        assert_eq!(dtype.byte_len(usize::MAX), usize::MAX);
    }

    #[test]
    fn display_names_the_quant_form() {
        let q = DType::quantised(
            ElementType::I8,
            QuantSpec::Block {
                block_size: 64,
                symmetric: true,
            },
        );
        assert_eq!(q.to_string(), "i8:block64-sym-zp");
        assert_eq!(DType::plain(ElementType::BF16).to_string(), "bf16");
    }
}
