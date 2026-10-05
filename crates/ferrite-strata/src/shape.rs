//! Tensor shapes, and the coarse shape classification the pull query uses.
//!
//! # Why shapes are static here
//!
//! Inference shapes are static in the overwhelmingly common case: weights are
//! loaded at fixed sizes, activations are batch × seq × hidden with a batch that
//! is fixed for a session. Nothing here needs symbolic dimensions, so every
//! dimension is a `usize` and a shape is fully known at graph-construction time.
//!
//! That is what makes the pull query in [`crate::capabilities`] possible at all.
//! A `supports(pattern_id, dtype, shape)` answer that has to carry symbolic
//! dimensions is a query a planner cannot answer before compiling, which is the
//! failure mode this design exists to avoid — see
//! [ADR-0005](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0005-capabilities-are-versioned-data-plus-a-pull-supports-query.md).
//!
//! What survives for a genuinely dynamic size is that a query names the *class*
//! rather than a shape: [`ShapeClass`] answers a promise about a family of sizes,
//! and a backend that needs finer detail uses a fused-pattern predicate rather than
//! growing this enum. Symbolic dimensions are not in B1, and a graph that needs them
//! does not fit [`Shape::new`] — which is the honest place for that to fail.

use core::fmt;

/// A tensor shape: a rank and its dimensions, all statically known.
///
/// Rank-0 is the scalar, and is a normal shape rather than a special case,
/// because a reduction over an empty axis list is how one is produced.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Shape {
    dims: [usize; Shape::MAX_RANK],
    rank: u8,
}

/// The highest rank the IR admits.
///
/// Chosen so a `Shape` fits in a register-width value alongside its length: 8
/// dimensions covers every tensor in an inference graph except a pathological
/// one, and a fixed ceiling means [`Shape`] is `Copy` with no heap allocation on
/// the partitioner's hot path.
pub const MAX_RANK: usize = 8;

impl Shape {
    /// The rank ceiling, as an associated constant so `Shape::MAX_RANK` reads
    /// naturally at a call site.
    pub const MAX_RANK: usize = MAX_RANK;

    /// A shape of the given rank and dimensions.
    ///
    /// # Errors
    ///
    /// If the rank exceeds [`Shape::MAX_RANK`].
    pub fn new(dims: &[usize]) -> Result<Self, ShapeError> {
        if dims.len() > Self::MAX_RANK {
            return Err(ShapeError::RankTooHigh {
                rank: dims.len(),
                limit: Self::MAX_RANK,
            });
        }
        let mut stored = [0usize; Self::MAX_RANK];
        stored[..dims.len()].copy_from_slice(dims);
        Ok(Self {
            dims: stored,
            // A rank above `MAX_RANK` was rejected above, so this cast is total.
            rank: dims.len() as u8,
        })
    }

    /// The rank-0 shape.
    #[must_use]
    pub const fn scalar() -> Self {
        Self {
            dims: [0; Self::MAX_RANK],
            rank: 0,
        }
    }

    /// A rank-1 shape.
    ///
    /// # Errors
    ///
    /// Never, in practice: rank 1 is within the limit. The `Result` is kept so that
    /// `vec![len]` needs no `.into()` and a caller cannot accidentally pass a
    /// slice of the wrong length.
    pub fn vector(len: usize) -> Result<Self, ShapeError> {
        Self::new(&[len])
    }

    /// The rank, from 0 to [`Shape::MAX_RANK`].
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank as usize
    }

    /// The dimension at `axis`, or `None` if `axis` is out of range.
    #[must_use]
    pub const fn dim(&self, axis: usize) -> Option<usize> {
        if axis >= self.rank as usize {
            return None;
        }
        Some(self.dims[axis])
    }

    /// Every dimension, in order.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank as usize]
    }

    /// The total number of elements.
    ///
    /// Saturates rather than wrapping, for the same reason as
    /// [`DType::byte_len`](crate::DType::byte_len): a product that exceeds
    /// `usize::MAX` is a rejected graph, and a wrapped element count is a buffer
    /// allocated at the wrong size.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.dims()
            .iter()
            .fold(1usize, |acc, &dim| acc.saturating_mul(dim))
    }

    /// Whether this is the rank-0 shape.
    #[must_use]
    pub const fn is_scalar(&self) -> bool {
        self.rank == 0
    }

    /// Whether any dimension is zero, making the tensor empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dims().contains(&0)
    }

    /// The shape with `dims` transposed by `permutation`.
    ///
    /// # Errors
    ///
    /// If `permutation` is not a permutation of `0..rank`. A short or long
    /// permutation is rejected rather than padded, because a partial transpose is
    /// a different — and plausible-looking — operation.
    pub fn transpose(&self, permutation: &[usize]) -> Result<Self, ShapeError> {
        if permutation.len() != self.rank() {
            return Err(ShapeError::NotAPermutation {
                rank: self.rank(),
                got: permutation.len(),
            });
        }
        let mut dims = [0usize; Self::MAX_RANK];
        let mut seen = [false; Self::MAX_RANK];
        for (position, &axis) in permutation.iter().enumerate() {
            if axis >= self.rank() {
                return Err(ShapeError::AxisOutOfRange {
                    axis,
                    rank: self.rank(),
                });
            }
            if seen[axis] {
                return Err(ShapeError::RepeatedAxis { axis });
            }
            seen[axis] = true;
            dims[position] = self.dims[axis];
        }
        Ok(Self {
            dims,
            rank: self.rank,
        })
    }

    /// The shape of a broadcast of `self` against `other` along the last axes.
    ///
    /// NumPy rules: rank-1 shapes right-align, and a dimension of 1 broadcasts to
    /// whatever it is paired with. Two shapes of equal rank must agree on every
    /// dimension except where one of them is 1.
    ///
    /// # Errors
    ///
    /// If the two shapes are not broadcastable.
    pub fn broadcast(&self, other: &Self) -> Result<Self, ShapeError> {
        let rank = self.rank().max(other.rank());
        let rank_isize = rank as isize;
        let mut dims = [0usize; Self::MAX_RANK];

        // Right-alignment: output axis `out` maps to this shape's axis
        // `out + self.rank() - rank`, which is *negative* for every axis beyond
        // this shape's rank. Those axes are absent and contribute 1 — hence the
        // signed arithmetic. `saturating_sub` would clamp the offset to 0 and
        // compare this shape's first dimension against the other's, which fails
        // for every rank-1-against-rank-2 broadcast.
        let aligned = |shape: &Self, out: usize| -> usize {
            let axis = out as isize + shape.rank() as isize - rank_isize;
            if axis < 0 {
                1
            } else {
                shape.dim(axis as usize).unwrap_or(1)
            }
        };

        for (out_axis, out) in dims.iter_mut().enumerate().take(rank) {
            let lhs = aligned(self, out_axis);
            let rhs = aligned(other, out_axis);
            *out = match (lhs, rhs) {
                (1, other) => other,
                (other, 1) => other,
                (lhs, rhs) if lhs == rhs => lhs,
                (lhs, rhs) => {
                    return Err(ShapeError::NotBroadcastable {
                        lhs,
                        rhs,
                        axis: out_axis,
                    });
                }
            };
        }
        Ok(Self {
            dims,
            rank: rank as u8,
        })
    }

    /// The coarse classification this shape presents to a capability query.
    ///
    /// Deliberately lossy: two shapes in the same class are promised the same
    /// answer by any backend whose capability data is expressed over classes, and
    /// a backend that needs finer detail uses a fused-pattern query with an
    /// explicit shape predicate rather than growing this enum.
    #[must_use]
    pub fn class(&self) -> ShapeClass {
        if self.is_scalar() {
            return ShapeClass::Scalar;
        }
        if self.is_empty() {
            return ShapeClass::Empty;
        }
        let rank = self.rank();
        let leading = self.dims()[0];
        // A leading 1 makes a tensor outer-product-shaped whatever its rank: a
        // matmul operand, a bias broadcast, or a single token. That is the case a
        // naive rank test misfiles, and it is the most common one in an inference
        // graph.
        if leading == 1 {
            return ShapeClass::OuterProduct;
        }
        if rank == 1 {
            return ShapeClass::Vector;
        }
        if rank == 2 {
            return ShapeClass::Matrix;
        }
        if rank == 3 {
            return ShapeClass::Batch3;
        }
        if rank == 4 {
            return ShapeClass::Batch4;
        }
        ShapeClass::HigherRank { rank: rank as u8 }
    }
}

/// A shape that could not be constructed or combined.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShapeError {
    /// A shape had more dimensions than [`Shape::MAX_RANK`] allows.
    RankTooHigh {
        /// The requested rank.
        rank: usize,
        /// The limit that was exceeded.
        limit: usize,
    },
    /// A permutation did not have exactly `rank` entries.
    NotAPermutation {
        /// The rank of the shape being permuted.
        rank: usize,
        /// How many entries the permutation had.
        got: usize,
    },
    /// A permutation named an axis the shape does not have.
    AxisOutOfRange {
        /// The offending axis.
        axis: usize,
        /// The rank of the shape.
        rank: usize,
    },
    /// A permutation named the same axis twice.
    RepeatedAxis {
        /// The repeated axis.
        axis: usize,
    },
    /// Two shapes could not broadcast against each other.
    NotBroadcastable {
        /// The left dimension.
        lhs: usize,
        /// The right dimension.
        rhs: usize,
        /// The axis they disagreed on, counted from the front of the result.
        axis: usize,
    },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::RankTooHigh { rank, limit } => write!(
                f,
                "rank {rank} exceeds the limit of {limit}. Raise `MAX_RANK` if the \
                 graph genuinely needs it: it is a fixed-size array, so raising it \
                 makes every `Shape` wider."
            ),
            ShapeError::NotAPermutation { rank, got } => write!(
                f,
                "a permutation of a rank-{rank} shape needs exactly {rank} entries, \
                 but this one has {got}. A short permutation is a partial transpose, \
                 which is a different operation rather than a mistaken one."
            ),
            ShapeError::AxisOutOfRange { axis, rank } => write!(
                f,
                "axis {axis} does not exist on a rank-{rank} shape, whose axes are \
                 0..{rank}."
            ),
            ShapeError::RepeatedAxis { axis } => write!(
                f,
                "axis {axis} appears twice in the permutation, so some axis was \
                 dropped. A permutation visits every axis exactly once."
            ),
            ShapeError::NotBroadcastable { lhs, rhs, axis } => write!(
                f,
                "dimension {axis} cannot broadcast {lhs} against {rhs}. One of them \
                 must be 1, or they must be equal; insert an explicit `reshape` or \
                 `expand` so the intent is visible in the graph."
            ),
        }
    }
}

impl std::error::Error for ShapeError {}

/// The coarse shape class a capability query is answered over.
///
/// This is our vocabulary, not a vendor's, because it is the framework's to keep
/// stable. Vendor-owned specificity lives in the fused-pattern vocabulary
/// ([ADR-0005](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0005-capabilities-are-versioned-data-plus-a-pull-supports-query.md)),
/// which is extensible strings — not here.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ShapeClass {
    /// Rank 0.
    Scalar,
    /// Any dimension is zero.
    Empty,
    /// Leading dimension 1, at any rank: a matmul operand, a bias, one token.
    OuterProduct,
    /// Rank 1 with a leading dimension above 1.
    Vector,
    /// Rank 2.
    Matrix,
    /// Rank 3.
    Batch3,
    /// Rank 4.
    Batch4,
    /// Rank 5 or more.
    HigherRank {
        /// The rank, from 5 up to [`Shape::MAX_RANK`].
        rank: u8,
    },
}

impl ShapeClass {
    /// Whether a shape in this class is guaranteed not to contain a dimension
    /// above 1.
    ///
    /// Capability data uses this to describe a "one-dimensional inner loop" limit
    /// without enumerating the empty and scalar cases.
    #[must_use]
    pub const fn is_unary_dim(&self) -> bool {
        matches!(self, Self::Empty | Self::OuterProduct)
    }
}

impl fmt::Display for Shape {
    /// NumPy-style: `2x3x4`, and `scalar` for rank 0.
    ///
    /// Because every shape error names two shapes, this is not a nicety — without it
    /// every error message has to fall back to `Debug` and print `[2, 3, 4]` where a
    /// person reads `2x3x4`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_scalar() {
            return f.write_str("scalar");
        }
        for (i, &dim) in self.dims().iter().enumerate() {
            if i > 0 {
                f.write_str("x")?;
            }
            write!(f, "{dim}")?;
        }
        Ok(())
    }
}

impl fmt::Display for ShapeClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeClass::Scalar => f.write_str("scalar"),
            ShapeClass::Empty => f.write_str("empty"),
            ShapeClass::OuterProduct => f.write_str("outer-product"),
            ShapeClass::Vector => f.write_str("vector"),
            ShapeClass::Matrix => f.write_str("matrix"),
            ShapeClass::Batch3 => f.write_str("batch3"),
            ShapeClass::Batch4 => f.write_str("batch4"),
            ShapeClass::HigherRank { rank } => write!(f, "rank{rank}+"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_RANK, Shape, ShapeClass, ShapeError};

    #[test]
    fn a_rank_0_shape_has_no_dimensions_and_one_element() {
        let s = Shape::scalar();
        assert!(s.is_scalar());
        assert_eq!(s.rank(), 0);
        assert_eq!(s.dims(), &[] as &[usize]);
        assert_eq!(s.element_count(), 1);
        assert_eq!(s.class(), ShapeClass::Scalar);
    }

    #[test]
    fn element_count_saturates() {
        let s = Shape::new(&[usize::MAX, usize::MAX]).expect("rank 2 is fine");
        assert_eq!(s.element_count(), usize::MAX);
    }

    #[test]
    fn rank_above_the_limit_is_rejected_with_the_limit_named() {
        let dims = vec![1usize; MAX_RANK + 1];
        let err = Shape::new(&dims).expect_err("rank 9 exceeds the limit of 8");
        assert_eq!(
            err,
            ShapeError::RankTooHigh {
                rank: 9,
                limit: MAX_RANK
            }
        );
        assert!(err.to_string().contains("limit of 8"), "{err}");
    }

    #[test]
    fn an_axis_past_the_rank_is_none_not_a_panic() {
        let s = Shape::new(&[2, 3]).expect("rank 2");
        assert_eq!(s.dim(0), Some(2));
        assert_eq!(s.dim(1), Some(3));
        assert_eq!(s.dim(2), None);
    }

    #[test]
    fn transpose_applies_the_permutation() {
        let s = Shape::new(&[2, 3, 4]).expect("rank 3");
        assert_eq!(
            s.transpose(&[2, 0, 1]).expect("a permutation").dims(),
            &[4, 2, 3]
        );
        assert_eq!(
            s.transpose(&[1, 0, 2]).expect("a permutation").dims(),
            &[3, 2, 4]
        );
    }

    #[test]
    fn a_short_permutation_is_rejected_rather_than_padded() {
        let s = Shape::new(&[2, 3, 4]).expect("rank 3");
        let err = s.transpose(&[0, 1]).expect_err("two entries for rank 3");
        assert_eq!(err, ShapeError::NotAPermutation { rank: 3, got: 2 });
    }

    #[test]
    fn a_repeated_axis_is_rejected() {
        let s = Shape::new(&[2, 3]).expect("rank 2");
        assert_eq!(
            s.transpose(&[0, 0]).expect_err("axis 0 twice"),
            ShapeError::RepeatedAxis { axis: 0 }
        );
    }

    #[test]
    fn broadcast_right_aligns_and_ones_expand() {
        let (rows, cols) = (2usize, 3usize);
        let matrix = Shape::new(&[rows, cols]).expect("rank 2");
        let vector = Shape::vector(cols).expect("rank 1");
        assert_eq!(
            matrix
                .broadcast(&vector)
                .expect("rank 2 with rank 1")
                .dims(),
            &[rows, cols]
        );
        assert_eq!(
            vector
                .broadcast(&matrix)
                .expect("rank 1 with rank 2")
                .dims(),
            &[rows, cols]
        );
        assert_eq!(
            vector
                .broadcast(&vector)
                .expect("rank 1 with rank 1")
                .dims(),
            &[cols]
        );
    }

    #[test]
    fn a_leading_one_expands_a_rank_1_against_a_rank_3() {
        let v = Shape::vector(5).expect("rank 1");
        let t = Shape::new(&[2, 5, 5]).expect("rank 3");
        assert_eq!(v.broadcast(&t).expect("right-aligned").dims(), &[2, 5, 5]);
    }

    #[test]
    fn mismatched_dimensions_are_rejected_with_the_axis() {
        let a = Shape::new(&[2, 3]).expect("rank 2");
        let b = Shape::new(&[2, 4]).expect("rank 2");
        let err = a.broadcast(&b).expect_err("3 against 4");
        assert_eq!(
            err,
            ShapeError::NotBroadcastable {
                lhs: 3,
                rhs: 4,
                axis: 1
            }
        );
        assert!(err.to_string().contains("dimension 1"), "{err}");
    }

    #[test]
    fn a_leading_dimension_of_one_classifies_as_outer_product() {
        // The case a naive rank test misfiles, and the most common one in an
        // inference graph.
        assert_eq!(
            Shape::new(&[1, 5, 5]).expect("rank 3").class(),
            ShapeClass::OuterProduct
        );
        assert_eq!(
            Shape::new(&[1, 4096]).expect("rank 2").class(),
            ShapeClass::OuterProduct
        );
    }

    #[test]
    fn classes_cover_every_rank() {
        assert_eq!(
            Shape::new(&[7]).expect("rank 1").class(),
            ShapeClass::Vector
        );
        assert_eq!(
            Shape::new(&[7, 5]).expect("rank 2").class(),
            ShapeClass::Matrix
        );
        assert_eq!(
            Shape::new(&[7, 5, 3]).expect("rank 3").class(),
            ShapeClass::Batch3
        );
        assert_eq!(
            Shape::new(&[7, 5, 3, 2]).expect("rank 4").class(),
            ShapeClass::Batch4
        );
        assert_eq!(
            Shape::new(&[7, 5, 3, 2, 1, 1]).expect("rank 6").class(),
            ShapeClass::HigherRank { rank: 6 }
        );
    }

    #[test]
    fn an_empty_tensor_has_the_empty_class() {
        assert_eq!(
            Shape::new(&[2, 0]).expect("rank 2").class(),
            ShapeClass::Empty
        );
        assert!(Shape::new(&[2, 0]).expect("rank 2").is_empty());
        assert_eq!(Shape::new(&[2, 0]).expect("rank 2").element_count(), 0);
    }
}
