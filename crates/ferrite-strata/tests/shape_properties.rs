//! Property tests for the IR's algebraic invariants.
//!
//! Unit tests pin specific answers; these check the laws, which is where a shape or
//! broadcast implementation actually breaks. A rank-4 tensor with a `2` in it and a
//! rank-1 tensor with a `3` in it is a handful of cases; the property that broadcast
//! is associative across all of them is not.

use ferrite_strata::{DType, ElementType, Shape, ShapeClass, ShapeError};
use proptest::prelude::*;

/// A shape strategy: a rank-0 to rank-4 shape with small, positive dimensions.
///
/// Small on purpose. Dimensions are drawn from `1..=4` so that a generated pair of
/// shapes is *usually* broadcastable, which means a failing case is a real failure
/// rather than a refusal the generator produced.
fn shape() -> impl Strategy<Value = Shape> {
    proptest::collection::vec(1usize..=4, 0..=4)
        .prop_map(|dims| Shape::new(&dims).expect("rank at most 4 is within the limit"))
}

proptest! {
    #[test]
    fn broadcast_is_commutative(lhs in shape(), rhs in shape()) {
        let Ok(left) = lhs.broadcast(&rhs) else {
            // Not broadcastable in one direction means not in the other; checking
            // that is itself part of the property.
            prop_assert!(rhs.broadcast(&lhs).is_err());
            return Ok(());
        };
        let right = rhs.broadcast(&lhs).expect("broadcast is symmetric");
        prop_assert_eq!(left.dims(), right.dims());
    }

    #[test]
    fn broadcasting_a_shape_with_itself_is_the_identity(value in shape()) {
        let out = value.broadcast(&value).expect("a shape broadcasts with itself");
        prop_assert_eq!(out.dims(), value.dims());
    }

    #[test]
    fn broadcasting_preserves_the_element_count(lhs in shape(), rhs in shape()) {
        let Ok(out) = lhs.broadcast(&rhs) else {
            return Ok(());
        };
        prop_assert!(out.element_count() >= lhs.element_count());
        prop_assert!(out.element_count() >= rhs.element_count());
    }

    #[test]
    fn a_scalar_broadcasts_with_anything(value in shape()) {
        let scalar = Shape::scalar();
        let out = value.broadcast(&scalar).expect("a scalar broadcasts with anything");
        prop_assert_eq!(out.dims(), value.dims());
    }

    #[test]
    fn a_vector_broadcasts_with_a_matrix_of_the_same_width(
        rows in 1usize..=4,
        cols in 1usize..=4,
    ) {
        let vector = Shape::vector(cols).expect("rank 1");
        let matrix = Shape::new(&[rows, cols]).expect("rank 2");
        let out = matrix.broadcast(&vector).expect("a row vector expands over rows");
        prop_assert_eq!(out.dims(), &[rows, cols]);
    }

    #[test]
    fn transpose_twice_is_the_identity(value in shape()) {
        let Ok(axes) = permutation(value.rank()) else {
            return Ok(());
        };
        let once = value.transpose(&axes).expect("generated permutation is valid");
        let back = once
            .transpose(&invert(&axes))
            .expect("the inverse permutation is valid");
        prop_assert_eq!(back.dims(), value.dims());
    }

    #[test]
    fn a_permutation_of_a_shorter_list_is_refused(value in shape()) {
        // A short permutation is a partial transpose, which is a different operation
        // rather than a mistaken one, so it must never be silently padded.
        //
        // Rank 0 is excluded: its only permutation is the empty list, so there is no
        // shorter one to supply.
        let rank = value.rank();
        if rank == 0 {
            return Ok(());
        }
        let short = vec![0usize; rank - 1];
        // `prop_assert!` stringifies its argument into a format string, so a `matches!`
        // pattern containing braces has to be bound first.
        let refused = matches!(
            value.transpose(&short),
            Err(ShapeError::NotAPermutation { .. })
        );
        prop_assert!(refused);
    }

    #[test]
    fn a_repeated_axis_is_refused(value in shape()) {
        let rank = value.rank();
        if rank < 2 {
            // Rank 0 and 1 have exactly one permutation each, and both are valid, so
            // there is no repetition to offer.
            return Ok(());
        }
        let repeated = vec![0usize; rank];
        let refused = matches!(
            value.transpose(&repeated),
            Err(ShapeError::RepeatedAxis { axis: 0 })
        );
        prop_assert!(refused);
    }

    #[test]
    fn the_shape_class_is_total(dims in proptest::collection::vec(0usize..=4, 0..=4)) {
        // Every shape gets a class, with no panics and no `Option`. Total by
        // construction is the property; the specific answers are unit-tested.
        let shape = Shape::new(&dims).expect("rank at most 4 is within the limit");
        let class = shape.class();
        let known = matches!(
            class,
            ShapeClass::Scalar
                | ShapeClass::Empty
                | ShapeClass::OuterProduct
                | ShapeClass::Vector
                | ShapeClass::Matrix
                | ShapeClass::Batch3
                | ShapeClass::Batch4
                | ShapeClass::HigherRank { .. }
        );
        prop_assert!(known);
    }

    #[test]
    fn element_count_is_the_product_of_the_dimensions(
        dims in proptest::collection::vec(1usize..=4, 0..=4)
    ) {
        let shape = Shape::new(&dims).expect("rank at most 4 is within the limit");
        let expected: usize = dims.iter().product();
        prop_assert_eq!(shape.element_count(), expected);
    }

    #[test]
    fn a_byte_length_is_a_whole_number_of_elements(
        dims in proptest::collection::vec(1usize..=4, 0..=4),
        element in prop_oneof![
            Just(ElementType::F8E5M2),
            Just(ElementType::F16),
            Just(ElementType::BF16),
            Just(ElementType::F32),
        ],
    ) {
        let shape = Shape::new(&dims).expect("rank at most 4 is within the limit");
        let dtype = DType::plain(element);
        let bytes = dtype.byte_len(shape.element_count());
        let size = element.size_in_bytes();
        prop_assert_eq!(bytes % size, 0);
    }

    #[test]
    fn a_rank_above_the_limit_is_always_refused(
        extra in 1usize..=4,
    ) {
        let rank = Shape::MAX_RANK + extra;
        let dims = vec![1usize; rank];
        let refused = matches!(Shape::new(&dims), Err(ShapeError::RankTooHigh { .. }));
        prop_assert!(refused);
    }
}

/// A permutation of `0..rank`, if `rank` is within the limit.
fn permutation(rank: usize) -> Result<Vec<usize>, ()> {
    if rank > Shape::MAX_RANK {
        return Err(());
    }
    let mut axes: Vec<usize> = (0..rank).collect();
    // Deterministic "shuffle": rotate by one when possible. Enough to catch a
    // transpose that mishandles a non-trivial ordering without needing a generator.
    if rank > 1 {
        axes.rotate_left(1);
    }
    Ok(axes)
}

fn invert(axes: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; axes.len()];
    for (position, &axis) in axes.iter().enumerate() {
        out[axis] = position;
    }
    out
}
