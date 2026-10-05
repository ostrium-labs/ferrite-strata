//! The tensor type: a shape and a contiguous run of `f32`.
//!
//! Row-major, contiguous, `Copy`-free and owned. No strides, no views, no lazy
//! axes. That is a real limitation and it is the right one here: this backend is an
//! oracle, and an oracle that can express a strided view is an oracle whose answer
//! depends on the view bookkeeping rather than only on the arithmetic.

use std::collections::HashMap;

use ferrite_strata::{DType, ElementType, Shape, ShapeError, ValueId};

/// A dense `f32` tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuTensor {
    shape: Shape,
    data: Vec<f32>,
}

impl CpuTensor {
    /// A tensor of `shape`, filled with `value`.
    ///
    /// # Errors
    ///
    /// Never in practice; the signature is `Result` only because the byte length
    /// computation can saturate, and a tensor whose element count wrapped would be
    /// silently the wrong size.
    pub fn filled(shape: Shape, value: f32) -> Result<Self, CpuError> {
        let count = shape.element_count();
        Ok(Self {
            shape,
            data: vec![value; count],
        })
    }

    /// A tensor from a row-major slice.
    ///
    /// # Errors
    ///
    /// If the slice length does not match the shape's element count. Checked rather
    /// than padded because an oracle that silently truncates produces an oracle that
    /// silently agrees with a wrong kernel.
    pub fn from_slice(data: &[f32], shape: Shape) -> Result<Self, CpuError> {
        if data.len() != shape.element_count() {
            return Err(CpuError::ShapeMismatch {
                expected: shape.element_count(),
                got: data.len(),
            });
        }
        Ok(Self {
            shape,
            data: data.to_vec(),
        })
    }

    /// An all-zero tensor of this shape.
    ///
    /// # Errors
    ///
    /// See [`CpuTensor::filled`].
    pub fn zeros(shape: Shape) -> Result<Self, CpuError> {
        Self::filled(shape, 0.0)
    }

    /// The shape.
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// The data, row-major.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// The data, mutably.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// How many elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor holds no elements.
    ///
    /// A tensor is empty exactly when some dimension is zero, so this is a real case
    /// and not a formality — a zero-sized batch reaches here.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The element at `(axis0, axis1)` of a 2-D tensor.
    ///
    /// # Errors
    ///
    /// If the tensor is not 2-D or the indices are out of range. A scalar-indexing
    /// helper rather than a general one on purpose: every kernel in this crate is
    /// written against explicit 2-D indexing so that the arithmetic under test is
    /// visible in the source.
    pub fn at(&self, row: usize, col: usize) -> Result<f32, CpuError> {
        let (rows, cols) = self.dims2()?;
        if row >= rows || col >= cols {
            return Err(CpuError::IndexOutOfRange {
                row,
                col,
                rows,
                cols,
            });
        }
        Ok(self.data[row * cols + col])
    }

    /// Set the element at `(row, col)`.
    ///
    /// # Errors
    ///
    /// See [`CpuTensor::at`].
    pub fn set(&mut self, row: usize, col: usize, value: f32) -> Result<(), CpuError> {
        let (rows, cols) = self.dims2()?;
        if row >= rows || col >= cols {
            return Err(CpuError::IndexOutOfRange {
                row,
                col,
                rows,
                cols,
            });
        }
        self.data[row * cols + col] = value;
        Ok(())
    }

    /// This tensor's dimensions as `(rows, cols)`.
    ///
    /// # Errors
    ///
    /// If the tensor is not rank 2. The rank is checked, not just the first two
    /// dimensions: a `[2, 2, 2]` tensor has a `dim(0)` and a `dim(1)` too, and
    /// reading it as a 2-D one would silently compute over the wrong elements.
    pub fn dims2(&self) -> Result<(usize, usize), CpuError> {
        match (self.shape.rank(), self.shape.dim(0), self.shape.dim(1)) {
            (2, Some(rows), Some(cols)) => Ok((rows, cols)),
            _ => Err(CpuError::NotRank2 {
                rank: self.shape.rank(),
            }),
        }
    }

    /// The dtype this backend computes in.
    #[must_use]
    pub fn dtype() -> DType {
        DType::plain(ElementType::F32)
    }
}

/// Collects the input tensors for an executable, keyed by value id.
pub type Inputs = HashMap<ValueId, CpuTensor>;

/// Why two shapes failed to broadcast against each other.
#[derive(Clone, Debug, PartialEq)]
pub struct BroadcastFailure {
    /// The left operand's shape.
    pub lhs: Shape,
    /// The right operand's shape.
    pub rhs: Shape,
    /// The specific disagreement.
    pub reason: ShapeError,
}

impl BroadcastFailure {
    /// A failure from two shapes and the IR's own verdict on them.
    #[must_use]
    pub fn new(lhs: Shape, rhs: Shape, reason: ShapeError) -> Self {
        Self { lhs, rhs, reason }
    }
}

/// Something wrong running on the CPU.
#[derive(Clone, Debug, PartialEq)]
pub enum CpuError {
    /// A slice length did not match its shape.
    ShapeMismatch {
        /// The element count the shape implies.
        expected: usize,
        /// How many elements were supplied.
        got: usize,
    },
    /// A 2-D operation was handed a tensor that is not 2-D.
    NotRank2 {
        /// The rank it actually had.
        rank: usize,
    },
    /// An index past the end of a 2-D tensor.
    IndexOutOfRange {
        /// The requested row.
        row: usize,
        /// The requested column.
        col: usize,
        /// How many rows there are.
        rows: usize,
        /// How many columns there are.
        cols: usize,
    },
    /// Two operands of the same elementwise op disagreed on shape.
    ///
    /// The detail is boxed so `CpuError` stays small enough to return by value.
    /// Two [`Shape`]s plus a [`ShapeError`] is well over the size clippy is
    /// right to complain about, and every kernel here returns this one.
    Broadcast {
        /// The shapes and why they did not broadcast.
        detail: Box<BroadcastFailure>,
    },
    /// Two matmul operands had incompatible inner dimensions.
    MatmulInner {
        /// The left operand's columns.
        lhs_cols: usize,
        /// The right operand's rows.
        rhs_rows: usize,
    },
    /// An executable was run without one of its operands.
    MissingInput {
        /// The value that was needed.
        id: ValueId,
    },
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpuError::ShapeMismatch { expected, got } => write!(
                f,
                "the data has {got} elements but the shape needs {expected}. \
                 Padding or truncating here would make this backend agree with a \
                 wrong kernel, which is the one thing it must never do."
            ),
            CpuError::NotRank2 { rank } => {
                write!(f, "expected a rank-2 tensor, got rank {rank}")
            }
            CpuError::IndexOutOfRange {
                row,
                col,
                rows,
                cols,
            } => write!(f, "index ({row}, {col}) is outside a {rows}x{cols} tensor"),
            CpuError::Broadcast { detail } => {
                write!(
                    f,
                    "cannot broadcast {} against {}: {}",
                    detail.lhs, detail.rhs, detail.reason
                )
            }
            CpuError::MatmulInner { lhs_cols, rhs_rows } => write!(
                f,
                "the matmul inner dimensions disagree: the left operand has {lhs_cols} \
                 columns and the right has {rhs_rows} rows"
            ),
            CpuError::MissingInput { id } => {
                write!(f, "no input supplied for {id}")
            }
        }
    }
}

impl std::error::Error for CpuError {}

impl From<ShapeError> for CpuError {
    fn from(error: ShapeError) -> Self {
        Self::Broadcast {
            detail: Box::new(BroadcastFailure::new(
                Shape::scalar(),
                Shape::scalar(),
                error,
            )),
        }
    }
}
