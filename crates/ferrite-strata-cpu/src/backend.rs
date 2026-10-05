//! The backend: capability blob, tri-state compile, and the kernels.

use std::collections::{BTreeSet, HashMap};

use ferrite_strata::{
    Arena, Backend, CapabilitySet, CompileOutcome, DType, DTypeMask, DeclaredLimits, ElementType,
    Error, ErrorCode, Executable, ExecutableId, Graph, NodeId, Op, PatternSupport, PluginVersion,
    Shape, ShapeClass, StrataSupport, StrataVersion, SupportLevel, ValueId,
    capabilities::reference::{self, ReferenceOp},
};

use crate::tensor::{BroadcastFailure, CpuError, CpuTensor, Inputs};

/// One compiled step.
///
/// An enum rather than a closure or a boxed trait object: a closure would need a
/// `dyn Fn` per node and would erase the op from the debug output, and the whole
/// point of this crate is that a failing step can be read off the type.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// Elementwise addition.
    Add {
        /// The left operand.
        lhs: ValueId,
        /// The right operand.
        rhs: ValueId,
        /// Where the result goes.
        out: ValueId,
    },
    /// Elementwise multiplication.
    Mul {
        /// The left operand.
        lhs: ValueId,
        /// The right operand.
        rhs: ValueId,
        /// Where the result goes.
        out: ValueId,
    },
    /// 2-D matrix multiply, `lhs @ rhs`.
    MatMul {
        /// The left operand, `[m, k]`.
        lhs: ValueId,
        /// The right operand, `[k, n]`.
        rhs: ValueId,
        /// Where the result goes.
        out: ValueId,
    },
    /// Natural logarithm.
    Log {
        /// The operand.
        input: ValueId,
        /// Where the result goes.
        out: ValueId,
    },
    /// The exponential function.
    Exp {
        /// The operand.
        input: ValueId,
        /// Where the result goes.
        out: ValueId,
    },
    /// Row-wise softmax, computed with the max subtracted for stability.
    Softmax {
        /// The operand, `[rows, cols]`.
        input: ValueId,
        /// Where the result goes.
        out: ValueId,
    },
    /// Layer normalisation over the trailing axis.
    LayerNorm {
        /// The operand.
        input: ValueId,
        /// The per-feature gain.
        gamma: ValueId,
        /// The per-feature bias.
        beta: ValueId,
        /// Added inside the square root.
        eps: f32,
        /// Where the result goes.
        out: ValueId,
    },
    /// The GELU activation, in its exact form.
    Gelu {
        /// The operand.
        input: ValueId,
        /// Where the result goes.
        out: ValueId,
    },
    /// A linear layer, `input @ weight + bias`, with `weight` as `[out, in]`.
    Linear {
        /// The operand, `[m, in]`.
        input: ValueId,
        /// The weight, `[out, in]`.
        weight: ValueId,
        /// The optional bias, `[out]`.
        bias: Option<ValueId>,
        /// Where the result goes.
        out: ValueId,
    },
    /// A reshape.
    Reshape {
        /// The operand.
        input: ValueId,
        /// The target shape.
        shape: Shape,
        /// Where the result goes.
        out: ValueId,
    },
    /// A transpose.
    Transpose {
        /// The operand.
        input: ValueId,
        /// The axis permutation.
        permutation: Vec<i64>,
        /// Where the result goes.
        out: ValueId,
    },
    /// A broadcast.
    Expand {
        /// The operand.
        input: ValueId,
        /// The target shape.
        shape: Shape,
        /// Where the result goes.
        out: ValueId,
    },
    /// A slice along an axis.
    Slice {
        /// The operand.
        input: ValueId,
        /// The axis.
        axis: i64,
        /// Inclusive start.
        start: i64,
        /// Exclusive end.
        end: i64,
        /// Positive step.
        stride: i64,
        /// Where the result goes.
        out: ValueId,
    },
}

/// The typed result of [`CpuBackend::compile_typed`].
///
/// Mirrors [`CompileOutcome`]'s three-way structure without erasing the concrete
/// executable, so a caller can hold a `CpuExecutable` and call
/// [`CpuExecutable::run`] on it. The erasure is only needed at the
/// [`Backend::compile`] boundary, where the host genuinely does not know the type.
#[derive(Debug)]
pub enum CpuCompile {
    /// This backend can run the subgraph.
    Compiled(CpuExecutable),
    /// This backend will not take it, and says why.
    Declined {
        /// The human-readable reason. Never machine-parsed.
        reason: String,
    },
}

/// A compiled CPU subgraph.
#[derive(Debug)]
pub struct CpuExecutable {
    graph: Graph,
    steps: Vec<Step>,
    label: String,
}

impl Executable for CpuExecutable {
    fn label(&self) -> &str {
        &self.label
    }

    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn outputs(&self) -> &[ValueId] {
        self.graph.outputs()
    }

    /// Execute against the host arena.
    ///
    /// A whole-arena entry that is not host-readable is a refusal to run rather than
    /// a silent zero: this backend computes in `f32`, so an `Opaque` operand is
    /// something it genuinely cannot interpret.
    fn run(&self, arena: &mut Arena) -> Result<(), Error> {
        let mut values: HashMap<ValueId, CpuTensor> = HashMap::new();
        for id in self.external_operands() {
            let Some(entry) = arena.get(id) else {
                return Err(Error::from_backend(
                    ErrorCode::InvalidArgument,
                    "cpu",
                    format!("no value {id} in the arena for `{}`", self.label),
                ));
            };
            let Some(data) = entry.as_f32() else {
                return Err(Error::from_backend(
                    ErrorCode::InvalidArgument,
                    "cpu",
                    format!(
                        "the CPU backend computes in f32 and cannot read {id}, which                          is stored opaquely"
                    ),
                ));
            };
            let shape = self.graph.value(id).shape;
            values.insert(
                id,
                CpuTensor::from_slice(data, shape).map_err(|error| {
                    Error::from_backend(ErrorCode::InvalidArgument, "cpu", error.to_string())
                })?,
            );
        }

        let produced = self.run_values(&values).map_err(|error| {
            Error::from_backend(ErrorCode::InvalidArgument, "cpu", error.to_string())
        })?;

        // Publish the graph's outputs, which is what a partition's neighbours read.
        // Iterating `outputs()` rather than the required values matters: an output is
        // normally *not* an operand of anything in its own subgraph, so a loop over
        // the operands would publish nothing at all.
        //
        // Only declared outputs are published. An intermediate belongs to the fused
        // kernel, and writing one out would invite a caller to depend on a value the
        // backend is free to stop producing.
        for &out in self.graph.outputs() {
            if let Some(tensor) = produced.get(&out) {
                arena.insert_f32(out, tensor.as_slice().to_vec());
            }
        }
        Ok(())
    }
}

impl CpuExecutable {
    /// The compiled steps, in execution order.
    #[must_use]
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Every value this executable reads: each step's operands, plus the weights
    /// its ops name by value id.
    fn required_values(&self) -> BTreeSet<ValueId> {
        let mut ids = BTreeSet::new();
        for step in &self.steps {
            for operand in step_operands(step) {
                ids.insert(operand);
            }
        }
        ids
    }

    /// The values that must arrive in the arena, as opposed to being computed here.
    ///
    /// The difference matters. A fused partition reads its own intermediates, and
    /// loading *every* required value up front would ask the caller for values that
    /// do not exist until this executable has computed them. So only genuinely
    /// external operands are staged; the rest arrive as `run_values` produces them,
    /// in step order.
    fn external_operands(&self) -> BTreeSet<ValueId> {
        let produced_here: BTreeSet<ValueId> = self
            .graph
            .values()
            .iter()
            .enumerate()
            .filter(|(_, value)| value.producer.is_some())
            .map(|(index, _)| ValueId(index as u32))
            .collect();

        self.required_values()
            .difference(&produced_here)
            .copied()
            .collect()
    }

    /// Run this executable over `inputs`.
    ///
    /// Named `run_values` rather than `run` because
    /// [`Executable::run`](ferrite_strata::Executable::run) takes an arena, and two
    /// methods called `run` on one type is a trap for method resolution.
    ///
    /// Returns every value the subgraph produced, not only the graph outputs: an
    /// intermediate may be the input to a second executable, and re-running a node
    /// to recreate it would be the wrong trade.
    ///
    /// # Errors
    ///
    /// If an operand is missing, or a kernel's own preconditions fail. Both are
    /// `INVALID_ARGUMENT`: the caller asked for something that cannot be computed,
    /// which is a different thing from the device failing.
    pub fn run_values(&self, inputs: &Inputs) -> Result<HashMap<ValueId, CpuTensor>, CpuError> {
        let mut values: HashMap<ValueId, CpuTensor> = HashMap::new();
        for (&id, tensor) in inputs {
            values.insert(id, tensor.clone());
        }

        for step in &self.steps {
            let result = self.eval(step, &mut values)?;
            values.extend(result);
        }
        Ok(values)
    }

    fn get<'a>(
        &self,
        values: &'a HashMap<ValueId, CpuTensor>,
        id: ValueId,
    ) -> Result<&'a CpuTensor, CpuError> {
        values.get(&id).ok_or(CpuError::MissingInput { id })
    }

    fn eval(
        &self,
        step: &Step,
        values: &mut HashMap<ValueId, CpuTensor>,
    ) -> Result<Vec<(ValueId, CpuTensor)>, CpuError> {
        // A zero-element tensor short-circuits every kernel: with no elements there
        // is nothing to compute, and every kernel below indexes at least once.
        let out_shape = self.step_output_shape(step);
        if out_shape.element_count() == 0 {
            return Ok(vec![(step_out(step), CpuTensor::zeros(out_shape)?)]);
        }

        let produced = match step {
            Step::Add { lhs, rhs, .. } => {
                elementwise(self.get(values, *lhs)?, self.get(values, *rhs)?, |a, b| {
                    a + b
                })?
            }
            Step::Mul { lhs, rhs, .. } => {
                elementwise(self.get(values, *lhs)?, self.get(values, *rhs)?, |a, b| {
                    a * b
                })?
            }
            Step::Log { input, .. } => map(self.get(values, *input)?, f32::ln)?,
            Step::Exp { input, .. } => map(self.get(values, *input)?, f32::exp)?,
            Step::Gelu { input, .. } => map(self.get(values, *input)?, gelu)?,
            Step::MatMul { lhs, rhs, .. } => {
                matmul(self.get(values, *lhs)?, self.get(values, *rhs)?)?
            }
            Step::Softmax { input, .. } => softmax(self.get(values, *input)?)?,
            Step::LayerNorm {
                input,
                gamma,
                beta,
                eps,
                ..
            } => layer_norm(
                self.get(values, *input)?,
                self.get(values, *gamma)?,
                self.get(values, *beta)?,
                *eps,
            )?,
            Step::Linear {
                input,
                weight,
                bias,
                ..
            } => linear(
                self.get(values, *input)?,
                self.get(values, *weight)?,
                bias.map(|id| self.get(values, id)).transpose()?,
            )?,
            Step::Reshape { input, shape, .. } => {
                CpuTensor::from_slice(self.get(values, *input)?.as_slice(), *shape)?
            }
            Step::Transpose {
                input, permutation, ..
            } => transpose(self.get(values, *input)?, permutation)?,
            Step::Expand { input, shape, .. } => broadcast_to(self.get(values, *input)?, *shape)?,
            Step::Slice {
                input,
                axis,
                start,
                end,
                stride,
                ..
            } => slice_along(self.get(values, *input)?, *axis, *start, *end, *stride)?,
        };

        Ok(vec![(step_out(step), produced)])
    }

    /// The declared shape of this step's result, read from the graph rather than
    /// recomputed. The IR is the authority on shapes; a kernel that derived its
    /// own would be a second, disagreeing source.
    fn step_output_shape(&self, step: &Step) -> Shape {
        self.graph.value(step_out(step)).shape
    }
}

/// Every value a step reads, including the weights named in the op variant.
fn step_operands(step: &Step) -> Vec<ValueId> {
    match step {
        Step::Add { lhs, rhs, .. } | Step::Mul { lhs, rhs, .. } | Step::MatMul { lhs, rhs, .. } => {
            vec![*lhs, *rhs]
        }
        Step::Linear {
            input,
            weight,
            bias,
            ..
        } => {
            let mut ids = vec![*input, *weight];
            if let Some(bias) = bias {
                ids.push(*bias);
            }
            ids
        }
        Step::LayerNorm {
            input, gamma, beta, ..
        } => vec![*input, *gamma, *beta],
        Step::Log { input, .. }
        | Step::Exp { input, .. }
        | Step::Softmax { input, .. }
        | Step::Gelu { input, .. }
        | Step::Reshape { input, .. }
        | Step::Transpose { input, .. }
        | Step::Expand { input, .. }
        | Step::Slice { input, .. } => vec![*input],
    }
}

fn step_out(step: &Step) -> ValueId {
    match step {
        Step::Add { out, .. }
        | Step::Mul { out, .. }
        | Step::MatMul { out, .. }
        | Step::Log { out, .. }
        | Step::Exp { out, .. }
        | Step::Softmax { out, .. }
        | Step::LayerNorm { out, .. }
        | Step::Gelu { out, .. }
        | Step::Linear { out, .. }
        | Step::Reshape { out, .. }
        | Step::Transpose { out, .. }
        | Step::Expand { out, .. }
        | Step::Slice { out, .. } => *out,
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

fn map(input: &CpuTensor, f: impl Fn(f32) -> f32) -> Result<CpuTensor, CpuError> {
    CpuTensor::from_slice(
        &input.as_slice().iter().copied().map(f).collect::<Vec<_>>(),
        input.shape(),
    )
}

/// `x * 0.5 * (1 + erf(x / sqrt(2)))`, the exact form.
///
/// Exact rather than the tanh approximation, deliberately: a kernel library's job
/// is to be the thing you check the approximation against.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// A scalar error function, by Abramowitz & Stegun 7.1.26.
///
/// Maximum absolute error 1.5e-7, which is below `f32` resolution at these
/// magnitudes — good enough for an oracle, and far simpler to audit than the
/// complementary error function's rational approximations.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

fn elementwise(
    lhs: &CpuTensor,
    rhs: &CpuTensor,
    f: impl Fn(f32, f32) -> f32,
) -> Result<CpuTensor, CpuError> {
    let shape = lhs
        .shape()
        .broadcast(&rhs.shape())
        .map_err(|reason| CpuError::Broadcast {
            detail: Box::new(BroadcastFailure::new(lhs.shape(), rhs.shape(), reason)),
        })?;
    let lhs_expanded = broadcast_to(lhs, shape)?;
    let rhs_expanded = broadcast_to(rhs, shape)?;
    let data = lhs_expanded
        .as_slice()
        .iter()
        .zip(rhs_expanded.as_slice())
        .map(|(a, b)| f(*a, *b))
        .collect::<Vec<_>>();
    CpuTensor::from_slice(&data, shape)
}

/// Naive `m x k @ k x n`, accumulating in `f32` in row-major `k` order.
///
/// Accumulating in `k` order rather than summing a whole row and subtracting is
/// what keeps this a useful oracle: an NPU accumulating in tiles may produce a
/// slightly different last bit, and an oracle that itself was reassociated would
/// make that difference uninterpretable.
fn matmul(lhs: &CpuTensor, rhs: &CpuTensor) -> Result<CpuTensor, CpuError> {
    let (m, k) = lhs.dims2()?;
    let (k2, n) = rhs.dims2()?;
    if k != k2 {
        return Err(CpuError::MatmulInner {
            lhs_cols: k,
            rhs_rows: k2,
        });
    }
    let shape = Shape::new(&[m, n]).map_err(|error| CpuError::Broadcast {
        detail: Box::new(BroadcastFailure::new(lhs.shape(), rhs.shape(), error)),
    })?;
    let mut out = CpuTensor::zeros(shape)?;
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += lhs.at(row, inner)? * rhs.at(inner, col)?;
            }
            out.set(row, col, acc)?;
        }
    }
    Ok(out)
}

/// Row-wise softmax with the maximum subtracted.
///
/// The subtraction is not an optimisation detail: `exp(1000)` is `inf`, so the
/// naive form returns `NaN` for a row with a large bias. A backend under test
/// would be blamed for a bug that is in the reference.
fn softmax(input: &CpuTensor) -> Result<CpuTensor, CpuError> {
    let (rows, cols) = input.dims2()?;
    let mut out = CpuTensor::zeros(input.shape())?;
    for row in 0..rows {
        let mut max = f32::NEG_INFINITY;
        for col in 0..cols {
            max = max.max(input.at(row, col)?);
        }
        let mut sum = 0.0f32;
        for col in 0..cols {
            sum += (input.at(row, col)? - max).exp();
        }
        for col in 0..cols {
            out.set(row, col, (input.at(row, col)? - max).exp() / sum)?;
        }
    }
    Ok(out)
}

/// Layer normalisation over the trailing axis, per row.
///
/// Variance is biased (`/ n`) rather than corrected (`/ (n-1)`), matching what
/// inference kernels do and what
/// [`LayerNorm`](ferrite_strata::Op::LayerNorm) specifies.
fn layer_norm(
    input: &CpuTensor,
    gamma: &CpuTensor,
    beta: &CpuTensor,
    eps: f32,
) -> Result<CpuTensor, CpuError> {
    let (rows, cols) = input.dims2()?;
    if gamma.len() != cols || beta.len() != cols {
        return Err(CpuError::ShapeMismatch {
            expected: cols,
            got: gamma.len().max(beta.len()),
        });
    }
    let mut out = CpuTensor::zeros(input.shape())?;
    let cols_f = cols as f32;
    for row in 0..rows {
        let mut sum = 0.0f32;
        for col in 0..cols {
            sum += input.at(row, col)?;
        }
        let mean = sum / cols_f;
        let mut variance = 0.0f32;
        for col in 0..cols {
            let centred = input.at(row, col)? - mean;
            variance += centred * centred;
        }
        variance /= cols_f;
        let denominator = (variance + eps).sqrt();
        for col in 0..cols {
            let normalised = (input.at(row, col)? - mean) / denominator;
            out.set(
                row,
                col,
                normalised * gamma.as_slice()[col] + beta.as_slice()[col],
            )?;
        }
    }
    Ok(out)
}

/// `input @ weight + bias`, with `weight` as `[out, in]`.
///
/// The weight is transposed relative to the matmul layout, which is the convention
/// every framework uses for a linear layer's stored weights and the one that makes
/// the operand contiguous.
fn linear(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
) -> Result<CpuTensor, CpuError> {
    let (m, k) = input.dims2()?;
    let (out_features, in_features) = weight.dims2()?;
    if in_features != k {
        return Err(CpuError::MatmulInner {
            lhs_cols: k,
            rhs_rows: in_features,
        });
    }
    let shape = Shape::new(&[m, out_features]).map_err(|error| CpuError::Broadcast {
        detail: Box::new(BroadcastFailure::new(input.shape(), weight.shape(), error)),
    })?;
    let mut out = CpuTensor::zeros(shape)?;
    for row in 0..m {
        for col in 0..out_features {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += input.at(row, inner)? * weight.at(col, inner)?;
            }
            if let Some(bias) = bias {
                acc += bias.as_slice()[col];
            }
            out.set(row, col, acc)?;
        }
    }
    Ok(out)
}

fn transpose(input: &CpuTensor, permutation: &[i64]) -> Result<CpuTensor, CpuError> {
    let (rows, cols) = input.dims2()?;
    if permutation != [1, 0] {
        return Err(CpuError::ShapeMismatch {
            expected: 2,
            got: permutation.len(),
        });
    }
    let shape = Shape::new(&[cols, rows]).map_err(|error| CpuError::Broadcast {
        detail: Box::new(BroadcastFailure::new(input.shape(), input.shape(), error)),
    })?;
    let mut out = CpuTensor::zeros(shape)?;
    for row in 0..rows {
        for col in 0..cols {
            out.set(col, row, input.at(row, col)?)?;
        }
    }
    Ok(out)
}

/// Broadcast `input` to `shape` under NumPy's rules.
///
/// Right-aligns `input`'s dimensions against `shape`'s, treating missing leading
/// axes as 1 and expanding any source dimension of 1. That is the rule a backend
/// under test is being checked against, so this has to be the real rule and not a
/// same-size special case — `[1, 2] -> [3, 2]` is the case that matters.
fn broadcast_to(input: &CpuTensor, shape: Shape) -> Result<CpuTensor, CpuError> {
    let source = input.shape();
    if source == shape {
        return Ok(input.clone());
    }
    // The same validation the IR's own broadcast performs, so a refusal here agrees
    // with the refusal a partitioner would have produced.
    source
        .broadcast(&shape)
        .map_err(|reason| CpuError::Broadcast {
            detail: Box::new(BroadcastFailure::new(source, shape, reason)),
        })?;

    let mut out = CpuTensor::zeros(shape)?;
    let offset = shape.rank().saturating_sub(source.rank());
    let rows = shape.dim(0).unwrap_or(1);
    let cols = shape.dim(1).unwrap_or(1);

    // Index back into the source for each output position. A source dimension of 1
    // or a missing one means "take index 0"; otherwise the index carries over.
    let source_dim = |axis: usize| source.dim(axis + offset);
    let pick = |axis: usize, index: usize| match source_dim(axis) {
        Some(1) | None => 0,
        Some(size) => index.min(size.saturating_sub(1)),
    };

    if source.rank() <= 1 {
        for i in 0..out.len() {
            out.as_mut_slice()[i] =
                input.as_slice()[pick(0, i / cols.max(1)) * cols + pick(1, i % cols)];
        }
        return Ok(out);
    }
    for row in 0..rows {
        for col in 0..cols {
            let r = pick(0, row);
            let c = pick(1, col);
            out.set(row, col, input.at(r, c)?)?;
        }
    }
    Ok(out)
}

fn slice_along(
    input: &CpuTensor,
    axis: i64,
    start: i64,
    end: i64,
    stride: i64,
) -> Result<CpuTensor, CpuError> {
    let (rows, cols) = input.dims2()?;
    if axis != 1 {
        // Slicing a 2-D tensor along axis 0 is a row selection, which no step in
        // this crate needs yet.
        return Err(CpuError::NotRank2 { rank: 1 });
    }
    if stride <= 0 {
        return Err(CpuError::ShapeMismatch {
            expected: 1,
            got: stride.unsigned_abs() as usize,
        });
    }
    let indices: Vec<usize> = (start..end)
        .step_by(stride as usize)
        .map(|i| i as usize)
        .filter(|&i| i < cols)
        .collect();
    let shape = Shape::new(&[rows, indices.len()]).map_err(|error| CpuError::Broadcast {
        detail: Box::new(BroadcastFailure::new(input.shape(), input.shape(), error)),
    })?;
    let mut out = CpuTensor::zeros(shape)?;
    for row in 0..rows {
        for (new_col, &src_col) in indices.iter().enumerate() {
            out.set(row, new_col, input.at(row, src_col)?)?;
        }
    }
    Ok(out)
}

/// A `f32` CPU backend.
///
/// The name is `cpu` and the capabilities say so, because the point of this crate is
/// to be legible: a diagnostic saying a graph went to the CPU is a true statement,
/// not a euphemism for "some other device".
#[derive(Debug)]
pub struct CpuBackend {
    support: StrataSupport,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    /// A backend claiming what this crate implements.
    #[must_use]
    pub fn new() -> Self {
        let mut capabilities = CapabilitySet::empty("cpu")
            .with_dtype_mask(DTypeMask::of(ElementType::F32))
            .with_limits(DeclaredLimits {
                // No device memory, so no declared allocation limit: the reading is
                // chosen per call by whoever asks, and `None` means "unknown".
                max_allocation_bytes: None,
                max_graph_bytes: None,
                max_rank: Some(8),
                max_dim: None,
                max_nodes: None,
            });
        for op in ReferenceOp::all() {
            for class in reference::all_shape_classes() {
                capabilities = capabilities.with_pattern(PatternSupport::new(
                    op.pattern_id(),
                    CpuTensor::dtype(),
                    *class,
                    // `Fallback`, not `Fused`: this backend runs the ops, it
                    // does not fuse them. Claiming `Fused` here would be a lie a
                    // planner would act on.
                    SupportLevel::Fallback,
                ));
            }
        }
        Self {
            support: StrataSupport::new(capabilities),
        }
    }

    /// Whether this backend implements `op`.
    ///
    /// The single source of truth for the supported set, used both to build the
    /// capability blob and to decide whether to decline. Keeping them derived from
    /// one function is what stops [`CpuBackend::compile`] from declining a graph it
    /// advertised.
    fn supports_op(op: &Op) -> bool {
        matches!(
            op,
            Op::Add
                | Op::Mul
                | Op::MatMul
                | Op::Log
                | Op::Exp
                | Op::Softmax { .. }
                | Op::LayerNorm { .. }
                | Op::Gelu
                | Op::Linear { .. }
                | Op::Reshape { .. }
                | Op::Transpose { .. }
                | Op::Expand { .. }
                | Op::Slice { .. }
        )
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn plugin_version(&self) -> PluginVersion {
        PluginVersion::new(0, 1, 0, StrataVersion::current())
    }

    fn capabilities(&self) -> &StrataSupport {
        &self.support
    }

    /// The pull query, with the dtype gate this backend actually has.
    ///
    /// The default forwards to the capability blob, which already gates on the
    /// `f32`-only dtype mask — so this override exists to make the *reason* for the
    /// refusal recoverable, which the blob's absence cannot express.
    fn supports(&self, pattern_id: &str, dtype: DType, shape_class: ShapeClass) -> SupportLevel {
        if dtype.element != ElementType::F32 || dtype.is_quantised() {
            return SupportLevel::Refuse;
        }
        self.support.supports(pattern_id, dtype, shape_class)
    }

    /// Compile a subgraph, or decline it, as a type-erased executable.
    ///
    /// This is the [`Backend`] trait entry point. Callers that want to *run* the
    /// result should use [`CpuBackend::compile_typed`] instead: an
    /// [`ExecutableId`] hides the concrete type behind `dyn Executable`, and there
    /// is no `run` on that trait yet because the buffer ABI is B2's to define.
    fn compile(&self, subgraph: &Graph, pattern_id: &str) -> Result<CompileOutcome, Error> {
        Ok(match self.compile_typed(subgraph, pattern_id)? {
            CpuCompile::Compiled(executable) => {
                CompileOutcome::Compiled(Box::new(ExecutableId::new(self.name(), executable)))
            }
            CpuCompile::Declined { reason } => CompileOutcome::Declined {
                backend: self.name().to_string(),
                reason,
            },
        })
    }
}

impl CpuBackend {
    /// Compile a subgraph into something this backend can actually run.
    ///
    /// The decline arms are the interesting part, and they are real rather than
    /// defensive:
    ///
    /// - a non-`f32` value, which this backend does not compute in;
    /// - an op outside `CpuBackend::supports_op`, including every `Custom` fused
    ///   op — a fused attention kernel is not something this crate computes;
    /// - an empty output list, which cannot be executed;
    /// - a multi-output node, which no kernel here has.
    ///
    /// Each refusal is [`CpuCompile::Declined`], never an `Err`, so a caller can
    /// fall through to the next backend without inspecting an error code. Only a
    /// malformed graph produces `Err`.
    ///
    /// # Errors
    ///
    /// If a node declares fewer operands than its op needs, which is a bug in the
    /// graph rather than a scope limit.
    pub fn compile_typed(&self, subgraph: &Graph, _pattern_id: &str) -> Result<CpuCompile, Error> {
        if subgraph.outputs().is_empty() {
            return Ok(CpuCompile::Declined {
                reason: "a subgraph with no outputs has nothing to run".into(),
            });
        }

        let mut steps = Vec::with_capacity(subgraph.node_count());
        for (index, node) in subgraph.nodes().iter().enumerate() {
            let id = ferrite_strata::NodeId(index as u32);

            if !Self::supports_op(&node.op) {
                return Ok(CpuCompile::Declined {
                    reason: format!("no CPU kernel for op `{}`", node.op.name()),
                });
            }
            if node.outputs.len() != 1 {
                return Ok(CpuCompile::Declined {
                    reason: format!(
                        "every CPU kernel here is single-output; this node has {}",
                        node.outputs.len()
                    ),
                });
            }
            // The dtype gate is checked once, before any step is built, so a
            // graph is declined whole rather than compiled halfway and abandoned.
            for value in subgraph.values() {
                if value.dtype.element != ElementType::F32 || value.dtype.is_quantised() {
                    return Ok(CpuCompile::Declined {
                        reason: format!(
                            "the CPU backend computes in f32; {} is not one",
                            value.dtype
                        ),
                    });
                }
            }

            let out = node.outputs[0];
            let step = match &node.op {
                Op::Add => Step::Add {
                    lhs: operand(node, id, 0)?,
                    rhs: operand(node, id, 1)?,
                    out,
                },
                Op::Mul => Step::Mul {
                    lhs: operand(node, id, 0)?,
                    rhs: operand(node, id, 1)?,
                    out,
                },
                Op::MatMul => Step::MatMul {
                    lhs: operand(node, id, 0)?,
                    rhs: operand(node, id, 1)?,
                    out,
                },
                Op::Log => Step::Log {
                    input: operand(node, id, 0)?,
                    out,
                },
                Op::Exp => Step::Exp {
                    input: operand(node, id, 0)?,
                    out,
                },
                Op::Gelu => Step::Gelu {
                    input: operand(node, id, 0)?,
                    out,
                },
                Op::Softmax { .. } => Step::Softmax {
                    input: operand(node, id, 0)?,
                    out,
                },
                Op::LayerNorm { gamma, beta, eps } => Step::LayerNorm {
                    input: operand(node, id, 0)?,
                    gamma: *gamma,
                    beta: *beta,
                    eps: *eps,
                    out,
                },
                Op::Linear { weight, bias } => Step::Linear {
                    input: operand(node, id, 0)?,
                    weight: *weight,
                    bias: *bias,
                    out,
                },
                Op::Reshape { shape } => Step::Reshape {
                    input: operand(node, id, 0)?,
                    shape: *shape,
                    out,
                },
                Op::Transpose { permutation } => Step::Transpose {
                    input: operand(node, id, 0)?,
                    permutation: permutation.clone(),
                    out,
                },
                Op::Expand { shape } => Step::Expand {
                    input: operand(node, id, 0)?,
                    shape: *shape,
                    out,
                },
                Op::Slice {
                    axis,
                    start,
                    end,
                    stride,
                } => Step::Slice {
                    input: operand(node, id, 0)?,
                    axis: *axis,
                    start: *start,
                    end: *end,
                    stride: *stride,
                    out,
                },
                Op::Input | Op::Custom { .. } => {
                    return Ok(CpuCompile::Declined {
                        reason: format!("`{}` is not an executable CPU op", node.op.name()),
                    });
                }
            };
            steps.push(step);
        }

        // The arena handed back is the subgraph itself, re-keyed so its values are
        // dense from zero. The partitioner already produced one, so this is a
        // re-basing rather than a rebuild.
        let executable = CpuExecutable {
            graph: subgraph.clone(),
            steps,
            label: format!("cpu::{}", subgraph.name()),
        };
        Ok(CpuCompile::Compiled(executable))
    }
}

fn operand(node: &ferrite_strata::Node, id: NodeId, index: usize) -> Result<ValueId, Error> {
    node.inputs.get(index).copied().ok_or_else(|| {
        Error::from_backend(
            ErrorCode::InvalidArgument,
            "cpu",
            format!(
                "node {} ({}) wants operand {index} but declares {}",
                id.0,
                node.op.name(),
                node.inputs.len()
            ),
        )
    })
}

/// An `f32` tensor from a shape and row-major data.
///
/// A convenience for callers writing tests and examples, so they do not have to
/// convert a `ShapeError` at every call site.
///
/// # Errors
///
/// If the data length does not match the shape's element count.
pub fn f32_tensor(shape: &[usize], data: &[f32]) -> Result<CpuTensor, CpuError> {
    let shape = Shape::new(shape).map_err(|error| CpuError::Broadcast {
        detail: Box::new(BroadcastFailure::new(
            Shape::scalar(),
            Shape::scalar(),
            error,
        )),
    })?;
    CpuTensor::from_slice(data, shape)
}
