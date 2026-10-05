//! Planning and compiling: turning a graph into something runnable.

use ferrite_strata::{
    Arena, CompileOutcome, Error, ErrorCode, ExecutableId, Graph, NodeId, Op, ValueId,
};

use crate::session::{Plan, Session, Unit};

/// One compiled, runnable unit of work.
///
/// The unit is `ExecutableId` for a backend, or a node list for the host. Both are
/// behind one enum so a [`Step`] can hold them in execution order without the caller
/// having to know which is which — and so adding a third kind later is an additive
/// match rather than a new trait.
/// A compiled unit plus the id translation needed to publish its results.
#[derive(Debug)]
struct Work {
    /// The extracted graph, in its own dense id space.
    subgraph: Graph,
    /// How to execute it.
    executor: Executor,
    /// Extracted value id to the original graph's value id.
    ///
    /// Without this a partitioned run writes its outputs under ids the caller has no
    /// way to name, and `Arena::f32_at` on a graph output returns `None` for a run
    /// that actually succeeded.
    back: std::collections::HashMap<ValueId, ValueId>,
}

impl Work {
    /// Run this unit, staging its operands and publishing its outputs.
    ///
    /// The staging is not an optimisation, it is a correctness requirement. The
    /// caller's arena is keyed by the *original* graph's value ids, while this unit's
    /// subgraph is keyed by extracted ids — running the executor against the shared
    /// arena would look up `v1` and find nothing, because the caller wrote `v2`.
    ///
    /// So a unit gets a private arena holding only its own operands, and only its
    /// declared outputs leave. That also means an intermediate never reaches the
    /// caller, which is what lets a backend stop producing one freely.
    fn execute(&self, arena: &mut Arena) -> Result<(), Error> {
        let mut local = Arena::new();
        let operands: Vec<ValueId> = self
            .subgraph
            .nodes()
            .iter()
            .flat_map(|node| node.inputs.iter().copied())
            .collect();
        for extracted in operands {
            // An operand produced by an earlier unit is already in the caller's arena
            // under its original id, published when that unit finished.
            let Some(&original) = self.back.get(&extracted) else {
                continue;
            };
            if let Some(entry) = arena.get(original) {
                local.insert(extracted, entry.clone());
            }
        }

        match &self.executor {
            Executor::Compiled(executable) => executable.run(&mut local)?,
            Executor::Host => run_host_graph(&self.subgraph, &mut local)?,
        }

        for &extracted in self.subgraph.outputs() {
            let Some(&original) = self.back.get(&extracted) else {
                continue;
            };
            if let Some(entry) = local.get(extracted) {
                arena.insert(original, entry.clone());
            }
        }
        Ok(())
    }
}

/// Which executor runs an extracted subgraph.
#[derive(Debug)]
enum Executor {
    /// A backend-compiled subgraph.
    Compiled(Box<ExecutableId>),
    /// Computed by the host's own kernels.
    Host,
}

/// A compiled plan: everything a graph needs to run, in order.
///
/// Compiling produces this and running consumes it, and the split is deliberate. The
/// expensive, failure-prone half is compiling — minutes for a real backend — so it
/// happens once and the result is inspectable before anything runs. A caller that
/// wants to know what a plan would do can ask before committing to it.
#[derive(Debug)]
pub struct Step {
    units: Vec<(Unit, Work)>,
    graph: Graph,
}

impl Step {
    /// How many units this step runs.
    #[must_use]
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// The units, in execution order.
    ///
    /// Reported rather than hidden because "which backend ran this" is the first
    /// question anyone asks when a run is slow.
    #[must_use]
    pub fn units(&self) -> Vec<Unit> {
        self.units.iter().map(|(unit, _)| *unit).collect()
    }

    /// How many of the units are host-computed.
    #[must_use]
    pub fn host_unit_count(&self) -> usize {
        self.units
            .iter()
            .filter(|(unit, _)| matches!(unit, Unit::Host))
            .count()
    }

    /// The graph this step came from.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Run every unit in order over `arena`.
    ///
    /// # Errors
    ///
    /// If any unit fails. The arena is left holding that unit's completed work, so a
    /// caller that wants to retry from a failure can inspect what did run — which is
    /// the reason this is not transactional.
    pub fn run(&self, arena: &mut Arena) -> Result<(), ExecutionError> {
        for (unit, work) in &self.units {
            work.execute(arena)
                .map_err(|error| ExecutionError::Backend { unit: *unit, error })?;
        }
        Ok(())
    }

    /// Run against an arena seeded from plain `f32` inputs.
    ///
    /// A convenience for the common case, and the one the doctests use. Returns the
    /// arena so a caller can read the graph's outputs without knowing their ids.
    ///
    /// # Errors
    ///
    /// See [`Step::run`].
    pub fn run_with_inputs(
        &self,
        inputs: &std::collections::HashMap<ValueId, Vec<f32>>,
    ) -> Result<Arena, ExecutionError> {
        let mut arena = Arena::new();
        for (&id, data) in inputs {
            arena.insert_f32(id, data.clone());
        }
        self.run(&mut arena)?;
        Ok(arena)
    }
}

/// Something that went wrong while running.
#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionError {
    /// A unit failed.
    Backend {
        /// Which unit failed.
        unit: Unit,
        /// What it reported.
        error: Error,
    },
    /// A host-computed node could not run: a missing or unusable value.
    Host {
        /// The value that was needed.
        value: ValueId,
        /// Why it was unusable.
        reason: String,
    },
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::Backend { unit, error } => write!(f, "{unit}: {error}"),
            ExecutionError::Host { value, reason } => {
                write!(f, "host: cannot compute {value}: {reason}")
            }
        }
    }
}

impl std::error::Error for ExecutionError {}

impl ExecutionError {
    /// Whether this error poisoned the device, and so should end the session.
    ///
    /// The host path is CPU arithmetic and cannot poison anything, which is one more
    /// reason a fallback is preferable to a failure.
    #[must_use]
    pub const fn is_device_poisoning(&self) -> bool {
        match self {
            ExecutionError::Backend { error, .. } => error.code.is_device_poisoning(),
            ExecutionError::Host { .. } => false,
        }
    }
}

/// Compile a plan into a runnable [`Step`].
///
/// Fails only on a hard error. A backend that declines at compile time — after the
/// partitioner said it would accept — is not a failure: its partition falls back to
/// the host, which is the same path a declined partition takes during planning.
pub(crate) fn compile_plan(session: &Session, plan: &Plan, graph: &Graph) -> Result<Step, Error> {
    let mut units: Vec<(Unit, Work)> = Vec::new();

    for partition in &plan.inner.partitions {
        let backend = session.backend(partition.backend);
        let (subgraph, back) = graph
            .subgraph_with_map(&partition.nodes)
            .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?;

        let executor = match backend.compile(&subgraph, &partition.pattern_id)? {
            CompileOutcome::Compiled(executable) => Executor::Compiled(executable),
            CompileOutcome::Declined { .. } => {
                // The capability query said yes and the compiler said no. That
                // happens — a vendor's table is a summary, its compiler is the
                // authority — and the honest response is to compute those nodes on
                // the host rather than fail a plan that was otherwise fine.
                //
                // This is the design working: being wrong about support costs time,
                // not correctness and not an error.
                Executor::Host
            }
        };

        units.push((
            Unit::Backend(partition.backend),
            Work {
                subgraph,
                executor,
                back,
            },
        ));
    }

    if !plan.inner.host_nodes.is_empty() {
        let (subgraph, back) = graph
            .subgraph_with_map(&plan.inner.host_nodes)
            .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?;
        units.push((
            Unit::Host,
            Work {
                subgraph,
                executor: Executor::Host,
                back,
            },
        ));
    }

    Ok(Step {
        units,
        graph: graph.clone(),
    })
}

/// Run every node of an extracted subgraph on the host, in place.
///
/// The arena arrives already holding this subgraph's operands, so this only computes
/// and writes. Outputs stay under the subgraph's own ids; [`Work::execute`] is what
/// translates them back.
fn run_host_graph(subgraph: &Graph, arena: &mut Arena) -> Result<(), Error> {
    for index in 0..subgraph.node_count() {
        host_node(subgraph, NodeId(index as u32), arena)?;
    }
    Ok(())
}

/// Compute one node into `arena`.
fn host_node(graph: &Graph, node: NodeId, arena: &mut Arena) -> Result<(), Error> {
    let reference = graph.node(node);
    let Some(&output) = reference.outputs.first() else {
        return Ok(());
    };

    let mut operands: Vec<&[f32]> = Vec::with_capacity(reference.inputs.len());
    for &id in &reference.inputs {
        let data = arena.f32_at(id).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidArgument,
                format!("{id} is not in the arena, or is not readable as f32"),
            )
        })?;
        operands.push(data);
    }

    let computed = match &reference.op {
        Op::Gelu => map1(operands[0], gelu),
        Op::Exp => map1(operands[0], f32::exp),
        Op::Log => map1(operands[0], f32::ln),
        Op::Add => zip2(operands[0], operands[1], |a, b| a + b),
        Op::Mul => zip2(operands[0], operands[1], |a, b| a * b),
        other => {
            return Err(Error::new(
                ErrorCode::Unimplemented,
                format!(
                    "the host path has no kernel for `{}`. Nodes reach the host only \
                     because no backend claimed them, so this is a gap in the host \
                     fallback rather than an expected condition.",
                    other.name()
                ),
            ));
        }
    };

    arena.insert_f32(output, computed);
    Ok(())
}

fn map1(input: &[f32], f: impl Fn(f32) -> f32) -> Vec<f32> {
    input.iter().copied().map(f).collect()
}

fn zip2(lhs: &[f32], rhs: &[f32], f: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    let n = lhs.len().min(rhs.len());
    (0..n).map(|i| f(lhs[i], rhs[i])).collect()
}

/// The exact GELU, matching [`ferrite_strata_cpu`]'s kernel.
///
/// Duplicated rather than shared because a dependency on the CPU backend would invert
/// the layering: the CPU backend is one *provider* of host execution, and the host
/// path has to work with no backends registered at all. Two copies of one erf is a
/// cheaper price than a cycle, and the cross-check test keeps them honest.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

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
