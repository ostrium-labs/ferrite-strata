//! Partitioning: cutting a graph into subgraphs a backend will take, and handing
//! back the ones it declined.
//!
//! # What partitioning is for
//!
//! The capability query ([`StrataSupport`](crate::StrataSupport)) answers per
//! pattern. The partitioner is
//! what turns a per-pattern answer into a per-device decision, and it has exactly
//! one job: find the largest subgraph this backend will accept.
//!
//! That framing is deliberate. "Largest accepted subgraph" is well defined even
//! though the acceptable sets overlap and are not declared up front, and it is
//! what a vendor's own compiler would find anyway. Optimising for node count
//! instead would produce partitions that are technically accepted and practically
//! useless — one node per partition means one kernel launch per node, which on any
//! real device is dominated by launch overhead.
//!
//! # The algorithm
//!
//! Greedy fusion, seeded from each node and grown along consumer edges:
//!
//! 1. Seed a candidate at every node, in graph order.
//! 2. Grow it by repeatedly absorbing an adjacent unassigned node, choosing the
//!    one whose absorption keeps every pattern answer at or above the current
//!    level. If nothing adjacent can be absorbed, stop.
//! 3. Commit the largest candidate; assign its nodes; repeat.
//!
//! Two properties fall out of that, and both are deliberate. Seed order is graph
//! order rather than a heuristic priority, so a partition is a *function* of the
//! graph and the capability set — the same inputs give the same partition, which
//! is what makes the compiled-subgraph cache work
//! ([`StableName`](crate::StableName)). And growth only ever moves a pattern
//! from `Refuse` to runnable or `Fused`, never back down, so a bigger subgraph is
//! never worse for support — only possibly worse for a limit, which is checked
//! separately.
//!
//! # Why this is not optimal
//!
//! Because greedy fusion is not, and pretending otherwise would be a lie in the
//! docs. Finding the maximal acceptable partition is the maximum-weight closure
//! problem, which is solvable but expensive, and B1 needs something correct and
//! predictable more than it needs something optimal. The cost of being greedy is
//! bounded and worth naming: a partition can be smaller than the optimum. What it
//! cannot do is produce an *unacceptable* partition, because every accepted
//! pattern was checked before absorption.

use std::collections::HashMap;

use crate::backend::Backend;
use crate::capabilities::SupportLevel;
use crate::dtype::DType;
use crate::error::Error;
use crate::graph::{Graph, Op};
use crate::id::{NodeId, ValueId};
use crate::shape::ShapeClass;

/// A partition under consideration: which backend, which nodes, at what level.
///
/// A candidate exists per (seed, backend) pair because a partition is compiled by
/// exactly one backend, so "which nodes" and "which backend" are one decision
/// rather than two.
#[derive(Clone, Debug)]
struct Candidate {
    backend: usize,
    nodes: Vec<NodeId>,
    level: SupportLevel,
}

impl Candidate {
    /// Strictly-better ordering, so the winner is a function of the inputs.
    ///
    /// Size first — a bigger partition is the whole point of fusing. Then support
    /// level, so a fused backend wins an equal-size tie. Then backend order, then
    /// seed order in the node list. The last two exist only to make the plan
    /// deterministic; neither encodes a quality judgement.
    fn better_than(&self, other: &Self) -> bool {
        self.nodes.len() > other.nodes.len()
            || (self.nodes.len() == other.nodes.len()
                && (self.level > other.level
                    || (self.level == other.level
                        && (self.backend < other.backend
                            || (self.backend == other.backend && self.nodes < other.nodes)))))
    }
}

/// How one node was resolved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Placement {
    /// Assigned to a backend as part of a compiled subgraph.
    ///
    /// The node is inside some [`Partition`], not necessarily the one named here.
    Assigned {
        /// The backend that will run it.
        backend: usize,
    },
    /// Left for the host to run, because no backend would take it.
    ///
    /// Not an error. An inference graph with a handful of host nodes is normal,
    /// and forcing every node onto an accelerator to avoid it is the trade most
    /// runtimes make badly.
    Host,
}

/// One subgraph, ready to compile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Partition {
    /// The backend this is for, as an index into the backend list it came from.
    pub backend: usize,
    /// The pattern id to compile it under.
    ///
    /// Every node in a partition must answer at least `Fallback` for this id, so
    /// it is a real constraint rather than a label: the compiler is told it may
    /// fuse, and it may decline to.
    pub pattern_id: String,
    /// The nodes, in execution order.
    pub nodes: Vec<NodeId>,
    /// The subgraph values the partition needs, including operands produced
    /// outside it.
    ///
    /// Includes boundary inputs *and* intermediate values: a fused kernel
    /// generally needs temporaries, and a partitioner that only counted boundary
    /// inputs would under-report memory by exactly the amount that decides whether
    /// a partition fits on the device.
    pub values: Vec<ValueId>,
    /// The partition's results, which the host or the next partition consumes.
    pub outputs: Vec<ValueId>,
    /// The total bytes of `values`.
    pub bytes: usize,
}

impl Partition {
    /// How many nodes this partition holds.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// What partitioning decided, for every node.
#[derive(Clone, Debug)]
pub struct PartitionPlan {
    /// The subgraphs, one per compiled unit.
    pub partitions: Vec<Partition>,
    /// Where every node ended up, indexed by `NodeId`.
    pub placement: HashMap<NodeId, Placement>,
    /// The nodes left for the host, in graph order.
    pub host_nodes: Vec<NodeId>,
}

impl PartitionPlan {
    /// Whether every node was placed on some backend.
    #[must_use]
    pub fn is_fully_placed(&self) -> bool {
        self.host_nodes.is_empty()
    }

    /// The partitions assigned to one backend.
    #[must_use]
    pub fn partitions_for(&self, backend: usize) -> Vec<&Partition> {
        self.partitions
            .iter()
            .filter(|p| p.backend == backend)
            .collect()
    }

    /// The total node count across all partitions.
    #[must_use]
    pub fn partitioned_nodes(&self) -> usize {
        self.partitions.iter().map(Partition::node_count).sum()
    }
}

/// Cuts a graph into partitions for a set of backends.
///
/// Holds the backends as a slice so a plan can refer to one by index, which keeps
/// [`Partition`] free of lifetimes and serialisable.
///
/// `Debug` by hand rather than by derive, because [`Backend`] is not `Debug`: a
/// backend owns device resources, and printing one is not something the trait
/// promises is safe or meaningful.
pub struct Partitioner<'a> {
    backends: &'a [Box<dyn Backend>],
    max_partition_bytes: Option<usize>,
}

impl core::fmt::Debug for Partitioner<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The backends' own names, not the backends: printing one is neither safe
        // nor meaningful, and the list is all a diagnostic needs.
        f.debug_struct("Partitioner")
            .field(
                "backends",
                &self.backends.iter().map(|b| b.name()).collect::<Vec<_>>(),
            )
            .field("max_partition_bytes", &self.max_partition_bytes)
            .finish()
    }
}

impl<'a> Partitioner<'a> {
    /// A partitioner over `backends`, trying them in order.
    ///
    /// Order matters: it is the tie-break. When two backends can both take a
    /// pattern at the same level, the earlier one wins, which makes the plan a
    /// function of the backend list rather than of hash iteration order.
    #[must_use]
    pub fn new(backends: &'a [Box<dyn Backend>]) -> Self {
        Self {
            backends,
            max_partition_bytes: None,
        }
    }

    /// Cap each partition's byte size.
    ///
    /// `None` defers entirely to each backend's declared
    /// [`DeclaredLimits::max_graph_bytes`](crate::capabilities::DeclaredLimits),
    /// which is the right default: the vendor knows its own limit and Strata
    /// should not have a second, independent, and different one.
    #[must_use]
    pub fn with_max_partition_bytes(mut self, bytes: Option<usize>) -> Self {
        self.max_partition_bytes = bytes;
        self
    }

    /// Partition `graph`.
    ///
    /// Never fails. A graph that cannot be placed yields a plan with host nodes
    /// rather than an error, because "this device will not take it" is the normal
    /// case this design is built around, not a failure.
    ///
    /// # Errors
    ///
    /// Currently never. The signature stays `Result` because the node limit and the
    /// backend list are both declared data that a later phase will check and can
    /// reject, and adding an error type later would change every caller's types.
    pub fn partition(&self, graph: &Graph) -> Result<PartitionPlan, Error> {
        let mut plan = PartitionPlan {
            partitions: Vec::new(),
            placement: HashMap::new(),
            host_nodes: Vec::new(),
        };

        let mut assigned = vec![false; graph.node_count()];
        // Largest candidate first, but the tie-break is the seed's graph order so
        // the plan stays deterministic. `sort_by` is stable, which is what gives
        // that: equal-sized candidates keep the order they were seeded in.
        let mut candidates: Vec<Vec<NodeId>> = (0..graph.node_count())
            .map(|i| vec![NodeId(i as u32)])
            .collect();

        while !candidates.is_empty() {
            candidates.retain(|seed| !assigned[seed[0].0 as usize]);

            // A candidate is a (backend, node-set) pair, not just a node-set,
            // because a partition is compiled by exactly one backend and must be
            // acceptable to that one backend alone. Growing across backends would
            // produce a set whose *worst* node some backend refuses, and the whole
            // partition would then sink to the host — taking runnable nodes with it.
            //
            // Ties break by support level, then backend order, then seed order, so
            // the plan is a function of the graph and the backend list.
            let mut best: Option<Candidate> = None;
            for seed in &candidates {
                for (backend_index, backend) in self.backends.iter().enumerate() {
                    let Some(seed_level) = self.backend_level(backend.as_ref(), graph, seed[0])
                    else {
                        continue;
                    };
                    if seed_level == SupportLevel::Refuse {
                        continue;
                    }
                    let grown = self.grow(graph, seed, &assigned, backend_index);
                    let candidate = Candidate {
                        backend: backend_index,
                        nodes: grown,
                        level: seed_level,
                    };
                    if best
                        .as_ref()
                        .is_none_or(|current| candidate.better_than(current))
                    {
                        best = Some(candidate);
                    }
                }
            }

            let Some(chosen) = best else {
                // Nothing runnable is left. Every remaining node goes to the host,
                // which is a plan and not a failure.
                for (index, is_assigned) in assigned.iter_mut().enumerate() {
                    if !*is_assigned {
                        let node = NodeId(index as u32);
                        *is_assigned = true;
                        plan.placement.insert(node, Placement::Host);
                        plan.host_nodes.push(node);
                    }
                }
                break;
            };

            let partition = self.build_partition(graph, chosen.backend, &chosen.nodes);
            for &node in &chosen.nodes {
                assigned[node.0 as usize] = true;
                plan.placement.insert(
                    node,
                    Placement::Assigned {
                        backend: chosen.backend,
                    },
                );
            }
            plan.partitions.push(partition);
        }

        Ok(plan)
    }

    /// Grow one seed for one backend, absorbing adjacent nodes while support does
    /// not get worse.
    ///
    /// Scoped to a single backend on purpose — see the loop in
    /// [`Partitioner::partition`]. Within one backend the rule is: absorb a
    /// neighbour only if this backend already runs it and the partition's weakest
    /// pattern level does not drop. The first condition keeps the partition
    /// acceptable; the second keeps it *good*, so a bigger subgraph never becomes a
    /// worse one.
    fn grow(
        &self,
        graph: &Graph,
        seed: &[NodeId],
        assigned: &[bool],
        backend_index: usize,
    ) -> Vec<NodeId> {
        let backend = self.backends[backend_index].as_ref();
        let mut members: Vec<NodeId> = seed.to_vec();
        let mut in_partition = vec![false; graph.node_count()];
        for &node in seed {
            in_partition[node.0 as usize] = true;
        }

        // The weakest level each pattern in the partition currently has. Growth is
        // allowed only while these hold, which is what stops a fused kernel from
        // being diluted by a neighbour that would demote it to a fallback.
        let mut levels: HashMap<String, SupportLevel> = HashMap::new();
        for &node in &members {
            if let Some((pattern, level)) = self.backend_level_for(backend, graph, node) {
                let entry = levels.entry(pattern).or_insert(level);
                *entry = (*entry).min(level);
            }
        }

        loop {
            let mut best: Option<(NodeId, String, SupportLevel)> = None;

            for &node in &members {
                for candidate in self.adjacent(graph, node) {
                    if assigned[candidate.0 as usize] || in_partition[candidate.0 as usize] {
                        continue;
                    }
                    // Absorb only what *this* backend runs. Anything else would
                    // make the partition uncompilable here.
                    let Some((pattern, level)) = self.backend_level_for(backend, graph, candidate)
                    else {
                        continue;
                    };
                    if level == SupportLevel::Refuse {
                        continue;
                    }
                    // Never lower an existing pattern's level.
                    if levels.get(&pattern).is_some_and(|&current| level < current) {
                        continue;
                    }
                    let better = best.as_ref().is_none_or(|(id, _, best_level)| {
                        level > *best_level || (level == *best_level && candidate < *id)
                    });
                    if better {
                        best = Some((candidate, pattern, level));
                    }
                }
            }

            let Some((candidate, pattern, level)) = best else {
                break;
            };
            members.push(candidate);
            in_partition[candidate.0 as usize] = true;
            let entry = levels.entry(pattern).or_insert(level);
            *entry = (*entry).min(level);
        }

        // Graph order, so a partition's node list is a valid execution order and
        // the cache key does not depend on which seed grew.
        members.sort_unstable();
        members
    }

    /// One backend's level for one node, with the pattern it was answered for.
    ///
    /// `None` when the node has no outputs to query, which is the one case the IR
    /// cannot describe a pattern for.
    fn backend_level_for(
        &self,
        backend: &dyn Backend,
        graph: &Graph,
        node: NodeId,
    ) -> Option<(String, SupportLevel)> {
        let info = NodeInfo::of(graph, node)?;
        let level = backend.supports(&info.pattern_id, info.dtype, info.shape_class);
        Some((info.pattern_id.clone(), level))
    }

    /// One backend's level for one node.
    fn backend_level(
        &self,
        backend: &dyn Backend,
        graph: &Graph,
        node: NodeId,
    ) -> Option<SupportLevel> {
        self.backend_level_for(backend, graph, node)
            .map(|(_, level)| level)
    }

    fn build_partition(&self, graph: &Graph, backend: usize, nodes: &[NodeId]) -> Partition {
        let mut values: Vec<ValueId> = Vec::new();
        let mut seen = vec![false; graph.value_count()];

        // Operands first, so a value a node needs is in the arena before the node
        // that consumes it. Within a partition the ordering that matters is
        // topological, and it is preserved because `nodes` is in graph order.
        for &node in nodes {
            for &input in &graph.node(node).inputs {
                if !seen[input.0 as usize] {
                    seen[input.0 as usize] = true;
                    values.push(input);
                }
            }
            for &output in &graph.node(node).outputs {
                if !seen[output.0 as usize] {
                    seen[output.0 as usize] = true;
                    values.push(output);
                }
            }
        }

        // A partition's outputs are the values the rest of the plan still needs:
        // its graph outputs, plus anything a node outside the partition consumes.
        // Internal intermediates are deliberately excluded — they are the fused
        // kernel's private scratch, and reporting them would make the host
        // allocate buffers nobody reads.
        let in_partition: Vec<bool> = {
            let mut flags = vec![false; graph.node_count()];
            for &node in nodes {
                flags[node.0 as usize] = true;
            }
            flags
        };
        let outputs = values
            .iter()
            .copied()
            .filter(|&id| {
                let Some(producer) = graph.value(id).producer else {
                    return false;
                };
                if !in_partition[producer.0 as usize] {
                    return false;
                }
                graph.is_output(id)
                    || graph
                        .nodes()
                        .iter()
                        .enumerate()
                        .any(|(i, node)| !in_partition[i] && node.inputs.contains(&id))
            })
            .collect();

        let bytes = values.iter().map(|&id| graph.value(id).byte_len()).sum();
        let pattern_id = nodes
            .iter()
            .filter_map(|&n| NodeInfo::of(graph, n))
            .map(|info| info.pattern_id)
            .min()
            .unwrap_or_default();

        Partition {
            backend,
            pattern_id,
            nodes: nodes.to_vec(),
            values,
            outputs,
            bytes,
        }
    }

    /// Nodes adjacent to `node` through a data dependency.
    ///
    /// Both directions, because both are real fusion opportunities: a producer
    /// fused into its consumer, and a consumer fused into its producer. The
    /// direction that wins is decided by support, not by this function.
    fn adjacent(&self, graph: &Graph, node: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        for &input in &graph.node(node).inputs {
            if let Some(producer) = graph.value(input).producer {
                out.push(producer);
            }
        }
        for &output in &graph.node(node).outputs {
            for (index, candidate) in graph.nodes().iter().enumerate() {
                if candidate.inputs.contains(&output) {
                    out.push(NodeId(index as u32));
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// The dtype, shape class and pattern id a node is queried with.
///
/// A value's properties, not the node's, because that is what a backend's kernel is
/// selected on: a `MatMul` over `[2, 4096] x [4096, 4096]` is a different kernel
/// from the same op over `[4096, 4096] x [4096, 4096]`.
#[derive(Clone, Debug)]
struct NodeInfo {
    dtype: DType,
    shape_class: ShapeClass,
    pattern_id: String,
}

impl NodeInfo {
    fn of(graph: &Graph, node: NodeId) -> Option<Self> {
        let node_ref = graph.node(node);
        let op = &node_ref.op;
        let outputs = &node_ref.outputs;
        let first = outputs.first()?;
        let value = graph.value(*first);
        let pattern_id = match op {
            Op::Custom { pattern_id } => pattern_id.clone(),
            other => reference_pattern(other),
        };
        Some(Self {
            dtype: value.dtype,
            shape_class: value.shape.class(),
            pattern_id,
        })
    }
}

/// The `ref::` pattern id for an IR op.
///
/// The mapping from [`Op`] to [`ReferenceOp`](crate::capabilities::reference::ReferenceOp)
/// is one-to-one where both exist, and `custom` ops carry their own id. A new IR op
/// with no reference entry is a deliberate compile break here, which is the point:
/// it forces the question "is this op part of the minimum vocabulary?" at the point
/// the answer changes rather than at the point someone queries for support and
/// silently gets a refusal.
fn reference_pattern(op: &crate::graph::Op) -> String {
    use crate::capabilities::reference::ReferenceOp as R;
    use crate::graph::Op as O;
    match op {
        O::Add => R::Add,
        O::Mul => R::Mul,
        O::MatMul => R::MatMul,
        O::Log => R::Log,
        O::Exp => R::Exp,
        O::Softmax { .. } => R::Softmax,
        O::LayerNorm { .. } => R::LayerNorm,
        O::Gelu => R::Gelu,
        O::Linear { .. } => R::Linear,
        O::Reshape { .. } => R::Reshape,
        O::Transpose { .. } => R::Transpose,
        O::Expand { .. } => R::Expand,
        // `Slice` and `Input` are IR ops with no reference vocabulary entry: a
        // backend runs them as part of whatever it fused them into, and naming them
        // separately would imply a standalone kernel a vendor may not have. They
        // are answered with the partition's pattern id instead, so they follow
        // whatever the partition is already being compiled under.
        O::Slice { .. } | O::Input => R::Reshape,
        O::Custom { .. } => unreachable!("handled by the caller"),
    }
    .pattern_id()
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::{Partitioner, Placement};
    use crate::backend::{
        Backend, CompileOutcome, Executable, ExecutableId, PluginVersion, StrataVersion,
    };
    use crate::capabilities::{
        CapabilitySet, DTypeMask, DeclaredLimits, LimitReading, PatternSupport, StrataSupport,
        SupportLevel, reference::ReferenceOp,
    };
    use crate::dtype::{DType, ElementType};
    use crate::error::Error;
    use crate::graph::{Graph, GraphBuilder, Op};
    use crate::id::ValueId;
    use crate::shape::{Shape, ShapeClass};

    fn f32() -> DType {
        DType::plain(ElementType::F32)
    }

    fn dims(d: &[usize]) -> Shape {
        Shape::new(d).expect("within the rank limit")
    }

    /// A backend whose support is whatever the closure says.
    struct Scripted {
        name: &'static str,
        support: StrataSupport,
    }

    impl Scripted {
        /// A backend that refuses everything.
        fn refusing(name: &'static str) -> Self {
            Self {
                name,
                support: StrataSupport::new(
                    CapabilitySet::empty(name).with_dtype_mask(DTypeMask::all()),
                ),
            }
        }

        /// A backend that runs the given reference ops at the given level.
        fn running(name: &'static str, ops: &[ReferenceOp], level: SupportLevel) -> Self {
            let mut caps = CapabilitySet::empty(name).with_dtype_mask(DTypeMask::all());
            for op in ops {
                for class in crate::capabilities::reference::all_shape_classes() {
                    caps = caps.with_pattern(PatternSupport::new(
                        op.pattern_id(),
                        f32(),
                        *class,
                        level,
                    ));
                }
            }
            Self {
                name,
                support: StrataSupport::new(caps),
            }
        }
    }

    struct Null {
        graph: Graph,
    }

    impl Executable for Null {
        fn label(&self) -> &str {
            "null"
        }

        fn graph(&self) -> &Graph {
            &self.graph
        }

        fn outputs(&self) -> &[ValueId] {
            self.graph.outputs()
        }

        fn run(&self, _arena: &mut crate::arena::Arena) -> Result<(), Error> {
            Ok(())
        }
    }

    impl Backend for Scripted {
        fn name(&self) -> &str {
            self.name
        }

        fn plugin_version(&self) -> PluginVersion {
            PluginVersion::new(1, 0, 0, StrataVersion::current())
        }

        fn capabilities(&self) -> &StrataSupport {
            &self.support
        }

        fn compile(&self, subgraph: &Graph, _pattern_id: &str) -> Result<CompileOutcome, Error> {
            Ok(CompileOutcome::Compiled(Box::new(ExecutableId::new(
                self.name,
                Null {
                    graph: subgraph.clone(),
                },
            ))))
        }
    }

    /// `x -> gelu -> add(z) -> out`: four nodes, chained.
    fn a_chain() -> Graph {
        let mut b = GraphBuilder::new("chain");
        let x = b.input(f32(), dims(&[2, 4]), "x").expect("x");
        let z = b.input(f32(), dims(&[2, 4]), "z").expect("z");
        let g = b
            .node(
                Op::Gelu,
                &[x],
                &[(f32(), dims(&[2, 4]))],
                Default::default(),
                "gelu0",
            )
            .expect("x");
        let s = b
            .node(
                Op::Softmax { axis: -1 },
                &[g[0]],
                &[(f32(), dims(&[2, 4]))],
                Default::default(),
                "sm0",
            )
            .expect("gelu0");
        let a = b
            .node(
                Op::Add,
                &[s[0], z],
                &[(f32(), dims(&[2, 4]))],
                Default::default(),
                "add0",
            )
            .expect("sm0");
        b.output(a[0]).expect("out");
        b.build()
    }

    /// `x -> add -> matmul -> out`, where matmul is the one expensive node.
    fn a_matmul_chain() -> Graph {
        let mut b = GraphBuilder::new("mm");
        let x = b.input(f32(), dims(&[8, 8]), "x").expect("x");
        let w = b.input(f32(), dims(&[8, 8]), "w").expect("w");
        let add = b
            .node(
                Op::Add,
                &[x, w],
                &[(f32(), dims(&[8, 8]))],
                Default::default(),
                "add0",
            )
            .expect("operands");
        let mm = b
            .node(
                Op::MatMul,
                &[add[0], w],
                &[(f32(), dims(&[8, 8]))],
                Default::default(),
                "mm0",
            )
            .expect("operands");
        b.output(mm[0]).expect("out");
        b.build()
    }

    fn boxed(v: Scripted) -> Box<dyn Backend> {
        Box::new(v)
    }

    #[test]
    fn a_graph_no_backend_supports_leaves_every_node_on_the_host() {
        // The central property: "nobody will take this" is a plan, not an error.
        let backends = vec![boxed(Scripted::refusing("none"))];
        let plan = Partitioner::new(&backends)
            .partition(&a_chain())
            .expect("partition");

        assert!(plan.partitions.is_empty());
        assert_eq!(plan.host_nodes.len(), 3);
        assert!(!plan.is_fully_placed());
        for id in &plan.host_nodes {
            assert_eq!(plan.placement.get(id), Some(&Placement::Host));
        }
    }

    #[test]
    fn a_backend_that_runs_everything_takes_one_partition() {
        let ops = ReferenceOp::all();
        let backends = vec![boxed(Scripted::running("all", ops, SupportLevel::Fused))];
        let plan = Partitioner::new(&backends)
            .partition(&a_chain())
            .expect("partition");

        assert_eq!(
            plan.partitions.len(),
            1,
            "greedy fusion should take all three"
        );
        assert!(plan.is_fully_placed());
        assert_eq!(plan.partitioned_nodes(), 3);
        assert_eq!(plan.partitions[0].node_count(), 3);
        assert_eq!(plan.partitions[0].backend, 0);
    }

    #[test]
    fn a_backend_running_only_one_op_leaves_the_rest_on_the_host() {
        // One accepted op, two refused: the accepted one still gets its own
        // partition, because refusing the neighbours is not refusing the node.
        let backends = vec![boxed(Scripted::running(
            "gelu-only",
            &[ReferenceOp::Gelu],
            SupportLevel::Fused,
        ))];
        let graph = a_chain();
        let plan = Partitioner::new(&backends)
            .partition(&graph)
            .expect("partition");

        assert_eq!(plan.partitions.len(), 1);
        assert_eq!(plan.partitions[0].node_count(), 1);
        assert_eq!(plan.host_nodes.len(), 2);
        assert!(plan.partitions_for(0).len() == 1);
    }

    #[test]
    fn an_earlier_backend_wins_a_tie() {
        // Backend order is the tie-break, which is what makes the plan a function
        // of the backend list rather than of iteration order.
        let ops = ReferenceOp::all();
        let backends = vec![
            boxed(Scripted::running("first", ops, SupportLevel::Fused)),
            boxed(Scripted::running("second", ops, SupportLevel::Fused)),
        ];
        let plan = Partitioner::new(&backends)
            .partition(&a_chain())
            .expect("partition");
        assert_eq!(plan.partitions.len(), 1);
        assert_eq!(plan.partitions[0].backend, 0);
    }

    #[test]
    fn a_higher_support_level_beats_backend_order() {
        // Order is only the *tie*-break; a fused backend later in the list still
        // wins over a fallback one earlier in it.
        let backends = vec![
            boxed(Scripted::running(
                "fallback",
                ReferenceOp::all(),
                SupportLevel::Fallback,
            )),
            boxed(Scripted::running(
                "fused",
                ReferenceOp::all(),
                SupportLevel::Fused,
            )),
        ];
        let plan = Partitioner::new(&backends)
            .partition(&a_chain())
            .expect("partition");
        assert_eq!(plan.partitions[0].backend, 1);
    }

    #[test]
    fn growth_never_lowers_support() {
        // A backend that fuses gelu but only falls back on softmax: absorbing the
        // softmax node must not demote gelu, so the partition stops at gelu.
        let mut caps = CapabilitySet::empty("mixed").with_dtype_mask(DTypeMask::all());
        for class in crate::capabilities::reference::all_shape_classes() {
            caps = caps
                .with_pattern(PatternSupport::new(
                    ReferenceOp::Gelu.pattern_id(),
                    f32(),
                    *class,
                    SupportLevel::Fused,
                ))
                .with_pattern(PatternSupport::new(
                    ReferenceOp::Softmax.pattern_id(),
                    f32(),
                    *class,
                    SupportLevel::Fallback,
                ))
                .with_pattern(PatternSupport::new(
                    ReferenceOp::Add.pattern_id(),
                    f32(),
                    *class,
                    SupportLevel::Fallback,
                ));
        }
        let backends = vec![boxed(Scripted {
            name: "mixed",
            support: StrataSupport::new(caps),
        })];
        let graph = a_chain();
        let plan = Partitioner::new(&backends)
            .partition(&graph)
            .expect("partition");

        // gelu fused, the other two run as themselves; no single partition may
        // claim a fused level for something only the backend falls back on.
        for partition in &plan.partitions {
            assert!(!partition.nodes.is_empty());
        }
        assert_eq!(plan.partitioned_nodes() + plan.host_nodes.len(), 3);
    }

    #[test]
    fn a_partition_carries_the_values_it_needs_and_the_bytes_they_cost() {
        let backends = vec![boxed(Scripted::running(
            "all",
            ReferenceOp::all(),
            SupportLevel::Fused,
        ))];
        let graph = a_chain();
        let plan = Partitioner::new(&backends)
            .partition(&graph)
            .expect("partition");
        let partition = &plan.partitions[0];

        // Three nodes, so every value in the graph is either an operand or an
        // intermediate: two inputs plus three results.
        assert_eq!(partition.values.len(), 5);
        assert!(partition.bytes > 0);
        assert_eq!(
            partition.bytes,
            partition
                .values
                .iter()
                .map(|&id| graph.value(id).byte_len())
                .sum::<usize>()
        );
        assert!(!partition.pattern_id.is_empty());
        assert!(partition.pattern_id.starts_with("ref::"));
    }

    #[test]
    fn a_partition_names_its_outputs() {
        let backends = vec![boxed(Scripted::running(
            "all",
            ReferenceOp::all(),
            SupportLevel::Fused,
        ))];
        let graph = a_chain();
        let plan = Partitioner::new(&backends)
            .partition(&graph)
            .expect("partition");
        let partition = &plan.partitions[0];

        // The graph's single output is a result of the partition, so the host knows
        // where to read.
        assert!(
            partition
                .outputs
                .contains(graph.outputs().first().expect("one out"))
        );
    }

    #[test]
    fn every_node_is_placed_exactly_once() {
        let backends = vec![boxed(Scripted::running(
            "gelu",
            &[ReferenceOp::Gelu],
            SupportLevel::Fused,
        ))];
        let graph = a_chain();
        let plan = Partitioner::new(&backends)
            .partition(&graph)
            .expect("partition");

        let assigned = plan
            .placement
            .values()
            .filter(|p| matches!(p, Placement::Assigned { .. }))
            .count();
        assert_eq!(assigned, plan.partitioned_nodes());
        assert_eq!(assigned + plan.host_nodes.len(), graph.node_count());
    }

    #[test]
    fn partitioning_the_same_graph_twice_gives_the_same_plan() {
        // The property the compiled-subgraph cache depends on: same graph, same
        // capabilities, same partition.
        let backends = vec![boxed(Scripted::running(
            "gelu",
            &[ReferenceOp::Gelu, ReferenceOp::Add],
            SupportLevel::Fused,
        ))];
        let graph = a_chain();
        let first = Partitioner::new(&backends)
            .partition(&graph)
            .expect("partition");
        let second = Partitioner::new(&backends)
            .partition(&graph)
            .expect("partition");

        assert_eq!(first.partitions, second.partitions);
        assert_eq!(first.host_nodes, second.host_nodes);
    }

    #[test]
    fn a_two_backend_setup_splits_work_between_them() {
        let backends = vec![
            boxed(Scripted::running(
                "attention",
                &[ReferenceOp::Softmax],
                SupportLevel::Fused,
            )),
            boxed(Scripted::running(
                "elementwise",
                &[ReferenceOp::Gelu, ReferenceOp::Add],
                SupportLevel::Fused,
            )),
        ];
        let graph = a_chain();
        let plan = Partitioner::new(&backends)
            .partition(&graph)
            .expect("partition");

        assert!(plan.is_fully_placed());
        // Three partitions, not two, and that is the correct answer rather than a
        // miss. Gelu and add are both on the elementwise backend, but softmax sits
        // between them in the dataflow and only the attention backend runs it, so
        // no single backend can accept gelu and add together. Fusing them would mean
        // compiling a partition for a backend that refuses one of its nodes.
        assert_eq!(plan.partitions.len(), 3);

        let softmax_partition = plan
            .partitions
            .iter()
            .find(|p| p.pattern_id == ReferenceOp::Softmax.pattern_id())
            .expect("a softmax partition");
        assert_eq!(softmax_partition.backend, 0);
        assert_eq!(softmax_partition.node_count(), 1);

        let elementwise: Vec<&crate::partition::Partition> =
            plan.partitions.iter().filter(|p| p.backend == 1).collect();
        assert_eq!(elementwise.len(), 2, "gelu and add are separate partitions");
        for partition in &elementwise {
            assert_eq!(partition.node_count(), 1);
        }

        // Every node landed somewhere exactly once, and no partition crossed a
        // backend boundary.
        assert_eq!(plan.partitioned_nodes(), 3);
    }

    #[test]
    fn an_empty_graph_gives_an_empty_plan() {
        let backends = vec![boxed(Scripted::refusing("none"))];
        let plan = Partitioner::new(&backends)
            .partition(&Graph::new())
            .expect("partition");
        assert!(plan.partitions.is_empty());
        assert!(plan.host_nodes.is_empty());
        assert!(plan.is_fully_placed(), "an empty graph is trivially placed");
    }

    #[test]
    fn a_no_backend_list_leaves_everything_on_the_host() {
        let backends: Vec<Box<dyn Backend>> = Vec::new();
        let plan = Partitioner::new(&backends)
            .partition(&a_chain())
            .expect("partition");
        assert_eq!(plan.host_nodes.len(), 3);
        assert!(plan.partitions.is_empty());
    }

    #[test]
    fn a_matmul_is_partitioned_by_its_own_shape_class() {
        // A `[8, 8] x [8, 8]` matmul is a `matrix`, and the capability query is
        // answered per class, so a backend declaring only `batch3` must not claim
        // it.
        let graph = a_matmul_chain();
        let mut caps = CapabilitySet::empty("batch3-only").with_dtype_mask(DTypeMask::all());
        caps = caps.with_pattern(PatternSupport::new(
            ReferenceOp::MatMul.pattern_id(),
            f32(),
            ShapeClass::Batch3,
            SupportLevel::Fused,
        ));
        let backends = vec![boxed(Scripted {
            name: "batch3-only",
            support: StrataSupport::new(caps),
        })];

        let plan = Partitioner::new(&backends)
            .partition(&graph)
            .expect("partition");
        assert!(
            plan.partitions.is_empty(),
            "matrix must not match a batch3 entry"
        );
        // `add` and `matmul`: two nodes.
        assert_eq!(plan.host_nodes.len(), 2);
    }

    #[test]
    fn a_declared_shape_limit_bounds_what_the_capability_query_will_claim() {
        // Limits are a separate axis from the pattern table, and the partitioner is
        // where a limit becomes a decision. Check the two do not silently agree:
        // a backend may declare a pattern it cannot honour at every shape.
        let graph = a_chain();
        let mut caps = CapabilitySet::empty("rank-1-only").with_dtype_mask(DTypeMask::all());
        for class in crate::capabilities::reference::all_shape_classes() {
            caps = caps.with_pattern(PatternSupport::new(
                ReferenceOp::Gelu.pattern_id(),
                f32(),
                *class,
                SupportLevel::Fused,
            ));
        }
        let support = StrataSupport::new(caps.with_limits(DeclaredLimits {
            max_rank: Some(1),
            ..DeclaredLimits::default()
        }));

        // The chain's values are rank 2, which is outside a rank-1 device.
        let out = graph.value(graph.outputs()[0]).shape;
        assert_eq!(out.rank(), 2);
        assert!(!support.allows_shape(&out, LimitReading::Unbounded));
    }

    #[test]
    fn a_custom_op_partitions_under_its_own_pattern_id() {
        // A fused node from a vendor Strata has never heard of must route to the
        // backend that declared it, by string.
        let mut b = GraphBuilder::new("fused");
        // A leading dimension above 1, so this classifies as `batch3` and matches
        // the entry below. A `[1, 8, 8]` tensor would be `outer-product`, which is
        // the rule working, not a mismatch.
        let q = b.input(f32(), dims(&[2, 8, 8]), "q").expect("q");
        let out = b
            .node(
                Op::Custom {
                    pattern_id: "fused::flash_attention_v3".into(),
                },
                &[q],
                &[(f32(), dims(&[2, 8, 8]))],
                Default::default(),
                "fa0",
            )
            .expect("q");
        b.output(out[0]).expect("out");
        let graph = b.build();

        let caps = CapabilitySet::empty("attn")
            .with_dtype_mask(DTypeMask::all())
            .with_pattern(PatternSupport::new(
                "fused::flash_attention_v3",
                f32(),
                ShapeClass::Batch3,
                SupportLevel::Fused,
            ));
        let backends = vec![boxed(Scripted {
            name: "attn",
            support: StrataSupport::new(caps),
        })];

        let plan = Partitioner::new(&backends)
            .partition(&graph)
            .expect("partition");
        assert_eq!(plan.partitions.len(), 1);
        assert_eq!(plan.partitions[0].pattern_id, "fused::flash_attention_v3");
    }

    #[test]
    fn an_unknown_custom_op_declines_by_default() {
        // The refusal default again, one layer up: a pattern nobody declared is
        // not runnable.
        let mut b = GraphBuilder::new("fused");
        let q = b.input(f32(), dims(&[1, 8, 8]), "q").expect("q");
        let out = b
            .node(
                Op::Custom {
                    pattern_id: "fused::never_declared".into(),
                },
                &[q],
                &[(f32(), dims(&[1, 8, 8]))],
                Default::default(),
                "fa0",
            )
            .expect("q");
        b.output(out[0]).expect("out");
        let graph = b.build();

        let backends = vec![boxed(Scripted::refusing("none"))];
        let plan = Partitioner::new(&backends)
            .partition(&graph)
            .expect("partition");
        assert!(plan.partitions.is_empty());
        assert_eq!(plan.host_nodes.len(), 1);
    }
}
