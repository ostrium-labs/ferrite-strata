//! The session: the backends a run may use, and the plan for one graph.

use ferrite_strata::{
    Backend, Error, Graph, Partition, PartitionPlan, Partitioner, Placement, ValueId,
};

use crate::compile::{Step, compile_plan};

/// Which executor runs a unit.
///
/// A backend index, or [`Unit::Host`]. Not a `Box<dyn Backend>`: a plan has to be
/// serialisable and comparable, and an index into a session's backend list is
/// something a report can print.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Unit {
    /// A registered backend, by index.
    Backend(usize),
    /// The host's own CPU path.
    Host,
}

impl std::fmt::Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unit::Backend(index) => write!(f, "backend#{index}"),
            Unit::Host => f.write_str("host"),
        }
    }
}

/// The result of planning: what goes where, for one graph.
///
/// Inspectable before anything runs. The reason is that a plan is where a surprising
/// answer is cheapest to find — a graph that lands mostly on the host, or a set of
/// partitions far smaller than expected, is obvious here and invisible in a profile.
#[derive(Clone, Debug)]
pub struct Plan {
    pub(crate) inner: PartitionPlan,
    graph_name: String,
}

impl Plan {
    /// The partitions, in the order they were committed.
    #[must_use]
    pub fn partitions(&self) -> &[Partition] {
        &self.inner.partitions
    }

    /// Where every node ended up.
    #[must_use]
    pub fn placement(&self) -> &std::collections::HashMap<ferrite_strata::NodeId, Placement> {
        &self.inner.placement
    }

    /// The nodes left for the host.
    #[must_use]
    pub fn host_nodes(&self) -> &[ferrite_strata::NodeId] {
        &self.inner.host_nodes
    }

    /// Whether every node found a backend.
    #[must_use]
    pub fn is_fully_placed(&self) -> bool {
        self.inner.is_fully_placed()
    }

    /// How many nodes ended up on some backend, ignoring which.
    #[must_use]
    pub fn placed_node_count(&self) -> usize {
        self.inner.partitioned_nodes()
    }

    /// The graph's name.
    #[must_use]
    pub fn graph_name(&self) -> &str {
        &self.graph_name
    }
}

/// The backends a run may use, and the plans made with them.
///
/// Built by registering backends, then [`Session::plan`] per graph and
/// [`Session::compile`] per plan. Registration order is the tie-break the partitioner
/// uses, so it is a real decision: registering the CPU first makes it the fallback of
/// last resort rather than the first choice.
#[derive(Default)]
pub struct Session {
    backends: Vec<Box<dyn Backend>>,
    /// Cached plan for the most recently planned graph.
    ///
    /// Not a general cache — one slot, deliberately. The multi-entry compiled-subgraph
    /// cache is keyed on [`StableName`](ferrite_strata::StableName) and belongs with
    /// the runtime that owns device handles; keeping a single slot here is enough to
    /// make `plan` then `compile` cheap without inventing an eviction policy.
    last_plan: Option<(String, Plan)>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field(
                "backends",
                &self.backends.iter().map(|b| b.name()).collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl Session {
    /// A session with no backends.
    ///
    /// Not useless: with no backends every node lands on the host, and that is the
    /// configuration a framework starts in before a plugin has loaded.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a backend, returning its index.
    ///
    /// # Errors
    ///
    /// If `backend` is not `Send + Sync`, which the trait already guarantees, so this
    /// exists to give a single call site rather than to reject anything. Kept because
    /// every other fallible registration in this ecosystem should have been.
    pub fn add(&mut self, backend: Box<dyn Backend>) -> Result<usize, Error> {
        self.backends.push(backend);
        Ok(self.backends.len() - 1)
    }

    /// How many backends are registered.
    #[must_use]
    pub fn backend_count(&self) -> usize {
        self.backends.len()
    }

    /// The backends' names, in registration order.
    #[must_use]
    pub fn backend_names(&self) -> Vec<&str> {
        self.backends.iter().map(|b| b.name()).collect()
    }

    /// One backend by index.
    ///
    /// # Panics
    ///
    /// If `index` is out of range. Every caller gets an index from
    /// [`Session::add`] or from a plan this session produced, so a miss means two
    /// different sessions have been confused — a bug, not a runtime condition.
    #[must_use]
    pub fn backend(&self, index: usize) -> &dyn Backend {
        &*self.backends[index]
    }

    /// Plan `graph` across the registered backends.
    ///
    /// Never fails for want of support: a graph no backend will take comes back with
    /// its nodes on the host. [`Session::plan`] returning `Err` means the backend list
    /// or the graph is malformed, neither of which is a support question.
    ///
    /// # Errors
    ///
    /// Currently never. The signature is `Result` because a later phase validates
    /// declared node limits, and adding an error type later would change every
    /// caller's types for no benefit now.
    pub fn plan(&mut self, graph: &Graph) -> Result<Plan, Error> {
        let partitioner = Partitioner::new(&self.backends);
        let inner = partitioner.partition(graph)?;
        let plan = Plan {
            inner,
            graph_name: graph.name().to_string(),
        };
        self.last_plan = Some((plan.graph_name.clone(), plan.clone_for_cache()));
        Ok(plan)
    }

    /// Compile the plan made for `graph`'s name into something runnable.
    ///
    /// Takes the graph rather than a plan so the compiled artefact cannot outlive the
    /// graph it was extracted from — a plan is cheap to rebuild, a stale one is not
    /// safe to run.
    ///
    /// # Errors
    ///
    /// If a backend reports a hard error, or a partition cannot be extracted.
    pub fn compile(&mut self, graph: &Graph) -> Result<Step, Error> {
        let plan = match &self.last_plan {
            Some((name, plan)) if *name == graph.name() => plan.clone_for_cache(),
            _ => self.plan(graph)?,
        };
        compile_plan(self, &plan, graph)
    }

    /// Compile a specific plan.
    ///
    /// # Errors
    ///
    /// See [`Session::compile`].
    pub fn compile_plan(&self, graph: &Graph, plan: &Plan) -> Result<Step, Error> {
        compile_plan(self, plan, graph)
    }
}

impl Plan {
    fn clone_for_cache(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            graph_name: self.graph_name.clone(),
        }
    }
}

/// Build an input map from a list of `(value, data)` pairs.
///
/// A convenience so the common case is not a `.collect()` in every example and test.
/// Infallible by construction, so it returns the map directly rather than a `Result`
/// that every call site would have to unwrap.
#[must_use]
pub fn inputs(entries: &[(ValueId, Vec<f32>)]) -> std::collections::HashMap<ValueId, Vec<f32>> {
    entries
        .iter()
        .map(|(id, data)| (*id, data.clone()))
        .collect()
}
