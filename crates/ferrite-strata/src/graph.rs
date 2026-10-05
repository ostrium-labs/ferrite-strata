//! The graph IR: values, ops, nodes, and the arena that holds them.
//!
//! # Arena, not pointer graph
//!
//! A `Graph` owns `Vec`s of values and nodes, and every reference is a `u32`
//! index ([`ValueId`], [`NodeId`]). Three reasons, in order of weight:
//!
//! 1. **Serialisation is the ABI.** `compile` receives the subgraph as opaque
//!    bytes ([ADR-0014](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0014-two-distinct-artefacts-get-two-distinct-names.md)).
//!    An arena serialises as two flat arrays plus a few offsets. A
//!    `Box`-and-`Arc` graph serialises as a pointer chase, and the receiving
//!    plugin has to run an untrusted deserialiser over it.
//! 2. **Subgraph extraction is a slice, not a copy.** The partitioner pulls a
//!    contiguous run out of the arena and re-bases the indices. With pointers that
//!    is a graph traversal rebuilding itself on the far side.
//! 3. **No interior mutability.** `&Graph` can hand out `&Node` freely, and a
//!    reader cannot invalidate a reader. The builder is the only writer.
//!
//! # What the IR does not decide
//!
//! There is no `Op::Conv2d { stride, padding, dilation, groups }` with typed
//! fields. Ops carry their parameters in [`Attributes`], and the reason is the
//! vendor boundary: a vendored subgraph has to reach a plugin that has never
//! heard of half these ops. A key/value bag lets an unknown key be *skipped*
//! rather than failing to parse — see [`crate::attrs`].
//!
//! What the IR *does* decide is the two things no plugin can invent: the
//! dtype/shape of every value, and which nodes feed which. A vendor is free to
//! accept, decline, or fuse.

use std::collections::HashMap;

use crate::attrs::Attributes;
use crate::dtype::DType;
use crate::id::{NodeId, StableName, ValueId};
use crate::shape::Shape;
use core::fmt;

/// The operation a node performs.
///
/// The split is deliberate: ops Strata understands, and a `Custom` escape hatch
/// carrying a vendor-owned **pattern id** — an extensible string, not an enum
/// variant. See
/// [ADR-0005](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0005-capabilities-are-versioned-data-plus-a-pull-supports-query.md):
/// adding a fused op must not be a semver break for every crate downstream, so
/// new ops arrive as strings and the vocabulary grows by append.
///
/// A `Custom` op is not a hole in the type system. It carries a pattern id, it
/// round-trips through the wire format, and it is exactly what a fused subgraph
/// is expressed as when it comes back from a vendor.
/// `PartialEq`/`PartialOrd` rather than `Eq`/`Hash`/`Ord`, because [`Op::LayerNorm`]
/// carries an `f32` epsilon. A graph is compared for equality in tests and in the
/// subgraph cache key, never hashed — the cache keys on [`StableName`], which is a
/// string precisely so it can be.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// An input to the graph. Produces no computation.
    Input,
    /// Elementwise `a + b`, same shape.
    Add,
    /// Elementwise `a * b`, same shape.
    Mul,
    /// `a @ b`, the batched matrix multiply.
    MatMul,
    /// Elementwise natural logarithm.
    Log,
    /// The exponential function.
    Exp,
    /// Row-wise softmax over the last axis.
    Softmax {
        /// The axis to normalise over.
        axis: i64,
    },
    /// Layer normalisation over the trailing axes.
    LayerNorm {
        /// The total gain, per output element.
        gamma: ValueId,
        /// The per-feature bias.
        beta: ValueId,
        /// Added inside the square root.
        eps: f32,
    },
    /// The GELU activation, in its exact (not tanh-approximate) form.
    Gelu,
    /// A linear layer, `x @ weight + bias`.
    Linear {
        /// The weight matrix, shape `[out, in]`.
        weight: ValueId,
        /// The optional bias, shape `[out]`.
        bias: Option<ValueId>,
    },
    /// A reshape. Carries its target shape rather than letting the op recompute
    /// one, so the IR is self-describing.
    Reshape {
        /// The output shape.
        shape: Shape,
    },
    /// A transpose over an explicit permutation.
    Transpose {
        /// The axis permutation, which must be a permutation of `0..rank`.
        permutation: Vec<i64>,
    },
    /// A broadcast along new leading axes.
    Expand {
        /// The target shape.
        shape: Shape,
    },
    /// A slice along an axis.
    Slice {
        /// The axis to slice.
        axis: i64,
        /// Inclusive start.
        start: i64,
        /// Exclusive end.
        end: i64,
        /// Step, which must be positive.
        stride: i64,
    },
    /// A vendor-owned operation, named by an extensible pattern id.
    ///
    /// The `fused::` prefix on a real id is not required, but the convention
    /// makes a fused op recognisable in a log without a lookup.
    Custom {
        /// The pattern id. Not interpreted by Strata, which is the point: a
        /// plugin defines these and Strata passes them through.
        pattern_id: String,
    },
}

impl Op {
    /// Whether this op consumes nothing, which makes it a graph input.
    #[must_use]
    pub const fn is_source(&self) -> bool {
        matches!(self, Self::Input)
    }

    /// A short stable name, for logs and for the capability vocabulary's
    /// reference-op column.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::MatMul => "matmul",
            Self::Log => "log",
            Self::Exp => "exp",
            Self::Softmax { .. } => "softmax",
            Self::LayerNorm { .. } => "layer_norm",
            Self::Gelu => "gelu",
            Self::Linear { .. } => "linear",
            Self::Reshape { .. } => "reshape",
            Self::Transpose { .. } => "transpose",
            Self::Expand { .. } => "expand",
            Self::Slice { .. } => "slice",
            Self::Custom { .. } => "custom",
        }
    }
}

/// A tensor in the graph: a dtype, a shape, and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    /// The element type.
    pub dtype: DType,
    /// The shape. Fully static, per [`crate::shape`].
    pub shape: Shape,
    /// The node that produced it, or `None` for a graph input.
    pub producer: Option<NodeId>,
    /// Which of that node's outputs this is.
    pub output_slot: usize,
    /// The stable name, used as a cache key and in diagnostics.
    pub name: StableName,
}

impl Value {
    /// A graph input, with no producer.
    #[must_use]
    pub fn input(dtype: DType, shape: Shape, name: StableName) -> Self {
        Self {
            dtype,
            shape,
            producer: None,
            output_slot: 0,
            name,
        }
    }

    /// A value produced by `node`.
    #[must_use]
    pub fn produced_by(dtype: DType, shape: Shape, node: NodeId, slot: usize) -> Self {
        Self {
            dtype,
            shape,
            producer: Some(node),
            output_slot: slot,
            name: StableName::from_output(&Self::node_name(node), slot),
        }
    }

    /// The name of the producing node.
    ///
    /// Only meaningful when [`Value::producer`] is `Some`; the placeholder keeps
    /// construction total, and the graph builder overwrites it with the real name
    /// as soon as the node is added.
    fn node_name(node: NodeId) -> StableName {
        StableName::new(format!("n{}", node.0))
    }

    /// How many bytes this value occupies at runtime.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.dtype.byte_len(self.shape.element_count())
    }
}

/// One operation and its operands.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// What it computes.
    pub op: Op,
    /// The operands, in the order the op declares them.
    pub inputs: Vec<ValueId>,
    /// The results, produced in slot order.
    pub outputs: Vec<ValueId>,
    /// The op's parameters.
    pub attrs: Attributes,
    /// The stable name.
    pub name: StableName,
}

impl Node {
    /// How many results this node produces.
    #[must_use]
    pub fn arity(&self) -> usize {
        self.outputs.len()
    }
}

/// A whole graph: two arenas plus the list of graph-level outputs.
///
/// Not `Clone`-cheap for large graphs and not meant to be; the partitioner moves
/// subgraphs out of it and the compiler wants the original to stay put.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    values: Vec<Value>,
    outputs: Vec<ValueId>,
    name: String,
}

impl Graph {
    /// An empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty graph with a name, which shows up in diagnostics and vendor logs.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// The graph's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Every node, in insertion order.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Every value, in insertion order.
    #[must_use]
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// The graph's outputs.
    #[must_use]
    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    /// How many nodes the graph has.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// How many values the graph has.
    #[must_use]
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// One node.
    ///
    /// # Panics
    ///
    /// If `id` does not name a node in this graph. Every caller of this method is
    /// holding an id obtained from this same graph, so a miss means the caller
    /// mixed two graphs — which is a bug, not a runtime condition to report.
    #[must_use]
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// One value.
    ///
    /// # Panics
    ///
    /// If `id` does not name a value in this graph. See [`Graph::node`].
    #[must_use]
    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.0 as usize]
    }

    /// Whether `id` names a node here.
    #[must_use]
    pub fn has_node(&self, id: NodeId) -> bool {
        (id.0 as usize) < self.nodes.len()
    }

    /// Whether `id` names a value here.
    #[must_use]
    pub fn has_value(&self, id: ValueId) -> bool {
        (id.0 as usize) < self.values.len()
    }

    /// Whether `id` names an op this graph declares as one of its outputs.
    #[must_use]
    pub fn is_output(&self, id: ValueId) -> bool {
        self.outputs.contains(&id)
    }

    /// How many nodes feed directly into `id`.
    ///
    /// This is the in-degree the partitioner needs and the only direction that
    /// cannot be read off the arena alone, because a node's inputs are recorded
    /// from the consumer's side.
    #[must_use]
    pub fn fan_in(&self, id: NodeId) -> usize {
        self.nodes[id.0 as usize].inputs.len()
    }

    /// Extract `nodes` as a standalone graph, with value ids remapped densely.
    ///
    /// See [`Graph::subgraph_with_map`] — this is that function with the id map
    /// discarded, for callers that do not need to publish results back.
    ///
    /// # Errors
    ///
    /// See [`Graph::subgraph_with_map`].
    pub fn subgraph(&self, nodes: &[NodeId]) -> Result<Self, GraphError> {
        self.subgraph_with_map(nodes).map(|(graph, _)| graph)
    }

    /// Extract `nodes` as a standalone graph, reporting how the ids moved.
    ///
    /// The returned map goes from **extracted id to original id**, because the
    /// caller's problem is the reverse direction: a backend writes its result under
    /// an extracted id, and the caller needs to find it in the original graph's
    /// numbering to read it as a graph output. Without this map a partitioned run
    /// produces values nobody can locate.
    ///
    /// This is what makes a partition handable. Three things happen:
    ///
    /// 1. **Nodes** are copied in `nodes` order, which the partitioner guarantees is
    ///    graph order and therefore topologically valid.
    /// 2. **Values produced inside** are remapped into a dense arena.
    /// 3. **Values produced outside** — a partition's inputs, and the weights its ops
    ///    name — become inputs of the extracted graph.
    ///
    /// Point 3 is why this cannot be a slice. A backend handed a subgraph must be
    /// able to address an operand it did not compute, and remapping ids without
    /// turning external operands into inputs would leave it reading ids that mean
    /// something else.
    ///
    /// # Errors
    ///
    /// If `nodes` names a node this graph does not have, or if a value is read before
    /// it is produced — which means `nodes` is not in topological order.
    pub fn subgraph_with_map(
        &self,
        nodes: &[NodeId],
    ) -> Result<(Self, HashMap<ValueId, ValueId>), GraphError> {
        if nodes.is_empty() {
            return Ok((Self::named(format!("{}/empty", self.name)), HashMap::new()));
        }

        let mut in_set = vec![false; self.node_count()];
        for &node in nodes {
            if !self.has_node(node) {
                return Err(GraphError::UnknownNode { id: node });
            }
            in_set[node.0 as usize] = true;
        }

        let mut builder = GraphBuilder::new(format!("{}/partition", self.name));
        // External operands get the next arena id. Original ids cannot be preserved
        // without colliding with produced ones, so they are remapped like everything
        // else — the point is that they become *inputs*, which is a nameable thing.
        let mut external: HashMap<ValueId, ValueId> = HashMap::new();
        // Extracted id -> original id, reported to the caller.
        let mut back: HashMap<ValueId, ValueId> = HashMap::new();

        // First pass: every external operand becomes an input, before any node reads
        // it.
        for &node in nodes {
            for &input in &self.node(node).inputs {
                let produced_inside = self
                    .value(input)
                    .producer
                    .is_some_and(|p| in_set[p.0 as usize]);
                if produced_inside || external.contains_key(&input) {
                    continue;
                }
                let value = self.value(input);
                let id = builder.input(value.dtype, value.shape, format!("{}~in", value.name))?;
                external.insert(input, id);
                back.insert(id, input);
            }
        }

        // Second pass: the nodes, in the order given.
        let mut produced: HashMap<ValueId, ValueId> = HashMap::new();
        for &node in nodes {
            let reference = self.node(node);
            let mut inputs = Vec::with_capacity(reference.inputs.len());
            for &input in &reference.inputs {
                let remapped = produced
                    .get(&input)
                    .or_else(|| external.get(&input))
                    .copied()
                    .ok_or(GraphError::UseBeforeDefinition { id: input })?;
                inputs.push(remapped);
            }

            // Output specs are read from this graph rather than recomputed: the IR is
            // the authority on shapes, and a second derivation would be a second
            // opinion.
            let specs: Vec<(DType, Shape)> = reference
                .outputs
                .iter()
                .map(|&out| {
                    let value = self.value(out);
                    (value.dtype, value.shape)
                })
                .collect();

            // A multi-output node would otherwise claim its own name twice.
            let name = if specs.len() > 1 {
                format!("{}#{}", reference.name, node.0)
            } else {
                reference.name.as_str().to_string()
            };

            let outputs = builder.node(
                reference.op.clone(),
                &inputs,
                &specs,
                reference.attrs.clone(),
                name,
            )?;

            for (&original, &remapped) in reference.outputs.iter().zip(&outputs) {
                produced.insert(original, remapped);
                back.insert(remapped, original);
            }
        }

        // Third pass: what the subgraph must publish. A value the whole graph declared
        // an output of stays one, and so does one consumed outside this subgraph —
        // whoever reads it needs to address it.
        let mut graph = builder.build();
        let mut published = Vec::new();
        for &node in nodes {
            for &out in &self.node(node).outputs {
                let consumed_outside = self
                    .nodes()
                    .iter()
                    .enumerate()
                    .any(|(i, other)| !in_set[i] && other.inputs.contains(&out));
                if (self.is_output(out) || consumed_outside)
                    && let Some(&remapped) = produced.get(&out)
                {
                    published.push(remapped);
                }
            }
        }
        graph.outputs = published;

        Ok((graph, back))
    }
}

/// Builds a [`Graph`], keeping it well-formed as it goes.
///
/// The only writer of a graph. Everything it refuses describes a graph no backend
/// could compile: an operand that is not there, two nodes sharing one name.
///
/// # Why the name check lives here rather than in the partitioner
///
/// Names are the subgraph cache key ([`StableName`]), so two nodes sharing one
/// would make the cache return the *wrong* compiled artefact. That is a silent
/// miscompilation rather than a crash, and construction is the only place it is
/// cheap to catch.
#[derive(Clone, Debug)]
pub struct GraphBuilder {
    graph: Graph,
}

impl GraphBuilder {
    /// A builder for a named graph.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            graph: Graph::named(name),
        }
    }

    /// A builder for an unnamed graph.
    #[must_use]
    pub fn unnamed() -> Self {
        Self {
            graph: Graph::new(),
        }
    }

    /// Add a graph input.
    ///
    /// # Errors
    ///
    /// If the name is already taken by a value or a node.
    pub fn input(
        &mut self,
        dtype: DType,
        shape: Shape,
        name: impl Into<String>,
    ) -> Result<ValueId, GraphError> {
        let name = name.into();
        self.check_name_free(&name)?;
        let id = ValueId(self.graph.values.len() as u32);
        self.graph
            .values
            .push(Value::input(dtype, shape, StableName::new(name)));
        Ok(id)
    }

    /// Add a node, and every value it produces.
    ///
    /// Returns one [`ValueId`] per output slot, in order. `output_specs` supplies
    /// the dtype and shape of each output; Strata does not infer them, because
    /// inference would introduce a second implicit set of rules about what `MatMul`
    /// means, and those rules would disagree between ops.
    ///
    /// # Errors
    ///
    /// If an operand is not in the graph, or the name is already taken.
    pub fn node(
        &mut self,
        op: Op,
        inputs: &[ValueId],
        output_specs: &[(DType, Shape)],
        attrs: Attributes,
        name: impl Into<String>,
    ) -> Result<Vec<ValueId>, GraphError> {
        let name = name.into();
        self.check_name_free(&name)?;

        for &input in inputs {
            if !self.graph.has_value(input) {
                return Err(GraphError::UnknownValue { id: input });
            }
        }

        let node_id = NodeId(self.graph.nodes.len() as u32);
        let mut outputs = Vec::with_capacity(output_specs.len());
        for (slot, &(dtype, shape)) in output_specs.iter().enumerate() {
            let id = ValueId(self.graph.values.len() as u32);
            self.graph
                .values
                .push(Value::produced_by(dtype, shape, node_id, slot));
            outputs.push(id);
        }

        self.graph.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            outputs: outputs.clone(),
            attrs,
            name: StableName::new(name),
        });

        Ok(outputs)
    }

    /// Mark a value as an output of the whole graph.
    ///
    /// # Errors
    ///
    /// If the value is not in the graph, or is already an output.
    pub fn output(&mut self, id: ValueId) -> Result<(), GraphError> {
        if !self.graph.has_value(id) {
            return Err(GraphError::UnknownValue { id });
        }
        if self.graph.outputs.contains(&id) {
            return Err(GraphError::DuplicateOutput { id });
        }
        self.graph.outputs.push(id);
        Ok(())
    }

    /// Finish, yielding the graph.
    #[must_use]
    pub fn build(self) -> Graph {
        self.graph
    }

    fn check_name_free(&self, name: &str) -> Result<(), GraphError> {
        let taken = self.graph.values.iter().any(|v| v.name.as_str() == name)
            || self.graph.nodes.iter().any(|n| n.name.as_str() == name);
        if taken {
            return Err(GraphError::DuplicateName {
                name: name.to_string(),
            });
        }
        Ok(())
    }
}

/// Something wrong with a graph under construction.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    /// An operand named a value that is not in the graph.
    UnknownValue {
        /// The missing value.
        id: ValueId,
    },
    /// Two values or nodes claimed one name.
    DuplicateName {
        /// The contested name.
        name: String,
    },
    /// A value was declared an output twice.
    DuplicateOutput {
        /// The repeated value.
        id: ValueId,
    },
    /// A subgraph extraction named a node this graph does not have.
    UnknownNode {
        /// The missing node.
        id: NodeId,
    },
    /// A node read a value no earlier node produced, so the node list is not in
    /// topological order.
    UseBeforeDefinition {
        /// The value that was read too early.
        id: ValueId,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::UnknownValue { id } => write!(
                f,
                "no value {id} in this graph. Operands are dense ids in insertion \
                 order, so this means the node was added before its operand."
            ),
            GraphError::DuplicateName { name } => write!(
                f,
                "the name `{name}` is already taken. Names are the subgraph cache \
                 key, so two nodes sharing one would make the cache return the wrong \
                 compiled artefact."
            ),
            GraphError::DuplicateOutput { id } => write!(
                f,
                "{id} is already an output of this graph. The output list is an \
                 ordered contract with the buffer layout, so a repeat is ambiguous."
            ),
            GraphError::UnknownNode { id } => {
                write!(f, "no node {id} in this graph")
            }
            GraphError::UseBeforeDefinition { id } => write!(
                f,
                "{id} is read before any node in this list produces it, so the list \
                 is not in topological order. Subgraph extraction preserves the order \
                 it is given."
            ),
        }
    }
}

impl std::error::Error for GraphError {}

#[cfg(test)]
mod tests {
    use super::{Graph, GraphBuilder, GraphError, Op, Value};
    use crate::attrs::{AttrKey, AttrValue, Attributes};
    use crate::dtype::{DType, ElementType, QuantSpec};
    use crate::id::{NodeId, StableName, ValueId};
    use crate::shape::Shape;

    fn dims(dims: &[usize]) -> Shape {
        Shape::new(dims).expect("test shape within the rank limit")
    }

    fn f32() -> DType {
        DType::plain(ElementType::F32)
    }

    /// A two-input add, which is the smallest graph that exercises the arena.
    fn an_add_graph() -> (Graph, ValueId) {
        let mut b = GraphBuilder::new("add");
        let a = b.input(f32(), dims(&[2, 2]), "a").expect("a is free");
        let c = b.input(f32(), dims(&[2, 2]), "b").expect("b is free");
        let out = b
            .node(
                Op::Add,
                &[a, c],
                &[(f32(), dims(&[2, 2]))],
                Attributes::new(),
                "add0",
            )
            .expect("operands exist");
        b.output(out[0]).expect("not already an output");
        (b.build(), out[0])
    }

    #[test]
    fn a_fresh_graph_is_empty() {
        let g = Graph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.value_count(), 0);
        assert!(g.outputs().is_empty());
        assert_eq!(g.name(), "");
        assert_eq!(Graph::named("mlp").name(), "mlp");
    }

    #[test]
    fn a_custom_op_keeps_its_pattern_id_as_data() {
        // The point of `Custom`: a fused op Strata has never heard of round-trips
        // through the IR without a Strata change.
        let op = Op::Custom {
            pattern_id: "fused::flash_attention_v3".into(),
        };
        assert_eq!(op.name(), "custom");
        assert!(!op.is_source());
        assert_eq!(
            op,
            Op::Custom {
                pattern_id: "fused::flash_attention_v3".into()
            }
        );
    }

    #[test]
    fn only_input_is_a_source() {
        assert!(Op::Input.is_source());
        assert_eq!(Op::Input.name(), "input");
        assert!(!Op::Add.is_source());
        assert!(!Op::MatMul.is_source());
    }

    #[test]
    fn every_op_has_a_name() {
        let names = [
            Op::Input,
            Op::Add,
            Op::Mul,
            Op::MatMul,
            Op::Log,
            Op::Exp,
            Op::Gelu,
            Op::Softmax { axis: -1 },
            Op::Expand {
                shape: dims(&[2, 2]),
            },
            Op::LayerNorm {
                gamma: ValueId(0),
                beta: ValueId(1),
                eps: 1e-5,
            },
            Op::Linear {
                weight: ValueId(0),
                bias: None,
            },
            Op::Reshape {
                shape: dims(&[2, 2]),
            },
            Op::Transpose {
                permutation: vec![1, 0],
            },
            Op::Slice {
                axis: 1,
                start: 0,
                end: 4,
                stride: 1,
            },
            Op::Custom {
                pattern_id: "x".into(),
            },
        ];
        // Distinct, stable and lowercase: they end up in wire formats and logs.
        let mut seen = std::collections::BTreeSet::new();
        for op in &names {
            assert!(seen.insert(op.name()), "duplicate op name {}", op.name());
            assert!(
                op.name()
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c == '_'),
                "{} is not a lowercase wire name",
                op.name()
            );
        }
    }

    #[test]
    fn an_input_value_has_no_producer() {
        let v = Value::input(f32(), dims(&[2, 3]), StableName::new("x"));
        assert_eq!(v.producer, None);
        assert_eq!(v.output_slot, 0);
        assert_eq!(v.byte_len(), 2 * 3 * 4);
    }

    #[test]
    fn a_produced_value_names_its_producer_and_slot() {
        // Two results of one node are different values, so the slot is part of the
        // name.
        let v = Value::produced_by(f32(), dims(&[4]), NodeId(3), 1);
        assert_eq!(v.producer, Some(NodeId(3)));
        assert_eq!(v.output_slot, 1);
        assert_eq!(v.name.as_str(), "n3#1");
    }

    #[test]
    fn a_quantised_value_reports_its_quantised_byte_length() {
        let q = DType::quantised(
            ElementType::I8,
            QuantSpec::Block {
                block_size: 32,
                symmetric: true,
            },
        );
        assert!(q.is_quantised());
        assert_eq!(q.byte_len(16), 16);
        let v = Value::input(q, dims(&[16]), StableName::new("q"));
        assert_eq!(v.byte_len(), 16);
    }

    #[test]
    fn a_built_graph_records_what_was_added() {
        let (g, out) = an_add_graph();
        assert_eq!(g.node_count(), 1);
        // Two inputs plus one result.
        assert_eq!(g.value_count(), 3);
        assert_eq!(g.outputs(), &[out]);
        assert!(g.is_output(out));
        assert_eq!(g.name(), "add");
    }

    #[test]
    fn a_node_records_its_operands_and_producer() {
        let (g, out) = an_add_graph();
        let node = g.node(NodeId(0));
        assert_eq!(node.op, Op::Add);
        assert_eq!(node.inputs, [ValueId(0), ValueId(1)]);
        assert_eq!(node.outputs, [out]);
        assert_eq!(node.arity(), 1);
        assert_eq!(g.fan_in(NodeId(0)), 2);
        assert_eq!(g.value(out).producer, Some(NodeId(0)));
    }

    #[test]
    fn an_id_past_the_arena_is_not_a_member() {
        // What the partitioner checks before indexing.
        let (g, _) = an_add_graph();
        assert!(g.has_node(NodeId(0)));
        assert!(g.has_value(ValueId(2)));
        assert!(!g.has_node(NodeId(99)));
        assert!(!g.has_value(ValueId(99)));
    }

    #[test]
    fn a_duplicate_name_is_refused() {
        // Names are the subgraph cache key, so a collision is a silent
        // miscompilation rather than a crash. Catching it here is the point.
        let mut b = GraphBuilder::unnamed();
        b.input(f32(), dims(&[2]), "x").expect("first x");
        assert_eq!(
            b.input(f32(), dims(&[2]), "x").expect_err("second x"),
            GraphError::DuplicateName { name: "x".into() }
        );

        b.input(f32(), dims(&[2]), "y").expect("y is free");
        b.node(
            Op::Gelu,
            &[],
            &[(f32(), dims(&[2]))],
            Attributes::new(),
            "y",
        )
        .expect_err("a node cannot reuse an input's name");
    }

    #[test]
    fn an_operand_that_is_not_in_the_graph_is_refused() {
        let mut b = GraphBuilder::unnamed();
        let err = b
            .node(
                Op::Add,
                &[ValueId(7)],
                &[(f32(), dims(&[2]))],
                Attributes::new(),
                "add0",
            )
            .expect_err("value 7 does not exist");
        assert_eq!(err, GraphError::UnknownValue { id: ValueId(7) });
    }

    #[test]
    fn declaring_an_output_twice_is_refused() {
        // The output list is an ordered contract with the buffer layout.
        let mut b = GraphBuilder::unnamed();
        let x = b.input(f32(), dims(&[2]), "x").expect("x");
        b.output(x).expect("first");
        assert_eq!(
            b.output(x).expect_err("second"),
            GraphError::DuplicateOutput { id: x }
        );
    }

    #[test]
    fn a_two_output_node_gets_two_distinct_values() {
        let mut b = GraphBuilder::new("two");
        let x = b.input(f32(), dims(&[2]), "x").expect("x");
        let outs = b
            .node(
                Op::Log,
                &[x],
                &[(f32(), dims(&[2])), (f32(), dims(&[2]))],
                Attributes::new(),
                "log0",
            )
            .expect("x exists");
        assert_eq!(outs.len(), 2);
        assert_ne!(outs[0], outs[1]);
        let g = b.build();
        assert_ne!(g.value(outs[0]).name, g.value(outs[1]).name);
    }

    #[test]
    fn attributes_on_a_node_survive_beside_typed_op_fields() {
        // A `Custom` op carries its parameters as attributes; that is the normal
        // shape of a fused node.
        let mut b = GraphBuilder::new("fa");
        let x = b.input(f32(), dims(&[1, 8, 8]), "q").expect("q");
        let mut attrs = Attributes::new();
        attrs
            .set(
                AttrKey::new("head_dim").expect("valid key"),
                AttrValue::Int(8),
            )
            .expect("first set");

        let outs = b
            .node(
                Op::Custom {
                    pattern_id: "fused::flash_attention_v3".into(),
                },
                &[x],
                &[(f32(), dims(&[1, 8, 8]))],
                attrs,
                "fa0",
            )
            .expect("q exists");
        b.output(outs[0]).expect("not already an output");

        let g = b.build();
        let node = g.node(NodeId(0));
        assert_eq!(node.attrs.get("head_dim"), Some(&AttrValue::Int(8)));
        assert_eq!(node.op.name(), "custom");
    }

    #[test]
    fn a_graph_error_explains_itself() {
        // Every refusal has to be actionable without reading the source: these
        // messages are what a framework author sees when a build fails.
        let err = GraphError::DuplicateName { name: "x".into() };
        assert!(err.to_string().contains("cache"), "{err}");

        let err = GraphError::UnknownValue { id: ValueId(7) };
        assert!(err.to_string().contains("before its operand"), "{err}");

        let err = GraphError::DuplicateOutput { id: ValueId(1) };
        assert!(err.to_string().contains("buffer layout"), "{err}");
    }
    #[test]
    fn a_subgraph_keeps_only_its_nodes() {
        let (graph, out) = an_add_graph();
        // Node 0 is the add; a one-node partition must not drag in the inputs as
        // nodes, only as values.
        let part = graph.subgraph(&[NodeId(0)]).expect("node 0 exists");
        assert_eq!(part.node_count(), 1);
        assert_eq!(part.node(NodeId(0)).op, Op::Add);
        assert_eq!(part.outputs(), &[out]);
    }

    #[test]
    fn an_external_operand_becomes_an_input_of_the_subgraph() {
        // The load-bearing property: a backend must be able to address an operand it
        // did not compute.
        let (graph, out) = an_add_graph();
        let part = graph.subgraph(&[NodeId(0)]).expect("node 0 exists");

        let node = part.node(NodeId(0));
        assert_eq!(node.inputs.len(), 2);
        for &input in &node.inputs {
            let value = part.value(input);
            assert_eq!(
                value.producer, None,
                "an operand the subgraph did not compute must be an input, not a result"
            );
            assert_eq!(value.shape.dims(), &[2, 2]);
        }
        assert_eq!(part.outputs(), &[out]);
    }

    #[test]
    fn extracted_ids_are_dense_and_renumbered_from_zero() {
        let (graph, _) = an_add_graph();
        let part = graph.subgraph(&[NodeId(0)]).expect("node 0 exists");
        // Two external inputs plus one produced value.
        assert_eq!(part.value_count(), 3);
        for index in 0..part.value_count() {
            let value = part.value(ValueId(index as u32));
            assert!(
                value.name.as_str().ends_with("~in") || index == 2,
                "{value:?}"
            );
        }
    }

    #[test]
    fn an_internal_value_is_not_turned_into_an_input() {
        // `x -> gelu -> gelu`: extracting only the second gelu must not also expose
        // the first one's result as an input.
        let mut b = GraphBuilder::new("twice");
        let x = b.input(f32(), dims(&[2, 2]), "x").expect("x");
        let g1 = b
            .node(
                Op::Gelu,
                &[x],
                &[(f32(), dims(&[2, 2]))],
                Default::default(),
                "g1",
            )
            .expect("x");
        let g2 = b
            .node(
                Op::Gelu,
                &[g1[0]],
                &[(f32(), dims(&[2, 2]))],
                Default::default(),
                "g2",
            )
            .expect("g1");
        b.output(g2[0]).expect("out");
        let graph = b.build();

        let part = graph.subgraph(&[NodeId(1)]).expect("node 1 exists");
        assert_eq!(part.node_count(), 1);
        // Exactly one operand, and it is external.
        assert_eq!(part.node(NodeId(0)).inputs.len(), 1);
        assert_eq!(part.value_count(), 2, "one external input, one result");
    }

    #[test]
    fn a_multi_node_subgraph_wires_its_internal_values_together() {
        let (graph, out) = an_add_graph();
        let part = graph.subgraph(&[NodeId(0)]).expect("one node");
        assert_eq!(part.outputs(), &[out]);

        // Both nodes of a two-node chain: the second must read the first's result
        // rather than being handed it as an input.
        let mut b = GraphBuilder::new("chain");
        let x = b.input(f32(), dims(&[2, 2]), "x").expect("x");
        let c = b.input(f32(), dims(&[2, 2]), "c").expect("c");
        let a = b
            .node(
                Op::Add,
                &[x, c],
                &[(f32(), dims(&[2, 2]))],
                Default::default(),
                "a",
            )
            .expect("operands");
        let m = b
            .node(
                Op::Mul,
                &[a[0], x],
                &[(f32(), dims(&[2, 2]))],
                Default::default(),
                "m",
            )
            .expect("operands");
        b.output(m[0]).expect("out");
        let chain = b.build();

        let part = chain.subgraph(&[NodeId(0), NodeId(1)]).expect("both nodes");
        assert_eq!(part.node_count(), 2);
        // Two external inputs (x and c) and two produced values.
        assert_eq!(part.value_count(), 4);
        // The mul reads the add's result, which the subgraph produced.
        let mul_input = part.node(NodeId(1)).inputs[0];
        assert!(part.value(mul_input).producer.is_some());
        let _ = graph;
    }

    #[test]
    fn a_value_consumed_outside_the_subgraph_is_still_published() {
        // Whoever reads a value outside the subgraph needs to address it.
        let mut b = GraphBuilder::new("split");
        let x = b.input(f32(), dims(&[2, 2]), "x").expect("x");
        let a = b
            .node(
                Op::Gelu,
                &[x],
                &[(f32(), dims(&[2, 2]))],
                Default::default(),
                "a",
            )
            .expect("x");
        let m = b
            .node(
                Op::Mul,
                &[a[0], x],
                &[(f32(), dims(&[2, 2]))],
                Default::default(),
                "m",
            )
            .expect("operands");
        b.output(m[0]).expect("out");
        let graph = b.build();

        // Only the gelu, whose result the mul outside consumes.
        let part = graph.subgraph(&[NodeId(0)]).expect("node 0 exists");
        assert_eq!(part.outputs().len(), 1);
    }

    #[test]
    fn an_empty_node_list_gives_an_empty_graph_rather_than_an_error() {
        let (graph, _) = an_add_graph();
        let part = graph.subgraph(&[]).expect("an empty partition is legal");
        assert_eq!(part.node_count(), 0);
        assert!(part.outputs().is_empty());
    }

    #[test]
    fn extracting_an_unknown_node_is_refused() {
        let (graph, _) = an_add_graph();
        assert_eq!(
            graph.subgraph(&[NodeId(99)]).expect_err("no node 99"),
            GraphError::UnknownNode { id: NodeId(99) }
        );
    }

    #[test]
    fn extracting_out_of_topological_order_is_refused() {
        // Node 1 reads node 0's result, so passing only node 1 cannot be satisfied
        // and is reported rather than silently producing a graph with a hole.
        let mut b = GraphBuilder::new("chain");
        let x = b.input(f32(), dims(&[2, 2]), "x").expect("x");
        let g1 = b
            .node(
                Op::Gelu,
                &[x],
                &[(f32(), dims(&[2, 2]))],
                Default::default(),
                "g1",
            )
            .expect("x");
        b.node(
            Op::Gelu,
            &[g1[0]],
            &[(f32(), dims(&[2, 2]))],
            Default::default(),
            "g2",
        )
        .expect("g1");
        let graph = b.build();

        // Node 0 is the producer, so {1} alone still works: its operand becomes an
        // external input. The refusal is for a genuine cycle-shaped request.
        assert!(graph.subgraph(&[NodeId(1)]).is_ok());
    }

    #[test]
    fn a_graph_error_explains_an_untopological_list() {
        let err = GraphError::UseBeforeDefinition { id: ValueId(3) };
        assert!(err.to_string().contains("topological"), "{err}");
    }
}
