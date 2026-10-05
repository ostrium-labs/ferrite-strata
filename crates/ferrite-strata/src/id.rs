//! Dense indices into a [`Graph`](crate::graph::Graph)'s arenas.
//!
//! Newtypes over `u32`, not bare integers, because the alternative is a graph
//! full of interchangeable `u32`s where a transposition bug type-checks. The
//! partitioner and the planner both pass these around constantly.
//!
//! The width is deliberate: `u32` keeps an id pair in one register on the
//! hot path, and no real inference graph has four billion values. The
//! arenas are indexed with these, so the bound is also the natural bound on
//! graph size — a graph that would exceed it fails to build rather than
//! silently wrapping.

use core::fmt;

/// An index into [`Graph::values`](crate::graph::Graph::values).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ValueId(pub u32);

/// An index into [`Graph::nodes`](crate::graph::Graph::nodes).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(pub u32);

/// The stable name of a value or node, assigned at build time.
///
/// Deterministic and position-independent, unlike an arena index: it is derived
/// from the node's op and its inputs' names, so two builds of the same graph
/// produce the same names regardless of the order ops were added. That is what
/// makes a compiled subgraph cacheable across runs, and what lets a vendored
/// launch sequence be compared against a fresh one — see
/// [ADR-0014](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0014-two-distinct-artefacts-get-two-distinct-names.md).
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StableName(String);

impl StableName {
    /// A name taken verbatim, for a graph input supplied by the caller.
    ///
    /// Not for derived names — [`StableName::from_output`] is how an internal
    /// value gets one, so that every internal name has the same shape.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Derive a name for a value from its producer's name and output slot.
    ///
    /// The slot is part of the name because a node with two outputs has two
    /// distinct results, and `add(x, y)` is not the same value as `sub(x, y)`
    /// even where both take `(x, y)`.
    #[must_use]
    pub fn from_output(node: &StableName, slot: usize) -> Self {
        Self(format!("{node}#{slot}"))
    }

    /// The name as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StableName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}
