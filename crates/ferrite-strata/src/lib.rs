//! Ferrite Strata: a vendor-neutral runtime for AI accelerators.
//!
//! # What this crate is
//!
//! Strata's graph IR and the traits a backend implements. It is the layer that
//! knows what a computation *is* and refuses to know what a device *is*.
//!
//! The split is the whole design. Strata owns the IR, the dtype and shape
//! facts, the partitioner, and the error taxonomy. A vendor plugin owns the
//! hardware, the compiler, and the choice of whether to take a given subgraph.
//!
//! # The one idea everything follows from
//!
//! A backend **may decline**.
//!
//! The alternative designs all fail the same way. A trait method that returns
//! `Result<Compiled, Error>` leaves the backend no way to say "not mine, send it
//! somewhere else" — so vendors accept everything and fail at run time, or
//! refuse everything and the runtime cannot use them at all. The evidence is in
//! [ADR-0006](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0006-compile-returns-a-tri-state-with-decline-as-a-first-class-outcome.md):
//!
//! - **PyTorch/XLA**, with a [tri-state](https://github.com/openxla/xla/blob/main/xla/service/gpu/llvm/parallel_task_assignment.cc)
//!   lowering step, added one for Strata.
//! - **Burn**, which needed
//!   [both](https://github.com/tracel-ai/burn/blob/main/burn-core/src/burn/serde/backend.rs)
//!   `is_refusal()` and `is_device_poisoned()` on `CompilationError` — that is,
//!   it had to encode decline in a type that had no room for it.
//!
//! So [`ErrorCode::Declined`] is a first-class outcome, not an error, and
//! [`StrataSupport`] is a pull query a planner calls *before* compiling. See
//! [`capabilities`] and [`backend`].
//!
//! # Inference only
//!
//! There is no gradient, no optimiser, and no autograd here. A subgraph is a
//! DAG of tensor ops; if a framework needs to differentiate it, it does so
//! outside Strata and produces a new forward subgraph.
//!
//! # Module map
//!
//! | Module | What it decides |
//! |---|---|
//! | [`dtype`] | Element types, and the narrow quantisation overlay |
//! | [`shape`] | Static shapes, and the coarse class capabilities are queried over |
//! | [`attrs`] | Op parameters, as validated key/value data |
//! | [`arena`] | The host's value store, which an executable reads and writes |
//! | [`graph`] | Values, ops, nodes, and the arena |
//! | [`capabilities`] | What a backend can do, as versioned data plus a pull query |
//! | [`backend`] | The traits a plugin implements |
//! | [`error`] | The six failure codes that cross the ABI |
//! | [`partition`] | Cutting a graph into subgraphs a backend will take |
//!
//! # Example
//!
//! ```
//! use ferrite_strata::{DType, ElementType, GraphBuilder, Shape, ShapeClass};
//!
//! let mut b = GraphBuilder::new("example");
//! let x = b.input(
//!     DType::plain(ElementType::F32),
//!     Shape::new(&[1, 4096])?,
//!     "x",
//! )?;
//!
//! // A leading 1 makes this an outer-product operand whatever its rank, which is
//! // the shape class a capability query is answered over.
//! let g = b.build();
//! assert_eq!(g.value(x).shape.class(), ShapeClass::OuterProduct);
//! assert_eq!(g.value(x).byte_len(), 4096 * 4);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![doc(html_root_url = "https://docs.rs/ferrite-strata/0.1.0")]

pub mod arena;
pub mod attrs;
pub mod backend;
pub mod capabilities;
pub mod dtype;
pub mod error;
pub mod graph;
pub mod id;
pub mod partition;
pub mod shape;

pub use arena::{Arena, ArenaEntry};
pub use attrs::{AttrError, AttrKey, AttrValue, Attributes};
pub use backend::{
    Backend, CompileOutcome, Executable, ExecutableId, PluginVersion, StrataVersion, Support,
};
pub use capabilities::{
    CapabilitySet, ComputeUnit, DTypeMask, DeclaredLimits, LimitReading, PatternSupport,
    StrataSupport, SupportLevel,
};
pub use dtype::{DType, ElementType, QuantSpec};
pub use error::{Error, ErrorCode};
pub use graph::{Graph, GraphBuilder, GraphError, Node, Op, Value};
pub use id::{NodeId, StableName, ValueId};
pub use partition::{Partition, PartitionPlan, Partitioner, Placement};
pub use shape::{Shape, ShapeClass, ShapeError};
