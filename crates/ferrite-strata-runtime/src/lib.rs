//! Host-side execution: plan a graph, compile it, run it.
//!
//! # What this crate adds to `ferrite-strata`
//!
//! The core crate has the pieces — a graph, a partitioner, a backend trait — and no
//! way to use them together. That is deliberate: the core has to be usable by a
//! vendor plugin, which compiles subgraphs and never plans. This crate is the other
//! half, the one a framework or an application holds.
//!
//! # The shape of a run
//!
//! ```text
//! graph -> plan -> compile -> run
//! ```
//!
//! Four steps, four types, and each one is separately inspectable, because a run that
//! misbehaves has to be diagnosable without a debugger:
//!
//! - [`Session`] holds the backends and the partition plan for one graph.
//! - [`Plan`] says which nodes go where, including the ones that go nowhere
//!   because no backend would take them.
//! - [`Step`] is one compiled, runnable unit, produced by [`Session::compile`].
//! - [`Step::run`] executes it.
//!
//! # Host nodes are not a failure
//!
//! The central property, and the one worth reading the code for: a graph no backend
//! fully supports still **runs**. Whatever falls to [`Placement::Host`](ferrite_strata::Placement::Host) is executed
//! on the CPU by the host, in the same value arena, so the results are the same
//! numbers a fully-accelerated run would produce. A node on the host costs time and
//! nothing else.
//!
//! This is the payoff of modelling decline as a first-class outcome rather than an
//! error. An inference graph where one op is unsupported is the normal case for a
//! young backend, and the alternative — refusing to run until every op is
//! supported — is why most frameworks end up with a single fused-everything path
//! and no fallback at all.
//!
//! # Example
//!
//! ```
//! use ferrite_strata::{DType, ElementType, GraphBuilder, Op, Shape};
//! use ferrite_strata_cpu::CpuBackend;
//! use ferrite_strata_runtime::{Session, inputs};
//!
//! let mut b = GraphBuilder::new("mlp");
//! let f32 = DType::plain(ElementType::F32);
//! let x = b.input(f32, Shape::new(&[1, 4])?, "x")?;
//! let sq = b.node(Op::Mul, &[x, x], &[(f32, Shape::new(&[1, 4])?)], Default::default(), "sq")?;
//! b.output(sq[0])?;
//! let graph = b.build();
//!
//! let mut session = Session::new();
//! session.add(Box::new(CpuBackend::new()))?;
//!
//! let plan = session.plan(&graph)?;
//! assert!(plan.is_fully_placed(), "the CPU backend takes everything here");
//!
//! let step = session.compile(&graph)?;
//! let results = step.run_with_inputs(&inputs(&[(x, vec![1.0, 2.0, 3.0, 4.0])]))?;
//!
//! assert_eq!(results.f32_at(sq[0]), Some([1.0, 4.0, 9.0, 16.0].as_slice()));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![doc(html_root_url = "https://docs.rs/ferrite-strata-runtime/0.0.1")]

mod compile;
mod session;

pub use compile::{ExecutionError, Step};
pub use session::{Plan, Session, Unit, inputs};

/// The flat tensor representation a [`Step`] reads and writes.
///
/// Row-major `f32`. Deliberately not the IR's [`DType`](ferrite_strata::DType):
/// this is the host's own value arena for the CPU path, and widening it to every
/// element type is the buffer ABI's job in B2. A backend that computes in `bf16`
/// converts at its own boundary.
pub type Tensor = Vec<f32>;
