//! A reference CPU backend for [`ferrite_strata`].
//!
//! # Why a CPU backend exists
//!
//! Not as a fallback. As an **oracle**.
//!
//! Every other backend in this ecosystem has the same test problem: how do you know
//! an NPU kernel is right? You compare it against something, and the something has
//! to be simple enough to be obviously correct. That is this crate. A f32 scalar
//! loop whose every line can be checked by eye is the only available ground truth
//! for a fused bf16 attention kernel, and there is no other candidate for the job —
//! a GPU backend is a second implementation to disagree with, not an authority.
//!
//! It is also the backend that makes the decline path testable. A real vendor plugin
//! is not available in CI, so the first thing a test needs is *a* backend that
//! genuinely refuses work it will not take. This one declines everything that is
//! not `f32` and every op it does not implement, which is honest rather than
//! convenient.
//!
//! # Scope, stated plainly
//!
//! `f32` only, 2-D matmul only, single-threaded, no autograd. Each of those is a
//! refusal of something the IR can express, and [`CpuBackend::compile_typed`] declines
//! those graphs rather than computing something subtly different. Correctness is
//! the entire point of this crate; speed is not a goal, and a version of this
//! backend that is fast but wrong would be worthless.
//!
//! # Example
//!
//! ```
//! use ferrite_strata::{Backend, DType, ElementType, GraphBuilder, Op, Shape, ValueId};
//! use ferrite_strata_cpu::{CpuBackend, CpuCompile, CpuTensor};
//!
//! let mut b = GraphBuilder::new("square");
//! let f32 = DType::plain(ElementType::F32);
//! let x = b.input(f32, Shape::new(&[2, 2])?, "x")?;
//! let sq = b.node(Op::Mul, &[x, x], &[(f32, Shape::new(&[2, 2])?)], Default::default(), "sq")?;
//! b.output(sq[0])?;
//! let graph = b.build();
//!
//! let backend = CpuBackend::new();
//! // The typed entry point, because `Backend::compile` erases the executable
//! // behind `dyn Executable` and the buffer ABI is not settled yet.
//! let executable = match backend.compile_typed(&graph, "ref::mul")? {
//!     CpuCompile::Compiled(executable) => executable,
//!     CpuCompile::Declined { reason } => panic!("unexpectedly declined: {reason}"),
//! };
//!
//! let inputs = std::collections::HashMap::from([(
//!     ValueId(0),
//!     CpuTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], Shape::new(&[2, 2])?)?,
//! )]);
//! let outputs = executable.run_values(&inputs)?;
//!
//! let result = outputs.get(&sq[0]).expect("the square is an output");
//! assert_eq!(result.as_slice(), &[1.0, 4.0, 9.0, 16.0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![doc(html_root_url = "https://docs.rs/ferrite-strata-cpu/0.0.1")]

mod backend;
mod tensor;

pub use backend::{CpuBackend, CpuCompile, CpuExecutable, Step, f32_tensor};
pub use tensor::{BroadcastFailure, CpuError, CpuTensor, Inputs};
