Ferrite Strata
==============

A vendor-neutral accelerator runtime, so hardware vendors can ship tuned LLM
kernels without exposing their architecture.

Status
------

**Phase B1, core implemented.** Three crates build and test: the graph IR and
backend traits, the CPU reference backend, and the host runtime. The C ABI
plugin (`B2`) and the `torch.compile` entry point (`B3`) are not started, and
no crates are published yet.

```console
$ cargo test --workspace     # 173 tests
$ cargo clippy --workspace --all-targets -- -D warnings
```

What works today is the whole loop from graph to numbers: build a graph, plan it
across registered backends, compile the partitions, run them, and fall back to
the host for anything no backend will take.

The idea
--------

Traits abstract the **API**, not the **performance**. So the boundary has to be
coarse enough that a vendor keeps its architecture private, and fine enough that
they can still optimise inside it.

**The vendor interface is subgraph-level, not kernel-level.** This is the
central design decision:

1. Strata partitions a model graph into subgraphs per backend.
2. A backend declares which ops and which *fused patterns* it supports, plus
   dtypes, quant formats, memory limits, and collectives.
3. The vendor compiles and schedules the subgraph internally and returns an
   opaque executable.

A vendor never publishes an ISA, a scheduling model, or kernel source. Weights
and activations do cross into the plugin, so vendors see tensors — but the user
never sees the vendor's architecture.

Crate layout
------------

| Crate | Purpose | Phase |
|---|---|---|
| `ferrite-strata` | Graph IR, `Backend`/`Executable` traits, capabilities, partitioning | B1 **done** |
| `ferrite-strata-runtime` | Host execution: session, plan, compile, run | B1 **done** |
| `ferrite-strata-cpu` | CPU reference backend: the correctness oracle for every op | B1 **done** |
| `ferrite-strata-cubecl` | Portable GPU path via CubeCL (wgpu/CUDA/ROCm/Metal) | B1 not started |
| `ferrite-strata-plugin-api` | `#[repr(C)]` versioned vtable, `dlopen` loader, safe wrapper | B2 |
| `ferrite-strata-pytorch` | `torch.compile` entry point, via PyO3 | B3 |

The interface, as implemented
----------------------------

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn plugin_version(&self) -> PluginVersion;
    fn capabilities(&self) -> &StrataSupport;

    /// The pull query: can this backend take this pattern, for this dtype and
    /// shape class? Answered *before* compiling, which is what makes a partition
    /// plannable. PJRT has no equivalent; Burn can only run a search.
    fn supports(&self, pattern_id: &str, dtype: DType, shape_class: ShapeClass) -> SupportLevel;

    /// Compile. Three outcomes, and the middle one is the design.
    fn compile(&self, subgraph: &Graph, pattern_id: &str) -> Result<CompileOutcome, Error>;
}

pub enum CompileOutcome {
    Compiled(Box<ExecutableId>),
    Declined { backend: String, reason: String },   // not an error
}

pub trait Executable: Send {
    fn label(&self) -> &str;
    fn graph(&self) -> &Graph;
    fn outputs(&self) -> &[ValueId];
    fn run(&self, arena: &mut Arena) -> Result<(), Error>;
}
```

Two things in there are the whole project:

**Decline is not an error.** A backend that has no way to say "not mine" accepts
everything and fails at run time, or refuses everything and cannot be used at
all. Both prior arts had to retrofit this: PyTorch/XLA with a tri-state lowering
step, Burn with `is_refusal()` bolted onto an error type that had no room for
it. Here it is a third arm of the return type, so declining is cheaper than
pretending and a planner can route around it.

**The query is a pull.** `supports` is asked before compiling, so a partition can
be costed rather than discovered.

Host fallback is not a fallback plan
------------------------------------

A graph no backend fully supports still runs. Nodes that fall to
`Placement::Host` are computed on the CPU in the same arena, so the numbers are
the ones a fully-accelerated run would produce — a node on the host costs time
and nothing else.

That is the practical payoff of treating decline as a first-class outcome. One
unsupported op is the normal case for a young backend, and refusing to run
until everything is supported is why most frameworks end up with a single
fused-everything path and no fallback at all.

Rust has no stable ABI, so trait objects cannot cross a closed-source library
boundary. The plugin boundary is a versioned `#[repr(C)]` vtable loaded with
`dlopen`, with a safe Rust wrapper implementing `Backend` on top of it.

Open questions, in `docs/design-notes.md`
------------------------------------------

- Adopt the PJRT C API wholesale, or define a smaller Strata vtable?
- Which real vendor or hardware target is first, besides the FPGA backend?
- Does the AOT compile-service mode ship, where the vendor returns a compiled
  blob with no source and no schedule?

Non-goals
---------

Replacing CUDA/cuDNN/cuBLAS on NVIDIA hardware, training support (inference
first), and a general Rust-to-FPGA path.

Licence
-------

Apache-2.0, chosen for the explicit patent grant: third-party hardware vendors
are the ones we need to trust with a plugin ABI, and patent exposure is the
first question a vendor legal team will ask. See `LICENSE` and `NOTICE`.
