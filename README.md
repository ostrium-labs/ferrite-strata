Ferrite Strata
==============

A vendor-neutral accelerator runtime, so hardware vendors can ship tuned LLM
kernels without exposing their architecture.

Status
------

**Phase 0 (design).** No crates are published yet. The interface and ABI
decisions are in `docs/design-notes.md`; the crate layout below is the intended
shape.

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
| `ferrite-strata` | Graph IR, `Backend`/`Executable` traits, partitioning | B1 |
| `ferrite-strata-runtime` | Host runtime: buffers, streams, execution | B1 |
| `ferrite-strata-plugin-api` | `#[repr(C)]` versioned vtable, `dlopen` loader, safe wrapper | B2 |
| `ferrite-strata-cpu` | CPU reference backend: the correctness oracle for every op | B1 |
| `ferrite-strata-cubecl` | Portable GPU path via CubeCL (wgpu/CUDA/ROCm/Metal) | B1 |
| `ferrite-strata-pytorch` | `torch.compile` entry point, via PyO3 | B3 |

Interface sketch
----------------

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Capabilities;   // dtypes, ops, fused patterns, memory, collectives
    fn alloc(&self, bytes: usize, align: usize) -> Result<DeviceBuffer>;
    fn compile(&self, graph: &Subgraph, opts: &CompileOpts) -> Result<Box<dyn Executable>>;
}

pub trait Executable: Send + Sync {
    fn run(&self, stream: &Stream, inputs: &[BufferRef], outputs: &[BufferMut]) -> Result<()>;
}
```

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
