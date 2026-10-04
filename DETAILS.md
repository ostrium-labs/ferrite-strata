# Ferrite Strata: details

Everything known about this project in one place. Companion documents:

- `docs/design-notes.md` — the vendor interface and plugin ABI, read from PJRT, CubeCL and Burn source
- `docs/adr/` — 15 accepted architecture decision records
- `../ferrite-lithic/DETAILS.md` — the companion hardware DSL, and the ASIC competition chip

Status: **Phase 0 complete**, ABI settled, no crates published yet.

## 1. What it is

A vendor-neutral accelerator runtime, so hardware vendors can ship tuned LLM kernels
without exposing their architecture.

Traits abstract the **API**, not the **performance**. So the boundary must be coarse
enough that a vendor keeps its architecture private, and fine enough that they can
still optimise inside it.

The immediate deliverable is the **ABI and the graph IR**, not a backend. A working
runtime with no vendor is a library; a working ABI is a platform.

## 2. Identity

| | |
|---|---|
| Repository | `ostrium-labs/ferrite-strata`, default branch `dev` |
| Crates | `ferrite-strata`, `-runtime`, `-plugin-api`, `-cpu`, `-cubecl`, `-pytorch` |
| Licence | Apache-2.0, copyright appendix filled |

Names are prefixed `ferrite-` because `strata` and `strata-core` are permanently
taken on crates.io and `strata` collides with OpenGamma Strata. See ADR-0002.

`NOTICE` records PJRT/OpenXLA, CubeCL and Burn as design references with **no code
copied**, and is written to be extended if we end up adopting the PJRT C API.

## 3. The central decision

**The vendor interface is subgraph-level, not kernel-level.** Strata partitions a
model graph into subgraphs per backend; a backend declares which ops and fused
patterns it supports; the vendor compiles and schedules the subgraph internally and
returns an opaque executable. A vendor never publishes an ISA, a scheduling model, or
kernel source.

Kernel-level was rejected on evidence. CubeCL's boundary is
`Server::launch(&mut self, kernel: Box<dyn CubeKernel>, ...)` — a Rust trait object,
in-process, one kernel at a time, with a live `pliron` IR and no wire format. A closed
vendor cannot participate in a `Box<dyn>` boundary at all.

## 4. Why not PJRT

The plan asked whether to adopt the PJRT C API behind a Rust wrapper, on the reasoning
that reuse wins on stability. Reading the primary sources reverses that. Three
verified facts:

**PJRT has no ABI stability today.** `docs/pjrt/pjrt_integration.md` on `main` still
carries *"you need to match the jaxlib version with the PJRT C API version... We will
start supporting ABI compatibility soon"*, and the header moved from minor 111 to 116
inside the window one doc link spans. A project whose central promise is "your closed
plugin keeps working" cannot rest on a contract whose own guide says versions must be
matched per build.

**The unit of exchange is wrong.** One whole `HloModuleProto` in, one whole
`PJRT_Executable` out, **no partial-acceptance contract**. Subgraph partitioning with
decline is not expressible; a vendor supporting part of a graph can only fail the
whole compile with `UNIMPLEMENTED`.

**The concept surface is foreign as well as oversized.** Sharding is serialized
`xla.OpSharding` protos, compile options are `CompileOptionsProto`, and four separate
"capability surfaces" (`Plugin_Attributes`, `TopologyDescription_Attributes`,
`Device_GetAttributes`, `Executable_GetCostAnalysis`) are all the same untyped
`PJRT_NamedValue` string bag. Adopting PJRT means every vendor links XLA's protobuf
schema and IR semantics — precisely the coupling the project exists to avoid.

PJRT is at `PJRT_API_MINOR 116` with **138 vtable fields**, and 20-plus opaque handle
types. That is a permanent obligation for an inference-first runtime that largely does
not need sharding, multi-device or collectives.

**Three names in the original brief do not exist**, recorded so they never enter our
vocabulary: `PJRT_CompiledExecutable` (it is `PJRT_Executable` versus
`PJRT_LoadedExecutable`), `PJRT_Sharding` (no such type; sharding is an extension
returning serialized protos), and `PJRT_ScanningExecutable` (not a PJRT concept at
all).

## 5. What we take from PJRT anyway

Its *mechanics* are excellent and cost nothing to copy:

1. **`struct_size` on every struct plus `offsetof`-based field-presence macros**, so a
   caller built against an older header can binary-search for fields it knows. The
   source is explicit: *"Always add new fields to the end of the struct."*
2. **The `PJRT_Extension_Base` singly-linked chain**, making extensions additive.
3. **Errors as an object** — `PJRT_Error` with `PJRT_Error_Code` and
   `PJRT_Error_ForEachPayload`. Ours is a smaller closed enum:
   `INVALID_ARGUMENT`, `NOT_FOUND`, `UNIMPLEMENTED`, `RESOURCE_EXHAUSTED`,
   `INTERNAL`, `DECLINED`, and nothing else.
4. **The opaque-executable contract with its caveat intact**:
   `PJRT_Executable_{Serialize,DeserializeAndLoad,Fingerprint}` and
   `PJRT_TopologyDescription_Fingerprint`. PJRT's own header says a blob *"must have
   been produced by the same platform and library version as this one"* and *"the
   serialization is not guaranteed to be stable over time."*
5. **`PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable`** — executable portability is
   a different question from plugin-build compatibility, and conflating them is a trap.

PJRT's dtype enum is also worth watching: it already carries `BF16` and MX-style
`F8E8M0FNU` / `F4E2M1FN`. But **int8 block-quant scales and zero-points have no
representation in the ABI** — quantisation rides inside HLO. That gap is ours to close.

**PJRT stays reachable** as an interop tier, with adapters in both directions, and as
a future *third* ABI tier rather than a redesign.

## 6. The one genuine invention

**Capabilities as versioned data, plus a pull query:**

    supports(pattern_id, dtype_mask, shape_class) -> Refuse | Fallback | Fused

PJRT cannot express this at all — its two nearest mechanisms are *push* callbacks that
hand the vendor XLA HLO and wait. Burn can only approximate it by **running a search**:
`FusionRuntime::fusers(device)` returns the set of matchers, not a boolean, so
answering "supports flash-attn-v3 for bf16?" requires executing the search, and a
planner cannot cost a partition before committing.

Answering in **data** means we can snapshot a vendor's capabilities, cache them, print
them, diff them between plugin versions, and cost a partition up front. Pattern ids
are vendor-owned extensible strings, not a closed enum — Burn's vocabulary is a closed
enum in a 5,292-line file, so adding a pattern means editing their IR crate.

## 7. The ABI

The plan's sketch `{ abi_version, capabilities, alloc, compile, run, free }` is the
right size and under-specified in five places. The revised shape, each delta with a
precedent:

```c
typedef struct StrataAbiVersion { uint32_t major; uint32_t minor; } StrataAbiVersion;

typedef struct StrataExtBase {
  size_t struct_size;
  StrataExtType type;
  struct StrataExtBase* next;        // PJRT's extension chain, copied
} StrataExtBase;

typedef struct StrataPluginV1 {
  size_t struct_size;
  StrataExtBase* extension_start;
  StrataAbiVersion abi_version;       // split, not one int

  void  (*error_free)(StrataError*);
  const char* (*error_code)(const StrataError*);
  void  (*error_payload)(const StrataError*, StrataPayloadVisitor, void*);

  int   (*capabilities)(void*, const uint8_t** out, size_t* out_len);   // data
  StrataSupport (*supports)(void*, const StrataPatternQuery*);          // pull

  int   (*compile)(void*, const StrataCompileArgs*, StrataExecutable**);
  int   (*alloc)(void*, const StrataAllocDesc*, StrataBuffer**);
  int   (*buffer_release)(void*, StrataBuffer*);
  int   (*buffer_event)(void*, const StrataBuffer*, StrataEvent**);
  int   (*event_ready)(void*, const StrataEvent*);
  int   (*event_await)(void*, const StrataEvent*);
  int   (*run)(void*, const StrataRunArgs*);
  int   (*free_executable)(void*, StrataExecutable*);
  int   (*free_buffer)(void*, StrataBuffer*);

  int   (*serialize)(void*, const StrataExecutable*, StrataBlob**);
  int   (*deserialize_load)(void*, const StrataBlob*, StrataExecutable**);
  int   (*abi_compatible)(void*, const StrataBlob*);
  void  (*free_blob)(void*, StrataBlob*);
} StrataPluginV1;
```

**Loading.** `dlopen`, resolve one symbol, negotiate major/minor (minor admits
tail-appended vtable entries, a major bump means re-`dlopen`), walk the extension
chain. A safe Rust wrapper implements `Backend` on top, so vendors see only
`#[repr(C)]` and never a Rust trait object — which is the entire point, since
`Box<dyn Backend>` cannot cross the boundary either.

## 8. The subgraph contract

`compile` is tri-state, and this is what makes vendor adoption possible:

| Outcome | Meaning | Strata's response |
|---|---|---|
| executable | vendor compiled and scheduled the subgraph | run it |
| `DECLINED` | vendor cannot or will not take this subgraph | run generically, or re-partition |
| hard error | vendor accepted it and failed | surface the `StrataError` |

Modelled on Burn's `ExecutionStrategy::{Optimization, Operations, Composed}`. A
subgraph *containing* a decline becomes the `Operations` arm; a region mixing both
becomes `Composed`. Decline-as-first-class-outcome is why partial adoption is possible
at all.

`CompileArgs` carries the subgraph blob, the capability blob the vendor was
advertised, and a `probe: bool` — CubeCL's `LaunchMode::Skip` discipline, where the
plugin must do *everything* a real launch does except dispatch, so a partitioner can
cost a subgraph without running it.

## 9. What Burn gets right, and what to avoid

Burn's split is **four traits at two levels**, not the two the plan assumed:

| Trait | Means |
|---|---|
| `DeviceOps` | an opaque `DeviceId` plus `defaults()`. Tiny, remote-capable |
| `Backend` | "I can execute ops eagerly" — ~500 methods across 8 supertraits |
| `BackendIr` | "I can hand my resources to the graph layer as handles" |
| `FusionBackend` + `FusionRuntime` | "I can compile a graph" and "here are the patterns I know" |

`Fusion<B>` implements the entire eager `Backend` **without calling `B`'s ops at all**,
recording IR into a queue instead.

**Steal:** `ExecutionStrategy` with an explicit unfused arm; `FuserStatus::Closed` as
a pull answer carrying `score` and `ready`; `is_refusal()` versus
`is_device_poisoned()`; serialised name-dispatched optimisation state; and a tiny
device surface.

**Avoid:** the ~500-method eager `Backend` no closed vendor would implement; a closed
pattern vocabulary enum; capability that can only be learned by running a search;
everything as an in-process trait object; three opt-in traits meaning graph support is
a compile-time bound a `dlopen`ed plugin cannot satisfy; `alias_handle` and
`free_handle`, which exist purely because remote handles break; the overloaded word
"graph"; and plain serde with no schema versioning across a trust boundary.

One Burn idea we *want* and cannot have: `GraphPrimitive` defaults to
`pub enum GraphUnsupported {}`, an uninhabited type, so "no graph support" is enforced
at compile time. Behind `dlopen` the impl does not exist, which is exactly why the
runtime equivalent is necessary.

## 10. Backends

| Backend | Purpose | Phase |
|---|---|---|
| CPU reference | correctness oracle for every op | B1 |
| CubeCL | portable GPU path, optional and pinned | B1 |
| sample closed plugin | a dummy vendor `.so` proving the ABI | B2 |
| `cutile` | tuned NVIDIA path on stable Rust | B3 |
| `torch.compile` entry point | PyTorch model through Strata | B3 |
| Lithic FPGA | graph to RTL to bitstream | bridge |

**CubeCL is default-off and pinned.** It is the closest prior art and independently
published, but `cubecl-core` pulls four git dependencies pinned to bare revs, two of
them to an in-development MLIR framework (`pliron`) and a fork of `rspirv`. MSRV 1.95.
Stable 0.10.0, pre-release `0.11.0-pre.4`. Its own README calls it alpha. A project
whose premise is ABI stability should not inherit a moving MLIR IR through unversioned
git revs, so the core runtime does not depend on it.

## 11. NVIDIA CUDA Rust, and a licence hazard

**`cuda-oxide` on crates.io is an unrelated abandoned 2021 crate, GPL-3.0-or-later,
25,781 downloads.** NVIDIA's lives at `NVlabs/cuda-oxide`, renamed to
`NVIDIA/cuda-rust`, and is not published to crates.io at all. `cargo add cuda-oxide`
in this Apache-2.0 tree would silently pull GPL-3.0 code in. Any future wiring must be
a git dependency behind a default-off feature with provenance in `NOTICE`, and this
belongs in **CI as a dependency-policy check**, not in review. See ADR-0009.

**`cutile-rs` is not a crate name** — it is the GitHub repository. The crates are
`cutile` and friends, newest 0.4.0, MSRV 1.89 consistently, Apache-2.0, with candle
(`cutile = "=0.3.1"`) and mistral.rs as verified dependents. Hardware floor sm_80.

**`oxide-artifacts` 0.2.1** (Apache-2.0, 14,272 downloads) is NVIDIA's published crate
for *"architecture-neutral embedded device artifact metadata"* — almost exactly our
opaque-executable plus ABI-version plus can-this-run-here primitive. Read it before
inventing the envelope. See ADR-0011.

**`std::offload` is not a foundation.** Nightly-only experimental, tracking issue open
since 2024, and its own champion describing the device side as not yet reviewed or
sufficiently tested.

## 12. The Loams connection, honestly

The bridge premise is that Lithic's FPGA flow becomes a Strata backend. An audit of all
~293k lines of the Loams workspace found the entire arithmetic core is roughly **200
lines across six functions**, and that **Loams owns no ANN algorithm at all** — it
configures IVF-PQ and delegates to `lance-index`, configures HNSW/SQ/PQ/BQ and
delegates to `qdrant-edge`. Aggregations have no numeric kernels and delegate to
Tantivy. Text is allocation- and Unicode-table-bound. Vector quantisation is
configuration only, in exact-pinned dependencies.

The one real kernel, `loams-query/src/vector.rs:20-44`, is textbook systolic-array
input and is deliberately shaped away from it: **D79**, owner-approved, makes
summarization order part of the public API contract — *"SIMD kernels differ in
summarization order between libraries"*. So no tree reduction inside a row, forcing
row-parallelism with sequential per-lane f64 accumulators, and f64 MACs cost 4–8x the
area of f32.

**No acceleration is claimed for Loams graph, text search or analytical workloads.**
There is nothing to accelerate. The honest future hardware target is **D92 sampled
continuous recall** — batch, throughput-oriented, f64, not yet implemented — which is
a follow-on chip.

Two Loams-side blockers for whenever integration starts: `Cargo.toml:205` sets
`unsafe_code = "forbid"` workspace-wide, so a DMA/MMIO ring buffer driver needs a lint
carve-out; and `lance = "=12.0.0"` / `qdrant-edge = "=0.8.0"` are exact-pinned under a
documented lockstep policy.

## 13. Flash backend: parked

No such product as "MIMX 3.1 flash" exists in any public source. `MIMX` is a real NXP
part-number prefix, but NXP has never sold stand-alone flash under it, and no vendor
ships a "3.1" flash architecture. Nothing is scheduled to launch on the stated date
either.

If the i.MX RT reading was intended, the opportunity is real and recorded for later:
there is **no Rust-native stack** for xSPI-octal LUT sequences, on-die SECDED ECC
reporting, hybrid 256KB/4KB sectors, or the FlexSPI FCB and eFuse auto-probe. On these
parts every flash operation is a **lookup table of SPI sequences**, not an opcode.

One trap to carry: on the Infineon parts, **programming the same 16-byte half-page
twice after an erase silently disables ECC**, a live hazard for a log-structured
writer.

## 14. Phases

| Phase | Deliverable |
|---|---|
| 0 | Design notes, naming and ABI settled — **done** |
| B1 | Strata traits, graph IR, CPU reference, optional CubeCL |
| B2 | C ABI plugin plus a separately built sample closed plugin |
| B3 | `torch.compile` entry point, `cutile` backend |
| bridge | Lithic FPGA backend through the Strata ABI |

**Inference only.** Non-goals: replacing CUDA/cuDNN/cuBLAS on NVIDIA, training
support, and any general Rust-to-FPGA path. Fast-moving LLM ops are handled by
versioning capability and dtype declarations and falling back, not by chasing every
new attention variant.

## 15. Open questions

1. **First real vendor or hardware target** (the plan's own open question 2). Still
   unanswered. The FPGA backend is the demo; a real edge NPU is the test.
2. **AOT compile-service mode** — a vendor returning a compiled blob with no source
   and no schedule. Our `serialize`/`deserialize_load` already have the right shape;
   whether it ships as a separate product is undecided.
3. **Quantisation representation.** int8 block scales and zero-points need a
   first-class ABI concept; PJRT has none. Block-scaled formats also interact with MX
   dtypes we may want in the capability blob.
4. **Whether `supports` returns data or a callback.** Data is plannable and cacheable;
   a callback can be exact. Starting with data, with an extension for the callback
   case.
5. **StableHLO/ONNX export** — how much belongs in B1 versus B3, and whether it
   constrains the IR we can evolve.