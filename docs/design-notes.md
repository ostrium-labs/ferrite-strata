# Ferrite Strata: design notes

Phase 0 output. Settles the vendor interface and the plugin ABI, so that Phase B1
is not built on an unexamined assumption about PJRT.

Claims about upstream are tagged **[V]** verified against a primary source
fetched 2026-10-04, **[S]** secondary (upstream README or doc, not
independently corroborated), **[U]** unverifiable. Primary files read:
`openxla/xla@main` `xla/pjrt/c/*` including `pjrt_c_api.h` (3241 lines) and
`docs/pjrt/pjrt_integration.md`; `tracel-ai/cubecl@main` runtime/core/ir/opt
crates; `tracel-ai/burn@main` `burn-backend`, `burn-ir`, `burn-fusion`,
`burn-cubecl-fusion`; the crates.io and PyPI JSON APIs.

## Settled

| Question | Decision |
|---|---|
| Repository | `ostrium-labs/ferrite-strata` |
| Crate names | `ferrite-strata*` — bare `strata` and `strata-core` are permanently taken on crates.io |
| Licence | Apache-2.0, for the explicit patent grant on the vendor-facing ABI |
| Open question 1 | **Define our own smaller vtable. Do not adopt PJRT wholesale.** Transplant PJRT's *mechanics*, not its *surface*. |
| Unit of exchange | Subgraph in, opaque executable out, with **decline as a first-class outcome** |
| Capability model | Capabilities as versioned **data**, plus a **pull** `supports(pattern, dtypes, shape)` query |

## 1. Why not PJRT

The plan asked whether to adopt the PJRT C API behind a Rust wrapper. Three
verified facts settle it against.

**PJRT has no ABI stability, today.** `docs/pjrt/pjrt_integration.md` on `main`
still carries: *"you need to match the jaxlib version with the PJRT C API
version… **We will start supporting ABI compatibility soon.**"* **[V]** The
header moved from `PJRT_API_MINOR 111` to `116` inside the window spanned by a
single doc link — roughly monthly churn. A project whose central promise is
"your closed plugin keeps working" cannot be built on a contract whose own guide
says versions must be matched per build.

**The unit of exchange is wrong.** `PJRT_Compile` takes one whole `PJRT_Program`
(`hlo`, `hlo_with_config`, or `mlir`) and returns one whole `PJRT_Executable`,
with **no partial-acceptance contract** **[V]**. Strata's entire thesis —
partition into subgraphs, let the vendor compile and schedule each, get back an
opaque executable — is not expressible. A vendor supporting only part of a graph
can only fail the whole compile with `UNIMPLEMENTED`.

**The concept surface is both larger than needed and foreign.** Sharding is
serialized `xla.OpSharding` protobuf; compile options are serialized
`CompileOptionsProto`; topologies serialise `HloModuleMetadataProto` **[V]**.
Adopting PJRT means every hardware vendor links or reimplements XLA's protobuf
schema and IR semantics — exactly the coupling the project exists to avoid. And
sharding, multi-device, collectives and megascale would become permanent vtable
obligations for what is an inference-first runtime.

### Three names in the original brief do not exist **[V]**

Worth recording so they never enter our design vocabulary:

| Assumed | Reality |
|---|---|
| `PJRT_CompiledExecutable` | Does not exist. It is `PJRT_Executable` (compile product) vs `PJRT_LoadedExecutable` (device-resident product). |
| `PJRT_Sharding` | No such type. Sharding is an *extension*, and returns serialized `xla.OpSharding` protos, not an object. |
| `PJRT_ScanningExecutable` | Not a PJRT concept at all. Nearest are `PJRT_Executable_NumPartitions` and the experimental `PJRT_PhaseCompile`. |

### What we take from PJRT verbatim **[V]**

PJRT's *mechanics* are excellent and cost us nothing to copy:

1. **`struct_size` on every struct, plus `offsetof`-based field-presence
   macros**, so a caller built against an older header can binary-search for
   fields it knows. The rule is explicit in the source: *"Always add new fields
   to the end of the struct."*
2. **The `PJRT_Extension_Base` singly-linked chain** — an extension mechanism
   that is additive rather than a redesign.
3. **Errors as an object**: `PJRT_Error` with 17 abseil-shaped
   `PJRT_Error_Code`s and `PJRT_Error_ForEachPayload`. Our codes are a small
   closed enum: `INVALID_ARGUMENT`, `NOT_FOUND`, `UNIMPLEMENTED`,
   `RESOURCE_EXHAUSTED`, `INTERNAL`, `DECLINED`, and nothing else.
4. **The opaque-executable serialisation contract, with its honest caveat
   intact**: `PJRT_Executable_{Serialize,DeserializeAndLoad,Fingerprint}` and
   `PJRT_TopologyDescription_Fingerprint`. PJRT's own header says a blob
   *"must have been produced by the same platform and library version as this
   one"* and *"the serialization is not guaranteed to be stable over time."* We
   adopt that language rather than pretending cross-version portability exists.
5. **`PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable`** — executable
   portability is a separate question from plugin-build compatibility, and
   conflating them is a trap.

PJRT's dtype enum is also worth watching: it already carries `BF16`, and MX-style
formats `F8E8M0FNU` / `F4E2M1FN` **[V]**. But **int8 block-quant scale and
zero-point have no representation in the ABI** — quantisation rides inside HLO,
not in the buffer surface. That is gap #3 in the list below and it is ours to
close.

### PJRT as an interop tier, not the vendor ABI

Adopting PJRT is not all-or-nothing, and the ruling is compatible with keeping
it reachable:

- Our own vtable is the vendor contract. Non-negotiable.
- Build **Strata→PJRT** and **PJRT→Strata** adapters, so an existing vendor's
  PJRT plugin becomes a Strata backend, and Strata can be presented to
  JAX/PyTorch. The PJRT README lists an IREE plugin **[V]**, so there is an
  ecosystem to interoperate with.
- If PJRT-native reach is genuinely needed later, add it as a **third ABI tier**
  (`StrataPluginV2`), not a redesign. The extension chain exists for exactly
  this.

## 2. The gap list, and the one thing we actually invent

What PJRT lacks:

1. Decline-a-subgraph.
2. A capability vocabulary. Its only structured advertisement is
   `PJRT_NamedValue` string keys via `PJRT_Plugin_Attributes` — four
   "capability surfaces" (`Plugin_Attributes`, `TopologyDescription_Attributes`,
   `Device_GetAttributes`, `Executable_GetCostAnalysis`) that are all the *same*
   untyped string bag **[V]**. No schema, no versioning, no enumeration.
3. Quantisation metadata in the ABI.
4. ABI stability.
5. A pull capability query. `PJRT_XlaTransform` and
   `PJRT_Custom_Partitioner` exist, but both are *push* callbacks where the
   framework hands the vendor XLA HLO and waits. There is no
   "do you support pattern X for these dtypes?" pull query.

What CubeCL lacks: any plugin concept at all **[V]**. Its vendor boundary is
`Server::launch(&mut self, kernel: Box<dyn CubeKernel>, ...)` — a Rust trait
object, in-process, kernel-granular **[V]**. `Box<dyn CubeKernel>` cannot cross
a `dlopen` boundary, `KernelDefinition` is a live `pliron` IR with no wire
format, and the refusal path is kernel-granular and only *after* the IR is
built.

**The one genuinely new invention, versus both prior arts:**

> Capabilities as versioned, serialised **data**, plus a **pull** query
> `supports(pattern_id, dtype_mask, shape_class) -> Refuse | Fallback | Fused`.

This is what makes the partitioner *plannable*. PJRT cannot express it at all.
Burn can only approximate it by running a search (below). Answering in data means
Strata can snapshot a vendor's capabilities, cache them, print them, diff them
between plugin versions, and cost a partition **before** committing to a compile.

## 3. What Burn gets right, and what to avoid

Burn's split is **four traits at two levels**, not the two the plan assumed, and
the real shape is more useful than "a Device trait and a Compiled trait":

| Trait | Means |
|---|---|
| `DeviceOps` **[V]** | An opaque `DeviceId` plus `defaults()`. Tiny, and remote-capable. |
| `Backend` **[V]** | "I can execute these ops eagerly." ~500 methods across 8 supertraits. |
| `BackendIr` **[V]** | "I can hand my resources to the graph layer as handles." The eager→graph seam. |
| `FusionBackend` + `FusionRuntime` **[V]** | "I can compile a graph" and "here are the patterns I know". |

`Fusion<B>` is a decorator that implements the entire eager `Backend` **without
calling `B`'s ops at all** — it records IR into a queue and delegates.

**Steal these:**

1. **`ExecutionStrategy { Optimization, Operations, Composed }`** **[V]** — a
   capability system with no first-class unfused arm forces every planner to
   invent a fallback. `Composed` is what lets *part* of a region be vendor-fused
   while the rest falls back. This is the proof our tri-state `compile` result is
   the workable shape.
2. **Negative capability on demand.** `FuserStatus::Closed`, carrying `score` and
   `ready` **[V]** — a pull answer to "can you absorb this?", asked per candidate
   subgraph. The only working fused-pattern query in the prior art.
3. **`CompilationError::is_refusal()` vs `is_device_poisoned()`** **[V]** —
   "vendor declines" is machine-actionable and distinct from "the device died"
   and "the kernel is buggy". Encode exactly this in `StrataSupport`.
4. **Serialised, name-dispatched optimisation state** **[V]** — a fused plan can
   be cached and restored without the runtime knowing its type. Directly
   applicable to a vendor-supplied opaque subgraph plan.
5. **Tiny device surface**: opaque id plus defaults.

**Avoid these:**

1. **`Backend` is ~500 eager methods.** A closed vendor would have to implement
   all of it. This is the strongest argument for subgraph-level granularity and
   the reason we do not model the vendor interface at op granularity.
2. **The pattern vocabulary is a closed enum.** `burn-ir/src/operation.rs` is
   **5292 lines** with an `OperationIr` dispatching into eleven sub-enums **[V]**;
   adding a fused pattern means editing Burn's IR crate. **Our pattern ids must be
   extensible, vendor-owned strings or blobs**, versioned independently of our
   release train.
3. **Capability cannot be queried ahead of time.** `fusers(device)` returns the
   *set of matchers*, not a boolean — answering "supports flash-attn-v3 for
   bf16?" requires running the search. A planner cannot cost a partition in
   advance.
4. **Everything is an in-process trait object**, so no closed vendor can
   participate.
5. **Three opt-in traits on one type** means graph support is a compile-time
   bound. A `dlopen`ed plugin has no trait bounds; **we must register capability
   at load time, not implement it.**
6. **`alias_handle` / `free_handle` exist only because remote handles break**
   **[V]** — the doc comments say so outright. Designing for a remote handle from
   day one avoids two escape hatches.
7. **"Graph" is overloaded**: `GraphPrimitive` means device capture/replay while
   `Fusion` means kernel fusion. Our two artefacts — a vendor's compiled subgraph
   and a host launch-sequence capture — must be two distinct types with two
   distinct names.
8. **No versioned wire format** across the trust boundary: `burn-ir` is plain
   serde + ciborium with no schema versioning.

One Burn idea we *want* and cannot have: `GraphPrimitive` defaults to
`pub enum GraphUnsupported {}`, an uninhabited type, so "no graph support" is
enforced at compile time **[V]**. Behind `dlopen` the impl does not exist, which
is precisely why we need the runtime equivalent — the tri-state `StrataSupport`.

## 4. CubeCL as the B1 GPU backend

Usable, and a poor structural fit. Independently published, not coupled to
Burn's public surface **[V]**, but:

- MSRV **1.95**, workspace licence `MIT OR Apache-2.0`, stable **0.10.0**
  (2026-05-07), current pre-release `0.11.0-pre.4` **[V]**.
- **Four git dependencies pinned to bare revs**, two of them to an in-development
  MLIR framework (`pliron`, crates.io 0.18.0) and a fork of `rspirv` **[V]**.
- CubeCL's own README calls it alpha and recommends pinning a version **[V]**.

For a project whose premise is vendor-neutral ABI stability, inheriting a moving
MLIR IR through unversioned git revs is the wrong trade. We depend on it
**optionally, pinned, behind a default-off feature**, so the core runtime never
carries it.

CubeCL contributes three things worth taking regardless:

- **`LaunchMode::Skip`** **[V]** — a probe mode where the plugin must do
  *everything* a real launch does except dispatch: parse, plan, allocate,
  validate. This is exactly what a partitioner needs to cost a subgraph without
  running it.
- **The refusal taxonomy**, above.
- **The three-trait layering**, `Runtime` / `ServerStorage` / `Compiler`, whose
  boundary sits between `Server::launch` and `Compiler::compile`. Ours is
  `Backend` / `Executable`, one level coarser by design.

## 5. NVIDIA CUDA Rust — three corrections to the plan

The plan's risk 5 says to depend on stable crates and isolate nightly features.
Two of its names are wrong in ways that matter.

**`cutile-rs` is not a crate; it is the GitHub repository name** **[V]**. The
crates are `cutile`, `cutile-macro`, `cutile-compiler`, `cutile-ir`, `cuda-core`,
`cuda-async`, `cuda-bindings` — newest **0.4.0** (2026-09-25), MSRV **1.89**
consistently across every version, Apache-2.0 **[V]**. Real dependents, verified
in their manifests: HuggingFace **candle** (`cutile = "=0.3.1"`), **mistral.rs**,
`singe-kernel`, `axis` **[V]**. So the plan's substance is right — stable Rust,
in production use — but `cargo add cutile-rs` would fail. Hardware floor is
sm_80; sm_70/sm_75 are unsupported **[V]**.

**`cuda-oxide` on crates.io is an unrelated squatter and a licence hazard.**
It is a 2021, **GPL-3.0-or-later**, abandoned crate by `Protryon`, 25,781
downloads **[V]**. NVIDIA's cuda-oxide lives at `NVlabs/cuda-oxide`, which has
been **renamed to `NVIDIA/cuda-rust`** **[V]**.

> A project whose entire premise is vendor adoption cannot have `cargo add
> cuda-oxide` silently pull GPL-3.0 code into an Apache-2.0 tree. If we ever wire
> this up, it must be by git dependency on `NVIDIA/cuda-rust`, behind a
> default-off feature, with the provenance recorded in `NOTICE` — never by crate
> name. This belongs in CI as a dependency-policy check, not just in review.

NVIDIA's cuda-oxide is pinned to `nightly-2026-08-28` with `rust-src`,
`rustc-dev`, `llvm-tools` **[V]**, and the pipeline is Rust → MIR → Pliron → LLVM
→ PTX **[V]**. It is **not published to crates.io**.

**One crate from that tree is directly useful to us**: `oxide-artifacts` 0.2.1,
Apache-2.0, 14,272 downloads — NVIDIA-authored *"architecture-neutral embedded
device artifact metadata"* **[V]**. That is precisely the opaque-executable +
ABI-version + can-this-run-here primitive we need. Study it before inventing one.

**`std::offload` is not a foundation.** Served at 1.99.0 and marked
*"🔬 nightly-only experimental API (`gpu_offload` #131513)"*, tracking issue open
since 2024, and the device side described by its own champion as *"not yet
reviewed or sufficiently tested"* **[V]**. The LLVM 22 migration issue has 5 of 7
checklist items unchecked **[V]**. Do not design around it.

## 6. The ABI

The plan's `{ abi_version, capabilities, alloc, compile, run, free }` is the
right size and spirit, under-specified in five places. Revised shape, with each
delta's precedent:

```c
typedef struct StrataAbiVersion { uint32_t major; uint32_t minor; } StrataAbiVersion;

typedef struct StrataExtBase {
  size_t struct_size;
  StrataExtType type;
  struct StrataExtBase* next;          // PJRT's extension chain, copied
} StrataExtBase;

typedef struct StrataPluginV1 {
  size_t struct_size;                   // sizeof(StrataPluginV1) as built
  StrataExtBase* extension_start;
  StrataAbiVersion abi_version;         // (a) split, not one int

  void  (*error_free)(StrataError*);
  const char* (*error_code)(const StrataError*);
  void  (*error_payload)(const StrataError*, StrataPayloadVisitor, void* user_arg);

  int   (*capabilities)(void*, const uint8_t** out, size_t* out_len);  // (b) data
  StrataSupport (*supports)(void*, const StrataPatternQuery*);          // (c) pull

  int   (*compile)(void*, const StrataCompileArgs*, StrataExecutable** out);
  //                                                       // (d) tri-state, §7

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

**Loading.** `dlopen`, resolve one symbol, negotiate
`abi_version.major/minor` (minor admits tail-appended vtable entries; major means
re-`dlopen`), walk the extension chain for optional capabilities. A major bump
rejects rather than adapts.

**The safe Rust wrapper** implements `Backend` on top of this, so vendors see
only `#[repr(C)]` and never a Rust trait object — which is the entire point, since
`Box<dyn Backend>` cannot cross the boundary either.

## 7. The subgraph contract

`compile` is tri-state, and this is the contract that makes vendor adoption
possible:

| Outcome | Meaning | Strata's response |
|---|---|---|
| executable | Vendor compiled and scheduled the subgraph | Run it |
| `DECLINED` | Vendor cannot or will not take this subgraph | Run it generically on CPU/CubeCL, or re-partition |
| hard error | Vendor accepted it and failed | Surface `StrataError` |

Modelled on Burn's `ExecutionStrategy::{Optimization, Operations, Composed}`
**[V]**. A subgraph *containing* a decline becomes the `Operations` arm; a
region mixing both becomes `Composed`. This is why the decline is tri-state and
not a boolean.

`CompileArgs` carries the subgraph blob, the capability blob the vendor was
advertised, and a `probe: bool` flag — the `LaunchMode::Skip` discipline from
CubeCL, so a partitioner can cost a subgraph without dispatching it.

## 8. Open questions

1. **First real vendor or hardware target** (plan open question 2). Unchanged and
   still unanswered. The FPGA backend is the demo; a real edge NPU is the test.
2. **Does the AOT compile-service mode ship?** A vendor returning a compiled blob
   with no source and no schedule. Our `serialize`/`deserialize_load` already
   have the right shape; whether it is a separate product is undecided.
3. **Quantisation representation.** int8 block scales and zero-points need a
   first-class ABI concept — PJRT has none, riding in HLO instead. Block-scaled
   formats also interact with MX dtypes we may want in the capability blob.
4. **Whether `supports` returns data or a callback.** Answering in data is
   plannable and cacheable; a callback can be exact. We start with data and
   allow an extension for the callback case.
5. **StableHLO/ONNX export** — how much of this is in scope for B1 versus B3, and
   whether it constrains the IR we can evolve.

## Verification notes

Unverified, carried forward honestly **[U]**: whether NVIDIA will publish the rest
of `cuda-rust`'s crates; `cutile`'s TileGym/Grout integration and its B200
performance figures (README and blog claims only); whether jaxlib 0.11.2 embeds
PJRT exactly 0.116 (both were confirmed independently, the mapping between them
was not); and CubeCL's full resolved dependency count, where only the notable
direct dependencies and four git-pinned revs were enumerated.