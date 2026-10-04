# 0003. Define our own ABI rather than adopting PJRT

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

The plan asked whether to adopt the PJRT C API behind a safe Rust wrapper, on the
reasoning that reuse wins on stability. Reading the primary sources reverses that.
Three verified facts decide it.

1. PJRT has no ABI stability today. `docs/pjrt/pjrt_integration.md` on main still says
   "you need to match the jaxlib version with the PJRT C API version... We will
   start supporting ABI compatibility soon", and the header moved from minor 111 to
   116 inside the window one doc link spans.
2. The unit of exchange is wrong. One whole `HloModuleProto` in, one whole
   `PJRT_Executable` out, with no partial-acceptance contract. Subgraph
   partitioning with decline is not expressible.
3. The concept surface is foreign as well as oversized. Sharding is serialized
   `xla.OpSharding` protos, compile options are `CompileOptionsProto`, and four
   separate "capability surfaces" are all the same untyped `PJRT_NamedValue` string
   bag. Adopting PJRT means every vendor links XLA's protobuf schema and IR
   semantics, which is the coupling this project exists to avoid.

## Decision

A versioned `#[repr(C)]` vtable of roughly 10-15 functions, loaded with `dlopen`,
transplanting PJRT's *mechanics* rather than its surface:

- `struct_size` on every struct plus `offsetof`-based field-presence macros, so a
  caller built against an older header can binary-search for fields it knows.
- The `PJRT_Extension_Base` singly-linked chain, so extensions are additive.
- Errors as an object: a small closed code enum and a payload visitor, in place of
  PJRT's 17 abseil-shaped codes.
- The opaque-executable contract with its honest caveat intact: a serialised blob
  "must have been produced by the same platform and library version as this one" and
  "the serialization is not guaranteed to be stable over time".
- `abi_version` split into major and minor, where minor admits tail-appended vtable
  entries and a major bump means re-`dlopen`.

## Consequences

We own the ABI and its stability guarantees, which is the point. The cost is that we
cannot lean on PJRT's ecosystem for vendor onboarding, and interop has to be built
deliberately.

A vendor implements about a dozen functions against one vendored header, with no
XLA, no protobuf and no fork of a compiler codebase.
