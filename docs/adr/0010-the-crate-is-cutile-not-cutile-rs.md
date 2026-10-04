# 0010. The crate is cutile, not cutile-rs

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

The plan names `cutile-rs` as a crates.io dependency. It does not exist on crates.io.
`cutile-rs` is the **GitHub repository** name at `nvlabs/cutile-rs`. The crates are
`cutile`, `cutile-macro`, `cutile-compiler`, `cutile-ir`, `cuda-core`, `cuda-async`
and `cuda-bindings`, newest 0.4.0 (2026-09-25), MSRV 1.89 consistently across every
version, Apache-2.0.

## Decision

Where a tuned NVIDIA backend is wanted, the dependency is `cutile`, pinned. Real
dependents verified in their manifests: HuggingFace `candle`, `mistral.rs`,
`singe-kernel` and `axis`. Hardware floor is sm_80; sm_70 and sm_75 are unsupported.

The plan's substance holds, since this is stable Rust already in production use. Only
the name to type into Cargo was wrong.

## Consequences

Phase B3 scope. `cutile` is a tile DSL plus host launcher, i.e. kernel-level, and a
closed-IP vendor cannot adopt it: it is Apache-2.0 source you must read, JIT-only
through a CUDA toolkit the vendor does not control, and hard-gated to sm_80+.
