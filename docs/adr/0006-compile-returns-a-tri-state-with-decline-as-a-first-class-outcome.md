# 0006. compile returns a tri-state with decline as a first-class outcome

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

A vendor that supports only part of a graph has no way to say so in PJRT. Its only
lever is failing the entire compile with `UNIMPLEMENTED`. That makes partial
adoption impossible and pushes every vendor toward either "accept everything" or
"refuse everything".

Burn's `ExecutionStrategy` proves the three-way answer is the workable one:
`Optimization { opt, ordering, score }`, `Operations { ordering }`, and
`Composed(Vec<...>)`. The `Operations` arm is the unfused case, and `Composed` lets
part of a region be vendor-fused while the rest falls back.

## Decision

`compile` returns an executable, `DECLINED`, or a hard error:

| Outcome | Strata's response |
|---|---|
| executable | run it |
| `DECLINED` | run generically on CPU/CubeCL, or re-partition |
| hard error | surface the `StrataError` |

`StrataSupport` encodes the same distinction as `Refuse | Fallback | Fused`, in the
spirit of CubeCL's `CompilationError::is_refusal()` versus `is_device_poisoned()`.
"Vendor declines" is machine-actionable and distinct from "the device died" and "the
kernel is buggy".

`CompileArgs` also carries a `probe: bool`, the `LaunchMode::Skip` discipline, so a
partitioner can cost a subgraph without dispatching it.

## Consequences

Every backend must handle its own declines, which is more work for the CPU reference
backend than a pure-fallback design would be. That is the point: it keeps the
fallback path explicit and tested rather than implicit.
