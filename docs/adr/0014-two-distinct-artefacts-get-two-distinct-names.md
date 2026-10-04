# 0014. Two distinct artefacts get two distinct names

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

Burn overloads the word "graph" for two unrelated things: `BackendTypes::GraphPrimitive`
means device graph capture and replay, while `Fusion` means kernel fusion. The
consequence is visible in Burn's own code, where `Fusion::<B>` passes
`type GraphPrimitive = B::GraphPrimitive` straight through with a comment about fused
kernels being recorded on the inner backend's stream.

## Decision

Strata names its two artefacts separately and never overloads either term:

- a **compiled subgraph** — the opaque executable a vendor returns
- a **captured launch sequence** — host-side recording and replay of dispatches

The same discipline applies to "backend" versus "plugin": the Rust trait is a
backend, the closed binary loaded over `dlopen` is a plugin.

## Consequences

Slightly more vocabulary than one overloaded term. Cheap compared to the debugging
cost of a plugin and a capture buffer being confused in a log line.
