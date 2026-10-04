# 0004. The vendor interface is subgraph-level, not kernel-level

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

Traits abstract the API, not the performance, so the boundary must be coarse enough
that a vendor keeps its architecture private and fine enough that they can still
optimise inside it.

Kernel-level was considered and rejected on evidence. CubeCL's boundary is
`Server::launch(&mut self, kernel: Box<dyn CubeKernel>, ...)`, which is a Rust trait
object, in-process, one kernel at a time, and `KernelDefinition` is a live `pliron`
IR with no wire format. A closed vendor cannot participate in a `Box<dyn>` boundary
at all.

## Decision

Strata partitions a model graph into subgraphs per backend. A backend declares which
ops and fused patterns it supports; the vendor compiles and schedules the subgraph
internally and returns an opaque executable.

A vendor never publishes an ISA, a scheduling model, or kernel source. Weights and
activations cross into the plugin, so vendors see tensors, but the user never sees
the vendor's architecture.

## Consequences

Fusion and granularity may leave performance on the table at subgraph boundaries. The
compensation is capability queries for fused patterns plus measurement against
cuBLAS and FlashAttention baselines.

It also means the unit of currency is a subgraph, which is a harder interface for a
vendor to implement well than a kernel. That is a real adoption cost and the reason
the first real vendor target is still an open question.
