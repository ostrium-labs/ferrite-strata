# 0015. Inference first; no training support

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

Training brings autograd, optimiser state, checkpointing and a different performance
model. Supporting it would roughly double the interface surface before a single
vendor has implemented anything.

## Decision

Inference only. Non-goals also include replacing CUDA, cuDNN or cuBLAS on NVIDIA
hardware, and any general Rust-to-FPGA path.

Fast-moving LLM ops are handled by versioning capability and dtype declarations and
falling back to CPU or CubeCL for anything unsupported, rather than by trying to keep
up with every new attention variant.

## Consequences

Training workloads are out of scope until the inference path has a real vendor. If
that proves wrong, the capability-blob versioning already in place is where a new
dtype or pattern would be declared.
