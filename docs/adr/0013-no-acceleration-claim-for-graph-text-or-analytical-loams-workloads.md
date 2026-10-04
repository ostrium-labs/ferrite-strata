# 0013. No acceleration claim for graph, text or analytical Loams workloads

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

The bridge premise is that Lithic's FPGA flow becomes a Strata backend, so the
obvious question is which Loams workloads to accelerate.

An audit of all ~293k lines of the Loams workspace found the entire arithmetic core is
roughly 200 lines across six functions, and that Loams owns no ANN algorithm at all.
It configures IVF-PQ and delegates to `lance-index`, and configures HNSW, SQ, PQ and
BQ and delegates to `qdrant-edge`. `loams-query/src/exec/aggs.rs` contains no numeric
kernel: sum, avg, cardinality, percentiles, histogram and date_histogram are all
Tantivy collectors. The Porter stemmer allocates a `Vec<char>` per word and
tokenisation walks Unicode word-boundary tables. Vector quantisation is configuration
only, inside exact-pinned dependencies under a documented lockstep policy.

## Decision

Strata makes no acceleration claim for Loams graph traversal, text search or
analytical workloads. There is nothing to accelerate.

Sparse scoring and the sparse weight codec are recorded as SIMD-only: both are
branch-dominated over small trip counts, which is what SIMD cannot fix and what an
ASIC would still pay for in a merge network.

The honest future hardware target is **D92 sampled continuous recall**, which
re-runs 1% of vector queries exactly in the background under a CPU budget plus a
`POST /recall` endpoint pushing 25-100 sampled vectors through the exact kernel.
Batch, throughput-oriented, f64, and not yet implemented. That is a follow-on chip,
not this one.

## Consequences

Narrower claim, stronger evidence. The bridge stays a research track rather than a
product commitment, which is the honest position given the source.

Two Loams-side blockers are recorded for whenever integration starts:
`Cargo.toml:205` sets `unsafe_code = "forbid"` workspace-wide, so a DMA/MMIO ring
buffer driver cannot be written without a lint carve-out, and `lance = "=12.0.0"`
and `qdrant-edge = "=0.8.0"` are exact-pinned.
