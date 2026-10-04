# 0008. CubeCL is optional, pinned, and default-off

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

CubeCL is the closest existing design to Strata's goals and the intended portable GPU
path. It is independently published and not coupled to Burn's public surface, which
makes it usable.

The dependency tree is the problem. `cubecl-core` pulls four git dependencies pinned
to bare revs, two of them to an in-development MLIR framework (`pliron`, crates.io
0.18.0) and a fork of `rspirv`. MSRV is 1.95. CubeCL's own README calls it alpha and
recommends pinning a version. Stable is 0.10.0 with `0.11.0-pre.4` in pre-release.

## Decision

`ferrite-strata-cubecl` is an optional crate behind a default-off feature, pinned to
an exact version. The core runtime does not depend on CubeCL.

CubeCL contributes three things regardless of whether we depend on it: the
`LaunchMode::Skip` probe discipline, the refusal taxonomy, and the
Runtime/ServerStorage/Compiler layering whose boundary informs ours.

## Consequences

A project whose premise is vendor-neutral ABI stability inheriting a moving MLIR IR
through unversioned git revs is the wrong trade. The cost is that the portable GPU
path is not always available in a default build.
