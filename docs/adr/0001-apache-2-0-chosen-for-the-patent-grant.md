# 0001. Apache-2.0, chosen for the patent grant

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

The plugin ABI in this repository is the surface third-party hardware vendors will
link against without reading our source. Their first legal question is patent
exposure.

MIT is simpler and matches Hardcaml, but it carries no patent grant and leaves that
question open on the one surface where adoption is the entire goal.

## Decision

`ferrite-strata` is Apache-2.0, copyright appendix filled as "Copyright 2026 The
Ferrite Authors". `ferrite-lithic` uses the same licence so the suite has one legal
posture.

## Consequences

Vendors get an explicit patent grant. Dual-licensing remains available per crate if
that ever conflicts with a dependency.
