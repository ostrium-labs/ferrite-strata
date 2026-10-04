# 0002. Prefix crate names with ferrite-

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

`strata` and `strata-core` are both permanently taken on crates.io (0.1.1 with
3,647 downloads, and 0.1.0 with 742). Crate names are never released once claimed.
`strata` also collides with OpenGamma Strata, a mature market-risk library, and with
a popular AI repository of the same name. `ferrite` itself is taken too, by an image
viewer with 18,355 downloads.

GitHub organisation names were free, so the collision is specifically the crate
namespace and search.

## Decision

Repositories `ostrium-labs/ferrite-strata` and `ostrium-labs/ferrite-lithic`. Crates
`ferrite-strata`, `-runtime`, `-plugin-api`, `-cpu`, `-cubecl`, `-pytorch`, and
`ferrite-lithic*`. Repository and crate names match so the suite reads as one.

## Consequences

No top-level `ferrite` crate exists, so the suite's identity lives in the prefix.
Acceptable. Naming is settled and was a Phase 0 exit criterion.
