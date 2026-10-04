# 0009. Never depend on the crates.io crate named cuda-oxide

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

`cuda-oxide` on crates.io is an unrelated, abandoned 2021 crate by a third party,
licensed **GPL-3.0-or-later**, with 25,781 downloads. It is not NVIDIA's.

NVIDIA's cuda-oxide lives at `NVlabs/cuda-oxide`, which has been **renamed to
`NVIDIA/cuda-rust`**, and it is not published to crates.io at all. A contributor
following the plan's risk-5 note and typing `cargo add cuda-oxide` would silently
pull GPL-3.0 code into an Apache-2.0 tree and contaminate the licence of anything
built from it.

## Decision

No dependency may be added by the bare crate name `cuda-oxide`. If this is ever
wired up, it must be a git dependency on `NVIDIA/cuda-rust`, behind a default-off
feature, with the provenance recorded in `NOTICE`.

This belongs in CI as an automated dependency-policy check rather than in review,
because it is exactly the kind of mistake that passes a human read.

## Consequences

There is no published `cuda-oxide` to depend on, so this costs us nothing today. The
ADR exists so the trap is documented before someone falls into it, since the plan
still names the crate.

`std::offload` is likewise not a foundation: nightly-only experimental, its tracking
issue open since 2024, and its own champion describing the device side as not yet
reviewed or sufficiently tested.
