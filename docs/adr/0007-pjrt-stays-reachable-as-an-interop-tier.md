# 0007. PJRT stays reachable as an interop tier

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

Not adopting PJRT is a decision about the vendor contract, not about cutting PJRT
out of the ecosystem. An existing vendor's PJRT plugin becoming a Strata backend for
free is valuable reach, and PJRT's README lists plugins such as IREE's.

## Decision

Build two adapters: Strata-to-PJRT, so Strata can consume PJRT plugins, and
PJRT-to-Strata, so Strata can be presented to JAX and PyTorch.

If PJRT-native vendor reach is genuinely needed later, add it as a **third ABI tier**
(`StrataPluginV2`) rather than a redesign. The `struct_size` plus `extension_start`
chain exists precisely so that extensions are additive.

## Consequences

Adapters are real maintenance surface against an API with no stability guarantee, so
they will need periodic work. Worth it: interoperability is the difference between a
usable runtime and a theoretical one.
