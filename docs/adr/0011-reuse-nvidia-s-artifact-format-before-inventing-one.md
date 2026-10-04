# 0011. Reuse NVIDIA's artifact format before inventing one

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

Our `serialize` / `deserialize_load` / `abi_compatible` triple needs to express "an
opaque executable blob plus the ABI version it was built for plus whether it can run
here". Designing that from scratch means inventing a second artifact format with the
same job.

## Decision

Study `oxide-artifacts` (0.2.1, Apache-2.0, 14,272 downloads), NVIDIA's published
crate for "architecture-neutral embedded device artifact metadata". It is the closest
existing thing to our opaque-executable primitive, it is real and mature, and reading
it before inventing ours costs a day.

We still define our own ABI; this is about not reinventing the *envelope*.

## Consequences

`oxide-artifacts` comes from an alpha tree that is not published to crates.io beyond
this one crate, so it is a reference rather than a dependency. Depending on it would
couple us to an alpha NVIDIA release train.

`PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable` remains the model for separating
executable portability from plugin-build compatibility.
