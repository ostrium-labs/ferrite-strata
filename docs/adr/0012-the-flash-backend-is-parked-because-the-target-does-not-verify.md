# 0012. The flash backend is parked, because the target does not verify

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

A Strata flash backend was to target "MIMX 3.1 flash", described as unreleased and
due in two days.

No such product exists in any public source. `MIMX` is a real NXP part-number prefix
(`MIMXRT1062`, `MIMXRT1186`) but NXP has never sold stand-alone flash under it, and
no vendor ships a "3.1" flash architecture. Searches across Micron, Winbond, Macronix,
GigaDevice, ISSI, Infineon, Renesas, Microchip, Everspin, Kioxia and Samsung found
versioned architectures (BiCS gen 8/9/10, XM, SEMPER, OctaBus), none named 3.1.

## Decision

No flash backend is built. Designing against an unverifiable product would be
fabrication.

If the i.MX RT reading was intended, the opportunity is real and is recorded for
later: there is **no Rust-native stack** for xSPI-octal LUT sequences, on-die SECDED
ECC reporting, hybrid 256KB/4KB sectors, or the FlexSPI Flash Configuration Block and
eFuse auto-probe. `embedded-hal`, `sfdp` and `nor-quad` cover none of it, and on
these parts every flash operation is a lookup table of SPI sequences rather than an
opcode.

One trap to carry forward: on the Infineon parts, programming the same 16-byte
half-page twice after an erase silently disables ECC, which is a live hazard for a
log-structured writer.

## Consequences

No Strata storage backend exists yet, which is consistent with an inference-first,
graph-in-graph-out runtime that assumes tensors are already resident.

Revisit only against a part whose datasheet can be read.
