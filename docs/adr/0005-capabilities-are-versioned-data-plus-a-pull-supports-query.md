# 0005. Capabilities are versioned data plus a pull supports query

- Status: Accepted
- Date: 2026-10-04
- Applies to: `ferrite-strata`

## Context

This is the one genuinely new invention relative to both prior arts, and it is what
makes the partitioner plannable.

PJRT cannot express it. Its nearest mechanisms, `PJRT_XlaTransform` and
`PJRT_Custom_Partitioner`, are *push* callbacks where the framework hands the vendor
XLA HLO and waits. There is no "do you support pattern X for these dtypes?" pull
query anywhere in PJRT.

Burn can only approximate it by running a search. `FusionRuntime::fusers(device)`
returns the *set of matchers*, not a boolean, so answering "supports flash-attn-v3
for bf16?" requires executing the search. A planner therefore cannot cost a partition
before committing to a compile. Burn's pattern vocabulary is also a closed enum in a
5,292-line file, so adding a pattern means editing their IR crate.

## Decision

`capabilities` returns a self-describing, independently versioned blob. `supports`
is a pull function:

    supports(pattern_id, dtype_mask, shape_class) -> Refuse | Fallback | Fused

The answer is *data*, so Strata can snapshot a vendor's capabilities, cache them,
print them, diff them between plugin versions, and cost a partition before compiling.
Pattern ids are vendor-owned extensible strings, not a closed enum.

Numeric formats are first-class in that blob, including int8 block scales and
zero-points, which PJRT has no representation for at all: quantisation rides inside
HLO rather than in the ABI.

## Consequences

Answering in data cannot be exact for every case. A callback can consider state the
vendor does not want to serialise. We start with data and allow an extension for the
callback case.

A capability blob format is a second thing to version and keep stable, which is a
cost of this choice.
