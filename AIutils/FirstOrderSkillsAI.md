# First-order EoS guidance for AI assistants

Use this resource when metadata explicitly declares a first-order transition,
or when interpreting outputs produced by the 2D LINE/BINODALS/Region-S route.
This route uses phase information declared in metadata. It does not discover
phase structure from an ordinary table and is not a general higher-dimensional
multibranch thermodynamic solver.

## Core rule

Treat metadata and its referenced scientific source as authoritative. Do not
infer a first-order transition, coexistence boundary, phase-fraction meaning,
or critical point from filenames, visual irregularities, Jacobian behavior, or
coarse table structure alone. If required physics is missing, ask the user.

Use the lowercase JSON values `"line"` and `"binodals"`. The uppercase terms
LINE and BINODALS below are descriptive names for the corresponding runtime
routes.

Consult:

- [`templates/metadata_first_order_line.json`](../templates/metadata_first_order_line.json);
- [`templates/metadata_first_order_binodals.json`](../templates/metadata_first_order_binodals.json);
- [`docs/METADATA.md`](../docs/METADATA.md);
- [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md);
- [`docs/KNOWN_LIMITATIONS.md`](../docs/KNOWN_LIMITATIONS.md).

## LINE representation

For a declared `"line"` representation:

1. verify the referenced coexistence file exists;
2. verify the declared `muB` and `T` columns and units;
3. check finite values, usable ordering, and coverage relative to the main EoS;
4. verify the declared critical-point reference and matching tolerances;
5. preserve the metadata's interpretation of the two sides of the line.

Do not reconstruct a line from apparent table discontinuities, approximate a
missing line, or infer undocumented column meanings.

## BINODALS representation

For a declared `"binodals"` representation, the supported sidecar normally
contains both branches in one file:

```text
muB  T_dilute  T_dense
```

Verify the file, all three declared columns and units, finite values, branch
ordering required by the declared convention, and overlap with the main-table
domain. Do not infer branch meaning from a filename or silently exchange the
dilute and dense columns.

## Phase-fraction auxiliary field

Use a phase-composition or phase-fraction field only when metadata explicitly
names it and documents its meaning. A value between zero and one may describe
position inside the coexistence region, but that interpretation is not
universal. Never assign this meaning from a suggestive column name alone.

Check consistency with the declared boundaries when useful, but do not redefine
the source's phase convention.

## Critical point

Record or use a critical point only when it is explicitly supported by the
metadata and source documentation. Do not estimate it from:

- table resolution or apparent endpoints;
- abrupt derivatives or Jacobian signs;
- visible mixed-phase points;
- plotting alone.

Report a mismatch between the declared critical point, sidecar, and main table;
do not silently correct it.

## Validate the complete first-order input

Before inversion, verify that:

- the main table and sidecar use compatible coordinate conventions and units;
- all referenced files and fields exist;
- declared ranges overlap appropriately;
- the phase-fraction field exists when required;
- metadata validation and source-table sanity checks pass.

Use [`MDSkillsAI.md`](MDSkillsAI.md) when creating or repairing metadata, then
run `eos-trensmuther preprocess path/to/parameters.py`.

## Multiple verified inverse roots

A target inside or near the coexistence region may produce more than one
verified inverse root. Never select a root using discovery order, worker order,
smallest residual, continuation from a neighboring target, or a globally
preferred Jacobian sign.

Root verification, duplicate handling, bounds checks, and phase-membership
checks occur before any branch selection.

### Local Jacobian-consistency recovery

A finite-resolution table can create a small local multilinear fold. The
implementation may select a branch only when:

1. all ambiguous candidates are fully verified;
2. regular cells in the one-cell surrounding halo have one unanimous
   Jacobian orientation after candidate, singular, sign-changing, and invalid
   cells are excluded;
3. exactly one candidate has that orientation;
4. that candidate is not Jacobian-near-singular.

The surrounding orientation may be positive or negative. Positive orientation
is not intrinsically preferred. Majority voting is not used.

The selected point remains explicitly marked:

```text
SELECTED_FROM_JACOBIAN
```

In the generic text table this is represented by
`selected_from_jacobian=1`; `phase_multivalued=1` records that another distinct
physical root was verified. The authoritative `.eosinv` artifact retains the
selection-reason code and verified mixed-root count. A distinct alternate
single-phase source coordinate is stored when that separate diagnostic case
applies; rejected mixed-root coordinates are not promised as public output.

If the surrounding geometry is mixed, insufficient, or singular; if no unique
candidate matches it; or if several candidates have the expected sign, the
case remains ambiguous. Do not add a local selection rule of your own.

This recovery is a conservative table/interpolation rule, not a universal
thermodynamic branch-selection principle.
