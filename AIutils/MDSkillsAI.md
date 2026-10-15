# Preparing EoS metadata with an AI assistant

Use this resource when creating, repairing, or auditing metadata for an EoS
table. The objective is a metadata file supported by evidence and accepted by
the loader—not a plausible-looking description based on guesses.

## Non-negotiable rule

If a column meaning, unit, normalization, charge constraint, phase convention,
critical point, or coexistence boundary cannot be established from the table
header, accompanying documentation, generator, or publication, **DO NOT GUESS.**
Record the uncertainty and ask the user. Never manufacture scientific
information merely to make validation pass.

Use these maintained references:

- [`templates/metadata_minimal.json`](../templates/metadata_minimal.json) for
  the smallest regular-table input;
- [`templates/metadata_regular.json`](../templates/metadata_regular.json) for
  the recommended ordinary starting point;
- [`templates/metadata_first_order_line.json`](../templates/metadata_first_order_line.json)
  or [`metadata_first_order_binodals.json`](../templates/metadata_first_order_binodals.json)
  for first-order tables;
- [`templates/metadata_master.json`](../templates/metadata_master.json) and
  [`METADATA_REFERENCE.md`](../templates/METADATA_REFERENCE.md) for every
  supported field and default;
- [`docs/METADATA.md`](../docs/METADATA.md) for scientific explanations.

## 1. Establish the source-file structure

Inspect the header and bounded samples from the beginning, middle, and end.
Record:

- filename and encoding;
- delimiter, comment prefix, and rows to skip;
- total numeric column count, including unused columns;
- missing-value tokens;
- whether every data row has the same width.

Do not use the position of a similarly named quantity in another table or
paper as evidence for this file.

## 2. Identify retained fields

For every retained field, record its zero-based column index, source label,
scientific meaning, source unit, and stored normalization. Classify it as:

- `coordinate`: `T`, followed by the independent `muB`, `muQ`, and `muS`;
- `target`: `e`, followed by `nB`, `nQ`, and `nS` in the same dimensional
  order;
- `auxiliary`: pressure, entropy, sound speed, susceptibility, phase fraction,
  or another quantity interpolated at returned states.

Coordinates and targets are required. Undocumented or unused source columns
may remain unmapped. Never assign a physical name merely to eliminate an
unmapped-column warning.

## 3. Establish canonical conversion

Canonical coordinates use GeV. Energy-like densities use GeV^4, and number or
entropy densities use GeV^3. Determine whether the file stores direct values
or quantities such as `e/T^4`, `nB/T^3`, MeV/fm^3, or fm^-3.

The supported multiplicative conversion is:

```text
canonical = raw * scale_factor * T_GeV^temperature_power
                * HBARC_GEV_FM^hbarc_power
```

Use only transformations justified by the source convention. Verify the
temperature unit before applying a temperature power. If the needed conversion
cannot be represented this way, stop and ask the user.

## 4. Establish the source grid

For every coordinate column:

1. count exact or tolerance-justified unique values;
2. verify finite, strictly increasing canonical axis values;
3. record bounds, spacing, and whether spacing is uniform;
4. compare the product of axis lengths with the row count;
5. verify that every Cartesian coordinate tuple occurs exactly once;
6. identify the row ordering and block structure.

The supported table is a complete rectilinear Cartesian product; axes may be
nonuniform. Canonical order is ascending, with the last coordinate changing
fastest. `grid.reorder: true` may reorder a complete grid in memory. It cannot
repair duplicates, holes, or scattered data.

Use `expected_shape`, `expected_bounds`, and `expected_spacing` only when they
are independently known. Values derived from the same possibly damaged input
do not provide an independent integrity check.

## 5. Determine dimension and charge constraints

Independent coordinate/target pairs determine dimensions 1–4. Document the
physics of omitted charges. The software default is zero omitted densities,
but do not accept that default when the scientific source might impose another
constraint.

Relations such as `nQ = 0.4*nB` and `nS = 0` use the documented homogeneous
linear constraint block. The tilde transform does not require tilde columns in
the source table; it uses canonical targets and the declared charge
constraints when a run requests `Variables = "tilde"`.

## 6. Record phase information only from evidence

Establish, when documented:

- whether a first-order transition is represented;
- the meaning and range of a phase-fraction field;
- a `line` sidecar with `muB,T` columns;
- a `binodals` sidecar with `muB,T_dilute,T_dense` columns;
- critical-point coordinates and whether they are known or approximate;
- intended pure- and mixed-phase interpretation.

Metadata JSON uses the lowercase representation values `"line"` and
`"binodals"`; they are mutually exclusive. Confirm sidecar units and column
indices. Use phase status `"unknown"` when information is unavailable rather
than claiming `"none"` without evidence. Read
[`FirstOrderSkillsAI.md`](FirstOrderSkillsAI.md) for the dedicated route.

## 7. Validate and report

Create a parameter file that points to the metadata, then run:

```bash
eos-trensmuther preprocess path/to/parameters.py
```

For a direct Python check, use the supported loader:

```python
from eos_trensmuther import ForwardTable

table = ForwardTable.load_validated("path/to/metadata.json")
```

Review errors, warnings, detected shape, row reordering, canonical ranges, and
the source-table sanity summary. Independently verify a few converted values.
Report provenance, publication identifiers, revision, validity limits, and
every unresolved scientific assumption.

A successful software validation proves structural interpretability under the
declared metadata. For ordinary tables, observed positive and negative regular
Jacobian orientations are a validation error; either uniform sign is allowed.
Declared first-order tables retain orientation reversals as diagnostics for
their phase-aware route. Passing the bounded diagnostic still does not prove
that the EoS is physically correct, monotonic, or globally one-to-one. Do not
add a persistent `validated` flag; validation is recomputed from the metadata
and numeric files.
