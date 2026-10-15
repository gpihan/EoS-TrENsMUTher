# Interpreting EoS-TrENsMUTher outputs with an AI assistant

Use this resource when reading, filtering, reshaping, comparing, or plotting a
completed run. Let the user choose the scientific quantity and visualization;
do not silently replace requested data with projected or filtered values.

The maintained format reference is
[`docs/OUTPUT_FORMAT.md`](../docs/OUTPUT_FORMAT.md).

## Choose the appropriate source

A run can contain:

- `inverse.eosinv`: authoritative checksummed result with all arrays, statuses,
  proof codes, phase information, and provenance;
- `eos_table.dat` or another configured filename: convenient numeric text
  view;
- `diagnostic/text_export_manifest.json`: exact text-column contract, counts,
  digest, and provenance;
- `diagnostic/preprocess.json`: interpreted source-table and configuration
  summary;
- checkpoint or Slurm state intended for restart/assembly, not ordinary data
  analysis.

Use the text table for straightforward plotting when it contains every needed
field. Use `inverse.eosinv` or the manifest for exact statuses, retained
phase-root diagnostics, and fields omitted from text. Never parse the binary
artifact as an undocumented archive; load it through the public API:

```python
from eos_trensmuther import load_pipeline_result

result = load_pipeline_result("path/to/inverse.eosinv")
```

## Read the text table by declaration

Comment lines begin with `#` and are skipped normally by `numpy.loadtxt`. Read
the `Fields` line rather than assuming fixed positions. Public column order is:

```text
canonical requested targets
optional requested tilde coordinates
returned source coordinates
selected auxiliary fields
optional explicit phase fields
ExtraInfo fields
```

Names, units, dimension, grid space, shape, and C ordering are declared in the
header and JSON manifest. Do not assume two runs have identical auxiliary
fields or layouts.

## ExtraInfo flags

With `ExtraInfo=true`, the current deterministic fields are:

```text
projected
certified_unreachable
mixed_phase
phase_boundary
phase_multivalued
selected_from_jacobian
```

Interpret them as follows:

- `projected=1`: returned state is a nearest admissible projection, not an
  exact inverse of the requested target.
- `certified_unreachable=1`: the supported proof route certified no root for
  the requested target. A projected row may retain this provenance, so flags
  are not necessarily mutually exclusive.
- `mixed_phase=1`: selected returned state belongs to the declared mixed-phase
  route.
- `phase_boundary=1`: selected state lies on a declared phase boundary.
- `phase_multivalued=1`: the first-order solver verified another distinct
  physical root for the same target; the table contains the selected branch.
- `selected_from_jacobian=1`: multiple verified roots were reduced to one by
  the conservative local Jacobian-orientation rule. This implies explicit
  branch-selection provenance, not an ordinary unambiguous root.

An ordinary exact solved row needs no `solved` flag. Unresolved, invalid, and
error rows prevent generic text export, so `unresolved` and `error` are not text
columns. Their exact status remains available in `inverse.eosinv`.

Do not interpret `phase_multivalued` as an error or as proof that every possible
global branch was found. It records a distinct verified alternate root within
the supported first-order logic.

## Grid reconstruction

Rows use canonical C order: the last grid axis changes fastest. For a column
with one value per target, reshape using the declared grid shape:

```python
values_on_grid = flat_values.reshape(grid_shape, order="C")
```

Use coordinate values from the result/grid declaration, not an assumed
`linspace`, especially when comparing separate runs. For a one-dimensional
slice, hold the other requested grid indices fixed and state which values are
held constant.

The rectangular inverse grid can contain physically unreachable combinations.
Do not connect, interpolate across, or average projected/undefined points as if
they were exact roots unless the user explicitly requests that analysis.

## Auxiliary fields and units

Auxiliary quantities depend on table metadata. Use declared names, units, and
descriptions; do not infer scientific meaning from a suggestive label. `NaN`
can mean that a state or auxiliary value is unavailable under the declared
export policy. Consult the manifest and artifact status before dropping or
replacing it.

For tilde runs, use the transform definitions recorded in the header/manifest.
`Export.include_tilde=true` adds all four requested transformed-grid
coordinates as `GeV`-valued numeric columns. They remain requested coordinates
on projected rows, rather than coordinates recomputed from the returned
projected state. Do not reconstruct a transformed coordinate from memory or
assume it is an independent source-table column. Genuinely dimensionless
columns are labeled explicitly as `dimensionless`, not with the shorthand
unit `1`.

## Preserve provenance in analysis

When reporting or plotting results:

- distinguish exact and projected rows;
- preserve certified-unreachable, phase, and Jacobian-selection markers;
- report any filtering rule and counts before/after filtering;
- label synthetic/non-physical examples as such;
- use the manifest or artifact when the header lacks requested detail.

For first-order interpretation, also read
[`FirstOrderSkillsAI.md`](FirstOrderSkillsAI.md). For projection semantics, read
[`docs/PROJECTION.md`](../docs/PROJECTION.md).
