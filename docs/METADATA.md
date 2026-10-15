# Metadata reference

Each EoS table is described by one JSON file. Start with
[`metadata_regular.json`](../templates/metadata_regular.json) for an
ordinary table, the matching LINE/BINODALS starter in that directory for a
first-order table, or
[`metadata_master.json`](../templates/metadata_master.json) when you
want to see optional provenance, conversion, and grid controls.

The master is a copy-ready complete ordinary example next to a concise
[`field-by-field guide`](../templates/METADATA_REFERENCE.md). LINE and
BINODALS use separate starters because their fields are mutually exclusive.

The words below have precise practical meanings:

- **REQUIRED**: the table cannot be interpreted safely without it.
- **OPTIONAL — default: ...**: omit it when the default describes your table.
- **ADVANCED**: useful for unusual formats, stricter checks, or first-order
  data; normal users should leave it out unless it applies to their table.

## Smallest ordinary metadata file

```json
{
  "name": "My 2D EoS",
  "dimension": 2,
  "data_file": "table.dat",
  "fields": {
    "T":   {"role": "coordinate", "column": 0, "source_unit": "GeV",   "canonical_unit": "GeV"},
    "muB": {"role": "coordinate", "column": 1, "source_unit": "GeV",   "canonical_unit": "GeV"},
    "e":   {"role": "target",     "column": 2, "source_unit": "GeV^4", "canonical_unit": "GeV^4"},
    "nB":  {"role": "target",     "column": 3, "source_unit": "GeV^3", "canonical_unit": "GeV^3"}
  }
}
```

This uses a whitespace-delimited UTF-8 file, `#` comments, identity unit
conversion, linear interpolation, automatic row reordering, a complete
possibly nonuniform rectilinear grid, default zero omitted-charge constraints,
and no first-order transition. Scientific provenance is optional for loading
but strongly recommended for real scientific work; missing provenance emits
warnings rather than blocking the table.

## Required top-level fields

| Field | Meaning |
| --- | --- |
| `name` | Human-readable table name. |
| `dimension` | `1`, `2`, `3`, or `4`. This fixes the canonical source and target fields. |
| `data_file` | Numerical filename relative to the metadata file. Absolute paths are refused. |
| `fields` | Column meanings, units, and conversions for every quantity the program should retain. |

The canonical fields are:

| Dimension | Source coordinates | Inversion targets |
| ---: | --- | --- |
| 1 | `T` | `e` |
| 2 | `T`, `muB` | `e`, `nB` |
| 3 | `T`, `muB`, `muQ` | `e`, `nB`, `nQ` |
| 4 | `T`, `muB`, `muQ`, `muS` | `e`, `nB`, `nQ`, `nS` |

## Optional top-level fields

| Field | Status and default | Meaning |
| --- | --- | --- |
| `schema_version` | **OPTIONAL — default: `1`** | Metadata format version. Change only when a future schema requires it. |
| `table_revision` | **OPTIONAL — default: `null`** | Scientific/data revision label. Recommended for reproducibility. |
| `description` | **OPTIONAL — default: `null`** | Short explanation of the table. |
| `format` | **OPTIONAL — defaults below** | Text parsing rules. |
| `grid` | **OPTIONAL — complete rectilinear defaults below** | Structural validation and expected geometry. |
| `charge_constraints` | **OPTIONAL — dimensional zero-charge defaults** | Relations for omitted conserved-charge densities. |
| `scientific` | **OPTIONAL — undocumented/unknown warnings** | Provenance, validity, publications, and phase structure. Strongly recommended for physical tables. |

`validated` is deliberately not metadata. A table becomes validated only
after the loader checks the actual metadata and numeric file together.

## Field descriptors

Every `fields.NAME` object accepts:

| Key | Status and default | Meaning |
| --- | --- | --- |
| `role` | **REQUIRED** | `coordinate`, `target`, or `auxiliary`. Noncanonical quantities must be auxiliary. |
| `column` | **REQUIRED** | Zero-based numeric column index. Indices must be unique. |
| `source_unit` | **REQUIRED** | Unit/normalization as stored in the file. |
| `canonical_unit` | **REQUIRED** | Unit after conversion. Canonical `T`/chemical potentials use `GeV`, `e` uses `GeV^4`, and charge densities use `GeV^3`. |
| `transform` | **OPTIONAL — identity conversion** | Multiplicative reconstruction described below. |
| `nonfinite_policy` | **OPTIONAL — default: `"forbid"`** | `allow_nan` is permitted only for an auxiliary field. Infinity is always refused. |
| `output` | **OPTIONAL — default: `true`** | Include this auxiliary in the default result/output selection. |
| `interpolation` | **OPTIONAL — default: `"linear"`** | `linear` is implemented. Explicit `null` disables interpolation; `nearest` is reserved for a future backend. |
| `source_label` | **OPTIONAL — default: `null`** | Original file/header label. |
| `stored_as` | **OPTIONAL — default: `null`** | Human-readable representation such as `e/T^4`. |
| `description` | **OPTIONAL — default: `null`** | Scientific convention or caveat. |

Write `dimensionless` for quantities with no physical unit. Public output
headers and manifests use that same explicit spelling; the older shorthand
`1` is still accepted when reading existing metadata.

The transform is:

```text
canonical = raw * scale_factor * T_GeV^temperature_power
                * HBARC_GEV_FM^hbarc_power
```

Its keys are all optional: `scale_factor` defaults to `1`, and
`temperature_power` and `hbarc_power` default to `0`.

| Stored quantity | Typical transform | Canonical unit |
| --- | --- | --- |
| `T`, `mu` in MeV | `scale_factor: 0.001` | `GeV` |
| `e/T^4`, `P/T^4` | `temperature_power: 4` | `GeV^4` |
| `n/T^3`, `s/T^3` | `temperature_power: 3` | `GeV^3` |
| MeV/fm³ energy | `scale_factor: 0.001`, `hbarc_power: 3` | `GeV^4` |
| fm⁻³ density | `hbarc_power: 3` | `GeV^3` |

## `format`

| Key | Status and default | Meaning |
| --- | --- | --- |
| `delimiter` | **OPTIONAL — default: `"whitespace"`** | `"whitespace"` or one literal character such as `","`. |
| `comment_prefix` | **OPTIONAL — default: `"#"`** | Comment marker; `null` disables comments. |
| `skip_rows` | **OPTIONAL — default: `0`** | Initial physical lines to ignore. |
| `encoding` | **OPTIONAL — default: `"utf-8"`** | Text encoding known to Python. |
| `expected_columns` | **OPTIONAL — default: unchecked** | Exact source-column count. Recommended because unmapped columns are allowed. |
| `missing_values` | **ADVANCED — default: `[]`** | Tokens converted to `NaN`; requires `expected_columns`. Field policy still decides whether `NaN` is legal. |

## `grid`

Complete rectilinear Cartesian products are accepted. Axes may be nonuniform.

| Key | Status and default | Meaning |
| --- | --- | --- |
| `kind` | **OPTIONAL — default: `"rectilinear"`** | Currently supported grid kind. |
| `require_complete` | **OPTIONAL — default: `true`** | Must remain true. Holes are refused. |
| `require_uniform` | **OPTIONAL — default: `false`** | Require constant spacing on every source axis. |
| `reorder` | **OPTIONAL — default: `true`** | Reorder rows in memory into canonical Cartesian order. |
| `duplicate_policy` | **ADVANCED — default: `"error"`** | Currently supported duplicate policy. |
| `absolute_tolerance` | **ADVANCED — default: `0`** | Absolute coordinate clustering/comparison tolerance. |
| `relative_tolerance` | **ADVANCED — default: `0`** | Relative coordinate tolerance. |
| `min_axis_points` | **ADVANCED — default: `2`** | Minimum distinct nodes per source axis. |
| `expected_shape` | **OPTIONAL — default: unchecked** | One expected length per axis. Recommended to detect a wholly missing outer plane. |
| `expected_bounds` | **OPTIONAL — default: `{}`** | Expected canonical source bounds by axis name. |
| `expected_spacing` | **OPTIONAL — default: `{}`** | Expected positive spacing for axes known to be uniform. |

Canonical row order is ascending source-coordinate order with the last axis
changing fastest. `reorder` never fills holes, removes duplicates, or turns
scattered data into a rectilinear grid.

## `charge_constraints`

When omitted, missing densities are zero: dimension 1 uses `nB=nQ=nS=0`,
dimension 2 uses `nQ=nS=0`, and dimension 3 uses `nS=0`. Dimension 4 has no
omitted density.

For a relation such as strangeness neutrality and `nQ=0.4*nB`, use:

```json
"charge_constraints": {
  "model": "linear_homogeneous",
  "dependent_densities": {"nQ": {"nB": 0.4}, "nS": {}},
  "validation": {"absolute_tolerance": 1e-12, "relative_tolerance": 1e-8}
}
```

`model` and every omitted density relation are required when this optional
block is present. The validation tolerances are advanced and default to the
values shown.

## `scientific`

These fields do not change ordinary interpolation, but they are important for
scientific provenance. The complete supported structure is:

| Key | Status and default | Meaning |
| --- | --- | --- |
| `description` | **OPTIONAL — default: `null`** | Model and intended use. |
| `origin` | **OPTIONAL — default: `null` with a warning** | Object accepting `generator`, `converter`, `version`, and `url` strings, plus model-specific documentary keys. |
| `publications` | **OPTIONAL — default: `[]` with a warning** | Publication objects accepting `type`, `title`, `authors`, `doi`, `arxiv`, and `url`. At least one title or identifier is required per object. |
| `constraints` | **OPTIONAL — default: `[]`** | Human-readable physics constraints as strings or objects. |
| `validity` | **OPTIONAL — default: `null`** | Documentary object describing the domain/model validity. |
| `open_questions` | **OPTIONAL — default: `[]`** | Non-empty strings recording information still needing confirmation. |
| `phase_structure` | **OPTIONAL — default: undocumented with a warning** | Object containing the fields below. |
| `phase_structure.status` | **REQUIRED when `phase_structure` is present** | `none`, `unknown`, or `known`. |
| `phase_structure.critical_points` | **OPTIONAL — default: `[]`** | Named known, approximate, or not-provided critical points. |
| `phase_structure.first_order_transition` | **OPTIONAL — default: absent** | Use `{ "present": false }` to state explicitly that none is represented. |

Each critical point accepts `name`, `status`, `location`, and `note`. `name`
and `status` are required. A `known` or `approximate` point must give every
independent coordinate under `location`, with one `{ "value": ..., "unit":
"..." }` object per coordinate. A `not_provided` point may use a null
location.

Unknown origin/publications/phase information produces warnings, not a false
claim or an execution block.

### Advanced first-order metadata

A present transition is supported only in 2D. Its complete field set is:

| Key | Status and default | Meaning |
| --- | --- | --- |
| `present` | **REQUIRED** | Boolean. When false, no other transition key is allowed. |
| `representation` | **REQUIRED when present** | `line` or `binodals`. |
| `kind` | **REQUIRED when present** | Non-empty scientific label. |
| `phase_fraction_field` | **ADVANCED — default: `null`** | Name of an auxiliary field carrying the phase fraction. |
| `mixed_phase_condition` | **ADVANCED — default: `null`** | Documentary non-empty condition string. |
| `critical_point` | **REQUIRED for `line`; forbidden for `binodals`** | Name of one declared known/approximate critical point. |
| `critical_point_tolerance` | **REQUIRED for `line`** | Object containing non-negative `absolute` and `relative` values. |
| `coexistence_data` | **REQUIRED when present** | Sidecar table description below. |

`coexistence_data.file`, `expected_columns`, and `columns` are required.
`delimiter`, `comment_prefix`, `skip_rows`, and `encoding` have the same
defaults as the main table; documentary `fields` defaults to absent. LINE
`columns` must define `muB` and `T`; BINODALS must define `muB`, `T_dilute`,
and `T_dense`. Each column object requires a zero-based `index` and `unit`
(`GeV` or `MeV`).

For example, the shape of a LINE declaration is:

```json
{
  "scientific": {
    "phase_structure": {
      "status": "known",
      "critical_points": [{
        "name": "CP",
        "status": "known",
        "location": {
          "T": {"value": 0.15, "unit": "GeV"},
          "muB": {"value": 0.35, "unit": "GeV"}
        },
        "note": null
      }],
      "first_order_transition": {
        "present": true,
        "representation": "line",
        "kind": "model-specific coexistence line",
        "phase_fraction_field": null,
        "mixed_phase_condition": null,
        "critical_point": "CP",
        "critical_point_tolerance": {"absolute": 1e-12, "relative": 1e-8},
        "coexistence_data": {
          "file": "coexistence.dat",
          "delimiter": "whitespace",
          "comment_prefix": "#",
          "skip_rows": 0,
          "encoding": "utf-8",
          "expected_columns": 2,
          "fields": ["muB", "T"],
          "columns": {
            "muB": {"index": 0, "unit": "GeV"},
            "T": {"index": 1, "unit": "GeV"}
          }
        }
      }
    }
  }
}
```

This route uses dedicated phase geometry; see
[Methodology](METHODOLOGY.md#first-order-isolation).

The complete BINODALS declaration is shown in
[`metadata_first_order_binodals.json`](../templates/metadata_first_order_binodals.json).
It uses no named critical point and requires a retained auxiliary phase
fraction plus `muB`, `T_dilute`, and `T_dense` sidecar columns.

## Structural and local-geometry sanity checks

Validation exhaustively rejects nonfinite required fields, duplicate
coordinate tuples, holes, malformed Cartesian products, and disallowed row
ordering. In 1D and 2D it also inspects every interpolation cell. The 2D
classification uses the exact determinant range of the bilinear map, so a
cell is reported as positive, negative, sign-changing/singular, or invalid.
For an ordinary table, observing both positive and negative regular cells is
a validation error: the orientation reversal is substantial evidence of a
fold or non-injective forward map. The required orientation is not hard-coded;
a consistently positive or consistently negative table is accepted. A table
with declared first-order structure keeps the reversal as an explicit warning
and is handled by the dedicated phase-aware route. An isolated
sign-changing/singular cell remains a warning when no opposite regular
orientations are observed.

For 3D and 4D, the geometry diagnostic is intentionally bounded: it uses a
deterministic regular sample over the complete domain (at most 50,000 initial
cells by default), includes boundaries, and refines a one-cell halo around
suspicious samples (at most 10,000 additional cells). It reports orientation
transitions, singular/nonfinite Jacobians, and condition numbers. An observed
regular orientation reversal applies the same ordinary/first-order policy
above. Because the scan is bounded, passing it is **not a proof of global
injectivity**.

## Validate a table

After installation:

```python
from eos_trensmuther import ForwardTable

table = ForwardTable("path/to/metadata.json")
report = table.validate()
for issue in report.issues:
    print(issue.severity, issue.code, issue.path, issue.message)
if report.is_valid:
    print(table.shape, table.find_boundaries(space="target").to_dict())
```

Warnings preserve uncertainty; errors prevent inversion. Successful loading
does not prove that the forward map is globally one-to-one or that every point
inside its rectangular target envelope is physically reachable.
