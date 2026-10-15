# Master metadata reference

Use [`metadata_master.json`](metadata_master.json) as the complete ordinary
table example. It is valid JSON, so it intentionally contains no comments.
This companion explains every supported input. Replace example values and
remove optional sections that do not apply.

For the smallest usable file, copy [`metadata_minimal.json`](metadata_minimal.json).
For a first-order table, use
[`metadata_first_order_line.json`](metadata_first_order_line.json) or
[`metadata_first_order_binodals.json`](metadata_first_order_binodals.json).
Those alternatives are mutually exclusive and therefore cannot both appear
in one valid master JSON.

The longer scientific explanations and examples are in
[`docs/METADATA.md`](../docs/METADATA.md).

The labels below mean:

- **REQUIRED**: needed to interpret the table safely.
- **OPTIONAL — default: ...**: omit when the default is correct.
- **ADVANCED**: normally leave unchanged.

## Top level

| Input | Requirement | Meaning |
| --- | --- | --- |
| `name` | **REQUIRED** | Human-readable table name. |
| `dimension` | **REQUIRED** | Number of independent coordinates, from 1 through 4. |
| `data_file` | **REQUIRED** | Numerical table filename, relative to this JSON file. |
| `fields` | **REQUIRED** | Meaning, column, units, and conversion of every retained quantity. |
| `schema_version` | **OPTIONAL — default: `1`** | Metadata format version. |
| `table_revision` | **OPTIONAL — default: `null`** | Scientific/data revision identifier. |
| `description` | **OPTIONAL — default: `null`** | Short table description. |
| `format` | **OPTIONAL — defaults below** | Text-file parsing rules. |
| `grid` | **OPTIONAL — defaults below** | Rectilinear-grid validation rules. |
| `charge_constraints` | **OPTIONAL — omitted charges default to zero** | Relations for charge densities absent from the independent coordinates. |
| `scientific` | **OPTIONAL, strongly recommended** | Provenance, validity, publications, and phase information. |

Canonical coordinate/target names are `T -> e`, then successively
`muB -> nB`, `muQ -> nQ`, and `muS -> nS` as the dimension increases.

## Each `fields.NAME` object

| Input | Requirement | Meaning |
| --- | --- | --- |
| `role` | **REQUIRED** | `coordinate`, `target`, or `auxiliary`. |
| `column` | **REQUIRED** | Zero-based source-file column index. |
| `source_unit` | **REQUIRED** | Unit or normalization stored in the file. |
| `canonical_unit` | **REQUIRED** | Unit after conversion: normally `GeV`, `GeV^4`, or `GeV^3`. |
| `transform` | **OPTIONAL — identity** | Multiplicative reconstruction described below. |
| `nonfinite_policy` | **OPTIONAL — default: `"forbid"`** | Only auxiliary fields may use `"allow_nan"`; infinity is always invalid. |
| `output` | **OPTIONAL — default: `true`** | Include an auxiliary field in normal output. |
| `interpolation` | **OPTIONAL — default: `"linear"`** | `"linear"` is supported; `null` disables interpolation. |
| `source_label` | **OPTIONAL — default: `null`** | Original header or publication label. |
| `stored_as` | **OPTIONAL — default: `null`** | Representation such as `e/T^4`. |
| `description` | **OPTIONAL — default: `null`** | Scientific meaning or convention. |

Use `dimensionless` for a quantity with no physical unit. The shorthand `1`
is accepted for existing metadata, but new files and public exports use the
explicit spelling consistently.

`transform` computes

```text
canonical = raw * scale_factor * T_GeV^temperature_power
                * HBARC_GEV_FM^hbarc_power
```

Its inputs are `scale_factor` (**default `1`**), `temperature_power`
(**default `0`**), and `hbarc_power` (**default `0`**).

## `format`

| Input | Requirement | Meaning |
| --- | --- | --- |
| `delimiter` | **OPTIONAL — default: `"whitespace"`** | Whitespace or one literal character such as `","`. |
| `comment_prefix` | **OPTIONAL — default: `"#"`** | Comment marker; `null` disables comments. |
| `skip_rows` | **OPTIONAL — default: `0`** | Initial physical lines to ignore. |
| `encoding` | **OPTIONAL — default: `"utf-8"`** | File encoding. |
| `expected_columns` | **OPTIONAL — default: unchecked** | Exact source column count. |
| `missing_values` | **ADVANCED — default: `[]`** | Tokens converted to `NaN`; requires `expected_columns`. |

## `grid`

| Input | Requirement | Meaning |
| --- | --- | --- |
| `kind` | **OPTIONAL — default: `"rectilinear"`** | Supported grid type. |
| `require_complete` | **OPTIONAL — default: `true`** | Must remain true; holes are unsupported. |
| `require_uniform` | **OPTIONAL — default: `false`** | Require equal spacing on every axis. |
| `reorder` | **OPTIONAL — default: `true`** | Reorder rows into canonical Cartesian order. |
| `duplicate_policy` | **ADVANCED — default: `"error"`** | Duplicate coordinate policy. |
| `absolute_tolerance` | **ADVANCED — default: `0`** | Absolute coordinate clustering tolerance. |
| `relative_tolerance` | **ADVANCED — default: `0`** | Relative coordinate clustering tolerance. |
| `min_axis_points` | **ADVANCED — default: `2`** | Minimum distinct nodes per coordinate. |
| `expected_shape` | **OPTIONAL — default: unchecked** | Expected axis lengths in canonical coordinate order. |
| `expected_bounds` | **OPTIONAL — default: `{}`** | Canonical lower/upper bounds keyed by coordinate. |
| `expected_spacing` | **OPTIONAL — default: `{}`** | Expected positive spacing keyed by coordinate. |

## `charge_constraints`

| Input | Requirement | Meaning |
| --- | --- | --- |
| `model` | **REQUIRED when block is present** | Must be `"linear_homogeneous"`. |
| `dependent_densities` | **REQUIRED when block is present** | Coefficients expressing every omitted density through retained densities. `{}` means zero. |
| `validation.absolute_tolerance` | **ADVANCED — default: `1e-12`** | Absolute check for a supplied dependent-density column. |
| `validation.relative_tolerance` | **ADVANCED — default: `1e-8`** | Relative check for a supplied dependent-density column. |

## `scientific`

| Input | Requirement | Meaning |
| --- | --- | --- |
| `description` | **OPTIONAL — default: `null`** | Model and intended use. |
| `origin` | **OPTIONAL — default: unknown warning** | Provenance object. Supported documentary keys include `generator`, `converter`, `version`, and `url`. |
| `publications` | **OPTIONAL — default: `[]` warning** | Entries may contain `type`, `title`, `authors`, `doi`, `arxiv`, and `url`. |
| `constraints` | **OPTIONAL — default: `[]`** | Human-readable physics constraints. |
| `validity` | **OPTIONAL — default: `null`** | Documentary validity domain and caveats. |
| `phase_structure` | **OPTIONAL — default: unknown warning** | Critical points and optional first-order representation. |
| `open_questions` | **OPTIONAL — default: `[]`** | Unresolved provenance or physics questions. |

Each `phase_structure.critical_points` entry accepts `name` and `status`
(required), plus `location` and `note`. `status` is `known`, `approximate`, or
`not_provided`. A known/approximate location contains each independent
coordinate as `{ "value": number, "unit": "GeV" or "MeV" }`.

`phase_structure.status` is `none`, `unknown`, or `known`.
`phase_structure.first_order_transition.present` is required when that object
is supplied. Use `false` for no represented transition.

## First-order-only inputs

When `present` is true, these inputs are supported only for a 2D table:

| Input | Requirement | Meaning |
| --- | --- | --- |
| `representation` | **REQUIRED** | `"line"` or `"binodals"`. |
| `kind` | **REQUIRED** | Scientific label for the transition. |
| `phase_fraction_field` | **ADVANCED — default: `null`** | Auxiliary field holding phase fraction. |
| `mixed_phase_condition` | **ADVANCED — default: `null`** | Documentary mixed-phase condition. |
| `critical_point` | **REQUIRED for LINE; forbidden for BINODALS** | Name of a declared critical point. |
| `critical_point_tolerance.absolute` | **REQUIRED for LINE** | Non-negative absolute matching tolerance. |
| `critical_point_tolerance.relative` | **REQUIRED for LINE** | Non-negative relative matching tolerance. |
| `coexistence_data` | **REQUIRED** | Sidecar-table parser and columns. |

`coexistence_data` supports `file`, `delimiter`, `comment_prefix`, `skip_rows`,
`encoding`, `expected_columns`, optional documentary `fields`, and `columns`.
Every `columns.NAME` object requires `index` and `unit`. LINE requires `muB`
and `T`; BINODALS requires `muB`, `T_dilute`, and `T_dense`.

Do not add a `validated` flag. Validation is computed from the JSON and the
actual numeric files each time the table is loaded.
