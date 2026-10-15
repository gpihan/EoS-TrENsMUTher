# Output and generic text format

## Authoritative result

`inverse.eosinv` is the authoritative completed result. It stores the target
grid, returned source coordinates, auxiliary values, masks, status/proof
codes, projection information, phase information, and a checksummed manifest.
Load it with `load_pipeline_result()`.

## Generic text export

The public text exporter writes columns in this order:

```text
canonical requested targets
optional requested tilde coordinates
returned source coordinates
selected auxiliary fields
optional first-order phase columns
ExtraInfo state/provenance columns (enabled by default)
```

Within each group, names follow the result plan. Rows follow canonical C grid
order: the last grid axis changes fastest.

By default, the file begins with only the structural information needed to
read it: the EoS-TrENsMUTher version, field names and units, transformed-field
definitions when applicable, and the grid shape/order.

`numpy.loadtxt`, and other readers with `#` comments enabled, skip these lines
normally. Example:

```text
# EoS-TrENsMUTher version 1.0.0.dev0
# Fields: e[GeV^4] nB[GeV^3] T[GeV] muB[GeV] ... projected[dimensionless] ...
# Grid shape: 5 x 5; C order; last axis changes fastest
```

The optional JSON sidecar records the same column contract, file SHA-256,
counts, source result schema, table fingerprint, and projection/undefined
value policies.

For a transformed-grid run, `Export.include_tilde=true` adds the energy-valued
columns `T_tilde`, `muB_tilde`, `muQ_tilde`, and `muS_tilde`, all in `GeV`, after the
canonical targets. All four are written even when only a lower-dimensional
independent subset defines the grid; metadata charge constraints determine
the dependent chemical tilde values. These columns describe the requested
grid point. They are therefore intentionally not replaced by the achieved
state on projected rows.

The header does not use an opaque matrix placeholder. It prints the reduced
numerical equations actually used for the table, for example for
`nQ=0.4*nB, nS=0`:

```text
# Tilde: T_tilde = (e / (19*pi^2/12))^(1/4)
# Tilde density constraints: nQ = 0.4*nB; nS = 0
# Tilde: muB_tilde = (4.6*nB) / T_tilde^2
# Tilde: muQ_tilde = (-0.2*nB) / T_tilde^2
# Tilde: muS_tilde = (1.6*nB) / T_tilde^2
```

The JSON sidecar stores the same formulas and their machine-readable numerical
coefficients under `tilde_transform`.

When a first-order root was recovered from a local finite-resolution fold,
the authoritative result stores `phase_selection_reason_codes`; its codebook
contains `SELECTED_FROM_JACOBIAN`, and
`PipelineResult.jacobian_selected_mask` selects those rows. The run summary
also records `phase_diagnostics.jacobian_selected_roots`.

With `ExtraInfo=true`, the numeric `selected_from_jacobian` column is one on
those rows. Target-specific diagnostics do not lengthen the normal header.
The JSON sidecar retains the detailed flat indices under
`root_selection_provenance`, and the authoritative artifact retains the
selection-reason code.

## ExtraInfo columns

The default deterministic order is:

```text
projected certified_unreachable mixed_phase
phase_boundary phase_multivalued selected_from_jacobian
```

These are dimensionless `0/1` flags and are labeled `[dimensionless]` in the
header and JSON sidecar. They follow every physical, auxiliary, and
explicit phase column. Overlapping explicit phase columns are not duplicated.
Set `ExtraInfo=false` to omit the group; this changes only the text view.

An ordinary solved row needs no flag: it is the normal case. Unresolved,
invalid, and error rows are never written to a generic text table; any such
row causes export to fail, while the authoritative `.eosinv` artifact retains
the detailed diagnostic state. Consequently, the redundant `solved` column
and the inapplicable `unresolved` and `error` columns were removed in
text-export schema 4. Schema 5 added optional requested tilde columns and a
machine-readable explicit tilde-transform description. Schema 6 corrects the
tilde units to `GeV` and standardizes public dimensionless-unit labels as
`dimensionless` rather than the terse token `1`.

`phase_multivalued=1` means the first-order solver verified at least one
additional, distinct physical root for the same target. The exported source
coordinate is the selected branch. When `selected_from_jacobian=1`, that
branch was selected by the conservative local Jacobian-orientation rule; in
other multivalued cases, the retained alternate branch is recorded in the
authoritative artifact. This flag does not by itself mean that the exported
row is invalid.

## Projected and unusable rows

Projected states are included automatically when projection produced them.
They are nearest feasible states found by the configured projection
algorithm, not roots of the requested targets, and are identified by
`projected=1` with the default ExtraInfo layout.

Certified-unreachable rows are disabled in export unless
`Export.include_unusable` is true. When included, requested target columns
remain finite and unavailable state/auxiliary columns are `nan`. Unresolved,
invalid, failed, or error rows always prevent creation of the generic text
table. Use `inverse.eosinv` for exact point-level status and proof
interpretation.

## Header and precision

`Export.write_header` defaults to true and should normally remain enabled.
`Export.float_format` controls decimal text precision; the default is
`"%.8e"`. Text output can lose precision and status detail, so it does not
replace the binary artifact for restart, proof auditing, or exact round trips.
