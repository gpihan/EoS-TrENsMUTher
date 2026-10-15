# Parameter-file reference

The command-line interface reads a trusted Python file containing one
`general_parameters` dictionary. Because this is executable Python, do not
run a parameter file from an untrusted source.

The complete copyable reference is
[`templates/parameters.master.py`](../templates/parameters.master.py). It
shows every supported public setting, its default, and whether normal users
should normally leave it unchanged. The files in
[`examples/`](../examples/) are smaller runnable configurations.

## Required for a run

| Parameter | Meaning |
| --- | --- |
| `EoS_metadata` | Path to validated table metadata. |
| `GridAxes` | One count or bounded count for every axis in the selected grid space. |

`GridAxes` may use automatic boundaries:

```python
"GridAxes": {"e": 40, "nB": 40},
```

or explicit bounds:

```python
"GridAxes": {
    "e": {"count": 40, "lower": 0.01, "upper": 1.0},
    "nB": {"count": 40, "lower": -0.1, "upper": 0.1},
},
```

Automatic bounds are component-wise extrema of validated table nodes. Their
Cartesian product is a sampling envelope and can contain unreachable points;
the reachability/proof path handles them.

Regular `(e,nB,...)` variables are the default, so they require no mode
parameter. For transformed variables add:

```python
"Variables": "tilde",
"GridAxes": {"T": 40, "muB": 40},
```

The program applies the existing constraint-aware transformation to every
dimension and maps those convenient axis names to `T_tilde`, `muB_tilde`,
and so on. Explicit tilde names remain accepted. The older `GridSpace`
setting is compatibility input, not the preferred public form.

## Main optional groups

- `OutputFolder`, `EoSName`, `RunSpecification`, and `Overwrite` choose the
  run directory without changing the science.
- `AuxiliaryFields` chooses table quantities interpolated at returned states.
- `Execution` selects serial or local process execution.
- `Checkpoint` enables atomic restart state.
- `Projection` controls separately labeled nearest reachable states.
- `Export` controls the generic text table and self-describing header.
- `ExtraInfo` controls appended public `0/1` status/provenance columns.

## Defaults worth knowing

| Setting | Default |
| --- | --- |
| Coordinate variables | regular targets `(e,nB,...)`; add `Variables="tilde"` to transform |
| `OutputFolder` | `"Output"` beside the parameter file |
| `Execution.workers` | `1` |
| `Execution.worker_task_size` | `8` |
| `Checkpoint.enabled` / `.resume` | `true` / `true` |
| `Projection.enabled` | `true` |
| `Export.enabled` / `.write_header` | `true` / `true` |
| `Export.filename` | `"eos_table.dat"` |
| `Export.include_tilde` | `false`; for a tilde grid, add all four requested tilde columns |
| `ExtraInfo` | `true` |
| `ResidualScales` | automatic from validated table bounds |
| `Solver.root_residual_tolerance` | `1e-10` in the configured scaled residual |
| Ordinary subcell fallback | enabled, maximum depth `7`, one-million active-box guard |

Projection is a different scientific statement from inversion. Disable it if
you want only exact roots and certified/inconclusive statuses. When enabled,
projected states are included in text output automatically and tagged by the
`projected` ExtraInfo column. `Export.include_projected` is accepted only as
a compatibility spelling and no longer controls public behavior.

`ExtraInfo=true` (the default) appends these columns, in order, after physical,
auxiliary, and explicitly requested phase columns:

```text
projected certified_unreachable mixed_phase
phase_boundary phase_multivalued selected_from_jacobian
```

Every value is `0` or `1`. If an explicitly requested phase column already
has the same name, it appears only once. Set `ExtraInfo=false` only when the
consumer requires a physical-columns-only numeric layout; the `.eosinv`
artifact always retains the full state and provenance.

Ordinary solved rows need no separate flag. Unresolved, invalid, and error
rows are rejected rather than written as generic text.
`phase_multivalued=1` records that the first-order solver verified another
distinct physical root for the same target; the exported source coordinate is
the selected branch. See
[Output and generic text format](OUTPUT_FORMAT.md) for the detailed
interpretation.

For a run with `Variables="tilde"`, set

```python
"Export": {"include_tilde": True}
```

to append `T_tilde`, `muB_tilde`, `muQ_tilde`, and `muS_tilde` immediately
after the canonical target columns. These are the requested transformed-grid
coordinates. On a projected row they describe the requested point, while the
returned `(T,muB,...)` state is the projected admissible point; use the
`projected` flag to distinguish that case. The text header and JSON manifest
record the numerical transform after applying the metadata charge constraints.

## Residual normalization

`ResidualScales` is **optional and advanced**. It is a numerical
normalization used when comparing how accurately the forward EoS reproduces
target quantities with different units and magnitudes; it is not a physical
EoS parameter.

When the setting is omitted, `None`, or `"automatic"`, EoS-TrENsMUTher uses
the maximum absolute validated table-node value of each target component. The
same table therefore always gives the same scales, independently of the
requested target. An exactly zero component uses `1.0` in its canonical unit
so division by zero is impossible. Most users should leave this automatic.

For a controlled numerical study, an explicit mapping remains available:

```python
"ResidualScales": {"e": 0.02, "nB": 0.02},
```

Every canonical target must be present, and every supplied value must be
finite, positive, and safely invertible. Explicit values are used exactly.
The resolved values are stored in preprocessing, result, checkpoint, and run
identity information for reproducibility.

## Advanced settings

`Solver`, `Reachability`, `CoarseSeed`, memory guards, and first-order
auxiliary policies are public for reproducibility, but their defaults are the
validated configuration. Change them only with a reason you can record in
your scientific workflow. Resource limits yield an honest `UNRESOLVED`
outcome when exhausted; they do not weaken proof rules.

The three ordinary-subcell controls are:

```text
Solver.enable_subcell_fallback
Solver.max_subcell_depth
Solver.max_subcell_active_boxes
```

See [Methodology](METHODOLOGY.md) before changing a numerical tolerance and
[Output format](OUTPUT_FORMAT.md) before treating the text table as an
authoritative result.
