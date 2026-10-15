# Public API and configuration contract

This document freezes the supported Python and configuration surface.
New user code should import the names below from `eos_trensmuther`, not from
implementation submodules. The package root continues to expose older names
for compatibility, but those additional names are not part of the small
supported interface.

The contract is machine-readable:

```python
import eos_trensmuther

print(eos_trensmuther.__version__)
print(eos_trensmuther.PUBLIC_API_VERSION)
print(eos_trensmuther.PUBLIC_API_GROUPS)
print(eos_trensmuther.V1_CONTRACT_VERSIONS)
```

The current development version is `1.0.0.dev0`. Its single source is
`src/eos_trensmuther/_version.py`; package metadata reads that value rather than
duplicating it. `PUBLIC_API_VERSION` is `7`. Version 2 replaced hydro-code
specific exporter names with one generic text-export surface. Version 3 makes
residual normalization automatic by default while preserving the explicit
`PipelineConfig.residual_scales` override. Version 4 adds the public
`PhaseRootSelectionReason` needed to interpret conservative local-Jacobian
recovery of verified first-order roots. The corresponding result,
persistence, routing, and text-export contracts were bumped honestly.
Version 5 adds the public `Variables` and `ExtraInfo` configuration surface,
makes regular targets the default, always includes/tag projected text rows,
and gives the generic text layout a deterministic ExtraInfo group. Text-export
schema 4 removes the redundant `solved` column and the inapplicable
`unresolved` and `error` columns; none of these changes affects the
authoritative result artifact. Version 6 adds the public
`Export.include_tilde` option. Text-export schema 5 records its optional
requested tilde columns and the explicit constraint-reduced numerical
transformation in the header and JSON manifest; schema 6 corrects their units
to `GeV` and standardizes `dimensionless` labels. Version 7 renames the public
Python package, console command, and source-tree launcher to
`eos_trensmuther`, `eos-trensmuther`, and `EoS-TrENsMUTher.py` respectively.

## Supported top-level names

### Contract introspection

```text
__version__
PUBLIC_API_VERSION
PUBLIC_API_GROUPS
PUBLIC_API_NAMES
V1_CONTRACT_VERSIONS
```

### Tables and validation

```text
ForwardTable
BoundaryEnvelope
ValidationIssue
ValidationReport
EoSError
EoSValidationError
```

### Configuration

```text
ApplicationConfig
ApplicationConfigError
GridAxisSpec
InversionGridSpec
PipelineConfig
ExecutionConfig
ProjectionConfig
SolverConfig
ReachabilityConfig
CheckpointConfig
CoarseSeedConfig
ErrorPolicy
ProcessStartMethod
PARAMETER_GROUPS
ORDINARY_SUBCELL_CONFIG_FIELDS
load_application_config
load_parameter_file
```

### Pipeline and status interpretation

```text
EoSInversionPipeline
PipelineResult
PipelineError
PipelineConfigurationError
PointDisposition
PointRoute
PrescreenStatus
SolveStatus
ProofStatus
ProjectionQuality
PhaseLabel
PhaseRoute
PhaseRootSelectionReason
InitialGuessSource
pipeline_status_code
```

### Persistence and exporters

```text
save_pipeline_result
load_pipeline_result
PersistenceError
TextExportConfig
TextExportResult
ExportError
export_text
```

### Advanced direct inversion

```text
MultilinearInterpolator
EoSRootSolver
InversionResult
```

### Slurm arrays

```text
SlurmArrayConfig
ExecutorError
create_array_plan
save_array_plan
load_array_plan
render_slurm_script
execute_array_shard
assemble_array_shards
```

`PUBLIC_API_NAMES` is the exact flattened tuple above. `PUBLIC_API_GROUPS` is
a read-only mapping of the same groups. Names in `COMPATIBILITY_API_NAMES`
remain importable from the package root for compatibility, and the broad
historical `__all__` behavior is preserved, but new code should not depend on
those names. Modules whose names begin with `_` are private. Other package
submodules are implementation locations rather than an alternative public
surface, except for the declared console entry point
`eos_trensmuther.cli:main`.

## Typed Python workflow

The supported typed route loads metadata, builds an immutable configuration,
and runs the pipeline:

```python
from eos_trensmuther import (
    CheckpointConfig,
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    PipelineConfig,
    ProjectionConfig,
    load_pipeline_result,
    save_pipeline_result,
)

table = ForwardTable.load_validated("examples/regular/metadata.json")
config = PipelineConfig(
    grid=InversionGridSpec(
        "target",
        {"e": GridAxisSpec(5), "nB": GridAxisSpec(5)},
    ),
    projection=ProjectionConfig(enabled=True),
    execution=ExecutionConfig(
        workers=2,
        chunk_size=5,
        worker_task_size=2,
        max_pending_chunks=4,
        start_method="spawn",
    ),
    auxiliary_fields=("P", "s"),
)

result = EoSInversionPipeline(table, config).run(
    checkpoint=CheckpointConfig("Output/checkpoint")
)
save_pipeline_result(result, "Output/inverse.eosinv")
restored = load_pipeline_result("Output/inverse.eosinv")
```

Set `workers=1` for direct serial execution. Worker count, task size, pending
future bound, and process start method are execution settings and may change on
a compatible checkpoint resume. Atomic `chunk_size` defines checkpoint layout
and cannot change for that checkpoint.

## Interpreting results

`PipelineResult.status_counts` summarizes terminal dispositions. The following
row-level masks are the primary interpretation interface:

```text
solution_mask                 accepted physical roots
certified_unreachable_mask    proven no root in the supported domain
unknown_mask                  honest UNRESOLVED outcome
projected_mask                optional nearest admissible state
projection_failed_mask        requested projection did not complete
invalid_target_mask           invalid requested target
error_mask                    point-level execution error
```

The numeric arrays remain available for downstream storage, while
`result.codebooks` maps their integer values to declared meanings. Enum values
such as `PointDisposition`, `SolveStatus`, `ProofStatus`, and
`ProjectionQuality` plus `pipeline_status_code()` allow explicit comparisons.
Projection does not turn an unreachable target into a root: a projected row
retains its distinct disposition and provenance.

`result.jacobian_selected_mask` identifies solved rows selected from multiple
verified first-order roots by unanimous local Jacobian consistency. Their
`phase_selection_reason_codes` decode to `SELECTED_FROM_JACOBIAN`; ordinary
unambiguous roots remain distinguishable.

## Parameter-file configuration groups

`load_application_config()` remains the supported adapter for trusted Python
files containing `general_parameters`. `PARAMETER_GROUPS` is a read-only map
with four disjoint groups:

- `user`: normal input, grid, residual scale, projection, execution,
  checkpoint, `Variables`, `ExtraInfo`, generic export, output-root, and
  overwrite choices;
- `advanced`: solver/seeding/reachability, resource ceilings, auxiliary
  policies and artifact compression;
- `diagnostic`: preprocessing and artifact filenames; and
- `compatibility`: documented aliases and deliberately accepted legacy-only
  values.

Unknown top-level and nested keys fail with their configuration scope and a
close-name suggestion when one exists. Compatibility aliases keep their declared
behavior; ignored legacy-only values are reported in `ApplicationConfig` and
by the CLI.

The only public ordinary-subcell controls are exactly:

```text
Solver.enable_subcell_fallback
Solver.max_subcell_depth
Solver.max_subcell_active_boxes
```

They are also exposed as `ORDINARY_SUBCELL_CONFIG_FIELDS`. Other solver fields
are advanced numerical controls retained for existing configurations; no
additional subcell implementation knobs are public.

## Version registry

`V1_CONTRACT_VERSIONS` records the accepted versions without changing them:

| Contract | Version |
| --- | ---: |
| Public API | 7 |
| Metadata schema | 1 |
| Table snapshot fingerprint | 2 |
| Dense pipeline result | 4 |
| Run manifest | 11 |
| Numerical routing | 3 |
| Checkpoint | 4 |
| Completed artifact | 3 |
| Reachability routing | 1 |
| Shared worker state | 1 |
| Slurm array plan / shard | 3 / 3 |
| Generic text / phase export | 6 / 1 |

These values describe persisted or numerical compatibility boundaries. The
Text-export schema 5 recorded the optional requested tilde columns and their
explicit numerical transform. Schema 6 corrects those column units to `GeV`
and writes all genuinely dimensionless public column units explicitly as
`dimensionless`. No numerical routing, result, artifact,
checkpoint, or Slurm schema changed. Root verification,
residual tolerance, proof, projection, ordinary-EoS, and first-order
membership semantics did not change.

## CLI errors and version

The supported console program name is `eos-trensmuther`; the source-tree root
`EoS-TrENsMUTher.py` launcher provides the same commands without installation.
Configuration/runtime failures use
the concise form:

```text
eos-trensmuther: error: <actionable message>
```

The version is available without loading a table:

```bash
eos-trensmuther --version
```
