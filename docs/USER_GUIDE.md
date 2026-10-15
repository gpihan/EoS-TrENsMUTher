# User guide

EoS-TrENsMUTher validates a forward equation-of-state table, builds a regular
target or tilde grid, computes inverses where they exist, proves no-root
outcomes where possible, optionally projects unreachable targets, and saves a
restartable result.

## Installation

Python 3.9–3.13 is supported:

```bash
python -m pip install .
eos-trensmuther --help
eos-trensmuther --version
```

## Input table

Every table needs JSON metadata and a numeric text file. The truly required
metadata are the table name, dimension, data filename, and each retained
field's role, zero-based column, source unit, and canonical unit. Parsing,
identity transformations, complete rectilinear-grid behavior, and omitted
zero-charge constraints have documented defaults.

Start from [`metadata_minimal.json`](../templates/metadata_minimal.json) and
read the [metadata reference](METADATA.md). The loader accepts complete rectilinear
1D–4D grids; source axes may be nonuniform. Auxiliary quantities such as
pressure and entropy are interpolated at returned states.

## Parameter file

The CLI reads a trusted Python `general_parameters` dictionary. The three
scientifically essential run settings are:

```python
general_parameters = {
    "EoS_metadata": "metadata.json",
    "GridAxes": {"e": 40, "nB": 40},
}
```

Regular target variables are the default. To request the constraint-aware
transformed grid, add only `"Variables": "tilde"`; `GridAxes` may still use
the familiar source names (`T`, `muB`, ...) or the generated tilde names.

An integer axis count uses automatic metadata-derived bounds. Explicit
`count`, `lower`, and `upper` values select a smaller or scientifically chosen
domain. Residual normalization is also automatic: it is derived from the
validated table and is not a physical parameter users normally need to
choose. The full public configuration is documented in
[PARAMETERS.md](PARAMETERS.md) and
[`parameters.master.py`](../templates/parameters.master.py).

## Run locally

Serial:

```bash
eos-trensmuther run examples/regular/parameters_serial.py
```

Local processes:

```bash
eos-trensmuther run examples/regular/parameters_multicore.py
```

`Execution.workers=1` runs directly. Larger values use persistent worker
processes and shared read-only table/reachability state. `chunk_size` is the
atomic checkpoint range; `worker_task_size` is the smaller scheduled task.
Complete checkpoint chunks are reused after interruption, while unpublished
partial work is safely recomputed.

## Slurm

Planning never submits a job:

```bash
eos-trensmuther slurm-plan examples/regular/parameters_slurm.py \
  --partition cpu --wall-time 00:10:00 --cpus-per-task 1
```

Inspect and submit the generated `submit_array.sbatch`. After all tasks finish:

```bash
eos-trensmuther slurm-assemble --plan PATH/array_plan.json
eos-trensmuther export-text examples/regular/parameters_slurm.py
```

The table, environment, plan, mmap state, checkpoints, shards, and output must
be visible at the same absolute paths from all nodes. See the
[complete regular workflow](../examples/regular/README.md).

## CLI stages

```text
preprocess     validate and summarize the input
invert         create inverse.eosinv
export-text    create the generic text table from a completed artifact
run            perform all three stages
slurm-plan     create a first-axis array plan and sbatch script
slurm-task     execute one array shard
slurm-assemble validate and assemble all shards
```

`postprocess` remains a compatibility alias for `export-text`.

## Outputs

A normal run contains:

- `diagnostic/preprocess.json`: validated input and resolved configuration;
- `diagnostic/checkpoint/`: atomic restart state;
- `inverse.eosinv`: authoritative checksummed result;
- `eos_table.dat`: optional generic tabular view;
- `diagnostic/text_export_manifest.json`: export digest, columns, and policy.

The text file has compact `#` comment lines explaining version, column
names/units, and grid order. Projected and phase/provenance states are
machine-readable `0/1` columns when `ExtraInfo` is left at its default `true`.
See
[OUTPUT_FORMAT.md](OUTPUT_FORMAT.md).

## Scientific statuses

| Status | Meaning |
| --- | --- |
| `SOLVED` | Accepted in-domain root below the configured scaled residual tolerance. |
| `CERTIFIED_UNREACHABLE` | Complete prescreen or fallback proof established no root in the supported domain. |
| `UNRESOLVED` | Search/proof was inconclusive or reached a resource guard. |
| `PROJECTED` | Optional feasible approximation; not a root of the requested target. |
| `INVALID_TARGET` | Target or transform was invalid. |
| `PROJECTION_FAILED` | Requested projection did not return a usable state. |
| `ERROR` | Unexpected point-level failure under accumulate mode. |

A projected row retains its certified-no-root provenance, so status masks are
scientific facts rather than mutually exclusive file-format categories.

## Python API

```python
from eos_trensmuther import (
    EoSInversionPipeline, ExecutionConfig, ForwardTable, GridAxisSpec,
    InversionGridSpec, PipelineConfig, ProjectionConfig,
    load_pipeline_result, save_pipeline_result,
)

table = ForwardTable.load_validated("metadata.json")
config = PipelineConfig(
    grid=InversionGridSpec("target", {"e": GridAxisSpec(20), "nB": GridAxisSpec(20)}),
    projection=ProjectionConfig(enabled=False),
    execution=ExecutionConfig(workers=1),
)
result = EoSInversionPipeline(table, config).run()
save_pipeline_result(result, "inverse.eosinv")
restored = load_pipeline_result("inverse.eosinv")
```

Use an `if __name__ == "__main__":` guard around process-running Python
programs on platforms that use process spawning.

## First-order and external data

The 2D LINE/BINODALS/Region-S route uses dedicated phase geometry and is
isolated from the ordinary subcell route. It represents the declared
first-order construction; it is not a general higher-dimensional multibranch
phase-transition method. Physical EoS tables and their provenance remain
external scientific inputs; do not copy large production datasets into the
package merely to run the bounded example or CI.

Read [METHODOLOGY.md](METHODOLOGY.md),
[PROJECTION.md](PROJECTION.md), and
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) before scientific use.
