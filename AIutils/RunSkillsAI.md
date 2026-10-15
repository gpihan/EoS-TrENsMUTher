# Running EoS-TrENsMUTher with an AI assistant

Use this resource to prepare a supported run, choose its execution mode, and
report its outcome. Do not reconstruct missing scientific metadata here; use
[`MDSkillsAI.md`](MDSkillsAI.md).

Start from [`templates/parameters_regular.py`](../templates/parameters_regular.py)
or the applicable first-order template. The complete public configuration and
defaults are in [`templates/parameters.master.py`](../templates/parameters.master.py)
and [`docs/PARAMETERS.md`](../docs/PARAMETERS.md).

## Before running

Confirm that:

- metadata and source-table paths exist;
- the requested dimension and fields agree with validated metadata;
- any required first-order sidecar is declared and available;
- the output location and overwrite policy match the user's intent;
- the requested grid size and worker count are appropriate for available time
  and memory.

Preprocess before a costly run when the input is new or changed:

```bash
eos-trensmuther preprocess path/to/parameters.py
```

Review validation errors, warnings, detected shape, canonical ranges, and the
table-sanity summary. Do not adjust scientific tolerances merely to suppress a
failure.

## Target grid

`GridAxes` is required and supplies one count for each selected inverse-grid
axis. An integer count uses automatic metadata-derived bounds:

```python
"GridAxes": {"e": 400, "nB": 400}
```

These automatic component-wise bounds form a rectangular sampling envelope.
They are convenient and normally appropriate when the user has not requested a
smaller physical window, but correlations and phase structure can make points
inside the rectangle unreachable. Do not describe the envelope as the exact
physical image.

Use explicit lower/upper values when the user supplies a scientific domain or
when a bounded diagnostic is needed:

```python
"GridAxes": {
    "e": {"count": 100, "lower": 0.01, "upper": 0.20},
    "nB": {"count": 100, "lower": 0.00, "upper": 0.50},
}
```

Regular target variables are the default. Add `"Variables": "tilde"` only
when transformed coordinates are explicitly wanted. Do not invent or manually
rederive tilde formulas; the package uses canonical targets and metadata charge
constraints. Leave residual scaling automatic unless an expert user requests
and justifies an override.

## Projection

Projection is an optional recovery/output policy. It maps an unreachable
requested target to a nearby admissible thermodynamic state; it does not turn
that state into an exact inverse root.

- With projection disabled, preserve certified-unreachable and unresolved
  outcomes.
- With projection enabled, projected rows are included and tagged
  `projected=1`.
- Never report a projected row as an exact solution of the requested target.

Read [`docs/PROJECTION.md`](../docs/PROJECTION.md) before interpreting projected
scientific data.

## Execution modes

### Serial

Use for small examples, debugging, and bounded validation:

```bash
eos-trensmuther run path/to/parameters_serial.py
```

Set `Execution.workers` to `1`.

### Local multiprocessing

Use persistent local worker processes for larger independent target grids:

```bash
eos-trensmuther run path/to/parameters_multicore.py
```

Use the worker count requested by the user or a conservative value supported by
the machine's memory. Do not assume that using every CPU is faster or safe.
Worker count and `worker_task_size` are execution-only, so a compatible run may
resume with different values. `chunk_size` defines checkpoint layout and cannot
be changed on that resume.

### Slurm

Planning is local and does not submit a job:

```bash
eos-trensmuther slurm-plan path/to/parameters_slurm.py \
  --wall-time 01:00:00 --cpus-per-task 1
```

Inspect the generated plan and `submit_array.sbatch`. Submit with `sbatch` only
when the user explicitly authorizes submission and the environment is a Slurm
cluster. The repository, environment, inputs, shared state, checkpoints, and
shards must be visible at consistent absolute paths on compute nodes.

After all shards finish:

```bash
eos-trensmuther slurm-assemble --plan path/to/array_plan.json
eos-trensmuther export-text path/to/parameters_slurm.py
```

See the exact bounded workflow in
[`examples/regular/README.md`](../examples/regular/README.md).

## Checkpoints, overwrite, and expensive work

Keep checkpoint/restart enabled for production work. Do not delete or replace
existing results merely to restart; first establish whether the checkpoint is
compatible. Use `--overwrite` only when replacing the resolved run directory is
intentional and authorized.

For a long run, first perform a bounded pilot or inspect several completed
atomic chunks before estimating runtime. Early throughput is only an estimate
for a heterogeneous grid. Do not wait for a long external job or submit one
unless the user requested that action.

## Report the outcome

Report at least:

- resolved run directory and authoritative `inverse.eosinv` path;
- grid shape and execution mode/worker count;
- exact solved, projected, certified-unreachable, unresolved, error, and
  Jacobian-selected counts;
- checkpoint/resume state and elapsed time;
- whether generic text export succeeded.

Unresolved, invalid, or error rows prevent generic text export even if an
authoritative artifact was saved. Investigate them before calling a table
production-ready. For output interpretation, read
[`OutputSkillsAI.md`](OutputSkillsAI.md).
