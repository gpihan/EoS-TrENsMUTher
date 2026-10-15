# EoS-TrENsMUTher

EoS-TrENsMUTher inverts metadata-described tabulated equations of state. A
forward table such as

```text
(T, muB, muQ, muS) -> (e, nB, nQ, nS)
```

is sampled onto a regular grid in canonical target variables or independent,
constraint-aware tilde variables. The package distinguishes accepted roots,
rigorously certified no-root points, honest unresolved points, and optional
nearest-state projections.

The package supports complete rectilinear 1D–4D tables with multilinear
interpolation. Smooth ordinary tables use the ordinary route. Metadata-declared
first-order tables use the dedicated 2D LINE/BINODALS/Region-S route.

## Install

Python 3.9–3.13 is supported:

```bash
python -m pip install .
eos-trensmuther --version
```

Runtime dependencies are NumPy, SciPy, and `threadpoolctl`.

## Instructions for AI and LLM assistants

If you are an AI assistant reading or modifying this repository, inspect
[`AIutils/`](AIutils/) before starting the task. It is the canonical location
for project-specific AI skills, checklists, and supporting resources. Read the
[`AI resources index`](AIutils/README.md), then apply every resource relevant
to the requested work.

The index routes metadata preparation, run configuration/execution, output
interpretation, and first-order tasks to separate focused
resources. Additional resources may be added to `AIutils/` in later versions,
so assistants should enumerate the directory instead of assuming an older list
is exhaustive.

## Fast examples

The bundled examples are synthetic, non-physical, self-contained, and small
enough to run in a few seconds. They demonstrate file layout and program
behavior; they are not thermodynamic models. Run the commands below from the
repository root after installation. Paths inside a parameter file are
resolved relative to that file, so the commands do not depend on the current
shell directory once the parameter-file path is correct.

### Regular EoS: serial and local multicore

The regular example supplies a small 2D forward table
`(T,muB) -> (e,nB)`, metadata, and pressure/entropy auxiliary fields.

```bash
# Direct serial execution
eos-trensmuther run examples/regular/parameters_serial.py

# The same scientific problem with two persistent worker processes
eos-trensmuther run examples/regular/parameters_multicore.py
```

Each run:

1. validates the metadata, Cartesian grid, finite values, and local table
   geometry;
2. builds interpolation and reachability data;
3. inverts a 5-by-5 regular `(e,nB)` target grid;
4. projects requested points that have no exact inverse and marks them with
   `projected=1`;
5. writes atomic checkpoint chunks, an authoritative `.eosinv` artifact, and
   a generic text table.

The two configurations use separate run names and therefore do not overwrite
one another:

```text
example-output/RegularSynthetic2D/RegularSynthetic2D_serial/
example-output/RegularSynthetic2D/RegularSynthetic2D_multicore/
```

Inside either directory:

```text
inverse.eosinv                       authoritative saved result
eos_table.dat                        readable numeric text export
diagnostic/preprocess.json           source-table validation summary
diagnostic/text_export_manifest.json column/provenance description
diagnostic/checkpoint/               restartable atomic chunks
```

The expected regular-example result is 25 rows: 9 exact solved roots and 16
explicitly labeled projected rows, with no unresolved points or errors. This
count is a property of the bundled synthetic fixture, not a general promise
for physical tables.

The parameter files show the main execution difference:

```python
# serial
"Execution": {"workers": 1, "chunk_size": 5}

# local processes
"Execution": {
    "workers": 2,
    "start_method": "spawn",
    "chunk_size": 5,
    "worker_task_size": 2,
    "max_pending_chunks": 4,
}
```

If a named example run already exists, the program refuses to replace it.
Append `--overwrite` when you deliberately want to repeat that exact example:

```bash
eos-trensmuther run examples/regular/parameters_serial.py --overwrite
```

### Regular EoS: Slurm array workflow

Planning is local and does not submit a job:

```bash
eos-trensmuther slurm-plan examples/regular/parameters_slurm.py \
  --wall-time 00:10:00 --cpus-per-task 1
```

Add `--partition NAME`, `--account NAME`, `--memory SIZE`, or one or more
`--activate "COMMAND"` arguments when required by the cluster. Inspect the
generated files before submission:

```text
example-output/RegularSynthetic2D/RegularSynthetic2D_slurm/
  diagnostic/slurm/array_plan.json
  diagnostic/slurm/submit_array.sbatch
```

Submit the generated array script, wait for every array task to finish, then
assemble and export the result:

```bash
sbatch example-output/RegularSynthetic2D/RegularSynthetic2D_slurm/diagnostic/slurm/submit_array.sbatch

eos-trensmuther slurm-assemble \
  --plan example-output/RegularSynthetic2D/RegularSynthetic2D_slurm/diagnostic/slurm/array_plan.json

eos-trensmuther export-text examples/regular/parameters_slurm.py
```

The repository, installed environment, source table, plan, mmap state,
checkpoints, and shard files must be visible at the same absolute paths on the
login and compute nodes. `slurm-plan` only writes the plan and script; it never
runs `sbatch` automatically.

### First-order examples

The LINE example includes a coexistence-line sidecar and exercises the
isolated LINE/Region-S route:

```bash
eos-trensmuther run examples/first_order_line/parameters_serial.py
eos-trensmuther run examples/first_order_line/parameters_multicore.py
```

It requests a 2-by-2 target grid. The run should contain mixed-phase roots,
retain explicit labels for any unreachable rows, and never enter the ordinary
subcell route.

The BINODALS example supplies dilute/dense boundary temperatures and a
phase-fraction auxiliary field:

```bash
eos-trensmuther run examples/first_order_binodals/parameters_serial.py
eos-trensmuther run examples/first_order_binodals/parameters_multicore.py
```

Its four requested points are solved in the synthetic mixed band. In the text
export, all four rows have `mixed_phase=1`. These first-order constructions use
dedicated phase geometry; see the
[methodology](docs/METHODOLOGY.md#first-order-isolation) before applying them to
scientific data.

### Python API artifact round trip

The compact typed-Python utility runs a small inversion, saves the result,
reloads it independently, and checks the artifact:

```bash
python utilities/quickstart_api.py --output quickstart-api-output
```

More focused notes are available in the [examples index](examples/README.md)
and each example directory's README.

## Documentation

- [Installation and quick start](docs/USER_GUIDE.md)
- [Runnable examples](examples/README.md)
- [Parameters](docs/PARAMETERS.md)
- [Metadata and table format](docs/METADATA.md)
- [Complete metadata master and field guide](templates/METADATA_REFERENCE.md)
- [Copy-ready user templates](templates/README.md)
- [AI/LLM skills and resources](AIutils/README.md)
- [First-order EoS methodology](docs/METHODOLOGY.md#first-order-isolation)
- [Projection](docs/PROJECTION.md)
- [Output format](docs/OUTPUT_FORMAT.md)
- [Known limitations](docs/KNOWN_LIMITATIONS.md)
- [Methodology and proof semantics](docs/METHODOLOGY.md)
- [Public Python API](docs/PUBLIC_API.md)
- [Measured benchmarks](docs/BENCHMARKS.md)

Large physical EoS tables, production outputs, checkpoints, and mmap stores
are external scientific data and are not bundled.

## Result semantics

- `SOLVED`: accepted in-domain inverse satisfying the configured scaled
  residual tolerance.
- `CERTIFIED_UNREACHABLE`: the complete proof path established no root in the
  supported domain.
- `UNRESOLVED`: proof/search was inconclusive or reached a resource guard.
- `PROJECTED`: separately labeled feasible state; never an exact inverse of
  the original target.

The checksummed `.eosinv` artifact is authoritative. The generic text export
is a convenient tabular view with compact `#` header lines describing the
version, columns, units, and grid order. Public state flags are appended as
deterministic `0/1` columns by default.

For a transformed-grid run, add `"include_tilde": True` inside `Export` to
write `T_tilde` and all three chemical tilde coordinates as numeric columns.
The header then prints the actual constraint-reduced numerical formulas, and
the JSON sidecar stores the same coefficients in machine-readable form.

## How to cite

If EoS-TrENsMUTher contributes to scientific work, please cite the software
repository and version used:

```text
Grégoire Pihan, EoS-TrENsMUTher, version 1.0.0.dev0 (2026),
https://github.com/gpihan/EoS-TrENsMUTher
```

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).
A DOI may be added after the first public release. Citation is a scholarly
request and does not modify the terms of the MIT License.

## License

EoS-TrENsMUTher is released under the [MIT License](LICENSE).
