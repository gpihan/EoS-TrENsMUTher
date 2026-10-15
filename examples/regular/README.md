# Regular synthetic EoS example

This self-contained table is a small affine **non-physical** EoS used to
demonstrate the ordinary workflow.

From the repository root after installation:

```bash
eos-trensmuther run examples/regular/parameters_serial.py
eos-trensmuther run examples/regular/parameters_multicore.py
```

Both runs validate the table, invert a 5-by-5 `(e,nB)` grid, save a restartable
`inverse.eosinv`, and export `eos_table.dat`. Some envelope points are
unreachable; projection is enabled and those output rows have `projected=1`.

To generate, but not submit, a Slurm array plan:

```bash
eos-trensmuther slurm-plan examples/regular/parameters_slurm.py \
  --partition cpu --wall-time 00:10:00 --cpus-per-task 1
```

Inspect the generated `submit_array.sbatch`, submit it with `sbatch`, then use
the printed `array_plan.json` path:

```bash
eos-trensmuther slurm-assemble --plan PATH/array_plan.json
eos-trensmuther export-text examples/regular/parameters_slurm.py
```

The source table, installed environment, plan, mmap state, shards,
checkpoints, and output must be visible at identical absolute paths from the
login and compute nodes.
