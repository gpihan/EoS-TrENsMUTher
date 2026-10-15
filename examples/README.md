# Examples

Every bundled example is synthetic, non-physical, self-contained, and small
enough for a laptop or continuous integration.

- [`regular/`](regular/) demonstrates ordinary inversion, projection, serial
  execution, local process execution, and Slurm plan generation.
- [`first_order_line/`](first_order_line/) demonstrates the dedicated 2D
  LINE/Region-S route with a coexistence-line sidecar.
- [`first_order_binodals/`](first_order_binodals/) demonstrates the
  dedicated 2D BINODALS route and a phase-fraction auxiliary.

Each directory contains its source table, metadata, parameter files, exact
commands, and expected high-level behavior. Real CEC, Ising, and HRG4D data
remain external scientific inputs.

The complete parameter reference is
[`templates/parameters.master.py`](../templates/parameters.master.py).
The compact typed-Python artifact utility is
[`utilities/quickstart_api.py`](../utilities/quickstart_api.py).
