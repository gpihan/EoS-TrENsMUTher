# Synthetic LINE / Region-S example

This is a bounded **non-physical** fixture demonstrating the dedicated 2D
LINE route. `coexistence_line.dat` declares a short synthetic transition line;
the program reconstructs Region S from the source table on both sides.

```bash
eos-trensmuther run examples/first_order_line/parameters_serial.py
eos-trensmuther run examples/first_order_line/parameters_multicore.py
```

Expected high-level result: the run completes through the isolated
first-order route, some requested points are mixed-phase roots, any
unreachable rows remain explicitly tagged, and no ordinary-subcell route is
used. This example demonstrates file structure and execution only; it is not
a thermodynamic model.
