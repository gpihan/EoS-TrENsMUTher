# Synthetic BINODALS example

This is a bounded **non-physical** regression fixture demonstrating the
dedicated 2D BINODALS route. The `phase` auxiliary is zero below the band,
one above it, and between zero and one in the mixed region.

```bash
eos-trensmuther run examples/first_order_binodals/parameters_serial.py
eos-trensmuther run examples/first_order_binodals/parameters_multicore.py
```

The four requested points lie in the synthetic mixed band. Expected
high-level result: four solved mixed-phase points, no ordinary-subcell entry,
an authoritative artifact, and a generic text export whose `mixed_phase`
column is one.
