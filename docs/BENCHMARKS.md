# V1 benchmark and regression evidence

This page records measured validation evidence for the accepted v1 numerical
route. It is not a performance promise: table geometry, target composition,
hardware, filesystem, process start cost, and proof difficulty all affect
runtime.

## Decisive HRG4D regression

The final large regression used an external HRG4D source table with
6,250,000 nodes and a regular four-dimensional tilde target grid:

```text
target shape:                 30 x 30 x 30 x 30
target count:                 810,000
numerical routing contract:   2
root residual tolerance:      1e-10
projection:                   disabled
reachability hierarchy:       depth 7
ordinary subcells:            enabled, maximum depth 7
active-box guard:             1,000,000 per target
workers:                      8 persistent spawned processes
worker task size:             8 targets
maximum pending tasks:        16
atomic checkpoint chunk:      4,096 targets
checkpoint chunks:            198
```

The workflow loaded the normal source metadata/table and constructed a fresh
run-local shared mmap/reachability state. It did not require a historical
cache.

### Scientific results

| Outcome | Count |
| --- | ---: |
| Prescreen certified unreachable | 675,360 |
| Reachability candidates | 134,640 |
| FAST solved | 16,036 |
| Fallback certified no-root | 118,604 |
| Total certified unreachable | 793,964 |
| Unresolved | 0 |
| Errors | 0 |

All 16,036 accepted roots were independently reevaluated with the canonical
multilinear interpolator and were inside source bounds. Scaled infinity-norm
residuals were:

| Statistic | Value |
| --- | ---: |
| Median | `2.6652877010538158e-14` |
| 95th percentile | `2.0296703194118772e-11` |
| 99th percentile | `6.356504624022015e-11` |
| Maximum | `9.991416899453788e-11` |

Every compared per-index disposition, route, prescreen, solve, FAST, proof,
projection, phase, mixed-state field, and derived scientific mask agreed with
the frozen pre-subcell routing reference. No real solved target had a singleton
complete candidate-cell set, so global inverse uniqueness was not claimed and
returned-coordinate identity was diagnostic only.

The proof path evaluated 734,184,800 conservative subcell boxes. It produced
118,152 direct empty-union certificates; 452 remaining targets sent 223,428
residual boxes through the shared Krawczyk context, and all were rigorously
excluded without recursive subdivision.

### Measured setup, runtime, and memory

Measurements were made on Darwin/arm64 with 10 logical CPUs, CPython 3.9.6,
NumPy 1.26.4, and SciPy 1.13.1. The hardware report did not retain a more
specific machine model.

| Measurement | Value |
| --- | ---: |
| Source load and validation | `5.683820 s` |
| Fresh shared-state construction | `8.544591 s` |
| Run-local shared-state size | `1,405,620,808 bytes` |
| Inversion wall time | `6,668.341952 s` |
| Pipeline wall time | `6,668.113594 s` |
| Throughput | `121.473635 targets/s` |
| Artifact publication | `0.551379 s` |
| Parent peak RSS | `2,119,008,256 bytes` |
| Maximum reporting-worker peak RSS | `1,401,044,992 bytes` |
| Declared per-process memory ceiling | `3,221,225,472 bytes` |

The measured inversion time is about 1 hour 51 minutes 8 seconds. All 22
large retained array references were mmap-backed by 20 unique files totaling
1,405,316,104 bytes.

The earlier routing reference had an approximately four-hour active wall
measurement. That number is explicitly an estimate recovered from the
historical run record, because a later checkpoint-only replay overwrote its
artifact invocation timer. Comparing the measured new wall time to that
approximate reference gives about `2.1595x` speedup; it must not be presented
as two equally instrumented benchmark measurements.

## Scheduler selection

A separate fixed 128-target mixed HRG4D workload compared process task sizes
8, 16, and 32 with 1, 2, 4, and 8 workers. All 12 cells had exact scientific
status/proof parity with the serial result and zero returned-coordinate
difference. Task size 8 was the smallest size within 10% of the best aggregate
wall time and became the release default. On that bounded workload its
one-to-four-worker speedup was `2.34185x`; eight workers saturated and were
slower than four.

## Interpreting these numbers

- The counts and residual checks are regression contracts for this external
  HRG4D input and configuration.
- Runtime and memory are observations from one machine, not minimum hardware
  requirements or general speed guarantees.
- The public source distribution does not include the large HRG4D table,
  reference artifact, checkpoint tree, or mmap files.
- Bounded synthetic tests establish installed-package behavior; the accepted
  full-run evidence establishes the unchanged numerical implementation on the
  production-sized case.
