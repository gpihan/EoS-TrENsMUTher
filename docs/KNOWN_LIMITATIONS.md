# Known limitations

These limits are part of the current contract. They should be considered when
selecting input data and interpreting an inverse table.

## Input geometry

- Source tables must be complete rectilinear Cartesian products in dimensions
  one through four. Axis spacing may be nonuniform, but holes, duplicates, and
  truly scattered/unstructured data are rejected.
- Interpolation is piecewise multilinear. There is no triangulation, radial
  basis, spline, Gaussian-process, or learned interpolator.
- Automatic inverse-grid bounds are component-wise node envelopes. The
  Cartesian product contains points that may not be physically reachable.
- Nonlinear tilde node bounds are not a rigorous enclosure of every
  transformed cell interior. Explicit scientifically selected bounds may be
  preferable.
- The deterministic 3D/4D source-table Jacobian scan is bounded and
  diagnostic. Passing it is not a proof of global injectivity; an unsampled
  local defect can remain.
- Malformed coordinates, nonfinite required fields, nonfinite local
  Jacobians, and maps for which every inspected cell is singular fail normal
  loading. For ordinary tables, observed positive and negative regular
  Jacobian orientations also fail loading because they are substantial
  evidence of a fold or non-injective map. Either uniform sign is valid.
  Declared first-order tables retain such reversals as diagnostics for the
  phase-aware route. Isolated sign-changing/singular cells without observed
  opposite regular orientations remain warnings and must be reviewed.

## Inversion and branches

- An accepted FAST root satisfies the scaled residual contract but has no
  global uniqueness guarantee.
- Krawczyk uniqueness is box-local. A folded/non-injective EoS may have several
  valid inverse branches, and the software does not invent a physical branch-selection
  rule.
- For the 2D first-order route only, one narrow numerical fold
  can be recovered when exactly one verified root matches the unanimous
  Jacobian orientation of all usable regular cells in its local one-cell
  halo. The root is marked `SELECTED_FROM_JACOBIAN`. This is not a general
  thermodynamic branch rule; all cases without a unique local-orientation
  choice remain ambiguous.
- Candidate-box multiplicity is a routing fact, not proof of root
  multiplicity.
- `UNRESOLVED` is possible when proof work reaches a safety guard or interval
  analysis remains inconclusive. It must not be treated as no root.
- Subcell refinement is fixed at a maximum depth of seven; adaptive depth,
  box merging, GPU proofs, and intra-target parallelism are deferred.

## Projection

- Projection is deterministic local multistart over the admissible domain. It
  returns the best feasible state encountered, not a certificate of the
  globally closest state.
- Projected rows are not roots of their requested targets and remain labeled
  `PROJECTED` in artifacts and exports.
- Metadata does not yet derive a general active-cell mask from phase
  information. An explicit restricted mask is not supported by the ordinary
  table-wide FAST route.

## Constraints and phase structure

- Constraint-aware tilde transformations support the declared homogeneous
  linear charge relations. Arbitrary nonlinear constraints are not supported.
- First-order support is currently restricted to 2D
  `(T, muB) -> (e, nB)` LINE or BINODALS metadata plus the existing Region-S
  formulation. There is no general higher-dimensional phase topology,
  multibranch selection framework, or new critical-scaling model.
- Real first-order consumer-output golden fixtures remain external scientific
  validation data; the package carries synthetic isolation regressions.

## Execution and storage

- Local parallelism uses CPU processes. Large ordinary runs need temporary
  disk space for read-only mmap state and enough memory for the parent plus
  workers; measured numbers from one machine are not universal requirements.
- Slurm execution partitions only the first inverse-grid axis and assumes a
  shared filesystem visible at the same absolute paths from every task.
- Parameter files are trusted executable Python, not a safe declarative input
  format. Table metadata itself is JSON.
- Current executable checkpoints, Slurm plans, and shards are strict routing
  state. Older incompatible routing contracts are rejected. Some completed
  historical `.eosinv` artifacts remain readable only for compatibility and
  cannot be rewritten as current work.

## Data and exporters

- Large production CEC, Ising, and HRG4D tables are external scientific data;
  the source distribution includes only a metadata template and a small
  non-physical fixture.
- Users are responsible for table licensing, provenance, units, physical
  validity, constraints, publications, and sidecar correctness.
- The public exporter is a generic column-based text table. It deliberately
  does not generate hydro-code-specific binary layouts.
- Auxiliary interpolation is linear. Fields requesting `nearest` or no
  interpolation method are not silently substituted.

## Deferred research

Scattered-data triangulation, learned or symbolic coordinate maps, direct
transformed-space interpolation, FAST2, adaptive transformed coordinates,
adaptive process-task sizing, GPU interval work, and new phase-transition
methods are not currently implemented.
