# Methodology

This document describes the current release-level numerical method.
Projection details remain in [PROJECTION.md](PROJECTION.md); public inputs and
outputs are described in [METADATA.md](METADATA.md),
[PARAMETERS.md](PARAMETERS.md), and [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md).

## Problem

For a validated complete rectilinear table, EoS-TrENsMUTher constructs the
piecewise-multilinear forward map

```text
F: x -> y
x = (T, muB, muQ, muS)       up to the table dimension
y = (e, nB, nQ, nS)
```

and solves `F(x) = y_requested` independently at every point of a regular
target or independent-tilde grid. The source and target dimensions are equal
after applying metadata charge constraints. Auxiliary fields are evaluated
only after a source state has been accepted or projected.

## End-to-end state machine

```text
metadata + numerical table
            |
            v
validation, canonical units, Cartesian ordering
            |
            v
multilinear target/auxiliary interpolators
            |
            v
regular target or independent-tilde grid
            |
            v
conservative reachability + complete candidate cells C0(y)
       | excluded                         | candidate
       v                                  v
CERTIFIED_UNREACHABLE          one bounded FAST solve
                                          |
                          +---------------+---------------+
                          | accepted root                 | failure
                          v                               v
                       SOLVED              depth <= 7 local subcells
                                                          |
                              +---------------------------+----------------+
                              | empty complete union      | residual boxes
                              v                           v
                   CERTIFIED_UNREACHABLE       shared Krawczyk/FALLBACK
                                                          |
                              +---------------+-----------+-----------+
                              | root          | all excluded          | limit/
                              v               v                       | inconclusive
                           SOLVED     CERTIFIED_UNREACHABLE            v
                                                                     UNRESOLVED
```

If projection is explicitly enabled, a target is considered for projection
only after certified exclusion. Projection produces `PROJECTED`, not
`SOLVED`. `UNRESOLVED` is never projected because failure to finish a proof is
not proof that a root is absent.

## Interpolation and boundaries

The source grid may be nonuniform but must be a complete Cartesian product.
Multilinear interpolation is exact at nodes, continuous across faces, and
piecewise differentiable. It avoids interpolation-created cubic overshoot and
admits conservative per-cell value/Jacobian enclosures.

Before inversion, structural checks exhaustively cover finite required
values, coordinate tuples, holes, and row ordering. Local forward-map geometry
is exhaustively classified in 1D/2D (with exact bilinear determinant ranges
in 2D). In 3D/4D a deterministic domain-wide center sample is capped at
50,000 cells and a suspicious-cell halo at 10,000 further cells. This bounded
scan looks for folds, singularity, or extreme conditioning. If an ordinary
table contains both positive and negative regular Jacobian orientations,
validation stops before inversion because the observed reversal is
substantial evidence of a fold or non-injective forward map. Either uniform
orientation sign is accepted; positive orientation is not preferred.
Declared first-order tables retain orientation reversal as a diagnostic and
use their dedicated phase-aware route instead. Isolated singular or
sign-changing cells remain explicit warnings when no opposite regular
orientations are observed. Passing the scan is not a proof of global
injectivity, especially in the bounded 3D/4D case.

The automatic target boundary is the component-wise node minimum/maximum: the
smallest axis-aligned rectangle containing all table nodes. It is the natural
automatic sampling envelope, but correlations and folds mean many points
inside it may be unreachable. Conservative cell-image localization separates
that rectangular sampling choice from physical reachability.

The tilde transform is analytical and constraint-aware. Its independent
dimension always equals the EoS dimension; omitted charge directions become
derived tilde outputs. Because this transform is nonlinear, a node envelope
in tilde space is a sampling envelope rather than a rigorous enclosure of all
cell interiors.

## Reachability and complete localization

Every source cell has an outward-safe axis-aligned enclosure in target space.
A target outside the complete union can be rejected without running Newton.
Targets inside one or more enclosures retain the complete canonical list of
candidate source cells. Four-dimensional ordinary tables use the validated
depth-7 immutable hierarchy; dimensions one through three use the exhaustive
collector. Both preserve complete localization semantics.

Cell boxes are routing and exclusion devices. Membership in a conservative
box does not assert that the target is exactly reachable, and the number of
candidate cells does not by itself establish the number of roots.

## FAST solve

The fast path uses bounded, scaled numerical localization followed by
safeguarded Newton iterations with analytical cell-local Jacobians. Source
coordinates are normalized, target residual rows use one fixed positive scale
vector, and all steps remain in the closed source domain. A root is accepted
only after authoritative forward reevaluation meets the configured scaled
infinity-norm residual tolerance.

Optimizer termination or Newton failure is not evidence of no root. The
production policy makes one FAST attempt, then passes the complete candidate
set to the proof-safe fallback.

## Subcell-assisted rigorous fallback

Each original candidate cell is refined in local dyadic coordinates to a
maximum depth of seven. Outward-rounded value enclosures exclude children
that cannot contain the requested target. An empty union proves no root only
when refinement started from complete, provenance-matched `C0(y)` and every
branch was processed.

Residual boxes share one target and rigorous-cell context. The solver applies
Krawczyk exclusion/inclusion, then the established recursive interval queue
only where the first pass is inconclusive. A Krawczyk inclusion certifies
uniqueness in that box, not global uniqueness across the whole source domain.

Any active-box guard, numerical enclosure failure, interruption, or incomplete
coverage produces `UNRESOLVED/INCONCLUSIVE`. Resource exhaustion is never
converted to `CERTIFIED_UNREACHABLE`.

This hybrid is more robust than plain Newton plus scalar bisection for a
multivariable table: Newton supplies speed near a root, while complete cell
provenance, interval exclusions, and Krawczyk tests preserve branches and make
no-root claims auditable. There is no general multidimensional analogue of
one-dimensional sign-changing bisection that alone handles folds, multiple
roots, or piecewise-multilinear cell boundaries.

## Parallel and persistence semantics

Target nodes are independent jobs. Process workers reuse immutable mmap-backed
table, interpolation, reachability, and solver state. Small tasks are scheduled
dynamically across several atomic checkpoint chunks; returned slices are
assembled in deterministic C order by the parent. Workers never publish final
arrays or checkpoints.

Only a complete outer chunk becomes authoritative. On restart, validated
published chunks are restored and work from unpublished partial chunks is
recomputed. Scientific run fingerprints include the table, grid, scales,
reachability, solver/proof, projection, and routing contracts. Worker count
and small task size are execution-only; checkpoint `chunk_size` is layout
identity.

## Status contract

- `SOLVED`: accepted in-domain root. It does not imply global uniqueness.
- `CERTIFIED_UNREACHABLE`: a complete proof excluded every supported branch.
- `UNRESOLVED`: no conclusion within numerical/resource limits.
- `PROJECTED`: optional feasible approximation to an excluded target.
- `INVALID_TARGET`, `PROJECTION_FAILED`, `ERROR`: explicit input,
  projection, or unexpected execution failures.

Requested targets, achieved targets, source coordinates, residuals,
dispositions, solver/proof states, routes, phase information, projection
quality, and auxiliary definedness remain separate persisted data.

## First-order isolation

The current 2D LINE/BINODALS route uses dedicated phase geometry and solver
logic. Ordinary subcell fallback does not consume LINE, BINODALS, or Region-S
work. This prevents an ordinary no-root proof from silently crossing a mixed
phase model. The dedicated construction is regression-tested and represents
the declared 2D phase geometry; it is not a general higher-dimensional
multibranch phase-transition method.

Finite-resolution tabulated data can occasionally create a small local fold
in the bilinear `(T, muB) -> (e, nB)` map and therefore more than one fully
verified mixed-phase inverse root. The solver does not choose by discovery
order, residual size, continuation, or an assumed positive determinant. It
first removes duplicate roots and verifies bounds, residuals, and phase
membership. It then examines the exact bilinear Jacobian-determinant range in
a one-cell halo around the candidate cells, excluding the candidate cells
themselves and every singular, sign-changing, or invalid surrounding cell.

If all usable surrounding cells have one orientation and exactly one
candidate has that same non-singular orientation, that root is accepted with
the explicit reason `SELECTED_FROM_JACOBIAN`. The unanimous surrounding
orientation may be positive or negative; there is no majority vote. Mixed
surrounding signs, insufficient regular neighbors, two matching candidates,
or a near-singular matching candidate retain the existing ambiguity error.
This is a numerical/table-interpolation recovery rule, not a universal
thermodynamic branch-selection principle.

## Validation evidence

Validation includes dimensions 1--4, exact nodes/faces/corners and one-ULP
neighbors, tiny densities, high candidate multiplicity, ill-conditioned maps,
folded maps, proof guards, projection, first-order isolation,
serial/process/restart/Slurm parity, artifact/export round trips, and a full
`30^4` HRG4D regression. See [BENCHMARKS.md](BENCHMARKS.md) for the measured
large-run configuration and results.
