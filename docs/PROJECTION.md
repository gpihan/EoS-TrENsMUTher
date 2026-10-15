# Nearest-state projector: contract and code guide

This document explains the nearest-state projection contract. The full
inversion and proof sequence is summarized in
[`METHODOLOGY.md`](METHODOLOGY.md).

## 1. What projection means

For a requested canonical target vector `y`, the projector searches the
admissible source domain `D_x` for a state with small fixed-metric distance

\[
d(x)=\left\|(F(x)-y)/s\right\|_2,
\]

where `F` is the validated target-only multilinear interpolator and `s` is one
positive scale vector fixed for the complete run.

This is not root acceptance. A projection may have a nonzero residual, and a
projection whose residual happens to be zero is still returned as a
`ProjectionResult`; only `EoSRootSolver` assigns root statuses or proofs.

The current method is deterministic local multistart. It returns the best
feasible state encountered but does not claim to prove the global closest
state. `ProjectionQuality.GLOBAL_CERTIFIED` is reserved and is never emitted
by `EoSProjector`.

## 2. Public API

### `ProjectionConfig`

The immutable configuration records:

- whether projection is enabled at pipeline level;
- how many target-AABB-ranked cell centers become starts;
- the maximum function evaluations per start;
- explicit SciPy `ftol`, `xtol`, and `gtol` values;
- the cell-processing batch size; and
- a preallocation limit for the projection index.

An `EoSProjector` cannot be constructed from a disabled configuration. The
pipeline uses that setting to skip the projection branch entirely.

### `EoSProjector`

The constructor requires:

- a validated `MultilinearInterpolator` containing exactly the canonical
  targets in metadata order;
- one finite, positive, safely invertible scale per target; and
- optionally, a nonempty boolean active-cell mask.

Convenience constructors keep run-wide contracts aligned:

- `EoSProjector.from_plan(interpolator, plan)` uses the resolved pipeline
  scales and projection configuration;
- `EoSProjector.from_reachability(index, config)` reuses the reachability
  index's active-cell catalog, exact cell envelopes, and scales.

`project(target, seeds=...)` accepts one finite canonical target. Optional
`ProjectionSeed` values can carry `FAST_BEST_CANDIDATE` or user provenance.

### `ProjectionResult`

The immutable result keeps these values separate:

- `requested_target`;
- `source_coordinates`;
- `achieved_target` from a final authoritative forward reevaluation;
- physical and scaled residual vectors;
- scaled infinity norm, L2 distance, and `0.5 * distance**2` objective;
- quality and selected-seed provenance;
- the nearest active node's canonical flat index; and
- every `ProjectionAttempt` diagnostic.

The requested target is never overwritten by the achieved target.

## 3. Deterministic algorithm

For each target, `EoSProjector` performs the following sequence:

1. Scan active table nodes in canonical C order and install the nearest one as
   a guaranteed feasible incumbent. Exact distance ties retain the smaller
   flat node index.
2. Build starts in fixed order: nearest active node, full-domain midpoint when
   every cell is active, target-AABB-ranked active cell centers, then supplied
   seeds.
3. Reevaluate every seed before optimization. A valid seed therefore remains
   eligible even if SciPy stops or raises.
4. Run scaled bounded least squares from every distinct start.
5. Reevaluate every usable optimizer point with the authoritative
   interpolator and retain only strict objective improvements. Fixed ordering
   resolves exact ties reproducibly.
6. Reevaluate the selected source point once more and construct the result
   algebra from that value rather than trusting an optimizer cache or cost.

Cell target boxes rank starts only. Their lower-bound distances are not
presented as closest-state certificates.

## 4. Active-cell safety

Without a mask, numerical trials use the full source bounding box. With a
nontrivial active-cell mask, a global box would allow an optimizer to cross an
excluded phase region. The implementation therefore:

- derives feasible nodes only from corners incident to active cells;
- assigns generic seeds to a deterministic containing active cell;
- keeps each cell-center seed in its owning active cell; and
- bounds every optimizer call to that single cell.

At an active cell's internal upper face, the adapter requests the left-sided
Jacobian; at a lower face it requests the right-sided Jacobian. Thus an
excluded neighboring cell cannot supply the derivative used by the trial.

The metadata-to-active-mask scientific policy is not implemented yet. Callers
must currently provide an explicit mask, and the same mask must govern
screening, solving, and projection.

## 5. Quality labels

| Quality | Exact meaning |
| --- | --- |
| `OPTIMIZER_CONVERGED` | The selected strict improvement came from an optimizer point whose run reported convergence. This is local, not global. |
| `BEST_FEASIBLE_ENCOUNTERED` | A supplied/built-in seed or nonconverged optimizer point strictly improved the node incumbent. |
| `DISCRETE_NODE_FALLBACK` | No candidate strictly improved the guaranteed nearest active-node state. |
| `GLOBAL_CERTIFIED` | Reserved for a future proof-producing global projector. |

Optimizer failure alone is not a projection failure because the active node
remains usable. `ProjectionInvariantError` is reserved for a broken domain or
forward-reevaluation guarantee; the pipeline maps such a failure
to its explicit `PROJECTION_FAILED` disposition.

## 6. Minimal usage

```python
from eos_trensmuther import EoSProjector, MultilinearInterpolator

targets = MultilinearInterpolator(table)
projector = EoSProjector(
    targets,
    residual_scales={"e": 1.0, "nB": 0.1},
)
result = projector.project([requested_e, requested_nB])

print(result.source_coordinates)
print(result.achieved_target)
print(result.scaled_distance_l2, result.quality.value)
```

The scale values above are illustrative; production values are a scientific
run configuration, not defaults inferred separately for each target.
