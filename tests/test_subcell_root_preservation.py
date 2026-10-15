"""V1 Phase-5 known-root preservation regressions."""

import os
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    EoSRootSolver,
    MultilinearInterpolator,
    ReachabilityConfig,
    SolveStatus,
    load_pipeline_result,
)
from eos_trensmuther.shared_state import load_shared_reachability, load_shared_table
from tests._support.validate_v1_phase5_root_preservation import (
    CI_QUOTAS,
    CI_REAL_COUNT,
    DEFAULT_ARTIFACT,
    DEFAULT_OUTPUT,
    DEFAULT_SHARED_STATE,
    MAX_DEPTH,
    TIER_B_SYNTHETIC_COUNT,
    UNIQUE_ROOT_COUNT,
    _folded_synthetic_cases,
    _candidate_center_seed,
    _localization_statistics,
    check_kernel_root_preservation,
    run_unique_root_comparison,
)


CI_ARTIFACT = (
    Path(__file__).resolve().parent / "fixtures" / "phase5_ci_real_192.npz"
)
RUN_SLOW_REAL_CI = os.environ.get("EOSINVERTER_RUN_SLOW_HRG4D_PHASE5") == "1"


def _folded_interpolator():
    axes = (np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0]))
    x, q = np.meshgrid(*axes, indexing="ij")
    return MultilinearInterpolator.from_rectilinear_map(
        axes,
        {"e": np.abs(x - 1.0), "nB": q},
        coordinate_names=("T", "muB"),
        target_names=("e", "nB"),
        coordinate_units=("1", "1"),
        target_units=("1", "1"),
    )


class SyntheticRootPreservationTests(unittest.TestCase):
    def test_closed_face_edge_and_node_ownership_survive_every_depth(self):
        interpolator = _folded_interpolator()
        points = np.asarray(
            [
                [0.5, 0.5],       # dyadic crossing inside one cell
                [1.0, 0.5],       # shared original-cell face
                [0.0, 0.0],       # source-domain corner
                [2.0, 1.0],       # opposite source-domain corner
            ]
        )
        targets = interpolator.evaluate(points)
        for point, target in zip(points, targets):
            result = check_kernel_root_preservation(
                interpolator,
                target,
                point,
                np.ones(2),
                1.0e-10,
            )
            self.assertEqual(len(result["branches_per_level"]), MAX_DEPTH + 1)
            self.assertGreater(result["total_boxes_checked"], 0)
            self.assertGreaterEqual(result["minimum_scaled_enclosure_margin"], 0.0)

    def test_folded_noninjective_integrated_cases_preserve_both_branches(self):
        report = _folded_synthetic_cases()
        self.assertEqual(report["count"], TIER_B_SYNTHETIC_COUNT)
        for row in report["rows"]:
            self.assertEqual(row["candidate_cells"], 2)
            self.assertNotEqual(row["fallback_only_status"], "certified_no_root")
            self.assertEqual(row["status"], "converged_fast")
            self.assertLessEqual(row["scaled_residual_norm_inf"], 1.0e-10)

    def test_affine_unique_roots_return_expected_coordinates(self):
        report = run_unique_root_comparison()
        self.assertEqual(report["case_count"], UNIQUE_ROOT_COUNT)
        self.assertEqual(
            report["selection"]["category_counts"],
            {
                "ordinary_interior": 24,
                "near_boundary": 12,
                "domain_corner": 12,
            },
        )
        self.assertEqual(
            set(report["selection"]["condition_profiles"]),
            {
                "well_conditioned",
                "moderately_conditioned",
                "high_condition_number",
            },
        )
        self.assertTrue(report["acceptance"]["gate_satisfied"])
        for profile in report["profiles"].values():
            matrix = np.asarray(profile["affine_matrix"], dtype=float)
            inverse_norm_inf = float(np.linalg.norm(np.linalg.inv(matrix), ord=np.inf))
            self.assertAlmostEqual(float(np.linalg.norm(matrix, ord=np.inf)), 1.0)
            self.assertAlmostEqual(
                profile["normalized_inverse_jacobian_norm_inf"],
                inverse_norm_inf,
            )
            self.assertAlmostEqual(
                float(np.linalg.cond(matrix, p=np.inf)),
                inverse_norm_inf,
            )
        for row in report["rows"]:
            self.assertEqual(row["disposition"], "solved")
            self.assertTrue(row["owner_cells_present"])
            self.assertTrue(row["in_source_bounds"])
            self.assertTrue(row["coordinate_match"])
            self.assertLessEqual(
                row["scaled_residual_norm_inf"],
                report["acceptance_contract"]["root_residual_tolerance"],
            )
            self.assertLessEqual(
                row["independent_scaled_residual_norm_inf"],
                report["acceptance_contract"]["root_residual_tolerance"],
            )


class FixedHRG4DCISubsetTests(unittest.TestCase):
    @unittest.skipUnless(CI_ARTIFACT.is_file(), "Phase-5 fixed CI artifact not generated")
    def test_fixed_artifact_has_declared_difficult_composition(self):
        with np.load(CI_ARTIFACT, allow_pickle=False) as archive:
            categories = archive["categories"]
            conditions = archive["condition_numbers"]
            candidate_counts = archive["candidate_counts"]
            depths = archive["completed_subcell_depth"]
            self.assertEqual(categories.size, CI_REAL_COUNT)
            counts = {
                str(category): int(np.count_nonzero(categories == category))
                for category in np.unique(categories)
            }
            self.assertEqual(counts, dict(CI_QUOTAS))
            self.assertTrue(np.all(np.isfinite(conditions)))
            self.assertTrue(np.all(candidate_counts > 0))
            self.assertTrue(np.all((depths >= 0) & (depths <= MAX_DEPTH)))
            self.assertGreater(float(np.percentile(conditions, 95.0)), float(np.median(conditions)))
            self.assertGreaterEqual(int(np.max(candidate_counts)), int(np.median(candidate_counts)))

    @unittest.skipUnless(
        RUN_SLOW_REAL_CI
        and CI_ARTIFACT.is_file()
        and DEFAULT_ARTIFACT.is_file()
        and DEFAULT_SHARED_STATE.is_file(),
        "set EOSINVERTER_RUN_SLOW_HRG4D_PHASE5=1 with Phase-5 artifacts",
    )
    def test_fixed_real_subset_replays_complete_integrated_path(self):
        artifact = load_pipeline_result(DEFAULT_ARTIFACT)
        table = load_shared_table(DEFAULT_SHARED_STATE)
        interpolator = MultilinearInterpolator(table, artifact.plan.target_names)
        reachability = load_shared_reachability(
            DEFAULT_SHARED_STATE,
            interpolator,
            artifact.plan,
            ReachabilityConfig(method="hierarchical", hierarchy_levels=7),
        )
        config = replace(
            artifact.plan.config.solver,
            enable_subcell_fallback=True,
            max_subcell_depth=MAX_DEPTH,
        )
        solver = EoSRootSolver(
            interpolator, config, residual_scales=artifact.plan.residual_scales
        )
        with np.load(CI_ARTIFACT, allow_pickle=False) as archive:
            points = archive["source_coordinates"]
            targets = archive["targets"]
        localization, _, owners = _localization_statistics(
            reachability, targets, points, interpolator
        )
        self.assertTrue(np.all(owners))
        started = time.perf_counter()
        for row, (point, target) in enumerate(zip(points, targets)):
            seed, _, _ = _candidate_center_seed(
                interpolator,
                target,
                localization.cells_for(row),
                np.asarray(artifact.plan.residual_scales, dtype=float),
            )
            result = solver.solve_with_candidate_cells(
                target,
                localization,
                row,
                seed,
                expected_provenance=reachability.provenance,
                strategy="auto",
            )
            self.assertIsNot(result.status, SolveStatus.CERTIFIED_NO_ROOT)
            if result.converged:
                self.assertIsNotNone(result.solution)
                self.assertTrue(np.all(interpolator.contains(result.solution)))
                self.assertIsNotNone(result.scaled_residual_norm)
                self.assertLessEqual(
                    result.scaled_residual_norm, config.root_residual_tolerance
                )
            else:
                self.assertIs(result.status, SolveStatus.UNRESOLVED)
        self.assertGreater(time.perf_counter() - started, 0.0)


if __name__ == "__main__":
    unittest.main()
