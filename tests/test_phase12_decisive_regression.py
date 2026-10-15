"""Lightweight contracts for the post-run Phase-12 validator."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from tests._support.validate_v1_phase12_decisive_regression import (
    EXPECTED_COUNTS,
    GRID_SHAPE,
    GRID_SIZE,
    _EXACT_SCIENTIFIC_FIELDS,
    _MASK_NAMES,
    _distribution,
    _performance,
    compare_semantics,
)


class Phase12DecisiveRegressionDriverTests(unittest.TestCase):
    def test_frozen_grid_and_reference_counts(self):
        self.assertEqual(GRID_SHAPE, (30, 30, 30, 30))
        self.assertEqual(GRID_SIZE, 810_000)
        self.assertEqual(EXPECTED_COUNTS["prescreen_unreachable"], 675_360)
        self.assertEqual(EXPECTED_COUNTS["candidates"], 134_640)
        self.assertEqual(EXPECTED_COUNTS["solved"], 16_036)
        self.assertEqual(
            EXPECTED_COUNTS["fallback_certified_no_root"], 118_604
        )
        self.assertEqual(EXPECTED_COUNTS["unresolved"], 0)
        self.assertEqual(EXPECTED_COUNTS["errors"], 0)

    def test_distribution_records_bounded_summary(self):
        report = _distribution(np.asarray([0.0, 1.0, 2.0, 3.0]))
        self.assertEqual(report["count"], 4)
        self.assertEqual(report["minimum"], 0.0)
        self.assertEqual(report["median"], 1.5)
        self.assertEqual(report["maximum"], 3.0)

    def test_semantics_comparison_is_per_index_and_coordinate_is_diagnostic(self):
        def result():
            value = SimpleNamespace()
            for name in _EXACT_SCIENTIFIC_FIELDS:
                setattr(value, name, np.asarray([1, 2], dtype=np.uint8))
            for name in _MASK_NAMES:
                setattr(value, name, np.asarray([True, False]))
            value.solution_mask = np.asarray([True, False])
            value.source_coordinates = np.asarray(
                [[0.5, 0.5, 0.5, 0.5], [np.nan] * 4]
            )
            value.plan = SimpleNamespace(
                config=SimpleNamespace(
                    solver=SimpleNamespace(newton_step_tolerance=1.0e-13)
                )
            )
            return value

        reference = result()
        current = result()
        current.source_coordinates[0, 3] += 2.0e-8
        table = SimpleNamespace(
            axes=tuple(np.asarray([0.0, 1.0]) for _ in range(4))
        )
        report = compare_semantics(reference, current, table)
        self.assertTrue(report["all_exact_scientific_fields_equal"])
        self.assertTrue(report["all_scientific_masks_equal"])
        self.assertEqual(
            report["all_solved_coordinate_comparison"]["acceptance_role"],
            "diagnostic_only_until_global_inverse_uniqueness_is_proven",
        )
        current.route_codes = np.asarray([1, 3], dtype=np.uint8)
        changed = compare_semantics(reference, current, table)
        self.assertEqual(changed["field_discrepancy_counts"]["route_codes"], 1)
        self.assertFalse(changed["all_exact_scientific_fields_equal"])

    def test_performance_derives_zero_recursive_work_from_complete_box_counts(self):
        stage = {
            name: {"sum": value}
            for name, value in {
                "fallback": 40.0,
                "fast": 2.0,
                "initial_guess": 1.0,
                "output_evaluation": 1.0,
                "phase_solve": 0.0,
                "prescreen": 6.0,
                "projection": 0.0,
            }.items()
        }
        old = SimpleNamespace(
            manifest={
                "run": {
                    "performance": {
                        "chunk_stage_seconds": {"fallback": {"sum": 100.0}},
                        "chunk_processing_seconds": {"sum": 150.0},
                    }
                }
            }
        )
        counters = {
            "subcell_residual_boxes": 25,
            "subcell_krawczyk_boxes_processed": 25,
            "subcell_residual_krawczyk": 2,
            "fast_attempted": 20,
            "fallback_attempted": 10,
            "subcell_fallback_attempted": 10,
            "subcell_certified_no_root": 8,
            "subcell_boxes_evaluated": 100,
            "projection_attempted": 0,
        }
        counters.update(
            {"subcell_certificate_depth_%d" % depth: 0 for depth in range(1, 8)}
        )
        new = SimpleNamespace(
            manifest={
                "run": {
                    "performance": {
                        "chunk_stage_seconds": stage,
                        "chunk_processing_seconds": {"sum": 50.0},
                        "computed_points_per_second_this_invocation": 10.0,
                        "observed_peak_rss_bytes": {
                            "parent_before_result_finalization": 100,
                            "maximum_reporting_worker": 200,
                        },
                    },
                    "timing_seconds": {
                        "total_run": 10.0,
                        "worker_stage_sum": {
                            name: values["sum"] for name, values in stage.items()
                        },
                    },
                    "execution": {},
                    "software": {},
                    "stage_counters": counters,
                    "status_summary": {"route": {"fast": 10}},
                    "checkpoint": {
                        "restored_chunks": 0,
                        "computed_chunks_this_invocation": 2,
                    },
                }
            }
        )
        step1 = {
            "source_table_load_and_validation_seconds": 1.0,
            "shared_state_action": "constructed_fresh_from_source_table",
            "shared_state_setup_seconds": 2.0,
            "shared_state_bytes": 3,
            "artifact_write_seconds": 4.0,
            "inversion_wall_seconds": 10.0,
        }
        report = _performance(old, new, step1)
        work = report["solver_work"]
        self.assertEqual(work["residual_krawczyk_recursive_evaluations_derived"], 0)
        self.assertEqual(work["residual_krawczyk_recursive_subdivisions_derived"], 0)
        self.assertTrue(report["comparison"]["material_full_run_improvement"])


if __name__ == "__main__":
    unittest.main()
