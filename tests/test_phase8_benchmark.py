"""Pure contract tests for the V1 Phase-8 benchmark driver."""

import unittest

import numpy as np

from tests._support.validate_v1_phase8_scheduling import (
    compare_result_arrays,
    select_measured_default,
)


class Phase8BenchmarkContractTests(unittest.TestCase):
    def test_default_is_smallest_near_best_task_size(self):
        rows = [
            {"task_size": 8, "scheduler_wall_seconds": 10.8},
            {"task_size": 16, "scheduler_wall_seconds": 10.0},
            {"task_size": 32, "scheduler_wall_seconds": 11.2},
        ]
        selected = select_measured_default(rows)
        self.assertEqual(selected["eligible_task_sizes"], [8, 16])
        self.assertEqual(selected["selected_task_size"], 8)

    def test_parity_requires_exact_statuses_valid_roots_and_coordinates(self):
        base = {
            "prescreen": np.asarray(["candidate", "definitely_unreachable"]),
            "status": np.asarray(["converged_fast", "prescreen_excluded"]),
            "proof": np.asarray(["none", "reachability_no_candidate"]),
            "source_coordinates": np.asarray([[0.5, 0.5], [np.nan, np.nan]]),
            "scaled_residual_norm_inf": np.asarray([1.0e-12, np.nan]),
            "in_source_bounds": np.asarray([True, False]),
        }
        passed = compare_result_arrays(
            base,
            dict(base),
            lower=np.asarray([0.0, 0.0]),
            upper=np.asarray([1.0, 1.0]),
            residual_tolerance=1.0e-10,
            coordinate_tolerance=1.0e-11,
        )
        self.assertTrue(passed["parity_passed"])

        changed = dict(base)
        changed["proof"] = np.asarray(["unique_in_box", "reachability_no_candidate"])
        failed = compare_result_arrays(
            base,
            changed,
            lower=np.asarray([0.0, 0.0]),
            upper=np.asarray([1.0, 1.0]),
            residual_tolerance=1.0e-10,
            coordinate_tolerance=1.0e-11,
        )
        self.assertFalse(failed["parity_passed"])

        # Coordinate acceptance is component-wise in full-span-normalized
        # coordinates.  A wide unrelated axis must not enlarge the absolute
        # tolerance of a narrow axis.
        shifted = dict(base)
        shifted["source_coordinates"] = np.asarray(
            [[0.5, 0.5 + 5.0e-9], [np.nan, np.nan]]
        )
        coordinate_failed = compare_result_arrays(
            base,
            shifted,
            lower=np.asarray([0.0, 0.0]),
            upper=np.asarray([1000.0, 1.0]),
            residual_tolerance=1.0e-10,
            coordinate_tolerance=1.0e-11,
        )
        self.assertGreater(
            coordinate_failed[
                "maximum_full_span_normalized_coordinate_difference"
            ],
            1.0e-11,
        )
        self.assertFalse(coordinate_failed["parity_passed"])


if __name__ == "__main__":
    unittest.main()
