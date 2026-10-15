"""Lightweight contract tests for the restartable Phase-11 driver."""

from __future__ import annotations

import unittest

from tests._support.validate_v1_phase11_release_candidate import (
    CHUNK_SIZE,
    GRID_COUNT,
    GRID_SIZE,
    build_config,
    route_coverage,
)


class Phase11ReleaseCandidateDriverTests(unittest.TestCase):
    def test_frozen_medium_configuration(self):
        config = build_config(
            workers=4, worker_task_size=16, max_pending_tasks=8
        )
        self.assertEqual(
            tuple(axis.count for axis in config.grid.axes.values()),
            (GRID_COUNT,) * 4,
        )
        self.assertEqual(GRID_SIZE, 10_000)
        self.assertEqual(config.execution.chunk_size, CHUNK_SIZE)
        self.assertEqual(config.execution.workers, 4)
        self.assertEqual(config.execution.worker_task_size, 16)
        self.assertEqual(config.execution.max_pending_chunks, 8)
        self.assertEqual(config.auxiliary_fields, ("P", "s"))
        self.assertTrue(config.solver.enable_subcell_fallback)
        self.assertEqual(config.solver.max_subcell_depth, 7)
        self.assertFalse(config.projection.enabled)

    def test_interruption_and_resume_change_execution_only_settings(self):
        initial = build_config(
            workers=2, worker_task_size=8, max_pending_tasks=4
        )
        resumed = build_config(
            workers=4, worker_task_size=16, max_pending_tasks=8
        )
        initial_dict = initial.to_dict()
        resumed_dict = resumed.to_dict()
        initial_execution = initial_dict.pop("execution")
        resumed_execution = resumed_dict.pop("execution")
        self.assertEqual(initial_dict, resumed_dict)
        self.assertEqual(initial_execution["chunk_size"], CHUNK_SIZE)
        self.assertEqual(resumed_execution["chunk_size"], CHUNK_SIZE)
        self.assertNotEqual(
            initial_execution["workers"], resumed_execution["workers"]
        )
        self.assertNotEqual(
            initial_execution["worker_task_size"],
            resumed_execution["worker_task_size"],
        )

    def test_route_coverage_requires_each_release_candidate_path(self):
        counters = {
            "prescreen_excluded": 1,
            "fast_attempted": 2,
            "fallback_attempted": 1,
            "subcell_fallback_attempted": 1,
            "subcell_certified_no_root": 1,
            "subcell_residual_krawczyk": 0,
            "subcell_boxes_evaluated": 8,
            "subcell_residual_boxes": 0,
            "subcell_krawczyk_boxes_processed": 0,
        }
        counters.update(
            {"subcell_certificate_depth_%d" % depth: 0 for depth in range(1, 8)}
        )
        counters["subcell_certificate_depth_3"] = 1
        manifest = {
            "stage_counters": counters,
            "status_summary": {"route": {"fast": 1}},
        }
        report = route_coverage(manifest)
        self.assertTrue(report["all_required_routes_observed"])
        self.assertEqual(report["subcell_certificate_depth_counts"]["3"], 1)
        manifest["stage_counters"]["subcell_certified_no_root"] = 0
        self.assertFalse(
            route_coverage(manifest)["all_required_routes_observed"]
        )


if __name__ == "__main__":
    unittest.main()
