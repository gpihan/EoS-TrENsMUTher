"""Focused contracts for the V1 Phase-6 paired benchmark driver."""

import unittest

from tests._support.validate_v1_phase6_paired_benchmark import summarize_rows


def _route(seconds, *, subcell, residual=0, processed=0, subdivisions=0):
    return {
        "elapsed_seconds": seconds,
        "peak_rss_after_bytes": 1024,
        "production_solver_calls": 1,
        "status": "certified_no_root",
        "proof_status": "no_root_in_domain",
        "scan_complete": True,
        "localization_complete": True,
        "unresolved_boxes": 0,
        "unprocessed_candidate_cells": 0,
        "subcell_residual_box_count": residual,
        "boxes_processed": processed,
        "boxes_subdivided": subdivisions,
        "subcell": subcell,
    }


class Phase6SummaryTests(unittest.TestCase):
    def test_clean_pairs_satisfy_every_gate(self):
        complete_subcell = {
            "complete": True,
            "first_exclusion_depth": None,
        }
        rows = []
        for position, flat_index in enumerate((412906, 465165)):
            rows.append(
                {
                    "flat_index": flat_index,
                    "old": _route(10.0, subcell=None, processed=5000),
                    "new": _route(
                        1.0,
                        subcell=complete_subcell,
                        residual=4433,
                        processed=4433,
                    ),
                }
            )
        summary = summarize_rows(rows, expected_count=2)
        self.assertTrue(summary["acceptance"]["gate_satisfied"])
        self.assertEqual(summary["paired_status_and_proof_agreement"], 2)
        self.assertEqual(summary["new_krawczyk_evaluations"], 8866)

    def test_incomplete_certification_or_old_faster_fails_closed(self):
        rows = [
            {
                "flat_index": 412906,
                "old": _route(1.0, subcell=None, processed=4433),
                "new": _route(
                    2.0,
                    subcell={"complete": False, "first_exclusion_depth": None},
                    residual=4433,
                    processed=4433,
                ),
            },
            {
                "flat_index": 465165,
                "old": _route(1.0, subcell=None, processed=4433),
                "new": _route(
                    2.0,
                    subcell={"complete": True, "first_exclusion_depth": None},
                    residual=4433,
                    processed=4433,
                ),
            },
        ]
        summary = summarize_rows(rows, expected_count=2)
        self.assertFalse(summary["acceptance"]["gate_satisfied"])
        self.assertFalse(
            summary["acceptance"][
                "new_proof_safe_certified_no_root_400_of_400"
            ]
        )
        self.assertFalse(
            summary["acceptance"]["new_total_time_materially_below_old"]
        )


if __name__ == "__main__":
    unittest.main()
