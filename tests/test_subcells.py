"""Proof and topology tests for target-local dyadic subcells."""

import itertools
import unittest

import numpy as np

from eos_trensmuther.subcells import (
    SubcellBoxes,
    SubcellCompleteness,
    SubcellEnclosures,
    SubcellInputError,
    SubcellInvariantError,
    SubcellRefinementReport,
    SubcellRefinementResult,
    SubcellStopReason,
    refine_subcells,
)


def identity_enclosures(boxes):
    return SubcellEnclosures(
        boxes.unit_lower,
        boxes.unit_upper,
        np.ones(boxes.count, dtype=bool),
    )


class SubcellTopologyTests(unittest.TestCase):
    def test_children_tile_parent_in_c_order_for_dimensions_one_through_four(self):
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                parent = SubcellBoxes.original_cells([7], dimension)
                children = parent.dyadic_children()
                expected_offsets = np.asarray(
                    list(itertools.product((0, 1), repeat=dimension)),
                    dtype=float,
                )
                self.assertEqual(children.count, 2**dimension)
                self.assertEqual(children.depth, 1)
                np.testing.assert_array_equal(
                    children.cell_flat_indices,
                    np.full(2**dimension, 7),
                )
                np.testing.assert_array_equal(
                    children.unit_lower, expected_offsets / 2.0
                )
                np.testing.assert_array_equal(
                    children.unit_upper, (expected_offsets + 1.0) / 2.0
                )
                self.assertEqual(children.normalized_volume, 1.0)

                # Every corner is covered, and each non-boundary test point
                # lies in exactly one child: no gaps or positive-volume overlap.
                probes = np.asarray(
                    list(itertools.product((0.25, 0.75), repeat=dimension))
                )
                membership = np.all(
                    (probes[:, None, :] >= children.unit_lower[None, :, :])
                    & (probes[:, None, :] <= children.unit_upper[None, :, :]),
                    axis=2,
                )
                np.testing.assert_array_equal(
                    np.count_nonzero(membership, axis=1),
                    np.ones(probes.shape[0], dtype=int),
                )

    def test_boxes_are_canonical_unique_and_immutable(self):
        boxes = SubcellBoxes.original_cells([5, 2], 2)
        np.testing.assert_array_equal(boxes.cell_flat_indices, [2, 5])
        self.assertFalse(boxes.cell_flat_indices.flags.writeable)
        self.assertFalse(boxes.unit_lower.flags.writeable)
        with self.assertRaises(ValueError):
            boxes.unit_lower[0, 0] = 0.25
        with self.assertRaises(SubcellInputError):
            SubcellBoxes.original_cells([2, 2], 2)

    def test_closed_faces_corners_and_one_ulp_outside(self):
        corner = refine_subcells(
            [0.0, 0.0], [0], 2, identity_enclosures, maximum_depth=1
        )
        self.assertEqual(corner.boxes.count, 1)
        np.testing.assert_array_equal(corner.boxes.unit_lower, [[0.0, 0.0]])

        shared_face = refine_subcells(
            [0.5, 0.5], [0], 2, identity_enclosures, maximum_depth=1
        )
        self.assertEqual(shared_face.boxes.count, 4)

        outside = np.nextafter(np.float64(0.0), -np.inf)
        excluded = refine_subcells(
            [outside, 0.25], [0], 2, identity_enclosures, maximum_depth=1
        )
        self.assertTrue(excluded.certifies_no_root)
        self.assertEqual(excluded.report.first_exclusion_depth, 1)

    def test_depth_seven_retains_the_single_identity_interval(self):
        result = refine_subcells(
            [0.3], [4], 1, identity_enclosures, maximum_depth=7
        )
        self.assertTrue(result.report.complete)
        self.assertFalse(result.certifies_no_root)
        self.assertEqual(result.report.stop_reason, SubcellStopReason.MAXIMUM_DEPTH)
        self.assertEqual(result.boxes.depth, 7)
        self.assertEqual(result.boxes.count, 1)
        self.assertLessEqual(result.boxes.unit_lower[0, 0], 0.3)
        self.assertGreaterEqual(result.boxes.unit_upper[0, 0], 0.3)
        self.assertEqual(len(result.report.levels), 7)


class SubcellProofContractTests(unittest.TestCase):
    def test_complete_empty_union_is_the_only_no_root_certificate(self):
        def excluding(boxes):
            return SubcellEnclosures(
                np.full((boxes.count, boxes.dimension), 2.0),
                np.full((boxes.count, boxes.dimension), 3.0),
                np.ones(boxes.count, dtype=bool),
            )

        result = refine_subcells(
            [0.25, 0.25], [0, 1], 2, excluding, maximum_depth=7
        )
        self.assertTrue(result.certifies_no_root)
        self.assertEqual(result.boxes.count, 0)
        self.assertEqual(result.report.completed_depth, 1)
        self.assertEqual(result.report.total_boxes_evaluated, 8)
        self.assertEqual(result.report.stop_reason, SubcellStopReason.EMPTY_UNION)

    def test_undefined_enclosures_are_retained_conservatively(self):
        def undefined(boxes):
            shape = (boxes.count, boxes.dimension)
            return SubcellEnclosures(
                np.full(shape, np.nan),
                np.full(shape, np.nan),
                np.zeros(boxes.count, dtype=bool),
            )

        result = refine_subcells(
            [0.25, 0.25], [0], 2, undefined, maximum_depth=2
        )
        self.assertTrue(result.report.complete)
        self.assertFalse(result.certifies_no_root)
        self.assertEqual(result.boxes.count, 16)
        self.assertEqual(result.report.levels[0].undefined_boxes_retained, 4)
        self.assertEqual(result.report.levels[1].undefined_boxes_retained, 16)
        self.assertEqual(result.report.levels[1].original_volume_fraction, 1.0)

    def test_active_box_guard_returns_last_complete_union_without_proof(self):
        def retain_all(boxes):
            return SubcellEnclosures(
                np.full((boxes.count, boxes.dimension), -1.0),
                np.full((boxes.count, boxes.dimension), 1.0),
                np.ones(boxes.count, dtype=bool),
            )

        result = refine_subcells(
            [0.0, 0.0, 0.0, 0.0],
            [9],
            4,
            retain_all,
            maximum_depth=7,
            maximum_active_boxes=10,
        )
        self.assertEqual(result.report.completeness, SubcellCompleteness.INCOMPLETE)
        self.assertEqual(result.report.stop_reason, SubcellStopReason.ACTIVE_BOX_LIMIT)
        self.assertFalse(result.certifies_no_root)
        self.assertEqual(result.boxes.depth, 0)
        self.assertEqual(result.boxes.count, 1)
        self.assertFalse(result.report.levels[0].complete)
        self.assertIsNone(result.report.levels[0].original_volume_fraction)

    def test_incomplete_empty_result_is_rejected_by_invariant(self):
        empty = SubcellBoxes(
            dimension=1,
            depth=1,
            cell_flat_indices=np.empty(0, dtype=np.int64),
            unit_lower=np.empty((0, 1)),
            unit_upper=np.empty((0, 1)),
        )
        report = SubcellRefinementReport(
            dimension=1,
            requested_depth=1,
            completed_depth=1,
            completeness=SubcellCompleteness.INCOMPLETE,
            stop_reason=SubcellStopReason.ACTIVE_BOX_LIMIT,
            initial_cell_count=1,
            total_boxes_evaluated=1,
            first_exclusion_depth=None,
            levels=(),
        )
        with self.assertRaises(SubcellInvariantError):
            SubcellRefinementResult(empty, report)

    def test_malformed_enclosure_callback_fails_closed(self):
        def malformed(boxes):
            return SubcellEnclosures(
                np.zeros((boxes.count, boxes.dimension + 1)),
                np.ones((boxes.count, boxes.dimension + 1)),
                np.ones(boxes.count, dtype=bool),
            )

        with self.assertRaises(SubcellInvariantError):
            refine_subcells([0.5], [0], 1, malformed, maximum_depth=1)


if __name__ == "__main__":
    unittest.main()
