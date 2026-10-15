"""Topology and proof-safety tests for the private 4-D hierarchy builder."""

import itertools
import math
import unittest

import numpy as np

from eos_trensmuther.reachability_hierarchy import (
    ReachabilityHierarchyError,
    build_routing_hierarchy,
    classify_hierarchy_leaves,
    localize_hierarchy_leaves,
    profile_hierarchy_pruning,
)
from eos_trensmuther.reachability import ReachabilityConfig, _scan_target_boxes


def make_leaf_boxes(shape, seed=731):
    random = np.random.RandomState(seed)
    count = math.prod(shape)
    lower = random.normal(size=(count, 4))
    width = random.uniform(0.0, 0.5, size=(count, 4))
    upper = lower + width
    return lower, upper


class ExhaustiveOracleFreezeTests(unittest.TestCase):
    def test_closed_faces_shells_invalid_targets_and_sparse_ids_are_frozen(self):
        exact_lower = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        exact_upper = np.asarray([[1.0, 1.0], [2.0, 1.0]])
        routing_lower = np.nextafter(exact_lower - 0.1, -np.inf)
        routing_upper = np.nextafter(exact_upper + 0.1, np.inf)
        targets = np.asarray(
            [
                [1.0, 0.5],   # closed shared face: exact in both boxes
                [1.05, 1.05], # routing-shell-only in both boxes
                [1.05, 0.5],  # routing witness 4, exact witness 9
                [5.0, 5.0],   # outside both routing boxes
                [np.nan, 0.0],
            ]
        )
        valid = np.asarray([True, True, True, True, False])

        result = _scan_target_boxes(
            targets,
            valid,
            exact_lower,
            exact_upper,
            routing_lower,
            routing_upper,
            np.asarray([4, 9], dtype=np.int64),
            ReachabilityConfig(query_batch_size=1, cell_batch_size=1),
        )

        np.testing.assert_array_equal(
            result.routing_candidate, [True, True, True, False, False]
        )
        np.testing.assert_array_equal(
            result.exact_candidate, [True, False, True, False, False]
        )
        np.testing.assert_array_equal(
            result.routing_witness, [4, 4, 4, -1, -1]
        )
        np.testing.assert_array_equal(result.exact_witness, [4, -1, 9, -1, -1])


class RoutingHierarchyBuilderTests(unittest.TestCase):
    def test_balanced_uneven_partitions_cover_every_cell_once(self):
        shape = (5, 6, 7, 9)
        lower, upper = make_leaf_boxes(shape)
        hierarchy = build_routing_hierarchy(
            shape, lower, upper, hierarchy_levels=5
        )

        self.assertEqual(hierarchy.levels[0].block_shape, (1, 1, 1, 1))
        self.assertEqual(hierarchy.levels[-1].block_shape, shape)
        self.assertTrue(hierarchy.reaches_leaf)
        for level in hierarchy.levels:
            ownership = np.zeros(shape, dtype=np.uint8)
            for block in itertools.product(
                *(range(size) for size in level.block_shape)
            ):
                source_lower, source_upper = level.source_cell_bounds(block)
                slices = tuple(
                    slice(int(start), int(stop))
                    for start, stop in zip(source_lower, source_upper)
                )
                ownership[slices] += 1
            np.testing.assert_array_equal(ownership, np.ones(shape, dtype=np.uint8))
            for cells, boundaries in zip(shape, level.axis_boundaries):
                self.assertEqual(int(boundaries[0]), 0)
                self.assertEqual(int(boundaries[-1]), cells)
                self.assertTrue(np.all(np.diff(boundaries) > 0))

    def test_parent_bounds_outwardly_contain_every_child(self):
        shape = (5, 6, 7, 9)
        lower, upper = make_leaf_boxes(shape)
        hierarchy = build_routing_hierarchy(
            shape, lower, upper, hierarchy_levels=5
        )

        for parent, child in zip(hierarchy.levels[:-1], hierarchy.levels[1:]):
            for child_flat in range(child.block_count):
                child_index = np.asarray(
                    np.unravel_index(child_flat, child.block_shape, order="C")
                )
                source_lower, _ = child.source_cell_bounds(child_index)
                parent_index = np.asarray(
                    [
                        np.searchsorted(boundaries, source_lower[axis], side="right")
                        - 1
                        for axis, boundaries in enumerate(parent.axis_boundaries)
                    ]
                )
                parent_flat = np.ravel_multi_index(
                    tuple(parent_index), parent.block_shape, order="C"
                )
                self.assertTrue(
                    np.all(
                        parent.routing_lower[parent_flat]
                        <= child.routing_lower[child_flat]
                    )
                )
                self.assertTrue(
                    np.all(
                        parent.routing_upper[parent_flat]
                        >= child.routing_upper[child_flat]
                    )
                )

    def test_requested_depth_controls_retained_levels_and_memory_guard(self):
        shape = (9, 10, 11, 12)
        lower, upper = make_leaf_boxes(shape)
        hierarchy = build_routing_hierarchy(
            shape, lower, upper, hierarchy_levels=3
        )

        self.assertEqual(
            tuple(level.block_shape for level in hierarchy.levels),
            ((1, 1, 1, 1), (2, 2, 2, 2), (4, 4, 4, 4)),
        )
        self.assertFalse(hierarchy.reaches_leaf)
        self.assertGreater(hierarchy.owned_nbytes, 0)
        self.assertEqual(hierarchy.referenced_leaf_nbytes, 0)
        with self.assertRaisesRegex(
            ReachabilityHierarchyError, "max_hierarchy_bytes"
        ):
            build_routing_hierarchy(
                shape,
                lower,
                upper,
                hierarchy_levels=3,
                max_hierarchy_bytes=hierarchy.owned_nbytes - 1,
            )

    def test_leaf_level_alias_is_accounted_separately_and_arrays_are_immutable(self):
        shape = (5, 6, 7, 9)
        lower, upper = make_leaf_boxes(shape)
        lower.setflags(write=False)
        upper.setflags(write=False)
        hierarchy = build_routing_hierarchy(
            shape, lower, upper, hierarchy_levels=7
        )

        leaf = hierarchy.levels[-1]
        self.assertTrue(leaf.aliases_leaf)
        self.assertEqual(leaf.block_shape, shape)
        self.assertEqual(
            hierarchy.referenced_leaf_nbytes, lower.nbytes + upper.nbytes
        )
        np.testing.assert_array_equal(leaf.routing_lower, lower)
        np.testing.assert_array_equal(leaf.routing_upper, upper)
        self.assertTrue(np.shares_memory(leaf.routing_lower, lower))
        self.assertTrue(np.shares_memory(leaf.routing_upper, upper))
        for level in hierarchy.levels:
            self.assertFalse(level.routing_lower.flags.writeable)
            self.assertFalse(level.routing_upper.flags.writeable)
            self.assertTrue(
                all(not boundary.flags.writeable for boundary in level.axis_boundaries)
            )

    def test_invalid_inputs_fail_before_constructing_a_hierarchy(self):
        lower = np.zeros((16, 4))
        upper = np.ones((16, 4))
        with self.assertRaises(ReachabilityHierarchyError):
            build_routing_hierarchy(
                (2, 2, 2), lower, upper, hierarchy_levels=3
            )
        with self.assertRaises(ReachabilityHierarchyError):
            build_routing_hierarchy(
                (2, 2, 2, 2), lower, upper, hierarchy_levels=4
            )
        with self.assertRaises(ReachabilityHierarchyError):
            build_routing_hierarchy(
                (2, 2, 2, 2), lower[:-1], upper[:-1], hierarchy_levels=3
            )


class RoutingHierarchyProfilingTests(unittest.TestCase):
    def test_forward_targets_keep_a_branch_and_obvious_exclusion_is_rejected(self):
        shape = (5, 6, 7, 9)
        lower, upper = make_leaf_boxes(shape)
        hierarchy = build_routing_hierarchy(
            shape, lower, upper, hierarchy_levels=3
        )
        selected = np.asarray([0, 41, math.prod(shape) - 1])
        targets = np.vstack(
            [0.5 * (lower[selected] + upper[selected]), np.full((1, 4), 1.0e6)]
        )

        report = profile_hierarchy_pruning(
            hierarchy,
            targets,
            maximum_estimated_leaf_seconds=100.0,
            leaf_pairs_per_second=1.0e6,
            query_batch_size=2,
            maximum_pairwise_entries=1000,
        )

        self.assertTrue(report.traversal_completed)
        self.assertFalse(report.complete_classification)
        self.assertEqual(report.profiled_valid_target_count, 4)
        self.assertEqual(report.targets_rejected_before_leaf, 1)
        self.assertTrue(
            np.all(report.descendant_leaf_pairs_by_profiled_target[:3] > 0)
        )
        self.assertEqual(
            int(report.descendant_leaf_pairs_by_profiled_target[3]), 0
        )
        self.assertEqual(len(report.level_profiles), 3)

    def test_profile_counts_are_query_batch_invariant(self):
        shape = (5, 6, 7, 9)
        lower, upper = make_leaf_boxes(shape)
        hierarchy = build_routing_hierarchy(
            shape, lower, upper, hierarchy_levels=5
        )
        indices = np.arange(0, math.prod(shape), 37)
        targets = 0.5 * (lower[indices] + upper[indices])

        reports = [
            profile_hierarchy_pruning(
                hierarchy,
                targets,
                maximum_estimated_leaf_seconds=100.0,
                leaf_pairs_per_second=1.0e6,
                query_batch_size=batch_size,
                maximum_pairwise_entries=10000,
            )
            for batch_size in (1, 7, 128)
        ]

        expected = reports[0]
        for report in reports[1:]:
            np.testing.assert_array_equal(
                report.descendant_leaf_pairs_by_profiled_target,
                expected.descendant_leaf_pairs_by_profiled_target,
            )
            self.assertEqual(report.descendant_leaf_pairs, expected.descendant_leaf_pairs)
            self.assertEqual(
                [item.tested_pairs for item in report.level_profiles],
                [item.tested_pairs for item in expected.level_profiles],
            )
            self.assertEqual(
                [item.surviving_pairs for item in report.level_profiles],
                [item.surviving_pairs for item in expected.level_profiles],
            )

    def test_completion_ceiling_is_explicit_and_benchmark_only(self):
        shape = (3, 3, 3, 3)
        lower = np.zeros((math.prod(shape), 4))
        upper = np.ones((math.prod(shape), 4))
        hierarchy = build_routing_hierarchy(
            shape, lower, upper, hierarchy_levels=3
        )

        report = profile_hierarchy_pruning(
            hierarchy,
            np.full((1, 4), 0.5),
            maximum_estimated_leaf_seconds=0.0,
            leaf_pairs_per_second=1.0,
        )

        self.assertTrue(report.traversal_completed)
        self.assertFalse(report.leaf_completion_allowed)
        self.assertFalse(report.complete_classification)
        self.assertEqual(
            report.stopped_reason,
            "estimated_leaf_seconds_exceeds_configured_limit",
        )
        self.assertEqual(report.maximum_estimated_leaf_seconds, 0.0)
        self.assertIsNone(report.classification_leaf_pairs)
        self.assertIsNone(report.additional_minimum_witness_leaf_pairs)
        with self.assertRaises(ReachabilityHierarchyError):
            profile_hierarchy_pruning(
                hierarchy,
                np.full((1, 4), 0.5),
                maximum_estimated_leaf_seconds=-1.0,
                leaf_pairs_per_second=1.0,
            )

    def test_one_target_pair_overflow_stops_without_classifying(self):
        shape = (9, 9, 9, 9)
        lower = np.zeros((math.prod(shape), 4))
        upper = np.ones((math.prod(shape), 4))
        hierarchy = build_routing_hierarchy(
            shape, lower, upper, hierarchy_levels=5
        )

        report = profile_hierarchy_pruning(
            hierarchy,
            np.full((2, 4), 0.5),
            maximum_estimated_leaf_seconds=100.0,
            leaf_pairs_per_second=1.0e6,
            maximum_pairwise_entries=10,
        )

        self.assertFalse(report.traversal_completed)
        self.assertFalse(report.complete_classification)
        self.assertFalse(report.leaf_completion_allowed)
        self.assertEqual(report.profiled_valid_target_count, 0)
        self.assertEqual(
            report.stopped_reason,
            "maximum_pairwise_entries_exceeded_by_one_target",
        )

    def test_categories_and_invalid_targets_are_reported_separately(self):
        shape = (3, 3, 3, 3)
        lower, upper = make_leaf_boxes(shape)
        hierarchy = build_routing_hierarchy(
            shape, lower, upper, hierarchy_levels=3
        )
        targets = np.vstack(
            [
                0.5 * (lower[0] + upper[0]),
                np.full(4, 1.0e8),
                np.full(4, np.nan),
            ]
        )

        report = profile_hierarchy_pruning(
            hierarchy,
            targets,
            maximum_estimated_leaf_seconds=100.0,
            leaf_pairs_per_second=1.0e6,
            categories=("reachable", "outside", "invalid"),
        )

        self.assertEqual(report.target_count, 3)
        self.assertEqual(report.valid_target_count, 2)
        self.assertEqual(set(report.category_profiles), {"outside", "reachable"})
        self.assertEqual(report.category_profiles["outside"].rejected_before_leaf, 1)
        self.assertEqual(report.category_profiles["reachable"].requiring_leaf, 1)
        self.assertIn("maximum_estimated_leaf_seconds", report.to_dict())


class RoutingHierarchyLeafResolutionTests(unittest.TestCase):
    def assert_matches_oracle(
        self,
        shape,
        exact_lower,
        exact_upper,
        routing_lower,
        routing_upper,
        targets,
        valid,
        *,
        hierarchy_levels,
        query_batch_size,
        leaf_batch_size,
    ):
        hierarchy = build_routing_hierarchy(
            shape,
            routing_lower,
            routing_upper,
            hierarchy_levels=hierarchy_levels,
        )
        localization = localize_hierarchy_leaves(
            hierarchy,
            targets,
            exact_lower,
            exact_upper,
            routing_lower,
            routing_upper,
            valid_mask=valid,
            query_batch_size=query_batch_size,
            leaf_batch_size=leaf_batch_size,
            maximum_pairwise_entries=max(leaf_batch_size, 64),
        )
        result = localization.result
        oracle = _scan_target_boxes(
            np.asarray(targets, dtype=float),
            np.asarray(valid, dtype=bool) & np.all(np.isfinite(targets), axis=1),
            np.asarray(exact_lower),
            np.asarray(exact_upper),
            np.asarray(routing_lower),
            np.asarray(routing_upper),
            np.arange(math.prod(shape), dtype=np.int64),
            ReachabilityConfig(query_batch_size=3, cell_batch_size=7),
        )
        np.testing.assert_array_equal(
            result.routing_candidate, oracle.routing_candidate
        )
        np.testing.assert_array_equal(result.exact_candidate, oracle.exact_candidate)
        np.testing.assert_array_equal(result.routing_witness, oracle.routing_witness)
        np.testing.assert_array_equal(result.exact_witness, oracle.exact_witness)

        target_values = np.asarray(targets, dtype=float)
        evaluated = np.asarray(valid, dtype=bool) & np.all(
            np.isfinite(target_values), axis=1
        )
        expected_offsets = [0]
        expected_cells = []
        for target, is_valid in zip(target_values, evaluated):
            if is_valid:
                contains = np.all(
                    (target >= np.asarray(routing_lower))
                    & (target <= np.asarray(routing_upper)),
                    axis=1,
                )
                row = np.flatnonzero(contains).astype(np.int64, copy=False)
                expected_cells.extend(row.tolist())
            expected_offsets.append(len(expected_cells))
        np.testing.assert_array_equal(localization.offsets, expected_offsets)
        np.testing.assert_array_equal(
            localization.cell_flat_indices,
            np.asarray(expected_cells, dtype=np.int64),
        )
        self.assertFalse(localization.offsets.flags.writeable)
        self.assertFalse(localization.cell_flat_indices.flags.writeable)
        self.assertEqual(
            localization.pair_count,
            int(localization.offsets[-1]),
        )
        for target_index in range(target_values.shape[0]):
            start = expected_offsets[target_index]
            stop = expected_offsets[target_index + 1]
            np.testing.assert_array_equal(
                localization.cells_for(target_index),
                np.asarray(expected_cells[start:stop], dtype=np.int64),
            )
        return result

    def test_randomized_folded_boxes_match_every_oracle_column_at_all_depths(self):
        random = np.random.RandomState(1907)
        shape = (5, 5, 5, 5)
        count = math.prod(shape)
        centers = random.uniform(-1.0, 1.0, size=(count, 4))
        widths = random.uniform(0.02, 0.35, size=(count, 4))
        exact_lower = centers - widths
        exact_upper = centers + widths
        padding = random.uniform(0.0, 0.08, size=(count, 4))
        routing_lower = exact_lower - padding
        routing_upper = exact_upper + padding
        selected = np.asarray([0, 1, 18, 211, count - 1])
        targets = np.vstack(
            [
                centers[selected],
                random.uniform(-1.5, 1.5, size=(30, 4)),
                exact_upper[0],
                np.nextafter(exact_upper[1], np.inf),
                np.full(4, 10.0),
                np.full(4, np.nan),
            ]
        )
        valid = np.ones(targets.shape[0], dtype=bool)
        valid[3] = False

        references = []
        for depth in (3, 5, 7):
            result = self.assert_matches_oracle(
                shape,
                exact_lower,
                exact_upper,
                routing_lower,
                routing_upper,
                targets,
                valid,
                hierarchy_levels=depth,
                query_batch_size=1 if depth == 3 else 11,
                leaf_batch_size=1 if depth == 5 else 13,
            )
            references.append(result)
        for result in references[1:]:
            np.testing.assert_array_equal(
                result.routing_witness, references[0].routing_witness
            )
            np.testing.assert_array_equal(
                result.exact_witness, references[0].exact_witness
            )

    def test_faces_tolerance_shell_gaps_and_distinct_witnesses_match_oracle(self):
        shape = (2, 2, 2, 2)
        count = math.prod(shape)
        exact_lower = np.full((count, 4), 50.0)
        exact_upper = np.full((count, 4), 51.0)
        routing_lower = exact_lower.copy()
        routing_upper = exact_upper.copy()
        exact_lower[0], exact_upper[0] = 0.0, 1.0
        routing_lower[0], routing_upper[0] = -0.1, 1.1
        exact_lower[1] = np.asarray([1.0, 0.0, 0.0, 0.0])
        exact_upper[1] = np.asarray([2.0, 1.0, 1.0, 1.0])
        routing_lower[1] = exact_lower[1] - 0.1
        routing_upper[1] = exact_upper[1] + 0.1
        targets = np.asarray(
            [
                [1.0, 0.5, 0.5, 0.5],
                [1.05, 1.05, 1.05, 1.05],
                [1.05, 0.5, 0.5, 0.5],
                [25.0, 25.0, 25.0, 25.0],
                np.nextafter(np.ones(4), np.inf),
            ]
        )
        result = self.assert_matches_oracle(
            shape,
            exact_lower,
            exact_upper,
            routing_lower,
            routing_upper,
            targets,
            np.ones(targets.shape[0], dtype=bool),
            hierarchy_levels=3,
            query_batch_size=2,
            leaf_batch_size=3,
        )
        self.assertEqual(int(result.routing_witness[0]), 0)
        self.assertEqual(int(result.exact_witness[0]), 0)
        self.assertTrue(result.routing_candidate[1])
        self.assertFalse(result.exact_candidate[1])
        self.assertEqual(int(result.routing_witness[2]), 0)
        self.assertEqual(int(result.exact_witness[2]), 1)
        self.assertFalse(result.routing_candidate[3])

    def test_witness_overhead_is_counted_without_changing_minimum_identity(self):
        shape = (5, 5, 5, 5)
        count = math.prod(shape)
        exact_lower = np.zeros((count, 4))
        exact_upper = np.ones((count, 4))
        routing_lower = np.full((count, 4), -0.1)
        routing_upper = np.full((count, 4), 1.1)
        result = self.assert_matches_oracle(
            shape,
            exact_lower,
            exact_upper,
            routing_lower,
            routing_upper,
            np.full((1, 4), 0.5),
            np.ones(1, dtype=bool),
            hierarchy_levels=3,
            query_batch_size=1,
            leaf_batch_size=17,
        )

        self.assertEqual(int(result.routing_witness[0]), 0)
        self.assertEqual(int(result.exact_witness[0]), 0)
        self.assertEqual(result.diagnostics.actual_leaf_pairs, count)
        self.assertEqual(result.diagnostics.classification_leaf_pairs, 1)
        self.assertEqual(
            result.diagnostics.additional_minimum_witness_leaf_pairs,
            count - 1,
        )

    def test_leaf_level_vector_batches_separate_witness_and_batch_overhead(self):
        shape = (2, 2, 2, 2)
        count = math.prod(shape)
        exact_lower = np.zeros((count, 4))
        exact_upper = np.ones((count, 4))
        routing_lower = np.full((count, 4), -0.1)
        routing_upper = np.full((count, 4), 1.1)
        result = self.assert_matches_oracle(
            shape,
            exact_lower,
            exact_upper,
            routing_lower,
            routing_upper,
            np.asarray([[0.5, 0.5, 0.5, 0.5], [1.05, 1.05, 1.05, 1.05]]),
            np.ones(2, dtype=bool),
            hierarchy_levels=3,
            query_batch_size=2,
            leaf_batch_size=8,
        )

        np.testing.assert_array_equal(
            result.diagnostics.classification_leaf_pairs_by_target,
            np.asarray([1, 1]),
        )
        np.testing.assert_array_equal(
            result.diagnostics.additional_minimum_witness_leaf_pairs_by_target,
            np.asarray([0, count - 1]),
        )
        np.testing.assert_array_equal(
            result.diagnostics.additional_batch_leaf_pairs_by_target,
            np.asarray([count - 1, 0]),
        )

    def test_leaf_resolution_rejects_an_unbounded_temporary_batch(self):
        shape = (2, 2, 2, 2)
        lower = np.zeros((math.prod(shape), 4))
        upper = np.ones((math.prod(shape), 4))
        hierarchy = build_routing_hierarchy(
            shape, lower, upper, hierarchy_levels=3
        )
        with self.assertRaisesRegex(
            ReachabilityHierarchyError, "leaf_batch_size"
        ):
            classify_hierarchy_leaves(
                hierarchy,
                np.full((1, 4), 0.5),
                lower,
                upper,
                lower,
                upper,
                leaf_batch_size=65,
                maximum_pairwise_entries=64,
            )

    def test_localization_is_batch_invariant_and_profile_reports_stay_non_authoritative(self):
        shape = (3, 3, 3, 3)
        count = math.prod(shape)
        exact_lower = np.zeros((count, 4))
        exact_upper = np.ones((count, 4))
        routing_lower = np.full((count, 4), -0.1)
        routing_upper = np.full((count, 4), 1.1)
        targets = np.asarray(
            [
                [0.5, 0.5, 0.5, 0.5],
                [1.05, 1.05, 1.05, 1.05],
                [5.0, 5.0, 5.0, 5.0],
                [np.nan, 0.0, 0.0, 0.0],
            ]
        )
        hierarchy = build_routing_hierarchy(
            shape, routing_lower, routing_upper, hierarchy_levels=7
        )
        localizations = [
            localize_hierarchy_leaves(
                hierarchy,
                targets,
                exact_lower,
                exact_upper,
                routing_lower,
                routing_upper,
                query_batch_size=query_batch,
                leaf_batch_size=leaf_batch,
                maximum_pairwise_entries=maximum_pairs,
            )
            for query_batch, leaf_batch, maximum_pairs in (
                (1, 1, 64),
                (4, 7, 128),
                (2, 13, 256),
            )
        ]
        classification = classify_hierarchy_leaves(
            hierarchy,
            targets,
            exact_lower,
            exact_upper,
            routing_lower,
            routing_upper,
            query_batch_size=4,
            leaf_batch_size=7,
            maximum_pairwise_entries=128,
        )
        for name in (
            "valid",
            "routing_candidate",
            "exact_candidate",
            "routing_witness",
            "exact_witness",
        ):
            np.testing.assert_array_equal(
                getattr(localizations[0].result, name),
                getattr(classification, name),
            )
        for localization in localizations[1:]:
            np.testing.assert_array_equal(
                localization.offsets, localizations[0].offsets
            )
            np.testing.assert_array_equal(
                localization.cell_flat_indices,
                localizations[0].cell_flat_indices,
            )
            np.testing.assert_array_equal(
                localization.result.routing_witness,
                localizations[0].result.routing_witness,
            )

        report = profile_hierarchy_pruning(
            hierarchy,
            targets,
            maximum_estimated_leaf_seconds=0.0,
            leaf_pairs_per_second=1.0,
        )
        self.assertFalse(report.complete_classification)
        self.assertFalse(hasattr(report, "cell_flat_indices"))
        self.assertFalse(hasattr(report, "offsets"))

    def test_replayed_localization_after_pair_limit_splitting_is_duplicate_free(self):
        """A split query batch must retain unique global query/leaf pairs."""

        shape = (3, 3, 3, 3)
        leaf_count = math.prod(shape)
        exact_lower = np.zeros((leaf_count, 4))
        exact_upper = np.ones((leaf_count, 4))
        hierarchy = build_routing_hierarchy(
            shape,
            exact_lower,
            exact_upper,
            hierarchy_levels=7,
        )
        # Seven targets produce 567 terminal pairs.  The 128-pair guard forces
        # _classify_full_leaf_level to split the query batch recursively, while
        # one target's 81 terminal pairs still fit and complete authoritatively.
        targets = np.full((7, 4), 0.5)

        localizations = [
            localize_hierarchy_leaves(
                hierarchy,
                targets,
                exact_lower,
                exact_upper,
                exact_lower,
                exact_upper,
                query_batch_size=7,
                leaf_batch_size=64,
                maximum_pairwise_entries=128,
            )
            for _ in range(2)
        ]

        expected_offsets = np.arange(8, dtype=np.int64) * leaf_count
        expected_cells = np.tile(np.arange(leaf_count, dtype=np.int64), 7)
        for localization in localizations:
            np.testing.assert_array_equal(localization.offsets, expected_offsets)
            np.testing.assert_array_equal(
                localization.cell_flat_indices, expected_cells
            )
            for target_index in range(targets.shape[0]):
                row = localization.cells_for(target_index)
                self.assertTrue(np.all(np.diff(row) > 0))
        np.testing.assert_array_equal(
            localizations[1].offsets, localizations[0].offsets
        )
        np.testing.assert_array_equal(
            localizations[1].cell_flat_indices,
            localizations[0].cell_flat_indices,
        )


if __name__ == "__main__":
    unittest.main()
