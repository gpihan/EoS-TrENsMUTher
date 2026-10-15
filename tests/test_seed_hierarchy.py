"""Focused regression tests for production inverse-grid seed precedence."""

import itertools
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from eos_trensmuther import (
    CoarseSeedConfig,
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InitialGuessSource,
    InversionGridSpec,
    LocalizationCompleteness,
    PipelineConfig,
    PipelineStateError,
    PointDisposition,
    ProjectionConfig,
    ReachabilityConfig,
    ReachabilityLocalizationError,
    SolveStatus,
    SolveStrategy,
)

from tests.test_tables import TemporaryEoS, field_spec, metadata


class SeedHierarchyTests(unittest.TestCase):
    def make_identity_problem(self, *, coarse_enabled):
        coordinate_names = ("T", "muB", "muQ", "muS")
        target_names = ("e", "nB", "nQ", "nS")
        fields = {
            name: field_spec("coordinate", index)
            for index, name in enumerate(coordinate_names)
        }
        fields.update(
            {
                name: field_spec("target", index + 4)
                for index, name in enumerate(target_names)
            }
        )
        axes = (
            (0.2, 0.8),
            (0.0, 0.6),
            (-0.3, 0.3),
            (-0.4, 0.4),
        )
        rows = [list(point) + list(point) for point in itertools.product(*axes)]
        fixture = TemporaryEoS(metadata(fields, 4), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        envelope = table.find_boundaries(space="target")
        config = PipelineConfig(
            grid=InversionGridSpec(
                "target",
                {name: GridAxisSpec(3) for name in envelope.names},
            ),
            residual_scales={name: 1.0 for name in target_names},
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(workers=1, chunk_size=9),
            coarse_seed=CoarseSeedConfig(
                enabled=coarse_enabled,
                points_per_axis=2,
            ),
            auxiliary_fields=(),
        )
        pipeline = EoSInversionPipeline(table, config)
        pipeline._prepare()
        return pipeline

    def make_multicell_identity_problem(
        self,
        dimension,
        *,
        coarse_enabled=False,
        hierarchical=False,
        use_reachability_seed=True,
    ):
        coordinate_names = ("T", "muB", "muQ", "muS")[:dimension]
        target_names = ("e", "nB", "nQ", "nS")[:dimension]
        fields = {
            name: field_spec("coordinate", index)
            for index, name in enumerate(coordinate_names)
        }
        fields.update(
            {
                name: field_spec("target", index + dimension)
                for index, name in enumerate(target_names)
            }
        )
        axes = tuple((0.0, 1.0, 2.0) for _ in range(dimension))
        rows = [list(point) + list(point) for point in itertools.product(*axes)]
        fixture = TemporaryEoS(metadata(fields, dimension), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        envelope = table.find_boundaries(space="target")
        config = PipelineConfig(
            grid=InversionGridSpec(
                "target",
                {name: GridAxisSpec(3) for name in envelope.names},
            ),
            residual_scales={name: 1.0 for name in target_names},
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(workers=1, chunk_size=9),
            coarse_seed=CoarseSeedConfig(
                enabled=coarse_enabled,
                points_per_axis=2,
            ),
            use_reachability_seed=use_reachability_seed,
            auxiliary_fields=(),
        )
        reachability = (
            ReachabilityConfig(method="hierarchical", hierarchy_levels=7)
            if hierarchical
            else ReachabilityConfig(method="exhaustive")
        )
        pipeline = EoSInversionPipeline(
            table,
            config,
            reachability_config=reachability,
        )
        pipeline._prepare()
        return pipeline

    def make_correlated_no_root_problem(
        self, dimension, *, use_reachability_seed=True
    ):
        coordinate_names = ("T", "muB", "muQ", "muS")[:dimension]
        target_names = ("e", "nB", "nQ", "nS")[:dimension]
        fields = {
            name: field_spec("coordinate", index)
            for index, name in enumerate(coordinate_names)
        }
        fields.update(
            {
                name: field_spec("target", index + dimension)
                for index, name in enumerate(target_names)
            }
        )
        axes = ((0.0, 1.0),) * dimension
        rows = []
        for point in itertools.product(*axes):
            first, second = point[:2]
            rows.append(
                list(point)
                + [first + second, first - second]
                + list(point[2:])
            )
        fixture = TemporaryEoS(metadata(fields, dimension), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        grid_axes = {
            "e": GridAxisSpec(2, 0.24, 0.26),
            "nB": GridAxisSpec(2, 0.74, 0.76),
        }
        grid_axes.update(
            {name: GridAxisSpec(2, 0.4, 0.6) for name in target_names[2:]}
        )
        config = PipelineConfig(
            grid=InversionGridSpec(
                "target",
                grid_axes,
            ),
            residual_scales={name: 1.0 for name in target_names},
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(
                workers=1, chunk_size=2**dimension
            ),
            coarse_seed=CoarseSeedConfig(enabled=False),
            use_reachability_seed=use_reachability_seed,
            auxiliary_fields=(),
        )
        pipeline = EoSInversionPipeline(
            table,
            config,
            reachability_config=(
                ReachabilityConfig(
                    method="hierarchical", hierarchy_levels=7
                )
                if dimension == 4
                else ReachabilityConfig(method="exhaustive")
            ),
        )
        pipeline._prepare()
        return pipeline

    @staticmethod
    def flat(pipeline, multi):
        return int(np.ravel_multi_index(multi, pipeline.plan().grid.shape, order="C"))

    def test_direct_solved_neighbor_has_deterministic_priority(self):
        pipeline = self.make_identity_problem(coarse_enabled=True)
        target = (1, 1, 1, 1)
        axis_zero_plus = self.flat(pipeline, (2, 1, 1, 1))
        axis_three_minus = self.flat(pipeline, (1, 1, 1, 0))
        preferred = np.asarray([0.7, 0.4, 0.0, 0.0])
        other = np.asarray([0.5, 0.3, 0.0, -0.2])
        pipeline._register_solved_seed(axis_three_minus, other)
        pipeline._register_solved_seed(axis_zero_plus, preferred)

        seed, source, direction = pipeline._select_seed(
            self.flat(pipeline, target)
        )

        self.assertIs(source, InitialGuessSource.DIRECT_NEIGHBOR)
        self.assertEqual(direction, "axis0_plus")
        np.testing.assert_array_equal(seed, preferred)

    def test_nonadjacent_solved_point_is_not_searched(self):
        pipeline = self.make_identity_problem(coarse_enabled=False)
        pipeline._register_solved_seed(
            self.flat(pipeline, (0, 0, 0, 0)),
            np.asarray([0.2, 0.0, -0.3, -0.4]),
        )

        seed, source, direction = pipeline._select_seed(
            self.flat(pipeline, (2, 2, 2, 2))
        )

        self.assertIs(source, InitialGuessSource.DOMAIN_MIDPOINT)
        self.assertIsNone(direction)
        np.testing.assert_allclose(seed, [0.5, 0.3, 0.0, 0.0])

    def test_coarse_seed_is_used_without_direct_neighbor(self):
        pipeline = self.make_identity_problem(coarse_enabled=True)

        seed, source, direction = pipeline._select_seed(
            self.flat(pipeline, (1, 1, 1, 1))
        )

        self.assertIs(source, InitialGuessSource.COARSE_LEVEL)
        self.assertIsNone(direction)
        self.assertTrue(np.all(np.isfinite(seed)))
        self.assertGreater(pipeline._coarse_seeds.size, 0)

    def test_generic_seed_is_used_when_neighbor_and_coarse_are_unavailable(self):
        pipeline = self.make_identity_problem(coarse_enabled=False)

        seed, source, direction = pipeline._select_seed(
            self.flat(pipeline, (1, 1, 1, 1))
        )

        self.assertIs(source, InitialGuessSource.DOMAIN_MIDPOINT)
        self.assertIsNone(direction)
        np.testing.assert_allclose(seed, [0.5, 0.3, 0.0, 0.0])

    def test_only_solved_dispositions_enter_neighbor_cache(self):
        pipeline = self.make_identity_problem(coarse_enabled=False)
        neighbor = self.flat(pipeline, (0, 1, 1, 1))
        point = np.asarray([0.2, 0.3, 0.0, 0.0])
        rejected = (
            PointDisposition.PROJECTED,
            PointDisposition.CERTIFIED_UNREACHABLE,
            PointDisposition.ROOT_CERTIFIED_UNREFINED,
            PointDisposition.UNRESOLVED,
            PointDisposition.INVALID_TARGET,
            PointDisposition.PROJECTION_FAILED,
            PointDisposition.ERROR,
        )
        for disposition in rejected:
            with self.subTest(disposition=disposition):
                self.assertFalse(
                    pipeline._register_solved_seed(
                        neighbor, point, disposition=disposition
                    )
                )

        _, source, _ = pipeline._select_seed(
            self.flat(pipeline, (1, 1, 1, 1))
        )
        self.assertIs(source, InitialGuessSource.DOMAIN_MIDPOINT)

    def test_different_valid_seeds_reach_same_smooth_4d_inverse(self):
        pipeline = self.make_identity_problem(coarse_enabled=False)
        target = np.asarray([0.37, 0.41, -0.11, 0.17])
        first = pipeline._solver.solve(
            target,
            np.asarray([0.2, 0.0, -0.3, -0.4]),
            strategy=SolveStrategy.FAST,
        )
        second = pipeline._solver.solve(
            target,
            np.asarray([0.8, 0.6, 0.3, 0.4]),
            strategy=SolveStrategy.FAST,
        )

        self.assertIs(first.status, SolveStatus.CONVERGED_FAST)
        self.assertIs(second.status, SolveStatus.CONVERGED_FAST)
        np.testing.assert_allclose(first.solution, target, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(second.solution, target, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            first.solution, second.solution, rtol=0.0, atol=1.0e-12
        )

    def test_candidate_centers_are_vectorized_and_c_order_ties_are_deterministic(self):
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension, backend="exhaustive"):
                pipeline = self.make_multicell_identity_problem(dimension)
                target = np.ones(dimension)
                localization = pipeline._reachability.localize([target])
                seed = pipeline._candidate_center_seed(target, localization, 0)

                self.assertEqual(seed.candidate_count, 2**dimension)
                self.assertEqual(seed.cell_flat_index, 0)
                self.assertAlmostEqual(seed.scaled_residual_norm_inf, 0.5)
                np.testing.assert_array_equal(
                    seed.point, np.full(dimension, 0.5)
                )
                self.assertFalse(seed.point.flags.writeable)
                self.assertGreaterEqual(seed.scoring_seconds, 0.0)
                self.assertEqual(
                    seed.diagnostics()["tie_break"],
                    "smallest_global_c_order_cell_id",
                )

        exhaustive = self.make_multicell_identity_problem(4)
        hierarchical = self.make_multicell_identity_problem(
            4, hierarchical=True
        )
        target = np.ones(4)
        exhaustive_seed = exhaustive._candidate_center_seed(
            target, exhaustive._reachability.localize([target]), 0
        )
        hierarchical_seed = hierarchical._candidate_center_seed(
            target, hierarchical._reachability.localize([target]), 0
        )
        self.assertEqual(
            hierarchical_seed.cell_flat_index,
            exhaustive_seed.cell_flat_index,
        )
        self.assertEqual(
            hierarchical_seed.candidate_count,
            exhaustive_seed.candidate_count,
        )
        np.testing.assert_array_equal(
            hierarchical_seed.point, exhaustive_seed.point
        )

        class CountingInterpolator:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.evaluate_calls = []

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def evaluate(self, points):
                supplied = np.asarray(points, dtype=float)
                self.evaluate_calls.append(supplied.copy())
                return self.wrapped.evaluate(points)

        counted = CountingInterpolator(exhaustive._target_interpolator)
        exhaustive._target_interpolator = counted
        exhaustive._candidate_center_seed(
            target, exhaustive._reachability.localize([target]), 0
        )
        self.assertEqual(len(counted.evaluate_calls), 1)
        self.assertEqual(counted.evaluate_calls[0].shape, (16, 4))

    def test_localized_selection_priority_and_safe_failure_paths(self):
        pipeline = self.make_multicell_identity_problem(
            4, coarse_enabled=True
        )
        target = np.ones(4)
        localization = pipeline._reachability.localize([target])
        flat_index = self.flat(pipeline, (1, 1, 1, 1))

        selected = pipeline._select_localized_seed(
            flat_index, target, localization, 0
        )
        self.assertIs(selected.source, InitialGuessSource.CONTAINING_CELL)
        self.assertIsNotNone(selected.candidate_center)
        self.assertEqual(selected.candidate_center.candidate_count, 16)

        neighbor = self.flat(pipeline, (0, 1, 1, 1))
        neighbor_seed = np.asarray([0.8, 1.0, 1.0, 1.0])
        pipeline._register_solved_seed(neighbor, neighbor_seed)
        selected = pipeline._select_localized_seed(
            flat_index, target, localization, 0
        )
        self.assertIs(selected.source, InitialGuessSource.DIRECT_NEIGHBOR)
        self.assertIsNone(selected.candidate_center)
        np.testing.assert_array_equal(selected.point, neighbor_seed)

        pipeline._solved_seed_cache.clear()
        pipeline._register_solved_seed(
            neighbor, np.asarray([0.0, 1.0, 1.0, 1.0])
        )
        selected = pipeline._select_localized_seed(
            flat_index, target, localization, 0
        )
        self.assertIs(selected.source, InitialGuessSource.CONTAINING_CELL)
        self.assertIsNotNone(selected.candidate_center)

        incomplete = replace(
            localization,
            completeness=LocalizationCompleteness.INCOMPLETE,
        )
        with self.assertRaisesRegex(
            ReachabilityLocalizationError,
            "incomplete candidate localization",
        ):
            pipeline._select_localized_seed(
                flat_index, target, incomplete, 0
            )

        pipeline._solved_seed_cache.clear()
        selected = pipeline._select_localized_seed(
            flat_index,
            target,
            localization,
            0,
            use_reachability_seed=False,
        )
        self.assertIs(selected.source, InitialGuessSource.COARSE_LEVEL)
        self.assertIsNone(selected.candidate_center)

        with self.assertRaisesRegex(
            ReachabilityLocalizationError,
            "incomplete candidate localization",
        ):
            pipeline._select_localized_seed(
                flat_index, target, incomplete, 0
            )
        with self.assertRaisesRegex(
            PipelineStateError, "does not match its localization row"
        ):
            pipeline._select_localized_seed(
                flat_index, target + 0.1, localization, 0
            )

        generic = self.make_multicell_identity_problem(
            4, coarse_enabled=False
        )
        generic_localization = generic._reachability.localize([target])
        selected = generic._select_localized_seed(
            self.flat(generic, (1, 1, 1, 1)),
            target,
            generic_localization,
            0,
            use_reachability_seed=False,
        )
        self.assertIs(selected.source, InitialGuessSource.DOMAIN_MIDPOINT)

        class FirstOrderTableProxy:
            first_order_transition = object()

        generic._table = FirstOrderTableProxy()
        with self.assertRaisesRegex(
            PipelineStateError, "disabled for first-order tables"
        ):
            generic._select_localized_seed(
                self.flat(generic, (1, 1, 1, 1)),
                target,
                generic_localization,
                0,
            )

    def test_selected_candidate_center_invokes_fast_exactly_once(self):
        pipeline = self.make_multicell_identity_problem(4)
        target = np.asarray([0.8, 1.2, 0.7, 1.3])
        localization = pipeline._reachability.localize([target])
        selected = pipeline._select_localized_seed(
            self.flat(pipeline, (1, 1, 1, 1)),
            target,
            localization,
            0,
        )
        original_solver = pipeline._solver

        class CountingSolver:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.calls = []

            def solve(self, requested, initial, *, strategy):
                self.calls.append(
                    (
                        np.asarray(requested, dtype=float).copy(),
                        np.asarray(initial, dtype=float).copy(),
                        strategy,
                    )
                )
                return self.wrapped.solve(
                    requested, initial, strategy=strategy
                )

        counting = CountingSolver(original_solver)
        pipeline._solver = counting
        result = pipeline._solve_selected_fast(target, selected)

        self.assertEqual(len(counting.calls), 1)
        self.assertIs(counting.calls[0][2], SolveStrategy.FAST)
        self.assertIs(result.status, SolveStatus.CONVERGED_FAST)
        np.testing.assert_allclose(
            result.solution, target, rtol=0.0, atol=1.0e-12
        )

    def test_depth7_chunk_flow_localizes_once_and_invokes_fast_once_per_target(self):
        pipeline = self.make_multicell_identity_problem(
            4, hierarchical=True
        )
        reachability = pipeline._reachability
        with mock.patch.object(
            reachability,
            "localize_chunk",
            wraps=reachability.localize_chunk,
        ) as localize_chunk, mock.patch.object(
            reachability,
            "classify_chunk",
            side_effect=AssertionError(
                "depth-7 production must not discard localization pairs"
            ),
        ), mock.patch.object(
            pipeline,
            "_solve_selected_fast",
            wraps=pipeline._solve_selected_fast,
        ) as selected_fast:
            result = pipeline.run()

        counters = result.manifest["run"]["stage_counters"]
        self.assertEqual(localize_chunk.call_count, 9)
        self.assertEqual(selected_fast.call_count, 81)
        self.assertEqual(counters["fast_attempted"], 81)
        self.assertEqual(counters["fallback_attempted"], 0)
        self.assertEqual(counters["seed_retry_coarse_level"], 0)
        self.assertEqual(counters["seed_retry_generic"], 0)
        self.assertGreaterEqual(
            result.manifest["run"]["status_summary"][
                "initial_guess_source"
            ]["containing_cell"],
            1,
        )
        self.assertEqual(
            result.manifest["run"]["initial_guess"][
                "fast_attempt_policy"
            ],
            "exactly_one_selected_seed",
        )
        localization_manifest = result.manifest["run"]["reachability"][
            "complete_leaf_localization"
        ]
        self.assertTrue(localization_manifest["enabled"])
        self.assertEqual(
            localization_manifest["lifetime"], "worker_chunk_only"
        )
        self.assertFalse(localization_manifest["persisted_in_checkpoint"])

    def test_ordinary_dimensions_one_to_four_honor_seed_option(self):
        for dimension in range(1, 5):
            for enabled in (True, False):
                with self.subTest(dimension=dimension, enabled=enabled):
                    pipeline = self.make_multicell_identity_problem(
                        dimension,
                        use_reachability_seed=enabled,
                    )
                    result = pipeline.run()
                    counters = result.manifest["run"]["stage_counters"]
                    candidate_count = (
                        counters["prescreen_evaluated"]
                        - counters["prescreen_excluded"]
                    )

                    self.assertEqual(
                        counters["fast_attempted"], candidate_count
                    )
                    self.assertEqual(counters["fallback_attempted"], 0)
                    self.assertEqual(counters["seed_retry_coarse_level"], 0)
                    self.assertEqual(counters["seed_retry_generic"], 0)
                    self.assertEqual(result.status_counts["solved"], candidate_count)
                    self.assertEqual(
                        result.manifest["run"]["initial_guess"][
                            "use_reachability_seed"
                        ],
                        enabled,
                    )
                    localization_manifest = result.manifest["run"][
                        "reachability"
                    ]["complete_leaf_localization"]
                    self.assertTrue(localization_manifest["enabled"])
                    self.assertEqual(
                        localization_manifest["backend"], "exhaustive"
                    )
                    containing = result.manifest["run"]["status_summary"][
                        "initial_guess_source"
                    ]["containing_cell"]
                    if enabled:
                        self.assertGreater(containing, 0)
                    else:
                        self.assertEqual(containing, 0)

    def test_ordinary_flow_uses_restricted_fallback_without_global_scan(self):
        for dimension in (2, 4):
            for enabled in (True, False):
                with self.subTest(
                    dimension=dimension, use_reachability_seed=enabled
                ):
                    pipeline = self.make_correlated_no_root_problem(
                        dimension,
                        use_reachability_seed=enabled,
                    )
                    solver = pipeline._solver
                    with mock.patch.object(
                        solver,
                        "solve_with_candidate_cells",
                        wraps=solver.solve_with_candidate_cells,
                    ) as restricted, mock.patch.object(
                        solver,
                        "_fallback",
                        side_effect=AssertionError(
                            "ordinary localized flow entered the global FALLBACK scan"
                        ),
                    ):
                        result = pipeline.run()

                    counters = result.manifest["run"]["stage_counters"]
                    expected = 2**dimension
                    self.assertEqual(counters["fast_attempted"], expected)
                    self.assertEqual(counters["fallback_attempted"], expected)
                    self.assertEqual(restricted.call_count, expected)
                    self.assertEqual(counters["seed_retry_coarse_level"], 0)
                    self.assertEqual(counters["seed_retry_generic"], 0)
                    self.assertEqual(
                        result.status_counts["certified_unreachable"], expected
                    )
                    self.assertEqual(
                        result.manifest["run"]["errors"]["count"], 0
                    )
                    self.assertEqual(
                        result.manifest["run"]["initial_guess"][
                            "fallback_domain"
                        ],
                        "complete_reachability_routing_cell_set",
                    )


if __name__ == "__main__":
    unittest.main()
