"""Tests for deterministic bounded nearest-state projection."""

import itertools
import math
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eos_trensmuther import (
    EoSProjector,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    InversionResult,
    MultilinearInterpolator,
    PipelineConfig,
    PipelinePlan,
    ProjectionConfig,
    ProjectionConfigurationError,
    ProjectionInputError,
    ProjectionQuality,
    ProjectionResult,
    ProjectionSeed,
    ProjectionSeedSource,
    ReachabilityIndex,
)
from eos_trensmuther.projection import _BoundedProjectionProblem

from tests.test_tables import TemporaryEoS, field_spec, metadata


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COORDINATE_NAMES = ("T", "muB", "muQ", "muS")
TARGET_NAMES = ("e", "nB", "nQ", "nS")


class ProjectionTestCase(unittest.TestCase):
    def make_interpolator(self, axes, function):
        dimension = len(axes)
        fields = {}
        for column, name in enumerate(COORDINATE_NAMES[:dimension]):
            fields[name] = field_spec("coordinate", column)
        for column, name in enumerate(
            TARGET_NAMES[:dimension], start=dimension
        ):
            fields[name] = field_spec("target", column)
        points = np.asarray(list(itertools.product(*axes)), dtype=float)
        values = np.asarray([function(*point) for point in points], dtype=float)
        fixture = TemporaryEoS(
            metadata(fields, dimension), np.column_stack((points, values))
        )
        self.addCleanup(fixture.close)
        table = ForwardTable(fixture.metadata_path)
        self.assertTrue(table.validate().is_valid)
        return table, MultilinearInterpolator(table)

    def make_identity(self, dimension=1, axes=None):
        selected_axes = (
            tuple(np.asarray([0.0, 1.0]) for _ in range(dimension))
            if axes is None
            else axes
        )
        return self.make_interpolator(
            selected_axes, lambda *point: tuple(point)
        )


class AnalyticProjectionTests(ProjectionTestCase):
    def make_affine(self, scales):
        table, interpolator = self.make_interpolator(
            (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])),
            lambda u, v: (1.0 + u, 2.0 + u + v),
        )
        projector = EoSProjector(
            interpolator,
            scales,
            ProjectionConfig(nearest_cell_starts=1),
        )
        return table, interpolator, projector

    def test_unweighted_analytic_projection_and_result_algebra(self):
        _, interpolator, projector = self.make_affine([1.0, 1.0])

        result = projector.project([1.0, 4.0])

        self.assertIsInstance(result, ProjectionResult)
        self.assertNotIsInstance(result, InversionResult)
        np.testing.assert_allclose(result.requested_target, [1.0, 4.0])
        np.testing.assert_allclose(
            result.source_coordinates, [0.5, 1.0], atol=2.0e-7
        )
        np.testing.assert_allclose(
            result.achieved_target, [1.5, 3.5], atol=2.0e-7
        )
        np.testing.assert_allclose(
            result.achieved_target,
            interpolator.evaluate(result.source_coordinates),
        )
        np.testing.assert_allclose(result.residual, [0.5, -0.5], atol=2.0e-7)
        np.testing.assert_allclose(
            result.scaled_residual,
            result.residual / result.residual_scales,
        )
        self.assertAlmostEqual(result.scaled_residual_norm_inf, 0.5, places=6)
        self.assertAlmostEqual(result.scaled_distance_l2, math.sqrt(0.5), places=6)
        self.assertAlmostEqual(result.objective, 0.25, places=6)
        self.assertEqual(result.quality, ProjectionQuality.OPTIMIZER_CONVERGED)
        self.assertFalse(result.globally_certified)
        for array in (
            result.requested_target,
            result.source_coordinates,
            result.achieved_target,
            result.residual,
            result.scaled_residual,
            result.residual_scales,
        ):
            self.assertFalse(array.flags.writeable)
        with self.assertRaises(ValueError):
            result.source_coordinates[0] = 7.0

    def test_fixed_metric_scales_change_the_projection(self):
        _, _, projector = self.make_affine([1.0, 2.0])

        result = projector.project([1.0, 4.0])

        np.testing.assert_allclose(
            result.source_coordinates, [0.2, 1.0], atol=2.0e-7
        )
        np.testing.assert_allclose(
            result.achieved_target, [1.2, 3.2], atol=2.0e-7
        )
        np.testing.assert_allclose(
            result.scaled_residual, [0.2, -0.4], atol=2.0e-7
        )
        self.assertAlmostEqual(result.scaled_distance_l2, math.sqrt(0.2), places=6)
        self.assertAlmostEqual(result.objective, 0.1, places=6)

    def test_separable_affine_smoke_in_dimensions_one_through_four(self):
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                _, interpolator = self.make_identity(dimension)
                projector = EoSProjector(
                    interpolator,
                    np.ones(dimension),
                    ProjectionConfig(nearest_cell_starts=1),
                )
                result = projector.project(np.full(dimension, 2.0))
                np.testing.assert_allclose(
                    result.source_coordinates, np.ones(dimension), atol=2.0e-7
                )
                np.testing.assert_allclose(
                    result.achieved_target, np.ones(dimension), atol=2.0e-7
                )
                self.assertTrue(
                    interpolator.contains(
                        result.source_coordinates,
                        include_boundary_tolerance=False,
                    )
                )
                self.assertNotEqual(result.quality, ProjectionQuality.GLOBAL_CERTIFIED)

    def test_large_representable_l2_metric_avoids_sum_of_squares_overflow(self):
        _, interpolator = self.make_identity(2)
        projector = EoSProjector(
            interpolator,
            [1.0, 1.0],
            ProjectionConfig(nearest_cell_starts=0),
        )

        with patch(
            "eos_trensmuther.projection.least_squares",
            side_effect=RuntimeError("metric-only regression"),
        ):
            result = projector.project([-1.0e154, -1.0e154])

        self.assertTrue(math.isfinite(result.scaled_distance_l2))
        self.assertTrue(math.isfinite(result.objective))
        self.assertAlmostEqual(
            result.scaled_distance_l2 / 1.0e154, math.sqrt(2.0), places=14
        )
        self.assertAlmostEqual(result.objective / 1.0e308, 1.0, places=14)


class QualityAndProvenanceTests(ProjectionTestCase):
    @staticmethod
    def optimizer_result(x, *, success, cost=-999.0):
        return types.SimpleNamespace(
            x=np.asarray(x, dtype=float),
            success=success,
            status=1 if success else 0,
            message="synthetic optimizer result",
            nfev=3,
            njev=2,
            cost=cost,
            fun=np.asarray([12345.0]),
        )

    def make_projector(self):
        _, interpolator = self.make_identity()
        return interpolator, EoSProjector(
            interpolator,
            [1.0],
            ProjectionConfig(nearest_cell_starts=1),
        )

    def test_selected_converged_trial_and_final_reevaluation(self):
        interpolator, projector = self.make_projector()
        synthetic = self.optimizer_result([0.25], success=True)

        with patch("eos_trensmuther.projection.least_squares", return_value=synthetic):
            result = projector.project([0.25])

        self.assertEqual(result.quality, ProjectionQuality.OPTIMIZER_CONVERGED)
        np.testing.assert_allclose(result.source_coordinates, [0.25])
        np.testing.assert_allclose(result.achieved_target, [0.25])
        np.testing.assert_allclose(
            result.achieved_target,
            interpolator.evaluate(result.source_coordinates),
        )
        self.assertEqual(result.objective, 0.0)
        self.assertNotEqual(result.objective, synthetic.cost)

    def test_selected_nonconverged_improvement_is_best_feasible(self):
        _, projector = self.make_projector()
        synthetic = self.optimizer_result([0.25], success=False)

        with patch("eos_trensmuther.projection.least_squares", return_value=synthetic):
            result = projector.project([0.25])

        self.assertEqual(
            result.quality, ProjectionQuality.BEST_FEASIBLE_ENCOUNTERED
        )
        self.assertFalse(result.attempts[result.selected_attempt_index].optimizer_success)

    def test_optimizer_failure_or_equal_candidate_keeps_node_incumbent(self):
        _, projector = self.make_projector()
        with patch(
            "eos_trensmuther.projection.least_squares",
            side_effect=RuntimeError("synthetic failure"),
        ):
            failed = projector.project([0.25])
        self.assertEqual(
            failed.quality, ProjectionQuality.DISCRETE_NODE_FALLBACK
        )
        self.assertIsNone(failed.selected_attempt_index)
        self.assertEqual(failed.nearest_node_flat_index, 0)
        np.testing.assert_array_equal(failed.source_coordinates, [0.0])
        self.assertTrue(all(attempt.usable for attempt in failed.attempts))
        self.assertTrue(
            all(not attempt.candidate_from_optimizer for attempt in failed.attempts)
        )

        equal = self.optimizer_result([0.0], success=True)
        with patch("eos_trensmuther.projection.least_squares", return_value=equal):
            tied = projector.project([0.25])
        self.assertEqual(tied.quality, ProjectionQuality.DISCRETE_NODE_FALLBACK)
        np.testing.assert_array_equal(tied.source_coordinates, [0.0])

    def test_invalid_optimizer_coordinates_cannot_be_clipped_into_success(self):
        _, projector = self.make_projector()
        invalid = self.optimizer_result([2.0], success=True)

        with patch("eos_trensmuther.projection.least_squares", return_value=invalid):
            result = projector.project([0.25])

        self.assertEqual(result.quality, ProjectionQuality.DISCRETE_NODE_FALLBACK)
        self.assertTrue(
            all(not attempt.candidate_from_optimizer for attempt in result.attempts)
        )
        np.testing.assert_array_equal(result.source_coordinates, [0.0])

    def test_nonfinite_trial_jacobian_retains_finite_node_fallback(self):
        _, interpolator = self.make_identity(
            axes=(np.asarray([0.0, 2.0]),)
        )
        projector = EoSProjector(
            interpolator,
            [1.0e-308],
            ProjectionConfig(nearest_cell_starts=0),
        )

        result = projector.project([0.0])

        self.assertEqual(result.quality, ProjectionQuality.DISCRETE_NODE_FALLBACK)
        np.testing.assert_array_equal(result.source_coordinates, [0.0])
        self.assertTrue(any("non-finite" in item.message for item in result.attempts))

    def test_external_seed_provenance_is_recorded(self):
        _, projector = self.make_projector()
        seed = ProjectionSeed(
            [0.4], ProjectionSeedSource.FAST_BEST_CANDIDATE
        )
        with patch(
            "eos_trensmuther.projection.least_squares",
            side_effect=RuntimeError("synthetic failure"),
        ):
            result = projector.project([0.4], seeds=(seed,))

        sources = tuple(attempt.seed_source for attempt in result.attempts)
        self.assertIn(ProjectionSeedSource.FAST_BEST_CANDIDATE, sources)
        self.assertEqual(
            result.selected_seed_source,
            ProjectionSeedSource.FAST_BEST_CANDIDATE,
        )
        self.assertEqual(
            result.quality, ProjectionQuality.BEST_FEASIBLE_ENCOUNTERED
        )
        np.testing.assert_array_equal(result.source_coordinates, [0.4])


class ActiveDomainTests(ProjectionTestCase):
    def test_disconnected_active_cells_never_project_into_the_gap(self):
        _, interpolator = self.make_interpolator(
            (np.asarray([0.0, 1.0, 2.0, 3.0]),),
            lambda x: (1.0 + x,),
        )
        mask = np.asarray([True, False, True])
        projector = EoSProjector(
            interpolator,
            [1.0],
            ProjectionConfig(nearest_cell_starts=3),
            active_cell_mask=mask,
        )

        first = projector.project([2.5])
        second = projector.project([2.5])

        np.testing.assert_array_equal(first.source_coordinates, [1.0])
        np.testing.assert_array_equal(first.achieved_target, [2.0])
        self.assertEqual(first.nearest_node_flat_index, 1)
        self.assertNotAlmostEqual(float(first.source_coordinates[0]), 1.5)
        for attempt in first.attempts:
            self.assertIn(
                (float(attempt.lower_bounds[0]), float(attempt.upper_bounds[0])),
                ((0.0, 1.0), (2.0, 3.0)),
            )
        np.testing.assert_array_equal(first.source_coordinates, second.source_coordinates)
        self.assertEqual(first.quality, second.quality)
        self.assertEqual(first.selected_attempt_index, second.selected_attempt_index)

    def test_only_nodes_incident_to_active_cells_are_feasible(self):
        _, interpolator = self.make_identity(
            axes=(np.asarray([0.0, 1.0, 2.0]),)
        )
        projector = EoSProjector(
            interpolator,
            [1.0],
            active_cell_mask=np.asarray([True, False]),
        )

        np.testing.assert_array_equal(projector.active_node_flat_indices, [0, 1])
        with self.assertRaises(ProjectionInputError):
            projector.project(
                [0.5], seeds=(ProjectionSeed([1.5]),)
            )

    def test_active_mask_is_a_defensive_snapshot(self):
        _, interpolator = self.make_identity(
            axes=(np.asarray([0.0, 1.0, 2.0]),)
        )
        supplied = np.asarray([True, False])
        projector = EoSProjector(
            interpolator, [1.0], active_cell_mask=supplied
        )

        supplied[:] = [False, True]

        np.testing.assert_array_equal(projector.active_cell_flat_indices, [0])
        np.testing.assert_array_equal(projector.active_node_flat_indices, [0, 1])
        result = projector.project([2.0])
        self.assertLessEqual(float(result.source_coordinates[0]), 1.0)

    def test_all_true_mask_matches_the_default_domain(self):
        _, interpolator = self.make_identity(
            axes=(np.asarray([0.0, 0.5, 1.0]),)
        )
        config = ProjectionConfig(nearest_cell_starts=2)
        ordinary = EoSProjector(interpolator, [1.0], config)
        explicit = EoSProjector(
            interpolator,
            [1.0],
            config,
            active_cell_mask=np.ones(interpolator.cell_shape, dtype=bool),
        )

        first = ordinary.project([1.4])
        second = explicit.project([1.4])
        self.assertFalse(ordinary.restricted_domain)
        self.assertFalse(explicit.restricted_domain)
        np.testing.assert_allclose(first.source_coordinates, second.source_coordinates)
        np.testing.assert_allclose(first.achieved_target, second.achieved_target)

    def test_cell_upper_face_uses_the_owning_cells_left_jacobian(self):
        _, interpolator = self.make_interpolator(
            (np.asarray([0.0, 1.0, 2.0]),),
            lambda x: (x if x <= 1.0 else 1.0 + 100.0 * (x - 1.0),),
        )
        problem = _BoundedProjectionProblem(
            interpolator,
            np.asarray([2.0]),
            np.asarray([1.0]),
            np.asarray([0.0]),
            np.asarray([1.0]),
            cell_bounded=True,
        )

        np.testing.assert_allclose(problem.jacobian(np.asarray([1.0])), [[1.0]])


class ValidationAndIntegrationTests(ProjectionTestCase):
    def test_configuration_inputs_and_masks_are_rejected_explicitly(self):
        invalid_configs = (
            lambda: ProjectionConfig(enabled=1),
            lambda: ProjectionConfig(nearest_cell_starts=-1),
            lambda: ProjectionConfig(max_evaluations_per_start=0),
            lambda: ProjectionConfig(least_squares_ftol=np.finfo(float).eps),
            lambda: ProjectionConfig(cell_batch_size=False),
            lambda: ProjectionConfig(max_index_bytes=0),
        )
        for constructor in invalid_configs:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ProjectionConfigurationError):
                    constructor()

        _, interpolator = self.make_identity()
        invalid_scales = (
            [0.0],
            [True],
            [np.nextafter(0.0, 1.0)],
            [[1.0]],
            {"bad": 1.0},
        )
        for scales in invalid_scales:
            with self.subTest(scales=scales):
                with self.assertRaises(ProjectionConfigurationError):
                    EoSProjector(interpolator, scales)
        with self.assertRaises(ProjectionConfigurationError):
            EoSProjector(
                interpolator,
                [1.0],
                ProjectionConfig(enabled=False),
            )
        with self.assertRaises(ProjectionConfigurationError):
            EoSProjector(interpolator, [1.0], active_cell_mask=np.asarray([1]))
        with self.assertRaises(ProjectionConfigurationError):
            EoSProjector(
                interpolator,
                [1.0],
                active_cell_mask=np.asarray([False]),
            )
        with patch.object(
            EoSProjector,
            "_active_node_mask",
            side_effect=AssertionError("allocation guard ran too late"),
        ):
            with self.assertRaises(ProjectionConfigurationError):
                EoSProjector(
                    interpolator,
                    [1.0],
                    ProjectionConfig(max_index_bytes=1),
                )

    def test_invalid_targets_and_seeds_are_rejected(self):
        _, interpolator = self.make_identity()
        projector = EoSProjector(interpolator, [1.0])
        for target in ([1.0, 2.0], [np.nan], [np.inf], "bad"):
            with self.subTest(target=target):
                with self.assertRaises(ProjectionInputError):
                    projector.project(target)
        with self.assertRaises(ProjectionInputError):
            projector.project([0.5], seeds=([0.2],))
        with self.assertRaises(ProjectionInputError):
            projector.project([0.5], seeds=(ProjectionSeed([2.0]),))

    def test_from_reachability_reuses_active_domain_and_scales(self):
        _, interpolator = self.make_identity(
            axes=(np.asarray([0.0, 1.0, 2.0, 3.0]),)
        )
        index = ReachabilityIndex(
            interpolator,
            [2.0],
            1.0e-10,
            active_cell_mask=np.asarray([True, False, True]),
        )
        projector = EoSProjector.from_reachability(
            index, ProjectionConfig(nearest_cell_starts=2)
        )

        np.testing.assert_array_equal(
            projector.active_cell_flat_indices,
            index.enclosures.cell_flat_indices,
        )
        np.testing.assert_array_equal(projector.residual_scales, [2.0])
        self.assertTrue(projector.restricted_domain)
        self.assertTrue(np.all(np.isfinite(projector.project([1.5]).achieved_target)))

    def test_from_plan_uses_the_resolved_run_metric_and_projection_config(self):
        table, interpolator = self.make_identity()
        plan = PipelinePlan.from_table(
            table,
            PipelineConfig(
                grid=InversionGridSpec(
                    "target", {"e": GridAxisSpec(3)}
                ),
                residual_scales={"e": 2.5},
                projection=ProjectionConfig(nearest_cell_starts=0),
                auxiliary_fields=(),
            ),
        )

        projector = EoSProjector.from_plan(interpolator, plan)

        self.assertIs(projector.config, plan.config.projection)
        np.testing.assert_array_equal(projector.residual_scales, [2.5])

    @unittest.skipUnless(
        (REPOSITORY_ROOT / "EoSTables/2DIsing/EoSmub602.dat").is_file()
        and (
            REPOSITORY_ROOT
            / "EoSTables/CEC/hic_table_T80-300_muB0-800_d1.dat"
        ).is_file(),
        "external CEC and Ising scientific tables are unavailable",
    )
    def test_real_ising_and_cec_tables_produce_finite_active_projections(self):
        for relative in ("2DIsing/metadata.json", "CEC/metadata.json"):
            with self.subTest(table=relative):
                table = ForwardTable(REPOSITORY_ROOT / "EoSTables" / relative)
                self.assertTrue(table.validate().is_valid)
                interpolator = MultilinearInterpolator(table)
                mask = np.zeros(interpolator.cell_shape, dtype=bool)
                mask[(0,) * interpolator.dimension] = True
                mask[tuple(size - 1 for size in interpolator.cell_shape)] = True
                bounds = interpolator.node_value_bounds
                target = np.asarray([bounds.lower[0], bounds.upper[1]])
                spans = np.maximum(bounds.upper - bounds.lower, 1.0e-12)
                projector = EoSProjector(
                    interpolator,
                    spans,
                    ProjectionConfig(
                        nearest_cell_starts=2,
                        max_evaluations_per_start=30,
                    ),
                    active_cell_mask=mask,
                )

                result = projector.project(target)
                self.assertTrue(np.all(np.isfinite(result.source_coordinates)))
                self.assertTrue(np.all(np.isfinite(result.achieved_target)))
                np.testing.assert_allclose(
                    result.achieved_target,
                    interpolator.evaluate(result.source_coordinates),
                )
                self.assertNotEqual(
                    result.quality, ProjectionQuality.GLOBAL_CERTIFIED
                )


if __name__ == "__main__":
    unittest.main()
