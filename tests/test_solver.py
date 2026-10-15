"""Analytical tests for the fast and certified inversion pipelines."""

import itertools
import unittest
from dataclasses import replace

import numpy as np

from eos_trensmuther import (
    EoSRootSolver,
    FallbackStatus,
    ForwardTable,
    MultilinearInterpolator,
    LocalizationCompleteness,
    ProofStatus,
    ReachabilityIndex,
    ReachabilityLocalization,
    SolveStatus,
    SolverConfig,
    SolverConfigurationError,
    SolverInputError,
)
from eos_trensmuther.solver import (
    _BATCH_KRAWCZYK_CERTIFIED,
    _BATCH_KRAWCZYK_CONTRACT,
    _BATCH_KRAWCZYK_EXCLUDED,
    _BATCH_KRAWCZYK_INCONCLUSIVE,
    _Interval,
    _RigorousCellModel,
    _RigorousSubcellEnclosureEvaluator,
    _bounded_midpoint,
    _interval_multiply,
    _krawczyk_step,
    _krawczyk_steps_for_cells,
)
from eos_trensmuther.subcells import SubcellBoxes, SubcellStopReason

from tests.test_tables import TemporaryEoS, field_spec, metadata


class SolverTestCase(unittest.TestCase):
    def make_solver(
        self,
        first_axis,
        second_axis,
        function,
        *,
        config=None,
        residual_scales=None,
    ):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }
        points = np.asarray(
            list(itertools.product(first_axis, second_axis)), dtype=float
        )
        targets = np.asarray([function(*point) for point in points], dtype=float)
        fixture = TemporaryEoS(
            metadata(fields, 2), np.column_stack((points, targets))
        )
        self.addCleanup(fixture.close)
        table = ForwardTable(
            fixture.metadata_path, _allow_unsuitable_geometry=True
        )
        self.assertTrue(table.validate().is_valid)
        interpolator = MultilinearInterpolator(table)
        solver = EoSRootSolver(
            interpolator,
            config,
            residual_scales=residual_scales,
        )
        return table, interpolator, solver

    def make_affine_solver(self, *, config=None):
        return self.make_solver(
            [0.5, 1.5],
            [0.0, 1.0],
            lambda temperature, chemical_potential: (
                temperature - 0.5 + chemical_potential,
                temperature - 0.5 - chemical_potential,
            ),
            config=config,
        )


class SubcellProofAdapterTests(SolverTestCase):
    def test_vectorized_cell_enclosures_equal_scalar_proof_kernel(self):
        _, interpolator, _ = self.make_affine_solver()
        model = _RigorousCellModel(
            interpolator,
            np.asarray([0, 0], dtype=np.intp),
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
        )
        boxes = SubcellBoxes.original_cells([0], 2).dyadic_children()
        boxes = boxes.dyadic_children()
        batched = model.value_enclosures(boxes.unit_lower, boxes.unit_upper)
        scalar = [
            model.value_enclosure(lower, upper)
            for lower, upper in zip(boxes.unit_lower, boxes.unit_upper)
        ]
        np.testing.assert_array_equal(
            batched.lower,
            np.stack([item.lower for item in scalar]),
        )
        np.testing.assert_array_equal(
            batched.upper,
            np.stack([item.upper for item in scalar]),
        )

    def test_adapter_reuses_cell_model_and_maps_normalized_bounds_safely(self):
        _, interpolator, solver = self.make_affine_solver()
        target = np.asarray([1.0, 0.0])
        evaluator = _RigorousSubcellEnclosureEvaluator(
            interpolator,
            target,
            solver._residual_scales(target),
            solver.config.root_residual_tolerance,
        )
        boxes = SubcellBoxes.original_cells([0], 2).dyadic_children()
        enclosures = evaluator(boxes)
        physical_lower, physical_upper = evaluator.physical_bounds(boxes)

        self.assertEqual(evaluator.cached_model_count, 1)
        self.assertTrue(np.all(enclosures.defined))
        np.testing.assert_array_equal(physical_lower[0], [0.5, 0.0])
        np.testing.assert_array_equal(physical_upper[-1], [1.5, 1.0])
        self.assertTrue(np.all(physical_lower >= interpolator.bounds[:, 0]))
        self.assertTrue(np.all(physical_upper <= interpolator.bounds[:, 1]))

        # Every physical subcell corner is contained in the corresponding
        # outward-rounded routing enclosure produced by the solver kernel.
        for row in range(boxes.count):
            corners = np.asarray(
                list(
                    itertools.product(
                        *zip(physical_lower[row], physical_upper[row])
                    )
                )
            )
            values = np.asarray(interpolator.evaluate(corners))
            self.assertTrue(np.all(values >= enclosures.routing_lower[row]))
            self.assertTrue(np.all(values <= enclosures.routing_upper[row]))

    def test_shared_dyadic_grid_is_array_exact_with_general_proof_kernel(self):
        axes = tuple(np.asarray([0.0, 1.0]) for _ in range(4))
        coordinates = np.meshgrid(*axes, indexing="ij")
        fields = {
            "y%d" % field: sum(
                (field + 1) * (axis + 1) * coordinates[axis]
                for axis in range(4)
            )
            + (field + 1) * coordinates[0] * coordinates[2]
            for field in range(4)
        }
        interpolator = MultilinearInterpolator.from_rectilinear_map(
            axes,
            fields,
            coordinate_names=("x0", "x1", "x2", "x3"),
            target_names=("y0", "y1", "y2", "y3"),
            coordinate_units=("1",) * 4,
            target_units=("1",) * 4,
        )
        evaluator = _RigorousSubcellEnclosureEvaluator(
            interpolator,
            np.asarray([0.3, 0.4, 0.5, 0.6]),
            np.ones(4),
            1.0e-10,
        )
        parents = SubcellBoxes.original_cells([0], 4).dyadic_children()
        children = parents.dyadic_children()
        general = evaluator(children)
        shared = evaluator.evaluate_dyadic_children(parents, children)

        np.testing.assert_array_equal(shared.defined, general.defined)
        np.testing.assert_array_equal(
            shared.routing_lower, general.routing_lower
        )
        np.testing.assert_array_equal(
            shared.routing_upper, general.routing_upper
        )

    def test_solver_subcell_entry_preserves_root_and_excludes_false_positive(self):
        _, _, solver = self.make_affine_solver()

        face_root = solver._refine_candidate_subcells(
            [1.0, 0.0], [0], maximum_depth=7
        )
        self.assertFalse(face_root.certifies_no_root)
        self.assertEqual(face_root.report.stop_reason, SubcellStopReason.MAXIMUM_DEPTH)
        self.assertEqual(face_root.boxes.count, 4)

        # This target is inside the original component AABB but its affine
        # inverse has muB=-0.4, outside the source cell.
        false_positive = solver._refine_candidate_subcells(
            [0.1, 0.9], [0], maximum_depth=7
        )
        self.assertTrue(false_positive.certifies_no_root)
        self.assertEqual(
            false_positive.report.stop_reason, SubcellStopReason.EMPTY_UNION
        )
        self.assertEqual(false_positive.report.first_exclusion_depth, 1)


class BatchedResidualBoxTests(SolverTestCase):
    def test_batched_first_steps_equal_scalar_proof_kernel(self):
        _, interpolator, solver = self.make_affine_solver()
        target = np.asarray([0.70, -0.04])
        factors = 1.0 / solver._residual_scales(target)
        boxes = SubcellBoxes.original_cells([0], 2).dyadic_children()
        cell_indices = np.zeros((boxes.count, 2), dtype=np.intp)
        batch = _krawczyk_steps_for_cells(
            interpolator,
            cell_indices,
            target,
            factors,
            boxes.unit_lower,
            boxes.unit_upper,
        )
        model = _RigorousCellModel(
            interpolator,
            np.asarray([0, 0], dtype=np.intp),
            target,
            factors,
        )
        codes = {
            "excluded": _BATCH_KRAWCZYK_EXCLUDED,
            "certified": _BATCH_KRAWCZYK_CERTIFIED,
            "contract": _BATCH_KRAWCZYK_CONTRACT,
            "inconclusive": _BATCH_KRAWCZYK_INCONCLUSIVE,
        }
        for row in range(boxes.count):
            scalar = _krawczyk_step(
                model, boxes.unit_lower[row], boxes.unit_upper[row]
            )
            self.assertEqual(batch.kinds[row], codes[scalar.kind])
            if scalar.lower is not None:
                np.testing.assert_array_equal(batch.lower[row], scalar.lower)
                np.testing.assert_array_equal(batch.upper[row], scalar.upper)

    def test_shared_box_context_matches_old_root_and_no_root_outcomes(self):
        _, _, solver = self.make_affine_solver()

        no_root_target = np.asarray([0.24, 0.26])
        old_no_root = solver.solve(no_root_target, strategy="fallback")
        refinement, new_no_root = solver._refine_and_fallback_candidate_subcells(
            no_root_target,
            [0],
            complete_source_coverage=True,
            active_cell_count=1,
            maximum_depth=1,
        )
        self.assertEqual(old_no_root.fallback.status, FallbackStatus.NO_ROOT_IN_DOMAIN)
        self.assertEqual(new_no_root.report.status, old_no_root.fallback.status)
        self.assertEqual(refinement.boxes.count, 1)
        self.assertIsNotNone(new_no_root.box_search)
        self.assertEqual(new_no_root.box_search.initial_boxes, 1)
        self.assertEqual(new_no_root.box_search.one_shot_boxes_processed, 1)
        self.assertEqual(new_no_root.box_search.one_shot_boxes_excluded, 1)
        self.assertEqual(new_no_root.box_search.recursive_boxes_processed, 0)

        root_target = np.asarray([0.70, -0.04])
        root = np.asarray([0.83, 0.37])
        old_root = solver.solve(root_target, root, strategy="fallback")
        refinement, new_root = solver._refine_and_fallback_candidate_subcells(
            root_target,
            [0],
            root,
            complete_source_coverage=True,
            active_cell_count=1,
            maximum_depth=2,
        )
        self.assertEqual(old_root.fallback.status, FallbackStatus.UNIQUE_IN_BOX)
        self.assertEqual(new_root.report.status, old_root.fallback.status)
        self.assertIsNotNone(new_root.certificate)
        self.assertEqual(new_root.box_search.initial_boxes, refinement.boxes.count)
        self.assertEqual(
            new_root.box_search.one_shot_boxes_processed,
            refinement.boxes.count,
        )
        self.assertEqual(new_root.box_search.one_shot_boxes_certified, 1)
        self.assertEqual(new_root.box_search.recursive_boxes_processed, 0)
        # Refinement is batched directly through the existing proof kernel;
        # Krawczyk constructs only the one model needed for the certificate.
        self.assertEqual(new_root.box_search.cell_models_constructed, 1)
        self.assertEqual(new_root.box_search.cell_models_total, 1)

    def test_all_residual_candidates_are_one_shot_tested_before_recursion(self):
        _, interpolator, solver = self.make_solver(
            [0.5, 1.0, 1.5],
            [0.0, 0.5, 1.0],
            lambda temperature, chemical_potential: (
                temperature - 0.5 + chemical_potential,
                temperature - 0.5 - chemical_potential,
            ),
            residual_scales=[1.0, 1.0],
        )
        target = np.asarray([0.63, 0.17])
        index = ReachabilityIndex(
            interpolator,
            [1.0, 1.0],
            solver.config.root_residual_tolerance,
        )
        localization = index.localize([target])
        cells = localization.cells_for(0)
        self.assertGreater(cells.size, 1)

        refinement, outcome = solver._refine_and_fallback_candidate_subcells(
            target,
            cells,
            [0.9, 0.23],
            complete_source_coverage=True,
            active_cell_count=index.provenance.active_cell_count,
            maximum_depth=2,
        )
        self.assertEqual(outcome.report.status, FallbackStatus.UNIQUE_IN_BOX)
        self.assertEqual(
            outcome.box_search.one_shot_boxes_processed,
            refinement.boxes.count,
        )
        self.assertEqual(outcome.box_search.recursive_boxes_processed, 0)
        self.assertEqual(
            outcome.box_search.one_shot_boxes_excluded
            + outcome.box_search.one_shot_boxes_certified,
            refinement.boxes.count,
        )

    def test_boundary_root_remains_unresolved_not_falsely_excluded(self):
        config = SolverConfig(fallback_max_boxes=32, fallback_max_depth=8)
        _, _, solver = self.make_affine_solver(config=config)
        target = np.asarray([0.5, 0.5])
        old = solver.solve(target, [1.25, 0.75], strategy="fallback")
        _, new = solver._refine_and_fallback_candidate_subcells(
            target,
            [0],
            [1.25, 0.75],
            complete_source_coverage=True,
            active_cell_count=1,
            maximum_depth=2,
        )
        self.assertEqual(old.fallback.status, FallbackStatus.UNRESOLVED)
        self.assertEqual(new.report.status, FallbackStatus.UNRESOLVED)
        self.assertGreater(new.report.unresolved_boxes, 0)
        self.assertIsNone(new.certificate)

    def test_unproven_or_guarded_coverage_cannot_certify_no_root(self):
        _, _, solver = self.make_affine_solver()
        target = np.asarray([0.24, 0.26])
        refinement, unproven = solver._refine_and_fallback_candidate_subcells(
            target,
            [0],
            complete_source_coverage=False,
            active_cell_count=1,
            maximum_depth=1,
        )
        self.assertTrue(refinement.report.complete)
        self.assertTrue(unproven.box_search.complete_no_root)
        self.assertEqual(unproven.report.status, FallbackStatus.UNRESOLVED)

        config = SolverConfig(fallback_max_boxes=8, fallback_max_depth=2)
        _, _, singular = self.make_solver(
            [0.5, 1.5],
            [0.0, 1.0],
            lambda temperature, chemical_potential: (0.0, 0.0),
            config=config,
            residual_scales=[1.0, 1.0],
        )
        guarded, guarded_outcome = (
            singular._refine_and_fallback_candidate_subcells(
                [0.0, 0.0],
                [0],
                complete_source_coverage=True,
                active_cell_count=1,
                maximum_depth=2,
                maximum_active_boxes=2,
            )
        )
        self.assertFalse(guarded.report.complete)
        self.assertEqual(guarded_outcome.report.status, FallbackStatus.UNRESOLVED)
        self.assertIsNone(guarded_outcome.box_search)

    def test_box_and_depth_limits_remain_inconclusive(self):
        box_limited_config = SolverConfig(
            fallback_max_boxes=2,
            fallback_max_depth=4,
        )
        _, _, box_limited = self.make_solver(
            [0.5, 1.5],
            [0.0, 1.0],
            lambda temperature, chemical_potential: (0.0, 0.0),
            config=box_limited_config,
            residual_scales=[1.0, 1.0],
        )
        refinement, outcome = (
            box_limited._refine_and_fallback_candidate_subcells(
                [0.0, 0.0],
                [0],
                complete_source_coverage=True,
                active_cell_count=1,
                maximum_depth=1,
                maximum_active_boxes=16,
            )
        )
        self.assertEqual(refinement.boxes.count, 4)
        self.assertEqual(outcome.report.status, FallbackStatus.UNRESOLVED)
        self.assertEqual(outcome.box_search.one_shot_boxes_processed, 2)
        self.assertEqual(outcome.box_search.unprocessed_initial_boxes, 2)

        depth_limited_config = SolverConfig(
            fallback_max_boxes=32,
            fallback_max_depth=1,
        )
        _, _, depth_limited = self.make_solver(
            [0.5, 1.5],
            [0.0, 1.0],
            lambda temperature, chemical_potential: (0.0, 0.0),
            config=depth_limited_config,
            residual_scales=[1.0, 1.0],
        )
        _, outcome = depth_limited._refine_and_fallback_candidate_subcells(
            [0.0, 0.0],
            [0],
            complete_source_coverage=True,
            active_cell_count=1,
            maximum_depth=1,
            maximum_active_boxes=16,
        )
        self.assertEqual(outcome.report.status, FallbackStatus.UNRESOLVED)
        self.assertEqual(outcome.box_search.maximum_depth_reached, 1)
        self.assertGreater(outcome.box_search.unresolved_boxes, 0)

    def test_nonfinite_interval_work_is_retained_as_unresolved(self):
        config = SolverConfig(fallback_max_boxes=4, fallback_max_depth=2)
        _, _, solver = self.make_solver(
            [0.5, 1.5],
            [0.0, 1.0],
            lambda temperature, chemical_potential: (
                1.0e308 if temperature < 1.0 else -1.0e308,
                1.0e308 if chemical_potential < 0.5 else -1.0e308,
            ),
            config=config,
            residual_scales=[1.0e308, 1.0e308],
        )
        refinement, outcome = solver._refine_and_fallback_candidate_subcells(
            [0.0, 0.0],
            [0],
            complete_source_coverage=True,
            active_cell_count=1,
            maximum_depth=1,
            maximum_active_boxes=16,
        )
        self.assertEqual(refinement.boxes.count, 4)
        self.assertEqual(outcome.report.status, FallbackStatus.UNRESOLVED)
        self.assertEqual(outcome.box_search.one_shot_boxes_excluded, 0)
        self.assertGreater(outcome.box_search.unresolved_boxes, 0)


class FastPipelineTests(SolverTestCase):
    def test_fast_pipeline_localizes_and_refines_an_affine_root(self):
        _, interpolator, solver = self.make_affine_solver()

        result = solver.solve(
            [0.75, 0.25], [1.375, 0.875], strategy="fast"
        )

        self.assertEqual(result.status, SolveStatus.CONVERGED_FAST)
        self.assertEqual(result.proof_status, ProofStatus.NONE)
        self.assertTrue(result.converged)
        self.assertFalse(result.certified)
        self.assertEqual(
            result.path, ("bounded_least_squares", "safeguarded_newton")
        )
        np.testing.assert_allclose(result.solution, [1.0, 0.25], atol=2e-12)
        np.testing.assert_allclose(
            interpolator(result.solution), [0.75, 0.25], atol=2e-12
        )
        self.assertLessEqual(
            result.scaled_residual_norm, solver.config.root_residual_tolerance
        )
        mantissas, _ = np.frexp(result.residual_scales)
        np.testing.assert_array_equal(mantissas, [0.5, 0.5])
        self.assertIsNotNone(result.localization)
        self.assertTrue(result.localization.accepted_for_newton)
        self.assertIsNotNone(result.newton)
        self.assertTrue(result.newton.converged)
        with self.assertRaises(ValueError):
            result.solution[0] = 7.0

    def test_continuation_uses_only_the_previous_converged_solution(self):
        _, _, solver = self.make_affine_solver()

        results = solver.solve_path(
            [[0.75, 0.25], [0.25, 0.75], [0.80, 0.20]],
            initial_guess=[1.375, 0.875],
            strategy="fast",
        )

        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].converged)
        self.assertFalse(results[1].converged)
        self.assertTrue(results[2].converged)
        self.assertEqual(results[0].initial_guess_source, "provided")
        self.assertEqual(results[1].initial_guess_source, "continuation")
        self.assertEqual(results[2].initial_guess_source, "domain_midpoint")
        np.testing.assert_allclose(results[1].initial_guess, results[0].solution)
        np.testing.assert_allclose(results[2].initial_guess, [1.0, 0.5])
        self.assertFalse(
            np.array_equal(results[2].initial_guess, results[1].best_candidate)
        )
        np.testing.assert_allclose(results[2].solution, [1.0, 0.30], atol=2e-12)


class FallbackPipelineTests(SolverTestCase):
    @staticmethod
    def complex_square(temperature, chemical_potential):
        u = temperature - 0.5
        v = chemical_potential - 0.5
        return 1.0 + u * u - v * v, 2.0 * u * v

    def make_complex_solver(self, *, config=None):
        dyadic_axis = [3.0 / 8.0, 0.5, 0.75, 1.0]
        return self.make_solver(
            dyadic_axis, dyadic_axis, self.complex_square, config=config
        )

    def test_optimizer_success_with_nonzero_residual_triggers_fallback(self):
        _, _, solver = self.make_complex_solver()
        target = [1.0, 25.0 / 128.0]
        stationary_guess = [0.5, 0.5]

        fast_only = solver.solve(target, stationary_guess, strategy="fast_only")

        self.assertEqual(fast_only.status, SolveStatus.FAST_PATH_FAILED)
        self.assertFalse(fast_only.converged)
        self.assertIsNone(fast_only.solution)
        np.testing.assert_array_equal(fast_only.best_candidate, stationary_guess)
        self.assertTrue(fast_only.localization.optimizer_success)
        self.assertFalse(fast_only.localization.accepted_for_newton)
        self.assertGreater(
            fast_only.localization.scaled_residual_norm,
            solver.config.localization_residual_tolerance,
        )

        automatic = solver.solve(target, stationary_guess, strategy="auto")

        self.assertEqual(automatic.status, SolveStatus.CONVERGED_FALLBACK)
        self.assertEqual(automatic.proof_status, ProofStatus.UNIQUE_IN_BOX)
        self.assertTrue(automatic.converged)
        self.assertTrue(automatic.certified)
        self.assertEqual(automatic.fallback.status, FallbackStatus.UNIQUE_IN_BOX)
        self.assertTrue(automatic.fallback.outward_rounded)
        self.assertTrue(automatic.certificate.rigorous)
        self.assertTrue(solver.interval_certification_available)
        bounds = solver.interpolator.node_value_bounds
        np.testing.assert_array_equal(
            automatic.residual_scales,
            np.maximum(np.abs(bounds.lower), np.abs(bounds.upper)),
        )
        self.assertEqual(automatic.certificate.uniqueness_scope, "box_only")
        np.testing.assert_array_equal(automatic.certificate.cell_indices, [2, 2])
        np.testing.assert_allclose(
            automatic.solution, [13.0 / 16.0, 13.0 / 16.0], atol=2e-12
        )
        self.assertTrue(
            np.all(automatic.solution > automatic.certificate.lower)
        )
        self.assertTrue(
            np.all(automatic.solution < automatic.certificate.upper)
        )
        self.assertIn("interval_krawczyk", automatic.path)
        self.assertIn("certified_box_newton", automatic.path)
        self.assertIsNone(automatic.newton)
        self.assertIsNotNone(automatic.fallback_newton)
        self.assertEqual(
            automatic.fallback_newton.normalized_coordinate_system,
            "interpolation_cell",
        )

    def test_candidate_truncation_cannot_become_a_no_root_proof(self):
        config = SolverConfig(fallback_max_candidate_cells=1)
        _, _, solver = self.make_complex_solver(config=config)

        result = solver.solve(
            [1.0, 25.0 / 128.0], [0.5, 0.5], strategy="fallback"
        )

        self.assertEqual(result.status, SolveStatus.UNRESOLVED)
        self.assertEqual(result.proof_status, ProofStatus.INCONCLUSIVE)
        self.assertGreater(result.fallback.candidate_cells, 1)
        self.assertEqual(result.fallback.retained_candidate_cells, 1)
        self.assertGreaterEqual(
            result.fallback.unresolved_boxes,
            result.fallback.candidate_cells
            - result.fallback.retained_candidate_cells,
        )

    def test_certified_root_is_retained_when_point_refinement_is_too_strict(self):
        config = SolverConfig(
            root_residual_tolerance=1.0e-30,
            newton_max_iterations=1,
        )
        _, _, solver = self.make_complex_solver(config=config)

        result = solver.solve(
            [1.0, 25.0 / 128.0], [0.5, 0.5], strategy="fallback"
        )

        self.assertEqual(result.status, SolveStatus.CERTIFIED_ROOT_UNREFINED)
        self.assertEqual(result.proof_status, ProofStatus.UNIQUE_IN_BOX)
        self.assertTrue(result.certified)
        self.assertFalse(result.converged)
        self.assertIsNone(result.solution)
        self.assertTrue(np.all(np.isfinite(result.best_candidate)))

    def test_failed_newton_enters_fallback_in_auto_mode(self):
        _, _, solver = self.make_solver(
            [0.5, 1.5],
            [0.0, 1.0],
            lambda temperature, chemical_potential: (
                temperature - 0.5,
                0.0,
            ),
            residual_scales=[1.0, 1.0],
        )
        target = [5.0 / 16.0, 5.0e-7]

        fast = solver.solve(target, strategy="fast")
        self.assertEqual(fast.status, SolveStatus.FAST_PATH_FAILED)
        self.assertTrue(fast.localization.accepted_for_newton)
        self.assertIsNotNone(fast.newton)
        self.assertIn("singular", fast.newton.reason)

        automatic = solver.solve(target, strategy="auto")
        self.assertEqual(automatic.status, SolveStatus.CERTIFIED_NO_ROOT)
        self.assertIsNotNone(automatic.newton)
        self.assertEqual(automatic.fallback.seed_source, "fast_candidate")

    def test_complete_interval_search_can_certify_no_root_inside_envelope(self):
        _, interpolator, solver = self.make_affine_solver()
        target = np.asarray([0.24, 0.26])
        global_bounds = interpolator.node_value_bounds
        self.assertTrue(np.all(target >= global_bounds.lower))
        self.assertTrue(np.all(target <= global_bounds.upper))

        result = solver.solve(target, strategy="fallback")

        self.assertEqual(result.status, SolveStatus.CERTIFIED_NO_ROOT)
        self.assertEqual(result.proof_status, ProofStatus.NO_ROOT_IN_DOMAIN)
        self.assertTrue(result.certified)
        self.assertFalse(result.converged)
        self.assertIsNone(result.solution)
        self.assertTrue(result.fallback.scan_complete)
        self.assertEqual(result.fallback.status, FallbackStatus.NO_ROOT_IN_DOMAIN)
        self.assertEqual(result.fallback.unresolved_boxes, 0)

    def test_nonisolated_root_set_is_unresolved_not_no_root(self):
        config = SolverConfig(fallback_max_boxes=32, fallback_max_depth=8)
        _, _, solver = self.make_solver(
            [0.5, 1.5],
            [0.0, 1.0],
            lambda temperature, chemical_potential: (
                temperature - 0.5,
                0.0,
            ),
            config=config,
            residual_scales=[1.0, 1.0],
        )

        result = solver.solve([5.0 / 16.0, 0.0], strategy="fallback_only")

        self.assertEqual(result.status, SolveStatus.UNRESOLVED)
        self.assertEqual(result.proof_status, ProofStatus.INCONCLUSIVE)
        self.assertEqual(result.fallback.status, FallbackStatus.UNRESOLVED)
        self.assertGreater(result.fallback.unresolved_boxes, 0)
        self.assertIsNone(result.solution)

    def test_boundary_root_is_admissible_but_not_falsely_certified(self):
        config = SolverConfig(fallback_max_boxes=32, fallback_max_depth=8)
        _, _, solver = self.make_affine_solver(config=config)
        target = [0.5, 0.5]
        boundary_root = [1.0, 0.0]

        fast = solver.solve(target, [1.25, 0.75], strategy="fast")
        self.assertEqual(fast.status, SolveStatus.CONVERGED_FAST)
        np.testing.assert_allclose(fast.solution, boundary_root, atol=2e-12)
        self.assertEqual(fast.proof_status, ProofStatus.NONE)

        fallback = solver.solve(target, [1.25, 0.75], strategy="fallback")
        self.assertEqual(fallback.status, SolveStatus.UNRESOLVED)
        self.assertNotEqual(fallback.status, SolveStatus.CERTIFIED_NO_ROOT)


class RestrictedFallbackTests(SolverTestCase):
    def make_affine_localized_solver(self, *, config=None):
        return self.make_solver(
            [0.5, 1.0, 1.5],
            [0.0, 0.5, 1.0],
            lambda temperature, chemical_potential: (
                temperature - 0.5 + chemical_potential,
                temperature - 0.5 - chemical_potential,
            ),
            config=config,
            residual_scales=[1.0, 1.0],
        )

    def localization(self, interpolator, solver, targets, *, active_mask=None):
        index = ReachabilityIndex(
            interpolator,
            [1.0, 1.0],
            solver.config.root_residual_tolerance,
            active_cell_mask=active_mask,
        )
        return index, index.localize(np.asarray(targets, dtype=float))

    def test_restricted_root_matches_global_and_avoids_the_domain_scan(self):
        _, interpolator, solver = self.make_affine_localized_solver(
            config=SolverConfig(enable_subcell_fallback=False)
        )
        target = np.asarray([0.70, -0.04])
        index, localization = self.localization(interpolator, solver, [target])

        global_result = solver.solve(target, strategy="fallback")
        restricted = solver.solve_with_candidate_cells(
            target,
            localization,
            0,
            expected_provenance=index.provenance,
            strategy="fallback",
        )

        self.assertEqual(restricted.status, global_result.status)
        self.assertEqual(restricted.proof_status, global_result.proof_status)
        np.testing.assert_allclose(restricted.solution, global_result.solution)
        self.assertEqual(
            restricted.fallback.candidate_source,
            "complete_reachability_localization",
        )
        self.assertTrue(restricted.fallback.localization_complete)
        self.assertEqual(
            restricted.fallback.supplied_candidate_cells,
            localization.pair_count,
        )
        self.assertLess(
            restricted.fallback.cells_scanned,
            global_result.fallback.cells_scanned,
        )
        self.assertEqual(restricted.fallback.unprocessed_candidate_cells, 0)

    def test_restricted_no_root_matches_complete_global_search(self):
        _, interpolator, solver = self.make_affine_localized_solver()
        target = np.asarray([0.25, 0.75])
        index, localization = self.localization(interpolator, solver, [target])

        global_result = solver.solve(target, strategy="fallback")
        restricted = solver.solve_with_candidate_cells(
            target,
            localization,
            0,
            expected_provenance=index.provenance,
        )

        self.assertEqual(global_result.status, SolveStatus.CERTIFIED_NO_ROOT)
        self.assertEqual(restricted.status, global_result.status)
        self.assertEqual(restricted.proof_status, ProofStatus.NO_ROOT_IN_DOMAIN)
        self.assertEqual(
            restricted.fallback.status, FallbackStatus.NO_ROOT_IN_DOMAIN
        )
        self.assertEqual(restricted.fallback.unresolved_boxes, 0)

        automatic = solver.solve_with_candidate_cells(
            target,
            localization,
            0,
            expected_provenance=index.provenance,
            strategy="auto",
        )
        self.assertEqual(automatic.status, SolveStatus.CERTIFIED_NO_ROOT)
        self.assertIsNotNone(automatic.localization)
        self.assertIn("interval_krawczyk", automatic.path)
        self.assertEqual(
            automatic.fallback.candidate_source,
            "complete_reachability_localization",
        )

    def test_guarded_subcell_path_preserves_status_and_reports_mechanism(self):
        config = SolverConfig(
            enable_subcell_fallback=True,
            max_subcell_depth=2,
            max_subcell_active_boxes=256,
        )
        _, interpolator, solver = self.make_affine_localized_solver(
            config=config
        )
        old_solver = EoSRootSolver(
            interpolator,
            replace(config, enable_subcell_fallback=False),
            residual_scales=[1.0, 1.0],
        )
        target = np.asarray([0.24, 0.26])
        index, localization = self.localization(interpolator, solver, [target])

        old = old_solver.solve_with_candidate_cells(
            target,
            localization,
            0,
            expected_provenance=index.provenance,
            strategy="fallback",
        )
        new = solver.solve_with_candidate_cells(
            target,
            localization,
            0,
            expected_provenance=index.provenance,
            strategy="fallback",
        )

        self.assertEqual(new.status, old.status)
        self.assertEqual(new.proof_status, old.proof_status)
        self.assertIn("subcell_refinement", new.path)
        self.assertNotIn("subcell_refinement", old.path)
        self.assertIsNotNone(new.fallback.subcell_report)
        self.assertIsNone(old.fallback.subcell_report)
        self.assertTrue(new.fallback.localization_complete)
        self.assertEqual(
            new.fallback.candidate_source, "complete_subcell_refinement"
        )

    def test_subcell_residual_krawczyk_certifies_root(self):
        config = SolverConfig(
            enable_subcell_fallback=True,
            max_subcell_depth=2,
            max_subcell_active_boxes=256,
        )
        _, interpolator, solver = self.make_affine_localized_solver(
            config=config
        )
        target = np.asarray([0.70, -0.04])
        index, localization = self.localization(interpolator, solver, [target])

        result = solver.solve_with_candidate_cells(
            target,
            localization,
            0,
            [0.83, 0.37],
            expected_provenance=index.provenance,
            strategy="fallback",
        )

        self.assertEqual(result.status, SolveStatus.CONVERGED_FALLBACK)
        self.assertEqual(result.proof_status, ProofStatus.UNIQUE_IN_BOX)
        self.assertIsNotNone(result.fallback.subcell_report)
        self.assertGreater(result.fallback.subcell_residual_box_count, 0)
        self.assertGreater(result.fallback.boxes_processed, 0)
        self.assertIsNotNone(result.certificate)

    def test_subcell_active_box_guard_is_unresolved(self):
        config = SolverConfig(
            enable_subcell_fallback=True,
            max_subcell_depth=2,
            max_subcell_active_boxes=1,
        )
        _, interpolator, solver = self.make_affine_localized_solver(
            config=config
        )
        target = np.asarray([0.65, 0.15])
        index, localization = self.localization(interpolator, solver, [target])

        result = solver.solve_with_candidate_cells(
            target,
            localization,
            0,
            expected_provenance=index.provenance,
            strategy="fallback",
        )

        self.assertEqual(result.status, SolveStatus.UNRESOLVED)
        self.assertEqual(result.proof_status, ProofStatus.INCONCLUSIVE)
        self.assertFalse(result.fallback.subcell_report.complete)
        self.assertNotEqual(
            result.fallback.status, FallbackStatus.NO_ROOT_IN_DOMAIN
        )

    def test_empty_complete_localization_uses_reachability_proof_directly(self):
        config = SolverConfig(enable_subcell_fallback=True)
        _, interpolator, solver = self.make_affine_localized_solver(
            config=config
        )
        target = np.asarray([0.25, 0.75])
        index, localization = self.localization(interpolator, solver, [target])
        self.assertEqual(localization.pair_count, 0)

        result = solver.solve_with_candidate_cells(
            target,
            localization,
            0,
            expected_provenance=index.provenance,
            strategy="fallback",
        )

        self.assertEqual(result.status, SolveStatus.CERTIFIED_NO_ROOT)
        self.assertEqual(result.proof_status, ProofStatus.NO_ROOT_IN_DOMAIN)
        self.assertIsNone(result.fallback.subcell_report)
        self.assertEqual(result.fallback.boxes_processed, 0)
        self.assertTrue(result.fallback.localization_complete)

    def test_routing_shell_cells_are_exactly_excluded_before_interval_work(self):
        config = SolverConfig(
            localization_residual_tolerance=0.1,
            root_residual_tolerance=0.1,
            enable_subcell_fallback=False,
        )
        _, interpolator, solver = self.make_solver(
            [0.5, 1.5],
            [0.0, 1.0],
            lambda temperature, chemical_potential: (
                temperature - 0.5 + chemical_potential,
                temperature - 0.5 - chemical_potential,
            ),
            config=config,
            residual_scales=[1.0, 1.0],
        )
        target = np.asarray([2.05, 0.0])
        index, localization = self.localization(interpolator, solver, [target])
        self.assertTrue(localization.result.tolerance_shell_mask[0])

        result = solver.solve_with_candidate_cells(
            target,
            localization,
            0,
            expected_provenance=index.provenance,
        )

        self.assertEqual(result.status, SolveStatus.CERTIFIED_NO_ROOT)
        self.assertEqual(result.fallback.boxes_processed, 0)
        self.assertEqual(result.fallback.supplied_candidate_cells, 1)
        self.assertEqual(result.fallback.supplied_cells_excluded_by_range, 1)
        self.assertEqual(result.fallback.cells_excluded_by_range, 1)

    def test_incomplete_or_mismatched_localization_is_rejected(self):
        _, interpolator, solver = self.make_affine_localized_solver()
        target = np.asarray([0.75, 0.25])
        index, complete = self.localization(interpolator, solver, [target])
        incomplete = ReachabilityLocalization(
            result=complete.result,
            offsets=complete.offsets,
            cell_flat_indices=complete.cell_flat_indices,
            completeness=LocalizationCompleteness.INCOMPLETE,
            provenance=complete.provenance,
        )

        with self.assertRaisesRegex(SolverInputError, "complete matching"):
            solver.solve_with_candidate_cells(
                target,
                incomplete,
                0,
                expected_provenance=index.provenance,
            )
        incompatible = replace(
            index.provenance,
            active_cell_fingerprint="different-active-domain",
        )
        with self.assertRaisesRegex(SolverInputError, "provenance"):
            solver.solve_with_candidate_cells(
                target,
                complete,
                0,
                expected_provenance=incompatible,
            )
        with self.assertRaisesRegex(SolverInputError, "bare cell IDs"):
            solver.solve_with_candidate_cells(
                target,
                complete.cell_flat_indices,
                0,
                expected_provenance=index.provenance,
            )

    def test_candidate_limit_cannot_create_restricted_no_root_proof(self):
        config = SolverConfig(
            fallback_max_candidate_cells=1,
            fallback_max_boxes=64,
            fallback_max_depth=8,
            enable_subcell_fallback=False,
        )
        def repeated_diamond(first, second):
            first_tent = 2.0 * first if first <= 0.5 else 2.0 * (1.0 - first)
            second_tent = (
                2.0 * second if second <= 0.5 else 2.0 * (1.0 - second)
            )
            return first_tent + second_tent, first_tent - second_tent

        _, interpolator, solver = self.make_solver(
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            repeated_diamond,
            config=config,
            residual_scales=[1.0, 1.0],
        )
        target = np.asarray([0.25, 0.75])
        index, localization = self.localization(interpolator, solver, [target])
        self.assertGreater(localization.pair_count, 1)

        result = solver.solve_with_candidate_cells(
            target,
            localization,
            0,
            expected_provenance=index.provenance,
        )

        self.assertEqual(result.status, SolveStatus.UNRESOLVED)
        self.assertEqual(result.proof_status, ProofStatus.INCONCLUSIVE)
        self.assertGreater(result.fallback.unprocessed_candidate_cells, 0)
        self.assertNotEqual(result.fallback.status, FallbackStatus.NO_ROOT_IN_DOMAIN)


class SolverContractTests(SolverTestCase):
    def test_subcell_configuration_is_explicit_and_bounded(self):
        self.assertTrue(SolverConfig().enable_subcell_fallback)
        self.assertEqual(SolverConfig().max_subcell_depth, 7)
        with self.assertRaises(TypeError):
            SolverConfig(enable_subcell_fallback=1)
        with self.assertRaises(ValueError):
            SolverConfig(max_subcell_depth=8)
        with self.assertRaises(ValueError):
            SolverConfig(max_subcell_active_boxes=0)

    def test_solver_rejects_noncanonical_field_order_and_bad_initial_guess(self):
        table, _, solver = self.make_affine_solver()
        with self.assertRaises(SolverConfigurationError):
            EoSRootSolver(MultilinearInterpolator(table, ("nB", "e")))
        with self.assertRaises(SolverInputError):
            solver.solve([0.75, 0.25], [2.0, 0.5])
        with self.assertRaises(SolverInputError):
            solver.solve([0.75])
        with self.assertRaises(SolverInputError):
            solver.solve([0.75, 0.25], strategy="unknown")

    def test_node_bounds_are_rigorous_and_immutable(self):
        _, interpolator, _ = self.make_affine_solver()
        bounds = interpolator.node_value_bounds
        self.assertTrue(bounds.rigorous)
        np.testing.assert_array_equal(bounds.lower, [0.0, -1.0])
        np.testing.assert_array_equal(bounds.upper, [2.0, 1.0])
        with self.assertRaises(ValueError):
            bounds.lower[0] = -1.0

    def test_invalid_backend_tolerances_and_tiny_residual_scales_are_rejected(self):
        with self.assertRaises(ValueError):
            SolverConfig(
                least_squares_ftol=1.0e-30,
                least_squares_xtol=1.0e-30,
                least_squares_gtol=1.0e-30,
            )
        _, interpolator, _ = self.make_affine_solver()
        with self.assertRaises(SolverConfigurationError):
            EoSRootSolver(interpolator, residual_scales=[5.0e-324, 1.0])

    def test_interval_zero_times_unbounded_and_subnormal_midpoint_are_safe(self):
        zero = _Interval(np.asarray(0.0), np.asarray(0.0))
        unbounded = _Interval(np.asarray(-np.inf), np.asarray(np.inf))
        product = _interval_multiply(zero, unbounded)
        self.assertEqual(float(product.lower), 0.0)
        self.assertEqual(float(product.upper), 0.0)

        smallest = np.nextafter(0.0, 1.0)
        midpoint = _bounded_midpoint(
            np.asarray([smallest]), np.asarray([smallest])
        )
        np.testing.assert_array_equal(midpoint, [smallest])

    def test_extreme_normalized_upper_face_maps_to_exact_axis_endpoint(self):
        lower = -5.592802544547773e299
        upper = 9.028825972070647e299
        _, _, solver = self.make_solver(
            [lower, upper],
            [0.0, 1.0],
            lambda temperature, chemical_potential: (
                0.0 if temperature == lower else 1.0,
                chemical_potential,
            ),
            residual_scales=[1.0, 1.0],
        )

        result = solver.solve(
            [1.0, 1.0], [upper, 1.0], strategy="fallback"
        )

        self.assertEqual(result.status, SolveStatus.UNRESOLVED)
        self.assertNotEqual(result.status, SolveStatus.CERTIFIED_NO_ROOT)

    def test_failed_runtime_capability_check_disables_interval_proofs(self):
        _, _, solver = self.make_affine_solver()
        solver._interval_certification_available = False

        result = solver.solve([0.75, 0.25], strategy="fallback")

        self.assertEqual(result.status, SolveStatus.UNRESOLVED)
        self.assertEqual(result.proof_status, ProofStatus.INCONCLUSIVE)
        self.assertIsNone(result.certificate)
        self.assertFalse(result.fallback.outward_rounded)


if __name__ == "__main__":
    unittest.main()
