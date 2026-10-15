"""Tests for restricted phase solves and deterministic mixed-root routing."""

import unittest

import numpy as np

from eos_trensmuther import (
    BinodalBand,
    BinodalSubdivisionConfig,
    EoSRootSolver,
    MultilinearInterpolator,
    PhaseAwareRootSolver,
    PhaseLabel,
    PhaseRootRoute,
    PhaseSolverAmbiguityError,
    PhaseSolverConfig,
    PhaseSolverResourceError,
    SolveStatus,
    SolverConfig,
    build_region_s,
)
from tests import test_region_s as region_s_fixtures


_FIRST_ORDER_SOLVER_CONFIG = SolverConfig(enable_subcell_fallback=False)


def folded_interpolator():
    temperature = np.asarray([0.0, 0.5, 1.0])
    chemical_potential = np.asarray([0.0, 1.0])
    T, muB = np.meshgrid(temperature, chemical_potential, indexing="ij")
    energy = np.broadcast_to(np.asarray([0.0, 1.0, 0.0])[:, None], T.shape)
    return MultilinearInterpolator.from_rectilinear_map(
        (temperature, chemical_potential),
        {"e": energy, "nB": muB},
        coordinate_names=("T", "muB"),
        target_names=("e", "nB"),
        coordinate_units=("GeV", "GeV"),
        target_units=("GeV^4", "GeV^3"),
    )


def identity_interpolator():
    temperature = np.asarray([0.0, 0.5, 1.0])
    chemical_potential = np.asarray([0.0, 1.0])
    T, muB = np.meshgrid(temperature, chemical_potential, indexing="ij")
    return MultilinearInterpolator.from_rectilinear_map(
        (temperature, chemical_potential),
        {"e": T, "nB": muB},
        coordinate_names=("T", "muB"),
        target_names=("e", "nB"),
        coordinate_units=("GeV", "GeV"),
        target_units=("GeV^4", "GeV^3"),
    )


def horizontal_band(lower=0.7, upper=0.8):
    return BinodalBand(
        muB=np.asarray([0.0, 1.0]),
        dilute_temperature=np.asarray([lower, lower]),
        dense_temperature=np.asarray([upper, upper]),
        sidecar_fingerprint="synthetic-phase-solver-band",
        kind="synthetic horizontal band",
    )


class RestrictedCellSolverTests(unittest.TestCase):
    def test_fallback_is_restricted_to_supplied_cell_box(self):
        interpolator = identity_interpolator()
        solver = EoSRootSolver(interpolator, residual_scales=[1.0, 1.0])

        excluded = solver.solve_in_cell_box(
            [0.75, 0.5],
            [0, 0],
            [0.0, 0.0],
            [0.5, 1.0],
            strategy="fallback",
        )
        included = solver.solve_in_cell_box(
            [0.25, 0.5],
            [0, 0],
            [0.0, 0.0],
            [0.5, 1.0],
            strategy="fallback",
        )

        self.assertEqual(excluded.status, SolveStatus.CERTIFIED_NO_ROOT)
        self.assertEqual(included.status, SolveStatus.CONVERGED_FALLBACK)
        np.testing.assert_allclose(included.solution, [0.25, 0.5], atol=1.0e-12)
        self.assertEqual(
            included.fallback_newton.normalized_coordinate_system,
            "interpolation_cell",
        )


class PhaseAwareRoutingTests(unittest.TestCase):
    def test_duplicate_source_tolerance_collapses_solver_jitter(self):
        ordinary = EoSRootSolver(
            identity_interpolator(),
            _FIRST_ORDER_SOLVER_CONFIG,
            residual_scales=[1.0, 1.0],
        )
        solver = PhaseAwareRootSolver(ordinary)

        self.assertTrue(
            solver._source_close(
                np.asarray([0.10698324913804576, 0.7451101075918518]),
                np.asarray([0.10698324932855649, 0.7451101065334654]),
            )
        )
        self.assertFalse(
            solver._source_close(
                np.asarray([0.106983249, 0.745110107]),
                np.asarray([0.106983249, 0.745210107]),
            )
        )

    def test_no_transition_delegates_to_ordinary_solver(self):
        interpolator = identity_interpolator()
        ordinary = EoSRootSolver(
            interpolator,
            _FIRST_ORDER_SOLVER_CONFIG,
            residual_scales=[1.0, 1.0],
        )
        expected = ordinary.solve([0.3, 0.4], [0.2, 0.3])

        routed = PhaseAwareRootSolver(ordinary).solve(
            [0.3, 0.4], [0.2, 0.3]
        )

        self.assertEqual(routed.phase_label, PhaseLabel.NOT_APPLICABLE)
        self.assertEqual(routed.route, PhaseRootRoute.NO_TRANSITION)
        self.assertEqual(routed.selected_result.status, expected.status)
        self.assertEqual(routed.selected_result.path, expected.path)
        np.testing.assert_array_equal(
            routed.selected_result.solution, expected.solution
        )
        np.testing.assert_array_equal(routed.source_coordinates, expected.solution)

    def test_binodal_mixed_root_precedes_distinct_ordinary_root(self):
        solver = PhaseAwareRootSolver.for_binodals(
            folded_interpolator(),
            horizontal_band(),
            [1.0, 1.0],
            _FIRST_ORDER_SOLVER_CONFIG,
            subdivision=BinodalSubdivisionConfig(
                maximum_depth=8,
                temperature_tolerance=0.005,
                chemical_potential_tolerance=0.005,
            ),
        )

        result = solver.solve([0.5, 0.5], initial_guess=[0.25, 0.5])

        self.assertTrue(result.solved)
        self.assertEqual(result.phase_label, PhaseLabel.MIXED_PHASE)
        self.assertEqual(result.route, PhaseRootRoute.BINODALS_ORDINARY)
        np.testing.assert_allclose(result.source_coordinates, [0.75, 0.5])
        np.testing.assert_allclose(
            result.alternate_single_phase_root, [0.25, 0.5]
        )
        self.assertTrue(result.multivalued)
        self.assertIsNone(result.line_mixture_fraction)
        self.assertGreater(result.mixed_candidate_entry_count, 0)

    def test_conservative_binodal_false_candidate_remains_single_phase(self):
        interpolator = identity_interpolator()
        band = BinodalBand(
            muB=np.asarray([0.0, 1.0]),
            dilute_temperature=np.asarray([0.4, 0.45]),
            dense_temperature=np.asarray([0.6, 0.65]),
            sidecar_fingerprint="synthetic-sloped-band",
            kind="synthetic sloped band",
        )
        solver = PhaseAwareRootSolver.for_binodals(
            interpolator,
            band,
            [1.0, 1.0],
            _FIRST_ORDER_SOLVER_CONFIG,
            subdivision=BinodalSubdivisionConfig(
                maximum_depth=0,
                temperature_tolerance=0.0,
                chemical_potential_tolerance=0.0,
            ),
        )

        result = solver.solve([0.1, 0.25], initial_guess=[0.1, 0.25])

        self.assertTrue(result.mixed_candidate)
        self.assertEqual(result.mixed_verified_root_count, 0)
        self.assertEqual(result.phase_label, PhaseLabel.SINGLE_PHASE)
        self.assertEqual(result.route, PhaseRootRoute.ORDINARY)
        np.testing.assert_allclose(result.source_coordinates, [0.1, 0.25])

    def test_binodal_boundary_is_inclusively_mixed_without_alternate(self):
        solver = PhaseAwareRootSolver.for_binodals(
            identity_interpolator(),
            horizontal_band(0.25, 0.75),
            [1.0, 1.0],
            _FIRST_ORDER_SOLVER_CONFIG,
            subdivision=BinodalSubdivisionConfig(maximum_depth=6),
        )

        result = solver.solve([0.25, 0.5], initial_guess=[0.25, 0.5])

        self.assertEqual(result.phase_label, PhaseLabel.MIXED_PHASE)
        self.assertTrue(result.phase_boundary)
        self.assertFalse(result.multivalued)
        self.assertIsNone(result.alternate_single_phase_root)

    def test_line_region_s_root_returns_physical_coordinates_and_fraction(self):
        fixture = region_s_fixtures.SyntheticRegionSTests()
        self.addCleanup(fixture.doCleanups)
        table, _ = fixture.make_line_table()
        strip = build_region_s(table, fixture.exact_config())
        ordinary = MultilinearInterpolator(table, ("e", "nB"))
        solver = PhaseAwareRootSolver.for_line(
            ordinary,
            strip,
            [1.0e-3, 1.0e-3],
            _FIRST_ORDER_SOLVER_CONFIG,
        )
        target = strip.evaluate(0.35, 0.4)

        result = solver.solve(target, initial_guess=[0.15, 0.35])

        self.assertEqual(result.phase_label, PhaseLabel.MIXED_PHASE)
        self.assertEqual(result.route, PhaseRootRoute.LINE_REGION_S)
        np.testing.assert_allclose(result.parameter_coordinates, [0.35, 0.4])
        np.testing.assert_allclose(result.source_coordinates, [0.145, 0.35])
        self.assertAlmostEqual(result.line_mixture_fraction, 0.4, places=10)

    def test_candidate_limit_fails_explicitly(self):
        solver = PhaseAwareRootSolver.for_binodals(
            folded_interpolator(),
            horizontal_band(),
            [1.0, 1.0],
            _FIRST_ORDER_SOLVER_CONFIG,
            subdivision=BinodalSubdivisionConfig(maximum_depth=6),
            config=PhaseSolverConfig(maximum_candidate_images_per_target=1),
        )

        with self.assertRaises(PhaseSolverResourceError):
            solver.solve([0.5, 0.5], initial_guess=[0.25, 0.5])

    def test_two_distinct_mixed_roots_fail_without_a_physics_policy(self):
        solver = PhaseAwareRootSolver.for_binodals(
            folded_interpolator(),
            horizontal_band(0.2, 0.8),
            [1.0, 1.0],
            _FIRST_ORDER_SOLVER_CONFIG,
            subdivision=BinodalSubdivisionConfig(maximum_depth=6),
        )

        with self.assertRaisesRegex(
            PhaseSolverAmbiguityError, "distinct verified mixed roots"
        ):
            solver.solve([0.5, 0.5], initial_guess=[0.25, 0.5])


if __name__ == "__main__":
    unittest.main()
