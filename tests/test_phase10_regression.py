"""Phase-10 boundary, low-dimensional, and solver edge-case regressions."""

import itertools
import unittest

import numpy as np

from eos_trensmuther import (
    EoSInversionPipeline,
    EoSRootSolver,
    ExecutionConfig,
    FallbackStatus,
    ForwardTable,
    GridAxisSpec,
    InterpolationDomainError,
    InversionGridSpec,
    MultilinearInterpolator,
    PipelineConfig,
    ProjectionConfig,
    ProofStatus,
    ReachabilityIndex,
    SolveStatus,
    SolverConfig,
)
from tests.test_tables import TemporaryEoS, field_spec, metadata


COORDINATE_NAMES = ("T", "muB", "muQ", "muS")
TARGET_NAMES = ("e", "nB", "nQ", "nS")


class Phase10RegressionTests(unittest.TestCase):
    def make_table(self, axes, function):
        dimension = len(axes)
        fields = {
            name: field_spec("coordinate", index)
            for index, name in enumerate(COORDINATE_NAMES[:dimension])
        }
        fields.update(
            {
                name: field_spec("target", dimension + index)
                for index, name in enumerate(TARGET_NAMES[:dimension])
            }
        )
        points = np.asarray(list(itertools.product(*axes)), dtype=float)
        targets = np.asarray([function(*point) for point in points], dtype=float)
        fixture = TemporaryEoS(
            metadata(fields, dimension), np.column_stack((points, targets))
        )
        self.addCleanup(fixture.close)
        return ForwardTable.load_validated(
            fixture.metadata_path, _allow_unsuitable_geometry=True
        )

    def make_identity_table(self, dimension, *, node_count=3):
        axes = tuple(
            np.linspace(0.0, 1.0, node_count, dtype=float)
            for _ in range(dimension)
        )
        return self.make_table(axes, lambda *point: tuple(point))

    def test_closed_nodes_faces_corners_ulps_and_outside_targets_in_1d_to_4d(self):
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                table = self.make_identity_table(dimension)
                interpolator = MultilinearInterpolator(table)
                nodes = np.asarray(list(itertools.product(*table.axes)))
                np.testing.assert_array_equal(interpolator.evaluate(nodes), nodes)

                lower_corner = np.zeros(dimension)
                upper_corner = np.ones(dimension)
                face = np.full(dimension, 0.25)
                face[0] = 0.5
                below_face = face.copy()
                below_face[0] = np.nextafter(0.5, -np.inf)
                above_face = face.copy()
                above_face[0] = np.nextafter(0.5, np.inf)
                below_domain = face.copy()
                below_domain[0] = np.nextafter(0.0, -np.inf)
                above_domain = face.copy()
                above_domain[0] = np.nextafter(1.0, np.inf)
                targets = np.asarray(
                    (
                        lower_corner,
                        upper_corner,
                        face,
                        below_face,
                        above_face,
                        below_domain,
                        above_domain,
                    )
                )

                strict = ReachabilityIndex(
                    interpolator,
                    np.ones(dimension),
                    0.0,
                )
                localized = strict.localize(targets)
                np.testing.assert_array_equal(
                    localized.candidate_counts,
                    np.asarray([1, 1, 2, 1, 1, 0, 0]),
                )
                np.testing.assert_array_equal(
                    localized.result.exact_candidate_mask,
                    np.asarray([True, True, True, True, True, False, False]),
                )
                face_cells = localized.cells_for(2)
                self.assertEqual(face_cells.size, 2)
                np.testing.assert_array_equal(localized.cells_for(3), face_cells[:1])
                np.testing.assert_array_equal(localized.cells_for(4), face_cells[1:])

                for point in (below_domain, above_domain):
                    with self.assertRaises(InterpolationDomainError):
                        interpolator.evaluate(point)

                # Production routing includes the solver's accepted residual
                # shell. One-ULP exterior targets are therefore candidates,
                # but remain explicitly outside every exact cell image.
                routed = ReachabilityIndex(
                    interpolator,
                    np.ones(dimension),
                    1.0e-10,
                ).classify(
                    np.asarray(
                        (
                            below_domain,
                            above_domain,
                            np.where(
                                np.arange(dimension) == 0,
                                -1.0e-8,
                                face,
                            ),
                            np.where(
                                np.arange(dimension) == 0,
                                1.0 + 1.0e-8,
                                face,
                            ),
                        )
                    )
                )
                np.testing.assert_array_equal(
                    routed.candidate_mask, [True, True, False, False]
                )
                np.testing.assert_array_equal(
                    routed.exact_candidate_mask, [False, False, False, False]
                )
                np.testing.assert_array_equal(
                    routed.tolerance_shell_mask, [True, True, False, False]
                )

    def test_complete_localization_preserves_one_and_large_multiplicity(self):
        unique = MultilinearInterpolator(self.make_identity_table(4, node_count=5))
        unique_localization = ReachabilityIndex(
            unique, np.ones(4), 0.0
        ).localize([[0.13, 0.37, 0.61, 0.83]])
        np.testing.assert_array_equal(unique_localization.candidate_counts, [1])

        constant_table = self.make_table(
            tuple(np.linspace(0.0, 1.0, 5) for _ in range(4)),
            lambda *_: (0.0, 0.0, 0.0, 0.0),
        )
        constant = MultilinearInterpolator(constant_table)
        many = ReachabilityIndex(constant, np.ones(4), 0.0).localize(
            [[0.0, 0.0, 0.0, 0.0]]
        )
        self.assertEqual(int(many.candidate_counts[0]), 4**4)
        np.testing.assert_array_equal(many.cells_for(0), np.arange(4**4))
        many.require_complete_for(
            ReachabilityIndex(constant, np.ones(4), 0.0).provenance
        )

    def test_tiny_density_and_ill_conditioned_roots_remain_solvable(self):
        tiny = 1.0e-300
        tiny_table = self.make_table(
            (np.asarray([0.1, 0.9]), np.asarray([-0.5, 0.5])),
            lambda temperature, chemical: (
                temperature,
                tiny * (1.0 + chemical),
            ),
        )
        tiny_interpolator = MultilinearInterpolator(tiny_table)
        tiny_solver = EoSRootSolver(
            tiny_interpolator,
            residual_scales=[1.0, tiny],
        )
        tiny_root = np.asarray([0.37, 0.23])
        tiny_result = tiny_solver.solve(
            tiny_interpolator.evaluate(tiny_root),
            [0.5, 0.0],
            strategy="auto",
        )
        self.assertEqual(tiny_result.status, SolveStatus.CONVERGED_FAST)
        self.assertLessEqual(
            tiny_result.scaled_residual_norm,
            tiny_solver.config.root_residual_tolerance,
        )
        np.testing.assert_allclose(tiny_result.solution, tiny_root, atol=2.0e-12)

        matrix = np.asarray([[1.0, 1.0], [1.0, 1.0 + 1.0e-7]])
        self.assertGreater(np.linalg.cond(matrix, p=np.inf), 1.0e7)
        ill_table = self.make_table(
            (np.asarray([0.0, 0.5, 1.0]),) * 2,
            lambda first, second: tuple(matrix @ np.asarray([first, second])),
        )
        ill_interpolator = MultilinearInterpolator(ill_table)
        ill_solver = EoSRootSolver(
            ill_interpolator,
            residual_scales=[1.0, 1.0],
        )
        known_root = np.asarray([0.27, 0.61])
        ill_result = ill_solver.solve(
            ill_interpolator.evaluate(known_root),
            [0.1, 0.9],
            strategy="auto",
        )
        self.assertEqual(ill_result.status, SolveStatus.CONVERGED_FAST)
        self.assertLessEqual(
            ill_result.scaled_residual_norm,
            ill_solver.config.root_residual_tolerance,
        )
        inverse_norm = np.linalg.norm(np.linalg.inv(matrix), ord=np.inf)
        coordinate_tolerance = max(
            4.0 * inverse_norm * ill_solver.config.root_residual_tolerance,
            64.0 * np.finfo(float).eps,
        )
        self.assertLessEqual(
            float(np.max(np.abs(ill_result.solution - known_root))),
            coordinate_tolerance,
        )

    def test_fast_failure_no_root_and_guarded_work_keep_proof_semantics(self):
        table = self.make_table(
            (np.asarray([0.5, 1.5]), np.asarray([0.0, 1.0])),
            lambda temperature, chemical: (
                temperature - 0.5 + chemical,
                temperature - 0.5 - chemical,
            ),
        )
        interpolator = MultilinearInterpolator(table)
        config = SolverConfig(
            enable_subcell_fallback=True,
            max_subcell_depth=3,
            max_subcell_active_boxes=512,
        )
        solver = EoSRootSolver(
            interpolator, config, residual_scales=[1.0, 1.0]
        )
        no_root_target = np.asarray([0.24, 0.26])
        fast = solver.solve(no_root_target, strategy="fast")
        self.assertEqual(fast.status, SolveStatus.FAST_PATH_FAILED)

        index = ReachabilityIndex(
            interpolator,
            [1.0, 1.0],
            config.root_residual_tolerance,
        )
        no_root_localization = index.localize([no_root_target])
        no_root = solver.solve_with_candidate_cells(
            no_root_target,
            no_root_localization,
            0,
            expected_provenance=index.provenance,
            strategy="auto",
        )
        self.assertEqual(no_root.status, SolveStatus.CERTIFIED_NO_ROOT)
        self.assertEqual(no_root.proof_status, ProofStatus.NO_ROOT_IN_DOMAIN)
        self.assertEqual(no_root.fallback.status, FallbackStatus.NO_ROOT_IN_DOMAIN)

        guard_config = SolverConfig(
            enable_subcell_fallback=True,
            max_subcell_depth=2,
            max_subcell_active_boxes=1,
        )
        guarded_solver = EoSRootSolver(
            interpolator, guard_config, residual_scales=[1.0, 1.0]
        )
        guarded_target = np.asarray([0.65, 0.15])
        guard_index = ReachabilityIndex(
            interpolator,
            [1.0, 1.0],
            guard_config.root_residual_tolerance,
        )
        guarded = guarded_solver.solve_with_candidate_cells(
            guarded_target,
            guard_index.localize([guarded_target]),
            0,
            expected_provenance=guard_index.provenance,
            strategy="fallback",
        )
        self.assertEqual(guarded.status, SolveStatus.UNRESOLVED)
        self.assertEqual(guarded.proof_status, ProofStatus.INCONCLUSIVE)
        self.assertEqual(guarded.fallback.status, FallbackStatus.UNRESOLVED)
        self.assertFalse(guarded.fallback.subcell_report.complete)

    def test_integrated_ordinary_pipeline_solves_dimensions_one_through_four(self):
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                table = self.make_identity_table(dimension)
                config = PipelineConfig(
                    grid=InversionGridSpec(
                        "target",
                        {
                            name: GridAxisSpec(3, lower=0.0, upper=1.0)
                            for name in TARGET_NAMES[:dimension]
                        },
                    ),
                    residual_scales={
                        name: 1.0 for name in TARGET_NAMES[:dimension]
                    },
                    projection=ProjectionConfig(enabled=False),
                    execution=ExecutionConfig(workers=1, chunk_size=17),
                )
                self.assertTrue(config.solver.enable_subcell_fallback)
                result = EoSInversionPipeline(table, config).run()
                expected = result.grid.chunk(
                    0, result.grid.size
                ).canonical_targets.reshape(result.grid.shape + (dimension,))

                self.assertTrue(np.all(result.solution_mask))
                self.assertFalse(np.any(result.projected_mask))
                self.assertEqual(result.manifest["run"]["errors"]["count"], 0)
                self.assertTrue(
                    np.all(
                        result.scaled_residual_norm_inf
                        <= config.solver.root_residual_tolerance
                    )
                )
                np.testing.assert_allclose(
                    result.source_coordinates,
                    expected,
                    rtol=0.0,
                    atol=2.0e-11,
                )


if __name__ == "__main__":
    unittest.main()
