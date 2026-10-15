"""Phase-8 scientific parity across serial, two-, and four-process runs."""

import itertools
import unittest
from dataclasses import replace

import numpy as np

from eos_trensmuther import (
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    PipelineConfig,
    PointDisposition,
    ProjectionConfig,
    SolverConfig,
    pipeline_status_code,
)

from tests.test_tables import TemporaryEoS, field_spec, metadata


_EXACT_SCIENTIFIC_FIELDS = (
    "disposition_codes",
    "prescreen_codes",
    "proof_status_codes",
)


class Phase8ExecutionParityTests(unittest.TestCase):
    def make_table(self, axes, function, *, allow_unsuitable_geometry=False):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }
        points = np.asarray(list(itertools.product(*axes)), dtype=float)
        targets = np.asarray([function(*point) for point in points], dtype=float)
        fixture = TemporaryEoS(
            metadata(fields, 2), np.column_stack((points, targets))
        )
        self.addCleanup(fixture.close)
        return ForwardTable.load_validated(
            fixture.metadata_path,
            _allow_unsuitable_geometry=allow_unsuitable_geometry,
        )

    @staticmethod
    def config(
        table,
        axes,
        *,
        projection=False,
        solver=None,
        workers=1,
    ):
        return PipelineConfig(
            grid=InversionGridSpec(
                "target",
                {
                    name: GridAxisSpec(count, lower=lower, upper=upper)
                    for name, (count, lower, upper) in axes.items()
                },
            ),
            residual_scales={"e": 1.0, "nB": 1.0},
            projection=ProjectionConfig(
                enabled=projection,
                nearest_cell_starts=4,
                max_evaluations_per_start=60,
            ),
            solver=SolverConfig() if solver is None else solver,
            execution=ExecutionConfig(
                workers=workers,
                start_method="spawn",
                chunk_size=12,
                worker_task_size=8,
                max_pending_chunks=max(1, 2 * workers),
            ),
            use_reachability_seed=True,
        )

    @staticmethod
    def run_modes(table, config):
        return {
            workers: EoSInversionPipeline(
                table,
                replace(
                    config,
                    execution=replace(config.execution, workers=workers),
                ),
            ).run()
            for workers in (1, 2, 4)
        }

    def assert_scientific_parity(self, table, results, *, unique_roots):
        reference = results[1]
        solved_code = pipeline_status_code(PointDisposition.SOLVED)
        solved = reference.disposition_codes == solved_code
        lower = np.asarray([axis[0] for axis in table.axes], dtype=float)
        upper = np.asarray([axis[-1] for axis in table.axes], dtype=float)
        span = upper - lower
        tolerance = max(
            10.0 * reference.plan.config.solver.newton_step_tolerance,
            1.0e-11,
        )
        self.assertTrue(np.all(reference.source_coordinates[solved] >= lower))
        self.assertTrue(np.all(reference.source_coordinates[solved] <= upper))
        self.assertTrue(
            np.all(
                reference.scaled_residual_norm_inf[solved]
                <= reference.plan.config.solver.root_residual_tolerance
            )
        )
        for workers in (2, 4):
            candidate = results[workers]
            for name in _EXACT_SCIENTIFIC_FIELDS:
                with self.subTest(workers=workers, field=name):
                    np.testing.assert_array_equal(
                        getattr(reference, name), getattr(candidate, name)
                    )
            candidate_solved = candidate.disposition_codes == solved_code
            np.testing.assert_array_equal(solved, candidate_solved)
            self.assertTrue(
                np.all(candidate.source_coordinates[candidate_solved] >= lower)
            )
            self.assertTrue(
                np.all(candidate.source_coordinates[candidate_solved] <= upper)
            )
            self.assertTrue(
                np.all(
                    candidate.scaled_residual_norm_inf[candidate_solved]
                    <= candidate.plan.config.solver.root_residual_tolerance
                )
            )
            if unique_roots and np.any(solved):
                normalized_error = np.max(
                    np.abs(
                        (
                            candidate.source_coordinates[solved]
                            - reference.source_coordinates[solved]
                        )
                        / span
                    ),
                    axis=1,
                )
                self.assertLessEqual(float(np.max(normalized_error)), tolerance)
            else:
                # A folded map may select either valid branch. Root identity is
                # not a scientific parity requirement without a branch policy.
                self.assertTrue(
                    np.all(
                        candidate.scaled_residual_norm_inf[candidate_solved]
                        <= candidate.plan.config.solver.root_residual_tolerance
                    )
                )

    def test_globally_unique_affine_roots_match_at_strict_coordinate_tolerance(self):
        table = self.make_table(
            ((0.0, 0.5, 1.0), (0.0, 0.5, 1.0)),
            lambda temperature, chemical: (temperature, chemical),
        )
        config = self.config(
            table,
            {"e": (7, 0.0, 1.0), "nB": (7, 0.0, 1.0)},
        )
        results = self.run_modes(table, config)
        self.assert_scientific_parity(table, results, unique_roots=True)
        expected = results[1].grid.chunk(0, results[1].grid.size).canonical_targets
        np.testing.assert_allclose(
            results[1].source_coordinates.reshape(-1, 2),
            expected,
            rtol=1.0e-11,
            atol=1.0e-11,
        )

    def test_projection_subcells_guard_and_folded_map_preserve_semantics(self):
        correlated = self.make_table(
            ((1.0, 2.0, 3.0), (0.0, 1.0, 2.0)),
            lambda temperature, chemical: (
                temperature,
                temperature + chemical,
            ),
        )
        projection = self.config(
            correlated,
            {"e": (3, 1.0, 3.0), "nB": (3, 1.0, 5.0)},
            projection=True,
        )
        projection_results = self.run_modes(correlated, projection)
        self.assert_scientific_parity(
            correlated, projection_results, unique_roots=True
        )
        self.assertGreater(
            projection_results[1].manifest["run"]["stage_counters"][
                "prescreen_excluded"
            ],
            0,
        )
        self.assertGreater(
            projection_results[1].manifest["run"]["stage_counters"][
                "projection_attempted"
            ],
            0,
        )

        diamond = self.make_table(
            ((0.5, 1.5), (0.0, 1.0)),
            lambda temperature, chemical: (
                temperature - 0.5 + chemical,
                temperature - 0.5 - chemical,
            ),
        )
        proof_solver = SolverConfig(
            enable_subcell_fallback=True,
            max_subcell_depth=3,
            max_subcell_active_boxes=512,
        )
        proof_config = self.config(
            diamond,
            # Stay off exact source-domain faces: boundary roots deliberately
            # retain conservative unresolved semantics and are not a useful
            # execution-scheduling parity fixture.
            {"e": (7, 0.1, 1.9), "nB": (7, -0.9, 0.9)},
            solver=proof_solver,
        )
        proof_results = self.run_modes(diamond, proof_config)
        self.assert_scientific_parity(diamond, proof_results, unique_roots=True)
        proof_counters = proof_results[1].manifest["run"]["stage_counters"]
        self.assertGreater(proof_counters["subcell_fallback_attempted"], 0)
        self.assertGreater(proof_counters["subcell_certified_no_root"], 0)

        guard_config = self.config(
            diamond,
            {"e": (3, 0.6, 0.7), "nB": (3, 0.1, 0.2)},
            solver=replace(
                proof_solver,
                least_squares_max_evaluations=1,
                newton_max_iterations=1,
                max_subcell_depth=2,
                max_subcell_active_boxes=1,
            ),
        )
        guard_results = self.run_modes(diamond, guard_config)
        self.assert_scientific_parity(diamond, guard_results, unique_roots=True)
        guard_counters = guard_results[1].manifest["run"]["stage_counters"]
        self.assertGreater(guard_counters["subcell_guarded_incomplete"], 0)
        unresolved_code = pipeline_status_code(PointDisposition.UNRESOLVED)
        self.assertGreater(
            int(np.count_nonzero(guard_results[1].disposition_codes == unresolved_code)),
            0,
        )

        folded = self.make_table(
            ((0.0, 1.0, 2.0), (0.0, 1.0)),
            lambda temperature, chemical: (abs(temperature - 1.0), chemical),
            # This regression deliberately exercises non-injective solver
            # semantics below the production table-validation boundary.
            allow_unsuitable_geometry=True,
        )
        folded_config = self.config(
            folded,
            {"e": (5, 0.0, 1.0), "nB": (5, 0.0, 1.0)},
        )
        folded_results = self.run_modes(folded, folded_config)
        self.assert_scientific_parity(folded, folded_results, unique_roots=False)


if __name__ == "__main__":
    unittest.main()
