"""Focused end-to-end tests for the Phase-4 serial inversion pipeline."""

import itertools
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from eos_trensmuther import (
    EoSInversionPipeline,
    ErrorPolicy,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InitialGuessSource,
    InversionGridSpec,
    PIPELINE_RESULT_SCHEMA_VERSION,
    PhaseLabel,
    PhaseRoute,
    PipelineConfig,
    PipelineConfigurationError,
    PipelineStateError,
    PointDisposition,
    PointRoute,
    PrescreenStatus,
    ProjectionConfig,
    SolveStatus,
    SolverConfig,
    pipeline_status_code,
)

from tests.test_tables import TemporaryEoS, field_spec, metadata


class SerialPipelineTestCase(unittest.TestCase):
    def make_table(self, fields, rows):
        dimension = sum(
            specification["role"] == "coordinate"
            for specification in fields.values()
        )
        fixture = TemporaryEoS(metadata(fields, dimension), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable(
            fixture.metadata_path, _allow_unsuitable_geometry=True
        )
        self.assertTrue(table.validate().is_valid)
        return table

    @staticmethod
    def make_config(
        table,
        *,
        space="target",
        count=2,
        chunk_size=2,
        projection_enabled=True,
        residual_scales=None,
        auxiliary_fields=(),
        error_policy=ErrorPolicy.FAIL_FAST,
    ):
        envelope = table.find_boundaries(space=space)
        axes = {name: GridAxisSpec(count) for name in envelope.names}
        scales = (
            {
                name: float(index + 1)
                for index, name in enumerate(table.target_names)
            }
            if residual_scales is None
            else residual_scales
        )
        return PipelineConfig(
            grid=InversionGridSpec(space, axes),
            residual_scales=scales,
            projection=ProjectionConfig(
                enabled=projection_enabled,
                nearest_cell_starts=4,
                max_evaluations_per_start=60,
            ),
            execution=ExecutionConfig(
                workers=1,
                chunk_size=chunk_size,
                error_policy=error_policy,
            ),
            auxiliary_fields=auxiliary_fields,
        )

    @staticmethod
    def code(value):
        return pipeline_status_code(value)


class FastSerialPipelineTests(SerialPipelineTestCase):
    def make_affine_table(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "P": field_spec("auxiliary", 2),
        }
        rows = []
        for temperature in (0.2, 0.5, 0.8):
            energy = 1.0 + 2.0 * temperature
            rows.append((temperature, energy, 3.0 * energy))
        return self.make_table(fields, rows)

    def test_fast_auxiliary_manifest_and_chunk_size_determinism(self):
        table = self.make_affine_table()
        results = []
        for chunk_size in (1, 4):
            config = self.make_config(
                table,
                count=5,
                chunk_size=chunk_size,
                auxiliary_fields=("P",),
            )
            pipeline = EoSInversionPipeline(table, config)
            self.assertIs(pipeline.plan(), pipeline.plan())
            results.append(pipeline.run())

        result = results[0]
        requested = result.grid.chunk(0, result.grid.size).canonical_targets
        expected_source = (requested[:, 0] - 1.0) / 2.0

        self.assertTrue(np.all(result.solution_mask))
        self.assertTrue(np.all(result.candidate_mask))
        self.assertFalse(np.any(result.projected_mask))
        np.testing.assert_array_equal(
            result.phase_codes,
            np.full(
                result.grid.shape,
                self.code(PhaseLabel.NOT_APPLICABLE),
                dtype=np.uint8,
            ),
        )
        np.testing.assert_array_equal(
            result.phase_route_codes,
            np.full(
                result.grid.shape,
                self.code(PhaseRoute.NO_TRANSITION),
                dtype=np.uint8,
            ),
        )
        np.testing.assert_array_equal(
            result.route_codes,
            np.full(result.grid.shape, self.code(PointRoute.FAST), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            result.solve_status_codes,
            np.full(
                result.grid.shape,
                self.code(SolveStatus.CONVERGED_FAST),
                dtype=np.uint8,
            ),
        )
        np.testing.assert_array_equal(
            result.initial_guess_source_codes,
            np.asarray(
                [
                    self.code(InitialGuessSource.CONTAINING_CELL),
                    self.code(InitialGuessSource.CONTAINING_CELL),
                    self.code(InitialGuessSource.DIRECT_NEIGHBOR),
                    self.code(InitialGuessSource.DIRECT_NEIGHBOR),
                    self.code(InitialGuessSource.DIRECT_NEIGHBOR),
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_allclose(
            result.source_coordinates[:, 0], expected_source, atol=2.0e-12
        )
        np.testing.assert_allclose(result.achieved_targets, requested, atol=2.0e-12)
        np.testing.assert_allclose(result.residual, 0.0, atol=2.0e-12)
        np.testing.assert_array_equal(result.auxiliary_defined[:, 0], True)
        np.testing.assert_allclose(
            result.auxiliary_values[:, 0], 3.0 * requested[:, 0], atol=2.0e-12
        )

        self.assertEqual(
            result.manifest["pipeline_result_schema_version"],
            PIPELINE_RESULT_SCHEMA_VERSION,
        )
        self.assertEqual(result.manifest["grid"]["ordering"], "C_last_axis_fastest")
        self.assertEqual(result.manifest["run"]["status_counts"]["solved"], 5)
        self.assertEqual(sum(result.status_counts.values()), result.grid.size)

        first, second = results
        numeric_names = (
            "source_coordinates",
            "achieved_targets",
            "residual",
            "scaled_residual",
            "scaled_residual_norm_inf",
            "scaled_distance_l2",
            "auxiliary_values",
        )
        for name in numeric_names:
            np.testing.assert_allclose(
                getattr(first, name), getattr(second, name), equal_nan=True
            )
        exact_names = (
            "auxiliary_defined",
            "disposition_codes",
            "route_codes",
            "prescreen_codes",
            "solve_status_codes",
            "fast_status_codes",
            "proof_status_codes",
            "projection_quality_codes",
            "initial_guess_source_codes",
        )
        for name in exact_names:
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


class ProjectionRouteTests(SerialPipelineTestCase):
    def make_correlated_table(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "P": field_spec("auxiliary", 4),
        }
        rows = []
        for temperature, chemical_potential in itertools.product(
            (1.0, 2.0, 3.0), (0.0, 1.0, 2.0)
        ):
            energy = temperature
            density = temperature + chemical_potential
            rows.append(
                (
                    temperature,
                    chemical_potential,
                    energy,
                    density,
                    energy + density,
                )
            )
        return self.make_table(fields, rows)

    def test_prescreen_exclusions_project_or_remain_certified_unreachable(self):
        table = self.make_correlated_table()
        enabled = EoSInversionPipeline(
            table,
            self.make_config(
                table,
                projection_enabled=True,
                auxiliary_fields=("P",),
            ),
        ).run()

        expected_excluded = np.asarray([[False, True], [True, False]])
        np.testing.assert_array_equal(enabled.prescreen_excluded_mask, expected_excluded)
        np.testing.assert_array_equal(enabled.projected_mask, expected_excluded)
        np.testing.assert_array_equal(enabled.solution_mask, ~expected_excluded)
        expected_routes = np.asarray(
            [
                [self.code(PointRoute.FAST), self.code(PointRoute.PROJECTED_PRESCREEN)],
                [self.code(PointRoute.PROJECTED_PRESCREEN), self.code(PointRoute.FAST)],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(enabled.route_codes, expected_routes)
        self.assertTrue(np.all(enabled.projection_quality_codes[expected_excluded] != 0))
        self.assertTrue(np.all(enabled.auxiliary_defined))
        np.testing.assert_allclose(
            enabled.auxiliary_values[..., 0],
            enabled.achieved_targets[..., 0] + enabled.achieved_targets[..., 1],
        )
        np.testing.assert_allclose(
            enabled.source_coordinates[0, 1], [1.4, 2.0], atol=2.0e-7
        )
        np.testing.assert_allclose(
            enabled.source_coordinates[1, 0], [2.6, 0.0], atol=2.0e-7
        )

        disabled = EoSInversionPipeline(
            table,
            self.make_config(
                table,
                projection_enabled=False,
                auxiliary_fields=("P",),
            ),
        ).run()
        np.testing.assert_array_equal(disabled.prescreen_excluded_mask, expected_excluded)
        np.testing.assert_array_equal(disabled.solution_mask, ~expected_excluded)
        self.assertFalse(np.any(disabled.projected_mask))
        np.testing.assert_array_equal(
            disabled.disposition_codes[expected_excluded],
            self.code(PointDisposition.CERTIFIED_UNREACHABLE),
        )
        np.testing.assert_array_equal(
            disabled.route_codes[expected_excluded],
            self.code(PointRoute.CERTIFIED_UNREACHABLE_PRESCREEN),
        )
        self.assertTrue(np.all(np.isnan(disabled.source_coordinates[expected_excluded])))
        self.assertTrue(np.all(np.isnan(disabled.auxiliary_values[expected_excluded])))
        self.assertFalse(np.any(disabled.auxiliary_defined[expected_excluded]))
        self.assertFalse(np.any(disabled.projection_quality_codes))


class SubcellFallbackPipelineTests(SerialPipelineTestCase):
    def make_diamond_table(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }
        rows = []
        for temperature, chemical_potential in itertools.product(
            (0.5, 1.5), (0.0, 1.0)
        ):
            shifted = temperature - 0.5
            rows.append(
                (
                    temperature,
                    chemical_potential,
                    shifted + chemical_potential,
                    shifted - chemical_potential,
                )
            )
        return self.make_table(fields, rows)

    def configured(self, table, *, projection_enabled):
        base = self.make_config(
            table,
            count=3,
            chunk_size=9,
            projection_enabled=projection_enabled,
            residual_scales={"e": 1.0, "nB": 1.0},
        )
        return replace(
            base,
            solver=SolverConfig(
                enable_subcell_fallback=True,
                max_subcell_depth=3,
                max_subcell_active_boxes=512,
            ),
        )

    def test_complete_localization_enters_subcells_and_aggregates_counters(self):
        table = self.make_diamond_table()
        result = EoSInversionPipeline(
            table, self.configured(table, projection_enabled=False)
        ).run()
        counters = result.manifest["run"]["stage_counters"]

        self.assertGreater(counters["subcell_fallback_attempted"], 0)
        self.assertGreater(counters["subcell_certified_no_root"], 0)
        self.assertGreater(counters["subcell_boxes_evaluated"], 0)
        self.assertEqual(counters["subcell_guarded_incomplete"], 0)
        self.assertGreater(
            counters["subcell_certificate_depth_1"]
            + counters["subcell_certificate_depth_2"]
            + counters["subcell_certificate_depth_3"],
            0,
        )
        self.assertTrue(np.any(result.certified_unreachable_mask))
        self.assertEqual(result.manifest["run"]["errors"]["count"], 0)
        contract = result.manifest["numerical_routing_contract"]
        self.assertEqual(
            contract["fallback_cell_resolution"],
            "dyadic_subcells_then_shared_krawczyk",
        )
        self.assertTrue(contract["subcell_fallback"]["enabled"])

    def test_projection_occurs_after_subcell_no_root_certificate(self):
        table = self.make_diamond_table()
        result = EoSInversionPipeline(
            table, self.configured(table, projection_enabled=True)
        ).run()
        projected_after_proof = result.route_codes == self.code(
            PointRoute.PROJECTED_CERTIFIED_NO_ROOT
        )

        self.assertTrue(np.any(projected_after_proof))
        np.testing.assert_array_equal(
            result.solve_status_codes[projected_after_proof],
            self.code(SolveStatus.CERTIFIED_NO_ROOT),
        )
        self.assertTrue(np.all(result.projected_mask[projected_after_proof]))
        self.assertGreater(
            result.manifest["run"]["stage_counters"][
                "subcell_certified_no_root"
            ],
            0,
        )


class TildeInvalidTargetTests(SerialPipelineTestCase):
    def make_overflow_cross_table(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }
        rows = []
        for temperature, chemical_potential in itertools.product(
            (0.2, 0.5), (0.0, 1.0)
        ):
            energy = 1.0e-16 if temperature == 0.2 else 1.0e300
            density_scale = 5.0e298 if temperature == 0.2 else 1.0
            rows.append(
                (
                    temperature,
                    chemical_potential,
                    energy,
                    density_scale * chemical_potential,
                )
            )
        return self.make_table(fields, rows)

    def test_one_invalid_tilde_cross_point_does_not_poison_valid_slots(self):
        table = self.make_overflow_cross_table()
        config = self.make_config(
            table,
            space="tilde",
            projection_enabled=False,
            residual_scales={"e": 1.0e300, "nB": 5.0e298},
            error_policy=ErrorPolicy.ACCUMULATE,
        )
        result = EoSInversionPipeline(table, config).run()

        expected_invalid = np.asarray([[False, False], [False, True]])
        np.testing.assert_array_equal(result.invalid_target_mask, expected_invalid)
        self.assertTrue(np.all(result.solution_mask[~expected_invalid]))
        self.assertEqual(
            int(result.disposition_codes[1, 1]),
            self.code(PointDisposition.INVALID_TARGET),
        )
        self.assertEqual(
            int(result.route_codes[1, 1]), self.code(PointRoute.INVALID_TRANSFORM)
        )
        self.assertEqual(
            int(result.prescreen_codes[1, 1]), self.code(PrescreenStatus.NOT_RUN)
        )
        self.assertTrue(np.all(np.isnan(result.source_coordinates[1, 1])))
        self.assertTrue(np.all(np.isnan(result.achieved_targets[1, 1])))
        self.assertFalse(bool(result.achieved_tilde_defined[1, 1]))
        self.assertTrue(np.all(result.achieved_tilde_defined[~expected_invalid]))


class ConfigurationAndErrorPolicyTests(SerialPipelineTestCase):
    def make_affine_table(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        rows = [(temperature, 1.0 + temperature) for temperature in (0.2, 0.5, 0.8)]
        return self.make_table(fields, rows)

    def test_restricted_active_domain_guards(self):
        table = self.make_affine_table()
        serial = self.make_config(table)

        with self.assertRaises(PipelineConfigurationError):
            EoSInversionPipeline(
                table, serial, active_cell_mask=np.asarray([True, False])
            )

        # An explicit all-active mask is semantically identical to the default.
        accepted = EoSInversionPipeline(
            table, serial, active_cell_mask=np.asarray([True, True])
        )
        self.assertEqual(accepted.plan().grid.size, serial.grid.axes["e"].count)

    def test_unexpected_point_error_obeys_execution_policy(self):
        table = self.make_affine_table()
        accumulate = self.make_config(
            table, count=3, error_policy=ErrorPolicy.ACCUMULATE
        )
        with patch(
            "eos_trensmuther.pipeline.EoSRootSolver.solve",
            side_effect=RuntimeError("synthetic point failure"),
        ):
            result = EoSInversionPipeline(table, accumulate).run()

        self.assertTrue(np.all(result.error_mask))
        self.assertFalse(np.any(result.usable_state_mask))
        np.testing.assert_array_equal(
            result.route_codes,
            np.full(result.grid.shape, self.code(PointRoute.ERROR), dtype=np.uint8),
        )
        self.assertEqual(result.manifest["run"]["errors"]["count"], 3)
        self.assertEqual(len(result.manifest["run"]["errors"]["records"]), 3)

        fail_fast = self.make_config(
            table, count=3, error_policy=ErrorPolicy.FAIL_FAST
        )
        with patch(
            "eos_trensmuther.pipeline.EoSRootSolver.solve",
            side_effect=RuntimeError("synthetic point failure"),
        ):
            with self.assertRaises(PipelineStateError):
                EoSInversionPipeline(table, fail_fast).run()


if __name__ == "__main__":
    unittest.main()
