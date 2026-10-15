"""Dense, persistent, and process-parallel first-order pipeline contracts."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from eos_trensmuther import (
    CheckpointConfig,
    ConstrainedTildeTransform,
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    GridSpace,
    InversionGridSpec,
    LineAuxiliaryPolicy,
    MultilinearInterpolator,
    PhaseLabel,
    PhaseRoute,
    PipelineConfig,
    PipelineConfigurationError,
    ProjectionConfig,
    SolverConfig,
    build_region_s,
    load_pipeline_result,
    pipeline_status_code,
    save_pipeline_result,
    transition_model_from_table,
)
from tests.test_region_s import SyntheticRegionSTests
from tests.test_tables import TemporaryEoS, field_spec, metadata


_FIRST_ORDER_SOLVER_CONFIG = SolverConfig(enable_subcell_fallback=False)


_PHASE_ARRAYS = (
    "phase_codes",
    "phase_route_codes",
    "phase_selection_reason_codes",
    "mixed_candidate",
    "exact_mixed_candidate",
    "mixed_candidate_entry_count",
    "mixed_verified_root_count",
    "phase_boundary",
    "phase_multivalued",
    "line_mixture_fraction",
    "line_mixture_fraction_defined",
    "alternate_single_phase_source",
    "alternate_single_phase_source_defined",
)

_DENSE_RESULT_ARRAYS = (
    "source_coordinates",
    "achieved_targets",
    "residual",
    "scaled_residual",
    "scaled_residual_norm_inf",
    "scaled_distance_l2",
    "auxiliary_values",
    "auxiliary_defined",
    "achieved_tilde_values",
    "achieved_tilde_defined",
    "disposition_codes",
    "route_codes",
    "prescreen_codes",
    "solve_status_codes",
    "fast_status_codes",
    "proof_status_codes",
    "projection_quality_codes",
    "initial_guess_source_codes",
    "phase_codes",
    "phase_route_codes",
    "phase_selection_reason_codes",
    "mixed_candidate",
    "exact_mixed_candidate",
    "mixed_candidate_entry_count",
    "mixed_verified_root_count",
    "phase_boundary",
    "phase_multivalued",
    "line_mixture_fraction",
    "line_mixture_fraction_defined",
    "alternate_single_phase_source",
    "alternate_single_phase_source_defined",
)


class PhasePipelineTests(unittest.TestCase):
    def assert_dense_results_byte_equal(self, left, right):
        self.assertEqual(left.nbytes, right.nbytes)
        for name in _DENSE_RESULT_ARRAYS:
            with self.subTest(array=name):
                left_array = getattr(left, name)
                right_array = getattr(right, name)
                self.assertEqual(left_array.dtype, right_array.dtype)
                self.assertEqual(left_array.shape, right_array.shape)
                self.assertEqual(
                    left_array.tobytes(order="C"),
                    right_array.tobytes(order="C"),
                )

    def run_without_ordinary_localization(self, table, config):
        pipeline = EoSInversionPipeline(table, config)
        pipeline._prepare()
        self.assertFalse(pipeline._uses_complete_ordinary_localization())
        self.assertEqual(
            pipeline._coarse_seed_report["disabled_reason"],
            "first_order_branch_aware_path",
        )
        failure = AssertionError(
            "first-order flow entered ordinary localization/seeding"
        )
        with mock.patch.object(
            pipeline._reachability,
            "localize_chunk",
            side_effect=failure,
        ), mock.patch.object(
            pipeline,
            "_candidate_center_seed",
            side_effect=failure,
        ), mock.patch.object(
            pipeline,
            "_select_localized_seed",
            side_effect=failure,
        ):
            return pipeline.run()

    def make_problem(
        self,
        *,
        workers=1,
        chunk_size=2,
        include_auxiliary=False,
        line_auxiliary_policies=None,
    ):
        fixture = SyntheticRegionSTests()
        self.addCleanup(fixture.doCleanups)
        table, _ = fixture.make_line_table(
            include_auxiliary=include_auxiliary
        )
        strip = build_region_s(table)
        hadronic = strip.evaluate(0.35, 0.0)
        qgp = strip.evaluate(0.35, 1.0)
        lower = np.minimum(hadronic, qgp)
        upper = np.maximum(hadronic, qgp)
        config = PipelineConfig(
            grid=InversionGridSpec(
                GridSpace.TARGET,
                {
                    "e": GridAxisSpec(2, float(lower[0]), float(upper[0])),
                    "nB": GridAxisSpec(2, float(lower[1]), float(upper[1])),
                },
            ),
            residual_scales={"e": 1.0e-3, "nB": 1.0e-3},
            solver=_FIRST_ORDER_SOLVER_CONFIG,
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(
                workers=workers,
                chunk_size=chunk_size,
                max_pending_chunks=2,
            ),
            auxiliary_fields=("P", "marker") if include_auxiliary else (),
            line_auxiliary_policies=(
                {} if line_auxiliary_policies is None
                else line_auxiliary_policies
            ),
        )
        return table, config

    def make_binodal_problem(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "phase": field_spec("auxiliary", 4),
        }
        document = metadata(fields, 2, expected_shape=[3, 3])
        document["scientific"]["phase_structure"] = {
            "status": "known",
            "critical_points": [],
            "first_order_transition": {
                "present": True,
                "representation": "binodals",
                "kind": "synthetic binodal band",
                "phase_fraction_field": "phase",
                "mixed_phase_condition": "0 < phase < 1",
                "coexistence_data": {
                    "file": "transition.dat",
                    "delimiter": "whitespace",
                    "comment_prefix": "#",
                    "skip_rows": 0,
                    "encoding": "utf-8",
                    "expected_columns": 3,
                    "fields": ["muB", "T_dilute", "T_dense"],
                    "columns": {
                        "muB": {"index": 0, "unit": "MeV"},
                        "T_dilute": {"index": 1, "unit": "MeV"},
                        "T_dense": {"index": 2, "unit": "MeV"},
                    },
                },
            },
        }
        rows = [
            [temperature, muB, temperature, muB, (temperature - 0.1) / 0.1]
            for temperature in (0.1, 0.15, 0.2)
            for muB in (0.0, 0.25, 0.5)
        ]
        fixture = TemporaryEoS(document, rows)
        self.addCleanup(fixture.close)
        fixture.write_sidecar(
            "transition.dat", "0 120 180\n500 120 180\n"
        )
        table = ForwardTable.load_validated(fixture.metadata_path)
        config = PipelineConfig(
            grid=InversionGridSpec(
                GridSpace.TARGET,
                {
                    "e": GridAxisSpec(2, 0.13, 0.17),
                    "nB": GridAxisSpec(2, 0.1, 0.4),
                },
            ),
            residual_scales={"e": 1.0e-3, "nB": 1.0e-3},
            solver=_FIRST_ORDER_SOLVER_CONFIG,
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(workers=1, chunk_size=2),
            auxiliary_fields=("phase",),
        )
        return table, config

    def test_line_phase_columns_and_manifest_are_complete(self):
        table, config = self.make_problem()
        result = EoSInversionPipeline(table, config).run()

        self.assertEqual(np.count_nonzero(result.mixed_phase_mask), 2)
        self.assertEqual(np.count_nonzero(result.single_phase_mask), 0)
        self.assertEqual(
            np.count_nonzero(result.phase_route_codes == pipeline_status_code(
                PhaseRoute.LINE_REGION_S
            )),
            2,
        )
        fractions = np.sort(
            result.line_mixture_fraction[result.line_mixture_fraction_defined]
        )
        np.testing.assert_allclose(fractions, [0.0, 1.0], atol=1.0e-9)
        self.assertTrue(np.all(result.mixed_verified_root_count[
            result.mixed_phase_mask
        ] == 1))
        self.assertTrue(np.all(result.phase_boundary[result.mixed_phase_mask]))
        first_order = result.manifest["run"]["first_order"]
        self.assertTrue(first_order["configured"])
        self.assertEqual(first_order["representation"], "line")
        self.assertEqual(
            result.manifest["run"]["status_summary"]["phase"][
                PhaseLabel.MIXED_PHASE.value
            ],
            2,
        )

    def test_first_order_tables_do_not_enter_ordinary_complete_localization(self):
        table, config = self.make_problem()
        pipeline = EoSInversionPipeline(table, config)
        pipeline._prepare()
        self.assertFalse(pipeline._uses_complete_ordinary_localization())

        with mock.patch.object(
            pipeline._reachability,
            "localize_chunk",
            side_effect=AssertionError(
                "first-order flow entered ordinary complete localization"
            ),
        ):
            result = pipeline.run()

        self.assertEqual(result.manifest["run"]["errors"]["count"], 0)
        self.assertFalse(
            result.manifest["run"]["reachability"][
                "complete_leaf_localization"
            ]["enabled"]
        )

    def test_subcell_feature_does_not_enter_first_order_path(self):
        table, base = self.make_problem()
        config = replace(
            base,
            solver=replace(
                base.solver,
                enable_subcell_fallback=True,
                max_subcell_depth=3,
            ),
        )
        pipeline = EoSInversionPipeline(table, config)
        pipeline._prepare()
        self.assertFalse(pipeline._uses_complete_ordinary_localization())

        with mock.patch.object(
            pipeline._reachability,
            "localize_chunk",
            side_effect=AssertionError(
                "first-order flow entered subcell-capable localization"
            ),
        ):
            result = pipeline.run()

        counters = result.manifest["run"]["stage_counters"]
        self.assertEqual(counters["subcell_fallback_attempted"], 0)
        self.assertEqual(counters["subcell_boxes_evaluated"], 0)
        self.assertTrue(config.solver.enable_subcell_fallback)
        self.assertEqual(result.manifest["run"]["errors"]["count"], 0)

    def test_reachability_seed_toggle_is_byte_inert_for_line_and_binodals(self):
        builders = (self.make_problem, self.make_binodal_problem)
        for builder in builders:
            with self.subTest(representation=builder.__name__):
                table, base = builder()
                enabled = self.run_without_ordinary_localization(
                    table,
                    replace(base, use_reachability_seed=True),
                )
                disabled = self.run_without_ordinary_localization(
                    table,
                    replace(base, use_reachability_seed=False),
                )

                self.assert_dense_results_byte_equal(enabled, disabled)
                enabled_run = enabled.manifest["run"]
                disabled_run = disabled.manifest["run"]
                for name in (
                    "stage_counters",
                    "status_counts",
                    "status_summary",
                    "errors",
                    "projection_failures",
                    "first_order",
                    "auxiliary_evaluation",
                    "initial_guess",
                    "reachability",
                ):
                    with self.subTest(manifest=name):
                        self.assertEqual(enabled_run[name], disabled_run[name])
                self.assertFalse(
                    enabled_run["reachability"][
                        "complete_leaf_localization"
                    ]["enabled"]
                )
                self.assertFalse(
                    disabled_run["reachability"][
                        "complete_leaf_localization"
                    ]["enabled"]
                )

                # The user configuration and run fingerprint intentionally
                # differ because the requested option is persisted, even
                # though first-order numerical/scientific output is inert.
                self.assertNotEqual(
                    enabled.plan.config.to_dict(),
                    disabled.plan.config.to_dict(),
                )

    def test_phase_columns_round_trip_and_checkpoint_resume(self):
        table, config = self.make_problem()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = CheckpointConfig(root / "checkpoint")
            result = EoSInversionPipeline(table, config).run(
                checkpoint=checkpoint
            )
            resumed = EoSInversionPipeline(table, config).run(
                checkpoint=checkpoint
            )
            artifact = root / "phase.eosinv"
            save_pipeline_result(result, artifact)
            loaded = load_pipeline_result(artifact)

        for name in _PHASE_ARRAYS:
            with self.subTest(name=name):
                np.testing.assert_array_equal(
                    getattr(result, name), getattr(resumed, name)
                )
                np.testing.assert_array_equal(
                    getattr(result, name), getattr(loaded, name)
                )
        self.assertEqual(
            result.manifest["run"]["first_order"],
            loaded.manifest["run"]["first_order"],
        )

    def test_line_phase_results_match_in_serial_and_process_modes(self):
        table, serial_config = self.make_problem(
            chunk_size=1,
            include_auxiliary=True,
            line_auxiliary_policies={"marker": LineAuxiliaryPolicy.EMPTY},
        )
        parallel_config = replace(
            serial_config,
            execution=replace(serial_config.execution, workers=2),
        )
        serial = EoSInversionPipeline(table, serial_config).run()
        parallel = EoSInversionPipeline(table, parallel_config).run()

        for name in _PHASE_ARRAYS + ("auxiliary_values", "auxiliary_defined"):
            with self.subTest(name=name):
                np.testing.assert_array_equal(
                    getattr(serial, name), getattr(parallel, name)
                )
        for name in (
            "stage_counters",
            "status_counts",
            "status_summary",
            "errors",
            "projection_failures",
            "first_order",
            "auxiliary_evaluation",
        ):
            self.assertEqual(
                serial.manifest["run"][name],
                parallel.manifest["run"][name],
            )

    def test_line_auxiliary_policies_are_field_explicit(self):
        table, config = self.make_problem(
            include_auxiliary=True,
            line_auxiliary_policies={
                "marker": LineAuxiliaryPolicy.EMPTY,
            },
        )
        result = EoSInversionPipeline(table, config).run()
        mixed = result.mixed_phase_mask
        self.assertEqual(int(np.count_nonzero(mixed)), 2)

        strip = build_region_s(table, auxiliary_fields=("P",))
        expected = np.asarray(
            [
                strip.evaluate_auxiliaries(source[1], fraction)[0]
                for source, fraction in zip(
                    result.source_coordinates[mixed],
                    result.line_mixture_fraction[mixed],
                )
            ]
        )
        p_index = result.plan.auxiliary_names.index("P")
        marker_index = result.plan.auxiliary_names.index("marker")
        np.testing.assert_allclose(
            result.auxiliary_values[..., p_index][mixed], expected
        )
        self.assertTrue(np.all(result.auxiliary_defined[..., p_index][mixed]))
        self.assertTrue(
            np.all(np.isnan(result.auxiliary_values[..., marker_index][mixed]))
        )
        self.assertFalse(
            np.any(result.auxiliary_defined[..., marker_index][mixed])
        )

        policy = result.manifest["run"]["auxiliary_evaluation"]["line"]
        self.assertEqual(
            policy["fields"],
            {
                "P": "linear",
                "marker": "empty",
            },
        )
        self.assertEqual(policy["counts"]["P"]["defined"], 2)
        self.assertEqual(policy["counts"]["marker"]["undefined"], 2)
        counters = result.manifest["run"]["stage_counters"]
        self.assertEqual(counters["line_auxiliary_values_defined"], 2)
        self.assertEqual(counters["line_auxiliary_values_undefined"], 2)
        restored = PipelineConfig.from_dict(config.to_dict())
        self.assertEqual(
            restored.line_auxiliary_policies["marker"],
            LineAuxiliaryPolicy.EMPTY,
        )

        invalid = replace(
            config,
            line_auxiliary_policies={"not_retained": "empty"},
        )
        with self.assertRaises(PipelineConfigurationError):
            EoSInversionPipeline(table, invalid)

        with self.assertRaises(PipelineConfigurationError):
            replace(config, line_auxiliary_default_policy="undefined")

    def test_binodal_auxiliary_uses_ordinary_interpolation(self):
        table, config = self.make_binodal_problem()
        result = EoSInversionPipeline(table, config).run()
        mixed = result.mixed_phase_mask
        self.assertEqual(int(np.count_nonzero(mixed)), result.grid.size)

        ordinary = MultilinearInterpolator(table, ("phase",))
        expected = ordinary.evaluate(result.source_coordinates[mixed])[:, 0]
        np.testing.assert_allclose(
            result.auxiliary_values[..., 0][mixed], expected
        )
        self.assertTrue(np.all(result.auxiliary_defined[..., 0][mixed]))
        policy = result.manifest["run"]["auxiliary_evaluation"]["binodals"]
        self.assertEqual(
            policy["policy"],
            "ordinary_table_interpolation_at_solved_source",
        )
        self.assertEqual(policy["counts"]["phase"]["defined"], 4)

    def test_line_policy_does_not_change_single_phase_auxiliaries(self):
        fixture = SyntheticRegionSTests()
        self.addCleanup(fixture.doCleanups)
        table, _ = fixture.make_line_table(include_auxiliary=True)
        source = np.asarray([0.06, 0.3])
        target_map = MultilinearInterpolator(table, ("e", "nB"))
        center = np.asarray(target_map.evaluate(source), dtype=float)
        width = np.maximum(np.abs(center) * 1.0e-7, 1.0e-14)
        config = PipelineConfig(
            grid=InversionGridSpec(
                GridSpace.TARGET,
                {
                    "e": GridAxisSpec(
                        2, float(center[0] - width[0]), float(center[0] + width[0])
                    ),
                    "nB": GridAxisSpec(
                        2, float(center[1] - width[1]), float(center[1] + width[1])
                    ),
                },
            ),
            residual_scales={"e": 1.0e-3, "nB": 1.0e-3},
            solver=_FIRST_ORDER_SOLVER_CONFIG,
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(workers=1, chunk_size=2),
            auxiliary_fields=("P", "marker"),
            line_auxiliary_policies={"marker": LineAuxiliaryPolicy.EMPTY},
        )
        result = EoSInversionPipeline(table, config).run()

        self.assertTrue(np.all(result.solution_mask))
        self.assertFalse(np.any(result.mixed_phase_mask))
        self.assertTrue(np.all(result.auxiliary_defined))
        ordinary = MultilinearInterpolator(table, ("P", "marker"))
        expected = ordinary.evaluate(result.source_coordinates)
        np.testing.assert_allclose(result.auxiliary_values, expected)

    def test_same_mixed_target_is_phase_equivalent_on_target_and_tilde_grids(self):
        fixture = SyntheticRegionSTests()
        self.addCleanup(fixture.doCleanups)
        table, _ = fixture.make_line_table()
        strip = build_region_s(table)
        target = np.asarray(strip.evaluate(0.35, 0.5), dtype=float)
        transform = ConstrainedTildeTransform(table.charge_constraints)
        tilde = transform.targets_to_coordinates(
            target[np.newaxis, :]
        ).independent_values[0]

        target_envelope = table.find_boundaries(space="target")
        tilde_envelope = table.find_boundaries(space="tilde")

        def axis_pair(value, maximum):
            step = max((float(maximum) - float(value)) * 1.0e-4, 1.0e-12)
            return GridAxisSpec(2, float(value), float(value) + step)

        common = {
            "residual_scales": {"e": 1.0e-3, "nB": 1.0e-3},
            "solver": _FIRST_ORDER_SOLVER_CONFIG,
            "projection": ProjectionConfig(enabled=False),
            "execution": ExecutionConfig(workers=1, chunk_size=2),
            "auxiliary_fields": (),
        }
        target_config = PipelineConfig(
            grid=InversionGridSpec(
                GridSpace.TARGET,
                {
                    name: axis_pair(value, maximum)
                    for name, value, maximum in zip(
                        target_envelope.names,
                        target,
                        target_envelope.upper,
                    )
                },
            ),
            **common,
        )
        tilde_config = PipelineConfig(
            grid=InversionGridSpec(
                GridSpace.TILDE,
                {
                    name: axis_pair(value, maximum)
                    for name, value, maximum in zip(
                        tilde_envelope.names,
                        tilde,
                        tilde_envelope.upper,
                    )
                },
            ),
            **common,
        )

        target_result = EoSInversionPipeline(table, target_config).run()
        tilde_result = EoSInversionPipeline(table, tilde_config).run()
        target_requested = target_result.grid.chunk(0, 1).canonical_targets[0]
        tilde_requested = tilde_result.grid.chunk(0, 1).canonical_targets[0]
        np.testing.assert_allclose(tilde_requested, target_requested, rtol=1e-13)
        np.testing.assert_allclose(target_requested, target, rtol=1e-13)

        self.assertTrue(target_result.mixed_phase_mask.reshape(-1)[0])
        self.assertTrue(tilde_result.mixed_phase_mask.reshape(-1)[0])
        self.assertEqual(
            target_result.phase_codes.reshape(-1)[0],
            tilde_result.phase_codes.reshape(-1)[0],
        )
        self.assertEqual(
            target_result.phase_route_codes.reshape(-1)[0],
            tilde_result.phase_route_codes.reshape(-1)[0],
        )
        np.testing.assert_allclose(
            target_result.source_coordinates.reshape(-1, 2)[0],
            tilde_result.source_coordinates.reshape(-1, 2)[0],
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            target_result.achieved_targets.reshape(-1, 2)[0],
            tilde_result.achieved_targets.reshape(-1, 2)[0],
            rtol=1e-10,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            float(target_result.line_mixture_fraction.reshape(-1)[0]),
            float(tilde_result.line_mixture_fraction.reshape(-1)[0]),
            places=10,
        )


class RealPhasePipelineValidationTests(unittest.TestCase):
    REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
    CEC_METADATA = REPOSITORY_ROOT / "EoSTables" / "CEC" / "metadata.json"
    CEC_TABLE = (
        REPOSITORY_ROOT
        / "EoSTables"
        / "CEC"
        / "hic_table_T80-300_muB0-800_d1.dat"
    )
    ISING_METADATA = (
        REPOSITORY_ROOT / "EoSTables" / "2DIsing" / "metadata.json"
    )
    ISING_TABLE = (
        REPOSITORY_ROOT / "EoSTables" / "2DIsing" / "EoSmub602.dat"
    )

    @unittest.skipUnless(CEC_TABLE.is_file(), "local CEC table is unavailable")
    def test_cec_binodal_geometry_fdense_and_small_pipeline_agree(self):
        table = ForwardTable.load_validated(self.CEC_METADATA)
        band = transition_model_from_table(table)
        temperature = table.field("T", shaped=False)
        chemical_potential = table.field("muB", shaped=False)
        fdense = table.field("fdense", shaped=False)
        geometric = band.contains_points(
            np.column_stack((temperature, chemical_potential))
        )
        fraction_mixed = (fdense > 0.0) & (fdense < 1.0)
        np.testing.assert_array_equal(geometric, fraction_mixed)
        self.assertEqual(int(np.count_nonzero(geometric)), 246)

        mixed_indices = np.flatnonzero(fraction_mixed)
        center_index = mixed_indices[
            np.argmin(np.abs(fdense[mixed_indices] - 0.5))
        ]
        center_e = float(table.field("e", shaped=False)[center_index])
        center_nB = float(table.field("nB", shaped=False)[center_index])
        config = PipelineConfig(
            grid=InversionGridSpec(
                GridSpace.TARGET,
                {
                    "e": GridAxisSpec(
                        2, center_e - 1.0e-6, center_e + 1.0e-6
                    ),
                    "nB": GridAxisSpec(
                        2, center_nB - 1.0e-6, center_nB + 1.0e-6
                    ),
                },
            ),
            residual_scales={"e": 0.1, "nB": 0.02},
            solver=_FIRST_ORDER_SOLVER_CONFIG,
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(workers=1, chunk_size=2),
            auxiliary_fields=("fdense",),
        )
        result = EoSInversionPipeline(table, config).run()

        self.assertTrue(np.all(result.solution_mask))
        self.assertTrue(np.all(result.mixed_phase_mask))
        solved_fdense = result.auxiliary_values[..., 0]
        self.assertTrue(np.all(result.auxiliary_defined[..., 0]))
        self.assertTrue(np.all((solved_fdense > 0.0) & (solved_fdense < 1.0)))
        run = result.manifest["run"]
        self.assertEqual(run["errors"]["count"], 0)
        self.assertEqual(run["first_order"]["representation"], "binodals")
        self.assertEqual(
            run["status_summary"]["phase"][PhaseLabel.MIXED_PHASE.value], 4
        )
        self.assertEqual(
            run["auxiliary_evaluation"]["binodals"]["counts"]["fdense"][
                "defined"
            ],
            4,
        )

    @unittest.skipUnless(
        ISING_TABLE.is_file(), "local Ising table is unavailable"
    )
    def test_ising_small_pipeline_solves_physical_region_s(self):
        table = ForwardTable.load_validated(self.ISING_METADATA)
        strip = build_region_s(table)
        center = np.asarray(strip.evaluate(0.65, 0.5), dtype=float)
        half_width = np.maximum(np.abs(center) * 1.0e-5, 1.0e-12)
        config = PipelineConfig(
            grid=InversionGridSpec(
                GridSpace.TARGET,
                {
                    "e": GridAxisSpec(
                        2,
                        float(center[0] - half_width[0]),
                        float(center[0] + half_width[0]),
                    ),
                    "nB": GridAxisSpec(
                        2,
                        float(center[1] - half_width[1]),
                        float(center[1] + half_width[1]),
                    ),
                },
            ),
            residual_scales={"e": 1.0e-3, "nB": 1.0e-3},
            solver=_FIRST_ORDER_SOLVER_CONFIG,
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(workers=1, chunk_size=2),
            auxiliary_fields=(),
        )
        result = EoSInversionPipeline(table, config).run()

        self.assertTrue(np.all(result.solution_mask))
        self.assertTrue(np.all(result.mixed_phase_mask))
        self.assertTrue(np.all(result.line_mixture_fraction_defined))
        fractions = result.line_mixture_fraction.reshape(-1)
        self.assertTrue(np.all((fractions > 0.0) & (fractions < 1.0)))
        sources = result.source_coordinates.reshape(-1, 2)
        achieved = result.achieved_targets.reshape(-1, 2)
        for source, fraction, target in zip(sources, fractions, achieved):
            np.testing.assert_allclose(
                strip.evaluate(float(source[1]), float(fraction)),
                target,
                rtol=1.0e-11,
                atol=1.0e-14,
            )
        run = result.manifest["run"]
        self.assertEqual(run["errors"]["count"], 0)
        self.assertEqual(run["first_order"]["representation"], "line")
        self.assertEqual(
            run["status_summary"]["phase"][PhaseLabel.MIXED_PHASE.value], 4
        )


if __name__ == "__main__":
    unittest.main()
