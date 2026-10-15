"""Phase-1 tests for full-table pipeline contracts and storage."""

import itertools
import unittest

import numpy as np

from eos_trensmuther import (
    EoSInversionPipeline,
    ErrorPolicy,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    GridSpace,
    InitialGuessSource,
    InversionGrid,
    InversionGridSpec,
    PipelineConfig,
    PipelineConfigurationError,
    PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
    PipelinePlan,
    PipelineStateError,
    PointDisposition,
    PointRoute,
    PrescreenStatus,
    ProcessStartMethod,
    ProjectionQuality,
    ProofStatus,
    SolveStatus,
    pipeline_status_code,
)
from eos_trensmuther.transforms import ConstrainedTildeTransform

from tests.test_tables import TemporaryEoS, field_spec, metadata


class PipelineTestCase(unittest.TestCase):
    def make_table(self, dimension, *, auxiliary_fields=()):
        coordinate_names = ("T", "muB", "muQ", "muS")[:dimension]
        target_names = ("e", "nB", "nQ", "nS")[:dimension]
        fields = {
            name: field_spec("coordinate", index)
            for index, name in enumerate(coordinate_names)
        }
        for offset, name in enumerate(target_names, start=dimension):
            fields[name] = field_spec("target", offset)
        for offset, (name, output) in enumerate(
            auxiliary_fields, start=2 * dimension
        ):
            fields[name] = field_spec("auxiliary", offset)
            fields[name]["output"] = output

        coordinate_axes = tuple(
            np.asarray([0.2 + 0.1 * index, 0.5 + 0.1 * index])
            for index in range(dimension)
        )
        rows = []
        for point in itertools.product(*coordinate_axes):
            point_array = np.asarray(point)
            energy = 1.0 + np.dot(
                np.arange(1.0, dimension + 1.0), point_array
            )
            targets = [energy]
            for index in range(1, dimension):
                targets.append(
                    0.02 * (index + 1.0) + 0.1 * point_array[index]
                )
            auxiliaries = [
                energy * (offset + 2.0)
                for offset in range(len(auxiliary_fields))
            ]
            rows.append(list(point) + targets + auxiliaries)

        fixture = TemporaryEoS(metadata(fields, dimension), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable(fixture.metadata_path)
        self.assertTrue(table.validate().is_valid)
        return table

    @staticmethod
    def make_config(
        table,
        *,
        space="target",
        count=2,
        max_points=1_000_000,
        max_result_bytes=100_000_000,
        max_peak_result_bytes=None,
        auxiliary_fields=(),
        chunk_size=3,
    ):
        envelope = table.find_boundaries(space=space)
        grid = InversionGridSpec(
            space=space,
            axes={name: GridAxisSpec(count) for name in envelope.names},
            max_points=max_points,
            max_result_bytes=max_result_bytes,
            max_peak_result_bytes=max_peak_result_bytes,
        )
        return PipelineConfig(
            grid=grid,
            residual_scales={name: 1.0 + index for index, name in enumerate(table.target_names)},
            auxiliary_fields=auxiliary_fields,
            execution=ExecutionConfig(chunk_size=chunk_size),
        )


class ConfigurationAndPlanningTests(PipelineTestCase):
    def test_configuration_is_deeply_immutable_and_validated(self):
        axes = {"e": GridAxisSpec(3)}
        scales = {"e": 2.0}
        grid = InversionGridSpec("target", axes, max_points=3)
        config = PipelineConfig(grid, scales, auxiliary_fields=())

        axes["e"] = GridAxisSpec(9)
        scales["e"] = 9.0
        self.assertEqual(config.grid.axes["e"].count, 3)
        self.assertEqual(config.residual_scales["e"], 2.0)
        with self.assertRaises(TypeError):
            config.grid.axes["new"] = GridAxisSpec(2)
        with self.assertRaises(TypeError):
            config.residual_scales["e"] = 3.0

        invalid_values = (
            lambda: GridAxisSpec(True),
            lambda: GridAxisSpec(1),
            lambda: GridAxisSpec(2, lower=1.0, upper=1.0),
            lambda: InversionGridSpec("bad", {"e": GridAxisSpec(2)}),
            lambda: InversionGridSpec("target", {"e": GridAxisSpec(2)}, max_points=True),
            lambda: PipelineConfig(grid, {"e": 0.0}),
            lambda: PipelineConfig(grid, {"e": np.nextafter(0.0, 1.0)}),
            lambda: PipelineConfig(grid, {"e": 1.0}, auxiliary_fields=("P", "P")),
            lambda: ExecutionConfig(workers=False),
            lambda: ExecutionConfig(start_method="invalid"),
        )
        for constructor in invalid_values:
            with self.subTest(constructor=constructor):
                with self.assertRaises(PipelineConfigurationError):
                    constructor()

        execution = ExecutionConfig(
            workers=2,
            start_method=ProcessStartMethod.SPAWN,
            chunk_size=7,
            worker_task_size=3,
            max_pending_chunks=4,
            error_policy=ErrorPolicy.ACCUMULATE,
        )
        self.assertEqual(execution.workers, 2)
        self.assertEqual(execution.worker_task_size, 3)
        self.assertEqual(execution.error_policy, ErrorPolicy.ACCUMULATE)

    def test_plan_requires_exact_axis_and_scale_names(self):
        table = self.make_table(2)
        envelope = table.find_boundaries(space="target")
        incomplete_grid = InversionGridSpec(
            "target", {envelope.names[0]: GridAxisSpec(2)}
        )
        with self.assertRaises(PipelineConfigurationError):
            PipelinePlan.from_table(
                table,
                PipelineConfig(
                    incomplete_grid,
                    {name: 1.0 for name in table.target_names},
                    auxiliary_fields=(),
                ),
            )

        complete_grid = InversionGridSpec(
            "target", {name: GridAxisSpec(2) for name in envelope.names}
        )
        with self.assertRaises(PipelineConfigurationError):
            PipelinePlan.from_table(
                table,
                PipelineConfig(
                    complete_grid,
                    {table.target_names[0]: 1.0},
                    auxiliary_fields=(),
                ),
            )

        fresh = ForwardTable(table.metadata_path)
        with self.assertRaises(Exception) as captured:
            PipelinePlan.from_table(fresh, self.make_config(table))
        self.assertIn("validat", str(captured.exception).lower())

    def test_target_grids_are_inclusive_and_c_ordered_in_all_dimensions(self):
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                table = self.make_table(dimension)
                config = self.make_config(
                    table, count=3, chunk_size=5, auxiliary_fields=()
                )
                plan = PipelinePlan.from_table(table, config)
                grid = plan.grid
                envelope = table.find_boundaries(space="target")

                self.assertEqual(grid.space, GridSpace.TARGET)
                self.assertEqual(grid.axis_names, table.target_names)
                self.assertEqual(grid.shape, (3,) * dimension)
                self.assertEqual(grid.size, 3**dimension)
                for index, axis in enumerate(grid.axes):
                    self.assertEqual(axis[0], envelope.lower[index])
                    self.assertEqual(axis[-1], envelope.upper[index])
                    self.assertFalse(axis.flags.writeable)

                chunks = list(plan.iter_chunks())
                indices = np.concatenate([chunk.flat_indices for chunk in chunks])
                values = np.concatenate([chunk.requested_values for chunk in chunks])
                np.testing.assert_array_equal(indices, np.arange(grid.size))
                expected = np.asarray(list(itertools.product(*grid.axes)))
                np.testing.assert_allclose(values, expected)
                np.testing.assert_allclose(
                    np.concatenate([chunk.canonical_targets for chunk in chunks]),
                    expected,
                )
                self.assertTrue(all(np.all(chunk.transform_valid) for chunk in chunks))
                self.assertTrue(all(not chunk.requested_values.flags.writeable for chunk in chunks))

    def test_two_dimensional_last_axis_varies_fastest(self):
        table = self.make_table(2)
        envelope = table.find_boundaries(space="target")
        bounds = {
            envelope.names[0]: GridAxisSpec(2),
            envelope.names[1]: GridAxisSpec(3),
        }
        config = PipelineConfig(
            InversionGridSpec("target", bounds),
            {name: 1.0 for name in table.target_names},
            auxiliary_fields=(),
        )
        grid = PipelinePlan.from_table(table, config).grid
        expected = np.asarray(
            [
                [grid.axes[0][0], grid.axes[1][0]],
                [grid.axes[0][0], grid.axes[1][1]],
                [grid.axes[0][0], grid.axes[1][2]],
                [grid.axes[0][1], grid.axes[1][0]],
                [grid.axes[0][1], grid.axes[1][1]],
                [grid.axes[0][1], grid.axes[1][2]],
            ]
        )
        np.testing.assert_allclose(grid.chunk(0, grid.size).requested_values, expected)
        self.assertEqual(grid.chunk(2, 5).requested_values.shape, (3, 2))
        self.assertEqual(grid.chunk(0, 0).requested_values.shape, (0, 2))
        with self.assertRaises(PipelineConfigurationError):
            grid.chunk(0, grid.size + 1)

    def test_parent_progress_callback_reports_monotonic_atomic_completion(self):
        table = self.make_table(2)
        config = self.make_config(table, count=3, chunk_size=2)
        observations = []

        result = EoSInversionPipeline(table, config).run(
            progress_callback=lambda completed, total: observations.append(
                (completed, total)
            )
        )

        self.assertTrue(observations)
        self.assertEqual(observations[-1], (result.grid.size, result.grid.size))
        self.assertEqual(
            [completed for completed, _ in observations],
            sorted({completed for completed, _ in observations}),
        )
        self.assertTrue(all(total == result.grid.size for _, total in observations))

    def test_bound_and_point_guards_run_before_grid_allocation(self):
        table = self.make_table(2)
        envelope = table.find_boundaries(space="target")
        axes = {
            envelope.names[0]: GridAxisSpec(3),
            envelope.names[1]: GridAxisSpec(3),
        }
        too_small = PipelineConfig(
            InversionGridSpec("target", axes, max_points=8),
            {name: 1.0 for name in table.target_names},
            auxiliary_fields=(),
        )
        with self.assertRaises(PipelineConfigurationError):
            PipelinePlan.from_table(table, too_small)

        outside = {
            envelope.names[0]: GridAxisSpec(
                2, lower=float(envelope.lower[0]) - 1.0
            ),
            envelope.names[1]: GridAxisSpec(2),
        }
        with self.assertRaises(PipelineConfigurationError):
            PipelinePlan.from_table(
                table,
                PipelineConfig(
                    InversionGridSpec("target", outside),
                    {name: 1.0 for name in table.target_names},
                    auxiliary_fields=(),
                ),
            )

    def test_tilde_grid_uses_only_independent_axes_and_converts_chunks(self):
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                table = self.make_table(dimension)
                config = self.make_config(
                    table, space="tilde", count=2, auxiliary_fields=()
                )
                plan = PipelinePlan.from_table(table, config)
                grid = plan.grid
                chunk = grid.chunk(0, grid.size)
                transform = ConstrainedTildeTransform(table.charge_constraints)

                self.assertEqual(
                    grid.axis_names,
                    table.charge_constraints.independent_tilde_names,
                )
                self.assertEqual(
                    grid.dependent_tilde_names,
                    table.charge_constraints.dependent_tilde_names,
                )
                self.assertEqual(len(grid.axis_names), dimension)
                self.assertEqual(
                    chunk.requested_dependent_tilde.shape[1], 4 - dimension
                )
                self.assertTrue(np.all(chunk.transform_valid))
                np.testing.assert_allclose(
                    chunk.canonical_targets,
                    transform.independent_to_targets(chunk.requested_values),
                )
                np.testing.assert_allclose(
                    chunk.requested_dependent_tilde,
                    transform.expand_independent(
                        chunk.requested_values
                    ).dependent_values,
                )

    def test_default_auxiliaries_honor_output_flag_and_explicit_selection(self):
        table = self.make_table(1, auxiliary_fields=(("P", True), ("hidden", False)))
        default_config = self.make_config(table, auxiliary_fields=None)
        default_plan = PipelinePlan.from_table(table, default_config)
        self.assertEqual(default_plan.auxiliary_names, ("P",))

        explicit = self.make_config(table, auxiliary_fields=("hidden",))
        explicit_plan = PipelinePlan.from_table(table, explicit)
        self.assertEqual(explicit_plan.auxiliary_names, ("hidden",))
        with self.assertRaises(PipelineConfigurationError):
            PipelinePlan.from_table(
                table, self.make_config(table, auxiliary_fields=("T",))
            )

    def test_result_memory_guard_matches_exact_allocation(self):
        table = self.make_table(2, auxiliary_fields=(("P", True),))
        first = PipelinePlan.from_table(
            table, self.make_config(table, count=3, auxiliary_fields=None)
        )
        self.assertEqual(
            first.allocate_result_buffer().nbytes, first.estimated_result_bytes
        )
        self.assertEqual(
            first.estimated_peak_result_bytes,
            2 * first.estimated_result_bytes,
        )
        exact = self.make_config(
            table,
            count=3,
            auxiliary_fields=None,
            max_result_bytes=first.estimated_result_bytes,
        )
        self.assertEqual(
            PipelinePlan.from_table(table, exact).estimated_result_bytes,
            first.estimated_result_bytes,
        )
        too_small = self.make_config(
            table,
            count=3,
            auxiliary_fields=None,
            max_result_bytes=first.estimated_result_bytes - 1,
        )
        with self.assertRaises(PipelineConfigurationError):
            PipelinePlan.from_table(table, too_small)

        peak_too_small = self.make_config(
            table,
            count=3,
            auxiliary_fields=None,
            max_peak_result_bytes=first.estimated_peak_result_bytes - 1,
        )
        with self.assertRaises(PipelineConfigurationError):
            PipelinePlan.from_table(table, peak_too_small)

        round_trip = PipelineConfig.from_dict(first.config.to_dict())
        self.assertEqual(round_trip.to_dict(), first.config.to_dict())
        self.assertTrue(round_trip.use_reachability_seed)
        self.assertEqual(
            round_trip.to_dict()["numerical_routing_contract_version"],
            PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
        )
        routing_contract = first.numerical_routing_contract()
        self.assertEqual(
            routing_contract["version"],
            PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
        )
        self.assertEqual(
            first.manifest()["numerical_routing_contract"], routing_contract
        )

        disabled_document = first.config.to_dict()
        disabled_document["use_reachability_seed"] = False
        disabled = PipelineConfig.from_dict(disabled_document)
        self.assertFalse(disabled.use_reachability_seed)
        self.assertFalse(disabled.to_dict()["use_reachability_seed"])

        subcell_document = first.config.to_dict()
        subcell_document["solver"]["enable_subcell_fallback"] = True
        subcell_document["solver"]["max_subcell_depth"] = 5
        subcell_document["solver"]["max_subcell_active_boxes"] = 12345
        subcell = PipelineConfig.from_dict(subcell_document)
        self.assertTrue(subcell.solver.enable_subcell_fallback)
        self.assertEqual(subcell.solver.max_subcell_depth, 5)
        self.assertEqual(subcell.solver.max_subcell_active_boxes, 12345)
        subcell_plan = PipelinePlan.from_table(table, subcell)
        self.assertEqual(
            subcell_plan.numerical_routing_contract()[
                "fallback_cell_resolution"
            ],
            "dyadic_subcells_then_shared_krawczyk",
        )

        legacy_document = first.config.to_dict()
        for name in (
            "enable_subcell_fallback",
            "max_subcell_depth",
            "max_subcell_active_boxes",
        ):
            legacy_document["solver"].pop(name)
        legacy = PipelineConfig.from_dict(legacy_document)
        self.assertFalse(legacy.solver.enable_subcell_fallback)
        self.assertEqual(legacy.solver.max_subcell_depth, 7)
        self.assertEqual(legacy.solver.max_subcell_active_boxes, 1_000_000)

        fixed_batch_document = first.config.to_dict()
        fixed_batch_document["execution"].pop("worker_task_size")
        fixed_batch = PipelineConfig.from_dict(fixed_batch_document)
        self.assertEqual(fixed_batch.execution.worker_task_size, 256)

        partial_document = first.config.to_dict()
        partial_document["solver"].pop("max_subcell_depth")
        with self.assertRaisesRegex(
            PipelineConfigurationError, "all three subcell fields or none"
        ):
            PipelineConfig.from_dict(partial_document)

        with self.assertRaisesRegex(
            PipelineConfigurationError, "use_reachability_seed must be boolean"
        ):
            PipelineConfig(
                grid=first.config.grid,
                residual_scales=first.config.residual_scales,
                use_reachability_seed="yes",
                auxiliary_fields=(),
            )
        incompatible = first.config.to_dict()
        incompatible["numerical_routing_contract_version"] += 1
        with self.assertRaisesRegex(
            PipelineConfigurationError,
            "unsupported numerical routing contract version",
        ):
            PipelineConfig.from_dict(incompatible)


class ResultContractTests(PipelineTestCase):
    @staticmethod
    def _set_code(array, index, value):
        array.reshape(-1)[index] = pipeline_status_code(value)

    def make_result_plan(self):
        table = self.make_table(1)
        config = self.make_config(
            table, count=12, auxiliary_fields=(), chunk_size=4
        )
        return PipelinePlan.from_table(table, config)

    def _populate_usable(self, buffer, indices):
        chunk = buffer.plan.grid.chunk(0, buffer.plan.grid.size)
        targets = chunk.canonical_targets
        source = buffer.source_coordinates.reshape(-1, 1)
        achieved = buffer.achieved_targets.reshape(-1, 1)
        residual = buffer.residual.reshape(-1, 1)
        scaled = buffer.scaled_residual.reshape(-1, 1)
        norm_inf = buffer.scaled_residual_norm_inf.reshape(-1)
        distance = buffer.scaled_distance_l2.reshape(-1)
        for index in indices:
            source[index, 0] = 0.25
            offset = 0.0 if index in (0, 1) else 0.05
            achieved[index, 0] = targets[index, 0] + offset
            residual[index, 0] = offset
            scaled[index, 0] = offset / buffer.plan.residual_scales[0]
            norm_inf[index] = abs(scaled[index, 0])
            distance[index] = abs(scaled[index, 0])

    def test_new_buffer_is_unassigned_and_cannot_be_finalized(self):
        plan = self.make_result_plan()
        buffer = plan.allocate_result_buffer()
        self.assertTrue(np.all(np.isnan(buffer.source_coordinates)))
        self.assertTrue(np.all(buffer.disposition_codes == 0))
        self.assertTrue(np.all(buffer.prescreen_codes == 0))
        with self.assertRaises(PipelineStateError):
            buffer.finalize()

    def test_terminal_truth_table_masks_and_immutable_arrays(self):
        plan = self.make_result_plan()
        buffer = plan.allocate_result_buffer()

        # (disposition, route, prescreen, solve, fast, proof, quality, guess)
        rows = (
            (PointDisposition.SOLVED, PointRoute.FAST, PrescreenStatus.CANDIDATE,
             SolveStatus.CONVERGED_FAST, SolveStatus.CONVERGED_FAST, ProofStatus.NONE,
             None, InitialGuessSource.NEAREST_NODE),
            (PointDisposition.SOLVED, PointRoute.FALLBACK_CERTIFIED, PrescreenStatus.CANDIDATE,
             SolveStatus.CONVERGED_FALLBACK, SolveStatus.FAST_PATH_FAILED,
             ProofStatus.UNIQUE_IN_BOX, None, InitialGuessSource.NEAREST_NODE),
            (PointDisposition.ROOT_CERTIFIED_UNREFINED, PointRoute.FALLBACK_ROOT_UNREFINED,
             PrescreenStatus.CANDIDATE, SolveStatus.CERTIFIED_ROOT_UNREFINED,
             SolveStatus.FAST_PATH_FAILED, ProofStatus.UNIQUE_IN_BOX, None,
             InitialGuessSource.NEAREST_NODE),
            (PointDisposition.UNRESOLVED, PointRoute.FALLBACK_UNRESOLVED,
             PrescreenStatus.CANDIDATE, SolveStatus.UNRESOLVED,
             SolveStatus.FAST_PATH_FAILED, ProofStatus.INCONCLUSIVE, None,
             InitialGuessSource.NEAREST_NODE),
            (PointDisposition.PROJECTED, PointRoute.PROJECTED_PRESCREEN,
             PrescreenStatus.DEFINITELY_UNREACHABLE, None, None, None,
             ProjectionQuality.OPTIMIZER_CONVERGED, None),
            (PointDisposition.PROJECTED, PointRoute.PROJECTED_CERTIFIED_NO_ROOT,
             PrescreenStatus.CANDIDATE, SolveStatus.CERTIFIED_NO_ROOT,
             SolveStatus.FAST_PATH_FAILED, ProofStatus.NO_ROOT_IN_DOMAIN,
             ProjectionQuality.DISCRETE_NODE_FALLBACK, InitialGuessSource.NEAREST_NODE),
            (PointDisposition.CERTIFIED_UNREACHABLE,
             PointRoute.CERTIFIED_UNREACHABLE_PRESCREEN,
             PrescreenStatus.DEFINITELY_UNREACHABLE, None, None, None, None, None),
            (PointDisposition.CERTIFIED_UNREACHABLE,
             PointRoute.CERTIFIED_UNREACHABLE_FALLBACK, PrescreenStatus.CANDIDATE,
             SolveStatus.CERTIFIED_NO_ROOT, SolveStatus.FAST_PATH_FAILED,
             ProofStatus.NO_ROOT_IN_DOMAIN, None, InitialGuessSource.NEAREST_NODE),
            (PointDisposition.PROJECTION_FAILED, PointRoute.PROJECTION_FAILED_PRESCREEN,
             PrescreenStatus.DEFINITELY_UNREACHABLE, None, None, None, None, None),
            (PointDisposition.PROJECTION_FAILED,
             PointRoute.PROJECTION_FAILED_CERTIFIED_NO_ROOT, PrescreenStatus.CANDIDATE,
             SolveStatus.CERTIFIED_NO_ROOT, SolveStatus.FAST_PATH_FAILED,
             ProofStatus.NO_ROOT_IN_DOMAIN, None, InitialGuessSource.NEAREST_NODE),
            (PointDisposition.INVALID_TARGET, PointRoute.INVALID_TRANSFORM,
             PrescreenStatus.NOT_RUN, None, None, None, None, None),
            (PointDisposition.ERROR, PointRoute.ERROR, PrescreenStatus.NOT_RUN,
             None, None, None, None, None),
        )
        for index, row in enumerate(rows):
            disposition, route, prescreen, solve, fast, proof, quality, guess = row
            self._set_code(buffer.disposition_codes, index, disposition)
            self._set_code(buffer.route_codes, index, route)
            self._set_code(buffer.prescreen_codes, index, prescreen)
            for array, value in (
                (buffer.solve_status_codes, solve),
                (buffer.fast_status_codes, fast),
                (buffer.proof_status_codes, proof),
                (buffer.projection_quality_codes, quality),
                (buffer.initial_guess_source_codes, guess),
            ):
                if value is not None:
                    self._set_code(array, index, value)

        self._populate_usable(buffer, (0, 1, 4, 5))
        result = buffer.finalize({"test": "truth_table"})

        np.testing.assert_array_equal(
            result.solution_mask,
            [True, True, False, False, False, False, False, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            result.projected_mask,
            [False, False, False, False, True, True, False, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            result.usable_state_mask,
            [True, True, False, False, True, True, False, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            result.certified_reachable_mask,
            [False, True, True, False, False, False, False, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            result.certified_unreachable_mask,
            [False, False, False, False, True, True, True, True, True, True, False, False],
        )
        np.testing.assert_array_equal(
            result.unknown_mask,
            [False, False, False, True, False, False, False, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            result.projection_failed_mask,
            [False, False, False, False, False, False, False, False, True, True, False, False],
        )
        np.testing.assert_array_equal(
            result.prescreen_evaluated_mask,
            [True] * 10 + [False, False],
        )
        self.assertFalse(result.source_coordinates.flags.writeable)
        self.assertFalse(result.solution_mask.flags.writeable)
        self.assertEqual(sum(result.status_counts.values()), plan.grid.size)
        self.assertEqual(result.manifest["run"]["test"], "truth_table")
        with self.assertRaises(ValueError):
            result.source_coordinates[0, 0] = 7.0
        with self.assertRaises(PipelineStateError):
            buffer.finalize()

    def test_finalize_rejects_route_status_and_residual_inconsistency(self):
        plan = self.make_result_plan()
        buffer = plan.allocate_result_buffer()
        for index in range(plan.grid.size):
            self._set_code(buffer.disposition_codes, index, PointDisposition.INVALID_TARGET)
            self._set_code(buffer.route_codes, index, PointRoute.INVALID_TRANSFORM)
            self._set_code(buffer.prescreen_codes, index, PrescreenStatus.NOT_RUN)
        # Deliberately pair an invalid target with a solver route.
        self._set_code(buffer.route_codes, 0, PointRoute.FAST)
        with self.assertRaises(PipelineStateError):
            buffer.finalize()

        buffer = plan.allocate_result_buffer()
        for index in range(plan.grid.size):
            self._set_code(buffer.disposition_codes, index, PointDisposition.INVALID_TARGET)
            self._set_code(buffer.route_codes, index, PointRoute.INVALID_TRANSFORM)
            self._set_code(buffer.prescreen_codes, index, PrescreenStatus.NOT_RUN)
        self._set_code(buffer.disposition_codes, 0, PointDisposition.SOLVED)
        self._set_code(buffer.route_codes, 0, PointRoute.FAST)
        self._set_code(buffer.prescreen_codes, 0, PrescreenStatus.CANDIDATE)
        self._set_code(buffer.solve_status_codes, 0, SolveStatus.CONVERGED_FAST)
        self._set_code(buffer.fast_status_codes, 0, SolveStatus.CONVERGED_FAST)
        self._set_code(buffer.proof_status_codes, 0, ProofStatus.NONE)
        self._set_code(buffer.initial_guess_source_codes, 0, InitialGuessSource.NEAREST_NODE)
        self._populate_usable(buffer, (0,))
        buffer.residual.reshape(-1, 1)[0, 0] = 99.0
        with self.assertRaises(PipelineStateError):
            buffer.finalize()


if __name__ == "__main__":
    unittest.main()
