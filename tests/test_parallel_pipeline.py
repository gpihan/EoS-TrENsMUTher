"""Focused tests for Phase-5 process-parallel pipeline execution.

The numerical parity tests intentionally exercise only the public
``EoSInversionPipeline.run`` contract.  Process-specific fault injection is
limited to ``fork`` platforms because an in-memory mock cannot be inherited by
workers started with ``spawn`` or ``forkserver``.
"""

import itertools
import json
import multiprocessing
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eos_trensmuther import (
    EoSInversionPipeline,
    ErrorPolicy,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    PipelineConfig,
    PipelineConfigurationError,
    PipelineStateError,
    ProcessStartMethod,
    ProjectionConfig,
    ReachabilityConfig,
)

from tests.test_tables import TemporaryEoS, field_spec, metadata


_RESULT_ARRAY_NAMES = (
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

_FLOAT_RESULT_ARRAY_NAMES = {
    "source_coordinates",
    "achieved_targets",
    "residual",
    "scaled_residual",
    "scaled_residual_norm_inf",
    "scaled_distance_l2",
    "auxiliary_values",
    "achieved_tilde_values",
    "line_mixture_fraction",
    "alternate_single_phase_source",
}

_ORIGINAL_WORKER_CHUNK = None
_WORKER_COMPLETION_ORDER = None


def _run_delayed_first_chunk(chunk):
    """Fork-only test wrapper that forces later work to finish first."""

    if chunk.start == 0:
        time.sleep(0.5)
    result = _ORIGINAL_WORKER_CHUNK(chunk)
    _WORKER_COMPLETION_ORDER.append(chunk.start)
    return result


class ParallelPipelineTestCase(unittest.TestCase):
    def make_table(self, fields, rows):
        dimension = sum(
            specification["role"] == "coordinate"
            for specification in fields.values()
        )
        fixture = TemporaryEoS(metadata(fields, dimension), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable(fixture.metadata_path)
        self.assertTrue(table.validate().is_valid)
        return table

    @staticmethod
    def make_config(
        table,
        *,
        space,
        count,
        workers,
        start_method=ProcessStartMethod.SPAWN,
        chunk_size=1,
        worker_task_size=256,
        max_pending_chunks=2,
        error_policy=ErrorPolicy.FAIL_FAST,
        auxiliary_fields=("P",),
        projection_enabled=True,
    ):
        envelope = table.find_boundaries(space=space)
        grid = InversionGridSpec(
            space,
            {name: GridAxisSpec(count) for name in envelope.names},
        )
        return PipelineConfig(
            grid=grid,
            residual_scales={name: 1.0 for name in table.target_names},
            projection=ProjectionConfig(
                enabled=projection_enabled,
                nearest_cell_starts=4,
                max_evaluations_per_start=60,
            ),
            execution=ExecutionConfig(
                workers=workers,
                start_method=start_method,
                chunk_size=chunk_size,
                worker_task_size=worker_task_size,
                max_pending_chunks=max_pending_chunks,
                error_policy=error_policy,
            ),
            auxiliary_fields=auxiliary_fields,
        )

    @staticmethod
    def assert_result_columns_identical(test_case, serial, parallel):
        test_case.assertEqual(serial.grid.shape, parallel.grid.shape)
        test_case.assertEqual(serial.grid.ordering, "C_last_axis_fastest")
        test_case.assertEqual(parallel.grid.ordering, "C_last_axis_fastest")
        for serial_axis, parallel_axis in zip(
            serial.grid.axes, parallel.grid.axes
        ):
            np.testing.assert_array_equal(serial_axis, parallel_axis)
        for name in _RESULT_ARRAY_NAMES:
            with test_case.subTest(result_column=name):
                if name == "initial_guess_source_codes":
                    # Opportunistic direct-neighbor availability is allowed to
                    # differ with worker scheduling.  It is seed provenance,
                    # not a physical output or terminal status.
                    continue
                if name in _FLOAT_RESULT_ARRAY_NAMES:
                    np.testing.assert_allclose(
                        getattr(serial, name),
                        getattr(parallel, name),
                        rtol=1.0e-9,
                        atol=1.0e-10,
                        equal_nan=True,
                    )
                else:
                    np.testing.assert_array_equal(
                        getattr(serial, name), getattr(parallel, name)
                    )
        test_case.assertEqual(dict(serial.status_counts), dict(parallel.status_counts))
        for name in (
            "status_counts",
            "errors",
            "projection_failures",
        ):
            test_case.assertEqual(
                serial.manifest["run"][name], parallel.manifest["run"][name]
            )
        serial_summary = dict(serial.manifest["run"]["status_summary"])
        parallel_summary = dict(parallel.manifest["run"]["status_summary"])
        serial_summary.pop("initial_guess_source", None)
        parallel_summary.pop("initial_guess_source", None)
        test_case.assertEqual(serial_summary, parallel_summary)
        invariant_counters = (
            "prescreen_evaluated",
            "prescreen_excluded",
            "exact_candidates",
            "tolerance_shell_candidates",
            "invalid_targets",
            "mixed_candidates",
            "exact_mixed_candidates",
            "projection_attempted",
            "auxiliary_points_evaluated",
            "tilde_points_evaluated",
        )
        for key in invariant_counters:
            test_case.assertEqual(
                serial.manifest["run"]["stage_counters"][key],
                parallel.manifest["run"]["stage_counters"][key],
            )
        test_case.assertEqual(
            serial.manifest["run"]["stage_counters"]["seed_global_nearest_node"],
            0,
        )
        test_case.assertEqual(
            parallel.manifest["run"]["stage_counters"]["seed_global_nearest_node"],
            0,
        )


class ParallelParityTests(ParallelPipelineTestCase):
    def make_correlated_target_table(self):
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

    def make_one_dimensional_tilde_table(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "P": field_spec("auxiliary", 2),
        }
        rows = [
            (temperature, temperature, 2.0 * temperature)
            for temperature in (0.2, 0.5, 0.8)
        ]
        return self.make_table(fields, rows)

    def make_four_dimensional_identity_table(self):
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
            (0.2, 0.5),
            (0.0, 0.3),
            (-0.1, 0.1),
            (-0.2, 0.2),
        )
        rows = [list(point) + list(point) for point in itertools.product(*axes)]
        return self.make_table(fields, rows)

    def run_serial_and_parallel(self, table, **configuration):
        serial_config = self.make_config(table, workers=1, **configuration)
        parallel_config = replace(
            serial_config,
            execution=replace(serial_config.execution, workers=2),
        )
        serial = EoSInversionPipeline(table, serial_config).run()
        parallel = EoSInversionPipeline(table, parallel_config).run()
        return serial, parallel

    def test_target_grid_two_processes_exactly_matches_serial(self):
        table = self.make_correlated_target_table()

        serial, parallel = self.run_serial_and_parallel(
            table,
            space="target",
            count=3,
            chunk_size=1,
            max_pending_chunks=None,
        )

        self.assert_result_columns_identical(self, serial, parallel)
        execution = parallel.manifest["run"]["execution"]
        self.assertEqual(execution["mode"], "process")
        self.assertEqual(execution["workers_requested"], 2)
        self.assertEqual(execution["executor_max_workers"], 2)
        self.assertEqual(execution["maximum_concurrent_chunks"], 4)
        self.assertEqual(execution["maximum_concurrent_internal_batches"], 4)
        self.assertEqual(execution["submitted_worker_tasks"], 9)
        self.assertEqual(execution["completed_worker_tasks"], 9)
        self.assertEqual(
            execution["immutable_worker_state_backend"],
            "numpy_npy_mmap_mode_r",
        )
        self.assertEqual(execution["start_method"], "spawn")
        self.assertEqual(execution["max_pending_chunks"], 4)
        self.assertIsNotNone(execution["shared_state_manifest"])
        setup_count = parallel.manifest["run"]["performance"][
            "worker_initialization_seconds"
        ]["count"]
        self.assertGreaterEqual(setup_count, 1)
        self.assertLessEqual(setup_count, 2)
        self.assertLess(setup_count, execution["submitted_worker_tasks"])
        self.assertTrue(np.all(parallel.source_coordinates.flags.c_contiguous))

    def test_tilde_grid_two_processes_exactly_matches_serial(self):
        table = self.make_one_dimensional_tilde_table()

        serial, parallel = self.run_serial_and_parallel(
            table,
            space="tilde",
            count=7,
            chunk_size=2,
            max_pending_chunks=1,
        )

        self.assert_result_columns_identical(self, serial, parallel)
        self.assertTrue(np.all(parallel.achieved_tilde_defined))
        self.assertEqual(
            parallel.manifest["run"]["execution"]["max_pending_chunks"], 1
        )

    def test_hierarchical_4d_config_is_identical_in_serial_and_spawn_workers(self):
        table = self.make_four_dimensional_identity_table()
        serial_config = self.make_config(
            table,
            space="target",
            count=2,
            workers=1,
            chunk_size=3,
            auxiliary_fields=(),
            projection_enabled=False,
        )
        parallel_config = replace(
            serial_config,
            execution=replace(serial_config.execution, workers=2),
        )
        reachability = ReachabilityConfig(
            method="hierarchical",
            hierarchy_levels=7,
            query_batch_size=2,
            cell_batch_size=1,
        )

        serial = EoSInversionPipeline(
            table,
            serial_config,
            reachability_config=reachability,
        ).run()
        parallel = EoSInversionPipeline(
            table,
            parallel_config,
            reachability_config=reachability,
        ).run()

        self.assert_result_columns_identical(self, serial, parallel)
        self.assertEqual(serial.manifest["run"]["reachability"]["method"], "hierarchical")
        self.assertEqual(
            parallel.manifest["run"]["reachability"]["configuration"][
                "hierarchy_levels"
            ],
            7,
        )


class ParallelConfigurationAndFailureTests(ParallelPipelineTestCase):
    def make_affine_table(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "P": field_spec("auxiliary", 2),
        }
        rows = [
            (temperature, 1.0 + temperature, 2.0 + 2.0 * temperature)
            for temperature in (0.2, 0.5, 0.8)
        ]
        return self.make_table(fields, rows)

    def test_execution_config_validates_workers_start_method_and_pending_bound(self):
        for value in (False, np.bool_(True), 0, -1, 1.5, "2"):
            with self.subTest(field="workers", value=value):
                with self.assertRaises(PipelineConfigurationError):
                    ExecutionConfig(workers=value)

        for value in (False, np.bool_(True), 0, -1, 1.5, "2"):
            with self.subTest(field="max_pending_chunks", value=value):
                with self.assertRaises(PipelineConfigurationError):
                    ExecutionConfig(max_pending_chunks=value)

        for value in (False, np.bool_(True), 0, -1, 1.5, "2"):
            with self.subTest(field="worker_task_size", value=value):
                with self.assertRaises(PipelineConfigurationError):
                    ExecutionConfig(worker_task_size=value)

        with self.assertRaises(PipelineConfigurationError):
            ExecutionConfig(start_method="not-a-method")

        for method in ProcessStartMethod:
            with self.subTest(start_method=method.value):
                execution = ExecutionConfig(
                    workers=2,
                    start_method=method.value,
                    chunk_size=3,
                    max_pending_chunks=1,
                )
                self.assertIs(execution.start_method, method)
                self.assertEqual(execution.max_pending_chunks, 1)

        self.assertIsNone(ExecutionConfig().max_pending_chunks)
        self.assertEqual(ExecutionConfig().worker_task_size, 8)

    def test_spawn_workers_reuse_the_explicit_data_path_override(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        document = metadata(fields, 1)
        document["data_file"] = "intentionally-missing.dat"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "metadata.json"
            override_path = root / "actual-table.dat"
            metadata_path.write_text(json.dumps(document), encoding="utf-8")
            np.savetxt(
                override_path,
                [(0.2, 1.2), (0.5, 1.5), (0.8, 1.8)],
            )
            table = ForwardTable.load_validated(
                metadata_path, data_path=override_path
            )
            config = self.make_config(
                table,
                space="target",
                count=5,
                workers=2,
                chunk_size=1,
                auxiliary_fields=(),
                projection_enabled=False,
            )
            result = EoSInversionPipeline(table, config).run()

        self.assertEqual(result.grid.size, 5)
        self.assertEqual(result.status_counts["solved"], 5)

    def test_spawn_workers_do_not_reload_source_data_after_snapshot(self):
        table = self.make_affine_table()
        pipelines = []
        for policy in (ErrorPolicy.FAIL_FAST, ErrorPolicy.ACCUMULATE):
            config = self.make_config(
                table,
                space="target",
                count=3,
                workers=2,
                start_method=ProcessStartMethod.SPAWN,
                error_policy=policy,
            )
            pipelines.append((policy, EoSInversionPipeline(table, config)))
        data_path = Path(table.data_path)
        unavailable_path = data_path.with_name(data_path.name + ".unavailable")
        data_path.rename(unavailable_path)

        self.addCleanup(
            lambda: unavailable_path.rename(data_path)
            if unavailable_path.exists() and not data_path.exists()
            else None
        )
        for policy, pipeline in pipelines:
            with self.subTest(error_policy=policy.value):
                result = pipeline.run()
                self.assertEqual(result.grid.size, 3)
                self.assertEqual(result.manifest["run"]["errors"]["count"], 0)
                self.assertEqual(
                    result.manifest["run"]["execution"][
                        "immutable_worker_state_backend"
                    ],
                    "numpy_npy_mmap_mode_r",
                )

    @unittest.skipUnless(
        "fork" in multiprocessing.get_all_start_methods(),
        "fork-specific completion-order injection is unavailable",
    )
    def test_out_of_order_cross_chunk_tasks_are_aggregated_in_c_order(self):
        table = self.make_affine_table()
        serial_config = self.make_config(
            table,
            space="target",
            count=300,
            workers=1,
            start_method=ProcessStartMethod.FORK,
            chunk_size=150,
            worker_task_size=75,
            max_pending_chunks=4,
        )
        parallel_config = replace(
            serial_config,
            execution=replace(serial_config.execution, workers=2),
        )
        serial = EoSInversionPipeline(table, serial_config).run()

        import eos_trensmuther.pipeline as pipeline_module

        global _ORIGINAL_WORKER_CHUNK, _WORKER_COMPLETION_ORDER
        _ORIGINAL_WORKER_CHUNK = pipeline_module._run_pipeline_worker_chunk
        try:
            with multiprocessing.Manager() as manager:
                _WORKER_COMPLETION_ORDER = manager.list()
                with patch.object(
                    pipeline_module,
                    "_run_pipeline_worker_chunk",
                    new=_run_delayed_first_chunk,
                ):
                    parallel = EoSInversionPipeline(
                        table, parallel_config
                    ).run()
                completion_order = list(_WORKER_COMPLETION_ORDER)
        finally:
            _ORIGINAL_WORKER_CHUNK = None
            _WORKER_COMPLETION_ORDER = None

        self.assertEqual(sorted(completion_order), [0, 75, 150, 225])
        self.assertNotEqual(completion_order[0], 0)
        execution = parallel.manifest["run"]["execution"]
        self.assertEqual(execution["maximum_concurrent_chunks"], 2)
        self.assertEqual(execution["submitted_worker_tasks"], 4)
        self.assertEqual(execution["internal_batch_size"], 75)
        self.assert_result_columns_identical(self, serial, parallel)

    @unittest.skipUnless(
        "fork" in multiprocessing.get_all_start_methods(),
        "fork-specific inherited fault injection is unavailable",
    )
    def test_parallel_point_error_obeys_accumulate_and_fail_fast_policies(self):
        table = self.make_affine_table()
        base = self.make_config(
            table,
            space="target",
            count=4,
            workers=2,
            start_method=ProcessStartMethod.FORK,
            chunk_size=1,
            max_pending_chunks=1,
            error_policy=ErrorPolicy.ACCUMULATE,
        )

        with patch(
            "eos_trensmuther.pipeline.EoSRootSolver.solve",
            side_effect=RuntimeError("synthetic parallel point failure"),
        ):
            accumulated = EoSInversionPipeline(table, base).run()

        self.assertTrue(np.all(accumulated.error_mask))
        self.assertEqual(
            accumulated.manifest["run"]["errors"]["count"],
            accumulated.grid.size,
        )
        error_indices = [
            record["flat_index"]
            for record in accumulated.manifest["run"]["errors"]["records"]
        ]
        self.assertEqual(error_indices, list(range(accumulated.grid.size)))

        fail_fast = replace(
            base,
            execution=replace(
                base.execution, error_policy=ErrorPolicy.FAIL_FAST
            ),
        )
        with patch(
            "eos_trensmuther.pipeline.EoSRootSolver.solve",
            side_effect=RuntimeError("synthetic parallel point failure"),
        ):
            with self.assertRaises(PipelineStateError):
                EoSInversionPipeline(table, fail_fast).run()


if __name__ == "__main__":
    unittest.main()
