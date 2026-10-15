"""Restart and compatibility tests for atomic pipeline checkpoints."""

import json
import multiprocessing
import tempfile
import time
import unittest
import zipfile
from dataclasses import fields as dataclass_fields, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eos_trensmuther import (
    CheckpointConfig,
    CheckpointStore,
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    PersistenceIntegrityError,
    PersistenceVersionError,
    PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
    PIPELINE_RUN_MANIFEST_VERSION,
    PipelineConfig,
    PipelineStateError,
    ProjectionConfig,
    ReachabilityConfig,
)
from eos_trensmuther import checkpoint as checkpoint_module
from eos_trensmuther import outputs as outputs_module

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


_ORIGINAL_PHASE9_WORKER_TASK = None
_PHASE9_COMPLETED_TASK_STARTS = None


def _run_phase9_interrupted_worker_task(chunk):
    """Deterministically publish one chunk, then fail another partial chunk."""

    start = int(chunk.start)
    if start in (0, 1):
        time.sleep(0.005)
    elif start == 2:
        time.sleep(0.01)
    elif start == 3:
        time.sleep(0.20)
        raise RuntimeError("synthetic Phase-9 interruption")
    else:
        time.sleep(0.30)
    result = _ORIGINAL_PHASE9_WORKER_TASK(chunk)
    _PHASE9_COMPLETED_TASK_STARTS.append(start)
    return result


class CheckpointTests(unittest.TestCase):
    def make_table_and_config(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "P": field_spec("auxiliary", 2),
        }
        rows = [
            (temperature, 1.0 + 2.0 * temperature, 3.0 * temperature)
            for temperature in (0.2, 0.5, 0.8)
        ]
        fixture = TemporaryEoS(metadata(fields, 1), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        envelope = table.find_boundaries(space="target")
        config = PipelineConfig(
            grid=InversionGridSpec(
                "target", {envelope.names[0]: GridAxisSpec(7)}
            ),
            residual_scales={"e": 1.0},
            projection=ProjectionConfig(
                enabled=True,
                nearest_cell_starts=2,
                max_evaluations_per_start=40,
            ),
            execution=ExecutionConfig(workers=1, chunk_size=2),
            auxiliary_fields=("P",),
        )
        return table, config

    @staticmethod
    def assert_columns_equal(test_case, first, second):
        for name in _RESULT_ARRAY_NAMES:
            if name == "initial_guess_source_codes":
                continue
            if name in _FLOAT_RESULT_ARRAY_NAMES:
                np.testing.assert_allclose(
                    getattr(first, name),
                    getattr(second, name),
                    rtol=1.0e-9,
                    atol=1.0e-10,
                    equal_nan=True,
                )
            else:
                np.testing.assert_array_equal(
                    getattr(first, name), getattr(second, name)
                )
        test_case.assertEqual(dict(first.status_counts), dict(second.status_counts))

    @staticmethod
    def assert_scientific_diagnostics_equal(test_case, first, second):
        for name in (
            "active_domain",
            "reachability",
            "first_order",
            "auxiliary_evaluation",
            "projection",
            "stage_counters",
            "status_counts",
            "status_summary",
            "errors",
            "projection_failures",
        ):
            first_value = first.manifest["run"][name]
            second_value = second.manifest["run"][name]
            if name == "stage_counters":
                first_value = {
                    key: value
                    for key, value in first_value.items()
                    if not key.startswith("seed_")
                }
                second_value = {
                    key: value
                    for key, value in second_value.items()
                    if not key.startswith("seed_")
                }
            elif name == "status_summary":
                first_value = dict(first_value)
                second_value = dict(second_value)
                first_value.pop("initial_guess_source", None)
                second_value.pop("initial_guess_source", None)
            test_case.assertEqual(
                first_value,
                second_value,
                msg="scientific run diagnostic differs at %s" % name,
            )

    def make_partial_checkpoint(self, table, config, directory):
        pipeline = EoSInversionPipeline(table, config)
        store = CheckpointStore(
            CheckpointConfig(directory),
            pipeline.plan(),
            table_fingerprint=table.snapshot_fingerprint(),
            reachability_configuration={
                field.name: getattr(pipeline.reachability_config, field.name)
                for field in dataclass_fields(pipeline.reachability_config)
            },
        )
        pipeline._prepare()
        first = next(pipeline.plan().iter_chunks())
        store.save_chunk(pipeline._execute_chunk(first))
        return store

    def test_current_persistence_versions_are_frozen(self):
        self.assertEqual(PIPELINE_RUN_MANIFEST_VERSION, 11)
        self.assertEqual(PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION, 3)

    def test_subcell_policy_but_not_execution_settings_separates_fingerprints(self):
        table, config = self.make_table_and_config()
        reachability = ReachabilityConfig()
        reachability_document = {
            field.name: getattr(reachability, field.name)
            for field in dataclass_fields(reachability)
        }
        changed_execution = replace(
            config,
            execution=replace(
                config.execution,
                workers=3,
                start_method="spawn",
                worker_task_size=2,
                max_pending_chunks=6,
            ),
        )
        scientific_configs = (
            config,
            replace(
                config,
                solver=replace(
                    config.solver,
                    enable_subcell_fallback=False,
                ),
            ),
            replace(
                config,
                solver=replace(config.solver, max_subcell_depth=6),
            ),
            replace(
                config,
                solver=replace(
                    config.solver,
                    max_subcell_active_boxes=12345,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def store(name, selected):
                return CheckpointStore(
                    CheckpointConfig(root / name),
                    EoSInversionPipeline(table, selected).plan(),
                    table_fingerprint=table.snapshot_fingerprint(),
                    reachability_configuration=reachability_document,
                )

            baseline = store("baseline", config)
            execution_only = store("execution", changed_execution)
            scientific = tuple(
                store("scientific_%d" % index, selected)
                for index, selected in enumerate(scientific_configs[1:])
            )

        self.assertEqual(
            baseline.run_fingerprint, execution_only.run_fingerprint
        )
        self.assertEqual(
            baseline.layout_fingerprint, execution_only.layout_fingerprint
        )
        scientific_fingerprints = {
            baseline.run_fingerprint,
            *(item.run_fingerprint for item in scientific),
        }
        self.assertEqual(len(scientific_fingerprints), 4)

    def test_reachability_method_and_depth_separate_checkpoint_fingerprints(self):
        table, config = self.make_table_and_config()
        plan = EoSInversionPipeline(table, config).plan()
        exhaustive = ReachabilityConfig()
        exhaustive_document = {
            field.name: getattr(exhaustive, field.name)
            for field in dataclass_fields(exhaustive)
        }
        hierarchical_seven = dict(
            exhaustive_document,
            method="hierarchical",
            hierarchy_levels=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            stores = [
                CheckpointStore(
                    CheckpointConfig(Path(directory) / str(index)),
                    plan,
                    table_fingerprint=table.snapshot_fingerprint(),
                    reachability_configuration=document,
                )
                for index, document in enumerate(
                    (exhaustive_document, hierarchical_seven)
                )
            ]
        self.assertNotEqual(stores[0].run_fingerprint, stores[1].run_fingerprint)

    def test_reachability_seed_option_separates_checkpoint_fingerprints(self):
        table, enabled = self.make_table_and_config()
        disabled = replace(enabled, use_reachability_seed=False)
        reachability = ReachabilityConfig()
        reachability_document = {
            field.name: getattr(reachability, field.name)
            for field in dataclass_fields(reachability)
        }
        with tempfile.TemporaryDirectory() as directory:
            stores = [
                CheckpointStore(
                    CheckpointConfig(Path(directory) / str(index)),
                    EoSInversionPipeline(table, config).plan(),
                    table_fingerprint=table.snapshot_fingerprint(),
                    reachability_configuration=reachability_document,
                )
                for index, config in enumerate((enabled, disabled))
            ]
        self.assertNotEqual(stores[0].run_fingerprint, stores[1].run_fingerprint)

    def test_partial_serial_checkpoint_resumes_to_identical_columns(self):
        table, config = self.make_table_and_config()
        baseline = EoSInversionPipeline(table, config).run()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            self.make_partial_checkpoint(table, config, checkpoint)
            abandoned = checkpoint / "chunks" / ".abandoned.eoschunk.tmp"
            abandoned.write_bytes(b"incomplete")
            resumed = EoSInversionPipeline(table, config).run(
                checkpoint=CheckpointConfig(checkpoint)
            )
            self.assert_columns_equal(self, baseline, resumed)
            provenance = resumed.manifest["run"]["checkpoint"]
            self.assertEqual(provenance["restored_chunks"], 1)
            self.assertEqual(provenance["computed_chunks_this_invocation"], 3)
            self.assertTrue((checkpoint / "checkpoint.json").is_file())
            self.assertEqual(len(list((checkpoint / "chunks").glob("*.eoschunk"))), 4)
            self.assertTrue(abandoned.exists())
            self.assertEqual(
                list((checkpoint / "chunks").glob("[!.]*.tmp")), []
            )
            descriptor = json.loads(
                (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
            )
            run_contract = descriptor["run_contract"]
            self.assertEqual(
                run_contract["pipeline_run_manifest_version"],
                PIPELINE_RUN_MANIFEST_VERSION,
            )
            self.assertEqual(
                run_contract["numerical_routing_contract"]["version"],
                PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
            )
            with zipfile.ZipFile(
                sorted((checkpoint / "chunks").glob("*.eoschunk"))[0]
            ) as archive:
                self.assertFalse(
                    any("localization" in name for name in archive.namelist())
                )

    def test_older_routing_contract_checkpoint_is_rejected_for_resume(self):
        table, config = self.make_table_and_config()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            self.make_partial_checkpoint(table, config, checkpoint)
            manifest_path = checkpoint / "checkpoint.json"
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            document.pop("integrity")
            run_contract = document["run_contract"]
            run_contract["pipeline_run_manifest_version"] = (
                PIPELINE_RUN_MANIFEST_VERSION - 1
            )
            run_contract["numerical_routing_contract"]["version"] = (
                PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION - 1
            )
            run_contract["configuration"][
                "numerical_routing_contract_version"
            ] = PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION - 1
            legacy = checkpoint_module._with_integrity(
                "PipelineCheckpoint", document
            )
            manifest_path.write_bytes(outputs_module._canonical_json(legacy))

            with self.assertRaisesRegex(
                PersistenceVersionError,
                "older numerical routing contract",
            ):
                EoSInversionPipeline(table, config).run(
                    checkpoint=CheckpointConfig(checkpoint)
                )

    @unittest.skipUnless(
        "fork" in multiprocessing.get_all_start_methods(),
        "fork-specific deterministic interruption injection is unavailable",
    )
    def test_dynamic_interruption_resumes_without_republishing_complete_chunks(self):
        table, serial_config = self.make_table_and_config()
        baseline = EoSInversionPipeline(table, serial_config).run()
        interrupted_config = replace(
            serial_config,
            execution=replace(
                serial_config.execution,
                workers=2,
                start_method="fork",
                worker_task_size=1,
                max_pending_chunks=4,
            ),
        )
        continuous_process = EoSInversionPipeline(
            table, interrupted_config
        ).run()
        self.assert_columns_equal(self, baseline, continuous_process)
        self.assert_scientific_diagnostics_equal(
            self, baseline, continuous_process
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            import eos_trensmuther.pipeline as pipeline_module

            global _ORIGINAL_PHASE9_WORKER_TASK
            global _PHASE9_COMPLETED_TASK_STARTS
            _ORIGINAL_PHASE9_WORKER_TASK = (
                pipeline_module._run_pipeline_worker_chunk
            )
            try:
                with multiprocessing.Manager() as manager:
                    _PHASE9_COMPLETED_TASK_STARTS = manager.list()
                    with patch.object(
                        pipeline_module,
                        "_run_pipeline_worker_chunk",
                        new=_run_phase9_interrupted_worker_task,
                    ):
                        with self.assertRaisesRegex(
                            PipelineStateError,
                            "synthetic Phase-9 interruption",
                        ):
                            EoSInversionPipeline(
                                table, interrupted_config
                            ).run(checkpoint=CheckpointConfig(checkpoint))
                    completed_task_starts = list(
                        _PHASE9_COMPLETED_TASK_STARTS
                    )
            finally:
                _ORIGINAL_PHASE9_WORKER_TASK = None
                _PHASE9_COMPLETED_TASK_STARTS = None

            # Task [2, 3) returned, but sibling [3, 4) failed. The containing
            # atomic chunk must therefore remain unpublished.
            self.assertIn(2, completed_task_starts)
            incomplete = (
                checkpoint
                / "chunks"
                / "000000000002-000000000004.eoschunk"
            )
            self.assertFalse(incomplete.exists())
            published_paths = sorted(
                (checkpoint / "chunks").glob("*.eoschunk")
            )
            self.assertGreaterEqual(len(published_paths), 1)
            published_bytes = {
                path.name: path.read_bytes() for path in published_paths
            }
            descriptor = json.loads(
                (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertFalse(descriptor["completed"])
            # The descriptor remains the immutable incomplete base document
            # until final completion. Resume discovers and validates each
            # independently atomic chunk file rather than trusting a mutable
            # progress counter.
            self.assertEqual(descriptor["completed_chunk_count"], 0)

            resume_config = replace(
                interrupted_config,
                execution=replace(
                    interrupted_config.execution,
                    workers=3,
                    worker_task_size=2,
                    max_pending_chunks=6,
                ),
            )
            resumed = EoSInversionPipeline(table, resume_config).run(
                checkpoint=CheckpointConfig(checkpoint)
            )
            self.assert_columns_equal(self, baseline, resumed)
            self.assert_columns_equal(self, continuous_process, resumed)
            self.assert_scientific_diagnostics_equal(self, baseline, resumed)
            self.assert_scientific_diagnostics_equal(
                self, continuous_process, resumed
            )
            for path in published_paths:
                self.assertEqual(path.read_bytes(), published_bytes[path.name])
            run = resumed.manifest["run"]
            self.assertEqual(
                run["checkpoint"]["restored_chunks"], len(published_paths)
            )
            self.assertEqual(
                run["checkpoint"]["computed_chunks_this_invocation"],
                4 - len(published_paths),
            )
            self.assertEqual(
                len(list((checkpoint / "chunks").glob("*.eoschunk"))), 4
            )

    def test_completed_checkpoint_can_resume_with_different_worker_and_task_size(self):
        table, config = self.make_table_and_config()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            first = EoSInversionPipeline(table, config).run(
                checkpoint=CheckpointConfig(checkpoint)
            )
            two_workers = replace(
                config,
                execution=replace(
                    config.execution,
                    workers=2,
                    worker_task_size=1,
                ),
            )
            resumed = EoSInversionPipeline(table, two_workers).run(
                checkpoint=CheckpointConfig(checkpoint)
            )
            self.assert_columns_equal(self, first, resumed)
            run = resumed.manifest["run"]
            self.assertEqual(run["execution"]["mode"], "checkpoint_resume_only")
            self.assertEqual(run["checkpoint"]["restored_chunks"], 4)
            self.assertEqual(run["checkpoint"]["computed_chunks_this_invocation"], 0)

    def test_partial_serial_checkpoint_resumes_in_process_pool(self):
        table, config = self.make_table_and_config()
        baseline = EoSInversionPipeline(table, config).run()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            self.make_partial_checkpoint(table, config, checkpoint)
            parallel = replace(
                config,
                execution=replace(
                    config.execution,
                    workers=2,
                    worker_task_size=1,
                    max_pending_chunks=4,
                ),
            )
            resumed = EoSInversionPipeline(table, parallel).run(
                checkpoint=CheckpointConfig(checkpoint)
            )
            self.assert_columns_equal(self, baseline, resumed)
            run = resumed.manifest["run"]
            self.assertEqual(run["execution"]["mode"], "process")
            self.assertEqual(run["checkpoint"]["restored_chunks"], 1)
            self.assertEqual(run["checkpoint"]["computed_chunks_this_invocation"], 3)
            self.assertEqual(run["execution"]["submitted_worker_tasks"], 5)
            self.assertEqual(run["execution"]["maximum_concurrent_chunks"], 3)
            self.assertEqual(
                len(list((checkpoint / "chunks").glob("*.eoschunk"))), 4
            )

    def test_scientific_and_layout_mismatches_are_rejected(self):
        table, config = self.make_table_and_config()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            EoSInversionPipeline(table, config).run(
                checkpoint=CheckpointConfig(checkpoint)
            )
            changed_scale = replace(config, residual_scales={"e": 2.0})
            with self.assertRaises(PersistenceIntegrityError):
                EoSInversionPipeline(table, changed_scale).run(
                    checkpoint=CheckpointConfig(checkpoint)
                )
            changed_layout = replace(
                config,
                execution=replace(config.execution, chunk_size=3),
            )
            with self.assertRaises(PersistenceIntegrityError):
                EoSInversionPipeline(table, changed_layout).run(
                    checkpoint=CheckpointConfig(checkpoint)
                )

    def test_corrupt_completed_chunk_is_not_silently_recomputed(self):
        table, config = self.make_table_and_config()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            EoSInversionPipeline(table, config).run(
                checkpoint=CheckpointConfig(checkpoint)
            )
            chunk = sorted((checkpoint / "chunks").glob("*.eoschunk"))[0]
            with zipfile.ZipFile(chunk, mode="r") as archive:
                members = [
                    (item.filename, archive.read(item))
                    for item in archive.infolist()
                ]
            with zipfile.ZipFile(chunk, mode="w") as archive:
                changed = False
                for name, payload in members:
                    if not changed and name.startswith("arrays/"):
                        payload = payload[:-1] + bytes([payload[-1] ^ 1])
                        changed = True
                    archive.writestr(name, payload)
            with self.assertRaises(PersistenceIntegrityError):
                EoSInversionPipeline(table, config).run(
                    checkpoint=CheckpointConfig(checkpoint)
                )


if __name__ == "__main__":
    unittest.main()
