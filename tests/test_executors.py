"""First-axis array planning, restart, shard, and assembly tests."""

import itertools
import json
import shutil
import tempfile
import unittest
import zipfile
from dataclasses import fields as dataclass_fields, replace
from pathlib import Path

import numpy as np
import eos_trensmuther.executors as executors_module

from eos_trensmuther import (
    ARRAY_PLAN_SCHEMA_VERSION,
    ArrayPlanError,
    CheckpointConfig,
    CheckpointStore,
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    PipelineConfig,
    PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
    PersistenceFormatError,
    PersistenceIntegrityError,
    ProjectionConfig,
    ReachabilityConfig,
    ShardError,
    SHARD_SCHEMA_VERSION,
    SlurmArrayConfig,
    assemble_array_shards,
    create_array_plan,
    execute_array_shard,
    load_array_plan,
    render_slurm_script,
    save_array_plan,
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


def _assert_scientific_result_equal(first, second):
    """Compare physical results while allowing opportunistic seed provenance."""

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


def _assert_scientific_diagnostics_equal(first, second):
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
        if first_value != second_value:
            raise AssertionError(
                "scientific run diagnostic differs at %s" % name
            )


class FirstAxisExecutorTests(unittest.TestCase):
    def make_problem(self, directory, *, workers=1, worker_task_size=256):
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
            rows.append(
                (
                    temperature,
                    chemical_potential,
                    temperature,
                    temperature + chemical_potential,
                    3.0 * temperature + 2.0 * chemical_potential,
                )
            )
        fixture = TemporaryEoS(metadata(fields, 2), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        envelope = table.find_boundaries(space="target")
        config = PipelineConfig(
            grid=InversionGridSpec(
                "target",
                {
                    envelope.names[0]: GridAxisSpec(3),
                    envelope.names[1]: GridAxisSpec(4),
                },
            ),
            residual_scales={"e": 1.0, "nB": 1.0},
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(
                workers=workers,
                chunk_size=2,
                worker_task_size=worker_task_size,
            ),
            auxiliary_fields=("P",),
        )
        root = Path(directory)
        scheduler = SlurmArrayConfig(
            cpus_per_task=workers,
            account="science",
            partition="cpu",
            wall_time="00:20:00",
            memory="2G",
            working_directory=root,
            log_directory=root / "logs with spaces",
            activation_commands=("module load python", "source .venv/bin/activate"),
            python_executable="python3",
            entrypoint=root / "EoS-TrENsMUTher.py",
        )
        plan = create_array_plan(
            table,
            config,
            output_directory=root / "diagnostic" / "slurm",
            final_artifact=root / "inverse.eosinv",
            slurm=scheduler,
        )
        return table, config, plan

    def make_four_dimensional_plan(self, directory):
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
        axes = ((0.2, 0.5), (0.0, 0.3), (-0.1, 0.1), (-0.2, 0.2))
        rows = [list(point) + list(point) for point in itertools.product(*axes)]
        fixture = TemporaryEoS(metadata(fields, 4), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        envelope = table.find_boundaries(space="target")
        config = PipelineConfig(
            grid=InversionGridSpec(
                "target",
                {name: GridAxisSpec(2) for name in envelope.names},
            ),
            residual_scales={name: 1.0 for name in table.target_names},
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(workers=1, chunk_size=2),
            use_reachability_seed=False,
            auxiliary_fields=(),
        )
        reachability = ReachabilityConfig(
            method="hierarchical", hierarchy_levels=7
        )
        return create_array_plan(
            table,
            config,
            reachability_configuration=reachability,
            output_directory=Path(directory) / "slurm",
            final_artifact=Path(directory) / "inverse.eosinv",
        )

    def test_hierarchical_method_and_depth_round_trip_through_array_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_four_dimensional_plan(directory)
            path = save_array_plan(plan, Path(directory) / "array_plan.json")
            restored = load_array_plan(path)

        self.assertEqual(
            restored.reachability_configuration["method"], "hierarchical"
        )
        self.assertEqual(
            restored.reachability_configuration["hierarchy_levels"], 7
        )
        self.assertFalse(
            restored.pipeline_configuration["use_reachability_seed"]
        )
        self.assertEqual(ARRAY_PLAN_SCHEMA_VERSION, 3)
        self.assertEqual(SHARD_SCHEMA_VERSION, 3)
        self.assertEqual(
            restored.numerical_routing_contract["version"],
            PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
        )
        self.assertEqual(restored.fingerprint, plan.fingerprint)

    def test_older_array_plan_and_shard_contracts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.make_problem(directory)
            plan_path = save_array_plan(
                plan, Path(directory) / "array_plan.json"
            )
            document = json.loads(plan_path.read_text(encoding="utf-8"))
            document["schema_version"] = 1
            plan_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                ArrayPlanError, "older numerical routing contract"
            ):
                load_array_plan(plan_path)

        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.make_problem(directory)
            plan_path = save_array_plan(
                plan, Path(directory) / "array_plan.json"
            )
            document = json.loads(plan_path.read_text(encoding="utf-8"))
            document["numerical_routing_contract"]["version"] = (
                PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION - 1
            )
            document["pipeline_configuration"][
                "numerical_routing_contract_version"
            ] = PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION - 1
            without_integrity = dict(document)
            without_integrity.pop("integrity")
            document["integrity"]["digest"] = executors_module._digest(
                "FirstAxisArrayPlanDocument:v2", without_integrity
            )
            plan_path.write_bytes(
                executors_module._canonical_json(document)
            )
            with self.assertRaisesRegex(
                ArrayPlanError, "older numerical routing contract"
            ):
                load_array_plan(plan_path)

        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.make_problem(directory)
            shard_path = execute_array_shard(plan, 0)
            with zipfile.ZipFile(shard_path, mode="r") as archive:
                members = [
                    (item.filename, archive.read(item))
                    for item in archive.infolist()
                ]
            with zipfile.ZipFile(shard_path, mode="w") as archive:
                for name, payload in members:
                    if name == "manifest.json":
                        document = json.loads(payload.decode("utf-8"))
                        document["schema_version"] = 1
                        payload = json.dumps(document).encode("utf-8")
                    archive.writestr(name, payload)
            with self.assertRaisesRegex(
                ShardError, "older numerical routing contract"
            ):
                execute_array_shard(plan, 0)

        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.make_problem(directory)
            shard_path = execute_array_shard(plan, 0)
            with zipfile.ZipFile(shard_path, mode="r") as archive:
                members = [
                    (item.filename, archive.read(item))
                    for item in archive.infolist()
                ]
            with zipfile.ZipFile(shard_path, mode="w") as archive:
                for name, payload in members:
                    if name == "manifest.json":
                        document = json.loads(payload.decode("utf-8"))
                        document["numerical_routing_contract"]["version"] = (
                            PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION - 1
                        )
                        without_integrity = dict(document)
                        without_integrity.pop("integrity")
                        document["integrity"]["digest"] = (
                            executors_module._shard_digest(
                                without_integrity
                            )
                        )
                        payload = executors_module._canonical_json(document)
                    archive.writestr(name, payload)
            with self.assertRaisesRegex(
                ShardError, "older numerical routing contract"
            ):
                execute_array_shard(plan, 0)

    def test_plan_owns_exact_first_axis_slices_and_script_does_not_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.make_problem(directory)
            plan_path = save_array_plan(plan, Path(directory) / "array_plan.json")
            restored = load_array_plan(plan_path)
            self.assertEqual(restored.axis_names, ("e", "nB"))
            self.assertEqual(restored.shape, (3, 4))
            self.assertEqual(
                [(item.start, item.stop) for item in restored.shards],
                [(0, 4), (4, 8), (8, 12)],
            )
            self.assertEqual(
                sum(item.size for item in restored.shards),
                int(np.prod(restored.shape)),
            )
            script = render_slurm_script(
                plan_path, Path(directory) / "submit.sbatch"
            )
            text = script.read_text(encoding="utf-8")
            self.assertIn("#SBATCH --array=0-2", text)
            self.assertIn("#SBATCH --cpus-per-task=1", text)
            self.assertIn("#SBATCH --account=science", text)
            self.assertIn("#SBATCH --partition=cpu", text)
            self.assertIn("slurm-task", text)
            self.assertNotIn("\nsbatch ", text)

            table, _, _ = self.make_problem(Path(directory) / "tilde-problem")
            tilde_envelope = table.find_boundaries(space="tilde")
            tilde_config = PipelineConfig(
                grid=InversionGridSpec(
                    "tilde",
                    {
                        name: GridAxisSpec(2)
                        for name in tilde_envelope.names
                    },
                ),
                residual_scales={"e": 1.0, "nB": 1.0},
                projection=ProjectionConfig(enabled=False),
                execution=ExecutionConfig(workers=1, chunk_size=2),
                auxiliary_fields=("P",),
            )
            tilde_plan = create_array_plan(
                table,
                tilde_config,
                output_directory=Path(directory) / "tilde-slurm",
                final_artifact=Path(directory) / "tilde.eosinv",
            )
            self.assertEqual(tilde_plan.axis_names[0], "T_tilde")

            document = json.loads(plan_path.read_text(encoding="utf-8"))
            document["grid"]["shape"][0] = 4
            tampered = Path(directory) / "tampered.json"
            tampered.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ArrayPlanError):
                load_array_plan(tampered)

    def test_serial_shards_restart_and_assemble_exact_local_result(self):
        with tempfile.TemporaryDirectory() as directory:
            table, config, plan = self.make_problem(directory)
            local = EoSInversionPipeline(table, config).run()

            # Seed exactly one owned checkpoint chunk, simulating interruption.
            pipeline = EoSInversionPipeline(table, config)
            first_shard = plan.shard(0)
            store = CheckpointStore(
                CheckpointConfig(plan.shard_checkpoint(0)),
                pipeline.resolved_plan,
                table_fingerprint=table.snapshot_fingerprint(),
                reachability_configuration={
                    item.name: getattr(pipeline.reachability_config, item.name)
                    for item in dataclass_fields(pipeline.reachability_config)
                },
                owned_range=(first_shard.start, first_shard.stop),
            )
            pipeline._prepare()
            first_range = store.ranges[0]
            store.save_chunk(
                pipeline._execute_chunk(pipeline.resolved_plan.grid.chunk(*first_range))
            )
            seeded_path = sorted(
                (plan.shard_checkpoint(0) / "chunks").glob("*.eoschunk")
            )[0]
            seeded_bytes = seeded_path.read_bytes()

            # Arrival order is deliberately different from grid order.
            for index in (2, 0, 1):
                execute_array_shard(plan, index)
            self.assertEqual(seeded_path.read_bytes(), seeded_bytes)
            first_path = plan.shard_path(0)
            with zipfile.ZipFile(first_path, mode="r") as archive:
                first_manifest = json.loads(
                    archive.read("manifest.json").decode("utf-8")
                )
            self.assertEqual(
                first_manifest["execution"]["restored_chunks"], 1
            )
            self.assertEqual(
                first_manifest["execution"][
                    "computed_chunks_this_invocation"
                ],
                1,
            )
            first_mtime = first_path.stat().st_mtime_ns
            self.assertEqual(execute_array_shard(plan, 0), first_path)
            self.assertEqual(first_path.stat().st_mtime_ns, first_mtime)

            assembled = assemble_array_shards(plan)
            _assert_scientific_result_equal(assembled, local)
            _assert_scientific_diagnostics_equal(assembled, local)
            self.assertEqual(dict(assembled.status_counts), dict(local.status_counts))
            self.assertEqual(
                assembled.manifest["run"]["execution"]["mode"],
                "slurm_array_assembled",
            )
            self.assertTrue(plan.final_artifact.is_file())

    def test_assembly_rejects_missing_and_unexpected_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.make_problem(directory)
            execute_array_shard(plan, 0)
            with self.assertRaises(ShardError):
                assemble_array_shards(plan)
            for index in (1, 2):
                execute_array_shard(plan, index)
            extra = plan.shard_directory / "shard_999999.eosshard"
            extra.write_bytes(b"unexpected")
            with self.assertRaises(ShardError):
                assemble_array_shards(plan)

    def test_process_workers_inside_each_array_task_match_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            table, process_config, plan = self.make_problem(
                directory, workers=2, worker_task_size=1
            )
            serial_config = replace(
                process_config,
                execution=replace(process_config.execution, workers=1),
            )
            expected = EoSInversionPipeline(table, serial_config).run()
            for index in reversed(range(plan.shape[0])):
                execute_array_shard(plan, index)
                shared_manifest = plan.checkpoint_directory / (
                    "shard_%06d_shared_worker_state_v1/shared_state.json"
                    % index
                )
                self.assertTrue(shared_manifest.is_file())
            assembled = assemble_array_shards(plan)
            _assert_scientific_result_equal(assembled, expected)
            _assert_scientific_diagnostics_equal(assembled, expected)

    def test_assembly_rejects_corrupt_and_wrong_index_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.make_problem(directory)
            for index in range(plan.shape[0]):
                execute_array_shard(plan, index)
            wrong = plan.shard_path(1)
            wrong.unlink()
            shutil.copyfile(plan.shard_path(0), wrong)
            with self.assertRaises(PersistenceIntegrityError):
                assemble_array_shards(plan)

        with tempfile.TemporaryDirectory() as directory:
            _, _, plan = self.make_problem(directory)
            for index in range(plan.shape[0]):
                execute_array_shard(plan, index)
            corrupt = plan.shard_path(1)
            corrupt.write_bytes(b"not a zip artifact")
            with self.assertRaises(PersistenceFormatError):
                assemble_array_shards(plan)


if __name__ == "__main__":
    unittest.main()
