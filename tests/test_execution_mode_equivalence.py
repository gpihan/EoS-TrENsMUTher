"""Phase-10 numerical equivalence across local, resumed, and Slurm modes."""

import itertools
import tempfile
import unittest
from dataclasses import fields as dataclass_fields, replace
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    CheckpointConfig,
    CheckpointStore,
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InitialGuessSource,
    InversionGridSpec,
    PipelineConfig,
    PointDisposition,
    ProjectionConfig,
    SlurmArrayConfig,
    assemble_array_shards,
    create_array_plan,
    execute_array_shard,
    pipeline_status_code,
)

from tests.test_tables import TemporaryEoS, field_spec, metadata


_PHYSICAL_FLOAT_COLUMNS = (
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
)

_SCIENTIFIC_STATUS_COLUMNS = (
    "disposition_codes",
    "prescreen_codes",
    "projection_quality_codes",
    "phase_codes",
    "phase_route_codes",
    "phase_selection_reason_codes",
    "mixed_candidate",
    "exact_mixed_candidate",
    "mixed_candidate_entry_count",
    "mixed_verified_root_count",
    "phase_boundary",
    "phase_multivalued",
    "line_mixture_fraction_defined",
    "alternate_single_phase_source_defined",
    "auxiliary_defined",
    "achieved_tilde_defined",
)

_ALLOWED_EXECUTION_ROUTE_DIFFERENCES = frozenset(
    {
        "initial_guess_source_codes",
        "route_codes",
        "solve_status_codes",
        "fast_status_codes",
        "proof_status_codes",
    }
)


class ExecutionModeEquivalenceTests(unittest.TestCase):
    """Validate physical/status parity without requiring seed-route identity."""

    def make_problem(self, directory):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "P": field_spec("auxiliary", 4),
        }
        rows = [
            (temperature, chemical, temperature, chemical, temperature + chemical)
            for temperature, chemical in itertools.product(
                (1.0, 2.0, 3.0, 4.0),
                (0.0, 1.0, 2.0, 3.0),
            )
        ]
        fixture = TemporaryEoS(metadata(fields, 2), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        envelope = table.find_boundaries(space="target")
        config = PipelineConfig(
            grid=InversionGridSpec(
                "target",
                {name: GridAxisSpec(5) for name in envelope.names},
            ),
            residual_scales={"e": 1.0, "nB": 1.0},
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(workers=1, chunk_size=4),
            auxiliary_fields=("P",),
            use_reachability_seed=True,
        )
        slurm = SlurmArrayConfig(
            cpus_per_task=1,
            working_directory=Path(directory),
            log_directory=Path(directory) / "logs",
            entrypoint=Path(directory) / "EoS-TrENsMUTher.py",
        )
        return table, config, slurm

    def assert_scientific_equivalence(self, expected, actual):
        self.assertEqual(expected.grid.shape, actual.grid.shape)
        for left, right in zip(expected.grid.axes, actual.grid.axes):
            np.testing.assert_array_equal(left, right)
        for name in _PHYSICAL_FLOAT_COLUMNS:
            with self.subTest(physical_column=name):
                np.testing.assert_allclose(
                    getattr(expected, name),
                    getattr(actual, name),
                    rtol=1.0e-9,
                    atol=1.0e-10,
                    equal_nan=True,
                )
        for name in _SCIENTIFIC_STATUS_COLUMNS:
            with self.subTest(scientific_status_column=name):
                np.testing.assert_array_equal(
                    getattr(expected, name), getattr(actual, name)
                )
        self.assertEqual(dict(expected.status_counts), dict(actual.status_counts))
        self.assertEqual(
            actual.manifest["run"]["errors"], {"count": 0, "records": ()}
        )

    @staticmethod
    def differing_route_columns(first, second):
        names = _ALLOWED_EXECUTION_ROUTE_DIFFERENCES | {
            "disposition_codes", "prescreen_codes"
        }
        return {
            name
            for name in names
            if not np.array_equal(getattr(first, name), getattr(second, name))
        }

    def test_serial_process_resume_and_slurm_are_scientifically_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table, serial_config, slurm = self.make_problem(root)

            serial = EoSInversionPipeline(table, serial_config).run()
            one_process_fresh = EoSInversionPipeline(table, serial_config).run()
            process_config = replace(
                serial_config,
                execution=replace(serial_config.execution, workers=2),
            )
            two_process = EoSInversionPipeline(table, process_config).run()

            def seed_partial_checkpoint(checkpoint_path):
                partial_pipeline = EoSInversionPipeline(table, serial_config)
                reachability = partial_pipeline.reachability_config
                store = CheckpointStore(
                    CheckpointConfig(checkpoint_path),
                    partial_pipeline.resolved_plan,
                    table_fingerprint=table.snapshot_fingerprint(),
                    reachability_configuration={
                        descriptor.name: getattr(reachability, descriptor.name)
                        for descriptor in dataclass_fields(reachability)
                    },
                )
                partial_pipeline._prepare()
                chunks = tuple(partial_pipeline.resolved_plan.iter_chunks())
                for chunk in chunks[:2]:
                    store.save_chunk(partial_pipeline._execute_chunk(chunk))

            checkpoint_path = root / "partial-checkpoint"
            seed_partial_checkpoint(checkpoint_path)
            resumed = EoSInversionPipeline(table, serial_config).run(
                checkpoint=CheckpointConfig(checkpoint_path)
            )
            process_checkpoint_path = root / "partial-process-checkpoint"
            seed_partial_checkpoint(process_checkpoint_path)
            resumed_process = EoSInversionPipeline(table, process_config).run(
                checkpoint=CheckpointConfig(process_checkpoint_path)
            )

            array_plan = create_array_plan(
                table,
                serial_config,
                output_directory=root / "slurm",
                final_artifact=root / "slurm-result.eosinv",
                slurm=slurm,
            )
            for shard_index in reversed(range(array_plan.shape[0])):
                execute_array_shard(array_plan, shard_index)
            assembled = assemble_array_shards(array_plan)

            for label, result in (
                ("one_process_fresh", one_process_fresh),
                ("two_process", two_process),
                ("checkpoint_resume", resumed),
                ("process_checkpoint_resume", resumed_process),
                ("slurm_assembled", assembled),
            ):
                with self.subTest(execution_mode=label):
                    self.assert_scientific_equivalence(serial, result)

            solved_code = pipeline_status_code(PointDisposition.SOLVED)
            self.assertTrue(np.all(serial.disposition_codes == solved_code))
            self.assertTrue(np.all(assembled.disposition_codes == solved_code))

            serial_counters = serial.manifest["run"]["stage_counters"]
            slurm_counters = assembled.manifest["run"]["stage_counters"]
            self.assertEqual(
                serial_counters["seed_direct_neighbor_axis0_minus"], 9
            )
            self.assertEqual(
                slurm_counters["seed_direct_neighbor_axis0_minus"], 0
            )
            self.assertEqual(
                slurm_counters["seed_direct_neighbor_axis0_plus"], 0
            )

            shard_starts = np.arange(0, serial.grid.size, serial.grid.shape[1])
            containing_code = pipeline_status_code(
                InitialGuessSource.CONTAINING_CELL
            )
            np.testing.assert_array_equal(
                assembled.initial_guess_source_codes.reshape(-1)[shard_starts],
                np.full(shard_starts.shape, containing_code, dtype=np.uint8),
            )

            provenance_differences = self.differing_route_columns(
                serial, assembled
            )
            self.assertEqual(
                provenance_differences, {"initial_guess_source_codes"}
            )
            self.assertEqual(
                int(
                    np.count_nonzero(
                        serial.initial_guess_source_codes
                        != assembled.initial_guess_source_codes
                    )
                ),
                10,
            )
            self.assertEqual(
                int(
                    np.count_nonzero(
                        serial.initial_guess_source_codes
                        != resumed.initial_guess_source_codes
                    )
                ),
                2,
            )
            for result in (
                one_process_fresh,
                two_process,
                resumed,
                resumed_process,
                assembled,
            ):
                differences = self.differing_route_columns(serial, result)
                self.assertTrue(
                    differences <= _ALLOWED_EXECUTION_ROUTE_DIFFERENCES
                )

            resume_manifest = resumed.manifest["run"]
            self.assertEqual(resume_manifest["checkpoint"]["restored_chunks"], 2)
            self.assertGreater(
                resume_manifest["checkpoint"]["computed_chunks_this_invocation"],
                0,
            )
            self.assertEqual(
                two_process.manifest["run"]["execution"]["mode"], "process"
            )
            self.assertEqual(
                resumed_process.manifest["run"]["execution"]["mode"],
                "process",
            )
            self.assertEqual(
                resumed_process.manifest["run"]["checkpoint"][
                    "restored_chunks"
                ],
                2,
            )
            self.assertEqual(
                assembled.manifest["run"]["execution"]["mode"],
                "slurm_array_assembled",
            )


if __name__ == "__main__":
    unittest.main()
