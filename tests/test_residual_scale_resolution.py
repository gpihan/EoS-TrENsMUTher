"""Automatic residual-normalization regression coverage."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    CheckpointConfig,
    EoSInversionPipeline,
    EoSRootSolver,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    MultilinearInterpolator,
    PersistenceIntegrityError,
    PipelineConfig,
    PipelinePlan,
    ProjectionConfig,
    load_application_config,
    load_pipeline_result,
    save_pipeline_result,
)
from eos_trensmuther.cli import _preprocess_document

from tests.test_tables import TemporaryEoS, field_spec, metadata


class AutomaticResidualScaleTests(unittest.TestCase):
    def make_table(self, *, zero_target: bool = False) -> ForwardTable:
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        rows = []
        for temperature in (0.2, 0.5, 0.8):
            energy = 0.0 if zero_target else 1.0 + 2.0 * temperature
            rows.append((temperature, energy))
        fixture = TemporaryEoS(metadata(fields, 1), rows)
        self.addCleanup(fixture.close)
        return ForwardTable.load_validated(
            fixture.metadata_path, _allow_unsuitable_geometry=True
        )

    @staticmethod
    def config(
        table: ForwardTable,
        *,
        residual_scales=None,
        workers: int = 1,
    ) -> PipelineConfig:
        return PipelineConfig(
            grid=InversionGridSpec(
                "target", {"e": GridAxisSpec(5)}
            ),
            residual_scales=residual_scales,
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(
                workers=workers,
                chunk_size=2,
                worker_task_size=1,
                max_pending_chunks=2,
            ),
            auxiliary_fields=(),
        )

    def test_omitted_scale_uses_validation_extrema_and_resolves_config(self):
        table = self.make_table()
        config = self.config(table)
        self.assertIsNone(config.residual_scales)
        self.assertIsNone(config.to_dict()["residual_scales"])
        self.assertIsNone(
            PipelineConfig.from_dict(config.to_dict()).residual_scales
        )

        statistics = table.report.statistics["fields"]["e"]
        expected = max(
            abs(statistics["finite_minimum"]),
            abs(statistics["finite_maximum"]),
        )
        self.assertEqual(expected, 2.6)

        plan = PipelinePlan.from_table(table, config)
        np.testing.assert_array_equal(plan.residual_scales, [expected])
        self.assertEqual(dict(plan.config.residual_scales), {"e": expected})
        self.assertEqual(
            plan.manifest()["target"]["residual_scales"], {"e": expected}
        )

    def test_explicit_scale_is_used_exactly(self):
        table = self.make_table()
        plan = PipelinePlan.from_table(
            table, self.config(table, residual_scales={"e": 0.125})
        )
        np.testing.assert_array_equal(plan.residual_scales, [0.125])
        self.assertEqual(dict(plan.config.residual_scales), {"e": 0.125})

    def test_constant_zero_component_uses_one_canonical_unit(self):
        table = self.make_table(zero_target=True)
        interpolator = MultilinearInterpolator(table, field_names=("e",))
        solver = EoSRootSolver(interpolator)

        np.testing.assert_array_equal(
            solver._residual_scales(np.asarray([0.0])), [1.0]
        )

    def test_application_config_and_preprocess_record_automatic_resolution(self):
        table = self.make_table()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parameter_file = root / "parameters.py"
            parameter_file.write_text(
                "general_parameters = %r\n"
                % {
                    "EoS_metadata": str(table.metadata_path),
                    "OutputFolder": str(root / "Output"),
                    "GridSpace": "target",
                    "GridAxes": {"e": 3},
                    "Projection": {"enabled": False},
                    "Checkpoint": False,
                    "Export": {"enabled": False},
                },
                encoding="utf-8",
            )
            application, loaded = load_application_config(
                parameter_file, operation="preprocess"
            )
            self.assertIsNone(application.pipeline.residual_scales)
            document = _preprocess_document(application, loaded)

        resolution = document["residual_scale_resolution"]
        self.assertEqual(resolution["mode"], "automatic")
        self.assertEqual(
            resolution["rule"],
            "maximum_absolute_validated_table_node_value",
        )
        self.assertEqual(resolution["zero_component_fallback"], 1.0)
        self.assertEqual(resolution["values"], {"e": 2.6})
        self.assertEqual(
            document["resolved_pipeline_configuration"]["residual_scales"],
            {"e": 2.6},
        )

    def test_automatic_keyword_is_equivalent_to_omission(self):
        table = self.make_table()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parameter_file = root / "parameters.py"
            parameter_file.write_text(
                "general_parameters = %r\n"
                % {
                    "EoS_metadata": str(table.metadata_path),
                    "OutputFolder": str(root / "Output"),
                    "GridSpace": "target",
                    "GridAxes": {"e": 3},
                    "ResidualScales": "automatic",
                    "Projection": {"enabled": False},
                    "Checkpoint": False,
                    "Export": {"enabled": False},
                },
                encoding="utf-8",
            )
            application, _ = load_application_config(
                parameter_file, operation="preprocess"
            )
        self.assertIsNone(application.pipeline.residual_scales)

    def test_serial_process_and_artifact_use_identical_resolved_scales(self):
        table = self.make_table()
        serial = EoSInversionPipeline(table, self.config(table)).run()
        process = EoSInversionPipeline(
            table, self.config(table, workers=2)
        ).run()

        np.testing.assert_array_equal(serial.plan.residual_scales, [2.6])
        np.testing.assert_array_equal(
            process.plan.residual_scales, serial.plan.residual_scales
        )
        np.testing.assert_array_equal(
            process.solve_status_codes, serial.solve_status_codes
        )
        np.testing.assert_array_equal(
            process.proof_status_codes, serial.proof_status_codes
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "result.eosinv"
            save_pipeline_result(serial, artifact)
            restored = load_pipeline_result(artifact)
        np.testing.assert_array_equal(
            restored.plan.residual_scales, serial.plan.residual_scales
        )
        self.assertEqual(
            restored.plan.config.to_dict()["residual_scales"], {"e": 2.6}
        )

    def test_checkpoint_identity_uses_resolved_values(self):
        table = self.make_table()
        automatic = self.config(table)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint"
            first = EoSInversionPipeline(table, automatic).run(
                checkpoint=CheckpointConfig(checkpoint_path)
            )
            fingerprint = first.manifest["run"]["checkpoint"]["run_fingerprint"]

            equivalent_explicit = replace(
                automatic,
                residual_scales={"e": 2.6},
                execution=replace(automatic.execution, workers=2),
            )
            resumed = EoSInversionPipeline(table, equivalent_explicit).run(
                checkpoint=CheckpointConfig(checkpoint_path)
            )
            self.assertEqual(
                resumed.manifest["run"]["checkpoint"]["run_fingerprint"],
                fingerprint,
            )
            self.assertEqual(
                resumed.manifest["run"]["checkpoint"]["restored_chunks"],
                3,
            )

            changed = replace(automatic, residual_scales={"e": 2.5})
            with self.assertRaises(PersistenceIntegrityError):
                EoSInversionPipeline(table, changed).run(
                    checkpoint=CheckpointConfig(checkpoint_path)
                )


if __name__ == "__main__":
    unittest.main()
