"""Frozen v1 public-interface and supported-workflow regressions."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from dataclasses import fields as dataclass_fields, replace
from pathlib import Path

import numpy as np

import eos_trensmuther
from eos_trensmuther import (
    COMPATIBILITY_API_NAMES,
    ORDINARY_SUBCELL_CONFIG_FIELDS,
    PARAMETER_GROUPS,
    PUBLIC_API_GROUPS,
    PUBLIC_API_NAMES,
    PUBLIC_API_VERSION,
    V1_CONTRACT_VERSIONS,
    CheckpointConfig,
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    PipelineConfig,
    PointDisposition,
    ProjectionConfig,
    SolverConfig,
    TextExportConfig,
    __version__,
    export_text,
    load_application_config,
    load_pipeline_result,
    load_parameter_file,
    pipeline_status_code,
    save_pipeline_result,
)
from eos_trensmuther.cli import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_METADATA = REPOSITORY_ROOT / "examples/regular/metadata.json"


EXPECTED_PUBLIC_GROUPS = {
    "contract": (
        "__version__",
        "PUBLIC_API_VERSION",
        "PUBLIC_API_GROUPS",
        "PUBLIC_API_NAMES",
        "V1_CONTRACT_VERSIONS",
    ),
    "tables": (
        "ForwardTable",
        "BoundaryEnvelope",
        "ValidationIssue",
        "ValidationReport",
        "EoSError",
        "EoSValidationError",
    ),
    "configuration": (
        "ApplicationConfig",
        "ApplicationConfigError",
        "GridAxisSpec",
        "InversionGridSpec",
        "PipelineConfig",
        "ExecutionConfig",
        "ProjectionConfig",
        "SolverConfig",
        "ReachabilityConfig",
        "CheckpointConfig",
        "CoarseSeedConfig",
        "ErrorPolicy",
        "ProcessStartMethod",
        "PARAMETER_GROUPS",
        "ORDINARY_SUBCELL_CONFIG_FIELDS",
        "load_application_config",
        "load_parameter_file",
    ),
    "pipeline": (
        "EoSInversionPipeline",
        "PipelineResult",
        "PipelineError",
        "PipelineConfigurationError",
    ),
    "status": (
        "PointDisposition",
        "PointRoute",
        "PrescreenStatus",
        "SolveStatus",
        "ProofStatus",
        "ProjectionQuality",
        "PhaseLabel",
        "PhaseRoute",
        "PhaseRootSelectionReason",
        "InitialGuessSource",
        "pipeline_status_code",
    ),
    "persistence": (
        "save_pipeline_result",
        "load_pipeline_result",
        "PersistenceError",
    ),
    "export": (
        "TextExportConfig",
        "TextExportResult",
        "ExportError",
        "export_text",
    ),
    "advanced_inversion": (
        "MultilinearInterpolator",
        "EoSRootSolver",
        "InversionResult",
    ),
    "slurm": (
        "SlurmArrayConfig",
        "ExecutorError",
        "create_array_plan",
        "save_array_plan",
        "load_array_plan",
        "render_slurm_script",
        "execute_array_shard",
        "assemble_array_shards",
    ),
}

EXPECTED_PARAMETER_GROUPS = {
    "user": (
        "AuxiliaryFields", "Checkpoint", "Dimension", "EoSName",
        "EoS_metadata", "EoS_table", "Execution", "Export", "ExtraInfo", "GridAxes",
        "OutputFolder", "Overwrite", "Projection",
        "RunSpecification",
        "Variables",
    ),
    "advanced": (
        "ArtifactCompression", "CoarseSeed", "InitialGuess",
        "LineAuxiliaryDefaultPolicy", "LineAuxiliaryPolicies", "MaxGridPoints",
        "MaxPeakResultBytes", "MaxResultBytes", "Reachability", "ResidualScales", "Solver",
        "UseReachabilitySeed",
    ),
    "diagnostic": (
        "PreprocessReport", "ResultFile",
    ),
    "compatibility": (
        "Accuracy", "AutoSetBoundaries", "CheckpointEnabled",
        "CheckpointFolder", "ChunkSize", "DataFile", "ErrorPolicy",
        "GridSpace", "MAXITER", "MaxPendingChunks", "MetadataFile", "NB", "NQ", "NS",
        "NT", "NTildeT", "NTildemuB", "Number_of_cores",
        "OutputMergedEoS", "ProcessStartMethod", "ProjectionEnabled",
        "RegionS", "Resume", "RunMode", "TransitionLine", "Ttilde",
        "WorkerTaskSize", "e", "muBtilde", "muQtilde", "muStilde", "nB",
        "nQ", "nS",
    ),
}


class PublicContractTests(unittest.TestCase):
    def test_version_and_public_names_are_frozen(self) -> None:
        self.assertEqual(__version__, "1.0.0.dev0")
        self.assertEqual(PUBLIC_API_VERSION, 7)
        self.assertEqual(dict(PUBLIC_API_GROUPS), EXPECTED_PUBLIC_GROUPS)
        expected_names = tuple(
            name for group in EXPECTED_PUBLIC_GROUPS.values() for name in group
        )
        self.assertEqual(PUBLIC_API_NAMES, expected_names)
        self.assertEqual(len(PUBLIC_API_NAMES), len(set(PUBLIC_API_NAMES)))
        for name in PUBLIC_API_NAMES:
            with self.subTest(public_name=name):
                self.assertTrue(hasattr(eos_trensmuther, name))
                self.assertIn(name, eos_trensmuther.__all__)

    def test_broad_v1_compatibility_exports_remain_available(self) -> None:
        self.assertFalse(set(PUBLIC_API_NAMES) & set(COMPATIBILITY_API_NAMES))
        for name in ("CellImageIndex", "PipelinePlan", "TransitionLine"):
            self.assertIn(name, COMPATIBILITY_API_NAMES)
            self.assertTrue(hasattr(eos_trensmuther, name))
            self.assertIn(name, eos_trensmuther.__all__)

    def test_contract_versions_are_frozen_and_read_only(self) -> None:
        self.assertEqual(
            dict(V1_CONTRACT_VERSIONS),
            {
                "public_api": 7,
                "metadata_schema": 1,
                "table_snapshot_fingerprint": 2,
                "pipeline_result_schema": 4,
                "pipeline_run_manifest": 11,
                "numerical_routing": 3,
                "checkpoint_schema": 4,
                "artifact_schema": 3,
                "reachability_routing": 1,
                "shared_worker_state_schema": 1,
                "array_plan_schema": 3,
                "array_shard_schema": 3,
                "text_export_schema": 6,
                "phase_export_schema": 1,
            },
        )
        with self.assertRaises(TypeError):
            V1_CONTRACT_VERSIONS["numerical_routing"] = 99

    def test_parameter_groups_and_subcell_surface_are_exact(self) -> None:
        self.assertEqual(dict(PARAMETER_GROUPS), EXPECTED_PARAMETER_GROUPS)
        grouped = [set(values) for values in PARAMETER_GROUPS.values()]
        for index, values in enumerate(grouped):
            for other in grouped[index + 1 :]:
                self.assertFalse(values & other)
        self.assertEqual(sum(map(len, grouped)), 63)
        self.assertEqual(
            ORDINARY_SUBCELL_CONFIG_FIELDS,
            (
                "enable_subcell_fallback",
                "max_subcell_depth",
                "max_subcell_active_boxes",
            ),
        )
        self.assertEqual(
            tuple(
                field.name
                for field in dataclass_fields(SolverConfig)
                if "subcell" in field.name
            ),
            ORDINARY_SUBCELL_CONFIG_FIELDS,
        )


class PublicWorkflowTests(unittest.TestCase):
    @staticmethod
    def _config(workers: int) -> PipelineConfig:
        return PipelineConfig(
            grid=InversionGridSpec(
                "target", {"e": GridAxisSpec(3), "nB": GridAxisSpec(3)}
            ),
            residual_scales={"e": 0.02, "nB": 0.02},
            projection=ProjectionConfig(
                enabled=True,
                nearest_cell_starts=2,
                max_evaluations_per_start=60,
            ),
            execution=ExecutionConfig(
                workers=workers,
                start_method="spawn",
                chunk_size=3,
                worker_task_size=2,
                max_pending_chunks=4,
            ),
            auxiliary_fields=("P", "s"),
        )

    def test_supported_serial_process_checkpoint_artifact_and_export_path(self) -> None:
        table = ForwardTable.load_validated(SYNTHETIC_METADATA)
        serial = EoSInversionPipeline(table, self._config(1)).run()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = CheckpointConfig(root / "checkpoint")
            process = EoSInversionPipeline(table, self._config(2)).run(
                checkpoint=checkpoint
            )
            np.testing.assert_array_equal(
                serial.disposition_codes, process.disposition_codes
            )
            np.testing.assert_allclose(
                serial.source_coordinates,
                process.source_coordinates,
                rtol=1.0e-9,
                atol=1.0e-10,
                equal_nan=True,
            )

            resumed_config = replace(
                self._config(2),
                execution=replace(
                    self._config(2).execution,
                    workers=1,
                    worker_task_size=3,
                ),
            )
            resumed = EoSInversionPipeline(table, resumed_config).run(
                checkpoint=checkpoint
            )
            self.assertEqual(
                resumed.manifest["run"]["checkpoint"]["restored_chunks"], 3
            )
            np.testing.assert_array_equal(
                process.disposition_codes, resumed.disposition_codes
            )

            artifact = save_pipeline_result(process, root / "inverse.eosinv")
            loaded = load_pipeline_result(artifact)
            np.testing.assert_array_equal(
                process.disposition_codes, loaded.disposition_codes
            )
            np.testing.assert_allclose(
                process.source_coordinates,
                loaded.source_coordinates,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )

            exported = export_text(
                loaded,
                root / "eos_table.dat",
                config=TextExportConfig(include_projected=True),
            )
            self.assertTrue(exported.table_path.is_file())

        self.assertGreater(int(np.count_nonzero(process.solution_mask)), 0)
        self.assertGreater(int(np.count_nonzero(process.projected_mask)), 0)
        self.assertFalse(np.any(process.unknown_mask))
        self.assertFalse(np.any(process.error_mask))
        self.assertEqual(sum(process.status_counts.values()), process.grid.size)
        solved_code = pipeline_status_code(PointDisposition.SOLVED)
        self.assertTrue(
            np.array_equal(process.solution_mask, process.disposition_codes == solved_code)
        )
        self.assertIn("proof_status", process.codebooks)


class PublicErrorTests(unittest.TestCase):
    def test_parameter_typo_suggests_the_supported_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parameter_file = Path(directory) / "parameters.py"
            parameter_file.write_text(
                "general_parameters = {'GridSpce': 'target'}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                r"(?s)Unknown parameter:\s+GridSpce.*Did you mean:\s+GridSpace",
            ):
                load_parameter_file(parameter_file)

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main([str(parameter_file)])
            self.assertEqual(status, 2)
            self.assertIn("eos-trensmuther: error:", stderr.getvalue())
            self.assertIn("Did you mean:\n  GridSpace", stderr.getvalue())

    def test_nested_solver_typo_lists_only_supported_subcell_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parameter_file = root / "parameters.py"
            values = {
                "EoS_metadata": str(SYNTHETIC_METADATA),
                "OutputFolder": str(root / "output"),
                "EoSName": "Typo",
                "Overwrite": "no",
                "GridSpace": "target",
                "GridAxes": {"e": 2, "nB": 2},
                "ResidualScales": {"e": 0.02, "nB": 0.02},
                "Solver": {"max_subcell_dept": 3},
                "Projection": {"enabled": False},
                "Execution": {"workers": 1},
                "Checkpoint": False,
            }
            parameter_file.write_text(
                "general_parameters = %r\n" % values,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                r"(?s)Unknown Solver:\s+max_subcell_dept.*Did you mean:\s+max_subcell_depth.*"
                r"enable_subcell_fallback, max_subcell_depth, "
                r"max_subcell_active_boxes",
            ):
                load_application_config(parameter_file)

    def test_regular_is_default_and_tilde_is_one_explicit_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def load(name, additions):
                parameter_file = root / (name + ".py")
                values = {
                    "EoS_metadata": str(SYNTHETIC_METADATA),
                    "OutputFolder": str(root / (name + "-output")),
                    "EoSName": name,
                    "Overwrite": "no",
                    "GridAxes": 2,
                    "Projection": {"enabled": False},
                    "Execution": {"workers": 1},
                    "Checkpoint": False,
                    "Export": {"enabled": False},
                }
                values.update(additions)
                parameter_file.write_text(
                    "general_parameters = %r\n" % values, encoding="utf-8"
                )
                return load_application_config(parameter_file)[0]

            regular = load("Regular", {})
            transformed = load("Tilde", {"Variables": "tilde"})
            compatibility = load("Compatibility", {"GridSpace": "target"})

            self.assertEqual(regular.pipeline.grid.space.value, "target")
            self.assertEqual(transformed.pipeline.grid.space.value, "tilde")
            self.assertEqual(compatibility.pipeline.grid.space.value, "target")
            self.assertTrue(regular.text_export.extra_info)

    def test_projected_export_legacy_switch_is_no_longer_a_public_choice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parameter_file = root / "parameters.py"
            values = {
                "EoS_metadata": str(SYNTHETIC_METADATA),
                "OutputFolder": str(root / "output"),
                "EoSName": "Projected",
                "GridAxes": 2,
                "Export": {"include_projected": False},
            }
            parameter_file.write_text(
                "general_parameters = %r\n" % values, encoding="utf-8"
            )
            application, _ = load_application_config(parameter_file)
            self.assertTrue(application.text_export.include_projected)

    def test_invalid_worker_task_size_is_precise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parameter_file = root / "parameters.py"
            values = {
                "EoS_metadata": str(SYNTHETIC_METADATA),
                "OutputFolder": str(root / "output"),
                "EoSName": "BadWorkerSize",
                "GridAxes": 2,
                "WorkerTaskSize": -4,
            }
            parameter_file.write_text(
                "general_parameters = %r\n" % values, encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError,
                r"Invalid value for WorkerTaskSize:\s+received: -4\s+expected: positive integer",
            ):
                load_application_config(parameter_file)

    def test_metadata_json_and_missing_file_errors_identify_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "metadata.json"
            malformed.write_text('{"name": "broken"\n', encoding="utf-8")

            def parameter(metadata_path, name):
                path = root / (name + ".py")
                path.write_text(
                    "general_parameters = %r\n"
                    % {
                        "EoS_metadata": str(metadata_path),
                        "OutputFolder": str(root / "output"),
                        "EoSName": name,
                        "GridAxes": 2,
                    },
                    encoding="utf-8",
                )
                return path

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main([str(parameter(malformed, "Malformed"))])
            self.assertEqual(status, 2)
            error = stderr.getvalue()
            self.assertIn("Metadata JSON syntax error:", error)
            self.assertIn("file: %s" % malformed.resolve(), error)
            self.assertIn("line:", error)
            self.assertIn("column:", error)

            missing = root / "absent.json"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main([str(parameter(missing, "Missing"))])
            self.assertEqual(status, 2)
            self.assertIn(
                "Metadata file not found:\n  %s" % missing,
                # ``_input_path`` reports the user-facing spelling before a
                # missing path can be canonicalized.
                stderr.getvalue(),
            )

    def test_cli_reports_the_single_source_version(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), "eos-trensmuther " + __version__)


if __name__ == "__main__":
    unittest.main()
