"""End-to-end tests for the Python-dictionary local user interface."""

import contextlib
import io
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eos_trensmuther import EoSValidationError, ForwardTable
from eos_trensmuther.cli import _format_user_error, _slurm_launcher, build_parser, main
from eos_trensmuther.outputs import load_pipeline_result

from tests.test_tables import TemporaryEoS, field_spec, metadata


class LocalInterfaceTests(unittest.TestCase):
    def make_fixture(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "P": field_spec("auxiliary", 4),
            "s": field_spec("auxiliary", 5),
            "cs2": field_spec("auxiliary", 6),
            "chi2B": field_spec("auxiliary", 7),
        }
        rows = []
        for temperature, chemical_potential in itertools.product(
            (0.2, 0.5, 0.8), (0.0, 0.5, 1.0)
        ):
            rows.append(
                (
                    temperature,
                    chemical_potential,
                    temperature,
                    chemical_potential,
                    temperature + chemical_potential,
                    2.0 * temperature + chemical_potential,
                    0.2,
                    0.5 * chemical_potential,
                )
            )
        fixture = TemporaryEoS(metadata(fields, 2), rows)
        self.addCleanup(fixture.close)
        return fixture

    def test_default_command_runs_preprocess_invert_and_text_export(self):
        fixture = self.make_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            parameter_file = root / "parameters.py"
            values = {
                "EoS_metadata": str(fixture.metadata_path),
                "OutputFolder": str(output),
                "EoSName": "TestEoS",
                "Overwrite": "no",
                "GridAxes": {"e": 2, "nB": 2},
                "ResidualScales": {"e": 1.0, "nB": 1.0},
                "AuxiliaryFields": ["P", "s", "cs2", "chi2B"],
                "ProjectionEnabled": False,
                "UseReachabilitySeed": False,
                "Execution": {"workers": 1, "chunk_size": 2},
                "Checkpoint": {"enabled": True, "directory": "checkpoint", "resume": True},
                "Export": {"include_unusable": True, "write_header": True},
            }
            parameter_file.write_text(
                "general_parameters = %r\n" % values, encoding="utf-8"
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main([str(parameter_file)])
            self.assertEqual(status, 0, stderr.getvalue())
            run = output / "TestEoS" / "TestEoS"
            diagnostics = run / "diagnostic"
            self.assertTrue((diagnostics / "preprocess.json").is_file())
            self.assertTrue((run / "inverse.eosinv").is_file())
            self.assertTrue((run / "eos_table.dat").is_file())
            self.assertTrue((diagnostics / "text_export_manifest.json").is_file())
            manifest = json.loads(
                (diagnostics / "text_export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["include_unusable"])
            reloaded = load_pipeline_result(run / "inverse.eosinv")
            self.assertEqual(reloaded.plan.source_units, ("GeV", "GeV"))
            self.assertFalse(reloaded.plan.config.use_reachability_seed)
            exported_values = np.loadtxt(run / "eos_table.dat")
            self.assertEqual(exported_values.shape, (4, len(manifest["columns"])))
            exported_names = tuple(column["name"] for column in manifest["columns"])
            self.assertNotIn("solved", exported_names)
            self.assertNotIn("unresolved", exported_names)
            self.assertNotIn("error", exported_names)
            output_text = stdout.getvalue()
            self.assertIn("EoS-TrENsMUTher v", output_text)
            self.assertNotIn("Thanks for using EoS-TrENsMUTher!", output_text)
            self.assertIn("########################################\n\n[1/6]", output_text)
            self.assertIn("[1/6] Reading configuration and EoS table", output_text)
            self.assertIn("coordinate axes: OK", output_text)
            self.assertIn("Solved: 4", output_text)
            self.assertIn("Jacobian-selected: 0", output_text)
            self.assertIn("Total execution time:", output_text)

            repeat_stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                repeat_stderr
            ):
                repeated = main([str(parameter_file)])
            self.assertEqual(repeated, 2)
            self.assertIn("--spec '_test2mub500'", repeat_stderr.getvalue())

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                overwritten = main([str(parameter_file), "--overwrite"])
            self.assertEqual(overwritten, 0)
            self.assertTrue(
                (diagnostics / "checkpoint" / "checkpoint.json").is_file()
            )

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                specified = main([str(parameter_file), "--spec", "_test"])
            self.assertEqual(specified, 0)
            self.assertTrue(
                (output / "TestEoS" / "TestEoS_test" / "inverse.eosinv").is_file()
            )

    def test_default_overwrite_policy_numbers_run_directories(self):
        fixture = self.make_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "Output"
            parameter_file = root / "parameters.py"
            parameters = {
                "EoS_metadata": str(fixture.metadata_path),
                "OutputFolder": str(output),
                "EoSName": "Numbered",
                "Overwrite": "default",
                "GridSpace": "target",
                "GridAxes": {"e": 2, "nB": 2},
                "ResidualScales": {"e": 1.0, "nB": 1.0},
                "ProjectionEnabled": False,
                "Execution": {"workers": 1, "chunk_size": 2},
                "Checkpoint": False,
            }
            parameter_file.write_text(
                "general_parameters = %r\n" % parameters, encoding="utf-8"
            )
            for expected in ("Numbered_1", "Numbered_2"):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    status = main([str(parameter_file)])
                self.assertEqual(status, 0)
                self.assertTrue(
                    (output / "Numbered" / expected / "inverse.eosinv").is_file()
                )

    def test_public_cli_has_only_generic_text_export(self):
        help_text = build_parser().format_help()
        self.assertIn("export-text", help_text)
        self.assertNotIn("MUSIC", help_text)
        self.assertNotIn("vHLLE", help_text)

    def test_orientation_reversal_has_clear_scientific_error(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }
        rows = [
            [temperature, mu_b, temperature, (temperature - 0.25) * mu_b]
            for temperature, mu_b in itertools.product(
                (0.1, 0.2, 0.3, 0.4), (0.0, 0.5, 1.0)
            )
        ]
        fixture = TemporaryEoS(metadata(fields, 2), rows)
        self.addCleanup(fixture.close)

        with self.assertRaises(EoSValidationError) as caught:
            ForwardTable.load_validated(fixture.metadata_path)
        message = _format_user_error(caught.exception)

        self.assertNotIn("Table validation failed", message)
        self.assertIn("not sufficiently well behaved", message)
        self.assertIn("\n\nThe inversion cannot be performed.\n\n", message)
        self.assertIn("Local orientation is the sign", message)
        self.assertIn("Cells inspected: 6", message)
        self.assertIn("Regular positive: 2", message)
        self.assertIn("Regular negative: 2", message)
        self.assertIn("Sign-changing or singular: 2", message)
        self.assertIn("declare it in the metadata", message)

    def test_staged_commands_share_one_explicit_specification(self):
        fixture = self.make_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "Output"
            parameter_file = root / "parameters_staged.py"
            parameters = {
                "EoS_metadata": str(fixture.metadata_path),
                "OutputFolder": str(output),
                "EoSName": "Staged",
                "Overwrite": "no",
                "GridSpace": "target",
                "GridAxes": {"e": 2, "nB": 2},
                "ResidualScales": {"e": 1.0, "nB": 1.0},
                "ProjectionEnabled": False,
                "Execution": {"workers": 1, "chunk_size": 2},
                "Checkpoint": False,
            }
            parameter_file.write_text(
                "general_parameters = %r\n" % parameters, encoding="utf-8"
            )
            for command in ("preprocess", "invert", "postprocess"):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    status = main([command, str(parameter_file), "--spec", "_stage"])
                self.assertEqual(status, 0, command)
            run = output / "Staged" / "Staged_stage"
            self.assertTrue((run / "diagnostic" / "preprocess.json").is_file())
            self.assertTrue((run / "inverse.eosinv").is_file())
            self.assertTrue((run / "eos_table.dat").is_file())

    def test_installed_console_is_the_default_slurm_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "eos-trensmuther"
            launcher.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            with patch.object(sys, "argv", [str(launcher)]):
                working_directory, entrypoint = _slurm_launcher()

        self.assertEqual(working_directory, Path.cwd().resolve())
        self.assertEqual(entrypoint, launcher.resolve())


if __name__ == "__main__":
    unittest.main()
