"""Smoke tests for the bundled end-to-end user examples."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eos_trensmuther import load_pipeline_result
from eos_trensmuther.cli import main


ROOT = Path(__file__).resolve().parents[1]


class PublicExampleTests(unittest.TestCase):
    def _run(self, relative_parameter, output, *extra):
        stdout = io.StringIO()
        stderr = io.StringIO()
        arguments = [
            "run",
            str(ROOT / relative_parameter),
            "--output-folder",
            str(output),
            "--overwrite",
        ]
        arguments.extend(extra)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(arguments)
        self.assertEqual(status, 0, stderr.getvalue())
        return stdout.getvalue()

    def test_regular_line_and_binodals_serial_examples_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cases = (
                (
                    "examples/regular/parameters_serial.py",
                    "RegularSynthetic2D/RegularSynthetic2D_serial",
                    0,
                ),
                (
                    "examples/first_order_line/parameters_serial.py",
                    "FirstOrderLineSynthetic/FirstOrderLineSynthetic_serial",
                    1,
                ),
                (
                    "examples/first_order_binodals/parameters_serial.py",
                    "FirstOrderBinodalsSynthetic/FirstOrderBinodalsSynthetic_serial",
                    4,
                ),
            )
            for parameter, relative_run, minimum_mixed in cases:
                with self.subTest(parameter=parameter):
                    text = self._run(parameter, output)
                    run = output / relative_run
                    result = load_pipeline_result(run / "inverse.eosinv")
                    self.assertTrue((run / "eos_table.dat").is_file())
                    self.assertEqual(int(np.count_nonzero(result.error_mask)), 0)
                    self.assertGreaterEqual(
                        int(np.count_nonzero(result.mixed_phase_mask)),
                        minimum_mixed,
                    )
                    self.assertIn("Completed:", text)

    def test_regular_multicore_example_and_slurm_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self._run(
                "examples/regular/parameters_multicore.py",
                output,
            )
            run = output / "RegularSynthetic2D/RegularSynthetic2D_multicore"
            result = load_pipeline_result(run / "inverse.eosinv")
            self.assertEqual(
                result.manifest["run"]["execution"]["workers_requested"], 2
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "slurm-plan",
                        str(ROOT / "examples/regular/parameters_slurm.py"),
                        "--output-folder",
                        str(output),
                        "--overwrite",
                        "--partition",
                        "cpu",
                        "--wall-time",
                        "00:10:00",
                    ]
                )
            self.assertEqual(status, 0, stderr.getvalue())
            slurm = (
                output
                / "RegularSynthetic2D/RegularSynthetic2D_slurm/diagnostic/slurm"
            )
            self.assertTrue((slurm / "array_plan.json").is_file())
            self.assertTrue((slurm / "submit_array.sbatch").is_file())
            self.assertIn("No job was submitted", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
