"""Guardrails and report tests for the solver-free Reach4D benchmark."""

import importlib.util
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_tables import TemporaryEoS, field_spec, metadata


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "tests/_support/benchmark_reachability_hrg4d.py"


def load_driver():
    specification = importlib.util.spec_from_file_location(
        "benchmark_reachability_hrg4d_test_module", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


class ReachabilityBenchmarkDriverTests(unittest.TestCase):
    def make_table(self):
        coordinates = ("T", "muB", "muQ", "muS")
        targets = ("e", "nB", "nQ", "nS")
        fields = {
            name: field_spec("coordinate", index)
            for index, name in enumerate(coordinates)
        }
        fields.update(
            {
                name: field_spec("target", index + 4)
                for index, name in enumerate(targets)
            }
        )
        axes = (
            (0.2, 0.4, 0.6),
            (0.0, 0.2, 0.4),
            (-0.1, 0.0, 0.1),
            (-0.2, 0.0, 0.2),
        )
        rows = [list(point) + list(point) for point in itertools.product(*axes)]
        fixture = TemporaryEoS(metadata(fields, 4), rows)
        self.addCleanup(fixture.close)
        return fixture

    def test_driver_completes_bounded_leaves_and_writes_auditable_reports(self):
        fixture = self.make_table()
        driver = load_driver()
        with tempfile.TemporaryDirectory() as directory:
            status = driver.main(
                [
                    str(fixture.metadata_path),
                    "--output-directory",
                    directory,
                    "--hierarchy-levels",
                    "3",
                    "--seed",
                    "417",
                    "--target-mode",
                    "representative",
                    "--known-reachable",
                    "2",
                    "--obvious-unreachable",
                    "2",
                    "--boundary-targets",
                    "2",
                    "--random-envelope",
                    "2",
                    "--root-residual-tolerance",
                    "1e-10",
                    "--query-batch-size",
                    "2",
                    "--maximum-pairwise-entries",
                    "64",
                    "--maximum-hierarchy-bytes",
                    "1048576",
                    "--maximum-benchmark-memory-bytes",
                    "104857600",
                    "--maximum-estimated-leaf-pairs",
                    "100000",
                    "--maximum-estimated-leaf-seconds",
                    "100",
                    "--maximum-benchmark-wall-seconds",
                    "100",
                    "--throughput-targets",
                    "2",
                    "--throughput-leaves",
                    "16",
                    "--complete-leaves",
                    "--oracle-equivalence-sample",
                    "4",
                ]
            )
            document = json.loads(
                (Path(directory) / "reachability_benchmark.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (
                Path(directory) / "reachability_benchmark.md"
            ).read_text(encoding="utf-8")

        self.assertEqual(status, 0)
        self.assertEqual(document["benchmark_schema_version"], 2)
        self.assertFalse(document["solver_invoked"])
        self.assertTrue(document["scientific_classification_complete"])
        self.assertEqual(
            document["configuration"]["maximum_estimated_leaf_seconds"],
            100.0,
        )
        self.assertIsNotNone(
            document["leaf_completion"][
                "additional_minimum_witness_leaf_pairs"
            ]
        )
        self.assertIsNotNone(
            document["leaf_completion"]["additional_batch_leaf_pairs"]
        )
        self.assertTrue(document["oracle_equivalence"]["completed"])
        self.assertEqual(
            document["oracle_equivalence"]["routing_witness_mismatches"], 0
        )
        self.assertIn("Total measured/estimated reachability time", markdown)
        self.assertIn("Additional vector-batch leaf pairs", markdown)


if __name__ == "__main__":
    unittest.main()
