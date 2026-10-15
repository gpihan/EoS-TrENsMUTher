"""Smoke coverage for the Phase-11 lower-dimensional benchmark driver."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tests._support import benchmark_lowdim_reachability_seed


class LowDimensionalSeedBenchmarkTests(unittest.TestCase):
    def test_quick_report_covers_every_lower_dimension_and_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            with redirect_stdout(io.StringIO()):
                status = benchmark_lowdim_reachability_seed.main(
                    [
                        "--output-directory",
                        str(output),
                        "--quick",
                        "--repeats",
                        "1",
                        "--localization-repeats",
                        "1",
                        "--maximum-seed-samples",
                        "8",
                    ]
                )

            self.assertEqual(status, 0)
            document = json.loads(
                (output / "benchmark.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["dimension"] for item in document["dimensions"]],
                [1, 2, 3],
            )
            self.assertTrue(document["first_order_tables_excluded"])
            self.assertEqual(document["hierarchy_scope_unchanged"], "4D only")
            for item in document["dimensions"]:
                localization = item["localization"]
                self.assertGreater(localization["csr_pairs"], 0)
                self.assertGreater(localization["csr_bytes"], 0)
                self.assertGreater(
                    localization["isolated_seed_sample_count"], 0
                )
                self.assertTrue(item["decision"]["parity_passed"])
                for comparison in item["pipeline"]["parity"]:
                    self.assertTrue(comparison["terminal_codes_equal"])
                    self.assertGreater(comparison["solutions_compared"], 0)
                    self.assertTrue(comparison["source_coordinates_close"])
            self.assertTrue((output / "BENCHMARK.md").is_file())


if __name__ == "__main__":
    unittest.main()
