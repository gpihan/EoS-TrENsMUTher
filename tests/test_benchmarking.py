"""Tests for structured Phase-6 performance observations."""

import json
import unittest

from eos_trensmuther import summarize_pipeline_performance

from tests.test_serial_pipeline import SerialPipelineTestCase


class BenchmarkingTests(SerialPipelineTestCase):
    def test_summary_is_complete_json_without_timing_thresholds(self):
        fields = {
            "T": self._field("coordinate", 0),
            "e": self._field("target", 1),
        }
        table = self.make_table(
            fields,
            [(temperature, 1.0 + temperature) for temperature in (0.2, 0.5, 0.8)],
        )
        result = self._pipeline(table).run()
        summary = summarize_pipeline_performance(result)
        document = summary.to_dict()
        json.dumps(document, allow_nan=False)

        performance = document["performance"]
        self.assertEqual(
            performance["restored_points"]
            + performance["computed_points_this_invocation"],
            result.grid.size,
        )
        self.assertEqual(
            performance["chunk_processing_seconds"]["count"],
            result.manifest["run"]["stage_counters"]["chunks"],
        )
        self.assertEqual(
            set(performance["chunk_stage_seconds"]),
            {
                "prescreen",
                "initial_guess",
                "phase_solve",
                "fast",
                "fallback",
                "projection",
                "output_evaluation",
            },
        )
        self.assertEqual(
            result.nbytes, result.plan.estimated_result_bytes
        )
        self.assertEqual(
            result.plan.estimated_peak_result_bytes, 2 * result.nbytes
        )

    @staticmethod
    def _field(role, column):
        from tests.test_tables import field_spec

        return field_spec(role, column)

    def _pipeline(self, table):
        from eos_trensmuther import EoSInversionPipeline

        return EoSInversionPipeline(
            table,
            self.make_config(
                table,
                count=5,
                chunk_size=2,
                projection_enabled=False,
                auxiliary_fields=(),
            ),
        )


if __name__ == "__main__":
    unittest.main()
