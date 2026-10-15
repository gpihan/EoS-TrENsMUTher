"""Real smooth HRG4D metadata and bounded end-to-end inversion validation."""

import copy
import itertools
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    CheckpointConfig,
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    MultilinearInterpolator,
    PhaseLabel,
    PhaseRoute,
    PipelineConfig,
    ProjectionConfig,
    ReachabilityConfig,
    pipeline_status_code,
)

from tests.test_tables import TemporaryEoS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HRG_METADATA = REPOSITORY_ROOT / "EoSTables" / "HRG4D" / "metadata.json"
EXPECTED_FINGERPRINT = (
    "9029081790e2d8079da28fe939ea433ec1a1a24f878e035981d31fcf1a9daa09"
)

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


class HRG4DValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HRG_METADATA.is_file():
            raise unittest.SkipTest("real HRG4D table is not available")
        cls.full_table = ForwardTable.load_validated(HRG_METADATA)
        selected = (23, 24, 25)
        flat_indices = [
            np.ravel_multi_index(index, cls.full_table.shape, order="C")
            for index in itertools.product(selected, repeat=4)
        ]
        rows = np.asarray(cls.full_table.raw_data)[flat_indices].copy()
        document = json.loads(HRG_METADATA.read_text(encoding="utf-8"))
        document["name"] = "HRG4D compact genuine-node regression fixture"
        document["table_revision"] = "source-snapshot-%s" % EXPECTED_FINGERPRINT[:16]
        document["data_file"] = "table.dat"
        document["description"] = (
            "3^4 interior Cartesian subset of genuine HRG4D source rows at "
            "axis indices 23, 24, and 25."
        )
        axes = tuple(axis[list(selected)] for axis in cls.full_table.axes)
        document["grid"]["expected_shape"] = [3, 3, 3, 3]
        document["grid"]["expected_bounds"] = {
            name: [float(axis[0]), float(axis[-1])]
            for name, axis in zip(cls.full_table.coordinate_names, axes)
        }
        document["grid"]["expected_spacing"] = {
            name: float(axis[1] - axis[0])
            for name, axis in zip(cls.full_table.coordinate_names, axes)
        }
        document["scientific"]["origin"] = {
            "generator": "test fixture selected from HRG4D.dat",
            "source_snapshot_fingerprint": EXPECTED_FINGERPRINT,
            "selected_axis_indices": list(selected),
        }
        cls.fixture = TemporaryEoS(document, rows)
        cls.table = ForwardTable.load_validated(cls.fixture.metadata_path)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.close()
        del cls.table
        del cls.full_table

    def make_config(self, *, workers):
        target_interpolator = MultilinearInterpolator(
            self.table, self.table.target_names
        )
        source_center = np.asarray(
            [0.5 * (axis[1] + axis[2]) for axis in self.table.axes], dtype=float
        )
        target_center = target_interpolator.evaluate(source_center)
        target_columns = np.column_stack(
            [self.table.field(name, shaped=False) for name in self.table.target_names]
        )
        spans = np.ptp(target_columns, axis=0)
        epsilons = 1.0e-5 * spans
        if np.any(epsilons <= 0.0):
            self.fail("compact HRG4D target image is degenerate")
        axes = {
            name: GridAxisSpec(
                2,
                float(center - epsilon),
                float(center + epsilon),
            )
            for name, center, epsilon in zip(
                self.table.target_names, target_center, epsilons
            )
        }
        return PipelineConfig(
            grid=InversionGridSpec("target", axes),
            residual_scales={
                name: float(span)
                for name, span in zip(self.table.target_names, spans)
            },
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(workers=workers, chunk_size=3),
            auxiliary_fields=("P", "s"),
        )

    def test_full_real_table_metadata_grid_units_and_snapshot(self):
        table = self.full_table
        self.assertEqual(table.shape, (50, 50, 50, 50))
        self.assertEqual(table.coordinate_names, ("T", "muB", "muQ", "muS"))
        self.assertEqual(table.target_names, ("e", "nB", "nQ", "nS"))
        self.assertEqual(table.auxiliary_names, ("P", "s"))
        self.assertEqual(table.grid.uniform_by_axis, (True, True, True, True))
        self.assertFalse(table.grid.was_reordered)
        self.assertEqual(len(table.report.errors), 0)
        self.assertEqual(
            {issue.code for issue in table.report.warnings},
            {"scientific.publications_missing", "scientific.origin_missing"},
        )
        self.assertEqual(table.snapshot_fingerprint(), EXPECTED_FINGERPRINT)
        self.assertEqual(
            tuple(
                table.metadata.fields[name].transform.temperature_power
                for name in ("e", "nB", "nQ", "nS", "P", "s")
            ),
            (4, 3, 3, 3, 4, 3),
        )

    def test_real_node_subset_inverts_and_forward_reevaluates(self):
        result = EoSInversionPipeline(self.table, self.make_config(workers=1)).run()
        self.assertEqual(result.grid.shape, (2, 2, 2, 2))
        self.assertGreater(int(np.count_nonzero(result.solution_mask)), 0)
        self.assertFalse(np.any(result.error_mask))
        self.assertFalse(np.any(result.invalid_target_mask))
        usable = result.solution_mask
        target_interpolator = MultilinearInterpolator(
            self.table, self.table.target_names
        )
        auxiliary_interpolator = MultilinearInterpolator(
            self.table, ("P", "s")
        )
        forward = target_interpolator.evaluate(result.source_coordinates[usable])
        auxiliary = auxiliary_interpolator.evaluate(
            result.source_coordinates[usable]
        )
        np.testing.assert_allclose(
            forward, result.achieved_targets[usable], rtol=3e-13, atol=3e-15
        )
        np.testing.assert_allclose(
            result.achieved_targets[usable],
            result.grid.chunk(0, result.grid.size).canonical_targets[usable.reshape(-1)],
            rtol=2e-8,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            auxiliary, result.auxiliary_values[usable], rtol=3e-13, atol=3e-15
        )
        self.assertTrue(np.all(result.auxiliary_defined[usable]))
        self.assertTrue(
            np.all(
                result.phase_codes
                == pipeline_status_code(PhaseLabel.NOT_APPLICABLE)
            )
        )
        self.assertTrue(
            np.all(
                result.phase_route_codes
                == pipeline_status_code(PhaseRoute.NO_TRANSITION)
            )
        )

    def test_real_node_subset_serial_process_equivalence(self):
        serial_config = self.make_config(workers=1)
        process_config = replace(
            serial_config,
            execution=replace(serial_config.execution, workers=2),
        )
        serial = EoSInversionPipeline(self.table, serial_config).run()
        process = EoSInversionPipeline(self.table, process_config).run()
        _assert_scientific_result_equal(serial, process)
        self.assertEqual(dict(serial.status_counts), dict(process.status_counts))

    def test_hierarchical_depth7_matches_exhaustive_process_and_checkpoint(self):
        serial_config = self.make_config(workers=1)
        process_config = replace(
            serial_config,
            execution=replace(serial_config.execution, workers=2),
        )
        reachability = ReachabilityConfig(
            method="hierarchical",
            hierarchy_levels=7,
            query_batch_size=4,
            cell_batch_size=4,
        )
        exhaustive = EoSInversionPipeline(self.table, serial_config).run()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = CheckpointConfig(Path(directory) / "checkpoint")
            hierarchical = EoSInversionPipeline(
                self.table,
                serial_config,
                reachability_config=reachability,
            ).run(checkpoint=checkpoint)
            process = EoSInversionPipeline(
                self.table,
                process_config,
                reachability_config=reachability,
            ).run()
            resumed = EoSInversionPipeline(
                self.table,
                process_config,
                reachability_config=reachability,
            ).run(checkpoint=checkpoint)
            checkpoint_members = []
            for path in sorted(
                (Path(directory) / "checkpoint" / "chunks").glob(
                    "*.eoschunk"
                )
            ):
                with zipfile.ZipFile(path, mode="r") as archive:
                    checkpoint_members.extend(archive.namelist())

        for reference in (hierarchical, process, resumed):
            _assert_scientific_result_equal(exhaustive, reference)
            self.assertEqual(
                dict(exhaustive.status_counts), dict(reference.status_counts)
            )
        self.assertEqual(
            hierarchical.manifest["run"]["reachability"]["method"],
            "hierarchical",
        )
        self.assertEqual(process.manifest["run"]["execution"]["mode"], "process")
        self.assertEqual(
            resumed.manifest["run"]["execution"]["mode"],
            "checkpoint_resume_only",
        )
        self.assertTrue(checkpoint_members)
        self.assertFalse(
            any(
                "localization" in member or "candidate_cell" in member
                for member in checkpoint_members
            )
        )


if __name__ == "__main__":
    unittest.main()
