"""Tests for the policy-explicit generic text exporter."""

import hashlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    EoSInversionPipeline,
    ExecutionConfig,
    ExportConfigurationError,
    ExportDataError,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    PHASE_EXPORT_COLUMN_NAMES,
    PhaseExportConfig,
    PipelineConfig,
    ProjectionConfig,
    TextExportConfig,
    export_text,
)
from eos_trensmuther.exporters import EXTRA_INFO_COLUMN_NAMES

from tests.test_tables import TemporaryEoS, field_spec, metadata
from tests.test_pipeline import PipelineTestCase


class TextExporterTests(PipelineTestCase):
    def make_projected_result(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "P": field_spec("auxiliary", 4),
            "s": field_spec("auxiliary", 5),
            "cs2": field_spec("auxiliary", 6),
            "chi2B": field_spec("auxiliary", 7),
            "extra": field_spec("auxiliary", 8),
        }
        rows = []
        for temperature, chemical_potential in itertools.product(
            (1.0, 2.0, 3.0), (0.0, 1.0, 2.0)
        ):
            energy = temperature
            density = temperature + chemical_potential
            rows.append(
                (
                    temperature,
                    chemical_potential,
                    energy,
                    density,
                    energy + density,
                    2.0 * energy + density,
                    0.25,
                    density / 10.0,
                    -0.1 * chemical_potential,
                )
            )
        fixture = TemporaryEoS(metadata(fields, 2), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        config = PipelineConfig(
            grid=InversionGridSpec(
                "target", {"e": GridAxisSpec(2), "nB": GridAxisSpec(2)}
            ),
            residual_scales={"e": 1.0, "nB": 1.0},
            projection=ProjectionConfig(
                enabled=True,
                nearest_cell_starts=2,
                max_evaluations_per_start=40,
            ),
            execution=ExecutionConfig(workers=1, chunk_size=2),
            auxiliary_fields=("P", "s", "cs2", "chi2B", "extra"),
        )
        return EoSInversionPipeline(table, config).run()

    def test_projected_rows_are_included_and_tagged_by_default(self):
        result = self.make_projected_result()
        self.assertEqual(int(np.count_nonzero(result.projected_mask)), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "EoS_all.dat"
            exported = export_text(result, path)
            values = np.loadtxt(path)
            self.assertEqual(values.shape, (4, 9 + len(EXTRA_INFO_COLUMN_NAMES)))
            requested = result.grid.chunk(0, result.grid.size).canonical_targets
            np.testing.assert_allclose(values[:, :2], requested)
            np.testing.assert_allclose(
                values[:, 2:4], result.source_coordinates.reshape(4, 2)
            )
            document = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
            header = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("# ")
            ]
            self.assertTrue(all(line.startswith("# ") for line in header))
            self.assertEqual(len(header), 3)
            self.assertIn("EoS-TrENsMUTher version", header[0])
            self.assertIn("Fields: e[GeV^4]", header[1])
            self.assertIn("projected[dimensionless]", header[1])
            self.assertIn("Grid shape: 2 x 2", header[2])
            self.assertFalse(any("Tilde:" in line for line in header))
            self.assertEqual(document["columns"][0]["name"], "e")
            self.assertEqual(document["columns"][7]["name"], "chi2B")
            self.assertEqual(
                tuple(
                    column["name"]
                    for column in document["columns"][-len(EXTRA_INFO_COLUMN_NAMES) :]
                ),
                EXTRA_INFO_COLUMN_NAMES,
            )
            self.assertEqual(exported.extra_info_column_names, EXTRA_INFO_COLUMN_NAMES)
            self.assertNotIn("solved", exported.column_names)
            self.assertNotIn("unresolved", exported.column_names)
            self.assertNotIn("error", exported.column_names)
            projected_column = exported.column_names.index("projected")
            np.testing.assert_array_equal(
                values[:, projected_column], result.projected_mask.reshape(-1)
            )
            self.assertEqual(document["counts"]["solved"], 2)
            self.assertEqual(document["counts"]["projected"], 2)
            self.assertEqual(document["target_columns"], "requested canonical targets")
            self.assertEqual(
                document["table_sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
            with self.assertRaises(FileExistsError):
                export_text(
                    result,
                    path,
                )

    def test_extra_info_can_be_disabled_without_changing_physical_columns(self):
        result = self.make_projected_result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.dat"
            exported = export_text(
                result,
                path,
                config=TextExportConfig(extra_info=False),
            )
            values = np.atleast_2d(np.loadtxt(path))
            self.assertEqual(values.shape, (4, 9))
            self.assertEqual(exported.extra_info_column_names, ())
            manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["extra_info"]["enabled"])

    def test_deprecated_projected_switch_no_longer_excludes_projected_rows(self):
        result = self.make_projected_result()
        with tempfile.TemporaryDirectory() as directory:
            exported = export_text(
                result,
                Path(directory) / "table.dat",
                config=TextExportConfig(include_projected=False),
            )
            self.assertEqual(exported.projected_count, 2)
            manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["include_projected"])

    def test_tilde_header_definitions_appear_only_for_tilde_grid(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }
        document = metadata(fields, 2)
        document["charge_constraints"] = {
            "model": "linear_homogeneous",
            "dependent_densities": {"nQ": {"nB": 0.4}, "nS": {}},
        }
        rows = [
            [0.2, 0.0, 1.2, 0.02],
            [0.2, 0.2, 1.4, 0.04],
            [0.5, 0.0, 1.5, 0.03],
            [0.5, 0.2, 1.7, 0.05],
        ]
        fixture = TemporaryEoS(document, rows)
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        result = EoSInversionPipeline(
            table,
            self.make_config(table, space="tilde", count=2),
        ).run()
        with tempfile.TemporaryDirectory() as directory:
            exported = export_text(
                result,
                Path(directory) / "tilde.dat",
                config=TextExportConfig(
                    include_unusable=True,
                    include_tilde=True,
                ),
            )
            header = exported.table_path.read_text(encoding="utf-8").splitlines()[:12]
            self.assertEqual(sum("Tilde:" in line for line in header), 4)
            self.assertFalse(any("A*n" in line for line in header))
            self.assertTrue(
                any("muB_tilde = (4.6*nB) / T_tilde^2" in line for line in header)
            )
            self.assertTrue(
                any("nQ = 0.4*nB; nS = 0" in line for line in header)
            )
            self.assertTrue(
                any("muQ_tilde = (-0.2*nB)" in line for line in header)
            )
            self.assertTrue(
                any("muS_tilde = (1.6*nB)" in line for line in header)
            )
            self.assertTrue(any("Grid shape:" in line for line in header))
            self.assertEqual(
                exported.column_names[:8],
                (
                    "e",
                    "nB",
                    "T_tilde",
                    "muB_tilde",
                    "muQ_tilde",
                    "muS_tilde",
                    "T",
                    "muB",
                ),
            )
            self.assertEqual(
                exported.column_units[:8],
                (
                    "GeV^4",
                    "GeV^3",
                    "GeV",
                    "GeV",
                    "GeV",
                    "GeV",
                    "GeV",
                    "GeV",
                ),
            )
            self.assertIn("T_tilde[GeV]", header[0] + " " + header[1])
            self.assertIn("muB_tilde[GeV]", header[0] + " " + header[1])
            values = np.atleast_2d(np.loadtxt(exported.table_path))
            chunk = result.grid.chunk(0, result.grid.size)
            expected_tilde = np.column_stack(
                (chunk.requested_values, chunk.requested_dependent_tilde)
            )
            np.testing.assert_allclose(values[:, 2:6], expected_tilde)
            manifest = json.loads(
                exported.manifest_path.read_text(encoding="utf-8")
            )
            transform = manifest["tilde_transform"]
            self.assertTrue(transform["columns_included"])
            self.assertEqual(
                transform["column_semantics"],
                "requested transformed-grid coordinates",
            )
            self.assertEqual(
                transform["reduced_chemical_coefficients"]["muB_tilde"],
                {"nB": 4.6},
            )
            self.assertEqual(
                transform["reduced_chemical_coefficients"]["muQ_tilde"],
                {"nB": -0.2},
            )

    def test_tilde_columns_require_a_tilde_grid(self):
        result = self.make_projected_result()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ExportDataError, "only for a tilde-grid run"
            ):
                export_text(
                    result,
                    Path(directory) / "target.dat",
                    config=TextExportConfig(include_tilde=True),
                )

    def test_dimension_generalized_columns_for_1d_and_4d(self):
        for dimension in (1, 4):
            with self.subTest(dimension=dimension):
                coordinate_names = ("T", "muB", "muQ", "muS")[:dimension]
                target_names = ("e", "nB", "nQ", "nS")[:dimension]
                fields = {
                    name: field_spec("coordinate", index)
                    for index, name in enumerate(coordinate_names)
                }
                for index, name in enumerate(target_names, start=dimension):
                    fields[name] = field_spec("target", index)
                fields["P"] = field_spec("auxiliary", 2 * dimension)
                fields["s"] = field_spec("auxiliary", 2 * dimension + 1)
                axes = tuple(
                    (0.2 + 0.1 * index, 0.5 + 0.1 * index)
                    for index in range(dimension)
                )
                rows = []
                for point in itertools.product(*axes):
                    rows.append(
                        list(point)
                        + list(point)
                        + [sum(point), 2.0 * sum(point)]
                    )
                fixture = TemporaryEoS(metadata(fields, dimension), rows)
                self.addCleanup(fixture.close)
                table = ForwardTable.load_validated(fixture.metadata_path)
                config = self.make_config(
                    table,
                    count=2,
                    auxiliary_fields=("P", "s"),
                )
                result = EoSInversionPipeline(table, config).run()
                unusable = int(np.count_nonzero(~result.usable_state_mask))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "EoS_all.dat"
                    if unusable:
                        with self.assertRaises(ExportDataError):
                            export_text(result, path)
                    exported = export_text(
                        result,
                        path,
                        config=TextExportConfig(
                            include_unusable=True,
                            write_header=True,
                        ),
                    )
                    values = np.loadtxt(path)
                    values = np.atleast_2d(values)
                    expected_names = (
                        result.plan.target_names
                        + result.plan.source_names
                        + result.plan.auxiliary_names
                        + EXTRA_INFO_COLUMN_NAMES
                    )
                    self.assertEqual(exported.column_names, expected_names)
                    self.assertEqual(
                        values.shape,
                        (
                            result.grid.size,
                            2 * dimension + 2 + len(EXTRA_INFO_COLUMN_NAMES),
                        ),
                    )
                    requested = result.grid.chunk(
                        0, result.grid.size
                    ).canonical_targets
                    np.testing.assert_allclose(values[:, :dimension], requested)
                    np.testing.assert_allclose(
                        values[:, dimension : 2 * dimension],
                        result.source_coordinates.reshape(
                            result.grid.size, dimension
                        ),
                        equal_nan=True,
                    )
                    np.testing.assert_allclose(
                        values[:, 2 * dimension : 2 * dimension + 2],
                        result.auxiliary_values.reshape(result.grid.size, 2),
                        equal_nan=True,
                    )
                    manifest = json.loads(
                        exported.manifest_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(manifest["schema_version"], 6)
                    self.assertEqual(manifest["dimension"], dimension)
                    self.assertEqual(
                        manifest["column_contract"],
                        "targets_sources_auxiliaries_extra_info",
                    )
                    self.assertEqual(manifest["counts"]["unusable"], unusable)
                    self.assertEqual(exported.unusable_count, unusable)

    def test_generic_text_requires_opt_in_for_certified_unreachable_rows(self):
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
            (1.0, 2.0), (-1.0, 1.0)
        ):
            density = temperature * chemical_potential
            rows.append(
                [
                    temperature,
                    chemical_potential,
                    temperature,
                    density,
                    temperature + density,
                    2.0 * temperature + density,
                    0.25,
                    0.1 * density,
                ]
            )
        fixture = TemporaryEoS(
            metadata(fields, 2),
            rows,
        )
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        result = EoSInversionPipeline(
            table,
            PipelineConfig(
                grid=InversionGridSpec(
                    "target",
                    {
                        "e": GridAxisSpec(2, 1.0, 2.0),
                        "nB": GridAxisSpec(3, -2.0, 2.0),
                    },
                ),
                residual_scales={"e": 1.0, "nB": 2.0},
                projection=ProjectionConfig(enabled=False),
                execution=ExecutionConfig(workers=1, chunk_size=6),
                auxiliary_fields=("P", "s", "cs2", "chi2B"),
            ),
        ).run()
        certified = result.certified_unreachable_mask.reshape(-1)
        self.assertGreater(int(np.count_nonzero(certified)), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "EoS_all.dat"
            with self.assertRaises(ExportDataError):
                export_text(result, path)
            exported = export_text(
                result,
                path,
                config=TextExportConfig(include_unusable=True),
            )
            values = np.atleast_2d(np.loadtxt(path))
            self.assertEqual(exported.unusable_count, int(np.count_nonzero(certified)))
            self.assertTrue(np.all(np.isfinite(values[:, :2])))
            self.assertTrue(np.all(np.isnan(values[certified, 2:8])))
            self.assertTrue(np.all(np.isfinite(values[~certified, 2:8])))
            self.assertTrue(np.all(np.isfinite(values[:, 8:])))
            certified_column = exported.column_names.index("certified_unreachable")
            np.testing.assert_array_equal(
                values[:, certified_column], certified.astype(float)
            )

    def test_phase_columns_are_versioned_opt_in(self):
        result = self.make_projected_result()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exported = export_text(
                result,
                root / "table.dat",
                config=TextExportConfig(
                    include_projected=True,
                    phase=PhaseExportConfig(),
                ),
            )
            values = np.loadtxt(exported.table_path)
            self.assertEqual(
                values.shape,
                (4, 14 + len(EXTRA_INFO_COLUMN_NAMES) - 2),
            )
            self.assertEqual(
                exported.phase_column_names, PHASE_EXPORT_COLUMN_NAMES
            )
            self.assertEqual(exported.undefined_phase_count, 4)
            np.testing.assert_allclose(
                values[:, 9:13],
                np.column_stack(
                    (
                        result.phase_codes.reshape(-1),
                        result.phase_route_codes.reshape(-1),
                        result.line_mixture_fraction.reshape(-1),
                        result.phase_boundary.reshape(-1),
                    )
                ),
                equal_nan=True,
            )
            manifest = json.loads(
                exported.manifest_path.read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["phase_export"]["enabled"])
            self.assertEqual(manifest["phase_export"]["schema_version"], 1)
            self.assertEqual(manifest["columns"][13]["role"], "phase")
            self.assertEqual(manifest["columns"][-1]["role"], "state")
            self.assertIn("source_auxiliary_evaluation", manifest)

    def test_phase_export_configuration_is_strict(self):
        with self.assertRaises(ExportConfigurationError):
            PhaseExportConfig(schema_version=2)
        with self.assertRaises(ExportConfigurationError):
            PhaseExportConfig(columns=("unknown",))
        with self.assertRaises(ExportConfigurationError):
            PhaseExportConfig(columns=())


if __name__ == "__main__":
    unittest.main()
