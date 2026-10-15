"""Public source-table sanity-check regressions."""

import itertools
import unittest

import numpy as np

from eos_trensmuther import EoSValidationError, ForwardTable
from eos_trensmuther._table_sanity import scan_table_geometry
from tests.test_tables import TemporaryEoS, field_spec, metadata


class TableSanityTests(unittest.TestCase):
    @staticmethod
    def _regular_fields(dimension):
        coordinates = ("T", "muB", "muQ", "muS")[:dimension]
        targets = ("e", "nB", "nQ", "nS")[:dimension]
        fields = {
            name: field_spec("coordinate", index)
            for index, name in enumerate(coordinates)
        }
        fields.update(
            {
                name: field_spec("target", dimension + index)
                for index, name in enumerate(targets)
            }
        )
        return coordinates, targets, fields

    def test_valid_regular_table_has_exhaustive_geometry_summary(self):
        coordinates, targets, fields = self._regular_fields(2)
        rows = [list(point) + list(point) for point in itertools.product((0.1, 0.2, 0.3), repeat=2)]
        fixture = TemporaryEoS(metadata(fields, 2), rows)
        self.addCleanup(fixture.close)

        table = ForwardTable.load_validated(fixture.metadata_path)
        geometry = table.report.statistics["geometry"]

        self.assertEqual(geometry["strategy"], "exhaustive_exact_bilinear_2d")
        self.assertEqual(geometry["sampled_cells"], 4)
        self.assertEqual(geometry["counts"]["regular_positive"], 4)
        self.assertEqual(geometry["counts"]["sign_changing_or_singular"], 0)
        self.assertEqual(table.coordinate_names, coordinates)
        self.assertEqual(table.target_names, targets)

    def test_isolated_bilinear_fold_is_warned_not_rejected(self):
        axis = np.arange(5.0)
        energy = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0706995088614608, 0.8501739450206383, 1.646901113471949, 1.0],
                [2.0, 2.6649835124141963, 1.9655512378269018, 2.2847059039273474, 2.0],
                [3.0, 3.137373035993125, 3.2735050151616023, 3.5086450310598862, 3.0],
                [4.0, 4.0, 4.0, 4.0, 4.0],
            ]
        )
        density = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.2870651158669402, 2.030696870597297, 2.7712730231242673, 4.0],
                [0.0, 0.7158395922527567, 1.9910616396601177, 3.405364590119404, 4.0],
                [0.0, 1.1051823043957822, 2.018569826136145, 3.0900450333343903, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )
        _, _, fields = self._regular_fields(2)
        rows = [
            [axis[i], axis[j], energy[i, j], density[i, j]]
            for i, j in itertools.product(range(5), repeat=2)
        ]
        fixture = TemporaryEoS(metadata(fields, 2), rows)
        self.addCleanup(fixture.close)

        table = ForwardTable.load_validated(fixture.metadata_path)
        geometry = table.report.statistics["geometry"]

        self.assertTrue(table.report.is_valid)
        self.assertEqual(geometry["counts"]["regular_positive"], 15)
        self.assertEqual(geometry["counts"]["regular_negative"], 0)
        self.assertEqual(geometry["counts"]["sign_changing_or_singular"], 1)
        self.assertEqual(geometry["isolated_defects"], 1)
        self.assertIn(
            "geometry.sign_changing_cells",
            [issue.code for issue in table.report.warnings],
        )

    @staticmethod
    def _orientation_reversal_rows():
        """Return a map with negative, singular, then positive T slabs."""

        temperatures = (0.1, 0.2, 0.3, 0.4)
        chemical_potentials = (0.0, 0.5, 1.0)
        return [
            [temperature, mu_b, temperature, (temperature - 0.25) * mu_b]
            for temperature, mu_b in itertools.product(
                temperatures, chemical_potentials
            )
        ]

    def test_regular_orientation_reversal_rejects_ordinary_table(self):
        _, _, fields = self._regular_fields(2)
        fixture = TemporaryEoS(
            metadata(fields, 2), self._orientation_reversal_rows()
        )
        self.addCleanup(fixture.close)

        report = ForwardTable(fixture.metadata_path).validate()
        geometry = report.statistics["geometry"]

        self.assertFalse(report.is_valid)
        self.assertGreater(geometry["counts"]["regular_positive"], 0)
        self.assertGreater(geometry["counts"]["regular_negative"], 0)
        issue = next(
            item
            for item in report.errors
            if item.code == "geometry.orientation_reversal"
        )
        self.assertFalse(issue.context["declared_first_order"])
        self.assertGreater(issue.context["sign_changing_or_singular_cells"], 0)
        self.assertIn("Either uniform orientation sign is acceptable", issue.message)
        with self.assertRaises(EoSValidationError):
            ForwardTable.load_validated(fixture.metadata_path)

    def test_declared_first_order_orientation_reversal_remains_warning(self):
        _, _, fields = self._regular_fields(2)
        document = metadata(fields, 2)
        document["scientific"]["phase_structure"] = {
            "status": "known",
            "critical_points": [
                {
                    "name": "synthetic critical point",
                    "status": "known",
                    "location": {
                        "T": {"value": 150.0, "unit": "MeV"},
                        "muB": {"value": 0.0, "unit": "GeV"},
                    },
                }
            ],
            "first_order_transition": {
                "present": True,
                "representation": "line",
                "kind": "synthetic first-order line",
                "critical_point": "synthetic critical point",
                "critical_point_tolerance": {
                    "absolute": 1e-12,
                    "relative": 1e-10,
                },
                "coexistence_data": {
                    "file": "transition.dat",
                    "delimiter": "whitespace",
                    "comment_prefix": "#",
                    "skip_rows": 0,
                    "encoding": "utf-8",
                    "expected_columns": 2,
                    "fields": ["muB", "T"],
                    "columns": {
                        "muB": {"index": 0, "unit": "GeV"},
                        "T": {"index": 1, "unit": "MeV"},
                    },
                },
            },
        }
        fixture = TemporaryEoS(document, self._orientation_reversal_rows())
        self.addCleanup(fixture.close)
        fixture.write_sidecar("transition.dat", "0.0 150.0\n1.0 100.0\n")

        table = ForwardTable.load_validated(fixture.metadata_path)
        issues = [
            item
            for item in table.report.warnings
            if item.code == "geometry.orientation_reversal"
        ]

        self.assertEqual(len(issues), 1)
        self.assertTrue(issues[0].context["declared_first_order"])
        self.assertIn("dedicated phase-aware route", issues[0].message)

    def test_noncanonical_rows_report_first_offending_coordinate(self):
        _, _, fields = self._regular_fields(1)
        document = metadata(fields, 1)
        document["grid"]["reorder"] = False
        fixture = TemporaryEoS(document, [[0.1, 0.1], [0.3, 0.3], [0.2, 0.2]])
        self.addCleanup(fixture.close)

        report = ForwardTable(fixture.metadata_path).validate()
        issue = next(item for item in report.errors if item.code == "grid.row_order")

        self.assertIn("Coordinate 'T'", issue.message)
        self.assertIn("rows 1 -> 2", issue.message)
        self.assertEqual(tuple(issue.context["row_indices"]), (1, 2))
        self.assertEqual(tuple(issue.context["values"]), (0.3, 0.2))

    def test_duplicate_coordinate_tuple_is_rejected(self):
        _, _, fields = self._regular_fields(1)
        fixture = TemporaryEoS(
            metadata(fields, 1), [[0.1, 0.1], [0.2, 0.2], [0.2, 0.2]]
        )
        self.addCleanup(fixture.close)
        report = ForwardTable(fixture.metadata_path).validate()
        self.assertIn("grid.duplicate_coordinates", [item.code for item in report.errors])

    def test_nonfinite_coordinate_reports_rows_and_values(self):
        _, _, fields = self._regular_fields(1)
        fixture = TemporaryEoS(metadata(fields, 1), [[0.1, 0.1], [np.nan, 0.2]])
        self.addCleanup(fixture.close)
        report = ForwardTable(fixture.metadata_path).validate()
        issue = next(
            item
            for item in report.errors
            if item.code == "field.nonfinite.coordinate.nan"
        )
        self.assertEqual(tuple(issue.context["row_indices"]), (1,))
        self.assertEqual(tuple(issue.context["values"]), ("nan",))

    def test_wholly_degenerate_map_is_rejected_in_normal_loading(self):
        _, _, fields = self._regular_fields(2)
        rows = [
            [first, second, 1.0, 2.0]
            for first, second in itertools.product((0.1, 0.2), repeat=2)
        ]
        fixture = TemporaryEoS(metadata(fields, 2), rows)
        self.addCleanup(fixture.close)

        report = ForwardTable(fixture.metadata_path).validate()

        self.assertFalse(report.is_valid)
        self.assertIn(
            "geometry.degenerate_map", [item.code for item in report.errors]
        )

    def test_large_four_dimensional_scan_is_bounded_and_deterministic(self):
        axis = np.linspace(0.0, 1.0, 18)
        mesh = np.meshgrid(axis, axis, axis, axis, indexing="ij")
        targets = {
            name: values
            for name, values in zip(("e", "nB", "nQ", "nS"), mesh)
        }
        first = scan_table_geometry(
            (axis,) * 4,
            targets,
            ("e", "nB", "nQ", "nS"),
            max_sampled_cells=1200,
            max_refinement_cells=200,
        )
        second = scan_table_geometry(
            (axis,) * 4,
            targets,
            ("e", "nB", "nQ", "nS"),
            max_sampled_cells=1200,
            max_refinement_cells=200,
        )

        self.assertGreater(first["total_cells"], 1200)
        self.assertLessEqual(first["sampled_cells"], 1400)
        self.assertEqual(first, second)
        self.assertEqual(
            first["strategy"],
            "bounded_deterministic_center_sample_with_refinement",
        )
        self.assertFalse(first["injectivity_proof"])


if __name__ == "__main__":
    unittest.main()
