"""Tests for metadata-driven forward-table ingestion and validation."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    BoundaryComputationError,
    ChargeConstraintSpec,
    CoexistenceTable,
    EoSNotValidatedError,
    EoSValidationError,
    ForwardTable,
    FirstOrderTransitionSpec,
    HBARC_GEV_FM,
    TransformConstraintError,
    TransformDomainError,
    expand_independent_tilde,
    targets_to_tilde,
    tilde_to_targets,
    transform_targets_to_tilde,
)
from eos_trensmuther.tables import TransformSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def field_spec(
    role,
    column,
    *,
    scale=1.0,
    temperature_power=0,
    hbarc_power=0,
    nonfinite_policy="forbid",
):
    return {
        "role": role,
        "column": column,
        "source_unit": "test-source-unit",
        "canonical_unit": "test-canonical-unit",
        "transform": {
            "scale_factor": scale,
            "temperature_power": temperature_power,
            "hbarc_power": hbarc_power,
        },
        "nonfinite_policy": nonfinite_policy,
        "output": True,
        "interpolation": "linear",
    }


def metadata(fields, dimension, *, require_uniform=False, expected_shape=None):
    coordinate_names = ("T", "muB", "muQ", "muS")[:dimension]
    target_names = ("e", "nB", "nQ", "nS")[:dimension]
    for name in coordinate_names:
        fields[name]["canonical_unit"] = "GeV"
    for name in target_names:
        fields[name]["canonical_unit"] = "GeV^4" if name == "e" else "GeV^3"
    grid = {
        "kind": "rectilinear",
        "require_complete": True,
        "require_uniform": require_uniform,
        "reorder": True,
        "duplicate_policy": "error",
        "absolute_tolerance": 1e-12,
        "relative_tolerance": 1e-10,
        "min_axis_points": 2,
        "expected_bounds": {},
        "expected_spacing": {},
    }
    if expected_shape is not None:
        grid["expected_shape"] = expected_shape
    return {
        "schema_version": 1,
        "name": "synthetic test EoS",
        "table_revision": "test-1",
        "description": "small generated test table",
        "dimension": dimension,
        "data_file": "table.dat",
        "format": {
            "delimiter": "whitespace",
            "comment_prefix": "#",
            "skip_rows": 0,
            "encoding": "utf-8",
            "expected_columns": len(fields),
            "missing_values": [],
        },
        "fields": fields,
        "grid": grid,
        "scientific": {
            "origin": {"generator": "unit test"},
            "publications": [],
            "phase_structure": {
                "status": "none",
                "critical_points": [],
                "first_order_transition": None,
            },
        },
    }


class TemporaryEoS:
    def __init__(self, metadata_document, table):
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.metadata_path = self.directory / "metadata.json"
        self.table_path = self.directory / "table.dat"
        self.metadata_path.write_text(json.dumps(metadata_document), encoding="utf-8")
        np.savetxt(str(self.table_path), np.asarray(table, dtype=float))

    def close(self):
        self._directory.cleanup()

    def write_sidecar(self, name, text):
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path


class EoSTestCase(unittest.TestCase):
    def make_eos(self, metadata_document, table):
        fixture = TemporaryEoS(metadata_document, table)
        self.addCleanup(fixture.close)
        return ForwardTable(
            fixture.metadata_path, _allow_unsuitable_geometry=True
        )

    def assert_issue(self, report, code):
        self.assertIn(code, [issue.code for issue in report.issues])


class TransformTests(EoSTestCase):
    def test_combined_transform_uses_converted_temperature(self):
        fields = {
            "T": field_spec("coordinate", 0, scale=0.001),
            "e": field_spec(
                "target",
                1,
                scale=0.5,
                temperature_power=3,
                hbarc_power=2,
            ),
        }
        eos = self.make_eos(metadata(fields, 1), [[200.0, 2.0], [300.0, 4.0]])

        report = eos.validate()

        self.assertTrue(report.is_valid)
        expected = 2.0 * 0.5 * 0.2**3 * HBARC_GEV_FM**2
        self.assertAlmostEqual(eos.field("e", shaped=False)[0], expected)
        np.testing.assert_allclose(eos.axes[0], [0.2, 0.3])

    def test_mev_per_fm3_and_inverse_fm3_conversions(self):
        fields = {
            "T": field_spec("coordinate", 0, scale=0.001),
            "e": field_spec("target", 1, scale=0.001, hbarc_power=3),
            "density": field_spec("auxiliary", 2, hbarc_power=3),
        }
        eos = self.make_eos(metadata(fields, 1), [[100.0, 5.0, 7.0], [200.0, 6.0, 8.0]])

        self.assertTrue(eos.validate().is_valid)
        self.assertAlmostEqual(eos.field("e", shaped=False)[0], 5e-3 * HBARC_GEV_FM**3)
        self.assertAlmostEqual(eos.field("density", shaped=False)[0], 7 * HBARC_GEV_FM**3)

    def test_default_constraint_round_trips_all_dimensions(self):
        random = np.random.RandomState(1729)
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                energy = random.uniform(0.01, 2.0, size=(25, 1))
                densities = random.uniform(
                    -0.2, 0.2, size=(25, dimension - 1)
                )
                targets = np.column_stack((energy, densities))

                transformed = transform_targets_to_tilde(targets, dimension)
                reconstructed = tilde_to_targets(
                    transformed.independent_values, dimension
                )

                self.assertEqual(transformed.independent_values.shape[1], dimension)
                self.assertEqual(transformed.all_values.shape[1], 4)
                np.testing.assert_allclose(reconstructed, targets, rtol=2e-14, atol=2e-14)

    def test_default_3d_constraint_has_consistent_dependent_muS(self):
        targets = np.asarray([[1.0, 0.12, -0.03], [2.0, -0.04, 0.08]])

        transformed = transform_targets_to_tilde(targets, 3)
        mu_b = transformed.independent_values[:, 1]
        mu_q = transformed.independent_values[:, 2]

        self.assertEqual(transformed.dependent_names, ("muS_tilde",))
        np.testing.assert_allclose(
            transformed.dependent_values[:, 0], (mu_b - mu_q) / 3.0
        )
        np.testing.assert_allclose(
            tilde_to_targets(transformed.independent_values, 3), targets
        )

    def test_heavy_ion_constraint_changes_axes_and_derives_omitted_potentials(self):
        constraints = ChargeConstraintSpec(
            dimension=2,
            dependent_densities={"nQ": {"nB": 0.4}, "nS": {}},
        )
        targets = np.asarray([[1.0, 0.2], [2.0, -0.1]])

        transformed = transform_targets_to_tilde(targets, 2, constraints)
        mu_b = transformed.independent_values[:, 1]

        np.testing.assert_allclose(
            transformed.dependent_values[:, 0], -mu_b / 23.0
        )
        np.testing.assert_allclose(
            transformed.dependent_values[:, 1], 8.0 * mu_b / 23.0
        )
        np.testing.assert_allclose(
            tilde_to_targets(transformed.independent_values, 2, constraints),
            targets,
        )
        expanded = expand_independent_tilde(
            transformed.independent_values, 2, constraints
        )
        np.testing.assert_allclose(expanded.all_values, transformed.all_values)

    def test_singular_constraint_reduction_is_rejected(self):
        with self.assertRaises(TransformConstraintError):
            ChargeConstraintSpec(
                dimension=2,
                dependent_densities={"nQ": {"nB": 5.0}, "nS": {}},
            )

    def test_finite_constraint_coefficients_that_overflow_are_rejected(self):
        with self.assertRaises(TransformConstraintError):
            ChargeConstraintSpec(
                dimension=2,
                dependent_densities={
                    "nQ": {"nB": 1e308},
                    "nS": {"nB": 1e308},
                },
            )

    def test_constraint_identifier_is_semantic_and_deterministic(self):
        first = ChargeConstraintSpec(
            dimension=3,
            dependent_densities={"nS": {"nB": 0.1, "nQ": 0.2}},
        )
        reordered = ChargeConstraintSpec(
            dimension=3,
            dependent_densities={"nS": {"nQ": 0.2, "nB": 0.1}},
        )
        with_explicit_zero = ChargeConstraintSpec(
            dimension=3,
            dependent_densities={"nS": {"nB": 0.1, "nQ": 0.0}},
        )
        without_explicit_zero = ChargeConstraintSpec(
            dimension=3,
            dependent_densities={"nS": {"nB": 0.1}},
        )

        self.assertEqual(first.identifier, reordered.identifier)
        self.assertEqual(with_explicit_zero.identifier, without_explicit_zero.identifier)

    def test_constraint_python_api_rejects_non_string_relation_keys(self):
        with self.assertRaises(TransformConstraintError):
            ChargeConstraintSpec(
                dimension=3,
                dependent_densities={"nS": {"nB": 0.1, 7: 0.2}},
            )


class GridTests(EoSTestCase):
    @staticmethod
    def two_dimensional_fields():
        return {
            "extra": field_spec("auxiliary", 4),
            "nB": field_spec("target", 3),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "T": field_spec("coordinate", 0),
        }

    @staticmethod
    def full_nonuniform_table():
        rows = []
        for temperature_index, temperature in enumerate([0.1, 0.3]):
            for chemical_index, chemical_potential in enumerate([0.0, 0.07, 0.2]):
                marker = 100 * temperature_index + chemical_index
                rows.append([temperature, chemical_potential, marker, -marker, marker + 0.5])
        return np.asarray(rows)

    def test_shuffled_complete_nonuniform_grid_is_reordered(self):
        table = self.full_nonuniform_table()[[4, 0, 5, 2, 1, 3]]
        eos = self.make_eos(metadata(self.two_dimensional_fields(), 2), table)

        report = eos.validate()

        self.assertTrue(report.is_valid)
        self.assertTrue(eos.grid.was_reordered)
        self.assertEqual(eos.shape, (2, 3))
        np.testing.assert_allclose(eos.axes[0], [0.1, 0.3])
        np.testing.assert_allclose(eos.axes[1], [0.0, 0.07, 0.2])
        np.testing.assert_allclose(eos.field("e"), [[0, 1, 2], [100, 101, 102]])
        np.testing.assert_allclose(eos.field("extra"), [[0.5, 1.5, 2.5], [100.5, 101.5, 102.5]])
        self.assert_issue(report, "grid.nonuniform")
        self.assert_issue(report, "grid.rows_reordered")

    def test_hole_is_rejected(self):
        table = np.delete(self.full_nonuniform_table(), 1, axis=0)
        eos = self.make_eos(metadata(self.two_dimensional_fields(), 2), table)

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assertFalse(eos.is_validated)
        self.assert_issue(report, "grid.missing_cells")

    def test_duplicate_coordinate_is_rejected(self):
        table = np.vstack([self.full_nonuniform_table(), self.full_nonuniform_table()[0]])
        eos = self.make_eos(metadata(self.two_dimensional_fields(), 2), table)

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "grid.duplicate_coordinates")

    def test_uniformity_is_policy_not_rectilinearity(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        table = [[0.1, 1], [0.2, 2], [0.4, 3]]
        accepted = self.make_eos(metadata(fields, 1, require_uniform=False), table)
        rejected = self.make_eos(metadata(fields, 1, require_uniform=True), table)

        self.assertTrue(accepted.validate().is_valid)
        report = rejected.validate()
        self.assertFalse(report.is_valid)
        self.assert_issue(report, "grid.nonuniform")

    def test_coordinate_tolerance_snaps_reported_canonical_coordinates(self):
        fields = self.two_dimensional_fields()
        table = [
            [0.1, 0.0, 1, 1, 1],
            [0.1000000000005, 0.2, 2, 2, 2],
            [0.3, 0.0, 3, 3, 3],
            [0.3, 0.2, 4, 4, 4],
        ]
        eos = self.make_eos(metadata(fields, 2), table)

        report = eos.validate()

        self.assertTrue(report.is_valid)
        self.assert_issue(report, "grid.coordinates_clustered")
        np.testing.assert_allclose(eos.field("T")[0], eos.axes[0][0])
        axis_stats = report.statistics["grid"]["axes"]["T"]
        self.assertGreater(axis_stats["clustered_rows"], 0)
        self.assertLessEqual(axis_stats["maximum_cluster_distance"], 1e-12)

    def test_coordinate_values_outside_tolerance_do_not_merge(self):
        fields = self.two_dimensional_fields()
        table = [
            [0.1, 0.0, 1, 1, 1],
            [0.10000000002, 0.2, 2, 2, 2],
            [0.3, 0.0, 3, 3, 3],
            [0.3, 0.2, 4, 4, 4],
        ]
        eos = self.make_eos(metadata(fields, 2), table)

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "grid.missing_cells")

    def test_tolerance_clustering_does_not_drift_from_its_anchor(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        eos = self.make_eos(metadata(fields, 1), [[0.1, 1], [0.2, 2]])
        values = np.asarray([0.0, 0.09, 0.14, 0.17, 0.19])

        axis, inverse = eos._cluster_axis(values, 0.1, 0.0)

        self.assertEqual(axis.size, 2)
        self.assertLessEqual(float(np.max(np.abs(values - axis[inverse]))), 0.1)

    def test_all_supported_dimensions_use_generic_cartesian_logic(self):
        coordinate_names = ("T", "muB", "muQ", "muS")
        target_names = ("e", "nB", "nQ", "nS")
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                names = coordinate_names[:dimension] + target_names[:dimension]
                fields = {
                    name: field_spec(
                        "coordinate" if index < dimension else "target", index
                    )
                    for index, name in enumerate(names)
                }
                rows = []
                for coordinate in itertools.product([0.1, 0.2], repeat=dimension):
                    rows.append(list(coordinate) + [sum(coordinate) + i for i in range(dimension)])
                eos = self.make_eos(
                    metadata(fields, dimension, expected_shape=[2] * dimension), rows
                )
                self.assertTrue(eos.validate().is_valid)
                self.assertEqual(eos.shape, (2,) * dimension)


class SnapshotFingerprintTests(EoSTestCase):
    def test_fingerprint_is_path_and_source_row_order_independent(self):
        fields = GridTests.two_dimensional_fields()
        document = metadata(fields, 2)
        rows = GridTests.full_nonuniform_table()
        first_fixture = TemporaryEoS(document, rows)
        second_fixture = TemporaryEoS(document, rows[[4, 0, 5, 2, 1, 3]])
        self.addCleanup(first_fixture.close)
        self.addCleanup(second_fixture.close)
        first = ForwardTable.load_validated(
            first_fixture.metadata_path, _allow_unsuitable_geometry=True
        )
        second = ForwardTable.load_validated(
            second_fixture.metadata_path, _allow_unsuitable_geometry=True
        )

        self.assertEqual(first.snapshot_fingerprint(), second.snapshot_fingerprint())
        self.assertEqual(first.snapshot_fingerprint(), first.snapshot_fingerprint())

    def test_fingerprint_changes_with_canonical_data_or_metadata(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        document = metadata(fields, 1)
        baseline = self.make_eos(document, [[0.1, 1.0], [0.2, 2.0]])
        changed_data = self.make_eos(
            document, [[0.1, 1.0], [0.2, 2.5]]
        )
        changed_document = json.loads(json.dumps(document))
        changed_document["table_revision"] = "test-2"
        changed_metadata = self.make_eos(
            changed_document, [[0.1, 1.0], [0.2, 2.0]]
        )
        for table in (baseline, changed_data, changed_metadata):
            self.assertTrue(table.validate().is_valid)

        self.assertNotEqual(
            baseline.snapshot_fingerprint(), changed_data.snapshot_fingerprint()
        )
        self.assertNotEqual(
            baseline.snapshot_fingerprint(), changed_metadata.snapshot_fingerprint()
        )


class FirstOrderMetadataTests(EoSTestCase):
    @staticmethod
    def fields():
        return {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "phase": field_spec("auxiliary", 4),
        }

    @staticmethod
    def table():
        return [
            [0.1, 0.0, 1.0, 0.0, 0.0],
            [0.1, 0.5, 1.5, 0.1, 1.0],
            [0.2, 0.0, 2.0, 0.0, 0.0],
            [0.2, 0.5, 2.5, 0.2, 1.0],
        ]

    @classmethod
    def document(cls, representation="line"):
        document = metadata(cls.fields(), 2, expected_shape=[2, 2])
        critical_points = []
        transition = {
            "present": True,
            "representation": representation,
            "kind": "synthetic transition",
            "coexistence_data": {
                "file": "transition.dat",
                "delimiter": "whitespace",
                "comment_prefix": "#",
                "skip_rows": 0,
                "encoding": "utf-8",
            },
        }
        if representation == "line":
            critical_points = [
                {
                    "name": "synthetic critical point",
                    "status": "known",
                    "location": {
                        "T": {"value": 150.0, "unit": "MeV"},
                        "muB": {"value": 0.0, "unit": "GeV"},
                    },
                }
            ]
            transition.update(
                {
                    "critical_point": "synthetic critical point",
                    "critical_point_tolerance": {
                        "absolute": 1e-12,
                        "relative": 1e-10,
                    },
                }
            )
            transition["coexistence_data"].update(
                {
                    "expected_columns": 2,
                    "fields": ["muB", "T"],
                    "columns": {
                        "muB": {"index": 0, "unit": "GeV"},
                        "T": {"index": 1, "unit": "MeV"},
                    },
                }
            )
        else:
            transition.update(
                {
                    "phase_fraction_field": "phase",
                    "mixed_phase_condition": "0 < phase < 1",
                }
            )
            transition["coexistence_data"].update(
                {
                    "expected_columns": 3,
                    "fields": ["muB", "T_dilute", "T_dense"],
                    "columns": {
                        "muB": {"index": 0, "unit": "MeV"},
                        "T_dilute": {"index": 1, "unit": "MeV"},
                        "T_dense": {"index": 2, "unit": "MeV"},
                    },
                }
            )
        document["scientific"]["phase_structure"] = {
            "status": "known",
            "critical_points": critical_points,
            "first_order_transition": transition,
        }
        return document

    def make_transition_eos(self, document, sidecar):
        fixture = TemporaryEoS(document, self.table())
        self.addCleanup(fixture.close)
        fixture.write_sidecar("transition.dat", sidecar)
        return ForwardTable(fixture.metadata_path)

    def test_line_metadata_loads_canonical_data_and_critical_endpoint(self):
        eos = self.make_transition_eos(
            self.document("line"), "0.0 150.0\n0.5 120.0\n"
        )

        report = eos.validate()

        self.assertTrue(report.is_valid)
        self.assertIsInstance(eos.first_order_transition, FirstOrderTransitionSpec)
        self.assertEqual(eos.first_order_transition.representation, "line")
        self.assertIsInstance(eos.coexistence_table, CoexistenceTable)
        self.assertEqual(eos.coexistence_table.names, ("muB", "T"))
        self.assertEqual(eos.coexistence_table.units, ("GeV", "GeV"))
        np.testing.assert_allclose(
            eos.coexistence_table.values, [[0.0, 0.15], [0.5, 0.12]]
        )
        self.assertEqual(eos.coexistence_table.critical_endpoint, "first")
        self.assertEqual(len(eos.coexistence_table.fingerprint), 64)
        self.assertEqual(
            report.statistics["first_order_transition"]["critical_endpoint"],
            "first",
        )

    def test_binodals_convert_mev_semantic_columns(self):
        eos = self.make_transition_eos(
            self.document("binodals"),
            "0 120 125\n500 110 130\n",
        )

        report = eos.validate()

        self.assertTrue(report.is_valid)
        self.assertEqual(eos.coexistence_table.representation, "binodals")
        self.assertEqual(
            eos.coexistence_table.names, ("muB", "T_dilute", "T_dense")
        )
        np.testing.assert_allclose(
            eos.coexistence_table.values,
            [[0.0, 0.12, 0.125], [0.5, 0.11, 0.13]],
        )
        self.assertIsNone(eos.coexistence_table.critical_endpoint)

    def test_snapshot_fingerprint_changes_with_sidecar_content(self):
        first = self.make_transition_eos(
            self.document("line"), "0.0 150.0\n0.5 120.0\n"
        )
        second = self.make_transition_eos(
            self.document("line"), "0.0 150.0\n0.5 121.0\n"
        )
        self.assertTrue(first.validate().is_valid)
        self.assertTrue(second.validate().is_valid)

        self.assertNotEqual(
            first.coexistence_table.fingerprint,
            second.coexistence_table.fingerprint,
        )
        self.assertNotEqual(first.snapshot_fingerprint(), second.snapshot_fingerprint())

    def test_missing_sidecar_is_rejected(self):
        fixture = TemporaryEoS(self.document("line"), self.table())
        self.addCleanup(fixture.close)

        report = ForwardTable(fixture.metadata_path).validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "scientific.coexistence_file_missing")

    def test_malformed_semantic_column_metadata_is_rejected(self):
        variants = []
        unsupported = self.document("line")
        unsupported["scientific"]["phase_structure"]["first_order_transition"][
            "representation"
        ] = "LINE"
        variants.append((unsupported, "scientific.first_order_representation"))

        missing = self.document("line")
        del missing["scientific"]["phase_structure"]["first_order_transition"][
            "coexistence_data"
        ]["columns"]["T"]
        variants.append((missing, "scientific.coexistence_semantic_columns"))

        duplicate = self.document("line")
        duplicate["scientific"]["phase_structure"]["first_order_transition"][
            "coexistence_data"
        ]["columns"]["T"]["index"] = 0
        variants.append((duplicate, "scientific.coexistence_duplicate_column"))

        outside = self.document("line")
        outside["scientific"]["phase_structure"]["first_order_transition"][
            "coexistence_data"
        ]["columns"]["T"]["index"] = 2
        variants.append((outside, "scientific.coexistence_column_index"))

        unit = self.document("line")
        unit["scientific"]["phase_structure"]["first_order_transition"][
            "coexistence_data"
        ]["columns"]["T"]["unit"] = "kelvin"
        variants.append((unit, "scientific.coexistence_column_unit"))

        for document, code in variants:
            with self.subTest(code=code):
                eos = self.make_transition_eos(document, "0.0 150.0\n0.5 120.0\n")
                report = eos.validate()
                self.assertFalse(report.is_valid)
                self.assert_issue(report, code)

    def test_first_order_transition_is_rejected_outside_dimension_two(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        document = metadata(fields, 1)
        document["scientific"]["phase_structure"] = {
            "status": "known",
            "critical_points": [],
            "first_order_transition": {
                "present": True,
                "representation": "binodals",
                "kind": "invalid dimensional transition",
                "coexistence_data": {
                    "file": "transition.dat",
                    "expected_columns": 3,
                    "columns": {
                        "muB": {"index": 0, "unit": "GeV"},
                        "T_dilute": {"index": 1, "unit": "GeV"},
                        "T_dense": {"index": 2, "unit": "GeV"},
                    },
                },
            },
        }
        fixture = TemporaryEoS(document, [[0.1, 1.0], [0.2, 2.0]])
        self.addCleanup(fixture.close)
        fixture.write_sidecar("transition.dat", "0.0 0.12 0.13\n0.1 0.11 0.14\n")

        report = ForwardTable(fixture.metadata_path).validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "scientific.first_order_dimension")

    def test_sidecar_value_failures_are_distinguished(self):
        cases = (
            (
                "0.0 150.0\n0.0 120.0\n",
                "scientific.coexistence_muB_duplicate",
            ),
            (
                "0.5 150.0\n0.0 120.0\n",
                "scientific.coexistence_muB_order",
            ),
            (
                "0.0 150.0\n0.5 nan\n",
                "scientific.coexistence_nonfinite",
            ),
            (
                "0.0 150.0\n0.6 120.0\n",
                "scientific.coexistence_out_of_domain",
            ),
            (
                "0.0 150.0 1.0\n0.5 120.0 2.0\n",
                "scientific.coexistence_column_count",
            ),
        )
        for sidecar, code in cases:
            with self.subTest(code=code):
                eos = self.make_transition_eos(self.document("line"), sidecar)
                report = eos.validate()
                self.assertFalse(report.is_valid)
                self.assert_issue(report, code)

    def test_critical_point_must_match_one_line_endpoint(self):
        document = self.document("line")
        document["scientific"]["phase_structure"]["critical_points"][0][
            "location"
        ]["T"]["value"] = 140.0
        eos = self.make_transition_eos(document, "0.0 150.0\n0.5 120.0\n")

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "scientific.coexistence_critical_endpoint")


class NonfiniteTests(EoSTestCase):
    def test_explicitly_allowed_auxiliary_nan_is_retained(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "optional": field_spec("auxiliary", 2, nonfinite_policy="allow_nan"),
        }
        eos = self.make_eos(metadata(fields, 1), [[0.1, 1, np.nan], [0.2, 2, 3]])

        report = eos.validate()

        self.assertTrue(report.is_valid)
        self.assertTrue(np.isnan(eos.field("optional", shaped=False)[0]))
        self.assert_issue(report, "field.nan_allowed")
        self.assertEqual(report.statistics["fields"]["optional"]["nan_count"], 1)

    def test_allow_nan_does_not_allow_infinity(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "optional": field_spec("auxiliary", 2, nonfinite_policy="allow_nan"),
        }
        eos = self.make_eos(metadata(fields, 1), [[0.1, 1, np.inf], [0.2, 2, 3]])

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "field.nonfinite.auxiliary.infinity")

    def test_target_nan_is_rejected(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        eos = self.make_eos(metadata(fields, 1), [[0.1, np.nan], [0.2, 2]])

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "field.nonfinite.target.nan")


class BoundaryTests(EoSTestCase):
    @staticmethod
    def fields():
        return {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }

    @staticmethod
    def table():
        return [
            [0.1, 0.0, 1.0, -2.0],
            [0.1, 0.2, 4.0, 3.0],
            [0.3, 0.0, 9.0, -1.0],
            [0.3, 0.2, 16.0, 5.0],
        ]

    def test_target_boundaries_are_tight_component_extrema(self):
        eos = self.make_eos(metadata(self.fields(), 2), self.table())

        bounds = eos.find_boundaries(space="target")

        self.assertTrue(eos.is_validated)
        self.assertEqual(bounds.names, ("e", "nB"))
        self.assertEqual(bounds.units, ("GeV^4", "GeV^3"))
        np.testing.assert_allclose(bounds.ranges, [[1.0, 16.0], [-2.0, 5.0]])
        with self.assertRaises(ValueError):
            bounds.lower.setflags(write=True)

    def test_tilde_boundaries_transform_every_target_point_before_extrema(self):
        eos = self.make_eos(metadata(self.fields(), 2), self.table())
        targets = np.asarray(self.table())[:, 2:4]
        transformed, names = targets_to_tilde(targets, 2)

        bounds = eos.find_boundaries(space="tilde")

        self.assertEqual(bounds.names, names)
        self.assertEqual(bounds.units, ("GeV", "GeV"))
        np.testing.assert_allclose(bounds.lower, np.min(transformed, axis=0))
        np.testing.assert_allclose(bounds.upper, np.max(transformed, axis=0))
        self.assertIsNotNone(bounds.transform_id)
        self.assertEqual(bounds.coverage, "validated_grid_nodes")

    def test_unsupported_space_is_rejected(self):
        eos = self.make_eos(metadata(self.fields(), 2), self.table())
        with self.assertRaises(ValueError):
            eos.find_boundaries(space="convex_hull")

    def test_degenerate_target_axis_is_rejected_by_default(self):
        table = self.table()
        for row in table:
            row[3] = 0.0
        eos = self.make_eos(metadata(self.fields(), 2), table)

        with self.assertRaises(BoundaryComputationError):
            eos.find_boundaries(space="target")
        envelope = eos.find_boundaries(
            space="target", require_non_degenerate=False
        )
        self.assertEqual(envelope.degenerate_names, ("nB",))

    def test_default_3d_tilde_boundaries_use_nS_zero_constraint(self):
        coordinate_names = ("T", "muB", "muQ")
        target_names = ("e", "nB", "nQ")
        names = coordinate_names + target_names
        fields = {
            name: field_spec("coordinate" if index < 3 else "target", index)
            for index, name in enumerate(names)
        }
        rows = [
            list(coordinate) + [1.0, 0.1, 0.2]
            for coordinate in itertools.product([0.1, 0.2], repeat=3)
        ]
        eos = self.make_eos(metadata(fields, 3), rows)

        bounds = eos.find_boundaries(
            space="tilde", require_non_degenerate=False
        )

        self.assertEqual(bounds.names, ("T_tilde", "muB_tilde", "muQ_tilde"))
        self.assertEqual(bounds.dependent_names, ("muS_tilde",))
        relation = bounds.dependent_relations["muS_tilde"]
        self.assertAlmostEqual(relation["muB_tilde"], 1.0 / 3.0)
        self.assertAlmostEqual(relation["muQ_tilde"], -1.0 / 3.0)
        serialized = bounds.to_dict()
        self.assertEqual(
            serialized["dependent_coordinates"]["muS_tilde"]["kind"],
            "derived_output",
        )

    def test_tilde_transform_domain_checks_energy(self):
        with self.assertRaises(TransformDomainError):
            targets_to_tilde(np.asarray([[-1.0, 0.0]]), 2)
        with self.assertRaises(TransformDomainError):
            targets_to_tilde(np.asarray([[0.0, 0.0]]), 2)
        transformed, _ = targets_to_tilde(np.asarray([[0.0]]), 1)
        np.testing.assert_allclose(transformed, [[0.0]])


class StateAndMetadataTests(EoSTestCase):
    def test_minimal_metadata_uses_documented_deterministic_defaults(self):
        table = ForwardTable.load_validated(
            REPOSITORY_ROOT / "examples/regular/metadata.json"
        )
        assert table.metadata is not None

        self.assertEqual(table.metadata.schema_version, 1)
        self.assertIsNone(table.metadata.table_revision)
        self.assertIsNone(table.metadata.format.delimiter)
        self.assertEqual(table.metadata.format.comment_prefix, "#")
        self.assertEqual(table.metadata.format.skip_rows, 0)
        self.assertEqual(table.metadata.format.encoding, "utf-8")
        self.assertEqual(table.metadata.grid.kind, "rectilinear")
        self.assertTrue(table.metadata.grid.require_complete)
        self.assertTrue(table.metadata.grid.reorder)
        self.assertFalse(table.metadata.grid.require_uniform)
        for spec in table.metadata.fields.values():
            self.assertEqual(spec.transform, TransformSpec())
            self.assertEqual(spec.nonfinite_policy, "forbid")
            self.assertEqual(spec.interpolation, "linear")

    def test_validation_state_lifecycle_and_read_only_arrays(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        eos = self.make_eos(metadata(fields, 1), [[0.1, 1], [0.2, 2]])
        self.assertFalse(eos.is_validated)

        eos.read_data()
        self.assertFalse(eos.is_validated)
        eos.interpret_data()
        self.assertFalse(eos.is_validated)
        self.assertTrue(eos.validate().is_valid)
        self.assertTrue(eos.is_validated)
        self.assertFalse(eos.field("e").flags.writeable)
        with self.assertRaises(ValueError):
            eos.field("e")[0] = -1
        for exposed in (
            eos.raw_data,
            eos.axes[0],
            eos.field("e", shaped=False),
            eos.field("e", shaped=True),
            eos.grid.row_permutation,
        ):
            with self.assertRaises(ValueError):
                exposed.setflags(write=True)

        eos.read_data()
        self.assertFalse(eos.is_validated)
        with self.assertRaises(EoSNotValidatedError):
            eos.require_validated()

    def test_invalid_json_and_duplicate_json_keys_are_reported(self):
        for document in ("{", '{"schema_version": 1, "schema_version": 1}'):
            with self.subTest(document=document):
                directory = tempfile.TemporaryDirectory()
                self.addCleanup(directory.cleanup)
                path = Path(directory.name) / "metadata.json"
                path.write_text(document, encoding="utf-8")
                report = ForwardTable(path).validate()
                self.assertFalse(report.is_valid)
                self.assert_issue(report, "metadata.invalid_json")

    def test_invalid_metadata_bytes_return_a_report(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "metadata.json"
        path.write_bytes(b"\xff\xfe")

        report = ForwardTable(path).validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "metadata.read_failed")

    def test_unknown_data_encoding_is_a_metadata_error(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        document = metadata(fields, 1)
        document["format"]["encoding"] = "definitely-not-an-encoding"
        eos = self.make_eos(document, [[0.1, 1], [0.2, 2]])

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "metadata.format_encoding")

    def test_zero_scale_factor_is_rejected(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1, scale=0.0),
        }
        eos = self.make_eos(metadata(fields, 1), [[0.1, 1], [0.2, 2]])

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "metadata.transform_scale")

    def test_canonical_units_are_enforced(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        document = metadata(fields, 1)
        document["fields"]["e"]["canonical_unit"] = "seconds"
        eos = self.make_eos(document, [[0.1, 1], [0.2, 2]])

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "metadata.canonical_unit")

    def test_extreme_json_numbers_return_diagnostics_instead_of_crashing(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        variants = []
        huge = 10**400
        scale_document = metadata(json.loads(json.dumps(fields)), 1)
        scale_document["fields"]["e"]["transform"]["scale_factor"] = huge
        variants.append((scale_document, "metadata.transform_scale"))
        power_document = metadata(json.loads(json.dumps(fields)), 1)
        power_document["fields"]["e"]["transform"]["temperature_power"] = 10**100
        variants.append((power_document, "metadata.transform_temperature_power"))
        tolerance_document = metadata(json.loads(json.dumps(fields)), 1)
        tolerance_document["grid"]["absolute_tolerance"] = huge
        variants.append((tolerance_document, "metadata.grid_tolerance"))
        columns_document = metadata(json.loads(json.dumps(fields)), 1)
        columns_document["format"]["expected_columns"] = 10_001
        columns_document["format"]["missing_values"] = ["MISSING"]
        variants.append((columns_document, "metadata.format_expected_columns"))

        for document, code in variants:
            with self.subTest(code=code):
                eos = self.make_eos(document, [[0.1, 1], [0.2, 2]])
                report = eos.validate()
                self.assertFalse(report.is_valid)
                self.assert_issue(report, code)

    def test_nul_in_data_path_is_rejected_as_metadata(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        document = metadata(fields, 1)
        document["data_file"] = "bad\x00path.dat"
        eos = self.make_eos(document, [[0.1, 1], [0.2, 2]])

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "metadata.data_file_nul")

    def test_metadata_path_cannot_change_behind_validation_state(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        eos = self.make_eos(metadata(fields, 1), [[0.1, 1], [0.2, 2]])
        self.assertTrue(eos.validate().is_valid)

        with self.assertRaises(AttributeError):
            eos.metadata_path = Path("different.json")
        self.assertTrue(eos.is_validated)

    def test_custom_missing_token_uses_field_policy(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "optional": field_spec("auxiliary", 2, nonfinite_policy="allow_nan"),
        }
        document = metadata(fields, 1)
        document["format"]["missing_values"] = ["MISSING"]
        fixture = TemporaryEoS(document, [[0.1, 1, 2], [0.2, 2, 3]])
        self.addCleanup(fixture.close)
        fixture.table_path.write_text("0.1 1 MISSING\n0.2 2 3\n", encoding="utf-8")
        eos = ForwardTable(fixture.metadata_path)

        report = eos.validate()

        self.assertTrue(report.is_valid)
        self.assertTrue(np.isnan(eos.field("optional", shaped=False)[0]))
        self.assert_issue(report, "field.nan_allowed")

    def test_malformed_scientific_metadata_is_rejected(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "phase": field_spec("auxiliary", 2),
        }
        document = metadata(fields, 1)
        document["scientific"] = {
            "origin": {"generator": 123},
            "publications": [42],
            "phase_structure": {
                "status": "known",
                "critical_points": [
                    {"name": "bad point", "status": "known", "location": "Mars"},
                    {
                        "name": "wrong coordinates",
                        "status": "known",
                        "location": {"muS": {"value": 1.0, "unit": "GeV"}},
                    },
                ],
                "first_order_transition": {
                    "present": True,
                    "kind": "test",
                    "phase_fraction_field": "misspelled",
                },
            },
        }
        eos = self.make_eos(document, [[0.1, 1, 0], [0.2, 2, 1]])

        report = eos.validate()

        self.assertFalse(report.is_valid)
        for code in (
            "scientific.origin_value",
            "scientific.publication_type",
            "scientific.critical_point_location",
            "scientific.critical_point_coordinate",
            "scientific.critical_point_coordinates_missing",
            "scientific.phase_fraction_field",
        ):
            self.assert_issue(report, code)

    def test_user_authored_validated_flag_is_rejected(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        document = metadata(fields, 1)
        document["validated"] = True
        eos = self.make_eos(document, [[0.1, 1], [0.2, 2]])

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assertFalse(eos.is_validated)
        self.assert_issue(report, "metadata.unknown_key")

    def test_omitted_charge_constraints_resolve_to_dimension_defaults(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }
        eos = self.make_eos(
            metadata(fields, 2),
            [[0.1, 0.0, 1.0, 0.0], [0.1, 0.2, 1.0, 0.1],
             [0.2, 0.0, 2.0, 0.0], [0.2, 0.2, 2.0, 0.2]],
        )

        self.assertTrue(eos.validate().is_valid)
        self.assertEqual(eos.charge_constraints.source, "default")
        self.assertEqual(
            dict(eos.charge_constraints.dependent_densities),
            {"nQ": {}, "nS": {}},
        )

    def test_declared_density_constraints_validate_auxiliary_fields(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "nQ": field_spec("auxiliary", 4),
            "nS": field_spec("auxiliary", 5),
        }
        fields["nQ"]["canonical_unit"] = "GeV^3"
        fields["nS"]["canonical_unit"] = "GeV^3"
        document = metadata(fields, 2)
        document["charge_constraints"] = {
            "model": "linear_homogeneous",
            "dependent_densities": {"nQ": {"nB": 0.4}, "nS": {}},
            "validation": {
                "absolute_tolerance": 1e-12,
                "relative_tolerance": 1e-10,
            },
        }
        table = [
            [0.1, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 1.0, 0.1, 0.04, 0.0],
            [0.2, 0.0, 2.0, 0.2, 0.08, 0.0],
            [0.2, 0.2, 2.0, 0.3, 0.12, 0.0],
        ]

        valid = self.make_eos(document, table)
        self.assertTrue(valid.validate().is_valid)
        invalid_table = np.asarray(table, dtype=float)
        invalid_table[-1, 4] = 0.2
        invalid = self.make_eos(document, invalid_table)
        report = invalid.validate()
        self.assertFalse(report.is_valid)
        self.assert_issue(report, "constraint.density_mismatch")

    def test_dependent_density_field_requires_canonical_unit(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "nQ": field_spec("auxiliary", 4),
        }
        document = metadata(fields, 2)
        eos = self.make_eos(
            document,
            [
                [0.1, 0.0, 1.0, 0.0, 0.0],
                [0.1, 0.2, 1.0, 0.1, 0.0],
                [0.2, 0.0, 2.0, 0.2, 0.0],
                [0.2, 0.2, 2.0, 0.3, 0.0],
            ],
        )

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "metadata.canonical_unit")

    def test_constraint_validation_rejects_overflowed_expected_density(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "nQ": field_spec("auxiliary", 4),
        }
        fields["nQ"]["canonical_unit"] = "GeV^3"
        document = metadata(fields, 2)
        document["charge_constraints"] = {
            "model": "linear_homogeneous",
            "dependent_densities": {"nQ": {"nB": 1e300}, "nS": {}},
        }
        eos = self.make_eos(
            document,
            [
                [0.1, 0.0, 1.0, 1e100, 0.0],
                [0.1, 0.2, 1.0, 1e100, 0.0],
                [0.2, 0.0, 2.0, 1e100, 0.0],
                [0.2, 0.2, 2.0, 1e100, 0.0],
            ],
        )

        report = eos.validate()

        self.assertFalse(report.is_valid)
        self.assert_issue(report, "constraint.nonfinite_expected")

    def test_malformed_and_singular_charge_constraints_are_rejected(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }
        variants = []
        missing = metadata(json.loads(json.dumps(fields)), 2)
        missing["charge_constraints"] = {
            "model": "linear_homogeneous",
            "dependent_densities": {"nQ": {}},
        }
        variants.append((missing, "metadata.dependent_density_missing"))
        invalid_source = metadata(json.loads(json.dumps(fields)), 2)
        invalid_source["charge_constraints"] = {
            "model": "linear_homogeneous",
            "dependent_densities": {"nQ": {"nQ": 0.4}, "nS": {}},
        }
        variants.append((invalid_source, "metadata.density_relation_source"))
        singular = metadata(json.loads(json.dumps(fields)), 2)
        singular["charge_constraints"] = {
            "model": "linear_homogeneous",
            "dependent_densities": {"nQ": {"nB": 5.0}, "nS": {}},
        }
        variants.append((singular, "metadata.charge_constraints_invalid"))

        table = [
            [0.1, 0.0, 1.0, 0.0],
            [0.1, 0.2, 1.0, 0.1],
            [0.2, 0.0, 2.0, 0.2],
            [0.2, 0.2, 2.0, 0.3],
        ]
        for document, code in variants:
            with self.subTest(code=code):
                report = self.make_eos(document, table).validate()
                self.assertFalse(report.is_valid)
                self.assert_issue(report, code)

    def test_strict_loader_raises_structured_validation_error(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        fixture = TemporaryEoS(metadata(fields, 1), [[0.1, 1], [0.1, 2]])
        self.addCleanup(fixture.close)

        with self.assertRaises(EoSValidationError) as context:
            ForwardTable.load_validated(fixture.metadata_path)
        self.assertFalse(context.exception.report.is_valid)


class RealTableIntegrationTests(unittest.TestCase):
    TEMPLATE_METADATA = REPOSITORY_ROOT / "templates" / "metadata_master.json"
    SYNTHETIC_METADATA = REPOSITORY_ROOT / "examples" / "regular" / "metadata.json"
    ISING_METADATA = REPOSITORY_ROOT / "EoSTables" / "2DIsing" / "metadata.json"
    ISING_TABLE = REPOSITORY_ROOT / "EoSTables" / "2DIsing" / "EoSmub602.dat"
    CEC_METADATA = REPOSITORY_ROOT / "EoSTables" / "CEC" / "metadata.json"
    CEC_TABLE = REPOSITORY_ROOT / "EoSTables" / "CEC" / "hic_table_T80-300_muB0-800_d1.dat"

    def test_example_metadata_schema_loads_without_table_io(self):
        for path in (self.TEMPLATE_METADATA, self.SYNTHETIC_METADATA):
            with self.subTest(path=path):
                eos = ForwardTable(path)
                loaded = eos.load_metadata()
                self.assertEqual(loaded.schema_version, 1)

    @unittest.skipUnless(ISING_TABLE.is_file(), "local gitignored Ising table is unavailable")
    def test_ising_table(self):
        eos = ForwardTable.load_validated(self.ISING_METADATA)
        self.assertEqual(eos.shape, (771, 701))
        np.testing.assert_allclose([eos.axes[0][0], eos.axes[0][-1]], [0.03, 0.8])
        np.testing.assert_allclose([eos.axes[1][0], eos.axes[1][-1]], [0.0, 0.7])
        self.assertAlmostEqual(eos.field("e", shaped=False)[0], 0.169847 * 0.03**4)
        self.assertEqual(len(eos.fields), 8)
        np.testing.assert_allclose(
            eos.find_boundaries(space="tilde").ranges,
            [[0.009686518220848856, 0.7558189702212839], [0.0, 1.2431136807478493]],
        )

    @unittest.skipUnless(CEC_TABLE.is_file(), "local gitignored CEC table is unavailable")
    def test_cec_table(self):
        eos = ForwardTable.load_validated(self.CEC_METADATA)
        self.assertEqual(eos.shape, (221, 801))
        np.testing.assert_allclose([eos.axes[0][0], eos.axes[0][-1]], [0.08, 0.3])
        np.testing.assert_allclose([eos.axes[1][0], eos.axes[1][-1]], [0.0, 0.8])
        self.assertAlmostEqual(
            eos.field("e", shaped=False)[0], 4.31173e-3 * HBARC_GEV_FM**3
        )
        for name in ("cs2", "chi2B", "chi3B", "chi4B"):
            self.assertEqual(eos.report.statistics["fields"][name]["nan_count"], 246)
        np.testing.assert_allclose(
            eos.find_boundaries(space="target").ranges,
            [
                [3.3129201468981235e-05, 0.09646180232115137],
                [0.0, 0.017419966992010066],
            ],
        )
        tilde = eos.find_boundaries(space="tilde")
        np.testing.assert_allclose(
            tilde.ranges,
            [
                [0.038157922088400416, 0.2802985285900713],
                [0.0, 1.0199139301960334],
            ],
        )
        self.assertEqual(tilde.dependent_names, ("muQ_tilde", "muS_tilde"))
        self.assertAlmostEqual(
            tilde.dependent_relations["muQ_tilde"]["muB_tilde"], -1.0 / 23.0
        )
        self.assertAlmostEqual(
            tilde.dependent_relations["muS_tilde"]["muB_tilde"], 8.0 / 23.0
        )
        constraint_stats = eos.report.statistics["charge_constraints"]["relations"]
        self.assertEqual(constraint_stats["nQ"]["mismatch_count"], 0)
        self.assertEqual(constraint_stats["nS"]["mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
