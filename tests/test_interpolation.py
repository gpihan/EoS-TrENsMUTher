"""Tests for boundary-safe multilinear interpolation."""

import itertools
import json
import unittest
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    EoSNotValidatedError,
    ForwardTable,
    InterpolationCellError,
    InterpolationConfigurationError,
    InterpolationDerivativeError,
    InterpolationDomainError,
    InterpolationInputError,
    MultilinearInterpolator,
)

from tests.test_tables import TemporaryEoS, field_spec, metadata


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COORDINATE_NAMES = ("T", "muB", "muQ", "muS")
TARGET_NAMES = ("e", "nB", "nQ", "nS")
TEST_AXES = (
    np.asarray([0.08, 0.17, 0.43]),
    np.asarray([-0.35, 0.06, 0.71]),
    np.asarray([-0.18, 0.24, 0.83]),
    np.asarray([-0.42, -0.03, 0.52]),
)


def multiaffine_values(points, dimension, field_count):
    """Evaluate deterministic full multi-affine polynomials."""

    points = np.asarray(points, dtype=float)
    values = np.empty(points.shape[:-1] + (field_count,), dtype=float)
    for field_index in range(field_count):
        result = np.full(points.shape[:-1], 4.0 + field_index, dtype=float)
        for mask in range(1, 1 << dimension):
            coefficient = 0.037 * (field_index + 1) * (mask + 2)
            if (bin(mask).count("1") + field_index) % 2:
                coefficient *= -1.0
            term = np.ones(points.shape[:-1], dtype=float)
            for axis_index in range(dimension):
                if mask & (1 << axis_index):
                    term *= points[..., axis_index]
            result += coefficient * term
        values[..., field_index] = result
    return values


def multiaffine_jacobians(points, dimension, field_count):
    points = np.asarray(points, dtype=float)
    result = np.zeros(
        points.shape[:-1] + (field_count, dimension), dtype=float
    )
    for field_index in range(field_count):
        for differentiated_axis in range(dimension):
            derivative = np.zeros(points.shape[:-1], dtype=float)
            for mask in range(1, 1 << dimension):
                if not mask & (1 << differentiated_axis):
                    continue
                coefficient = 0.037 * (field_index + 1) * (mask + 2)
                if (bin(mask).count("1") + field_index) % 2:
                    coefficient *= -1.0
                term = np.ones(points.shape[:-1], dtype=float)
                for axis_index in range(dimension):
                    if (
                        axis_index != differentiated_axis
                        and mask & (1 << axis_index)
                    ):
                        term *= points[..., axis_index]
                derivative += coefficient * term
            result[..., field_index, differentiated_axis] = derivative
    return result


class InterpolationTestCase(unittest.TestCase):
    def make_eos(self, document, table):
        fixture = TemporaryEoS(document, table)
        self.addCleanup(fixture.close)
        return ForwardTable(
            fixture.metadata_path, _allow_unsuitable_geometry=True
        )

    def make_multiaffine_eos(self, dimension, *, shuffle=True):
        coordinate_names = COORDINATE_NAMES[:dimension]
        target_names = TARGET_NAMES[:dimension]
        fields = {}
        for column, name in enumerate(coordinate_names + target_names):
            role = "coordinate" if name in coordinate_names else "target"
            fields[name] = field_spec(role, column)
        axes = TEST_AXES[:dimension]
        points = np.asarray(list(itertools.product(*axes)), dtype=float)
        values = multiaffine_values(points, dimension, dimension)
        rows = np.column_stack((points, values))
        if shuffle:
            permutation = np.random.RandomState(100 + dimension).permutation(len(rows))
            rows = rows[permutation]
        eos = self.make_eos(metadata(fields, dimension), rows)
        self.assertTrue(eos.validate().is_valid)
        return eos


class AnalyticInterpolationTests(InterpolationTestCase):
    def test_values_and_jacobians_are_exact_in_dimensions_one_through_four(self):
        random = np.random.RandomState(1729)
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                eos = self.make_multiaffine_eos(dimension)
                interpolator = MultilinearInterpolator(eos, chunk_size=3)
                lower = np.asarray([axis[0] for axis in TEST_AXES[:dimension]])
                upper = np.asarray([axis[-1] for axis in TEST_AXES[:dimension]])
                points = random.uniform(lower, upper, size=(17, dimension))

                expected_values = multiaffine_values(points, dimension, dimension)
                expected_jacobians = multiaffine_jacobians(
                    points, dimension, dimension
                )
                evaluation = interpolator.evaluate_with_jacobian(points)

                np.testing.assert_allclose(
                    evaluation.values, expected_values, rtol=2e-14, atol=2e-14
                )
                np.testing.assert_allclose(
                    evaluation.jacobians,
                    expected_jacobians,
                    rtol=3e-14,
                    atol=3e-14,
                )
                self.assertEqual(evaluation.values.shape, (17, dimension))
                self.assertEqual(
                    evaluation.jacobians.shape, (17, dimension, dimension)
                )
                self.assertTrue(np.all(evaluation.values_defined))
                self.assertTrue(np.all(evaluation.jacobians_defined))

                single = points[0]
                self.assertEqual(interpolator(single).shape, (dimension,))
                self.assertEqual(
                    interpolator.jacobian(single).shape, (dimension, dimension)
                )
                shaped = points[:6].reshape(2, 3, dimension)
                self.assertEqual(
                    interpolator(shaped).shape, (2, 3, dimension)
                )
                self.assertEqual(
                    interpolator.jacobian(shaped).shape,
                    (2, 3, dimension, dimension),
                )

    def test_grid_nodes_and_empty_batches_have_stable_shapes(self):
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                eos = self.make_multiaffine_eos(dimension)
                interpolator = MultilinearInterpolator(eos)
                nodes = np.asarray(list(itertools.product(*eos.axes)))
                np.testing.assert_allclose(
                    interpolator(nodes),
                    multiaffine_values(nodes, dimension, dimension),
                    rtol=2e-14,
                    atol=2e-14,
                )
                empty = np.empty((0, dimension))
                self.assertEqual(interpolator(empty).shape, (0, dimension))
                self.assertEqual(
                    interpolator.jacobian(empty).shape,
                    (0, dimension, dimension),
                )


class InMemoryRectilinearMapTests(InterpolationTestCase):
    def make_map(self, dimension=2):
        axes = tuple(np.array(axis, copy=True) for axis in TEST_AXES[:dimension])
        coordinate_names = COORDINATE_NAMES[:dimension]
        target_names = TARGET_NAMES[:dimension]
        points = np.asarray(list(itertools.product(*axes)), dtype=float)
        grid_values = multiaffine_values(
            points, dimension, dimension
        ).reshape(tuple(axis.size for axis in axes) + (dimension,))
        fields = {
            name: np.array(grid_values[..., index], copy=True)
            for index, name in enumerate(target_names)
        }
        interpolator = MultilinearInterpolator.from_rectilinear_map(
            axes,
            fields,
            coordinate_names=coordinate_names,
            target_names=target_names,
            coordinate_units=("GeV",) * dimension,
            target_units=("1",) * dimension,
            chunk_size=3,
        )
        return axes, fields, interpolator

    def test_in_memory_maps_reuse_exact_value_jacobian_and_bounds_code(self):
        random = np.random.RandomState(8128)
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                axes, fields, interpolator = self.make_map(dimension)
                lower = np.asarray([axis[0] for axis in axes])
                upper = np.asarray([axis[-1] for axis in axes])
                points = random.uniform(lower, upper, size=(11, dimension))
                evaluation = interpolator.evaluate_with_jacobian(points)

                np.testing.assert_allclose(
                    evaluation.values,
                    multiaffine_values(points, dimension, dimension),
                    rtol=3.0e-13,
                    atol=3.0e-14,
                )
                np.testing.assert_allclose(
                    evaluation.jacobians,
                    multiaffine_jacobians(points, dimension, dimension),
                    rtol=3.0e-12,
                    atol=3.0e-13,
                )
                self.assertEqual(interpolator.field_names, TARGET_NAMES[:dimension])
                self.assertEqual(interpolator.target_names, TARGET_NAMES[:dimension])
                self.assertEqual(interpolator.shape, tuple(axis.size for axis in axes))
                self.assertTrue(interpolator.node_value_bounds.rigorous)
                self.assertTrue(np.all(interpolator.node_value_bounds.defined))

                first_cell = np.zeros(dimension, dtype=np.int64)
                enclosure = interpolator.cell_value_enclosures(first_cell)
                corners = interpolator.cell_corner_values(first_cell)
                self.assertTrue(enclosure.rigorous)
                np.testing.assert_array_equal(
                    enclosure.lower, np.min(corners, axis=0)
                )
                np.testing.assert_array_equal(
                    enclosure.upper, np.max(corners, axis=0)
                )

                expected = interpolator(points)
                for axis in axes:
                    axis[:] = 123.0
                for field in fields.values():
                    field[:] = -456.0
                np.testing.assert_array_equal(interpolator(points), expected)
                self.assertFalse(interpolator.axes[0].flags.writeable)

    def test_in_memory_constructor_rejects_non_square_or_incomplete_maps(self):
        axes = ([0.0, 1.0], [0.0, 1.0])
        values = {
            "e": np.zeros((2, 2)),
            "nB": np.ones((2, 2)),
        }
        common = {
            "coordinate_names": ("muB", "lambda"),
            "target_names": ("e", "nB"),
            "coordinate_units": ("GeV", "1"),
            "target_units": ("GeV^4", "GeV^3"),
        }

        invalid_cases = (
            ({**common, "target_names": ("e",)}, values),
            (common, {"e": values["e"]}),
            (common, {**values, "extra": values["e"]}),
            (common, {"e": np.zeros((3, 2)), "nB": values["nB"]}),
            (common, {"e": np.full((2, 2), np.nan), "nB": values["nB"]}),
        )
        for keywords, supplied_values in invalid_cases:
            with self.subTest(keywords=keywords, fields=tuple(supplied_values)):
                with self.assertRaises(InterpolationConfigurationError):
                    MultilinearInterpolator.from_rectilinear_map(
                        axes, supplied_values, **keywords
                    )

        with self.assertRaises(InterpolationConfigurationError):
            MultilinearInterpolator.from_rectilinear_map(
                ([0.0, 1.0], [1.0, 0.0]), values, **common
            )

    def test_requested_field_order_and_convenience_accessors(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "zeta": field_spec("auxiliary", 2),
            "alpha": field_spec("auxiliary", 3),
        }
        document = metadata(fields, 1)
        rows = [[0.1, 1.0, 10.0, -2.0], [0.3, 3.0, 30.0, 4.0]]
        eos = self.make_eos(document, rows)
        self.assertTrue(eos.validate().is_valid)

        targets = MultilinearInterpolator(eos)
        self.assertEqual(targets.field_names, ("e",))
        self.assertEqual(targets.coordinate_names, ("T",))
        self.assertEqual(targets.coordinate_units, ("GeV",))
        self.assertEqual(targets.field_units, ("GeV^4",))

        outputs = MultilinearInterpolator(eos, ("zeta", "alpha", "e"))
        self.assertEqual(outputs.field_names, ("zeta", "alpha", "e"))
        np.testing.assert_allclose(outputs([0.2]), [20.0, 1.0, 2.0])
        self.assertAlmostEqual(outputs.evaluate_field("alpha", [0.2]), 1.0)
        mapped = outputs.evaluate_dict([[0.1], [0.3]])
        self.assertEqual(tuple(mapped), ("zeta", "alpha", "e"))
        np.testing.assert_allclose(mapped["e"], [1.0, 3.0])


class BoundaryAndLocationTests(InterpolationTestCase):
    def make_piecewise_eos(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
        }
        eos = self.make_eos(
            metadata(fields, 1),
            [[0.1, 0.0], [0.2, 1.0], [0.5, 5.0]],
        )
        self.assertTrue(eos.validate().is_valid)
        return eos

    def test_closed_domain_and_deterministic_one_sided_grid_plane_behavior(self):
        interpolator = MultilinearInterpolator(self.make_piecewise_eos())
        np.testing.assert_allclose(interpolator([[0.1], [0.5]]), [[0.0], [5.0]])
        self.assertTrue(interpolator.contains([0.1]))
        self.assertTrue(interpolator.contains([0.5]))

        right = interpolator.locate([0.2])
        left = interpolator.locate([0.2], side="left")
        np.testing.assert_array_equal(right.cell_indices, [1])
        np.testing.assert_allclose(right.normalized_coordinates, [0.0])
        np.testing.assert_array_equal(left.cell_indices, [0])
        np.testing.assert_allclose(left.normalized_coordinates, [1.0])
        self.assertTrue(right.on_internal_grid_plane[0])
        self.assertAlmostEqual(interpolator.jacobian([0.2])[0, 0], 4.0 / 0.3)
        self.assertAlmostEqual(
            interpolator.jacobian([0.2], plane_policy="left")[0, 0], 10.0
        )
        with self.assertRaises(InterpolationDerivativeError):
            interpolator.jacobian([0.2], plane_policy="raise")

        minimum = interpolator.locate([0.1])
        maximum = interpolator.locate([0.5])
        np.testing.assert_array_equal(minimum.cell_indices, [0])
        np.testing.assert_allclose(minimum.normalized_coordinates, [0.0])
        np.testing.assert_array_equal(maximum.cell_indices, [1])
        np.testing.assert_allclose(maximum.normalized_coordinates, [1.0])
        self.assertTrue(minimum.on_domain_boundary[0])
        self.assertTrue(maximum.on_domain_boundary[0])

    def test_per_coordinate_sides_cover_mixed_cells_at_a_knot_intersection(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }
        rows = []
        for temperature in (0.1, 0.2, 0.4):
            for chemical_potential in (-0.2, 0.0, 0.3):
                rows.append(
                    [
                        temperature,
                        chemical_potential,
                        temperature**2 + chemical_potential**2,
                        temperature * chemical_potential,
                    ]
                )
        eos = self.make_eos(metadata(fields, 2), rows)
        self.assertTrue(eos.validate().is_valid)
        interpolator = MultilinearInterpolator(eos)
        point = [0.2, 0.0]

        left_right = interpolator.locate(point, side=("left", "right"))
        right_left = interpolator.locate(point, side=("right", "left"))
        np.testing.assert_array_equal(left_right.cell_indices, [0, 1])
        np.testing.assert_array_equal(right_left.cell_indices, [1, 0])
        self.assertEqual(left_right.sides, ("left", "right"))
        self.assertEqual(right_left.sides, ("right", "left"))
        np.testing.assert_allclose(
            interpolator.jacobian(point, plane_policy=("left", "right"))[0],
            [0.3, 0.3],
        )
        np.testing.assert_allclose(
            interpolator.jacobian(point, plane_policy=("right", "left"))[0],
            [0.6, -0.2],
        )

    def test_exact_upper_face_does_not_activate_an_off_face_nan(self):
        first = 9_918_737.534611896
        middle = 12_674_648.858918732
        last = middle + 1_000_000.0
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "optional": field_spec(
                "auxiliary", 2, nonfinite_policy="allow_nan"
            ),
        }

        upper_eos = self.make_eos(
            metadata(fields, 1),
            [[first, 1.0, np.nan], [middle, 2.0, 7.0]],
        )
        self.assertTrue(upper_eos.validate().is_valid)
        upper = MultilinearInterpolator(upper_eos, ("optional",))
        upper_point = [upper_eos.axes[0][-1]]
        self.assertEqual(upper.locate(upper_point).normalized_coordinates[0], 1.0)
        np.testing.assert_allclose(upper(upper_point), [7.0])

        internal_eos = self.make_eos(
            metadata(fields, 1),
            [
                [first, 1.0, np.nan],
                [middle, 2.0, 7.0],
                [last, 3.0, 9.0],
            ],
        )
        self.assertTrue(internal_eos.validate().is_valid)
        internal = MultilinearInterpolator(internal_eos, ("optional",))
        internal_point = [internal_eos.axes[0][1]]
        location = internal.locate(internal_point, side="left")
        self.assertEqual(location.normalized_coordinates[0], 1.0)
        np.testing.assert_allclose(internal(internal_point, side="left"), [7.0])

    def test_no_extrapolation_and_explicit_boundary_snapping(self):
        eos = self.make_piecewise_eos()
        strict = MultilinearInterpolator(eos)
        just_below = np.nextafter(0.1, -np.inf)
        just_above = np.nextafter(0.5, np.inf)
        self.assertFalse(strict.contains([just_below]))
        self.assertFalse(strict.contains([just_above]))
        for point in ([just_below], [just_above], [0.09], [0.51]):
            with self.subTest(point=point):
                with self.assertRaises(InterpolationDomainError):
                    strict(point)

        tolerant = MultilinearInterpolator(
            eos, boundary_absolute_tolerance=1e-6
        )
        location = tolerant.locate([0.1 - 5e-7])
        self.assertTrue(location.was_snapped)
        self.assertAlmostEqual(location.requested_points[0], 0.1 - 5e-7)
        self.assertAlmostEqual(location.points[0], 0.1)
        np.testing.assert_allclose(tolerant([0.1 - 5e-7]), [0.0])
        self.assertTrue(tolerant.contains([0.1 - 5e-7]))
        upper_location = tolerant.locate([0.5 + 5e-7])
        self.assertTrue(upper_location.was_snapped)
        self.assertEqual(upper_location.points[0], 0.5)
        np.testing.assert_allclose(tolerant([0.5 + 5e-7]), [5.0])
        self.assertFalse(
            tolerant.contains([0.1 - 5e-7], include_boundary_tolerance=False)
        )
        with self.assertRaises(InterpolationDomainError):
            tolerant([0.1 - 2e-6])

        relative = MultilinearInterpolator(
            eos, boundary_relative_tolerance=1e-5
        )
        np.testing.assert_allclose(relative([0.5 + 4e-6]), [5.0])
        with self.assertRaises(InterpolationDomainError):
            relative([0.5 + 6e-6])

    def test_structured_batch_domain_error_preserves_query_indices(self):
        eos = self.make_multiaffine_eos(2)
        interpolator = MultilinearInterpolator(eos)
        points = np.asarray(
            [
                [[0.1, 0.0], [0.2, 0.1]],
                [[0.2, 5.0], [0.3, 0.2]],
            ]
        )
        with self.assertRaises(InterpolationDomainError) as context:
            interpolator(points)
        error = context.exception
        self.assertEqual(error.flat_indices, (2,))
        self.assertEqual(error.query_indices, ((1, 0),))
        np.testing.assert_allclose(error.points, [[0.2, 5.0]])
        np.testing.assert_allclose(error.bounds, interpolator.bounds)

    def test_malformed_and_nonfinite_queries_are_rejected(self):
        one_dimensional = MultilinearInterpolator(self.make_piecewise_eos())
        with self.assertRaises(InterpolationInputError):
            one_dimensional(0.2)
        with self.assertRaises(InterpolationInputError):
            one_dimensional([0.2, 0.3])
        for point in ([np.nan], [np.inf], [-np.inf]):
            with self.subTest(point=point):
                self.assertFalse(one_dimensional.contains(point))
                with self.assertRaises(InterpolationInputError):
                    one_dimensional(point)
        with self.assertRaises(ValueError):
            one_dimensional.jacobian([0.2], plane_policy="middle")
        with self.assertRaises(ValueError):
            one_dimensional.locate([0.2], side="middle")


class CellBoundsTests(InterpolationTestCase):
    def test_whole_cell_value_and_jacobian_bounds_are_tight(self):
        random = np.random.RandomState(8675309)
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                eos = self.make_multiaffine_eos(dimension)
                interpolator = MultilinearInterpolator(eos)
                cell = np.zeros(dimension, dtype=int)
                lower_coordinates, upper_coordinates = (
                    interpolator.cell_coordinate_bounds(cell)
                )
                np.testing.assert_allclose(
                    lower_coordinates,
                    [axis[0] for axis in TEST_AXES[:dimension]],
                )
                np.testing.assert_allclose(
                    upper_coordinates,
                    [axis[1] for axis in TEST_AXES[:dimension]],
                )

                corners = interpolator.cell_corner_values(cell)
                value_bounds = interpolator.cell_value_bounds(cell)
                np.testing.assert_allclose(value_bounds.lower, np.min(corners, axis=0))
                np.testing.assert_allclose(value_bounds.upper, np.max(corners, axis=0))
                self.assertFalse(value_bounds.rigorous)
                self.assertTrue(np.all(value_bounds.defined))

                corner_points = np.asarray(
                    list(
                        itertools.product(
                            *zip(lower_coordinates, upper_coordinates)
                        )
                    )
                )
                corner_gradients = multiaffine_jacobians(
                    corner_points, dimension, dimension
                )
                expected_lower = np.min(corner_gradients, axis=0)
                expected_upper = np.max(corner_gradients, axis=0)
                jacobian_bounds = interpolator.cell_jacobian_bounds(cell)
                np.testing.assert_allclose(
                    jacobian_bounds.lower,
                    expected_lower,
                    rtol=3e-14,
                    atol=3e-14,
                )
                np.testing.assert_allclose(
                    jacobian_bounds.upper,
                    expected_upper,
                    rtol=3e-14,
                    atol=3e-14,
                )
                self.assertFalse(jacobian_bounds.rigorous)

                points = random.uniform(
                    lower_coordinates, upper_coordinates, size=(30, dimension)
                )
                values = interpolator(points)
                jacobians = interpolator.jacobian(points)
                self.assertTrue(np.all(values >= value_bounds.lower - 1e-14))
                self.assertTrue(np.all(values <= value_bounds.upper + 1e-14))
                self.assertTrue(np.all(jacobians >= jacobian_bounds.lower - 1e-14))
                self.assertTrue(np.all(jacobians <= jacobian_bounds.upper + 1e-14))
                if dimension == 1:
                    np.testing.assert_allclose(
                        jacobian_bounds.lower, jacobian_bounds.upper
                    )

    def test_cell_batches_and_invalid_indices(self):
        interpolator = MultilinearInterpolator(self.make_multiaffine_eos(2))
        cells = np.asarray([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
        lower, upper = interpolator.cell_coordinate_bounds(cells)
        self.assertEqual(lower.shape, (2, 2, 2))
        self.assertEqual(upper.shape, (2, 2, 2))
        self.assertEqual(interpolator.cell_corner_values(cells).shape, (2, 2, 4, 2))
        value_bounds = interpolator.cell_value_bounds(cells)
        jacobian_bounds = interpolator.cell_jacobian_bounds(cells)
        self.assertEqual(value_bounds.lower.shape, (2, 2, 2))
        self.assertEqual(jacobian_bounds.lower.shape, (2, 2, 2, 2))
        for batch_index in np.ndindex(cells.shape[:-1]):
            cell = cells[batch_index]
            corner_points = np.asarray(
                list(
                    itertools.product(
                        *[
                            TEST_AXES[axis_index][index : index + 2]
                            for axis_index, index in enumerate(cell)
                        ]
                    )
                )
            )
            values = multiaffine_values(corner_points, 2, 2)
            jacobians = multiaffine_jacobians(corner_points, 2, 2)
            np.testing.assert_allclose(
                value_bounds.lower[batch_index], np.min(values, axis=0)
            )
            np.testing.assert_allclose(
                value_bounds.upper[batch_index], np.max(values, axis=0)
            )
            np.testing.assert_allclose(
                jacobian_bounds.lower[batch_index], np.min(jacobians, axis=0)
            )
            np.testing.assert_allclose(
                jacobian_bounds.upper[batch_index], np.max(jacobians, axis=0)
            )
        for invalid in ([[-1, 0]], [[2, 0]], [[0, 2]], [[0.0, 0.0]], [0]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(InterpolationCellError):
                    interpolator.cell_value_bounds(invalid)


class MetadataAndNaNPolicyTests(InterpolationTestCase):
    def make_auxiliary_eos(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "optional": field_spec(
                "auxiliary", 4, nonfinite_policy="allow_nan"
            ),
            "clean": field_spec("auxiliary", 5),
        }
        rows = [
            [0.1, 0.0, 1.0, 2.0, 10.0, -1.0],
            [0.1, 1.0, 2.0, 1.0, 20.0, 0.0],
            [0.2, 0.0, 3.0, 4.0, 30.0, 1.0],
            [0.2, 1.0, 4.0, 3.0, np.nan, 2.0],
        ]
        eos = self.make_eos(metadata(fields, 2), rows)
        self.assertTrue(eos.validate().is_valid)
        return eos

    def test_auxiliary_nan_is_field_local_and_zero_weight_corners_are_inactive(self):
        eos = self.make_auxiliary_eos()
        targets = MultilinearInterpolator(eos)
        auxiliaries = MultilinearInterpolator(eos, ("optional", "clean"))

        self.assertTrue(np.all(np.isfinite(targets([0.15, 0.5]))))
        interior = auxiliaries.evaluate_with_jacobian([0.15, 0.5])
        self.assertTrue(np.isnan(interior.values[0]))
        self.assertTrue(np.isfinite(interior.values[1]))
        self.assertFalse(interior.values_defined[0])
        self.assertTrue(interior.values_defined[1])

        # The NaN at the opposite corner has exactly zero weight here.
        np.testing.assert_allclose(auxiliaries([0.1, 0.0]), [10.0, -1.0])
        at_finite_node = auxiliaries.evaluate_with_jacobian([0.2, 0.0])
        np.testing.assert_allclose(at_finite_node.values, [30.0, 1.0])
        self.assertTrue(np.isnan(at_finite_node.jacobians[0, 1]))
        self.assertTrue(np.isfinite(at_finite_node.jacobians[1, 1]))

        bounds = auxiliaries.cell_value_bounds([0, 0])
        self.assertTrue(np.isnan(bounds.lower[0]))
        self.assertTrue(np.isnan(bounds.upper[0]))
        self.assertTrue(bounds.defined[1])
        self.assertFalse(bounds.defined[0])
        jacobian_bounds = auxiliaries.cell_jacobian_bounds([0, 0])
        self.assertTrue(np.all(np.isnan(jacobian_bounds.lower[0])))
        self.assertTrue(np.all(np.isnan(jacobian_bounds.upper[0])))
        self.assertTrue(np.all(np.isfinite(jacobian_bounds.lower[1])))
        self.assertTrue(np.all(np.isfinite(jacobian_bounds.upper[1])))

    def test_staged_contraction_preserves_tiny_positive_corner_contributions(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "signal": field_spec("auxiliary", 4),
            "optional": field_spec(
                "auxiliary", 5, nonfinite_policy="allow_nan"
            ),
        }
        rows = [
            [1e-200, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1e-200, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 1e308, np.nan],
        ]
        eos = self.make_eos(metadata(fields, 2), rows)
        self.assertTrue(eos.validate().is_valid)
        interpolator = MultilinearInterpolator(eos, ("signal", "optional"))
        point = [2e-200, 1e-200]

        values = interpolator(point)
        self.assertAlmostEqual(values[0] / 1e-92, 1.0, places=14)
        self.assertTrue(np.isnan(values[1]))
        jacobian = interpolator.jacobian(point)
        np.testing.assert_allclose(
            jacobian[0], [1e108, 1e108], rtol=3e-14, atol=0.0
        )
        self.assertTrue(np.all(np.isnan(jacobian[1])))

    def test_large_finite_divided_difference_avoids_intermediate_overflow(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "steep": field_spec("auxiliary", 2),
        }
        eos = self.make_eos(
            metadata(fields, 1),
            [[1e307, 1.0, -1e308], [2e307, 2.0, 1e308]],
        )
        self.assertTrue(eos.validate().is_valid)
        interpolator = MultilinearInterpolator(eos, ("steep",))
        midpoint = [1.5e307]
        np.testing.assert_allclose(interpolator(midpoint), [0.0], atol=1e292)
        np.testing.assert_allclose(
            interpolator.jacobian(midpoint), [[20.0]], rtol=3e-14
        )
        bounds = interpolator.cell_jacobian_bounds([0])
        np.testing.assert_allclose(bounds.lower, [[20.0]], rtol=3e-14)
        np.testing.assert_allclose(bounds.upper, [[20.0]], rtol=3e-14)

    def test_metadata_interpolation_policy_and_validation_lifecycle(self):
        eos = self.make_auxiliary_eos()
        unvalidated = ForwardTable(eos.metadata_path)
        with self.assertRaises(EoSNotValidatedError):
            MultilinearInterpolator(unvalidated)

        for names in (("e", "e"), ("missing",), ("T",), ()):
            with self.subTest(names=names):
                with self.assertRaises(InterpolationConfigurationError):
                    MultilinearInterpolator(eos, names)

        base_document = json.loads(eos.metadata_path.read_text(encoding="utf-8"))
        table = np.loadtxt(str(eos.data_path))
        for policy in (None, "nearest"):
            with self.subTest(policy=policy):
                document = json.loads(json.dumps(base_document))
                document["fields"]["clean"]["interpolation"] = policy
                candidate = self.make_eos(document, table)
                self.assertTrue(candidate.validate().is_valid)
                with self.assertRaises(InterpolationConfigurationError):
                    MultilinearInterpolator(candidate, ("clean",))

        # output=false controls a future writer, not explicit interpolation.
        document = json.loads(json.dumps(base_document))
        document["fields"]["clean"]["output"] = False
        candidate = self.make_eos(document, table)
        self.assertTrue(candidate.validate().is_valid)
        MultilinearInterpolator(candidate, ("clean",))

        with self.assertRaises(InterpolationConfigurationError):
            MultilinearInterpolator(
                eos,
                boundary_absolute_tolerance=np.finfo(float).max,
                boundary_relative_tolerance=np.finfo(float).max,
            )
        for keyword, value, error in (
            ("boundary_absolute_tolerance", -1.0, ValueError),
            ("boundary_relative_tolerance", np.inf, ValueError),
            ("boundary_absolute_tolerance", True, TypeError),
            ("chunk_size", 0, ValueError),
            ("chunk_size", 1.5, TypeError),
        ):
            with self.subTest(keyword=keyword, value=value):
                with self.assertRaises(error):
                    MultilinearInterpolator(eos, **{keyword: value})

    def test_interpolator_owns_a_stable_validated_snapshot(self):
        eos = self.make_auxiliary_eos()
        interpolator = MultilinearInterpolator(eos)
        expected = interpolator([0.15, 0.5])
        eos.read_data()
        self.assertFalse(eos.is_validated)
        np.testing.assert_allclose(interpolator([0.15, 0.5]), expected)


class RealTableInterpolationTests(unittest.TestCase):
    ISING_METADATA = REPOSITORY_ROOT / "EoSTables" / "2DIsing" / "metadata.json"
    ISING_TABLE = REPOSITORY_ROOT / "EoSTables" / "2DIsing" / "EoSmub602.dat"
    CEC_METADATA = REPOSITORY_ROOT / "EoSTables" / "CEC" / "metadata.json"
    CEC_TABLE = (
        REPOSITORY_ROOT / "EoSTables" / "CEC" / "hic_table_T80-300_muB0-800_d1.dat"
    )

    @unittest.skipUnless(ISING_TABLE.is_file(), "local Ising table is unavailable")
    def test_ising_nodes_midpoint_and_jacobian(self):
        eos = ForwardTable.load_validated(self.ISING_METADATA)
        interpolator = MultilinearInterpolator(eos)
        indices = (120, 345)
        point = [eos.axes[0][indices[0]], eos.axes[1][indices[1]]]
        expected = [eos.field(name)[indices] for name in eos.target_names]
        np.testing.assert_allclose(interpolator(point), expected)

        cell = np.asarray(indices)
        lower, upper = interpolator.cell_coordinate_bounds(cell)
        midpoint = (lower + upper) / 2.0
        values = interpolator(midpoint)
        bounds = interpolator.cell_value_bounds(cell)
        self.assertTrue(np.all(values >= bounds.lower))
        self.assertTrue(np.all(values <= bounds.upper))
        self.assertTrue(np.all(np.isfinite(interpolator.jacobian(midpoint))))

    @unittest.skipUnless(CEC_TABLE.is_file(), "local CEC table is unavailable")
    def test_cec_nan_auxiliary_does_not_contaminate_other_fields(self):
        eos = ForwardTable.load_validated(self.CEC_METADATA)
        target_interpolator = MultilinearInterpolator(eos)
        auxiliary_interpolator = MultilinearInterpolator(eos, ("cs2", "P", "fdense"))
        nan_node = np.argwhere(np.isnan(eos.field("cs2")))[0]
        point = [
            eos.axes[axis_index][coordinate_index]
            for axis_index, coordinate_index in enumerate(nan_node)
        ]
        self.assertTrue(np.all(np.isfinite(target_interpolator(point))))
        outputs = auxiliary_interpolator(point)
        self.assertTrue(np.isnan(outputs[0]))
        self.assertTrue(np.isfinite(outputs[1]))
        self.assertTrue(np.isfinite(outputs[2]))


if __name__ == "__main__":
    unittest.main()
