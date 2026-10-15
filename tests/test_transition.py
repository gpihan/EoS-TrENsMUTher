"""Tests for standalone physical first-order-transition models."""

import unittest
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    BinodalBand,
    EoSNotValidatedError,
    FirstOrderRepresentation,
    ForwardTable,
    PhaseLabel,
    TransitionConfigurationError,
    TransitionDomainError,
    TransitionInputError,
    TransitionLine,
    transition_model_from_table,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TransitionLineTests(unittest.TestCase):
    def make_line(self):
        return TransitionLine(
            muB=[0.4, 0.5, 0.7],
            temperature=[0.14, 0.12, 0.08],
            sidecar_fingerprint="a" * 64,
            critical_endpoint="first",
            critical_point_name="synthetic critical point",
            kind="synthetic line",
        )

    def test_bounded_piecewise_linear_evaluation(self):
        line = self.make_line()

        self.assertAlmostEqual(line.evaluate(0.4), 0.14)
        self.assertAlmostEqual(line.evaluate(0.45), 0.13)
        np.testing.assert_allclose(
            line.evaluate(np.asarray([[0.4, 0.6], [0.5, 0.7]])),
            [[0.14, 0.10], [0.12, 0.08]],
        )
        self.assertFalse(line.evaluate([0.4]).flags.writeable)
        with self.assertRaises(TransitionDomainError):
            line.evaluate(0.399999999)
        with self.assertRaises(TransitionDomainError):
            line.evaluate([0.5, 0.700000001])

    def test_exact_closed_source_membership(self):
        line = self.make_line()

        self.assertTrue(line.contains(0.13, 0.45))
        self.assertTrue(line.contains(0.14, 0.4))
        self.assertTrue(line.contains(0.08, 0.7))
        self.assertFalse(line.contains(0.13000001, 0.45))
        self.assertFalse(line.contains(0.14, 0.39))
        self.assertTrue(
            line.contains(
                0.13000001,
                0.45,
                absolute_tolerance=1.1e-8,
            )
        )
        np.testing.assert_array_equal(
            line.contains_points(
                [[0.14, 0.4], [0.13, 0.45], [0.13, 0.39]]
            ),
            [True, True, False],
        )

    def test_critical_direction_and_diagnostics(self):
        line = self.make_line()

        self.assertIs(line.representation, FirstOrderRepresentation.LINE)
        self.assertIs(line.phase_label, PhaseLabel.MIXED_PHASE)
        self.assertEqual(line.first_order_muB_direction, "increasing")
        self.assertEqual(dict(line.critical_point), {"T": 0.14, "muB": 0.4})
        self.assertEqual(line.muB_bounds, (0.4, 0.7))
        self.assertEqual(line.diagnostics.interpolation, "piecewise_linear")
        self.assertEqual(line.diagnostics.extrapolation, "forbidden")
        self.assertEqual(line.diagnostics.to_dict()["critical_endpoint"], "first")
        self.assertFalse(line.muB.flags.writeable)
        self.assertFalse(line.temperature.flags.writeable)

        reverse = TransitionLine(
            muB=[0.4, 0.7],
            temperature=[0.14, 0.08],
            sidecar_fingerprint="b" * 64,
            critical_endpoint="last",
            critical_point_name="other endpoint",
            kind="synthetic line",
        )
        self.assertEqual(reverse.first_order_muB_direction, "decreasing")
        self.assertEqual(dict(reverse.critical_point), {"T": 0.08, "muB": 0.7})

    def test_invalid_line_configuration_and_queries_are_rejected(self):
        base = {
            "temperature": [0.14, 0.12],
            "sidecar_fingerprint": "a" * 64,
            "critical_endpoint": "first",
            "critical_point_name": "critical",
            "kind": "line",
        }
        for muB in ([0.4], [0.5, 0.4], [0.4, np.nan]):
            with self.subTest(muB=muB):
                with self.assertRaises(TransitionConfigurationError):
                    TransitionLine(muB=muB, **base)
        with self.assertRaises(TransitionConfigurationError):
            TransitionLine(muB=[0.4, 0.5], **{**base, "temperature": [0.14, 0.12, 0.1]})
        line = self.make_line()
        for value in (np.nan, np.inf, True, "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(TransitionInputError):
                    line.evaluate(value)
        with self.assertRaises(TransitionInputError):
            line.contains_points([0.14, 0.4, 1.0])


class BinodalBandTests(unittest.TestCase):
    def make_band(self):
        return BinodalBand(
            muB=[0.0, 0.5, 1.0],
            dilute_temperature=[0.10, 0.12, 0.16],
            dense_temperature=[0.20, 0.18, 0.14],
            sidecar_fingerprint="c" * 64,
            kind="synthetic binodals",
            phase_fraction_field="phase",
            mixed_phase_condition="0 < phase < 1",
        )

    def test_curve_and_sorted_bound_evaluation(self):
        band = self.make_band()

        np.testing.assert_allclose(band.evaluate(0.25), [0.11, 0.19])
        np.testing.assert_allclose(band.temperature_bounds(0.25), [0.11, 0.19])
        # The supplied curve ordering reverses near the final endpoint.
        np.testing.assert_allclose(band.evaluate(1.0), [0.16, 0.14])
        np.testing.assert_allclose(band.temperature_bounds(1.0), [0.14, 0.16])
        np.testing.assert_allclose(
            band.temperature_bounds([0.0, 0.75, 1.0]),
            [[0.10, 0.20], [0.14, 0.16], [0.14, 0.16]],
        )
        self.assertFalse(band.evaluate([0.5]).flags.writeable)
        with self.assertRaises(TransitionDomainError):
            band.evaluate(-np.finfo(float).eps)
        with self.assertRaises(TransitionDomainError):
            band.temperature_bounds(1.000001)

    def test_exact_inclusive_mixed_phase_membership(self):
        band = self.make_band()

        for temperature in (0.10, 0.15, 0.20):
            with self.subTest(temperature=temperature):
                self.assertTrue(band.contains(temperature, 0.0))
        self.assertTrue(band.contains(0.15, 1.0))
        self.assertFalse(band.contains(0.139999, 1.0))
        self.assertFalse(band.contains(0.160001, 1.0))
        self.assertFalse(band.contains(0.15, -0.01))
        self.assertFalse(band.contains(0.15, 1.01))
        np.testing.assert_array_equal(
            band.contains_points(
                [[0.15, 0.0], [0.11, 0.25], [0.21, 0.0], [0.15, 1.1]]
            ),
            [True, True, False, False],
        )

    def test_binodal_diagnostics_and_immutability(self):
        band = self.make_band()

        self.assertIs(band.representation, FirstOrderRepresentation.BINODALS)
        self.assertIs(band.phase_label, PhaseLabel.MIXED_PHASE)
        self.assertEqual(band.muB_bounds, (0.0, 1.0))
        self.assertEqual(band.phase_fraction_field, "phase")
        self.assertIn("T_lower", band.diagnostics.temperature_bounds)
        self.assertEqual(
            band.diagnostics.to_dict()["source_membership_boundary"], "closed"
        )
        for array in (
            band.muB,
            band.dilute_temperature,
            band.dense_temperature,
        ):
            self.assertFalse(array.flags.writeable)

    def test_invalid_binodal_configuration_and_queries_are_rejected(self):
        base = {
            "dilute_temperature": [0.1, 0.11],
            "dense_temperature": [0.2, 0.19],
            "sidecar_fingerprint": "c" * 64,
            "kind": "binodals",
        }
        with self.assertRaises(TransitionConfigurationError):
            BinodalBand(muB=[0.5, 0.4], **base)
        with self.assertRaises(TransitionConfigurationError):
            BinodalBand(
                muB=[0.4, 0.5],
                **{**base, "dense_temperature": [0.2, np.inf]},
            )
        band = self.make_band()
        with self.assertRaises(TransitionInputError):
            band.contains(np.nan, 0.5)
        with self.assertRaises(TransitionInputError):
            band.contains([0.1, 0.2], [0.0, 0.5, 1.0])
        with self.assertRaises(TransitionInputError):
            band.contains_points([[0.1, np.nan]])


class ValidatedTableFactoryTests(unittest.TestCase):
    ISING_METADATA = REPOSITORY_ROOT / "EoSTables" / "2DIsing" / "metadata.json"
    ISING_TABLE = REPOSITORY_ROOT / "EoSTables" / "2DIsing" / "EoSmub602.dat"
    CEC_METADATA = REPOSITORY_ROOT / "EoSTables" / "CEC" / "metadata.json"
    CEC_TABLE = (
        REPOSITORY_ROOT
        / "EoSTables"
        / "CEC"
        / "hic_table_T80-300_muB0-800_d1.dat"
    )

    @unittest.skipUnless(ISING_TABLE.is_file(), "local Ising table is unavailable")
    def test_factory_builds_real_ising_line(self):
        table = ForwardTable.load_validated(self.ISING_METADATA)

        model = transition_model_from_table(table)

        self.assertIsInstance(model, TransitionLine)
        self.assertAlmostEqual(model.evaluate(0.602), 0.0928908)
        self.assertEqual(model.critical_endpoint, "first")
        self.assertEqual(model.diagnostics.point_count, 99)
        self.assertEqual(
            model.sidecar_fingerprint, table.coexistence_table.fingerprint
        )

    @unittest.skipUnless(CEC_TABLE.is_file(), "local CEC table is unavailable")
    def test_factory_builds_real_cec_binodal_band(self):
        table = ForwardTable.load_validated(self.CEC_METADATA)

        model = transition_model_from_table(table)

        self.assertIsInstance(model, BinodalBand)
        lower, upper = model.temperature_bounds(0.655)
        self.assertAlmostEqual(lower, 0.1140229)
        self.assertAlmostEqual(upper, 0.1144662)
        self.assertTrue(model.contains(lower, 0.655))
        self.assertTrue(model.contains(upper, 0.655))
        self.assertFalse(model.contains(lower - 1e-9, 0.655))
        self.assertFalse(model.contains(0.1142, 0.654999))
        self.assertEqual(model.diagnostics.point_count, 146)

    def test_factory_requires_validated_forward_table(self):
        table = ForwardTable(self.ISING_METADATA)
        with self.assertRaises(EoSNotValidatedError):
            transition_model_from_table(table)

    def test_factory_rejects_non_table_input(self):
        with self.assertRaises(TypeError):
            transition_model_from_table(object())


if __name__ == "__main__":
    unittest.main()
