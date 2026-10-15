"""Tests for physical LINE endpoint reconstruction and RegionS geometry."""

import unittest
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    CellImageKind,
    EoSRootSolver,
    ForwardTable,
    MixedPhaseStrip,
    MixedPhaseCandidateIndex,
    ProofStatus,
    ReachabilityIndex,
    RegionSBuilder,
    RegionSConfig,
    RegionSReconstructionError,
    SolveStatus,
    SolverConfig,
    TransitionConfigurationError,
    TransitionDomainError,
    build_region_s,
)
from tests.test_tables import TemporaryEoS, field_spec, metadata


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SyntheticRegionSTests(unittest.TestCase):
    def make_line_table(
        self,
        temperature_axis=None,
        line_muB_axis=None,
        muB_axis=None,
        transition_temperature=None,
        include_auxiliary=False,
    ):
        if temperature_axis is None:
            temperature_axis = np.linspace(0.05, 0.25, 21)
        temperature_axis = np.asarray(temperature_axis, dtype=float)
        if muB_axis is None:
            muB_axis = np.asarray([0.2, 0.3, 0.4])
        muB_axis = np.asarray(muB_axis, dtype=float)
        if line_muB_axis is None:
            line_muB_axis = muB_axis
        line_muB_axis = np.asarray(line_muB_axis, dtype=float)
        if transition_temperature is None:
            transition_temperature = 0.18 - 0.1 * muB_axis
        transition_temperature = np.asarray(
            transition_temperature, dtype=float
        )
        if transition_temperature.shape != muB_axis.shape:
            raise ValueError(
                "transition_temperature must have one value per muB slice"
            )
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }
        if include_auxiliary:
            fields.update(
                {
                    "P": field_spec("auxiliary", 4),
                    "marker": field_spec("auxiliary", 5),
                }
            )
        document = metadata(
            fields,
            2,
            expected_shape=[temperature_axis.size, muB_axis.size],
        )
        document["scientific"]["phase_structure"] = {
            "status": "known",
            "critical_points": [
                {
                    "name": "synthetic critical endpoint",
                    "status": "known",
                    "location": {
                        "T": {
                            "value": float(transition_temperature[0]),
                            "unit": "GeV",
                        },
                        "muB": {
                            "value": float(line_muB_axis[0]),
                            "unit": "GeV",
                        },
                    },
                }
            ],
            "first_order_transition": {
                "present": True,
                "representation": "line",
                "kind": "synthetic congruent transition",
                "critical_point": "synthetic critical endpoint",
                "critical_point_tolerance": {
                    "absolute": 1.0e-12,
                    "relative": 1.0e-10,
                },
                "coexistence_data": {
                    "file": "transition.dat",
                    "delimiter": "whitespace",
                    "comment_prefix": "#",
                    "skip_rows": 0,
                    "encoding": "utf-8",
                    "expected_columns": 2,
                    "fields": ["muB_GeV", "T_GeV"],
                    "columns": {
                        "muB": {"index": 0, "unit": "GeV"},
                        "T": {"index": 1, "unit": "GeV"},
                    },
                },
            },
        }
        rows = []
        for temperature in temperature_axis:
            for muB, critical_temperature in zip(
                muB_axis, transition_temperature
            ):
                if temperature < critical_temperature:
                    energy_scaled = 1.5 + 2.0 * temperature + 0.4 * muB
                    density_scaled = 0.8 - 0.5 * temperature + 0.2 * muB
                else:
                    energy_scaled = 4.0 + temperature + 0.6 * muB
                    density_scaled = 2.0 + 0.3 * temperature + 0.5 * muB
                row = [
                    temperature,
                    muB,
                    energy_scaled * temperature**4,
                    density_scaled * temperature**3,
                ]
                if include_auxiliary:
                    if temperature < critical_temperature:
                        pressure = 10.0 * temperature + muB
                        marker = 100.0 * temperature - muB
                    else:
                        pressure = 20.0 * temperature + 2.0 * muB + 1.0
                        marker = 200.0 * temperature - 2.0 * muB + 3.0
                    row.extend(
                        (
                            pressure,
                            marker,
                        )
                    )
                rows.append(row)
        fixture = TemporaryEoS(document, rows)
        self.addCleanup(fixture.close)
        fixture.write_sidecar(
            "transition.dat",
            "".join(
                "%s %s\n" % (muB, temperature)
                for muB, temperature in zip(
                    line_muB_axis, transition_temperature
                )
            ),
        )
        table = ForwardTable.load_validated(fixture.metadata_path)
        return table, transition_temperature

    @staticmethod
    def raw_endpoint_targets(muB, temperature):
        energy_h = (1.5 + 2.0 * temperature + 0.4 * muB) * temperature**4
        density_h = (0.8 - 0.5 * temperature + 0.2 * muB) * temperature**3
        energy_q = (4.0 + temperature + 0.6 * muB) * temperature**4
        density_q = (2.0 + 0.3 * temperature + 0.5 * muB) * temperature**3
        return np.stack(
            (
                np.column_stack((energy_h, density_h)),
                np.column_stack((energy_q, density_q)),
            ),
            axis=1,
        )

    @classmethod
    def exact_endpoints(cls, muB, temperature):
        targets = cls.raw_endpoint_targets(muB, temperature)
        energy_h = np.array(targets[:, 0, 0], copy=True)
        density_h = np.array(targets[:, 0, 1], copy=True)
        energy_q = np.array(targets[:, 1, 0], copy=True)
        density_q = np.array(targets[:, 1, 1], copy=True)
        energy_h[0] = energy_q[0]
        density_h[0] = density_q[0]
        return np.column_stack(
            (temperature, muB, energy_h, density_h, energy_q, density_q)
        )

    @staticmethod
    def exact_config(**overrides):
        values = {
            "hadronic_temperature_offset": 0.02,
            "qgp_temperature_offset": 0.02,
            "qgp_energy_gaussian_sigma": 0.0,
            "hadronic_energy_smoothing": 0.0,
            "hadronic_density_smoothing": 0.0,
            "qgp_energy_smoothing": 0.0,
            "qgp_density_smoothing": 0.0,
        }
        values.update(overrides)
        return RegionSConfig(**values)

    def test_physical_endpoint_extraction_matches_analytic_fixture(self):
        table, temperature = self.make_line_table()
        muB = table.axes[1]

        strip = build_region_s(table, self.exact_config())

        self.assertIsInstance(strip, MixedPhaseStrip)
        np.testing.assert_allclose(
            strip.endpoints.as_array(),
            self.exact_endpoints(muB, temperature),
            rtol=2.0e-12,
            atol=2.0e-14,
        )
        self.assertEqual(strip.shape, (3, 2))
        self.assertEqual(strip.cell_shape, (2, 1))
        self.assertEqual(strip.coordinate_names, ("muB", "lambda"))
        self.assertEqual(strip.endpoints.target_segment_norms[0], 0.0)
        self.assertFalse(strip.endpoints.as_array().flags.writeable)

    def test_strip_interpolation_source_coordinates_and_cell_boxes(self):
        table, _ = self.make_line_table()
        strip = RegionSBuilder(table, self.exact_config()).build()
        endpoints = strip.endpoints

        np.testing.assert_allclose(
            strip.evaluate(endpoints.muB, 0.0), endpoints.hadronic_targets
        )
        np.testing.assert_allclose(
            strip.evaluate(endpoints.muB, 1.0), endpoints.qgp_targets
        )
        midpoint_endpoints = 0.5 * (
            endpoints.target_segments[0] + endpoints.target_segments[1]
        )
        np.testing.assert_allclose(
            strip.evaluate_endpoints(0.25), midpoint_endpoints
        )
        np.testing.assert_allclose(
            strip.evaluate(0.25, 0.25),
            0.75 * midpoint_endpoints[0] + 0.25 * midpoint_endpoints[1],
        )
        np.testing.assert_allclose(
            strip.source_coordinates([0.2, 0.3, 0.4], [0.0, 0.5, 1.0]),
            [[0.16, 0.2], [0.15, 0.3], [0.14, 0.4]],
        )

        cells = strip.cell_target_enclosures()
        self.assertEqual(cells.cell_count, 2)
        np.testing.assert_allclose(cells.lower, np.min(cells.corners, axis=1))
        np.testing.assert_allclose(cells.upper, np.max(cells.corners, axis=1))
        self.assertTrue(cells.rigorous)
        self.assertFalse(cells.corners.flags.writeable)

    def test_generic_auxiliary_endpoints_follow_line_orientation_and_lever_rule(self):
        table, _ = self.make_line_table(include_auxiliary=True)
        strip = build_region_s(
            table,
            self.exact_config(),
            auxiliary_fields=("P", "marker"),
        )

        self.assertEqual(strip.auxiliary_names, ("P", "marker"))
        auxiliary_endpoints = strip.auxiliary_endpoints.evaluate_endpoints(0.35)
        np.testing.assert_allclose(
            strip.evaluate_auxiliaries(0.35, 0.0), auxiliary_endpoints[0]
        )
        np.testing.assert_allclose(
            strip.evaluate_auxiliaries(0.35, 1.0), auxiliary_endpoints[1]
        )
        np.testing.assert_allclose(
            strip.evaluate_auxiliaries(0.35, 0.25),
            0.75 * auxiliary_endpoints[0] + 0.25 * auxiliary_endpoints[1],
        )
        self.assertGreater(
            abs(float(auxiliary_endpoints[1, 0] - auxiliary_endpoints[0, 0])),
            0.1,
        )
        self.assertFalse(strip.auxiliary_endpoints.hadronic_values.flags.writeable)

    def test_strip_shared_interpolator_and_analytic_inversion(self):
        table, _ = self.make_line_table()
        strip = build_region_s(table, self.exact_config())
        interpolator = strip.to_interpolator(chunk_size=2)
        points = np.asarray(
            [[0.2, 0.0], [0.25, 0.25], [0.35, 0.4], [0.4, 1.0]]
        )

        self.assertEqual(interpolator.coordinate_names, ("muB", "lambda"))
        self.assertEqual(interpolator.target_names, ("e", "nB"))
        self.assertEqual(interpolator.field_names, ("e", "nB"))
        self.assertEqual(interpolator.coordinate_units, ("GeV", "1"))
        self.assertEqual(interpolator.field_units, ("GeV^4", "GeV^3"))
        np.testing.assert_array_equal(interpolator.axes[0], strip.endpoints.muB)
        np.testing.assert_array_equal(interpolator.axes[1], [0.0, 1.0])
        np.testing.assert_allclose(
            interpolator(points), strip.evaluate(points[:, 0], points[:, 1]),
            rtol=2.0e-15, atol=2.0e-17,
        )
        self.assertTrue(
            np.all(np.isfinite(interpolator.jacobian(points[1:])))
        )

        strip_boxes = strip.cell_target_enclosures()
        map_boxes = interpolator.cell_value_enclosures(
            np.column_stack(
                (
                    np.arange(strip.cell_shape[0], dtype=np.int64),
                    np.zeros(strip.cell_shape[0], dtype=np.int64),
                )
            )
        )
        self.assertTrue(map_boxes.rigorous)
        np.testing.assert_array_equal(map_boxes.lower, strip_boxes.lower)
        np.testing.assert_array_equal(map_boxes.upper, strip_boxes.upper)

        reachability = ReachabilityIndex(
            interpolator, residual_scales=[1.0, 1.0],
            root_residual_tolerance=0.0,
        )
        np.testing.assert_array_equal(
            reachability.enclosures.exact_lower, strip_boxes.lower
        )
        np.testing.assert_array_equal(
            reachability.enclosures.exact_upper, strip_boxes.upper
        )

        mixed_candidates = MixedPhaseCandidateIndex.for_line(
            strip, residual_scales=[1.0, 1.0],
            root_residual_tolerance=0.0,
        )
        np.testing.assert_array_equal(
            mixed_candidates.images.exact_lower, strip_boxes.lower
        )
        np.testing.assert_array_equal(
            mixed_candidates.images.exact_upper, strip_boxes.upper
        )
        self.assertTrue(
            np.all(
                mixed_candidates.images.kind_codes
                == int(CellImageKind.LINE_REGION_S)
            )
        )

        expected_source = np.asarray([0.35, 0.4])
        target = strip.evaluate(*expected_source)
        self.assertTrue(bool(reachability.classify(target).exact_candidate))
        mixed_result = mixed_candidates.classify(target)
        self.assertTrue(bool(mixed_result.exact_candidate))
        self.assertTrue(bool(mixed_result.has_kind(CellImageKind.LINE_REGION_S)))
        self.assertTrue(bool(mixed_candidates.verify_source(expected_source)))
        self.assertFalse(bool(mixed_candidates.verify_source([0.19, 0.4])))
        solver = EoSRootSolver(
            interpolator,
            SolverConfig(enable_subcell_fallback=False),
        )
        fast = solver.solve(target, [0.34, 0.5], strategy="fast")
        fallback = solver.solve(target, [0.34, 0.5], strategy="fallback")

        self.assertEqual(fast.status, SolveStatus.CONVERGED_FAST)
        self.assertEqual(fast.proof_status, ProofStatus.NONE)
        np.testing.assert_allclose(fast.solution, expected_source, atol=2.0e-11)
        self.assertEqual(fallback.status, SolveStatus.CONVERGED_FALLBACK)
        self.assertEqual(fallback.proof_status, ProofStatus.UNIQUE_IN_BOX)
        np.testing.assert_allclose(
            fallback.solution, expected_source, atol=2.0e-11
        )
        np.testing.assert_allclose(
            strip.source_coordinates(*fallback.solution), [0.145, 0.35]
        )

    def test_strip_queries_are_strictly_bounded(self):
        table, _ = self.make_line_table()
        strip = build_region_s(table, self.exact_config())

        for muB, fraction in ((0.199, 0.5), (0.401, 0.5), (0.3, -1e-9), (0.3, 1.0001)):
            with self.subTest(muB=muB, fraction=fraction):
                with self.assertRaises(TransitionDomainError):
                    strip.evaluate(muB, fraction)
        with self.assertRaises(TransitionDomainError):
            strip.evaluate_endpoints(0.5)

    def test_diagnostics_record_physical_closure_policy_and_are_immutable(self):
        table, _ = self.make_line_table()
        strip = build_region_s(table, self.exact_config())
        diagnostic = strip.diagnostics

        self.assertEqual(diagnostic.algorithm, "physical_side_spline_v3")
        self.assertEqual(diagnostic.reconstructed_slice_count, 3)
        self.assertEqual(diagnostic.config["endpoint_space"], "canonical_physical")
        self.assertEqual(diagnostic.config["cross_slice_smoothing"], "none")
        self.assertEqual(
            diagnostic.config["critical_endpoint_policy"],
            "numerical_physical_center_separation_closure",
        )
        self.assertEqual(
            diagnostic.config["critical_closure_weight_function"],
            "smoothstep_3s2_minus_2s3",
        )
        self.assertTrue(
            diagnostic.config["critical_closure_geometry_validated"]
        )
        self.assertTrue(diagnostic.config["critical_closure_applied"])
        self.assertEqual(len(diagnostic.endpoint_fingerprint), 64)
        self.assertEqual(
            sum(diagnostic.energy_orientation_counts.values()), 3
        )
        with self.assertRaises(TypeError):
            diagnostic.config["cross_slice_smoothing"] = "enabled"
        self.assertEqual(
            diagnostic.to_dict()["endpoint_fingerprint"],
            diagnostic.endpoint_fingerprint,
        )

    def test_numerical_cep_closure_matches_formula_and_preserves_exterior(self):
        muB = np.linspace(0.2, 0.22, 21)
        transition_temperature = 0.18 - 0.1 * muB
        table, _ = self.make_line_table(
            muB_axis=muB,
            transition_temperature=transition_temperature,
        )
        config = self.exact_config(
            critical_closure_width=0.005,
            critical_closure_max_slice_expansions=15,
        )

        strip = build_region_s(table, config)
        endpoints = strip.endpoints
        diagnostic = strip.diagnostics.config
        raw = self.raw_endpoint_targets(muB, transition_temperature)
        match = int(
            np.flatnonzero(
                np.isclose(
                    muB,
                    diagnostic["critical_closure_match_muB_GeV"],
                    rtol=0.0,
                    atol=1.0e-14,
                )
            )[0]
        )
        critical_state = np.asarray(
            [
                np.interp(
                    transition_temperature[0],
                    table.axes[0],
                    table.grid_fields[name][:, 0],
                )
                for name in ("e", "nB")
            ]
        )

        np.testing.assert_array_equal(
            endpoints.hadronic_targets[0], critical_state
        )
        np.testing.assert_array_equal(endpoints.qgp_targets[0], critical_state)
        np.testing.assert_allclose(
            endpoints.target_segments[match:], raw[match:],
            rtol=3.0e-12,
            atol=3.0e-14,
        )

        raw_center = 0.5 * (raw[:, 0] + raw[:, 1])
        raw_separation = 0.5 * (raw[:, 1] - raw[:, 0])
        match_distance = muB[match] - muB[0]
        expected = np.array(raw, copy=True)
        for index in range(match):
            s = (muB[index] - muB[0]) / match_distance
            weight = 3.0 * s**2 - 2.0 * s**3
            center = raw_center[index] + (1.0 - weight) * (
                critical_state - raw_center[0]
            )
            separation = weight * raw_separation[index]
            expected[index, 0] = center - separation
            expected[index, 1] = center + separation
        expected[0] = critical_state
        np.testing.assert_allclose(
            endpoints.target_segments[: match + 1],
            expected[: match + 1],
            rtol=3.0e-12,
            atol=3.0e-14,
        )
        separation_norm = endpoints.target_segment_norms[: match + 1]
        self.assertTrue(np.all(np.diff(separation_norm) >= -1.0e-14))
        self.assertGreaterEqual(
            diagnostic["critical_closure_effective_width_GeV"],
            config.critical_closure_width,
        )
        self.assertTrue(diagnostic["critical_closure_geometry_validated"])

    def test_closure_can_be_disabled_to_return_unmerged_endpoints(self):
        table, transition_temperature = self.make_line_table()
        muB = table.axes[1]
        unmerged = build_region_s(
            table,
            self.exact_config(critical_closure_enabled=False),
        )
        closed = build_region_s(table, self.exact_config())
        expected_raw = self.raw_endpoint_targets(muB, transition_temperature)

        np.testing.assert_allclose(
            unmerged.endpoints.target_segments,
            expected_raw,
            rtol=3.0e-12,
            atol=3.0e-14,
        )
        self.assertGreater(unmerged.endpoints.target_segment_norms[0], 0.0)
        self.assertEqual(
            unmerged.diagnostics.config["critical_endpoint_policy"],
            "unmerged_one_sided_endpoints",
        )
        self.assertFalse(
            unmerged.diagnostics.config["critical_closure_applied"]
        )
        self.assertIsNone(
            unmerged.diagnostics.config["critical_closure_match_muB_GeV"]
        )
        self.assertIsNone(
            unmerged.diagnostics.config[
                "critical_closure_geometry_validated"
            ]
        )

        match_muB = closed.diagnostics.config[
            "critical_closure_match_muB_GeV"
        ]
        match = int(np.flatnonzero(np.isclose(muB, match_muB))[0])
        np.testing.assert_array_equal(
            closed.endpoints.target_segments[match:],
            unmerged.endpoints.target_segments[match:],
        )

    def test_output_grid_space_cannot_change_physical_endpoints(self):
        table, _ = self.make_line_table()

        table.find_boundaries(space="target")
        target_grid_result = build_region_s(table, self.exact_config())
        table.find_boundaries(space="tilde")
        transformed_grid_result = build_region_s(table, self.exact_config())

        np.testing.assert_array_equal(
            transformed_grid_result.endpoints.as_array(),
            target_grid_result.endpoints.as_array(),
        )
        self.assertEqual(
            transformed_grid_result.diagnostics.endpoint_fingerprint,
            target_grid_result.diagnostics.endpoint_fingerprint,
        )

    def test_configuration_and_insufficient_side_support_fail_clearly(self):
        for keyword, value in (
            ("hadronic_temperature_offset", 0.0),
            ("qgp_energy_gaussian_sigma", -1.0),
            ("critical_closure_enabled", 1),
            ("critical_closure_width", 0.0),
            ("critical_closure_max_slice_expansions", -1),
            ("critical_closure_max_slice_expansions", True),
            ("spline_degree", 6),
            ("spline_degree", True),
        ):
            with self.subTest(keyword=keyword, value=value):
                with self.assertRaises(TransitionConfigurationError):
                    RegionSConfig(**{keyword: value})

        table, _ = self.make_line_table()
        with self.assertRaisesRegex(
            RegionSReconstructionError, "insufficient side samples"
        ):
            build_region_s(
                table,
                self.exact_config(hadronic_temperature_offset=0.14),
            )

    def test_declared_critical_muB_must_be_a_source_table_slice(self):
        table, _ = self.make_line_table(line_muB_axis=[0.25, 0.3, 0.4])

        with self.assertRaisesRegex(
            RegionSReconstructionError,
            "critical muB=.*must coincide with exactly one source-table",
        ):
            build_region_s(table, self.exact_config())

    def test_nonuniform_temperature_requires_disabling_legacy_gaussian(self):
        temperature = np.asarray(
            [0.05, 0.06, 0.071, 0.083, 0.096, 0.11, 0.125, 0.14,
             0.156, 0.173, 0.191, 0.21, 0.23, 0.25]
        )
        table, _ = self.make_line_table(temperature)

        with self.assertRaisesRegex(RegionSReconstructionError, "uniform T axis"):
            build_region_s(table)


class RealRegionSIntegrationTests(unittest.TestCase):
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
    def test_ising_closed_physical_endpoint_golden_values(self):
        table = ForwardTable.load_validated(self.ISING_METADATA)

        strip = build_region_s(table)

        expected = np.asarray(
            [
                [
                    0.0928908,
                    0.602,
                    0.0007763574576589513,
                    0.0006556514700511366,
                    0.0007763574576589513,
                    0.0006556514700511366,
                ],
                [
                    0.0802978,
                    0.651,
                    0.0005687184383965958,
                    0.0005138427219773829,
                    0.000588048857397477,
                    0.0005321305835067239,
                ],
                [
                    0.0673,
                    0.7,
                    0.0004094634379541152,
                    0.0003879667661805674,
                    0.0004219713300019678,
                    0.0004002219703685419,
                ],
            ]
        )
        np.testing.assert_allclose(
            strip.endpoints.as_array()[[0, 49, 98]],
            expected,
            rtol=5.0e-10,
            atol=1.0e-13,
        )
        self.assertEqual(strip.endpoints.point_count, 99)
        self.assertEqual(strip.cell_shape, (98, 1))
        self.assertEqual(strip.endpoints.target_segment_norms[0], 0.0)
        self.assertEqual(
            strip.diagnostics.energy_orientation_counts,
            {"qgp_greater": 98, "equal": 1, "hadronic_greater": 0},
        )
        self.assertEqual(
            strip.diagnostics.density_orientation_counts,
            {"qgp_greater": 98, "equal": 1, "hadronic_greater": 0},
        )
        self.assertEqual(strip.diagnostics.rejected_slices, ())
        self.assertEqual(
            strip.diagnostics.config["cross_slice_smoothing"], "none"
        )
        self.assertEqual(
            strip.diagnostics.config["critical_endpoint_policy"],
            "numerical_physical_center_separation_closure",
        )
        self.assertEqual(
            strip.diagnostics.config["critical_closure_match_muB_GeV"],
            0.615,
        )
        self.assertAlmostEqual(
            strip.diagnostics.config[
                "critical_closure_effective_width_GeV"
            ],
            0.013,
        )
        self.assertEqual(
            strip.diagnostics.config["critical_closure_adaptive_expansions"],
            8,
        )
        self.assertTrue(
            strip.diagnostics.config["critical_closure_geometry_validated"]
        )
        match = 13
        scaled_hadronic_energy = (
            strip.endpoints.e_hadronic / strip.endpoints.temperature**4
        )
        scaled_hadronic_density = (
            strip.endpoints.nB_hadronic / strip.endpoints.temperature**3
        )
        self.assertTrue(
            np.all(np.diff(scaled_hadronic_energy[: match + 2]) > 0.0)
        )
        self.assertTrue(
            np.all(np.diff(scaled_hadronic_density[: match + 2]) > 0.0)
        )
        self.assertTrue(
            np.all(
                np.diff(strip.endpoints.target_segment_norms[: match + 1])
                >= 0.0
            )
        )

    @unittest.skipUnless(CEC_TABLE.is_file(), "local CEC table is unavailable")
    def test_binodal_table_does_not_build_region_s(self):
        table = ForwardTable.load_validated(self.CEC_METADATA)

        with self.assertRaisesRegex(
            TransitionConfigurationError, "requires a validated LINE"
        ):
            build_region_s(table)


if __name__ == "__main__":
    unittest.main()
