"""Tests for phase-tagged LINE and BINODALS target-image candidates."""

import unittest

import numpy as np

from eos_trensmuther import (
    BinodalBand,
    BinodalSubdivisionConfig,
    CellImageIndex,
    CellImageKind,
    MixedPhaseCandidateIndex,
    MultilinearInterpolator,
    ReachabilityConfigurationError,
    build_rectilinear_cell_images,
)


class SyntheticBinodalImageTests(unittest.TestCase):
    @staticmethod
    def target_values(points):
        points = np.asarray(points, dtype=float)
        temperature = points[..., 0]
        chemical_potential = points[..., 1]
        return np.stack(
            (
                2.0 * temperature
                + chemical_potential
                + 0.25 * temperature * chemical_potential,
                temperature
                + 3.0 * chemical_potential
                - 0.1 * temperature * chemical_potential,
            ),
            axis=-1,
        )

    def make_interpolator(self):
        temperature = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
        chemical_potential = np.asarray([0.0, 0.5, 1.0])
        T, muB = np.meshgrid(temperature, chemical_potential, indexing="ij")
        targets = self.target_values(np.stack((T, muB), axis=-1))
        return MultilinearInterpolator.from_rectilinear_map(
            (temperature, chemical_potential),
            {"e": targets[..., 0], "nB": targets[..., 1]},
            coordinate_names=("T", "muB"),
            target_names=("e", "nB"),
            coordinate_units=("GeV", "GeV"),
            target_units=("GeV^4", "GeV^3"),
        )

    @staticmethod
    def make_band():
        return BinodalBand(
            muB=np.asarray([0.0, 0.5, 1.0]),
            dilute_temperature=np.asarray([0.25, 0.30, 0.35]),
            dense_temperature=np.asarray([0.65, 0.70, 0.75]),
            sidecar_fingerprint="synthetic-binodal-sidecar",
            kind="synthetic sloped band",
        )

    def test_subdivided_images_cover_every_sampled_mixed_point(self):
        interpolator = self.make_interpolator()
        band = self.make_band()
        mixed = MixedPhaseCandidateIndex.for_binodals(
            interpolator,
            band,
            residual_scales=[1.0, 1.0],
            root_residual_tolerance=0.0,
            subdivision=BinodalSubdivisionConfig(
                maximum_depth=10,
                temperature_tolerance=0.005,
                chemical_potential_tolerance=0.005,
            ),
        )

        muB = np.linspace(0.0, 1.0, 41)
        bounds = band.temperature_bounds(muB)
        fractions = np.linspace(0.0, 1.0, 11)
        temperatures = (
            bounds[:, 0, None] * (1.0 - fractions[None, :])
            + bounds[:, 1, None] * fractions[None, :]
        )
        source = np.stack(
            (
                temperatures,
                np.broadcast_to(muB[:, None], temperatures.shape),
            ),
            axis=-1,
        )
        targets = self.target_values(source)
        result = mixed.classify(targets)

        self.assertTrue(np.all(result.candidate))
        self.assertTrue(np.all(result.exact_candidate))
        self.assertTrue(np.all(mixed.verify_source(source)))
        self.assertGreater(mixed.images.image_count, 0)
        self.assertTrue(np.any(mixed.images.boundary_candidate_mask))
        self.assertTrue(
            np.any(
                mixed.images.kind_codes
                == int(CellImageKind.BINODALS_INTERIOR)
            )
        )
        self.assertFalse(mixed.images.source_lower.flags.writeable)

        # Sidecar knots are explicit split locations; no retained source box
        # may straddle the interior muB=0.5 knot.
        straddles_knot = (
            (mixed.images.source_lower[:, 1] < 0.5)
            & (mixed.images.source_upper[:, 1] > 0.5)
        )
        self.assertFalse(np.any(straddles_knot))

    def test_boundary_box_false_positive_fails_exact_phase_verification(self):
        interpolator = self.make_interpolator()
        band = self.make_band()
        coarse = MixedPhaseCandidateIndex.for_binodals(
            interpolator,
            band,
            residual_scales=[1.0, 1.0],
            root_residual_tolerance=0.0,
            subdivision=BinodalSubdivisionConfig(
                maximum_depth=0,
                temperature_tolerance=0.0,
                chemical_potential_tolerance=0.0,
            ),
        )
        outside_source = np.asarray([0.10, 0.25])
        outside_target = self.target_values(outside_source)

        result = coarse.classify(outside_target)

        self.assertTrue(bool(result.candidate))
        self.assertTrue(
            bool(result.has_kind(CellImageKind.BINODALS_BOUNDARY))
        )
        self.assertFalse(bool(coarse.verify_source(outside_source)))

    def test_tagged_collection_union_reports_overlapping_provenance(self):
        interpolator = self.make_interpolator()
        band = self.make_band()
        scales = [1.0, 1.0]
        ordinary = build_rectilinear_cell_images(
            interpolator, scales, 0.0, kind=CellImageKind.ORDINARY
        )
        mixed = MixedPhaseCandidateIndex.for_binodals(
            interpolator,
            band,
            scales,
            0.0,
            subdivision=BinodalSubdivisionConfig(maximum_depth=6),
        )
        combined = CellImageIndex((ordinary, mixed.images))
        source = np.asarray([0.5, 0.5])
        result = combined.classify(self.target_values(source))

        self.assertTrue(bool(result.has_kind(CellImageKind.ORDINARY)))
        self.assertTrue(
            bool(result.has_kind(CellImageKind.BINODALS_INTERIOR))
        )
        self.assertTrue(bool(mixed.verify_source(source)))

    def test_subdivision_resource_limit_fails_instead_of_truncating(self):
        with self.assertRaisesRegex(
            ReachabilityConfigurationError, "maximum_images"
        ):
            MixedPhaseCandidateIndex.for_binodals(
                self.make_interpolator(),
                self.make_band(),
                [1.0, 1.0],
                0.0,
                subdivision=BinodalSubdivisionConfig(
                    maximum_depth=6, maximum_images=1
                ),
            )


if __name__ == "__main__":
    unittest.main()
