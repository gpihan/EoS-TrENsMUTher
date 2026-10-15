"""Tests for proof-safe cell envelopes and serial reachability screening."""

import itertools
import unittest
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    LocalizationCompleteness,
    MultilinearInterpolator,
    PipelineConfig,
    PipelinePlan,
    PrescreenStatus,
    ReachabilityConfig,
    ReachabilityConfigurationError,
    ReachabilityIndex,
    ReachabilityInputError,
    ReachabilityLocalization,
    ReachabilityLocalizationError,
    ReachabilityProvenance,
    ReachabilityResult,
    pipeline_status_code,
)

from tests.test_tables import TemporaryEoS, field_spec, metadata


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ReachabilityTestCase(unittest.TestCase):
    def make_gap_table(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
        }
        rows = []
        for temperature_index, temperature in enumerate((0.1, 0.2, 0.3)):
            for chemical_potential in (0.0, 1.0):
                energy = 1.0 + temperature_index
                density = (
                    chemical_potential
                    if temperature_index < 2
                    else 10.0 + chemical_potential
                )
                rows.append([temperature, chemical_potential, energy, density])
        fixture = TemporaryEoS(metadata(fields, 2), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable(fixture.metadata_path)
        self.assertTrue(table.validate().is_valid)
        return table

    def make_affine_table(self, dimension):
        coordinate_names = ("T", "muB", "muQ", "muS")[:dimension]
        target_names = ("e", "nB", "nQ", "nS")[:dimension]
        fields = {
            name: field_spec("coordinate", index)
            for index, name in enumerate(coordinate_names)
        }
        for index, name in enumerate(target_names, start=dimension):
            fields[name] = field_spec("target", index)
        axes = tuple(
            np.asarray([0.1 + 0.05 * axis, 0.3 + 0.05 * axis, 0.6 + 0.05 * axis])
            for axis in range(dimension)
        )
        rows = []
        coefficient = np.arange(1.0, dimension + 1.0)
        for point in itertools.product(*axes):
            values = np.asarray(point)
            targets = [2.0 + np.dot(coefficient, values)]
            for field_index in range(1, dimension):
                weights = np.roll(coefficient, field_index)
                targets.append(-0.1 * field_index + 0.03 * np.dot(weights, values))
            rows.append(list(point) + targets)
        fixture = TemporaryEoS(metadata(fields, dimension), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable(fixture.metadata_path)
        self.assertTrue(table.validate().is_valid)
        return table


class CellEnclosureTests(ReachabilityTestCase):
    def test_proof_safe_cell_enclosures_match_corner_extrema(self):
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                table = self.make_affine_table(dimension)
                interpolator = MultilinearInterpolator(table)
                cell_flat = np.arange(np.prod(interpolator.cell_shape), dtype=np.int64)
                cells = np.column_stack(
                    np.unravel_index(cell_flat, interpolator.cell_shape, order="C")
                )
                corners = interpolator.cell_corner_values(cells)
                enclosures = interpolator.cell_value_enclosures(cells)

                self.assertTrue(enclosures.rigorous)
                np.testing.assert_array_equal(
                    enclosures.lower, np.min(corners, axis=1)
                )
                np.testing.assert_array_equal(
                    enclosures.upper, np.max(corners, axis=1)
                )
                self.assertFalse(enclosures.lower.flags.writeable)

    def test_exact_and_outward_routing_boxes_are_separate(self):
        interpolator = MultilinearInterpolator(self.make_gap_table())
        index = ReachabilityIndex(
            interpolator,
            residual_scales={"e": 1.0, "nB": 1.0},
            root_residual_tolerance=0.1,
        )
        boxes = index.enclosures

        self.assertTrue(boxes.rigorous)
        self.assertTrue(boxes.outward_rounded)
        self.assertTrue(np.all(boxes.routing_padding >= 0.1))
        self.assertTrue(np.all(boxes.routing_lower < boxes.exact_lower))
        self.assertTrue(np.all(boxes.routing_upper > boxes.exact_upper))
        self.assertFalse(boxes.exact_lower.flags.writeable)

        result = index.classify([[1.5, 1.05], [1.5, 1.2]])
        np.testing.assert_array_equal(result.candidate_mask, [True, False])
        np.testing.assert_array_equal(result.exact_candidate_mask, [False, False])
        np.testing.assert_array_equal(result.tolerance_shell_mask, [True, False])
        np.testing.assert_array_equal(
            result.definitely_unreachable_mask, [False, True]
        )


class ReferenceClassifierTests(ReachabilityTestCase):
    def test_configuration_rejects_noncanonical_fields_and_unsafe_inputs(self):
        table = self.make_gap_table()
        interpolator = MultilinearInterpolator(table)
        auxiliary_like = MultilinearInterpolator(table, ("e",))

        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityIndex(auxiliary_like, [1.0], 0.0)
        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityIndex(interpolator, [True, 1.0], 0.0)
        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityIndex(interpolator, [1.0, 0.0], 0.0)
        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityIndex(interpolator, {"e": 1.0, 7: 1.0}, 0.0)
        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityIndex(
                interpolator,
                [1.0, 1.0],
                0.0,
                active_cell_mask=np.ones(interpolator.cell_shape, dtype=int),
            )
        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityConfig(
                cell_batch_size=1024,
                query_batch_size=1024,
                max_pairwise_entries=1024,
            )
        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityConfig(method="unknown")
        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityConfig(method="exhaustive", hierarchy_levels=3)
        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityConfig(method="hierarchical")
        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityConfig(method="hierarchical", hierarchy_levels=4)
        for development_depth in (3, 5):
            with self.subTest(development_depth=development_depth):
                with self.assertRaises(ReachabilityConfigurationError):
                    ReachabilityConfig(
                        method="hierarchical",
                        hierarchy_levels=development_depth,
                    )

    def test_hierarchical_4d_method_matches_explicit_exhaustive_method(self):
        random = np.random.RandomState(4031)
        interpolator = MultilinearInterpolator(self.make_affine_table(4))
        scales = np.maximum(
            interpolator.node_value_bounds.upper
            - interpolator.node_value_bounds.lower,
            1.0,
        )
        source_points = np.column_stack(
            [
                random.uniform(axis[0], axis[-1], size=75)
                for axis in interpolator.axes
            ]
        )
        targets = np.vstack(
            [
                interpolator(source_points),
                interpolator.node_value_bounds.lower - 1.0,
                np.full(4, np.nan),
            ]
        )
        exhaustive_index = ReachabilityIndex(
            interpolator,
            scales,
            1.0e-10,
            config=ReachabilityConfig(
                method="exhaustive",
                query_batch_size=7,
                cell_batch_size=3,
            ),
        )
        exhaustive = exhaustive_index.classify(targets)
        exhaustive_localization = exhaustive_index.localize(targets)

        # The lower depths remain available through the private hierarchy
        # benchmark API.  Production ReachabilityConfig intentionally exposes
        # only the depth selected by the HRG4D viability study.
        for depth in (7,):
            with self.subTest(depth=depth):
                index = ReachabilityIndex(
                    interpolator,
                    scales,
                    1.0e-10,
                    config=ReachabilityConfig(
                        method="hierarchical",
                        hierarchy_levels=depth,
                        query_batch_size=11,
                        cell_batch_size=2,
                    ),
                )
                result = index.classify(targets)
                localization = index.localize(targets)
                self.assertIsNotNone(index.hierarchy)
                np.testing.assert_array_equal(
                    result.prescreen_codes, exhaustive.prescreen_codes
                )
                np.testing.assert_array_equal(
                    result.exact_candidate, exhaustive.exact_candidate
                )
                np.testing.assert_array_equal(
                    result.routing_witness_flat,
                    exhaustive.routing_witness_flat,
                )
                np.testing.assert_array_equal(
                    result.exact_witness_flat,
                    exhaustive.exact_witness_flat,
                )
                np.testing.assert_array_equal(
                    localization.result.prescreen_codes,
                    exhaustive_localization.result.prescreen_codes,
                )
                np.testing.assert_array_equal(
                    localization.offsets,
                    exhaustive_localization.offsets,
                )
                np.testing.assert_array_equal(
                    localization.cell_flat_indices,
                    exhaustive_localization.cell_flat_indices,
                )
                localization.require_complete_for(index.provenance)

    def test_hierarchical_method_rejects_non_4d_and_sparse_domains(self):
        two_dimensional = MultilinearInterpolator(self.make_affine_table(2))
        hierarchical = ReachabilityConfig(
            method="hierarchical", hierarchy_levels=7
        )
        with self.assertRaisesRegex(
            ReachabilityConfigurationError, "only in 4D"
        ):
            ReachabilityIndex(
                two_dimensional,
                [1.0, 1.0],
                0.0,
                config=hierarchical,
            )

        four_dimensional = MultilinearInterpolator(self.make_affine_table(4))
        active = np.ones(four_dimensional.cell_shape, dtype=bool)
        active.reshape(-1)[-1] = False
        with self.assertRaisesRegex(
            ReachabilityConfigurationError, "full active-cell lattice"
        ):
            ReachabilityIndex(
                four_dimensional,
                np.ones(4),
                0.0,
                active_cell_mask=active,
                config=hierarchical,
            )

    def test_reachability_configuration_dictionary_round_trip(self):
        configured = ReachabilityConfig(
            method="hierarchical",
            hierarchy_levels=7,
            query_batch_size=13,
            cell_batch_size=17,
            max_pairwise_entries=13 * 17,
            max_hierarchy_bytes=123456,
        )
        document = {
            field.name: getattr(configured, field.name)
            for field in dataclass_fields(configured)
        }
        self.assertEqual(ReachabilityConfig(**document), configured)

    def test_global_component_envelope_point_can_be_proved_unreachable(self):
        interpolator = MultilinearInterpolator(self.make_gap_table())
        index = ReachabilityIndex(interpolator, [1.0, 1.0], 0.0)

        # It lies inside global node ranges e=[1,3], nB=[0,11], but in neither
        # cell box: cell 0 has nB<=1 and cell 1 has e>=2.
        result = index.classify([1.5, 5.0])

        self.assertTrue(bool(result.evaluated_mask))
        self.assertFalse(bool(result.candidate_mask))
        self.assertTrue(bool(result.definitely_unreachable_mask))
        self.assertTrue(bool(result.exact_excluded_mask))
        self.assertEqual(int(result.routing_witness_flat), -1)

    def test_random_forward_samples_are_never_screened_out(self):
        random = np.random.RandomState(2943)
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                table = self.make_affine_table(dimension)
                interpolator = MultilinearInterpolator(table)
                scales = np.maximum(
                    interpolator.node_value_bounds.upper
                    - interpolator.node_value_bounds.lower,
                    1.0,
                )
                index = ReachabilityIndex(interpolator, scales, 1.0e-12)
                points = np.column_stack(
                    [
                        random.uniform(axis[0], axis[-1], size=250)
                        for axis in interpolator.axes
                    ]
                )
                targets = interpolator(points)
                result = index.classify(targets)

                self.assertTrue(np.all(result.evaluated_mask))
                self.assertTrue(np.all(result.candidate_mask))
                self.assertFalse(np.any(result.definitely_unreachable_mask))
                self.assertTrue(np.all(result.routing_witness_flat >= 0))

    def test_cell_and_domain_faces_are_included(self):
        table = self.make_gap_table()
        interpolator = MultilinearInterpolator(table)
        index = ReachabilityIndex(interpolator, [1.0, 1.0], 0.0)
        nodes = np.asarray(list(itertools.product(*interpolator.axes)))
        targets = interpolator(nodes)

        result = index.classify(targets)

        self.assertTrue(np.all(result.exact_candidate_mask))
        self.assertTrue(np.all(result.candidate_mask))
        # The shared cell face uses the smallest flat containing cell as the
        # deterministic witness.
        shared = index.classify([2.0, 0.5])
        self.assertEqual(int(shared.routing_witness_flat), 0)

    def test_exact_and_routing_witnesses_use_independent_c_order_minima(self):
        interpolator = MultilinearInterpolator(self.make_gap_table())
        index = ReachabilityIndex(interpolator, [1.0, 1.0], 0.1)

        # Cell 0 contains this point only in its expanded routing box. Cell 1
        # contains it in its exact box. The two public witnesses therefore
        # have distinct, independently minimized C-order cell identities.
        result = index.classify([2.05, 1.05])

        self.assertTrue(bool(result.candidate_mask))
        self.assertTrue(bool(result.exact_candidate_mask))
        self.assertFalse(bool(result.tolerance_shell_mask))
        self.assertEqual(int(result.routing_witness_flat), 0)
        self.assertEqual(int(result.exact_witness_flat), 1)

    def test_reference_classification_and_witnesses_are_batch_invariant(self):
        interpolator = MultilinearInterpolator(self.make_gap_table())
        targets = np.asarray(
            [
                [2.05, 1.05],  # routing witness 0, exact witness 1
                [2.0, 0.5],    # shared face, both witnesses 0
                [1.5, 1.05],   # tolerance-shell-only candidate
                [1.5, 5.0],    # definitely unreachable
                [np.nan, 0.0], # invalid and therefore not run
            ]
        )
        configurations = (
            ReachabilityConfig(query_batch_size=1, cell_batch_size=1),
            ReachabilityConfig(query_batch_size=2, cell_batch_size=2),
            ReachabilityConfig(query_batch_size=5, cell_batch_size=1),
        )
        results = [
            ReachabilityIndex(
                interpolator,
                [1.0, 1.0],
                0.1,
                config=configuration,
            ).classify(targets)
            for configuration in configurations
        ]

        reference = results[0]
        for result in results[1:]:
            np.testing.assert_array_equal(
                result.prescreen_codes, reference.prescreen_codes
            )
            np.testing.assert_array_equal(
                result.exact_candidate, reference.exact_candidate
            )
            np.testing.assert_array_equal(
                result.routing_witness_flat, reference.routing_witness_flat
            )
            np.testing.assert_array_equal(
                result.exact_witness_flat, reference.exact_witness_flat
            )

    def test_complete_exhaustive_localization_matches_direct_membership_in_1d_to_4d(self):
        random = np.random.RandomState(8817)
        for dimension in range(1, 5):
            with self.subTest(dimension=dimension):
                table = self.make_affine_table(dimension)
                interpolator = MultilinearInterpolator(table)
                scales = np.maximum(
                    interpolator.node_value_bounds.upper
                    - interpolator.node_value_bounds.lower,
                    1.0,
                )
                points = np.column_stack(
                    [
                        random.uniform(axis[0], axis[-1], size=12)
                        for axis in interpolator.axes
                    ]
                )
                targets = np.vstack(
                    [
                        interpolator(points),
                        interpolator.node_value_bounds.upper + 10.0,
                        np.full(dimension, np.nan),
                    ]
                )
                valid = np.ones(targets.shape[0], dtype=bool)
                valid[2] = False
                configurations = (
                    ReachabilityConfig(
                        method="exhaustive",
                        query_batch_size=1,
                        cell_batch_size=1,
                    ),
                    ReachabilityConfig(
                        method="exhaustive",
                        query_batch_size=7,
                        cell_batch_size=3,
                    ),
                )
                localizations = []
                for configuration in configurations:
                    index = ReachabilityIndex(
                        interpolator,
                        scales,
                        1.0e-10,
                        config=configuration,
                    )
                    localization = index.localize(targets, valid_mask=valid)
                    classification = index.classify(targets, valid_mask=valid)
                    np.testing.assert_array_equal(
                        localization.result.prescreen_codes,
                        classification.prescreen_codes,
                    )
                    np.testing.assert_array_equal(
                        localization.result.exact_candidate,
                        classification.exact_candidate,
                    )
                    np.testing.assert_array_equal(
                        localization.result.routing_witness_flat,
                        classification.routing_witness_flat,
                    )
                    np.testing.assert_array_equal(
                        localization.result.exact_witness_flat,
                        classification.exact_witness_flat,
                    )
                    localization.require_complete_for(index.provenance)
                    self.assertEqual(
                        index.provenance.table_fingerprint,
                        table.snapshot_fingerprint(),
                    )

                    expected_offsets = [0]
                    expected_cells = []
                    evaluated = valid & np.all(np.isfinite(targets), axis=1)
                    boxes = index.enclosures
                    for target, is_valid in zip(targets, evaluated):
                        if is_valid:
                            contains = np.all(
                                (target >= boxes.routing_lower)
                                & (target <= boxes.routing_upper),
                                axis=1,
                            )
                            expected_cells.extend(
                                boxes.cell_flat_indices[contains].tolist()
                            )
                        expected_offsets.append(len(expected_cells))
                    np.testing.assert_array_equal(
                        localization.offsets, expected_offsets
                    )
                    np.testing.assert_array_equal(
                        localization.cell_flat_indices,
                        np.asarray(expected_cells, dtype=np.int64),
                    )
                    localizations.append(localization)

                np.testing.assert_array_equal(
                    localizations[0].offsets, localizations[1].offsets
                )
                np.testing.assert_array_equal(
                    localizations[0].cell_flat_indices,
                    localizations[1].cell_flat_indices,
                )

    def test_sparse_active_domain_localization_retains_global_c_order_ids(self):
        interpolator = MultilinearInterpolator(self.make_affine_table(3))
        active = np.ones(interpolator.cell_shape, dtype=bool)
        active.reshape(-1, order="C")[[1, 3, 6]] = False
        scales = np.maximum(
            interpolator.node_value_bounds.upper
            - interpolator.node_value_bounds.lower,
            1.0,
        )
        sparse = ReachabilityIndex(
            interpolator,
            scales,
            0.0,
            active_cell_mask=active,
            config=ReachabilityConfig(query_batch_size=2, cell_batch_size=2),
        )
        full = ReachabilityIndex(interpolator, scales, 0.0)
        targets = interpolator(
            np.asarray(list(itertools.product(*interpolator.axes)))
        )
        localization = sparse.localize(targets)
        allowed = set(sparse.enclosures.cell_flat_indices.tolist())

        self.assertTrue(
            all(int(cell) in allowed for cell in localization.cell_flat_indices)
        )
        self.assertNotEqual(
            sparse.provenance.active_cell_fingerprint,
            full.provenance.active_cell_fingerprint,
        )
        localization.require_complete_for(sparse.provenance)
        with self.assertRaisesRegex(
            ReachabilityLocalizationError, "active_cell_fingerprint"
        ):
            localization.require_complete_for(full.provenance)

    def test_invalid_rows_remain_not_run_instead_of_becoming_no_root(self):
        interpolator = MultilinearInterpolator(self.make_gap_table())
        index = ReachabilityIndex(interpolator, [1.0, 1.0], 0.0)
        targets = np.asarray([[np.nan, 0.0], [1.5, 0.5], [1.5, 5.0]])
        valid_mask = np.asarray([True, False, True])

        result = index.classify(targets, valid_mask=valid_mask)

        not_run = pipeline_status_code(PrescreenStatus.NOT_RUN)
        self.assertEqual(int(result.prescreen_codes[0]), not_run)
        self.assertEqual(int(result.prescreen_codes[1]), not_run)
        self.assertFalse(result.evaluated_mask[0])
        self.assertFalse(result.evaluated_mask[1])
        self.assertTrue(result.definitely_unreachable_mask[2])
        self.assertFalse(result.candidate_mask[0])
        with self.assertRaises(ReachabilityInputError):
            index.classify(targets, valid_mask=np.asarray([1, 0, 1]))

    def test_active_cell_mask_changes_the_admissible_domain(self):
        table = self.make_gap_table()
        interpolator = MultilinearInterpolator(table)
        active = np.asarray([[True], [False]])
        index = ReachabilityIndex(
            interpolator,
            [1.0, 1.0],
            0.0,
            active_cell_mask=active,
        )
        target_in_active_cell = interpolator([0.15, 0.5])
        target_in_excluded_cell = interpolator([0.3, 1.0])

        result = index.classify(
            np.vstack((target_in_active_cell, target_in_excluded_cell))
        )

        np.testing.assert_array_equal(result.candidate_mask, [True, False])
        np.testing.assert_array_equal(
            result.definitely_unreachable_mask, [False, True]
        )
        np.testing.assert_array_equal(result.exact_candidate_mask, [True, False])
        np.testing.assert_array_equal(result.routing_witness_flat, [0, -1])
        np.testing.assert_array_equal(result.exact_witness_flat, [0, -1])
        np.testing.assert_array_equal(index.enclosures.cell_flat_indices, [0])
        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityIndex(
                interpolator,
                [1.0, 1.0],
                0.0,
                active_cell_mask=np.zeros(interpolator.cell_shape, dtype=bool),
            )

    def test_enclosure_memory_guard_precedes_box_allocation(self):
        interpolator = MultilinearInterpolator(self.make_gap_table())
        # 2 cells * (one int64 + four 2-component float64 boxes) = 144 bytes.
        with self.assertRaises(ReachabilityConfigurationError):
            ReachabilityIndex(
                interpolator,
                [1.0, 1.0],
                0.0,
                config=ReachabilityConfig(max_enclosure_bytes=143),
            )
        index = ReachabilityIndex(
            interpolator,
            [1.0, 1.0],
            0.0,
            config=ReachabilityConfig(max_enclosure_bytes=144),
        )
        self.assertEqual(index.estimated_enclosure_bytes, 144)

    def test_pipeline_plan_and_grid_chunk_integration(self):
        table = self.make_gap_table()
        envelope = table.find_boundaries(space="target")
        config = PipelineConfig(
            grid=InversionGridSpec(
                "target",
                {name: GridAxisSpec(4) for name in envelope.names},
            ),
            residual_scales={"e": 1.0, "nB": 1.0},
            auxiliary_fields=(),
        )
        plan = PipelinePlan.from_table(table, config)
        interpolator = MultilinearInterpolator(table)
        index = ReachabilityIndex.from_plan(
            interpolator,
            plan,
            config=ReachabilityConfig(query_batch_size=3, cell_batch_size=1),
        )
        chunks = list(plan.iter_chunks())
        results = [index.classify_chunk(chunk) for chunk in chunks]

        codes = np.concatenate(
            [result.prescreen_codes.reshape(-1) for result in results]
        )
        self.assertEqual(codes.size, plan.grid.size)
        self.assertTrue(
            np.all(
                np.isin(
                    codes,
                    (
                        pipeline_status_code(PrescreenStatus.CANDIDATE),
                        pipeline_status_code(
                            PrescreenStatus.DEFINITELY_UNREACHABLE
                        ),
                    ),
                )
            )
        )
        self.assertTrue(
            any(np.any(result.definitely_unreachable_mask) for result in results)
        )
        first_localization = index.localize_chunk(chunks[0])
        direct_localization = index.localize(
            chunks[0].canonical_targets,
            valid_mask=chunks[0].transform_valid,
        )
        np.testing.assert_array_equal(
            first_localization.offsets, direct_localization.offsets
        )
        np.testing.assert_array_equal(
            first_localization.cell_flat_indices,
            direct_localization.cell_flat_indices,
        )

    @unittest.skipUnless(
        (REPOSITORY_ROOT / "EoSTables/2DIsing/EoSmub602.dat").is_file()
        and (
            REPOSITORY_ROOT
            / "EoSTables/CEC/hic_table_T80-300_muB0-800_d1.dat"
        ).is_file(),
        "external CEC and Ising scientific tables are unavailable",
    )
    def test_real_ising_and_cec_forward_points_are_candidates(self):
        for relative_path in (
            "EoSTables/2DIsing/metadata.json",
            "EoSTables/CEC/metadata.json",
        ):
            with self.subTest(table=relative_path):
                table = ForwardTable.load_validated(REPOSITORY_ROOT / relative_path)
                interpolator = MultilinearInterpolator(table)
                spans = (
                    interpolator.node_value_bounds.upper
                    - interpolator.node_value_bounds.lower
                )
                scales = np.maximum(spans, np.finfo(float).tiny)
                index = ReachabilityIndex(interpolator, scales, 1.0e-10)
                source_points = np.asarray(
                    [
                        [axis[0] for axis in interpolator.axes],
                        [0.5 * (axis[0] + axis[-1]) for axis in interpolator.axes],
                        [axis[-1] for axis in interpolator.axes],
                    ]
                )
                result = index.classify(interpolator(source_points))

                self.assertTrue(np.all(result.candidate_mask))
                self.assertFalse(np.any(result.definitely_unreachable_mask))


class ReachabilityLocalizationContractTests(unittest.TestCase):
    def make_result(self):
        return ReachabilityResult(
            field_names=("e", "nB"),
            cell_shape=(2, 2),
            targets=np.asarray(
                [[0.2, 0.3], [0.4, 0.5], [2.0, 2.0], [np.nan, 0.0]]
            ),
            prescreen_codes=np.asarray(
                [
                    pipeline_status_code(PrescreenStatus.CANDIDATE),
                    pipeline_status_code(PrescreenStatus.CANDIDATE),
                    pipeline_status_code(PrescreenStatus.DEFINITELY_UNREACHABLE),
                    pipeline_status_code(PrescreenStatus.NOT_RUN),
                ]
            ),
            exact_candidate=np.asarray([True, True, False, False]),
            routing_witness_flat=np.asarray([0, 0, -1, -1]),
            exact_witness_flat=np.asarray([0, 1, -1, -1]),
        )

    def make_provenance(self, **overrides):
        values = {
            "table_fingerprint": "table-snapshot-a",
            "active_cell_fingerprint": "active-domain-a",
            "coordinate_names": ("T", "muB"),
            "field_names": ("e", "nB"),
            "cell_shape": (2, 2),
            "active_cell_count": 4,
            "residual_scales": np.asarray([2.0, 3.0]),
            "root_residual_tolerance": 1.0e-10,
            "method": "exhaustive",
            "hierarchy_levels": None,
        }
        values.update(overrides)
        return ReachabilityProvenance(**values)

    def make_localization(self, **overrides):
        values = {
            "result": self.make_result(),
            "offsets": np.asarray([0, 2, 4, 4, 4]),
            "cell_flat_indices": np.asarray([0, 2, 0, 1]),
            "completeness": LocalizationCompleteness.COMPLETE,
            "provenance": self.make_provenance(),
        }
        values.update(overrides)
        return ReachabilityLocalization(**values)

    def test_csr_rows_are_immutable_consistent_and_byte_accounted(self):
        localization = self.make_localization()

        self.assertEqual(localization.target_count, 4)
        self.assertEqual(localization.pair_count, 4)
        self.assertEqual(localization.csr_nbytes, 9 * np.dtype(np.int64).itemsize)
        self.assertEqual(localization.nbytes, localization.csr_nbytes)
        np.testing.assert_array_equal(localization.candidate_counts, [2, 2, 0, 0])
        np.testing.assert_array_equal(localization.cells_for(0), [0, 2])
        np.testing.assert_array_equal(localization.cells_for(1), [0, 1])
        self.assertTrue(
            np.shares_memory(localization.cells_for(1), localization.cell_flat_indices)
        )
        self.assertFalse(localization.offsets.flags.writeable)
        self.assertFalse(localization.cell_flat_indices.flags.writeable)
        self.assertFalse(localization.provenance.residual_scales.flags.writeable)
        with self.assertRaises(ReachabilityLocalizationError):
            localization.cells_for(True)
        with self.assertRaises(ReachabilityLocalizationError):
            localization.cells_for(4)

    def test_only_complete_matching_provenance_authorizes_global_exclusion(self):
        expected = self.make_provenance()
        complete = self.make_localization(provenance=expected)

        self.assertTrue(complete.is_authoritative_for(expected))
        self.assertIsNone(complete.require_complete_for(expected))

        incomplete = self.make_localization(
            completeness=LocalizationCompleteness.INCOMPLETE,
            provenance=expected,
        )
        self.assertFalse(incomplete.is_authoritative_for(expected))
        with self.assertRaisesRegex(
            ReachabilityLocalizationError, "incomplete candidate localization"
        ):
            incomplete.require_complete_for(expected)

        mismatched = self.make_provenance(
            active_cell_fingerprint="active-domain-b"
        )
        self.assertFalse(complete.is_authoritative_for(mismatched))
        self.assertEqual(
            complete.provenance.compatibility_mismatches(mismatched),
            ("active_cell_fingerprint",),
        )
        with self.assertRaisesRegex(
            ReachabilityLocalizationError, "active_cell_fingerprint"
        ):
            complete.require_complete_for(mismatched)

    def test_malformed_csr_or_witness_relationships_are_rejected(self):
        invalid_cases = {
            "offset length": {
                "offsets": np.asarray([0, 2, 4, 4]),
            },
            "offset endpoint": {
                "offsets": np.asarray([0, 2, 3, 3, 3]),
            },
            "candidate without row": {
                "offsets": np.asarray([0, 0, 2, 2, 2]),
                "cell_flat_indices": np.asarray([0, 1]),
            },
            "excluded target with row": {
                "offsets": np.asarray([0, 2, 4, 5, 5]),
                "cell_flat_indices": np.asarray([0, 2, 0, 1, 3]),
            },
            "duplicate row IDs": {
                "cell_flat_indices": np.asarray([0, 0, 0, 1]),
            },
            "unsorted row IDs": {
                "cell_flat_indices": np.asarray([2, 0, 0, 1]),
            },
            "routing witness mismatch": {
                "cell_flat_indices": np.asarray([1, 2, 0, 1]),
            },
            "exact witness missing": {
                "cell_flat_indices": np.asarray([0, 2, 0, 2]),
            },
            "out-of-domain cell": {
                "cell_flat_indices": np.asarray([0, 4, 0, 1]),
            },
        }
        for label, overrides in invalid_cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ReachabilityLocalizationError):
                    self.make_localization(**overrides)

    def test_result_and_provenance_identity_must_match(self):
        with self.assertRaisesRegex(
            ReachabilityLocalizationError, "fields do not match"
        ):
            self.make_localization(
                provenance=self.make_provenance(field_names=("nB", "e"))
            )
        with self.assertRaisesRegex(
            ReachabilityLocalizationError, "cell shape does not match"
        ):
            self.make_localization(
                provenance=self.make_provenance(
                    cell_shape=(1, 4), active_cell_count=4
                )
            )

    def test_microbatch_plan_never_splits_or_truncates_a_target_row(self):
        localization = self.make_localization()
        batches = localization.plan_microbatches(
            maximum_candidate_pairs=2,
            maximum_csr_bytes=32,
        )

        self.assertEqual(
            [
                (
                    item.target_start,
                    item.target_stop,
                    item.pair_start,
                    item.pair_stop,
                    item.csr_nbytes,
                )
                for item in batches
            ],
            [(0, 1, 0, 2, 32), (1, 2, 2, 4, 32), (2, 4, 4, 4, 24)],
        )
        with self.assertRaisesRegex(
            ReachabilityLocalizationError, "complete row cannot fit"
        ):
            localization.plan_microbatches(
                maximum_candidate_pairs=1,
                maximum_csr_bytes=1024,
            )
        with self.assertRaisesRegex(
            ReachabilityLocalizationError, "complete row cannot fit"
        ):
            localization.plan_microbatches(
                maximum_candidate_pairs=100,
                maximum_csr_bytes=31,
            )

    def test_provenance_rejects_unsafe_contract_fields(self):
        invalid_cases = {
            "empty table fingerprint": {"table_fingerprint": ""},
            "active count": {"active_cell_count": 5},
            "boolean scale": {"residual_scales": [True, 1.0]},
            "nonpositive scale": {"residual_scales": [1.0, 0.0]},
            "negative tolerance": {"root_residual_tolerance": -1.0},
            "exhaustive hierarchy": {"hierarchy_levels": 7},
            "unknown method": {"method": "unknown"},
            "non-C order": {"cell_order": "F"},
        }
        for label, overrides in invalid_cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ReachabilityLocalizationError):
                    self.make_provenance(**overrides)


if __name__ == "__main__":
    unittest.main()
