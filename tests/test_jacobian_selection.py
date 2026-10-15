"""Regressions for conservative local Jacobian branch selection."""

import json
import multiprocessing as mp
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    BinodalBand,
    BinodalSubdivisionConfig,
    CheckpointConfig,
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    GridSpace,
    InversionGridSpec,
    MultilinearInterpolator,
    PhaseAwareRootSolver,
    PhaseRootSelectionReason,
    PipelineConfig,
    ProcessStartMethod,
    ProjectionConfig,
    SolverConfig,
    SlurmArrayConfig,
    TextExportConfig,
    export_text,
    assemble_array_shards,
    create_array_plan,
    execute_array_shard,
    load_pipeline_result,
    save_pipeline_result,
    transition_model_from_table,
)
from eos_trensmuther.phase_solver import (
    _JacobianCellOrientation,
    _bilinear_cell_orientation,
    _local_orientation_decision,
    _scaled_determinant_orientation,
)
from eos_trensmuther.solver import _automatic_residual_scales_from_bounds
from tests.test_tables import TemporaryEoS, field_spec, metadata


def _orientation_interpolator(*, negate=False, mixed_surroundings=False):
    axis = np.arange(5.0)
    energy = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 1, 1, 3],
            [4, 4, 2, 2, 4],
        ],
        dtype=float,
    )
    if mixed_surroundings:
        energy[0, :] = 2.0
    if negate:
        energy = -energy
    density = np.broadcast_to(axis, energy.shape)
    return MultilinearInterpolator.from_rectilinear_map(
        (axis, axis),
        {"e": energy, "nB": density},
        coordinate_names=("T", "muB"),
        target_names=("e", "nB"),
        coordinate_units=("GeV", "GeV"),
        target_units=("GeV^4", "GeV^3"),
    )


def _positive_interpolator():
    axis = np.arange(5.0)
    temperature, chemical_potential = np.meshgrid(axis, axis, indexing="ij")
    return MultilinearInterpolator.from_rectilinear_map(
        (axis, axis),
        {"e": temperature, "nB": chemical_potential},
        coordinate_names=("T", "muB"),
        target_names=("e", "nB"),
        coordinate_units=("GeV", "GeV"),
        target_units=("GeV^4", "GeV^3"),
    )


def _band():
    return BinodalBand(
        muB=np.asarray([0.0, 4.0]),
        dilute_temperature=np.asarray([1.2, 1.2]),
        dense_temperature=np.asarray([2.8, 2.8]),
        sidecar_fingerprint="jacobian-selection-test-band",
        kind="synthetic non-physical band",
    )


class LocalJacobianSelectionTests(unittest.TestCase):
    def test_cell_classification_uses_complete_bilinear_determinant_range(self):
        interpolator = _orientation_interpolator()
        self.assertIs(
            _bilinear_cell_orientation(interpolator, (1, 2)),
            _JacobianCellOrientation.POSITIVE,
        )
        self.assertIs(
            _bilinear_cell_orientation(interpolator, (2, 1)),
            _JacobianCellOrientation.SIGN_CHANGING_OR_SINGULAR,
        )
        self.assertIs(
            _bilinear_cell_orientation(interpolator, (2, 2)),
            _JacobianCellOrientation.NEGATIVE,
        )

    def test_near_zero_rule_is_scaled_by_jacobian_column_norms(self):
        almost_singular = np.asarray(
            [[1.0, 1.0], [1.0, 1.0 + np.finfo(float).eps]]
        )
        self.assertIs(
            _scaled_determinant_orientation(almost_singular),
            _JacobianCellOrientation.SIGN_CHANGING_OR_SINGULAR,
        )
        self.assertIs(
            _scaled_determinant_orientation(almost_singular * 1.0e100),
            _JacobianCellOrientation.SIGN_CHANGING_OR_SINGULAR,
        )

    def test_unanimous_positive_selects_only_positive_candidate(self):
        decision = _local_orientation_decision(
            _orientation_interpolator(), [[1.5, 2.5], [2.5, 2.5]]
        )
        self.assertIs(
            decision.expected_orientation, _JacobianCellOrientation.POSITIVE
        )
        self.assertEqual(
            decision.candidate_orientations,
            (
                _JacobianCellOrientation.POSITIVE,
                _JacobianCellOrientation.NEGATIVE,
            ),
        )
        self.assertEqual(decision.selected_index, 0)

    def test_unanimous_negative_selects_only_negative_candidate(self):
        decision = _local_orientation_decision(
            _orientation_interpolator(negate=True),
            [[1.5, 2.5], [2.5, 2.5]],
        )
        self.assertIs(
            decision.expected_orientation, _JacobianCellOrientation.NEGATIVE
        )
        self.assertEqual(decision.selected_index, 0)

    def test_mixed_surroundings_remain_ambiguous(self):
        decision = _local_orientation_decision(
            _orientation_interpolator(mixed_surroundings=True),
            [[1.5, 2.5], [2.5, 2.5]],
        )
        self.assertIsNone(decision.expected_orientation)
        self.assertIsNone(decision.selected_index)

    def test_two_candidates_matching_unanimous_orientation_remain_ambiguous(self):
        decision = _local_orientation_decision(
            _positive_interpolator(), [[1.5, 1.5], [1.5, 2.5]]
        )
        self.assertIs(
            decision.expected_orientation, _JacobianCellOrientation.POSITIVE
        )
        self.assertEqual(
            decision.candidate_orientations,
            (
                _JacobianCellOrientation.POSITIVE,
                _JacobianCellOrientation.POSITIVE,
            ),
        )
        self.assertIsNone(decision.selected_index)

    def test_near_singular_candidate_remains_ambiguous(self):
        decision = _local_orientation_decision(
            _orientation_interpolator(), [[2.5, 1.5], [2.5, 2.5]]
        )
        self.assertIs(
            decision.expected_orientation, _JacobianCellOrientation.POSITIVE
        )
        self.assertIs(
            decision.candidate_orientations[0],
            _JacobianCellOrientation.SIGN_CHANGING_OR_SINGULAR,
        )
        self.assertIsNone(decision.selected_index)

    def test_selection_is_independent_of_candidate_discovery_order(self):
        interpolator = _orientation_interpolator()
        forward = _local_orientation_decision(
            interpolator, [[1.5, 2.5], [2.5, 2.5]]
        )
        reverse = _local_orientation_decision(
            interpolator, [[2.5, 2.5], [1.5, 2.5]]
        )
        self.assertEqual(forward.selected_index, 0)
        self.assertEqual(reverse.selected_index, 1)

    def test_phase_solver_records_jacobian_selection(self):
        solver = PhaseAwareRootSolver.for_binodals(
            _orientation_interpolator(),
            _band(),
            [1.0, 1.0],
            SolverConfig(enable_subcell_fallback=False),
            subdivision=BinodalSubdivisionConfig(maximum_depth=6),
        )
        result = solver.solve([1.5, 2.5], initial_guess=[1.5, 2.5])
        np.testing.assert_allclose(result.source_coordinates, [1.5, 2.5])
        self.assertEqual(result.mixed_verified_root_count, 2)
        self.assertTrue(result.multivalued)
        self.assertIs(
            result.selection_reason,
            PhaseRootSelectionReason.SELECTED_FROM_JACOBIAN,
        )


class CECJacobianSelectionTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]
    metadata_path = repository_root / "EoSTables" / "CEC_TEST" / "metadata.minimal.json"
    table_path = repository_root / "EoSTables" / "CEC_TEST" / "hic_table_T80-300_muB0-800_d1.dat"

    @unittest.skipUnless(table_path.is_file(), "external CEC table is unavailable")
    def test_known_cec_fold_selects_positive_local_orientation(self):
        table = ForwardTable.load_validated(self.metadata_path)
        interpolator = MultilinearInterpolator(table, ("e", "nB"))
        root_a = np.asarray([0.10399946838576644, 0.782841878028613])
        root_b = np.asarray([0.1040031776079852, 0.7828364113355247])
        decision = _local_orientation_decision(interpolator, [root_a, root_b])
        self.assertEqual(decision.candidate_cells, ((23, 782), (24, 782)))
        self.assertIs(
            decision.expected_orientation, _JacobianCellOrientation.POSITIVE
        )
        self.assertEqual(decision.selected_index, 0)
        self.assertAlmostEqual(
            float(np.linalg.det(interpolator.jacobian(root_a))),
            5.8949146e-3,
            places=10,
        )
        self.assertAlmostEqual(
            float(np.linalg.det(interpolator.jacobian(root_b))),
            -9.8037856e-4,
            places=10,
        )
        expected_stencil = {
            (temperature, chemical_potential): (
                _JacobianCellOrientation.SIGN_CHANGING_OR_SINGULAR
                if (temperature, chemical_potential) == (24, 782)
                else _JacobianCellOrientation.POSITIVE
            )
            for temperature in (23, 24, 25)
            for chemical_potential in (781, 782, 783)
        }
        observed_stencil = {
            cell: _bilinear_cell_orientation(interpolator, cell)
            for cell in expected_stencil
        }
        self.assertEqual(observed_stencil, expected_stencil)

        bounds = interpolator.node_value_bounds
        scales = _automatic_residual_scales_from_bounds(bounds.lower, bounds.upper)
        solver = PhaseAwareRootSolver.for_binodals(
            interpolator,
            transition_model_from_table(table),
            scales,
            SolverConfig(enable_subcell_fallback=False),
        )
        result = solver.solve(interpolator.evaluate(root_a), initial_guess=root_a)
        np.testing.assert_allclose(result.source_coordinates, root_a, atol=1.0e-10)
        self.assertEqual(result.mixed_verified_root_count, 2)
        self.assertIs(
            result.selection_reason,
            PhaseRootSelectionReason.SELECTED_FROM_JACOBIAN,
        )


class JacobianSelectionPersistenceTests(unittest.TestCase):
    def setUp(self):
        fields = {
            "T": field_spec("coordinate", 0),
            "muB": field_spec("coordinate", 1),
            "e": field_spec("target", 2),
            "nB": field_spec("target", 3),
            "phase": field_spec("auxiliary", 4),
        }
        document = metadata(fields, 2, expected_shape=[5, 5])
        document["scientific"]["phase_structure"] = {
            "status": "known",
            "critical_points": [],
            "first_order_transition": {
                "present": True,
                "representation": "binodals",
                "kind": "synthetic non-physical band",
                "phase_fraction_field": "phase",
                "mixed_phase_condition": "0 < phase < 1",
                "coexistence_data": {
                    "file": "transition.dat",
                    "delimiter": "whitespace",
                    "comment_prefix": "#",
                    "skip_rows": 0,
                    "encoding": "utf-8",
                    "expected_columns": 3,
                    "fields": ["muB", "T_dilute", "T_dense"],
                    "columns": {
                        "muB": {"index": 0, "unit": "GeV"},
                        "T_dilute": {"index": 1, "unit": "GeV"},
                        "T_dense": {"index": 2, "unit": "GeV"},
                    },
                },
            },
        }
        interpolator = _orientation_interpolator()
        rows = []
        for i, temperature in enumerate(interpolator.axes[0]):
            for j, chemical_potential in enumerate(interpolator.axes[1]):
                values = interpolator.evaluate([temperature, chemical_potential])
                fraction = 0.5 if 1.2 < temperature < 2.8 else 0.0
                rows.append(
                    [temperature, chemical_potential, values[0], values[1], fraction]
                )
        self.fixture = TemporaryEoS(document, rows)
        self.addCleanup(self.fixture.close)
        self.fixture.write_sidecar("transition.dat", "0 1.2 2.8\n4 1.2 2.8\n")
        self.table = ForwardTable.load_validated(self.fixture.metadata_path)
        self.config = PipelineConfig(
            grid=InversionGridSpec(
                GridSpace.TARGET,
                {
                    "e": GridAxisSpec(2, 1.5 - 1.0e-6, 1.5 + 1.0e-6),
                    "nB": GridAxisSpec(2, 2.5 - 1.0e-6, 2.5 + 1.0e-6),
                },
            ),
            residual_scales={"e": 1.0, "nB": 1.0},
            solver=SolverConfig(enable_subcell_fallback=False),
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(workers=1, chunk_size=2, worker_task_size=1),
            auxiliary_fields=("phase",),
        )

    def test_artifact_and_checkpoint_preserve_selection_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = CheckpointConfig(Path(directory) / "checkpoint")
            result = EoSInversionPipeline(self.table, self.config).run(
                checkpoint=checkpoint
            )
            self.assertTrue(np.all(result.jacobian_selected_mask))
            self.assertEqual(
                result.manifest["run"]["status_summary"]["phase_diagnostics"][
                    "jacobian_selected_roots"
                ],
                4,
            )
            artifact = save_pipeline_result(
                result, Path(directory) / "selected.eosinv"
            )
            loaded = load_pipeline_result(artifact)
            np.testing.assert_array_equal(
                loaded.phase_selection_reason_codes,
                result.phase_selection_reason_codes,
            )
            resumed = EoSInversionPipeline(self.table, self.config).run(
                checkpoint=checkpoint
            )
            np.testing.assert_array_equal(
                resumed.phase_selection_reason_codes,
                result.phase_selection_reason_codes,
            )
            exported = export_text(
                result,
                Path(directory) / "selected.dat",
                config=TextExportConfig(phase=None),
            )
            text = exported.table_path.read_text(encoding="utf-8")
            self.assertNotIn("target_index=", text)
            values = np.loadtxt(exported.table_path)
            selected_column = exported.column_names.index(
                "selected_from_jacobian"
            )
            np.testing.assert_array_equal(values[:, selected_column], 1.0)
            manifest = json.loads(
                exported.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["root_selection_provenance"],
                {
                    "selection": "SELECTED_FROM_JACOBIAN",
                    "count": 4,
                    "flat_target_indices": [0, 1, 2, 3],
                },
            )

    def test_serial_and_process_results_have_identical_provenance(self):
        methods = mp.get_all_start_methods()
        start = "fork" if "fork" in methods else "spawn"
        process_config = replace(
            self.config,
            execution=replace(
                self.config.execution,
                workers=2,
                start_method=ProcessStartMethod(start),
            ),
        )
        serial = EoSInversionPipeline(self.table, self.config).run()
        process = EoSInversionPipeline(self.table, process_config).run()
        np.testing.assert_array_equal(
            process.phase_selection_reason_codes,
            serial.phase_selection_reason_codes,
        )
        np.testing.assert_allclose(
            process.source_coordinates, serial.source_coordinates, atol=1.0e-12
        )

    def test_slurm_shards_and_assembly_preserve_selection_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = create_array_plan(
                self.table,
                self.config,
                output_directory=root / "slurm",
                final_artifact=root / "assembled.eosinv",
                slurm=SlurmArrayConfig(
                    cpus_per_task=1,
                    working_directory=root,
                    log_directory=root / "logs",
                    entrypoint=root / "EoS-TrENsMUTher.py",
                ),
            )
            for shard in plan.shards:
                execute_array_shard(plan, shard.index)
            assembled = assemble_array_shards(plan)
            self.assertTrue(np.all(assembled.jacobian_selected_mask))
            self.assertEqual(
                assembled.manifest["run"]["status_summary"][
                    "phase_diagnostics"
                ]["jacobian_selected_roots"],
                4,
            )


if __name__ == "__main__":
    unittest.main()
