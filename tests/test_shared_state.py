"""Read-only mmap worker-state and bounded numerical-equivalence tests."""

import itertools
import multiprocessing
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eos_trensmuther.interpolation import MultilinearInterpolator
from eos_trensmuther.pipeline import (
    EoSInversionPipeline,
    ExecutionConfig,
    GridAxisSpec,
    InversionGridSpec,
    PipelineConfig,
    ProjectionConfig,
    _initialize_pipeline_worker,
    _pipeline_worker_state_diagnostics,
)
from eos_trensmuther.reachability import ReachabilityConfig
from eos_trensmuther.shared_state import (
    create_shared_worker_state,
    load_shared_reachability,
    load_shared_table,
    shared_state_backing_report,
)
from eos_trensmuther.tables import ForwardTable
from tests.test_tables import TemporaryEoS, field_spec, metadata


# Keep the production-private list local to the test rather than making it a
# package API merely for assertions.
_RESULT_FIELDS = (
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


class SharedWorkerStateTests(unittest.TestCase):
    def _case(self):
        coordinate_names = ("T", "muB", "muQ", "muS")
        target_names = ("e", "nB", "nQ", "nS")
        fields = {
            name: field_spec("coordinate", index)
            for index, name in enumerate(coordinate_names)
        }
        fields.update(
            {
                name: field_spec("target", index + 4)
                for index, name in enumerate(target_names)
            }
        )
        axes = ((0.2, 0.5), (0.0, 0.3), (-0.1, 0.1), (-0.2, 0.2))
        rows = [list(point) + list(point) for point in itertools.product(*axes)]
        fixture = TemporaryEoS(metadata(fields, 4), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        envelope = table.find_boundaries(space="target")
        config = PipelineConfig(
            grid=InversionGridSpec(
                "target",
                {name: GridAxisSpec(2) for name in envelope.names},
            ),
            residual_scales={name: 1.0 for name in table.target_names},
            projection=ProjectionConfig(enabled=False),
            execution=ExecutionConfig(workers=1, chunk_size=16),
            auxiliary_fields=(),
        )
        reachability = ReachabilityConfig(
            method="hierarchical",
            hierarchy_levels=7,
            query_batch_size=2,
            cell_batch_size=1,
        )
        return table, config, reachability

    def test_loaded_arrays_remain_read_only_mmap_backed(self):
        table, config, reachability = self._case()
        pipeline = EoSInversionPipeline(
            table, config, reachability_config=reachability
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = create_shared_worker_state(
                table,
                pipeline.plan(),
                reachability,
                Path(directory) / "state",
            )
            mapped_table = load_shared_table(manifest)
            interpolator = MultilinearInterpolator(
                mapped_table, pipeline.plan().target_names
            )
            mapped_index = load_shared_reachability(
                manifest,
                interpolator,
                pipeline.plan(),
                reachability,
            )
            report = shared_state_backing_report(mapped_table, mapped_index)

            self.assertTrue(report["all_large_arrays_mmap_backed"])
            self.assertTrue(report["arrays"])
            self.assertTrue(all(row["mmap_backed"] for row in report["arrays"]))
            self.assertTrue(
                all(not values.flags.writeable for values in mapped_table.fields.values())
            )
            self.assertTrue(
                all(
                    not getattr(mapped_index.enclosures, name).flags.writeable
                    for name in (
                        "cell_flat_indices",
                        "exact_lower",
                        "exact_upper",
                        "routing_lower",
                        "routing_upper",
                    )
                )
            )

    def test_repeated_mmap_localization_is_stateless_and_duplicate_free(self):
        table, config, reachability = self._case()
        pipeline = EoSInversionPipeline(
            table, config, reachability_config=reachability
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = create_shared_worker_state(
                table,
                pipeline.plan(),
                reachability,
                Path(directory) / "state",
            )
            mapped_table = load_shared_table(manifest)
            interpolator = MultilinearInterpolator(
                mapped_table, pipeline.plan().target_names
            )
            mapped_index = load_shared_reachability(
                manifest,
                interpolator,
                pipeline.plan(),
                reachability,
            )
            targets = np.asarray(
                [
                    [0.35, 0.15, 0.0, 0.0],
                    [0.35, 0.15, 0.0, 0.0],
                    [0.2, 0.0, -0.1, -0.2],
                ]
            )

            first = mapped_index.localize(targets)
            second = mapped_index.localize(targets)

            np.testing.assert_array_equal(first.offsets, second.offsets)
            np.testing.assert_array_equal(
                first.cell_flat_indices, second.cell_flat_indices
            )
            np.testing.assert_array_equal(first.candidate_counts, [1, 1, 1])
            for target_index in range(targets.shape[0]):
                row = first.cells_for(target_index)
                self.assertEqual(row.size, np.unique(row).size)

    def test_private_and_mmap_one_worker_chunks_are_numerically_equivalent(self):
        table, config, reachability = self._case()
        private = EoSInversionPipeline(
            table, config, reachability_config=reachability
        )
        private._prepare()
        chunk = private.plan().grid.chunk(0, private.plan().grid.size)
        private_result = private._execute_chunk(chunk)

        with tempfile.TemporaryDirectory() as directory:
            manifest = create_shared_worker_state(
                table,
                private.plan(),
                reachability,
                Path(directory) / "state",
            )
            mapped_table = load_shared_table(manifest)
            mapped = EoSInversionPipeline(
                mapped_table, config, reachability_config=reachability
            )
            mapped._prepare(shared_state_manifest=manifest)
            mapped_result = mapped._execute_chunk(
                mapped.plan().grid.chunk(0, mapped.plan().grid.size)
            )

            for name in _RESULT_FIELDS:
                with self.subTest(field=name):
                    np.testing.assert_allclose(
                        getattr(private_result, name),
                        getattr(mapped_result, name),
                        rtol=1.0e-12,
                        atol=1.0e-13,
                        equal_nan=True,
                    )
            self.assertEqual(private_result.counters, mapped_result.counters)

    def test_spawned_worker_reports_mmap_backing(self):
        table, config, reachability = self._case()
        pipeline = EoSInversionPipeline(
            table, config, reachability_config=reachability
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = create_shared_worker_state(
                table,
                pipeline.plan(),
                reachability,
                Path(directory) / "state",
            )
            payload = pipeline._worker_initialization(
                shared_state_manifest=manifest
            )
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_pipeline_worker,
                initargs=(payload,),
            ) as executor:
                diagnostics = executor.submit(
                    _pipeline_worker_state_diagnostics
                ).result()
            report = diagnostics["shared_state"]
            self.assertIsNotNone(report)
            self.assertTrue(report["all_large_arrays_mmap_backed"])
            self.assertTrue(all(row["mmap_backed"] for row in report["arrays"]))


if __name__ == "__main__":
    unittest.main()
