"""Focused tests for versioned completed-result persistence."""

import json
import os
import tempfile
import unittest
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import eos_trensmuther.outputs as outputs_module

from eos_trensmuther import (
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    PersistenceFormatError,
    PersistenceIntegrityError,
    PersistencePathError,
    PersistenceVersionError,
    PipelineConfig,
    ProjectionConfig,
    SolverConfig,
    load_pipeline_result,
    save_pipeline_result,
)

from tests.test_tables import TemporaryEoS, field_spec, metadata


_RESULT_ARRAY_NAMES = (
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
    "initial_guess_source_codes",
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


def _plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class OutputPersistenceTests(unittest.TestCase):
    def make_result(self, *, space="target", auxiliary=True, solver=None):
        fields = {
            "T": field_spec("coordinate", 0),
            "e": field_spec("target", 1),
            "P": field_spec("auxiliary", 2),
        }
        rows = [
            (temperature, 1.0 + 2.0 * temperature, 3.0 * temperature)
            for temperature in (0.2, 0.5, 0.8)
        ]
        fixture = TemporaryEoS(metadata(fields, 1), rows)
        self.addCleanup(fixture.close)
        table = ForwardTable.load_validated(fixture.metadata_path)
        envelope = table.find_boundaries(space=space)
        config = PipelineConfig(
            grid=InversionGridSpec(
                space,
                {name: GridAxisSpec(4) for name in envelope.names},
            ),
            residual_scales={"e": 1.0},
            solver=SolverConfig() if solver is None else solver,
            projection=ProjectionConfig(
                enabled=True,
                nearest_cell_starts=2,
                max_evaluations_per_start=40,
            ),
            execution=ExecutionConfig(workers=1, chunk_size=2),
            auxiliary_fields=("P",) if auxiliary else (),
        )
        result = EoSInversionPipeline(table, config).run()
        return result, fixture

    def assert_results_equal(self, first, second):
        self.assertEqual(first.plan.config.to_dict(), second.plan.config.to_dict())
        self.assertEqual(first.plan.manifest(), second.plan.manifest())
        self.assertEqual(_plain(first.manifest), _plain(second.manifest))
        self.assertEqual(first.grid.shape, second.grid.shape)
        self.assertEqual(first.grid.constraints.identifier, second.grid.constraints.identifier)
        for first_axis, second_axis in zip(first.grid.axes, second.grid.axes):
            np.testing.assert_array_equal(first_axis, second_axis)
            self.assertFalse(second_axis.flags.writeable)
        for name in _RESULT_ARRAY_NAMES:
            with self.subTest(result_array=name):
                np.testing.assert_array_equal(
                    getattr(first, name), getattr(second, name)
                )
                self.assertEqual(
                    getattr(first, name).dtype, getattr(second, name).dtype
                )
                self.assertFalse(getattr(second, name).flags.writeable)
        self.assertEqual(dict(first.status_counts), dict(second.status_counts))

    def test_target_and_tilde_results_round_trip_without_source_files(self):
        for space, compression in (("target", "none"), ("tilde", "deflate")):
            with self.subTest(space=space, compression=compression):
                result, fixture = self.make_result(space=space)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / ("%s.eosinv" % space)
                    saved = save_pipeline_result(
                        result, path, compression=compression
                    )
                    fixture.metadata_path.unlink()
                    fixture.table_path.unlink()
                    loaded = load_pipeline_result(saved)
                    self.assert_results_equal(result, loaded)
                    if space == "target":
                        self.assertEqual(loaded.achieved_tilde_values.shape[-1], 0)
                    else:
                        self.assertEqual(loaded.achieved_tilde_values.shape[-1], 4)

    def test_zero_width_auxiliary_column_round_trips(self):
        result, _ = self.make_result(auxiliary=False)
        with tempfile.TemporaryDirectory() as directory:
            path = save_pipeline_result(
                result, Path(directory) / "without-auxiliary.eosinv"
            )
            loaded = load_pipeline_result(path)
        self.assertEqual(loaded.auxiliary_values.shape, result.grid.shape + (0,))
        self.assert_results_equal(result, loaded)

    def test_no_clobber_default_and_explicit_overwrite(self):
        result, _ = self.make_result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.eosinv"
            saved = save_pipeline_result(result, path)
            original = saved.read_bytes()
            with self.assertRaises(FileExistsError):
                save_pipeline_result(result, path)
            self.assertEqual(saved.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            save_pipeline_result(result, path, overwrite=True, compression="deflate")
            loaded = load_pipeline_result(path)
            self.assert_results_equal(result, loaded)

    def test_save_rejects_symlink_destination(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symbolic links are unavailable")
        result, _ = self.make_result()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            real = parent / "real.eosinv"
            real.write_bytes(b"do not replace")
            link = parent / "linked.eosinv"
            try:
                link.symlink_to(real)
            except OSError as exc:
                self.skipTest("could not create a symbolic link: %s" % exc)
            with self.assertRaises(PersistencePathError):
                save_pipeline_result(result, link, overwrite=True)
            self.assertEqual(real.read_bytes(), b"do not replace")

    @staticmethod
    def rewrite_archive(path, transform):
        with zipfile.ZipFile(path, "r") as archive:
            members = [(info.filename, archive.read(info)) for info in archive.infolist()]
        transformed = transform(members)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, payload in transformed:
                archive.writestr(name, payload)

    def test_array_byte_corruption_is_detected(self):
        result, _ = self.make_result()
        with tempfile.TemporaryDirectory() as directory:
            path = save_pipeline_result(result, Path(directory) / "corrupt.eosinv")

            def corrupt(members):
                output = []
                changed = False
                for name, payload in members:
                    if not changed and name.startswith("arrays/") and payload:
                        payload = payload[:-1] + bytes([payload[-1] ^ 1])
                        changed = True
                    output.append((name, payload))
                return output

            self.rewrite_archive(path, corrupt)
            with self.assertRaises(PersistenceIntegrityError):
                load_pipeline_result(path)

    def test_descriptor_tampering_and_newer_version_are_rejected(self):
        result, _ = self.make_result()
        with tempfile.TemporaryDirectory() as directory:
            path = save_pipeline_result(result, Path(directory) / "tamper.eosinv")

            def tamper(members):
                output = []
                for name, payload in members:
                    if name == "manifest.json":
                        document = json.loads(payload.decode("utf-8"))
                        document["plan"]["table"]["name"] = "tampered"
                        payload = json.dumps(document).encode("utf-8")
                    output.append((name, payload))
                return output

            self.rewrite_archive(path, tamper)
            with self.assertRaises(PersistenceIntegrityError):
                load_pipeline_result(path)

            path = save_pipeline_result(
                result, Path(directory) / "future.eosinv"
            )

            def future(members):
                output = []
                for name, payload in members:
                    if name == "manifest.json":
                        document = json.loads(payload.decode("utf-8"))
                        document["artifact_schema_version"] += 1
                        payload = json.dumps(document).encode("utf-8")
                    output.append((name, payload))
                return output

            self.rewrite_archive(path, future)
            with self.assertRaises(PersistenceVersionError):
                load_pipeline_result(path)

    def test_extra_unsafe_and_duplicate_members_are_rejected(self):
        result, _ = self.make_result()
        with tempfile.TemporaryDirectory() as directory:
            for label, extra_name in (("extra", "extra.txt"), ("unsafe", "../x")):
                with self.subTest(label=label):
                    path = save_pipeline_result(
                        result, Path(directory) / (label + ".eosinv")
                    )

                    def add_member(members, name=extra_name):
                        return members + [(name, b"x")]

                    self.rewrite_archive(path, add_member)
                    with self.assertRaises(PersistenceFormatError):
                        load_pipeline_result(path)

            duplicate = save_pipeline_result(
                result, Path(directory) / "duplicate.eosinv"
            )

            def duplicate_member(members):
                return members + [members[-1]]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.rewrite_archive(duplicate, duplicate_member)
            with self.assertRaises(PersistenceFormatError):
                load_pipeline_result(duplicate)

    def test_invalid_json_and_resource_limits_are_rejected(self):
        result, _ = self.make_result()
        with tempfile.TemporaryDirectory() as directory:
            path = save_pipeline_result(result, Path(directory) / "limits.eosinv")
            for keyword in (
                "max_archive_bytes",
                "max_uncompressed_bytes",
                "max_manifest_bytes",
                "max_result_bytes",
            ):
                with self.subTest(limit=keyword):
                    with self.assertRaises(PersistenceFormatError):
                        load_pipeline_result(path, **{keyword: 1})

            invalid = save_pipeline_result(
                result, Path(directory) / "invalid-json.eosinv"
            )

            def invalid_json(members):
                return [
                    (name, b'{"x":1,"x":2}' if name == "manifest.json" else payload)
                    for name, payload in members
                ]

            self.rewrite_archive(invalid, invalid_json)
            with self.assertRaises(PersistenceFormatError):
                load_pipeline_result(invalid)

    def test_writer_rejects_nonterminal_manifest(self):
        result, _ = self.make_result()
        manifest = _plain(result.manifest)
        manifest["run"]["completed"] = False
        nonterminal = replace(result, manifest=manifest)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PersistenceFormatError):
                save_pipeline_result(
                    nonterminal, Path(directory) / "nonterminal.eosinv"
                )

    def test_completed_version_8_artifact_remains_readable_but_not_rewritable(self):
        result, _ = self.make_result()
        with tempfile.TemporaryDirectory() as directory:
            path = save_pipeline_result(
                result, Path(directory) / "legacy-version-8.eosinv"
            )

            def downgrade(members):
                output = []
                for name, payload in members:
                    if name == "manifest.json":
                        document = json.loads(payload.decode("utf-8"))
                        document["pipeline_run_manifest_version"] = 8
                        document["plan"]["configuration"].pop(
                            "numerical_routing_contract_version", None
                        )
                        document["plan"]["configuration"].pop(
                            "use_reachability_seed", None
                        )
                        run_manifest = document["run_manifest"]
                        run_manifest.pop("numerical_routing_contract")
                        run_manifest["configuration"].pop(
                            "numerical_routing_contract_version", None
                        )
                        run_manifest["configuration"].pop(
                            "use_reachability_seed", None
                        )
                        run_manifest["run"][
                            "pipeline_run_manifest_version"
                        ] = 8
                        without_integrity = dict(document)
                        without_integrity.pop("integrity")
                        document["integrity"]["digest"] = (
                            outputs_module._artifact_digest(without_integrity)
                        )
                        payload = outputs_module._canonical_json(document)
                    output.append((name, payload))
                return output

            self.rewrite_archive(path, downgrade)
            loaded = load_pipeline_result(path)
            self.assertEqual(
                loaded.manifest["run"]["pipeline_run_manifest_version"], 8
            )
            self.assertNotIn(
                "numerical_routing_contract", loaded.manifest
            )
            np.testing.assert_array_equal(
                loaded.source_coordinates, result.source_coordinates
            )
            with self.assertRaises(PersistenceVersionError):
                save_pipeline_result(
                    loaded, Path(directory) / "legacy-copy.eosinv"
                )

    def test_version_9_artifact_without_default_subcell_fields_remains_readable(self):
        result, _ = self.make_result(
            solver=SolverConfig(enable_subcell_fallback=False)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_pipeline_result(
                result, Path(directory) / "pre-subcell-version-9.eosinv"
            )

            def remove_default_subcell_fields(members):
                output = []
                names = (
                    "enable_subcell_fallback",
                    "max_subcell_depth",
                    "max_subcell_active_boxes",
                )
                for name, payload in members:
                    if name == "manifest.json":
                        document = json.loads(payload.decode("utf-8"))
                        document["pipeline_run_manifest_version"] = 9
                        for owner in (
                            document["plan"],
                            document["run_manifest"],
                        ):
                            owner["configuration"][
                                "numerical_routing_contract_version"
                            ] = 1
                        document["run_manifest"][
                            "numerical_routing_contract"
                        ]["version"] = 1
                        document["run_manifest"]["run"][
                            "pipeline_run_manifest_version"
                        ] = 9
                        for owner in (document["plan"], document["run_manifest"]):
                            solver = owner["configuration"]["solver"]
                            for field_name in names:
                                solver.pop(field_name)
                        without_integrity = dict(document)
                        without_integrity.pop("integrity")
                        document["integrity"]["digest"] = (
                            outputs_module._artifact_digest(without_integrity)
                        )
                        payload = outputs_module._canonical_json(document)
                    output.append((name, payload))
                return output

            self.rewrite_archive(path, remove_default_subcell_fields)
            loaded = load_pipeline_result(path)
            self.assertFalse(loaded.plan.config.solver.enable_subcell_fallback)
            self.assertEqual(loaded.plan.config.solver.max_subcell_depth, 7)
            self.assertEqual(
                loaded.plan.config.solver.max_subcell_active_boxes,
                1_000_000,
            )
            np.testing.assert_array_equal(
                loaded.source_coordinates, result.source_coordinates
            )
            with self.assertRaises(PersistenceVersionError):
                save_pipeline_result(
                    loaded, Path(directory) / "legacy-version-9-copy.eosinv"
                )

    def test_version_9_artifact_without_worker_task_size_remains_readable(self):
        result, _ = self.make_result()
        with tempfile.TemporaryDirectory() as directory:
            path = save_pipeline_result(
                result, Path(directory) / "pre-dynamic-scheduler-version-9.eosinv"
            )

            def remove_worker_task_size(members):
                output = []
                for name, payload in members:
                    if name == "manifest.json":
                        document = json.loads(payload.decode("utf-8"))
                        document["pipeline_run_manifest_version"] = 9
                        for owner in (
                            document["plan"],
                            document["run_manifest"],
                        ):
                            owner["configuration"][
                                "numerical_routing_contract_version"
                            ] = 1
                        document["run_manifest"][
                            "numerical_routing_contract"
                        ]["version"] = 1
                        document["run_manifest"]["run"][
                            "pipeline_run_manifest_version"
                        ] = 9
                        for owner in (document["plan"], document["run_manifest"]):
                            owner["configuration"]["execution"].pop(
                                "worker_task_size"
                            )
                        without_integrity = dict(document)
                        without_integrity.pop("integrity")
                        document["integrity"]["digest"] = (
                            outputs_module._artifact_digest(without_integrity)
                        )
                        payload = outputs_module._canonical_json(document)
                    output.append((name, payload))
                return output

            self.rewrite_archive(path, remove_worker_task_size)
            loaded = load_pipeline_result(path)
            self.assertEqual(loaded.plan.config.execution.worker_task_size, 256)
            np.testing.assert_array_equal(
                loaded.source_coordinates, result.source_coordinates
            )

    def test_current_artifact_missing_current_execution_field_is_rejected(self):
        result, _ = self.make_result()
        with tempfile.TemporaryDirectory() as directory:
            path = save_pipeline_result(
                result, Path(directory) / "invalid-current-contract.eosinv"
            )

            def remove_worker_task_size(members):
                output = []
                for name, payload in members:
                    if name == "manifest.json":
                        document = json.loads(payload.decode("utf-8"))
                        for owner in (
                            document["plan"],
                            document["run_manifest"],
                        ):
                            owner["configuration"]["execution"].pop(
                                "worker_task_size"
                            )
                        without_integrity = dict(document)
                        without_integrity.pop("integrity")
                        document["integrity"]["digest"] = (
                            outputs_module._artifact_digest(without_integrity)
                        )
                        payload = outputs_module._canonical_json(document)
                    output.append((name, payload))
                return output

            self.rewrite_archive(path, remove_worker_task_size)
            with self.assertRaises(PersistenceFormatError):
                load_pipeline_result(path)


if __name__ == "__main__":
    unittest.main()
