"""Local command-line interface for validation, inversion, and export."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from ._version import __version__
from .config import ApplicationConfig, ApplicationConfigError, load_application_config
from .exporters import (
    ExportError,
    TextExportResult,
    export_text,
)
from .outputs import PersistenceError, load_pipeline_result, save_pipeline_result
from .pipeline import (
    EoSInversionPipeline,
    PipelineError,
    PipelinePlan,
    PipelineResult,
)
from .tables import EoSError, EoSValidationError, ForwardTable
from .executors import (
    ExecutorError,
    SlurmArrayConfig,
    assemble_array_shards,
    create_array_plan,
    execute_array_shard,
    load_array_plan,
    render_slurm_script,
    save_array_plan,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("value is not JSON serializable: %r" % (value,))


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(str(path), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
            raise


def _atomic_json(value: Dict[str, Any], path: Path, *, overwrite: bool) -> Path:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise OSError("output is not a regular file: %s" % path)
        if not overwrite:
            raise FileExistsError("output already exists: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(str(temporary), str(path))
        else:
            try:
                os.link(str(temporary), str(path))
            except FileExistsError as exc:
                raise FileExistsError("output already exists: %s" % path) from exc
            temporary.unlink()
        _fsync_directory(path.parent)
        return path
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare_output_directories(config: ApplicationConfig) -> None:
    config.diagnostic_directory.mkdir(parents=True, exist_ok=True)
    for label, directory in (
        ("OutputFolder", config.output_root),
        ("EoS output folder", config.eos_directory),
        ("run output folder", config.output_directory),
        ("diagnostic folder", config.diagnostic_directory),
    ):
        if directory.is_symlink() or not directory.is_dir():
            raise OSError("%s must be a non-symbolic directory: %s" % (label, directory))
    for path in (
        config.preprocess_report_path,
        config.result_path,
        config.text_export_path,
        config.text_export_manifest_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)


def _preprocess_document(
    config: ApplicationConfig, table: ForwardTable
) -> Dict[str, Any]:
    table.require_validated()
    assert table.metadata is not None
    assert table.report is not None
    resolved_plan = PipelinePlan.from_table(table, config.pipeline)
    residual_scale_mode = (
        "automatic"
        if config.pipeline.residual_scales is None
        else "explicit"
    )
    return {
        "report_type": "EoS-TrENsMUTher.preprocess",
        "schema_version": 1,
        "validated": True,
        "run": {
            "eos_name": config.eos_name,
            "run_name": config.run_name,
            "specification": config.specification,
            "overwrite_mode": config.overwrite_mode,
            "output_directory": str(config.output_directory),
            "diagnostic_directory": str(config.diagnostic_directory),
            "text_export_enabled": config.export_enabled,
        },
        "parameter_file": str(config.parameter_path),
        "metadata_file": str(config.metadata_path),
        "data_file": str(table.data_path),
        "table": {
            "name": table.metadata.name,
            "revision": table.metadata.table_revision,
            "dimension": table.metadata.dimension,
            "shape": list(table.shape),
            "coordinate_names": list(table.coordinate_names),
            "target_names": list(table.target_names),
            "auxiliary_names": list(table.auxiliary_names),
            "uniform_by_axis": list(table.grid.uniform_by_axis),
            "source_rows_reordered": table.grid.was_reordered,
            "snapshot_fingerprint": table.snapshot_fingerprint(),
        },
        "validation": table.report.to_dict(),
        "boundaries": {
            "target": table.find_boundaries(space="target").to_dict(),
            "tilde": table.find_boundaries(space="tilde").to_dict(),
        },
        "resolved_pipeline_configuration": resolved_plan.config.to_dict(),
        "residual_scale_resolution": {
            "mode": residual_scale_mode,
            "rule": (
                "maximum_absolute_validated_table_node_value"
                if residual_scale_mode == "automatic"
                else "explicit_user_values"
            ),
            "zero_component_fallback": (
                1.0 if residual_scale_mode == "automatic" else None
            ),
            "values": {
                name: float(value)
                for name, value in zip(
                    resolved_plan.target_names,
                    resolved_plan.residual_scales,
                )
            },
        },
        "ignored_legacy_parameters": list(config.ignored_legacy_parameters),
        "notes": {
            "preprocessing": (
                "The source file is validated, reordered, and converted to canonical "
                "units in memory; it is not rewritten."
            ),
            "legacy_accuracy": (
                "Accuracy/MAXITER are not used by the new scaled solver; configure "
                "Solver directly only for an advanced numerical study."
            ),
        },
    }


def preprocess(config: ApplicationConfig, table: ForwardTable) -> Path:
    """Validate/interpret a table and write the resolved preprocessing report."""

    _prepare_output_directories(config)
    document = _preprocess_document(config, table)
    return _atomic_json(
        document, config.preprocess_report_path, overwrite=config.overwrite
    )


def invert(
    config: ApplicationConfig,
    table: ForwardTable,
    *,
    progress_callback: Optional[Any] = None,
) -> PipelineResult:
    """Run the new table-wide pipeline and publish its authoritative artifact."""

    _prepare_output_directories(config)
    if config.result_path.exists() and not config.overwrite:
        raise FileExistsError("output already exists: %s" % config.result_path)
    if config.overwrite and config.checkpoint is not None:
        checkpoint_directory = config.checkpoint.directory
        if checkpoint_directory.exists():
            if checkpoint_directory.is_symlink() or not checkpoint_directory.is_dir():
                raise OSError(
                    "checkpoint overwrite target must be a directory: %s"
                    % checkpoint_directory
                )
            diagnostic = config.diagnostic_directory.resolve(strict=True)
            checkpoint = checkpoint_directory.resolve(strict=True)
            if os.path.commonpath((str(diagnostic), str(checkpoint))) != str(diagnostic):
                raise OSError("checkpoint overwrite target escaped diagnostic folder")
            shutil.rmtree(checkpoint)
    pipeline = EoSInversionPipeline(
        table,
        config.pipeline,
        reachability_config=config.reachability,
    )
    result = pipeline.run(
        checkpoint=config.checkpoint,
        progress_callback=progress_callback,
    )
    save_pipeline_result(
        result,
        config.result_path,
        overwrite=config.overwrite,
        compression=config.artifact_compression,
    )
    return result


def postprocess_text(
    config: ApplicationConfig, result: Optional[PipelineResult] = None
) -> TextExportResult:
    """Load or consume a completed result and write the generic text table."""

    _prepare_output_directories(config)
    if not config.export_enabled:
        raise ApplicationConfigError(
            "Export.enabled is false; enable it before requesting text export"
        )
    if result is None:
        result = load_pipeline_result(config.result_path)
    return export_text(
        result,
        config.text_export_path,
        config=config.text_export,
        overwrite=config.overwrite,
        manifest_path=config.text_export_manifest_path,
    )


ExportResult = TextExportResult


def postprocess(
    config: ApplicationConfig, result: Optional[PipelineResult] = None
) -> ExportResult:
    """Write the configured generic text export."""

    return postprocess_text(config, result)


def run_all(
    config: ApplicationConfig, table: ForwardTable
) -> Tuple[Path, PipelineResult, Optional[ExportResult]]:
    """Preprocess, invert, and export using one resolved configuration."""

    candidates = [
        config.preprocess_report_path,
        config.result_path,
    ]
    if config.export_enabled:
        candidates.append(config.text_export_path)
        if config.text_export.write_manifest:
            candidates.append(config.text_export_manifest_path)
    if not config.overwrite:
        existing = [path for path in candidates if path.exists()]
        if existing:
            raise FileExistsError(
                "run output already exists: %s" % ", ".join(map(str, existing))
            )
    report = preprocess(config, table)
    result = invert(config, table)
    exported = postprocess(config, result) if config.export_enabled else None
    return report, result, exported


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "parameters",
        type=Path,
        help="trusted Python file defining general_parameters",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="override OutputFolder (relative overrides use the current directory)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="override Overwrite and replace outputs in the selected run folder",
    )
    parser.add_argument(
        "--spec",
        help="run suffix beginning with '_', for example '_test2mub500'",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eos-trensmuther",
        description=(
            "Validate a metadata-described EoS, invert a regular target/tilde "
            "grid, and write a generic self-describing text table."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, help_text, aliases in (
        ("preprocess", "validate and summarize the input table", ()),
        ("invert", "run inversion and save inverse.eosinv", ()),
        ("export-text", "convert inverse.eosinv to a generic text table", ()),
        ("postprocess", "compatibility alias for export-text", ()),
        ("run", "preprocess, invert, and export", ()),
    ):
        command = subcommands.add_parser(name, help=help_text, aliases=aliases)
        _add_common_arguments(command)
    slurm_plan = subcommands.add_parser(
        "slurm-plan",
        help="create a first-axis array plan and sbatch script without submitting",
    )
    _add_common_arguments(slurm_plan)
    slurm_plan.add_argument("--account")
    slurm_plan.add_argument("--partition")
    slurm_plan.add_argument("--wall-time", default="01:00:00")
    slurm_plan.add_argument("--cpus-per-task", type=int, default=1)
    slurm_plan.add_argument("--memory")
    slurm_plan.add_argument("--job-name", default="eos-trensmuther")
    slurm_plan.add_argument("--working-directory", type=Path)
    slurm_plan.add_argument("--log-directory", type=Path)
    slurm_plan.add_argument(
        "--activate",
        action="append",
        default=[],
        help="one trusted shell activation command; repeat for multiple commands",
    )
    slurm_task = subcommands.add_parser(
        "slurm-task", help="execute/resume one first-axis array shard"
    )
    slurm_task.add_argument("--plan", type=Path, required=True)
    slurm_task.add_argument("--shard-index", type=int)
    slurm_assemble = subcommands.add_parser(
        "slurm-assemble", help="validate all shards and assemble inverse.eosinv"
    )
    slurm_assemble.add_argument("--plan", type=Path, required=True)
    slurm_assemble.add_argument("--output", type=Path)
    slurm_assemble.add_argument("--overwrite", action="store_true")
    return parser


def _normalize_arguments(argv: Sequence[str]) -> Sequence[str]:
    commands = {
        "preprocess",
        "invert",
        "export-text",
        "postprocess",
        "run",
        "slurm-plan",
        "slurm-task",
        "slurm-assemble",
    }
    if argv and argv[0] not in commands and argv[0] not in {
        "-h",
        "--help",
        "--version",
    }:
        return ("run",) + tuple(argv)
    return argv


def _slurm_launcher() -> Tuple[Path, Path]:
    """Return a stable working directory and executable Python entry point."""

    source_root = Path(__file__).resolve().parents[2]
    source_wrapper = source_root / "EoS-TrENsMUTher.py"
    invoked = Path(sys.argv[0]).expanduser()
    if invoked.name == "eos-trensmuther" and invoked.exists():
        return Path.cwd().resolve(), invoked.resolve()
    if source_wrapper.is_file():
        return source_root, source_wrapper
    installed = shutil.which("eos-trensmuther")
    if installed is not None:
        return Path.cwd().resolve(), Path(installed).resolve()
    raise ExecutorError(
        "cannot locate the eos-trensmuther console launcher for the Slurm script"
    )


def _print_banner() -> None:
    width = 40
    print("#" * width)
    print("   EoS-TrENsMUTher v%s" % __version__)
    print("#" * width)
    print()


class _ProgressPrinter:
    """One parent-owned terminal progress line; never enters worker state."""

    def __init__(self, label: str = "Inverting EoS") -> None:
        self.label = label
        self.interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.last_completed = -1

    def __call__(self, completed: int, total: int) -> None:
        if not self.interactive or total <= 0 or completed == self.last_completed:
            return
        self.last_completed = completed
        fraction = min(1.0, max(0.0, float(completed) / float(total)))
        width = 28
        filled = int(round(width * fraction))
        bar = "#" * filled + "-" * (width - filled)
        print(
            "\r%s: [%s] %6.2f%% (%d/%d)"
            % (self.label, bar, 100.0 * fraction, completed, total),
            end="",
            flush=True,
        )

    def finish(self) -> None:
        if self.interactive and self.last_completed >= 0:
            print()


def _print_validation_summary(table: ForwardTable) -> None:
    report = table.report
    if report is None:
        return
    geometry = report.statistics.get("geometry", {})
    counts = geometry.get("counts", {}) if isinstance(geometry, dict) else {}
    print("  coordinate axes: OK")
    print("  required finite fields: OK")
    if geometry:
        suspicious = int(counts.get("sign_changing_or_singular", 0))
        invalid = int(counts.get("invalid", 0))
        positive = int(counts.get("regular_positive", 0))
        negative = int(counts.get("regular_negative", 0))
        ill_conditioned = int(counts.get("ill_conditioned", 0))
        sampled = int(geometry.get("sampled_cells", 0))
        if suspicious:
            isolated = geometry.get("isolated_defects")
            suffix = (
                "; %d isolated" % int(isolated)
                if isolated is not None
                else ""
            )
            print(
                "  local geometry: %d sign-changing/singular of %d inspected%s"
                % (suspicious, sampled, suffix)
            )
        elif positive and negative:
            print(
                "  local geometry: mixed orientation (%d positive, %d negative)"
                % (positive, negative)
            )
        else:
            print("  local geometry: %d cells inspected; no obvious issue" % sampled)
        if invalid:
            print("  invalid Jacobians: %d" % invalid)
        if ill_conditioned:
            print("  ill-conditioned sampled cells: %d" % ill_conditioned)


def _print_completion_summary(
    result: PipelineResult, output_path: Path, elapsed: float
) -> None:
    counts = result.status_counts
    print("\nCompleted:")
    print("  Solved: %d" % counts.get("solved", 0))
    print("  Projected: %d" % counts.get("projected", 0))
    print(
        "  Jacobian-selected: %d"
        % int(result.jacobian_selected_mask.sum())
    )
    print("  Unresolved: %d" % counts.get("unresolved", 0))
    print("  Errors: %d" % counts.get("error", 0))
    print("  Output: %s" % output_path)
    print("  Total execution time: %.3f s" % elapsed)


def _format_user_error(exc: BaseException) -> str:
    if isinstance(exc, EoSValidationError):
        orientation_issues = [
            issue
            for issue in exc.report.errors
            if issue.code == "geometry.orientation_reversal"
        ]
        if orientation_issues:
            issue = orientation_issues[0]
            positive = int(issue.context.get("positive_cells", 0))
            negative = int(issue.context.get("negative_cells", 0))
            singular = int(
                issue.context.get("sign_changing_or_singular_cells", 0)
            )
            invalid = int(issue.context.get("invalid_cells", 0))
            sampled = int(issue.context.get("sampled_cells", 0))
            lines = [
                "The source EoS table shows strong evidence that it is not "
                "sufficiently well behaved for this inverter.",
                "",
                "The inversion cannot be performed.",
                "",
                "Local orientation is the sign of the Jacobian determinant "
                "of the complete forward map from source variables to target "
                "variables (for example, (T, muB) -> (e, nB)). A well-behaved "
                "ordinary table may have either orientation, but it should "
                "not switch between positive and negative regular cells.",
                "",
                "Cells inspected: %d" % sampled,
                "  Regular positive: %d" % positive,
                "  Regular negative: %d" % negative,
                "  Sign-changing or singular: %d" % singular,
            ]
            if invalid:
                lines.append("  Invalid Jacobian: %d" % invalid)
            lines.extend(
                [
                    "",
                    "Finding both regular signs indicates a likely fold or "
                    "non-injective region.",
                    "",
                    "If this EoS contains a genuine first-order phase "
                    "transition, declare it in the metadata so that the "
                    "dedicated phase-aware inversion is used.",
                ]
            )
            other_errors = [
                item
                for item in exc.report.errors
                if item.code != "geometry.orientation_reversal"
            ]
            if other_errors:
                lines.extend(["", "Other table problems were also found:"])
                lines.extend("  %s" % item.message for item in other_errors)
            return "\n".join(lines)
        lines = ["Table validation failed:"]
        for issue in exc.report.errors:
            lines.append("  %s" % issue.message)
            if issue.path:
                lines.append("    location: %s" % issue.path)
            row_indices = issue.context.get("row_indices")
            values = issue.context.get("values")
            if row_indices:
                lines.append("    row indices: %s" % list(row_indices))
            if values:
                lines.append("    values: %s" % list(values))
        return "\n".join(lines)
    return str(exc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    parsed = parser.parse_args(_normalize_arguments(arguments))
    started = time.perf_counter()
    try:
        _print_banner()
        if parsed.command == "slurm-task":
            shard = execute_array_shard(parsed.plan, parsed.shard_index)
            print("Shard artifact:", shard)
            return 0
        if parsed.command == "slurm-assemble":
            array_plan = load_array_plan(parsed.plan)
            result = assemble_array_shards(
                array_plan,
                output_path=parsed.output,
                overwrite=parsed.overwrite,
            )
            print(
                "Assembled inverse artifact:",
                parsed.output or array_plan.final_artifact,
            )
            print("Point dispositions:", dict(result.status_counts))
            return 0
        print("[1/6] Reading configuration and EoS table...")
        config, table = load_application_config(
            parsed.parameters,
            output_directory=parsed.output_folder,
            overwrite=True if parsed.overwrite else None,
            specification=parsed.spec,
            operation=parsed.command,
        )
        print("[2/6] Validating source table...")
        _print_validation_summary(table)
        if parsed.command == "slurm-plan":
            print("[3/3] Creating Slurm plan...")
            _prepare_output_directories(config)
            cluster_directory = config.diagnostic_directory / "slurm"
            cluster_directory.mkdir(parents=True, exist_ok=True)
            default_working_directory, entrypoint = _slurm_launcher()
            working_directory = (
                default_working_directory
                if parsed.working_directory is None
                else parsed.working_directory
            )
            log_directory = (
                cluster_directory / "logs"
                if parsed.log_directory is None
                else parsed.log_directory
            )
            scheduler = SlurmArrayConfig(
                cpus_per_task=parsed.cpus_per_task,
                account=parsed.account,
                partition=parsed.partition,
                wall_time=parsed.wall_time,
                memory=parsed.memory,
                job_name=parsed.job_name,
                working_directory=working_directory,
                log_directory=log_directory,
                activation_commands=tuple(parsed.activate),
                entrypoint=entrypoint,
            )
            array_plan = create_array_plan(
                table,
                config.pipeline,
                reachability_configuration=config.reachability,
                output_directory=cluster_directory,
                final_artifact=config.result_path,
                slurm=scheduler,
                compression=config.artifact_compression,
            )
            plan_path = save_array_plan(
                array_plan,
                cluster_directory / "array_plan.json",
                overwrite=config.overwrite,
            )
            script_path = render_slurm_script(
                plan_path,
                cluster_directory / "submit_array.sbatch",
                overwrite=config.overwrite,
            )
            print("Array plan:", plan_path)
            print("Slurm script:", script_path)
            print("Array tasks: 0-%d" % (array_plan.shape[0] - 1))
            print("No job was submitted; inspect the script and run sbatch explicitly.")
        elif parsed.command == "preprocess":
            print("[3/3] Writing preprocessing summary...")
            report = preprocess(config, table)
            print("Validated table:", table.metadata.name if table.metadata else "")
            print("Output:", report)
        elif parsed.command == "invert":
            print("[3/4] Preparing interpolation and reachability...")
            print("[4/4] Inverting EoS...")
            progress = _ProgressPrinter()
            result = invert(config, table, progress_callback=progress)
            progress.finish()
            _print_completion_summary(
                result, config.result_path, time.perf_counter() - started
            )
        elif parsed.command in {"export-text", "postprocess"}:
            print("[3/3] Exporting results...")
            exported = postprocess_text(config)
            print("Text table:", exported.table_path)
            print(
                "Rows: %d (solved=%d, projected=%d)"
                % (
                    exported.row_count,
                    exported.solved_count,
                    exported.projected_count,
                )
            )
        else:
            print("[3/6] Writing preprocessing summary...")
            preprocess(config, table)
            print("[4/6] Preparing interpolation and reachability...")
            print("[5/6] Inverting EoS...")
            progress = _ProgressPrinter()
            result = invert(config, table, progress_callback=progress)
            progress.finish()
            print("[6/6] Exporting results...")
            exported = postprocess(config, result) if config.export_enabled else None
            if exported is not None:
                print("Text table:", exported.table_path)
                print(
                    "Text rows: %d (solved=%d, projected=%d)"
                    % (
                        exported.row_count,
                        exported.solved_count,
                        exported.projected_count,
                    )
                )
            _print_completion_summary(
                result, config.output_directory, time.perf_counter() - started
            )
        if parsed.command not in {"run", "invert"}:
            print("Run directory:", config.output_directory)
        if config.ignored_legacy_parameters:
            print(
                "Note: ignored legacy-only parameters:",
                ", ".join(config.ignored_legacy_parameters),
            )
        return 0
    except (
        ApplicationConfigError,
        EoSError,
        ExecutorError,
        ExportError,
        FileExistsError,
        OSError,
        PersistenceError,
        PipelineError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            "%s: error: %s" % (parser.prog, _format_user_error(exc)),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "invert",
    "main",
    "postprocess",
    "postprocess_text",
    "preprocess",
    "run_all",
]
