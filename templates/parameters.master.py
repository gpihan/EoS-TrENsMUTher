"""Master public parameter reference.

Copy this file and remove settings you do not need. Values shown below are
the defaults unless marked REQUIRED or EXAMPLE. This file is a reference,
not a runnable example: replace the input path and grid axes.
"""

general_parameters = {
    # ------------------------------------------------------------------
    # INPUT / REQUIRED
    # ------------------------------------------------------------------
    # REQUIRED: JSON metadata describing the source EoS table.
    "EoS_metadata": "path/to/metadata.json",
    # OPTIONAL — default: the data_file named by metadata.
    # Use only to read the same metadata with another local data snapshot.
    # "EoS_table": "path/to/table.dat",
    # OPTIONAL — default: checked against metadata only when supplied.
    # "Dimension": 2,

    # ------------------------------------------------------------------
    # TARGET GRID
    # ------------------------------------------------------------------
    # OPTIONAL — default: regular target variables (e, nB, ...).
    # Add this one setting only when transformed coordinates are wanted:
    # "Variables": "tilde",
    # REQUIRED: one entry per target-grid axis. An integer uses automatic
    # metadata-derived bounds. To choose bounds, use
    # {"count": N, "lower": MIN, "upper": MAX}.
    "GridAxes": {"e": 20, "nB": 20},
    # OPTIONAL — default: every metadata auxiliary with output=true.
    "AuxiliaryFields": ["P", "s", "cs2", "chi2B"],

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------
    # OPTIONAL — default: "Output", resolved beside this parameter file.
    "OutputFolder": "Output",
    # OPTIONAL — default: metadata directory name.
    "EoSName": "MyEoS",
    # OPTIONAL — default: no suffix. Must start with "_" when supplied.
    # "RunSpecification": "_test",
    # OPTIONAL — default: "default" (create a numbered run when needed).
    # Use "no" to refuse a collision or "yes" to replace this run's output.
    "Overwrite": "default",
    # OPTIONAL — default: "none". "deflate" makes smaller .eosinv artifacts.
    "ArtifactCompression": "none",
    # OPTIONAL — default: True. Append deterministic 0/1 state/provenance
    # columns such as projected, mixed_phase, and selected_from_jacobian.
    "ExtraInfo": True,

    # ------------------------------------------------------------------
    # GENERIC TEXT EXPORT
    # ------------------------------------------------------------------
    "Export": {
        # OPTIONAL — default: True. The .eosinv artifact is always authoritative.
        "enabled": True,
        # OPTIONAL — default: "eos_table.dat", inside the run directory.
        "filename": "eos_table.dat",
        # ADVANCED — default: False. If True, certified-unreachable rows are
        # retained with unavailable state and auxiliary values written as nan.
        # Unresolved, invalid, and error rows always prevent text export.
        "include_unusable": False,
        # ADVANCED — default: False. Permit nan in explicitly allowed
        # auxiliary fields instead of refusing the export.
        "allow_undefined_auxiliary": False,
        # OPTIONAL — default: True. Conventional '# ' lines describe version,
        # columns, units, and grid order.
        "write_header": True,
        # OPTIONAL — default: False. For Variables="tilde", append the four
        # requested transformed-grid coordinates T_tilde and muB/Q/S_tilde.
        # These remain requested coordinates on projected rows; the projected
        # flag identifies rows whose returned physical state differs.
        "include_tilde": False,
        # OPTIONAL — default: True. Write a checksummed JSON sidecar.
        "write_manifest": True,
        # ADVANCED — default: "%.8e".
        "float_format": "%.8e",
        # ADVANCED — default: no phase columns. For first-order
        # output, enable selected names from docs/OUTPUT_FORMAT.md.
        # "phase": {"columns": ["phase_code", "phase_route_code"]},
    },

    # ------------------------------------------------------------------
    # PARALLEL EXECUTION
    # ------------------------------------------------------------------
    "Execution": {
        # OPTIONAL — default: 1. Values >1 use persistent local processes.
        "workers": 1,
        # ADVANCED — default: "spawn". Also accepts supported platform methods.
        "start_method": "spawn",
        # OPTIONAL — default: 256. Atomic target rows per checkpoint chunk.
        "chunk_size": 256,
        # OPTIONAL — default: 8. Maximum target rows in one process task.
        "worker_task_size": 8,
        # ADVANCED — default: automatic bounded queue size.
        "max_pending_chunks": None,
        # OPTIONAL — default: "fail_fast". "accumulate" records point errors.
        "error_policy": "fail_fast",
    },

    # ------------------------------------------------------------------
    # CHECKPOINT / RESTART
    # ------------------------------------------------------------------
    "Checkpoint": {
        # OPTIONAL — default: True.
        "enabled": True,
        # OPTIONAL — default: "checkpoint", below diagnostic/.
        "directory": "checkpoint",
        # OPTIONAL — default: True. Only compatible complete chunks are reused.
        "resume": True,
    },

    # ------------------------------------------------------------------
    # PROJECTION
    # ------------------------------------------------------------------
    "Projection": {
        # OPTIONAL — default: True. Project certified-unreachable targets to a
        # separately labeled feasible state. Set False for exact inverses only.
        "enabled": True,
        # ADVANCED — defaults below; normally leave unchanged.
        "nearest_cell_starts": 8,
        "max_evaluations_per_start": 200,
        "least_squares_ftol": 1.0e-10,
        "least_squares_xtol": 1.0e-10,
        "least_squares_gtol": 1.0e-12,
        "cell_batch_size": 4096,
        "max_index_bytes": 2147483648,
    },

    # ------------------------------------------------------------------
    # ADVANCED SOLVER SETTINGS — normal users should leave these unchanged
    # ------------------------------------------------------------------
    "Solver": {
        "localization_residual_tolerance": 1.0e-6,
        "root_residual_tolerance": 1.0e-10,
        "least_squares_max_evaluations": 200,
        "least_squares_ftol": 1.0e-10,
        "least_squares_xtol": 1.0e-10,
        "least_squares_gtol": 1.0e-12,
        "newton_max_iterations": 24,
        "newton_condition_limit": 1.0e12,
        "newton_step_tolerance": 1.0e-13,
        "newton_armijo": 1.0e-4,
        "newton_backtrack_factor": 0.5,
        "newton_min_step_fraction": 2.0 ** -24,
        "fallback_max_scanned_cells": 2_000_000,
        "fallback_max_candidate_cells": 100_000,
        "fallback_scan_chunk_size": 65_536,
        "fallback_max_boxes": 100_000,
        "fallback_max_depth": 40,
        "fallback_min_normalized_width": 2.0 ** -40,
        "fallback_contraction_volume_ratio": 0.8,
        "fallback_cell_cache_size": 64,
        # These are the three supported ordinary-subcell controls.
        "enable_subcell_fallback": True,
        "max_subcell_depth": 7,
        "max_subcell_active_boxes": 1_000_000,
    },
    # ADVANCED — defaults shown; control conservative reachability memory and
    # the validated exhaustive/hierarchical index choice.
    "Reachability": {
        "cell_batch_size": 4096,
        "query_batch_size": 256,
        "max_enclosure_bytes": 2147483648,
        "max_pairwise_entries": 1_048_576,
        "method": "exhaustive",
        "hierarchy_levels": None,
        "max_hierarchy_bytes": 2147483648,
    },
    # ADVANCED — deterministic low-resolution initial seeds.
    "CoarseSeed": {"enabled": True, "points_per_axis": 3, "maximum_points": 4096},
    # ADVANCED — default: "domain_midpoint".
    "InitialGuess": "domain_midpoint",
    # ADVANCED — default: True.
    "UseReachabilitySeed": True,
    # ADVANCED — default: automatic. Normally omit this setting. The program
    # derives one deterministic numerical normalization per target component
    # from the validated table's maximum absolute value. An explicit mapping
    # overrides those values exactly after positive/finite validation.
    # "ResidualScales": {"e": 0.02, "nB": 0.02},
    # ADVANCED first-order auxiliary interpolation policies.
    "LineAuxiliaryDefaultPolicy": "linear",
    "LineAuxiliaryPolicies": {},

    # ------------------------------------------------------------------
    # DIAGNOSTICS / RESOURCE GUARDS
    # ------------------------------------------------------------------
    # ADVANCED — defaults protect against accidental oversized allocations.
    "MaxGridPoints": 10_000_000,
    "MaxResultBytes": 2147483648,
    # OPTIONAL — default: None; explicit total peak-result guard when needed.
    "MaxPeakResultBytes": None,
    # OPTIONAL diagnostic filenames inside the run directory.
    "PreprocessReport": "preprocess.json",
    "ResultFile": "inverse.eosinv",
}
