# User templates

All reusable metadata and run-parameter examples live here. `EoSTables/`
contains actual table directories and their table-specific metadata only.

## Metadata progression

1. [`metadata_minimal.json`](metadata_minimal.json) contains only the required
   information for a simple regular 2D table.
2. [`metadata_regular.json`](metadata_regular.json) is the recommended
   ordinary-table starting point, with parsing, grid, and scientific
   provenance sections.
3. [`metadata_first_order_line.json`](metadata_first_order_line.json) and
   [`metadata_first_order_binodals.json`](metadata_first_order_binodals.json)
   are the two mutually exclusive first-order starters.
4. [`metadata_master.json`](metadata_master.json) shows every mutually
   compatible ordinary-table field, including conversions, constraints,
   provenance, validity, and a critical point.

Every supported input, its requirement level, and its default are explained
in [`METADATA_REFERENCE.md`](METADATA_REFERENCE.md). The AI-assisted table
identification checklist is
[`AIutils/MDSkillsAI.md`](../AIutils/MDSkillsAI.md).

JSON does not support comments. Keeping the examples valid JSON lets the
actual metadata parser check them; explanations therefore live in the
companion Markdown reference rather than as unsupported comment keys.

## Run parameters

- [`parameters_regular.py`](parameters_regular.py): short ordinary run.
- [`parameters_first_order_line.py`](parameters_first_order_line.py): LINE.
- [`parameters_first_order_binodals.py`](parameters_first_order_binodals.py):
  BINODALS.
- [`parameters.master.py`](parameters.master.py): complete supported public
  configuration surface with inline explanations.
