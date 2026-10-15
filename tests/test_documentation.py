"""Release-facing documentation and executable quick-start regressions."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from eos_trensmuther import ForwardTable
from eos_trensmuther.exporters import EXTRA_INFO_COLUMN_NAMES


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENTS = (
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    ROOT / "examples/README.md",
    ROOT / "examples/regular/README.md",
    ROOT / "examples/first_order_line/README.md",
    ROOT / "examples/first_order_binodals/README.md",
    ROOT / "AIutils/README.md",
    ROOT / "AIutils/MDSkillsAI.md",
    ROOT / "AIutils/RunSkillsAI.md",
    ROOT / "AIutils/OutputSkillsAI.md",
    ROOT / "AIutils/FirstOrderSkillsAI.md",
    ROOT / "templates/README.md",
    ROOT / "templates/METADATA_REFERENCE.md",
    ROOT / "docs/USER_GUIDE.md",
    ROOT / "docs/METADATA.md",
    ROOT / "docs/PARAMETERS.md",
    ROOT / "docs/METHODOLOGY.md",
    ROOT / "docs/KNOWN_LIMITATIONS.md",
    ROOT / "docs/BENCHMARKS.md",
    ROOT / "docs/PUBLIC_API.md",
    ROOT / "docs/PROJECTION.md",
    ROOT / "docs/OUTPUT_FORMAT.md",
    ROOT / "utilities/README.md",
)
LINK_PATTERN = re.compile(r"\[[^]]*\]\(([^)]+)\)")


class DocumentationTests(unittest.TestCase):
    def test_public_tree_excludes_development_history_and_hydro_specific_ui(self) -> None:
        for path in (
            ROOT / "docs/development",
            ROOT / "docs/Modifications.md",
            ROOT / "scripts",
        ):
            self.assertFalse(path.exists(), str(path))
        text = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCUMENTS)
        self.assertNotIn("export-music", text)
        self.assertNotIn("export-vhlle", text)
        self.assertNotIn("MUSICExportConfig", text)
        self.assertNotIn("VHLLEExportConfig", text)

    def test_metadata_and_parameter_templates_are_complete_user_entry_points(self) -> None:
        for path in (
            ROOT / "templates/metadata_minimal.json",
            ROOT / "templates/metadata_regular.json",
            ROOT / "templates/metadata_first_order_line.json",
            ROOT / "templates/metadata_first_order_binodals.json",
            ROOT / "templates/metadata_master.json",
            ROOT / "templates/METADATA_REFERENCE.md",
            ROOT / "templates/parameters_regular.py",
            ROOT / "templates/parameters_first_order_line.py",
            ROOT / "templates/parameters_first_order_binodals.py",
            ROOT / "templates/parameters.master.py",
        ):
            self.assertTrue(path.is_file(), str(path))
        metadata_text = (ROOT / "docs/METADATA.md").read_text(encoding="utf-8")
        parameter_text = (ROOT / "docs/PARAMETERS.md").read_text(encoding="utf-8")
        for marker in ("**REQUIRED**", "**OPTIONAL", "**ADVANCED"):
            self.assertIn(marker, metadata_text)
        self.assertIn("templates/parameters.master.py", parameter_text)
        root_master = json.loads(
            (ROOT / "templates/metadata_master.json").read_text(encoding="utf-8")
        )
        self.assertIn("format", root_master)
        self.assertIn("grid", root_master)
        self.assertIn("charge_constraints", root_master)
        self.assertIn("scientific", root_master)
        parsed_master = ForwardTable(
            ROOT / "templates/metadata_master.json"
        ).load_metadata()
        self.assertEqual(parsed_master.dimension, 2)
        self.assertEqual(tuple(parsed_master.fields), ("T", "muB", "e", "nB", "P"))
        reference = (ROOT / "templates/METADATA_REFERENCE.md").read_text(
            encoding="utf-8"
        )
        for marker in (
            "## Top level",
            "## Each `fields.NAME` object",
            "## `format`",
            "## `grid`",
            "## `charge_constraints`",
            "## `scientific`",
            "## First-order-only inputs",
        ):
            self.assertIn(marker, reference)
        for name in (
            "metadata_regular.json",
            "metadata_first_order_line.json",
            "metadata_first_order_binodals.json",
            "metadata_master.json",
        ):
            path = ROOT / "templates" / name
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("fields", document)
            self.assertIn("dimension", document)
            self.assertEqual(ForwardTable(path).load_metadata().dimension, 2)

        self.assertFalse((ROOT / "EoSTables/templates").exists())
        self.assertFalse((ROOT / "EoSTables/metadata.minimal.json").exists())
        self.assertFalse((ROOT / "EoSTables/metadata.template.json").exists())
        self.assertFalse((ROOT / "EoSTables/MDSkillsAI.md").exists())
        self.assertFalse((ROOT / "templates/MDSkillsAI.md").exists())

    def test_regular_examples_use_automatic_residual_normalization(self) -> None:
        for name in (
            "parameters_serial.py",
            "parameters_multicore.py",
            "parameters_slurm.py",
        ):
            text = (ROOT / "examples/regular" / name).read_text(encoding="utf-8")
            self.assertNotIn("ResidualScales", text, name)
            self.assertNotIn("GridSpace", text, name)
            self.assertNotIn("include_projected", text, name)
        master = (ROOT / "templates/parameters.master.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('# "ResidualScales": {', master)
        self.assertIn("default: automatic", master)

    def test_public_docs_use_current_behavior_without_release_shorthand(self) -> None:
        current_behavior_docs = (
            ROOT / "README.md",
            ROOT / "docs/USER_GUIDE.md",
            ROOT / "docs/METADATA.md",
            ROOT / "docs/PARAMETERS.md",
            ROOT / "docs/METHODOLOGY.md",
            ROOT / "docs/KNOWN_LIMITATIONS.md",
            ROOT / "docs/OUTPUT_FORMAT.md",
        )
        text = "\n".join(path.read_text(encoding="utf-8") for path in current_behavior_docs)
        for stale in ("For v1", "In v1", "V1 only", "outside v1", "in v1"):
            self.assertNotIn(stale, text)

    def test_first_order_docs_do_not_label_supported_route_experimental(self) -> None:
        documents = (
            ROOT / "README.md",
            ROOT / "AIutils/README.md",
            ROOT / "AIutils/MDSkillsAI.md",
            ROOT / "AIutils/FirstOrderSkillsAI.md",
            ROOT / "docs/KNOWN_LIMITATIONS.md",
            ROOT / "docs/METADATA.md",
            ROOT / "docs/METHODOLOGY.md",
            ROOT / "docs/OUTPUT_FORMAT.md",
            ROOT / "docs/USER_GUIDE.md",
            ROOT / "templates/README.md",
            ROOT / "templates/parameters.master.py",
            ROOT / "examples/README.md",
            ROOT / "examples/first_order_line/README.md",
            ROOT / "examples/first_order_binodals/README.md",
        )
        text = " ".join(
            re.sub(r"\s+", " ", path.read_text(encoding="utf-8").lower())
            for path in documents
        )
        self.assertNotRegex(text, r"experimental.{0,80}first-order")
        self.assertNotRegex(text, r"first-order.{0,80}experimental")

    def test_ai_metadata_instructions_forbid_scientific_guessing(self) -> None:
        text = (ROOT / "AIutils/MDSkillsAI.md").read_text(encoding="utf-8")
        for required in (
            "DO NOT GUESS",
            "count exact or tolerance-justified unique values",
            "Cartesian coordinate tuple",
            "phase-fraction field",
            "critical-point coordinates",
        ):
            self.assertIn(required, text)

    def test_ai_resources_are_indexed_complete_and_current(self) -> None:
        directory = ROOT / "AIutils"
        index = (directory / "README.md").read_text(encoding="utf-8")
        agent_entrypoint = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("AIutils/README.md", agent_entrypoint)
        resources = tuple(
            path
            for path in sorted(directory.glob("*.md"))
            if path.name != "README.md"
        )
        self.assertEqual(
            tuple(path.name for path in resources),
            (
                "FirstOrderSkillsAI.md",
                "MDSkillsAI.md",
                "OutputSkillsAI.md",
                "RunSkillsAI.md",
            ),
        )
        self.assertFalse((directory / "FirsOrderSkillsAI.md").exists())
        for resource in resources:
            with self.subTest(resource=resource.name):
                text = resource.read_text(encoding="utf-8")
                self.assertIn(resource.name, index)
                self.assertEqual(text.count("```") % 2, 0)
                self.assertGreater(len(text.splitlines()), 20)

        output_text = (directory / "OutputSkillsAI.md").read_text(
            encoding="utf-8"
        )
        for name in EXTRA_INFO_COLUMN_NAMES:
            self.assertIn(name, output_text)

    def test_ci_and_citation_are_repository_versioned(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        for version in ('"3.9"', '"3.10"', '"3.11"', '"3.12"', '"3.13"'):
            self.assertIn(version, workflow)
        self.assertIn("python -m pip install .", workflow)
        self.assertIn("unittest discover", workflow)
        self.assertIn("title: \"EoS-TrENsMUTher\"", citation)
        self.assertIn("license: MIT", citation)
        self.assertIn("How to cite", (ROOT / "README.md").read_text(encoding="utf-8"))

    def test_public_document_links_resolve_inside_source_tree(self) -> None:
        missing = []
        for document in PUBLIC_DOCUMENTS:
            text = document.read_text(encoding="utf-8")
            for target in LINK_PATTERN.findall(text):
                target = target.strip().strip("<>").split("#", 1)[0]
                target = target.split("?", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                # TeX such as ``[J](X)`` is not a Markdown resource link.
                if len(target) == 1 and target.isalpha():
                    continue
                resolved = (document.parent / target).resolve()
                if not resolved.exists():
                    missing.append("%s -> %s" % (document.relative_to(ROOT), target))
        self.assertEqual(missing, [])

    def test_primary_user_path_does_not_require_source_path_hacks(self) -> None:
        documents = (
            ROOT / "README.md",
            ROOT / "examples/README.md",
            ROOT / "docs/USER_GUIDE.md",
            ROOT / "docs/METHODOLOGY.md",
            ROOT / "docs/KNOWN_LIMITATIONS.md",
        )
        text = "\n".join(path.read_text(encoding="utf-8") for path in documents)
        for stale in (
            "PYTHONPATH=src",
            "python3 EoS-TrENsMUTher.py",
            "LEGACY.md",
            "PHASE6.md",
        ):
            with self.subTest(stale=stale):
                self.assertNotIn(stale, text)

    def test_benchmark_note_records_measurement_context(self) -> None:
        text = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        for required in (
            "810,000",
            "6,668.341952",
            "121.473635 targets/s",
            "16,036",
            "793,964",
            "Unresolved | 0",
            "Errors | 0",
            "approximately four-hour",
            "not a performance promise",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_public_api_quick_start_without_and_with_projection(self) -> None:
        environment = dict(os.environ)
        script = ROOT / "utilities/quickstart_api.py"
        with tempfile.TemporaryDirectory(prefix="eos-phase16-docs-") as directory:
            root = Path(directory)
            expected = (
                ((), {"solved": 3, "certified_unreachable": 6, "projected": 0}),
                (("--projection",), {"solved": 3, "certified_unreachable": 6, "projected": 6}),
            )
            for index, (options, counts) in enumerate(expected):
                command = [
                    sys.executable,
                    str(script),
                    *options,
                    "--output",
                    str(root / ("run-%d" % index)),
                ]
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                report = json.loads(completed.stdout)
                self.assertEqual(report["targets"], 9)
                self.assertEqual(report["unresolved"], 0)
                self.assertEqual(report["errors"], 0)
                self.assertTrue(report["artifact_round_trip"])
                for name, value in counts.items():
                    self.assertEqual(report[name], value)


if __name__ == "__main__":
    unittest.main()
