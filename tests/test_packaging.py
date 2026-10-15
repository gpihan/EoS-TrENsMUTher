"""Static packaging and dependency-contract regressions."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import eos_trensmuther


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/eos_trensmuther"


class PackagingContractTests(unittest.TestCase):
    def test_pyproject_uses_single_version_source_and_console_entry(self) -> None:
        document = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('dynamic = ["version"]', document)
        self.assertIn(
            'version = { attr = "eos_trensmuther._version.__version__" }',
            document,
        )
        self.assertIn('eos-trensmuther = "eos_trensmuther.cli:main"', document)
        self.assertIn('requires-python = ">=3.9,<3.14"', document)
        for minor in range(9, 14):
            self.assertIn(
                '"Programming Language :: Python :: 3.%d"' % minor,
                document,
            )
        self.assertEqual(eos_trensmuther.__version__, "1.0.0.dev0")

    def test_runtime_dependency_surface_is_minimal(self) -> None:
        external = set()
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    external.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    external.add(node.module.split(".")[0])
        third_party = external.intersection(
            {"matplotlib", "numpy", "pandas", "scipy", "threadpoolctl"}
        )
        self.assertEqual(third_party, {"numpy", "scipy", "threadpoolctl"})

    def test_requirements_match_runtime_dependencies(self) -> None:
        requirements = {
            line.split(">=", 1)[0]
            for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line and not line.startswith("#")
        }
        self.assertEqual(requirements, {"numpy", "scipy", "threadpoolctl"})

    def test_data_distribution_decision_is_explicit_and_bounded(self) -> None:
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("prune EoSTables", manifest)
        self.assertNotIn("recursive-include EoSTables", manifest)
        self.assertIn("/EoSTables/", gitignore)
        self.assertFalse((PACKAGE / "data").exists())

    def test_mit_license_is_declared_and_distributed(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertTrue(license_text.startswith("MIT License\n"))
        self.assertIn("Copyright (c) 2026 Gregoire Pihan", license_text)
        self.assertIn('license = "MIT"', pyproject)
        self.assertIn('license-files = ["LICENSE"]', pyproject)
        self.assertIn("include LICENSE", manifest)
        self.assertIn("include CITATION.cff", manifest)
        self.assertFalse((ROOT / "LICENSE_STATUS.md").exists())


if __name__ == "__main__":
    unittest.main()
