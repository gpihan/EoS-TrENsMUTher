"""Guards against reintroducing dependencies on the development repository."""

from __future__ import annotations

import pkgutil
import unittest
from pathlib import Path

import eos_trensmuther

from eos_trensmuther import ForwardTable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "eos_trensmuther"
OLD_REPOSITORY = Path.home() / "Documents" / "EoSInverterpAI" / "EoSInverter"
FORBIDDEN_TOP_LEVEL = {"TEMP", "Output_TEST", "checkpoints"}


class StandaloneRepositoryTests(unittest.TestCase):
    def test_package_resolves_inside_this_repository(self) -> None:
        package_path = Path(eos_trensmuther.__file__).resolve()
        old_repository = OLD_REPOSITORY.resolve()
        self.assertFalse(package_path.is_relative_to(old_repository))
        if package_path.is_relative_to(REPOSITORY_ROOT.resolve()):
            self.assertTrue(package_path.is_relative_to(PACKAGE_ROOT.resolve()))
        else:
            self.assertIn("site-packages", package_path.parts)
        for module in pkgutil.iter_modules(eos_trensmuther.__path__):
            __import__(f"eos_trensmuther.{module.name}")

    def test_development_output_trees_are_not_shipped(self) -> None:
        present = sorted(
            name for name in FORBIDDEN_TOP_LEVEL if (REPOSITORY_ROOT / name).exists()
        )
        self.assertEqual(present, [])

    def test_no_symlinks_or_absolute_development_tree_references(self) -> None:
        symlinks = [
            str(path.relative_to(REPOSITORY_ROOT))
            for path in REPOSITORY_ROOT.rglob("*")
            if ".git" not in path.parts and path.is_symlink()
        ]
        self.assertEqual(symlinks, [])

        violations = []
        legacy_relative_reference = ".." + "/EoSInverter/"
        for directory in ("src", "tests", "examples", "scripts"):
            for path in (REPOSITORY_ROOT / directory).rglob("*.py"):
                if path.resolve() == Path(__file__).resolve():
                    continue
                text = path.read_text(encoding="utf-8")
                if str(OLD_REPOSITORY) in text or legacy_relative_reference in text:
                    violations.append(str(path.relative_to(REPOSITORY_ROOT)))
        self.assertEqual(violations, [])

    def test_bundled_fixture_loads_without_external_data(self) -> None:
        metadata = REPOSITORY_ROOT / "examples/regular/metadata.json"
        table = ForwardTable.load_validated(metadata)
        self.assertEqual(table.shape, (4, 4))
        self.assertEqual(table.coordinate_names, ("T", "muB"))
        self.assertEqual(table.target_names, ("e", "nB"))


if __name__ == "__main__":
    unittest.main()
