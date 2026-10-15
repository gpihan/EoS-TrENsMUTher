"""Architecture guards for the clean standalone v1 repository."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "eos_trensmuther"
LEGACY_MODULES = {
    "Check",
    "FindVals",
    "PostProcessor",
    "mapEOS_1D",
    "mapEOS_2D",
    "mapEOS_3D",
    "mapEOS_4D",
    "merger_eos",
    "utils",
}
LEGACY_FILES = tuple(
    REPOSITORY_ROOT / f"{module_name}.py"
    for module_name in sorted(LEGACY_MODULES)
) + (REPOSITORY_ROOT / "parameters.py",)
PROCESS_CALLS = {
    "os.system",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.run",
}


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


class LegacyIsolationTests(unittest.TestCase):
    def test_package_does_not_import_or_execute_legacy_scripts(self) -> None:
        violations: list[str] = []
        legacy_filenames = {path.name for path in LEGACY_FILES}

        for module_path in sorted(PACKAGE_ROOT.glob("*.py")):
            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".", maxsplit=1)[0]
                        if root in LEGACY_MODULES:
                            violations.append(f"{module_path.name}: imports {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    root = node.module.split(".", maxsplit=1)[0]
                    if root in LEGACY_MODULES:
                        violations.append(f"{module_path.name}: imports from {node.module}")
                elif isinstance(node, ast.Call):
                    call_name = _qualified_name(node.func)
                    if call_name not in PROCESS_CALLS:
                        continue
                    constants = {
                        value.value
                        for value in ast.walk(node)
                        if isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    }
                    for command in constants:
                        for filename in legacy_filenames:
                            if filename in command:
                                violations.append(
                                    f"{module_path.name}: {call_name} executes {filename}"
                                )

        self.assertEqual(violations, [])

    def test_obsolete_legacy_implementations_are_not_shipped(self) -> None:
        self.assertEqual(
            [path.name for path in LEGACY_FILES if path.exists()],
            [],
        )

    def test_root_entry_point_delegates_to_package_cli(self) -> None:
        entry_point = REPOSITORY_ROOT / "EoS-TrENsMUTher.py"
        tree = ast.parse(entry_point.read_text(encoding="utf-8"))
        imports = {
            (node.module, tuple(alias.name for alias in node.names))
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertIn(("eos_trensmuther.cli", ("main",)), imports)


if __name__ == "__main__":
    unittest.main()
