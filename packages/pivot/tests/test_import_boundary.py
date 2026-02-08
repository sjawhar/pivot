"""Tests enforcing the import boundary between core pivot and pivot-tui.

After the TUI package extraction, core pivot must not depend on textual
or pivot.tui. These tests catch accidental re-introduction of those imports.
"""

from __future__ import annotations

import importlib
import pathlib
import re
import sys

# =============================================================================
# Import boundary enforcement
# =============================================================================

_TEXTUAL_IMPORT_RE = re.compile(r"\bfrom\s+textual\b|\bimport\s+textual\b")
_PIVOT_TUI_IMPORT_RE = re.compile(r"\bfrom\s+pivot\.tui\b|\bimport\s+pivot\.tui\b")
_PIVOT_TUI_PKG_IMPORT_RE = re.compile(r"\bfrom\s+pivot_tui\b|\bimport\s+pivot_tui\b")


def test_no_textual_imports_in_core() -> None:
    """Core pivot must not import Textual anywhere."""
    root = pathlib.Path("src/pivot")
    violations = list[str]()
    for p in root.rglob("*.py"):
        text = p.read_text("utf-8")
        if _TEXTUAL_IMPORT_RE.search(text):
            violations.append(str(p))
    assert not violations, f"Textual imports found in core: {violations}"


def test_no_pivot_tui_imports_in_core() -> None:
    """Core pivot must not import from pivot.tui (old package location)."""
    root = pathlib.Path("src/pivot")
    violations = list[str]()
    for p in root.rglob("*.py"):
        text = p.read_text("utf-8")
        if _PIVOT_TUI_IMPORT_RE.search(text):
            violations.append(str(p))
    assert not violations, f"pivot.tui imports found in core: {violations}"


def test_no_pivot_tui_imports_in_pivot_tui_package() -> None:
    """pivot-tui package must not import from pivot.tui (old package location)."""
    root = pathlib.Path("packages/pivot-tui")
    violations = list[str]()
    for p in root.rglob("*.py"):
        text = p.read_text("utf-8")
        if _PIVOT_TUI_IMPORT_RE.search(text):
            violations.append(str(p))
    assert not violations, f"pivot.tui imports found in pivot-tui package: {violations}"


# =============================================================================
# Core import smoke tests
# =============================================================================


def test_core_cli_modules_importable() -> None:
    """All core CLI modules must be importable without pivot-tui."""
    modules = [
        "pivot.cli",
        "pivot.cli.console",
        "pivot.cli.data",
        "pivot.cli.decorators",
        "pivot.cli.helpers",
        "pivot.cli.repro",
        "pivot.cli.run",
        "pivot.cli.status",
        "pivot.cli._run_common",
    ]
    failures = list[str]()
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            failures.append(f"{mod}: {e}")
    assert not failures, "Core CLI modules failed to import:\n" + "\n".join(failures)


def test_core_engine_modules_importable() -> None:
    """Core engine modules must be importable without pivot-tui."""
    modules = [
        "pivot.engine",
        "pivot.engine.engine",
        "pivot.engine.sinks",
        "pivot.engine.types",
    ]
    failures = list[str]()
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            failures.append(f"{mod}: {e}")
    assert not failures, "Core engine modules failed to import:\n" + "\n".join(failures)


# =============================================================================
# Lazy import verification
# =============================================================================


def test_core_cli_imports_are_lazy() -> None:
    """Importing core CLI modules must NOT load pivot_tui or textual."""
    saved_modules = {
        mod: sys.modules[mod]
        for mod in list(sys.modules.keys())
        if mod.startswith(("pivot_tui", "textual"))
    }

    try:
        # Clear any existing imports
        for mod in list(saved_modules.keys()):
            del sys.modules[mod]

        # Re-import core CLI modules (force fresh import)
        for mod_name in (
            "pivot.cli.run",
            "pivot.cli.repro",
            "pivot.cli.data",
            "pivot.cli._run_common",
        ):
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])
            else:
                importlib.import_module(mod_name)

        # Assert pivot_tui and textual were NOT loaded
        loaded_modules = set(sys.modules.keys())

        pivot_tui_loaded = [m for m in loaded_modules if m.startswith("pivot_tui")]
        textual_loaded = [m for m in loaded_modules if m.startswith("textual")]

        assert not pivot_tui_loaded, f"pivot_tui modules loaded: {pivot_tui_loaded}"
        assert not textual_loaded, f"textual modules loaded: {textual_loaded}"
    finally:
        # Remove any new modules and restore previous state to avoid class identity issues.
        for mod in list(sys.modules.keys()):
            if mod.startswith(("pivot_tui", "textual")):
                del sys.modules[mod]
        sys.modules.update(saved_modules)


# =============================================================================
# Allowlist enforcement
# =============================================================================


def test_pivot_tui_imports_only_from_cli() -> None:
    """pivot_tui may only be imported from src/pivot/cli/."""
    root = pathlib.Path("src/pivot")
    violations = list[str]()

    for p in root.rglob("*.py"):
        # Skip CLI directory - it's allowed
        if p.is_relative_to(root / "cli"):
            continue

        text = p.read_text("utf-8")

        # Check for pivot_tui imports
        if _PIVOT_TUI_PKG_IMPORT_RE.search(text):
            violations.append(str(p))

    assert not violations, (
        f"pivot_tui imported outside src/pivot/cli/: {violations}. "
        "Only CLI modules may import pivot_tui."
    )
