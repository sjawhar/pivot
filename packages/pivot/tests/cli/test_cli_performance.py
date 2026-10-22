"""``pivot --help`` and each command's ``--help`` load no heavy library.

Commands load lazily (``pivot.cli._LAZY_COMMANDS``), so help imports only the module that
defines the command. Each case runs the installed ``pivot`` script in a fresh interpreter
under ``-X importtime``, which lists every module the process imports, so the check does not
depend on the machine's load.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from pivot.cli import _LAZY_COMMANDS

_PIVOT_BIN = pathlib.Path(sys.executable).parent / "pivot"

# The engine, the executor's worker and the data libraries they bring in.
_HEAVY_MODULES = frozenset(
    {"pandas", "numpy", "loky", "pivot.engine.engine", "pivot.executor.worker"}
)

# Commands whose modules import heavy modules at their top level, so their --help does too.
_HEAVY_HELP_COMMANDS = frozenset(
    {"commit", "dag", "metrics", "params", "plots", "repro", "run", "status", "verify"}
)


def _helper_help_imports(*args: str) -> set[str]:
    """Run ``pivot <args> --help`` in a fresh interpreter; return the modules it imported."""
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", str(_PIVOT_BIN), *args, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"pivot {' '.join(args)} --help failed:\n{proc.stderr}"
    assert "Usage:" in proc.stdout, proc.stdout
    imported = {
        line.rsplit("|", 1)[1].strip()
        for line in proc.stderr.splitlines()
        if line.startswith("import time:") and not line.endswith("imported package")
    }
    assert "pivot.cli" in imported, f"-X importtime listed no pivot import:\n{proc.stderr}"
    return imported


def test_main_help_imports_no_heavy_module() -> None:
    heavy = sorted(_helper_help_imports() & _HEAVY_MODULES)
    assert not heavy, f"pivot --help imports {heavy}"


@pytest.mark.parametrize("command", sorted(_LAZY_COMMANDS.keys() - _HEAVY_HELP_COMMANDS))
def test_command_help_imports_no_heavy_module(command: str) -> None:
    heavy = sorted(_helper_help_imports(command) & _HEAVY_MODULES)
    assert not heavy, f"pivot {command} --help imports {heavy}"


@pytest.mark.parametrize("command", sorted(_HEAVY_HELP_COMMANDS))
def test_heavy_help_command_still_imports_a_heavy_module(command: str) -> None:
    """Fails once a listed command's --help stops importing heavy modules, so the command
    leaves _HEAVY_HELP_COMMANDS and the check above covers it."""
    assert command in _LAZY_COMMANDS, f"{command} is not a pivot command"
    assert _helper_help_imports(command) & _HEAVY_MODULES, (
        f"pivot {command} --help no longer imports {sorted(_HEAVY_MODULES)}; "
        + "remove it from _HEAVY_HELP_COMMANDS"
    )
