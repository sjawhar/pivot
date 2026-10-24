"""The state directory and the config modules import one another without a cycle.

`pivot.config_files` is a leaf that reads one config file. `pivot.state_dir` reads the project
config through it alone, and `pivot.config.io`, behind the `pivot.config` package, imports
`pivot.state_dir`. So importing `pivot.state_dir` never loads the `pivot.config` package, and no
module among them starts importing a module that has not finished its own import, whichever one
a process imports first.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_MODULES = ("pivot.config_files", "pivot.state_dir", "pivot.config.io", "pivot.config")

# Run in a fresh interpreter: records, for every import statement that names one of the modules,
# whether that module was still executing its own body at that moment. A submodule naming its
# own package (`from pivot.config import models` inside `pivot.config.io`) is how a package
# loads its parts, not a cycle, so that edge is left out.
_PROBE = """
import builtins
import importlib
import importlib.machinery
import json
import sys

watched = set(sys.argv[2:])
executing = []
cycles = []
real_exec = importlib.machinery.SourceFileLoader.exec_module
real_import = builtins.__import__


def exec_module(self, module):
    executing.append(module.__name__)
    try:
        return real_exec(self, module)
    finally:
        executing.pop()


def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0:
        targets = [name, *(f"{name}.{item}" for item in fromlist or () if item != "*")]
        importer = globals.get("__name__", "") if globals else ""
        for target in targets:
            inside_target = importer == target or importer.startswith(f"{target}.")
            if target in watched and target in executing and not inside_target:
                cycles.append([importer, target])
    return real_import(name, globals, locals, fromlist, level)


importlib.machinery.SourceFileLoader.exec_module = exec_module
builtins.__import__ = tracking_import
importlib.import_module(sys.argv[1])
print(json.dumps({"cycles": cycles, "loaded": sorted(m for m in sys.modules if m in watched)}))
"""


def _import_first(module: str) -> dict[str, list[str] | list[list[str]]]:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, module, *_MODULES],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("first", _MODULES)
def test_no_module_imports_one_that_is_still_initializing(first: str) -> None:
    assert _import_first(first)["cycles"] == []


def test_the_state_directory_module_never_loads_the_config_package() -> None:
    assert _import_first("pivot.state_dir")["loaded"] == ["pivot.config_files", "pivot.state_dir"]
