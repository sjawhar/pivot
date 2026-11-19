"""The worker's import hooks never run inside themselves.

On Debian's CPython 3.13.5 (`/usr/bin/python3.13`, `libpython3.13-stdlib` 3.13.5-2+deb13u5),
`pathlib/_local.py:117` imports `ntpath` in every `PurePath.__init__`; uv's CPython 3.13.17 imports
it once, at the top of the module. The hooks' own recording builds paths
(`fingerprint.is_user_code`), so there each hooked import's recording imports again, and pivot
0.10.4's worker stops at start with `RecursionError`. Each test makes the running Python's
`PurePath.__init__` import `ntpath` first, as Debian's does, in a child process: the hooks replace
`builtins.__import__` and the change replaces a `pathlib` method, both process-wide, and a fresh
process starts with no module judged yet, as a worker does.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest
import yaml

from tests import helpers

# Run first in each child, with NESTED_IMPORT replaced by the nested import's spelling: every
# `PurePath.__init__` imports `ntpath` first, as Debian's does, and `loaded_source.record_modules`
# keeps the names it receives and counts the paths built while it runs.
_CHILD_PRELUDE = """\
import importlib
import json
import pathlib

from pivot import loaded_source

_pure_path_init = pathlib.PurePath.__init__
recording = False
paths_built_while_recording = 0


def _init_as_debian(self, *args):
    global paths_built_while_recording
    paths_built_while_recording += recording
    NESTED_IMPORT
    _pure_path_init(self, *args)


pathlib.PurePath.__init__ = _init_as_debian
received = []
_record_modules = loaded_source.record_modules


def _record_and_keep(names):
    global recording
    names = list(names)
    received.append(names)
    recording = True
    try:
        _record_modules(names)
    finally:
        recording = False


loaded_source.record_modules = _record_and_keep
"""

# Debian's own spelling, and the other route the hooks replace.
_NESTED_IMPORTS = [
    pytest.param("import ntpath", id="import-statement"),
    pytest.param("importlib.import_module('ntpath')", id="import-module"),
]

# The user modules a child imports as a stage would: one by an `import` statement, one through
# `importlib.import_module`.
_STAGE_MODULES = ("hookwork_statement", "hookwork_dynamic")

_STAGE_IMPORTS = """\
from pivot.executor import worker

worker.install_import_hooks()
import hookwork_statement
importlib.import_module("hookwork_dynamic")
print(json.dumps({"received": received, "paths": paths_built_while_recording}))
"""

# As BET-1194's reproduction starts a worker (`core._init_worker`, loky's initializer), then
# imports as the stage would.
_WORKER_START = """\
import os

from pivot.executor import core

core._init_worker(os.getppid(), None, dict())
print("init ok")
received.clear()
paths_built_while_recording = 0
import hookwork_statement
importlib.import_module("hookwork_dynamic")
print(json.dumps({"received": received, "paths": paths_built_while_recording}))
"""

# A stage is running, as `execute_stage` runs one (`worker._track_stage_imports`, which installs
# the hooks for the call); `imported` is what the stage-import check recorded.
_STAGE_RUNNING = """\
from pivot.executor import worker

with worker._track_stage_imports(frozenset()) as stage:
    import hookwork_statement
    importlib.import_module("hookwork_dynamic")
print(json.dumps({
    "received": received,
    "imported": sorted(stage.imported),
    "paths": paths_built_while_recording,
}))
"""

# The main thread holds inside its recording of `hookwork_statement` until the other thread's
# import of `hookwork_dynamic` has returned.
_TWO_THREADS = """\
import threading

from pivot.executor import worker

a_recording = threading.Event()
b_imported = threading.Event()
b_imported_while_a_recorded = False
_keep = loaded_source.record_modules


def _hold_while_recording_a(names):
    global b_imported_while_a_recorded
    _keep(names)
    if names == ["hookwork_statement"]:
        a_recording.set()
        b_imported_while_a_recorded = b_imported.wait(30)


loaded_source.record_modules = _hold_while_recording_a


def _import_b():
    a_recording.wait(30)
    importlib.import_module("hookwork_dynamic")
    b_imported.set()


worker.install_import_hooks()
b = threading.Thread(target=_import_b)
b.start()
import hookwork_statement
b.join(30)
print(json.dumps({"received": received, "overlapped": b_imported_while_a_recorded}))
"""

# A stage is running and reloads `hookwork_statement`, which the child imported before any hook
# was installed, so nothing has judged it yet: the reload hook's recording judges it, and so
# builds paths, each counted while the reload runs.
_STAGE_RELOADS = """\
import hookwork_statement

from pivot.executor import worker

_init_with_nested_import = pathlib.PurePath.__init__
paths_built_while_reloading = 0
reloading = False


def _count_while_reloading(self, *args):
    global paths_built_while_reloading
    paths_built_while_reloading += reloading
    _init_with_nested_import(self, *args)


pathlib.PurePath.__init__ = _count_while_reloading
with worker._track_stage_imports(frozenset()) as stage:
    reloading = True
    importlib.reload(hookwork_statement)
    reloading = False
print(json.dumps({
    "received": received,
    "imported": sorted(stage.imported),
    "paths": paths_built_while_reloading,
}))
"""


def _helper_run_child(
    tmp_path: pathlib.Path, script: str, nested: str
) -> subprocess.CompletedProcess[str]:
    """Run `script` after `_CHILD_PRELUDE` in a new interpreter in `tmp_path`, which holds the
    user modules `_STAGE_MODULES`, none of them imported yet."""
    for name in _STAGE_MODULES:
        (tmp_path / f"{name}.py").write_text("VALUE = 1\n")
    source = (_CHILD_PRELUDE + script).replace("NESTED_IMPORT", nested)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _helper_watched(received: list[list[str]]) -> list[list[str]]:
    """The lists of names `record_modules` received that name `ntpath` or a stage module. The
    import system's own imports reach the hooks too (CPython's `io.open_code`, which reads each
    source file, imports `_io` through `builtins.__import__`), and the hooks record those."""
    return [names for names in received if names[0] in {"ntpath", *_STAGE_MODULES}]


@pytest.mark.parametrize("nested", _NESTED_IMPORTS)
def test_an_import_the_hooks_recording_makes_passes_through_and_the_stage_import_is_recorded(
    tmp_path: pathlib.Path, nested: str
) -> None:
    """With the hooks installed, the stage's own imports return, with no `RecursionError`, and
    `record_modules` receives their names and never `ntpath`, which the recording imports."""
    done = _helper_run_child(tmp_path, _STAGE_IMPORTS, nested)

    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["paths"] > 0, "the recording built no path, so it imported nothing"
    assert _helper_watched(result["received"]) == [["hookwork_statement"], ["hookwork_dynamic"]]


def test_a_worker_starts_and_records_the_stage_imports_where_pathlib_imports_ntpath_per_path(
    tmp_path: pathlib.Path,
) -> None:
    """`core._init_worker`, which judges every module the worker holds (and so builds paths
    under the hooks it has just installed), returns, and the worker then records the stage's own
    imports and no import of `ntpath`."""
    done = _helper_run_child(tmp_path, _WORKER_START, "import ntpath")

    assert done.returncode == 0, done.stderr
    started, imported = done.stdout.splitlines()
    assert started == "init ok"
    result = json.loads(imported)
    assert result["paths"] > 0, "the recording built no path, so it imported nothing"
    assert _helper_watched(result["received"]) == [["hookwork_statement"], ["hookwork_dynamic"]]


def test_a_running_stage_records_its_imports_where_pathlib_imports_ntpath_per_path(
    tmp_path: pathlib.Path,
) -> None:
    """While a stage runs, its imports return, the stage-import check records both, and
    `record_modules` receives their names and never `ntpath`. Nothing in the fresh process has
    judged `ntpath` yet, as at a worker's start."""
    done = _helper_run_child(tmp_path, _STAGE_RUNNING, "import ntpath")

    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["paths"] > 0, "the recording built no path, so it imported nothing"
    assert result["imported"] == ["hookwork_dynamic", "hookwork_statement"]
    assert _helper_watched(result["received"]) == [["hookwork_statement"], ["hookwork_dynamic"]]


def test_one_threads_recording_does_not_swallow_another_threads_import(
    tmp_path: pathlib.Path,
) -> None:
    """While the main thread is inside its recording of one import, another thread's import is
    recorded: the pass-through holds for the thread that is recording only."""
    done = _helper_run_child(tmp_path, _TWO_THREADS, "import ntpath")

    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["overlapped"], "the other thread's import did not run during the recording"
    assert _helper_watched(result["received"]) == [["hookwork_statement"], ["hookwork_dynamic"]]


def test_a_reload_while_a_stage_runs_is_recorded_and_the_import_its_recording_makes_passes_through(
    tmp_path: pathlib.Path,
) -> None:
    """While a stage runs, `importlib.reload` of a user module nothing has judged yet returns,
    the stage-import check records the module, and the import of `ntpath` that judging it makes
    passes through: `record_modules` never receives `ntpath`, as for the other hooks' recording.
    Were the reload hook's recording not the hooks' own work, that import would be recorded as
    one the stage made."""
    done = _helper_run_child(tmp_path, _STAGE_RELOADS, "import ntpath")

    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["paths"] > 0, "the reload's recording built no path, so it imported nothing"
    assert result["imported"] == ["hookwork_statement"]
    assert _helper_watched(result["received"]) == []


# Installed as `sitecustomize` for every process of the run, the coordinator and each loky worker.
_DEBIAN_PATHLIB = """\
import pathlib

_pure_path_init = pathlib.PurePath.__init__


def _init_as_debian(self, *args):
    import ntpath

    _pure_path_init(self, *args)


pathlib.PurePath.__init__ = _init_as_debian
"""

# The stage imports by a runtime name through each hook: `importlib.import_module`, by an alias
# (the walk refuses `x.import_module(...)` at definition), and `builtins.__import__`. Neither
# module is loaded at discovery, so the worker's check counts both.
_RUNTIME_IMPORT_PIPELINE = """\
import importlib
import pathlib

from pivot import Pipeline, stage

load = importlib.import_module


@stage
def count(text: str) -> dict:
    plugin = load("plugin")
    extra = __import__("extra")

    return {
        "v": plugin.FACTOR * extra.FACTOR * len(text.splitlines()),
        "pathlib_init": pathlib.PurePath.__init__.__module__,
    }


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=str))
"""


def test_a_pipeline_runs_in_workers_whose_pathlib_imports_ntpath_per_path(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot repro` with Debian's `pathlib` in every process: the worker starts, the stage
    runs, and its run-time imports of `plugin` and `extra` are recorded (the stage warns that its
    fingerprint does not cover them)."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    site = tmp_path / "site"
    files = {
        project / "plugin.py": "FACTOR = 2\n",
        project / "extra.py": "FACTOR = 5\n",
        project / "pipeline.py": _RUNTIME_IMPORT_PIPELINE,
        project / "data" / "raw" / "lines.txt": "a\nb\nc\n",
        site / "sitecustomize.py": _DEBIAN_PATHLIB,
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    init = helpers.run_pivot(project, home, "init")
    assert init.returncode == 0, init.stdout + init.stderr
    # `helpers.cli_env` passes this process's PYTHONPATH on; the run keeps it, after `site`.
    python_path = os.pathsep.join(filter(None, [str(site), os.environ.get("PYTHONPATH")]))

    run = helpers.run_pivot(project, home, "repro", extra_env={"PYTHONPATH": python_path})

    assert run.returncode == 0, run.stdout + run.stderr
    output = " ".join((run.stdout + run.stderr).split())
    assert (
        "Stage 't/count' imported user modules its fingerprint does not cover: extra, plugin."
        in output
    ), output
    written = yaml.safe_load((project / "data" / "t" / "count.yaml").read_text())
    assert written == {"v": 30, "pathlib_init": "sitecustomize"}
