# pyright: reportImplicitStringConcatenation=false

"""The covered thread rule through the CLI: what a covered module's import starts belongs to it.

Each test drives the installed ``pivot`` CLI in a fresh project.

D1: a loaded user module whose top level starts a loader thread that imports another user module
(`plugin`) after the module's own import returns; the stage body-imports the module and joins the
loader before it returns. The loader is a covered thread, so the stage caches, with no warning,
and an edit to the module reruns it. A thread the stage itself starts after its covered import
returned is not covered: its import is counted.

D2: a covered module that joins its loader thread inside its own import, in a worker where an
upstream stage `a` already made `plugin` resident. The loader's hooked import of the resident
`plugin` is not counted, so `count` caches whether or not `a` ran first in that worker.

Straddle: the loader's import makes `plugin` resident while the covered import is open and
returns after it closed; the stage joins the loader or keeps working until it ends. It caches.

Further cases: a loader started from a covered module's loader thread (transitive); a pool
created before the covered import, whose thread is not covered (stated limit); a covered thread
of one stage still running in the next stage; the stage's own unhooked load of a module a covered
import then imports; and the AD2 hand-off shapes (`threading.Timer`, an asyncio loop on a thread,
`concurrent.futures.wait`, a daemon thread that outlives the stage).
"""

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

_PIVOT = pathlib.Path(sys.executable).parent / "pivot"
_LINES = "Local Set Default Proof Using\nSet Default Proof Using\nLemma a\n"

# helpers/lib.py starts a loader named `plugin-loader*` at import time. The loader waits until the
# module has finished importing, then imports `plugin`. Two spellings: a thread with `__import__`,
# and a ThreadPoolExecutor task with `importlib.import_module`.
_LOADER_LIBS = {
    "thread-dunder-import": (
        "import threading\nimport time\n\n"
        "_ready = threading.Event()\n\n\n"
        "def _load_plugins():\n"
        "    _ready.wait(10)\n"
        "    time.sleep(0.05)\n"
        "    __import__('plugin')\n\n\n"
        "threading.Thread(target=_load_plugins, name='plugin-loader', daemon=True).start()\n\n"
        "VALUE = 1\n"
        "_ready.set()\n"
    ),
    "pool-import-module": (
        "import concurrent.futures\nimport importlib\nimport threading\nimport time\n\n"
        "_ready = threading.Event()\n\n\n"
        "def _load_plugins():\n"
        "    _ready.wait(10)\n"
        "    time.sleep(0.05)\n"
        "    importlib.import_module('plugin')\n\n\n"
        "_pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix='plugin-loader')\n"
        "_pool.submit(_load_plugins)\n"
        "_pool.shutdown(wait=False)\n\n"
        "VALUE = 1\n"
        "_ready.set()\n"
    ),
}

_BODY_IMPORT_PIPELINE = """\
import threading

import helpers.lib
from pivot import Pipeline, stage


@stage
def count(text: str) -> dict:
    import helpers.lib

    for thread in threading.enumerate():
        if thread.name.startswith("plugin-loader"):
            thread.join(10)
    return {"v": helpers.lib.VALUE + len(text.splitlines())}


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=str))
"""

# Control (round 13's AC1 item 2, which a fix must keep): the stage itself starts the thread after
# its covered import of helpers.lib returned. That import is counted, at the head as before.
_STAGE_THREAD_PIPELINE = """\
import threading

import helpers.lib
from pivot import Pipeline, stage


@stage
def count(text: str) -> dict:
    import helpers.lib

    worker = threading.Thread(target=lambda: __import__("plugin"))
    worker.start()
    worker.join()
    return {"v": helpers.lib.VALUE + len(text.splitlines())}


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=str))
"""


def _run(project: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PIVOT_UNSAFE_FINGERPRINTING")
    }
    return subprocess.run(
        [str(_PIVOT), *args], cwd=project, env=env, capture_output=True, text=True, timeout=300
    )


def _ok(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, result.stdout + result.stderr
    return " ".join((result.stdout + result.stderr).split())


def _project(tmp_path: pathlib.Path, lib: str, pipeline: str) -> pathlib.Path:
    project = tmp_path / "project"
    files = {
        "helpers/__init__.py": "",
        "helpers/lib.py": lib,
        "plugin.py": "FACTOR = 1\n",
        "pipeline.py": pipeline,
        "data/raw/lines.txt": _LINES,
    }
    for rel, text in files.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    _ok(_run(project, "init"))
    return project


def _value(project: pathlib.Path) -> object:
    return yaml.safe_load((project / "data" / "t" / "count.yaml").read_text())["v"]


@pytest.mark.parametrize("shape", sorted(_LOADER_LIBS))
def test_d1_body_import_of_module_whose_loader_thread_imports_after_it_returns_is_reused(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """Criterion (a): main 19fd78cb runs this stage once and caches it on the second `repro`, with no
    warning. At 2f820946 every run warns `does not cover: plugin` and the second `repro` reruns it
    (`Previous run imported modules its fingerprint does not cover: plugin`)."""
    project = _project(tmp_path, _LOADER_LIBS[shape], _BODY_IMPORT_PIPELINE)

    first = _ok(_run(project, "repro"))
    again = _run(project, "repro", "--json")
    _ok(again)

    assert "does not cover" not in first, first
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout


@pytest.mark.parametrize("shape", sorted(_LOADER_LIBS))
def test_d1_loader_thread_stage_reruns_on_a_lib_edit(tmp_path: pathlib.Path, shape: str) -> None:
    """Contract item 2 for the same layout: once it caches, an edit to helpers/lib.py reruns it
    (v 103) with no warning. Main 19fd78cb fails this one (it serves v 4); a fix must keep both."""
    project = _project(tmp_path, _LOADER_LIBS[shape], _BODY_IMPORT_PIPELINE)
    _ok(_run(project, "repro"))
    _ok(_run(project, "repro"))
    (project / "helpers" / "lib.py").write_text(
        _LOADER_LIBS[shape].replace("VALUE = 1\n", "VALUE = 100\n")
    )

    edited = _ok(_run(project, "repro"))

    assert "does not cover" not in edited, edited
    assert _value(project) == 103


def test_d1_control_a_thread_the_stage_starts_after_the_covered_import_still_warns(
    tmp_path: pathlib.Path,
) -> None:
    """Control (passes at 2f820946; AC1 item 2): the stage, not the covered module, starts the thread
    after its covered import returned. The import is counted and the stage reruns."""
    project = _project(tmp_path, "VALUE = 1\n", _STAGE_THREAD_PIPELINE)

    first = _ok(_run(project, "repro"))
    again = _run(project, "repro", "--json")
    _ok(again)

    assert "does not cover: plugin" in first, first
    assert '"stage": "t/count", "status": "ran"' in again.stdout, again.stdout


# D2: round 13's AC1 layouts (`test_fingerprint_gaps.py::_IMPORT_THREAD_LIBS`), where the module
# joins the thread inside its own import, so the thread's import lands inside the covered window.
_JOINED_THREAD_LIBS = {
    "thread-dunder-import": (
        "import threading\n\n"
        "_t = threading.Thread(target=lambda: __import__('plugin'))\n_t.start()\n_t.join()\n\n"
        "VALUE = 1\n"
    ),
    "pool-import-module": (
        "import concurrent.futures\nimport importlib\n\n"
        "with concurrent.futures.ThreadPoolExecutor(2) as _pool:\n"
        "    _plugins = list(_pool.map(importlib.import_module, ['plugin']))\n\n"
        "VALUE = 1\n"
    ),
}

# Stage `a` body-imports `plugin` (covered for `a`) and returns a value that does not depend on its
# text, so a data refresh reruns `a` with the same output. `count` consumes `a` and body-imports
# helpers.lib. With `core.max_workers 1`, `a` runs first in the worker `count` then uses.
_WARM_PIPELINE = """\
import helpers.lib
import plugin
from pivot import Pipeline, stage


@stage
def a(text: str) -> dict:
{a_body}


@stage
def count(a: dict) -> dict:
    import helpers.lib
    return {{"v": helpers.lib.VALUE + a["v"]}}


with Pipeline("t") as pipeline:
    count(a(pipeline.input("lines.txt", t=str)))
"""
_A_IMPORTS_PLUGIN = '    import plugin\n    return {"v": plugin.FACTOR + 0 * len(text)}'
_A_DOES_NOT_IMPORT_PLUGIN = '    return {"v": 1 + 0 * len(text)}'


def _warm_project(tmp_path: pathlib.Path, shape: str, a_body: str) -> pathlib.Path:
    project = _project(tmp_path, _JOINED_THREAD_LIBS[shape], _WARM_PIPELINE.format(a_body=a_body))
    _ok(_run(project, "config", "set", "core.max_workers", "1"))
    return project


def _refresh_input(project: pathlib.Path, index: int) -> None:
    path = project / "data" / "raw" / "lines.txt"
    path.write_text(path.read_text() + f"Lemma extra{index}\n")


@pytest.mark.parametrize("shape", sorted(_JOINED_THREAD_LIBS))
def test_d2_covered_import_whose_joined_thread_imports_a_module_an_upstream_stage_loaded_caches(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """Criterion (a), with (b) and (c): main 19fd78cb runs both stages once with no warning, then
    caches `count` on every data refresh. At 2f820946 every run warns `does not cover: plugin`
    for `t/count`, its lock is marked, and `count` reruns on every refresh that reruns `a`."""
    project = _warm_project(tmp_path, shape, _A_IMPORTS_PLUGIN)

    first = _ok(_run(project, "repro"))
    _refresh_input(project, 1)
    refreshed = _run(project, "repro", "--json")
    _ok(refreshed)

    assert "does not cover" not in first, first
    assert '"stage": "t/a", "status": "ran"' in refreshed.stdout, refreshed.stdout
    assert '"stage": "t/count", "status": "cached"' in refreshed.stdout, refreshed.stdout


@pytest.mark.parametrize("shape", sorted(_JOINED_THREAD_LIBS))
def test_d2_control_without_the_upstream_import_the_same_layout_caches(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """Control (passes at 2f820946 and at main): `a` does not import `plugin`, so `plugin` is not
    resident when `count`'s covered import starts the thread, and `count` caches."""
    project = _warm_project(tmp_path, shape, _A_DOES_NOT_IMPORT_PLUGIN)

    first = _ok(_run(project, "repro"))
    _refresh_input(project, 1)
    refreshed = _run(project, "repro", "--json")
    _ok(refreshed)

    assert "does not cover" not in first, first
    assert '"stage": "t/count", "status": "cached"' in refreshed.stdout, refreshed.stdout


# The AD2 hand-off shapes: how a covered module's import can hand work to a thread. Each value is
# the project files over `_project`'s defaults; every stage body-imports what it reads, so no
# thread or pool handle reaches a pickled module.
_JOIN_LOADERS = (
    "    for thread in threading.enumerate():\n"
    '        if thread.name.startswith("plugin-loader"):\n'
    "            thread.join(10)\n"
)
_KEEP_WORKING = "    time.sleep(1.5)\n"
_HANDOFF_PIPELINE = """\
import threading
import time

import helpers.lib
from pivot import Pipeline, stage


@stage
def count(text: str) -> dict:
    import helpers.lib

{wait}    return {{"v": helpers.lib.VALUE + len(text.splitlines())}}


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=str))
"""
# `plugin` is registered at once and returns 0.5 s later, after the covered import has returned.
_SLOW_PLUGIN = "import time\n\ntime.sleep(0.5)\n\nFACTOR = 1\n"
_WAIT_RESIDENT = "while 'plugin' not in sys.modules:\n    time.sleep(0.001)\n\n"
_STRADDLE_LIBS = {
    "dunder": (
        "import sys\nimport threading\nimport time\n\n"
        "_t = threading.Thread(target=lambda: __import__('plugin'), name='plugin-loader')\n"
        "_t.start()\ndel _t\n" + _WAIT_RESIDENT + "VALUE = 1\n"
    ),
    "pool-import-module": (
        "import concurrent.futures\nimport importlib\nimport sys\nimport time\n\n"
        "_pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix='plugin-loader')\n"
        "_pool.submit(importlib.import_module, 'plugin')\n"
        "_pool.shutdown(wait=False)\ndel _pool\n" + _WAIT_RESIDENT + "VALUE = 1\n"
    ),
}
_TIMER_LIB = (
    "import threading\n\n"
    "_timer = threading.Timer(0.05, __import__, args=('plugin',))\n"
    "_timer.name = 'plugin-loader'\n_timer.start()\ndel _timer\n\n"
    "VALUE = 1\n"
)
_NESTED_THREAD_LIB = (
    "import threading\nimport time\n\n"
    "_ready = threading.Event()\n\n\n"
    "def _load_plugins():\n"
    "    _ready.wait(10)\n"
    "    time.sleep(0.05)\n"
    "    __import__('plugin')\n\n\n"
    "def _start_loader():\n"
    "    inner = threading.Thread(target=_load_plugins)\n"
    "    inner.start()\n"
    "    inner.join()\n\n\n"
    "threading.Thread(target=_start_loader, name='plugin-loader', daemon=True).start()\n\n"
    "VALUE = 1\n"
    "_ready.set()\n"
)
_ASYNCIO_LIB = (
    "import asyncio\nimport importlib\nimport threading\n\n\n"
    "async def _load_plugins():\n"
    "    await asyncio.sleep(0.05)\n"
    "    loop = asyncio.get_running_loop()\n"
    "    await loop.run_in_executor(None, importlib.import_module, 'plugin')\n\n\n"
    "threading.Thread(target=lambda: asyncio.run(_load_plugins()), name='plugin-loader').start()\n\n"
    "VALUE = 1\n"
)
_FUTURES_WAIT_LIB = (
    "import concurrent.futures\nimport importlib\n\n"
    "_pool = concurrent.futures.ThreadPoolExecutor(2)\n"
    "concurrent.futures.wait([_pool.submit(importlib.import_module, 'plugin')])\n"
    "_pool.shutdown()\ndel _pool\n\n"
    "VALUE = 1\n"
)
_DAEMON_LIB = (
    "import threading\nimport time\n\n\n"
    "def _load_plugins():\n"
    "    time.sleep(2)\n"
    "    __import__('plugin')\n\n\n"
    "threading.Thread(target=_load_plugins, daemon=True).start()\n\n"
    "VALUE = 1\n"
)
# Residual (i): `scaled` first imports `helpers.pool`, whose pool thread then exists in the worker
# before `count`'s covered import of `helpers.lib` hands it the import of `plugin`.
_POOL_BEFORE_FILES = {
    "helpers/pool.py": (
        "import concurrent.futures\n\n"
        "SIZE = 1\n"
        "POOL = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix='plugin-loader')\n"
        "POOL.submit(int).result()\n"
    ),
    "helpers/lib.py": (
        "import importlib\n\n"
        "from helpers.pool import POOL\n\n"
        "POOL.submit(importlib.import_module, 'plugin').result()\n\n"
        "VALUE = 1\n"
    ),
    "pipeline.py": """\
import helpers.lib
import helpers.pool
from pivot import Pipeline, stage


@stage
def scaled(text: str) -> str:
    import helpers.pool

    return text * helpers.pool.SIZE


@stage
def count(scaled: str) -> dict:
    import helpers.lib

    return {"v": helpers.lib.VALUE + len(scaled.splitlines())}


with Pipeline("t") as pipeline:
    count(scaled(pipeline.input("lines.txt", t=str)))
""",
}
# Limit (iv): `scaled`'s covered import of `helpers.loader` starts a daemon thread that imports
# `late` once `count`, the next stage in the same worker, opens the gate.
_NEXT_STAGE_FILES = {
    "helpers/loader.py": (
        "import pathlib\nimport threading\nimport time\n\n"
        "SIZE = 1\n\n\n"
        "def _load_late():\n"
        "    gate = pathlib.Path('go')\n"
        "    while not gate.exists():\n"
        "        time.sleep(0.01)\n"
        "    __import__('late')\n\n\n"
        "threading.Thread(target=_load_late, daemon=True).start()\n"
    ),
    "late.py": "VALUE = 0\n",
    "pipeline.py": """\
import pathlib
import time

import helpers.loader
from pivot import Pipeline, stage


@stage
def scaled(text: str) -> str:
    import helpers.loader

    return text * helpers.loader.SIZE


@stage
def count(scaled: str) -> dict:
    pathlib.Path("go").write_text("open")
    time.sleep(0.5)
    return {"v": len(scaled.splitlines())}


with Pipeline("t") as pipeline:
    count(scaled(pipeline.input("lines.txt", t=str)))
""",
}


def _handoff(lib: str, wait: str, plugin: str = "FACTOR = 1\n") -> dict[str, str]:
    return {
        "helpers/lib.py": lib,
        "plugin.py": plugin,
        "pipeline.py": _HANDOFF_PIPELINE.format(wait=wait),
    }


_HANDOFFS = {
    "thread-joined-in-import": _handoff(_JOINED_THREAD_LIBS["thread-dunder-import"], ""),
    "thread-joined-by-stage": _handoff(_LOADER_LIBS["thread-dunder-import"], _JOIN_LOADERS),
    "straddle-dunder-join": _handoff(_STRADDLE_LIBS["dunder"], _JOIN_LOADERS, _SLOW_PLUGIN),
    "straddle-dunder-work": _handoff(_STRADDLE_LIBS["dunder"], _KEEP_WORKING, _SLOW_PLUGIN),
    "timer": _handoff(_TIMER_LIB, _JOIN_LOADERS),
    "pool-in-import": _handoff(_LOADER_LIBS["pool-import-module"], _JOIN_LOADERS),
    "straddle-pool-import-module-join": _handoff(
        _STRADDLE_LIBS["pool-import-module"], _JOIN_LOADERS, _SLOW_PLUGIN
    ),
    "straddle-pool-import-module-work": _handoff(
        _STRADDLE_LIBS["pool-import-module"], _KEEP_WORKING, _SLOW_PLUGIN
    ),
    "pool-before-import": _POOL_BEFORE_FILES,
    "nested-thread": _handoff(_NESTED_THREAD_LIB, _JOIN_LOADERS),
    "asyncio-loop-thread": _handoff(_ASYNCIO_LIB, _JOIN_LOADERS),
    "futures-wait": _handoff(_FUTURES_WAIT_LIB, ""),
    "daemon-outlives-stage": _handoff(_DAEMON_LIB, ""),
    "covered-thread-in-next-stage": _NEXT_STAGE_FILES,
}
# Under the covered thread rule every shape caches except these two, which are counted.
_COUNTED_HANDOFFS = ("pool-before-import", "covered-thread-in-next-stage")
_COVERED_HANDOFFS = sorted(set(_HANDOFFS) - set(_COUNTED_HANDOFFS))


def _handoff_project(tmp_path: pathlib.Path, shape: str) -> pathlib.Path:
    files = _HANDOFFS[shape]
    project = _project(tmp_path, files.get("helpers/lib.py", ""), files["pipeline.py"])
    for rel, text in files.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text)
    _ok(_run(project, "config", "set", "core.max_workers", "1"))
    return project


@pytest.mark.parametrize("shape", _COVERED_HANDOFFS)
def test_thread_a_covered_import_hands_work_to_is_covered(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """Each way a covered module's import hands an import of `plugin` to a thread it starts (a
    thread whose import makes `plugin` resident inside the covered import and returns after it,
    joined by the stage or ending while the stage works; a thread started by such a thread; a
    `threading.Timer`; an asyncio loop on such a thread; futures it waits on; a daemon thread
    that outlives the stage): the stage warns nothing, caches, and reruns on a lib edit."""
    project = _handoff_project(tmp_path, shape)

    first = _ok(_run(project, "repro"))
    again = _run(project, "repro", "--json")
    _ok(again)
    lib = project / "helpers" / "lib.py"
    lib.write_text(lib.read_text().replace("VALUE = 1\n", "VALUE = 100\n"))
    edited = _ok(_run(project, "repro"))

    assert "does not cover" not in first, first
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    assert "does not cover" not in edited, edited
    assert _value(project) == 103


def test_import_handed_to_a_thread_that_existed_before_the_covered_import_is_counted(
    tmp_path: pathlib.Path,
) -> None:
    """Stated limit (i): a thread that existed before the covered import (a pool an earlier
    stage's import started in the same worker) is not a covered thread, so the import a covered
    module hands it is counted; the warning names the module and the fix."""
    project = _handoff_project(tmp_path, "pool-before-import")

    first = _ok(_run(project, "repro"))

    assert (
        "Stage 't/count' imported user modules its fingerprint does not cover: plugin." in first
    ), first
    assert "Fix: import them statically" in first, first
    assert "Stage 't/scaled' imported" not in first, first


def test_covered_thread_of_one_stage_is_not_covered_in_the_next(tmp_path: pathlib.Path) -> None:
    """A thread `scaled`'s covered import started is covered only for `scaled`: its import while
    `count` runs next in the same worker is counted for `count`."""
    project = _handoff_project(tmp_path, "covered-thread-in-next-stage")

    first = _ok(_run(project, "repro"))

    assert "Stage 't/count' imported user modules its fingerprint does not cover: late." in first, (
        first
    )
    assert "Stage 't/scaled' imported" not in first, first


_OWN_LOAD_PIPELINE = """\
import pathlib

import helpers.lib
from pivot import Pipeline, stage

ROOT = str(pathlib.Path(__file__).resolve().parent)


@stage
def count(text: str) -> dict:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("late", ROOT + "/late.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["late"] = module
    spec.loader.exec_module(module)

    import helpers.lib

    return {"v": helpers.lib.VALUE + sys.modules["late"].VALUE + len(text.splitlines())}


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=str))
"""


def test_module_the_stage_loads_by_hand_stays_counted_when_a_covered_import_then_imports_it(
    tmp_path: pathlib.Path,
) -> None:
    """The stage registers `late` with `exec_module` while no covered thread runs (the manual
    loader case), then first-imports the covered `helpers.lib`, whose top level imports `late`.
    The load is counted: the stage warns and reruns, and an edit to `late` is seen."""
    project = _project(tmp_path, "import late\n\nVALUE = 1\n", _OWN_LOAD_PIPELINE)
    (project / "late.py").write_text("VALUE = 0\n")

    first = _ok(_run(project, "repro"))
    again = _run(project, "repro", "--json")
    _ok(again)
    (project / "late.py").write_text("VALUE = 100\n")
    _ok(_run(project, "repro"))

    assert "does not cover: late." in first, first
    assert '"stage": "t/count", "status": "ran"' in again.stdout, again.stdout
    assert _value(project) == 104
