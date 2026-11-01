# pyright: reportImplicitStringConcatenation=false

"""CLI regressions for fingerprint gaps found during PR #496 acceptance testing."""

import json
import pathlib
import py_compile
import shutil
import subprocess
import sys

import pytest
import yaml
from tests import helpers

from pivot import run_history
from pivot.storage import state

_LINES = "Local Set Default Proof Using\nSet Default Proof Using\nLemma a\n"
_HEADER = """\
import functools
import importlib
import os

from pivot import Pipeline, stage
"""
_NORMALIZE = """

@stage
def normalize(text: str) -> str:
    return "\\n".join(line.strip() for line in text.splitlines() if line.strip())
"""
_FOOTER = """

with Pipeline("t") as pipeline:
    count(normalize(pipeline.input("lines.txt", t=str)))
"""
_LIB_VALUE = {"helpers/__init__.py": "", "helpers/lib.py": "VALUE = 1\n"}
_EDIT_LIB_VALUE = ("helpers/lib.py", "VALUE = 1\n", "VALUE = 100\n")


def _pipeline(imports: str, body: str) -> str:
    return _HEADER + imports + _NORMALIZE + "\n" + body + _FOOTER


def _project(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    home = tmp_path / "home"
    project = tmp_path / "project"
    for rel, text in files.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text)
    (project / "data" / "raw").mkdir(parents=True)
    (project / "data" / "raw" / "lines.txt").write_text(_LINES)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    init = helpers.run_pivot(project, home, "init")
    assert init.returncode == 0, init.stdout + init.stderr
    return project


def _edit(project: pathlib.Path, rel: str, old: str, new: str) -> None:
    path = project / rel
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def _run_cache_entries(project: pathlib.Path) -> list[run_history.RunCacheEntry]:
    with state.StateDB(project / ".pivot", readonly=True) as db:
        entries = list(db.iter_prefix(b"runcache:t/count:"))
    return [run_history.deserialize_run_cache_entry(value) for _key, value in entries]


def _assert_not_reusable(
    project: pathlib.Path, home: pathlib.Path, result: subprocess.CompletedProcess[str]
) -> None:
    """Check a successful run's warning, empty run cache, and reported uncovered-import reason."""
    message = " ".join((result.stdout + result.stderr).split())
    assert result.returncode == 0, message
    assert message.count("imported user modules its fingerprint does not cover") == 1, message
    assert _run_cache_entries(project) == []
    status = helpers.run_pivot(project, home, "status", "--json")
    assert status.returncode == 0, status.stdout + status.stderr
    assert "Previous run imported modules its fingerprint does not cover" in status.stdout


def _count(project: pathlib.Path) -> object:
    return yaml.safe_load((project / "data" / "t" / "count.yaml").read_text())["v"]


def _repro_statuses(
    project: pathlib.Path, home: pathlib.Path, seed: str | None = None
) -> dict[str, str]:
    """Run `pivot repro --json` (with `PYTHONHASHSEED=seed` when given) and return each stage's
    completion status."""
    extra_env = None if seed is None else {"PYTHONHASHSEED": seed}
    result = helpers.run_pivot(project, home, "repro", "--json", extra_env=extra_env)
    assert result.returncode == 0, result.stdout + result.stderr
    statuses = dict[str, str]()
    for line in result.stdout.splitlines():
        if line.startswith("{") and (event := json.loads(line)).get("type") == "stage_complete":
            statuses[event["stage"]] = event["status"]
    return statuses


_DYNAMIC_IMPORT_SHAPES = {
    "local-alias-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n    load = importlib.import_module\n"
                '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "global-alias-dunder-import": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "\n_imp = __import__\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": _imp("helpers.lib", fromlist=["VALUE"]).VALUE'
                " + len(text.splitlines())}\n",
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "default-arg-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "\n\ndef _load(name, loader=importlib.import_module):\n    return loader(name)\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": _load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "partial-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "\nload = functools.partial(importlib.import_module)\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "literal-getattr-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                '    load = getattr(importlib, "import_module")\n'
                '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "local-alias-getattr-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                "    lookup = getattr\n"
                '    load = lookup(importlib, "import_module")\n'
                '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "chained-alias-getattr-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                "    first = getattr\n"
                "    lookup = first\n"
                '    load = lookup(importlib, "import_module")\n'
                '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "parameter-getattr-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "\n\ndef _load(lookup):\n"
                '    return lookup(importlib, "import_module")("helpers.lib").VALUE\n',
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": _load(getattr) + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
}


_RUNTIME_LOADERS = {
    "partial-getattr": (
        "lookup = functools.partial(getattr, importlib)\n"
        '    load = lookup("import_module")\n'
        '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}'
    ),
    "module-dict": (
        'load = importlib.__dict__["import_module"]\n'
        '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}'
    ),
    "vars-dict": (
        'load = vars(importlib)["import_module"]\n'
        '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}'
    ),
    "exec-import": (
        "scope = {}\n"
        '    exec("import helpers.lib as loaded", scope)\n'
        '    return {"v": scope["loaded"].VALUE + len(text.splitlines())}'
    ),
    "manual-loader": (
        "import importlib.util\n"
        "    import pathlib\n"
        "    import sys\n"
        "    path = pathlib.Path('helpers') / 'lib.py'\n"
        "    spec = importlib.util.spec_from_file_location('manual_loaded', path)\n"
        "    assert spec is not None and spec.loader is not None\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    sys.modules[spec.name] = module\n"
        "    spec.loader.exec_module(module)\n"
        '    return {"v": module.VALUE + len(text.splitlines())}'
    ),
}


@pytest.mark.parametrize("loader", sorted(_RUNTIME_LOADERS))
def test_runtime_dynamic_import_makes_result_not_reusable(
    tmp_path: pathlib.Path, loader: str
) -> None:
    """A user module loaded only at run time is outside the fingerprint: the stage still runs,
    warns, files no run-cache entry and reruns next time, so an edit to the module is seen."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n    " + _RUNTIME_LOADERS[loader] + "\n",
            ),
        },
    )

    first = helpers.run_pivot(project, home, "repro")

    _assert_not_reusable(project, home, first)
    assert _count(project) == 4
    _edit(project, *_EDIT_LIB_VALUE)
    second = helpers.run_pivot(project, home, "repro")
    assert second.returncode == 0, second.stdout + second.stderr
    assert _count(project) == 103


_RELATIVE_IMPORT_FALLBACKS = {
    "by-name": (
        'm = __import__("lib", {"__name__": "helpers.shim"}, None, ("VALUE",), 1)\n'
        '    return {"v": m.VALUE + len(text.splitlines())}'
    ),
    "by-spec": (
        "import helpers\n"
        "    import sys\n"
        '    g = {"__name__": "x", "__spec__": sys.modules["helpers"].__spec__}\n'
        '    m = __import__("lib", g, None, ("VALUE",), 1)\n'
        '    return {"v": m.VALUE + len(text.splitlines())}'
    ),
    "exec-moduletype": (
        "import types\n"
        '    m = types.ModuleType("helpers.shim")\n'
        '    exec("from . import lib as L", vars(m))\n'
        '    return {"v": vars(m)["L"].VALUE + len(text.splitlines())}'
    ),
    "by-package-control": (
        'm = __import__("lib", {"__package__": "helpers"}, None, ("VALUE",), 1)\n'
        '    return {"v": m.VALUE + len(text.splitlines())}'
    ),
}


def _relative_import_pipeline(statement: str, *, warm: bool) -> str:
    if not warm:
        return (
            "import helpers.lib\n\n"
            "from pivot import Pipeline, stage\n\n\n"
            "@stage\n"
            "def count(text: str) -> dict:\n"
            f"    {statement}\n\n\n"
            'with Pipeline("t") as pipeline:\n'
            '    count(pipeline.input("lines.txt", t=str))\n'
        )
    return (
        "import helpers.lib\n\n"
        "from pivot import Pipeline, stage\n\n\n"
        "@stage\n"
        "def a(text: str) -> str:\n"
        "    import helpers.lib\n\n"
        "    return text\n\n\n"
        "@stage\n"
        "def count(a: str) -> dict:\n"
        f"    text = a\n    {statement}\n\n\n"
        'with Pipeline("t") as pipeline:\n'
        '    count(a(pipeline.input("lines.txt", t=str)))\n'
    )


@pytest.mark.parametrize("fallback", sorted(_RELATIVE_IMPORT_FALLBACKS))
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_relative_builtins_import_uses_cpythons_package_fallback(
    tmp_path: pathlib.Path, fallback: str, warm: bool
) -> None:
    """A relative `builtins.__import__` records the same user module CPython imports when the
    package comes from `__name__`, `__spec__.parent` or a `ModuleType` fallback, cold or after an
    upstream stage made the module resident. The explicit `__package__` spelling is the control."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            **_LIB_VALUE,
            "pipeline.py": _relative_import_pipeline(
                _RELATIVE_IMPORT_FALLBACKS[fallback], warm=warm
            ),
        },
    )
    helpers.run_pivot(project, home, "config", "set", "core.max_workers", "1")

    first = helpers.run_pivot(project, home, "repro")

    _assert_not_reusable(project, home, first)
    assert "helpers" in first.stdout + first.stderr
    assert _count(project) == 4
    _edit(project, *_EDIT_LIB_VALUE)
    second = helpers.run_pivot(project, home, "repro")
    assert second.returncode == 0, second.stdout + second.stderr
    assert _count(project) == 103


_PIVOT_THREAD_LOADS = {
    "fileno-manual-loader": (
        "import sys\n\nsys.stdout.fileno()\n\nVALUE = 1\n",
        "import importlib.util\n"
        "    import sys\n"
        "    spec = importlib.util.spec_from_file_location('late', 'late.py')\n"
        "    assert spec is not None and spec.loader is not None\n"
        "    m = importlib.util.module_from_spec(spec)\n"
        "    sys.modules['late'] = m\n"
        "    spec.loader.exec_module(m)\n"
        '    return {"v": m.VALUE + len(text.splitlines())}',
    ),
    "faulthandler-manual-loader": (
        "import faulthandler\nimport sys\n\nfaulthandler.enable(sys.stderr)\n\nVALUE = 1\n",
        "import importlib.util\n"
        "    import sys\n"
        "    spec = importlib.util.spec_from_file_location('late', 'late.py')\n"
        "    assert spec is not None and spec.loader is not None\n"
        "    m = importlib.util.module_from_spec(spec)\n"
        "    sys.modules['late'] = m\n"
        "    spec.loader.exec_module(m)\n"
        '    return {"v": m.VALUE + len(text.splitlines())}',
    ),
    "fileno-entrypoint-loader": (
        "import sys\n\nsys.stdout.fileno()\n\nVALUE = 1\n",
        "import importlib.metadata\n"
        "    ep = importlib.metadata.EntryPoint(name='late', value='late:VALUE', group='g')\n"
        '    return {"v": ep.load() + len(text.splitlines())}',
    ),
}


@pytest.mark.parametrize("surface", sorted(_PIVOT_THREAD_LOADS))
def test_pivot_thread_started_inside_a_covered_import_does_not_excuse_a_later_manual_load(
    tmp_path: pathlib.Path, surface: str
) -> None:
    """A pipe reader, parent watcher or stop-grace thread Pivot starts is not user work. Starting
    the reader through `fileno()` or `faulthandler.enable()` inside a covered import must not
    excuse a later manual or `EntryPoint.load()` load of `late`."""
    home = tmp_path / "home"
    lib, body = _PIVOT_THREAD_LOADS[surface]
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/lib.py": lib,
            "late.py": "VALUE = 1\n",
            "pipeline.py": _pipeline(
                "import helpers.lib\n",
                f"@stage\ndef count(text: str) -> dict:\n    import helpers.lib\n\n    {body}\n",
            ),
        },
    )

    first = helpers.run_pivot(project, home, "repro")

    _assert_not_reusable(project, home, first)
    assert "late" in first.stdout + first.stderr
    assert _count(project) == 4
    _edit(project, "late.py", "VALUE = 1\n", "VALUE = 100\n")
    second = helpers.run_pivot(project, home, "repro")
    assert second.returncode == 0, second.stdout + second.stderr
    assert _count(project) == 103


def test_pivot_reader_started_before_a_covered_import_does_not_excuse_a_later_manual_load(
    tmp_path: pathlib.Path,
) -> None:
    """A later manual load stays counted when Pivot starts a pipe reader before or during a covered
    import, because Pivot-owned threads do not inherit user-code coverage."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/lib.py": "import sys\n\nsys.stdout.fileno()\n\nVALUE = 1\n",
            "late.py": "VALUE = 1\n",
            "pipeline.py": _pipeline(
                "import helpers.lib\n",
                "@stage\ndef count(text: str) -> dict:\n"
                "    import importlib.util\n"
                "    import sys\n\n"
                "    sys.stdout.fileno()\n"
                "    import helpers.lib\n"
                "    spec = importlib.util.spec_from_file_location('late', 'late.py')\n"
                "    assert spec is not None and spec.loader is not None\n"
                "    m = importlib.util.module_from_spec(spec)\n"
                "    sys.modules['late'] = m\n"
                "    spec.loader.exec_module(m)\n"
                '    return {"v": m.VALUE + len(text.splitlines())}\n',
            ),
        },
    )

    _assert_not_reusable(project, home, helpers.run_pivot(project, home, "repro"))


def test_pivot_thread_started_inside_a_covered_import_does_not_hide_a_hooked_load(
    tmp_path: pathlib.Path,
) -> None:
    """A later hooked `__import__` stays counted while a reader Pivot starts in a covered import
    remains alive."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/lib.py": "import sys\n\nsys.stdout.fileno()\n\nVALUE = 1\n",
            "late.py": "VALUE = 1\n",
            "pipeline.py": _pipeline(
                "import helpers.lib\n",
                "@stage\ndef count(text: str) -> dict:\n"
                "    import helpers.lib\n\n"
                "    m = __import__('late')\n"
                '    return {"v": m.VALUE + len(text.splitlines())}\n',
            ),
        },
    )

    _assert_not_reusable(project, home, helpers.run_pivot(project, home, "repro"))


def test_runtime_dynamic_import_under_no_commit_warns_that_commit_keeps_the_mark(
    tmp_path: pathlib.Path,
) -> None:
    """The `--no-commit` run records its marked manifest, which `pivot commit` writes for the
    outputs it wrote, so under `--no-commit` the warning says `pivot commit` keeps the mark, and
    after the commit the stage still reruns."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n    "
                + _RUNTIME_LOADERS["exec-import"]
                + "\n",
            ),
        },
    )

    result = helpers.run_pivot(project, home, "repro", "--no-commit")

    message = " ".join((result.stdout + result.stderr).split())
    assert result.returncode == 0, message
    assert message.count("imported user modules its fingerprint does not cover") == 1, message
    assert "`pivot commit` keeps that when it records these outputs." in message, message
    committed = helpers.run_pivot(project, home, "commit")
    assert committed.returncode == 0, committed.stdout + committed.stderr
    lock = (project / ".pivot" / "stages" / "t" / "count.lock").read_text()
    assert "uncovered-import:" in lock
    again = helpers.run_pivot(project, home, "repro")
    assert "`pivot commit`" not in " ".join((again.stdout + again.stderr).split())
    assert "1 ran" in " ".join((again.stdout + again.stderr).split())


def test_import_from_a_thread_outliving_its_stage_does_not_fail_the_next_stage(
    tmp_path: pathlib.Path,
) -> None:
    """A thread one stage leaves running can import while a later stage runs in the same
    worker. That later stage still succeeds; it only loses reuse and reruns next time."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "late.py": "VALUE = 1\n",
            "pipeline.py": (
                "import pathlib\nimport threading\nimport time\n\n"
                "from pivot import Pipeline, stage\n\n\n"
                "@stage\ndef start(text: str) -> dict:\n"
                '    gate = pathlib.Path("go")\n\n'
                "    def delayed_import() -> None:\n"
                "        while not gate.exists():\n"
                "            time.sleep(0.01)\n"
                '        exec("import late")\n\n'
                "    threading.Thread(target=delayed_import, daemon=True).start()\n"
                '    return {"v": len(text.splitlines())}\n\n\n'
                "@stage\ndef count(start: dict) -> dict:\n"
                '    pathlib.Path("go").write_text("open")\n'
                "    time.sleep(0.5)\n"
                '    return {"v": start["v"]}\n\n\n'
                'with Pipeline("t") as pipeline:\n'
                '    count(start(pipeline.input("lines.txt", t=str)))\n'
            ),
        },
    )
    configured = helpers.run_pivot(project, home, "config", "set", "core.max_workers", "1")
    assert configured.returncode == 0, configured.stdout + configured.stderr

    first = helpers.run_pivot(project, home, "repro")

    _assert_not_reusable(project, home, first)
    assert "late" in first.stdout + first.stderr
    assert _count(project) == 3
    second = helpers.run_pivot(project, home, "repro", "t/count")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "t/count" in second.stdout
    assert "cached" not in second.stdout.split("t/count", 1)[1].splitlines()[0]


@pytest.mark.parametrize("shape", sorted(_DYNAMIC_IMPORT_SHAPES))
def test_dynamic_import_spelling_of_a_user_module_makes_result_not_reusable(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """Spellings main does not refuse at definition run; the user module they load at run time
    makes the result not reusable, and an edit to it is seen without `--force`."""
    home = tmp_path / "home"
    files, edit_spec = _DYNAMIC_IMPORT_SHAPES[shape]
    project = _project(tmp_path, files)

    first = helpers.run_pivot(project, home, "repro")

    _assert_not_reusable(project, home, first)
    assert _count(project) == 4
    _edit(project, *edit_spec)
    second = helpers.run_pivot(project, home, "repro")
    assert second.returncode == 0, second.stdout + second.stderr
    assert _count(project) == 103


_LIBRARY_DYNAMIC_IMPORTS = {
    "dunder-import": 'json = __import__("json")\n    return {"v": len(json.dumps([1]))}',
    "dunder-import-keywords": (
        'json = __import__("json", globals=None, locals=None, fromlist=(), level=0)\n'
        '    return {"v": len(json.dumps([1]))}'
    ),
    "import-module-alias": (
        'load = importlib.import_module\n    return {"v": len(load("json").dumps([1]))}'
    ),
}


@pytest.mark.parametrize("shape", sorted(_LIBRARY_DYNAMIC_IMPORTS))
def test_dynamic_import_of_a_library_runs_and_caches(tmp_path: pathlib.Path, shape: str) -> None:
    """As on main: a dynamic import of a library module runs, warns nothing, and caches."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n    "
                + _LIBRARY_DYNAMIC_IMPORTS[shape]
                + "\n",
            )
        },
    )

    first = helpers.run_pivot(project, home, "repro")
    again = helpers.run_pivot(project, home, "repro", "--json")

    assert first.returncode == 0, first.stdout + first.stderr
    assert "does not cover" not in " ".join((first.stdout + first.stderr).split())
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    assert _count(project) == 3


_UNLOADED_BODY_IMPORTS = {
    "from-import-function": (
        {"runtime_helper.py": "def value():\n    return 1\n"},
        "from runtime_helper import value",
        "runtime_helper",
    ),
    "import-module": (
        {"runtime_helper.py": "VALUE = 1\n"},
        "import runtime_helper",
        "runtime_helper",
    ),
    "package-submodule": (
        {"pkg/__init__.py": "", "pkg/sub.py": "VALUE = 1\n"},
        "import pkg.sub",
        "pkg.sub",
    ),
    "package-attribute-submodule": (
        _LIB_VALUE,
        "import helpers\n    _ = helpers.lib.VALUE",
        "helpers",
    ),
    "metaclass-hook": (
        {
            "runtime_helper.py": (
                "import pathlib\n\n\nclass Meta(type):\n"
                "    def __new__(mcs, name, bases, ns):\n"
                '        ns["BONUS"] = int(pathlib.Path("runtime.txt").read_text())\n'
                "        return super().__new__(mcs, name, bases, ns)\n\n\n"
                "class Config(metaclass=Meta):\n    pass\n"
            ),
            "runtime.txt": "1\n",
        },
        "from runtime_helper import Config",
        "runtime_helper",
    ),
}


@pytest.mark.parametrize("shape", sorted(_UNLOADED_BODY_IMPORTS))
def test_unloaded_user_body_import_is_refused_at_definition(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """A body import of a user module discovery did not load is refused: Pivot would have to
    run the module to fingerprint it. The message names the module and the fix."""
    home = tmp_path / "home"
    files, statement, module = _UNLOADED_BODY_IMPORTS[shape]
    project = _project(
        tmp_path,
        {
            **files,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                f"    {statement}\n"
                '    return {"v": len(text.splitlines())}\n',
            ),
        },
    )

    result = helpers.run_pivot(project, home, "repro")

    message = result.stdout + result.stderr
    assert result.returncode != 0, message
    assert f"imports user module '{module}'" in message
    assert "import it at module level" in message
    assert "StageParams" in message
    assert not (project / "data" / "t" / "count.yaml").exists()


def test_body_import_of_a_library_module_is_not_refused(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                "    import json\n"
                "    import textwrap\n"
                '    return {"v": len(json.loads("[1, 2]")) + len(textwrap.dedent(text).split())}\n',
            ),
        },
    )

    result = helpers.run_pivot(project, home, "repro")

    assert result.returncode == 0, result.stdout + result.stderr


def test_resolved_helper_accepts_import_module_argument_name(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/names.py": (
                "def lookup(value: str, name: str) -> str:\n    return f'{value}:{name}'\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.names import lookup\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": lookup("safe", "import_module")}\n',
            ),
        },
    )

    result = helpers.run_pivot(project, home, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == "safe:import_module"


def test_decorated_method_wrapper_edit_reruns_stage(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import functools\n\n\ndef scaled(fn):\n    @functools.wraps(fn)\n"
                "    def wrapper(self, x):\n        return fn(self, x) * 2\n\n    return wrapper\n\n\n"
                "class Scaler:\n    @scaled\n    def apply(self, x):\n        return x + 1\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Scaler\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": Scaler().apply(len(text.splitlines()))}\n',
            ),
        },
    )
    assert helpers.run_pivot(project, home, "repro").returncode == 0
    _edit(project, "helpers/model.py", "fn(self, x) * 2\n", "fn(self, x) * 200\n")

    result = helpers.run_pivot(project, home, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == 800


def test_decorator_wrapper_global_edit_reruns_stage(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import functools\n\nSCALE = 2\n\n\ndef scaled(fn):\n    @functools.wraps(fn)\n"
                "    def wrapper(self, x):\n        return fn(self, x) * SCALE\n\n    return wrapper\n\n\n"
                "class Scaler:\n    @scaled\n    def apply(self, x):\n        return x + 1\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Scaler\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": Scaler().apply(len(text.splitlines()))}\n',
            ),
        },
    )
    assert helpers.run_pivot(project, home, "repro").returncode == 0
    _edit(project, "helpers/model.py", "SCALE = 2", "SCALE = 30")

    result = helpers.run_pivot(project, home, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == 120


def test_wrapped_helper_body_import_edit_reruns_stage(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "impl.py": "LIMIT = 1\n",
            "helpers/__init__.py": "",
            "helpers/lib.py": (
                "import functools\n\nimport impl  # noqa: F401 - discovery loads it\n\n\n"
                "def limited(fn):\n    @functools.wraps(fn)\n"
                "    def wrapper(x):\n        from impl import LIMIT\n\n"
                "        return fn(x) + LIMIT\n\n    return wrapper\n\n\n"
                "@limited\ndef step(x):\n    return x\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.lib import step\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": step(len(text.splitlines()))}\n',
            ),
        },
    )
    assert helpers.run_pivot(project, home, "repro").returncode == 0
    _edit(project, "impl.py", "LIMIT = 1", "LIMIT = 100")

    result = helpers.run_pivot(project, home, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == 103


_TRACKED_CLASS_MEMBER_SHAPES = {
    "dataclass-partial-member": (
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import dataclasses\nimport functools\n\n\ndef scale(x, k):\n    return x * k\n\n\n"
                "@dataclasses.dataclass\nclass Settings:\n    width: int = 1\n"
                "    multiply = functools.partial(scale, k=2)\n\n\n"
                "def compute(settings: Settings, x):\n    return settings.multiply(x)\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Settings, compute\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": compute(Settings(), len(text.splitlines()))}\n',
            ),
        },
        ("helpers/model.py", "return x * k\n", "return x * k * 3\n"),
        18,
    ),
    "nested-class-method": (
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import dataclasses\n\nSCALE = 2\n\n\n@dataclasses.dataclass\nclass Settings:\n"
                "    width: int = 1\n\n    class Rule:\n        def apply(self, x):\n"
                "            return x * SCALE\n\n\n"
                "def compute(settings: Settings, x):\n    return settings.Rule().apply(x)\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Settings, compute\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": compute(Settings(), len(text.splitlines()))}\n',
            ),
        },
        ("helpers/model.py", "SCALE = 2", "SCALE = 3"),
        9,
    ),
    "dataclass-field-default": (
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import dataclasses\n\nWIDTH = 2\n\n\n@dataclasses.dataclass\nclass Settings:\n"
                "    width: int = WIDTH\n\n"
                "    def slots(self, x):\n        return x * self.width\n\n\n"
                "def compute(settings: Settings, x):\n    return settings.slots(x)\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Settings, compute\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": compute(Settings(), len(text.splitlines()))}\n',
            ),
        },
        ("helpers/model.py", "WIDTH = 2", "WIDTH = 3"),
        9,
    ),
    "partialmethod-member": (
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import functools\n\n\ndef multiply(self, x, k):\n    return x * k\n\n\n"
                "class Scaler:\n    apply = functools.partialmethod(multiply, k=2)\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Scaler\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": Scaler().apply(len(text.splitlines()))}\n',
            ),
        },
        ("helpers/model.py", "return x * k\n", "return x * k * 3\n"),
        18,
    ),
}


@pytest.mark.parametrize("shape", sorted(_TRACKED_CLASS_MEMBER_SHAPES))
def test_class_member_edit_reruns_stage(tmp_path: pathlib.Path, shape: str) -> None:
    home = tmp_path / "home"
    files, edit, expected = _TRACKED_CLASS_MEMBER_SHAPES[shape]
    project = _project(tmp_path, files)
    assert helpers.run_pivot(project, home, "repro").returncode == 0
    _edit(project, *edit)

    result = helpers.run_pivot(project, home, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == expected


def test_dataclass_callable_member_is_refused(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import dataclasses\n\n\nclass Scaler:\n    def __call__(self, x):\n"
                "        return x * 2\n\n\n@dataclasses.dataclass\nclass Settings:\n"
                "    scale = Scaler()\n\n\n"
                "def compute(settings: Settings, x):\n    return settings.scale(x)\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Settings, compute\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": compute(Settings(), len(text.splitlines()))}\n',
            ),
        },
    )

    result = helpers.run_pivot(project, home, "repro")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "captures non-scalar variable" in result.stdout + result.stderr
    assert "Settings.scale" in result.stdout + result.stderr


def test_guarded_unloaded_body_import_is_refused_without_running_it(
    tmp_path: pathlib.Path,
) -> None:
    """A guarded body import of a module discovery did not load is refused at definition, and
    the module is never run: one that raises on import does not break discovery."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/gpu.py": 'raise RuntimeError("helpers.gpu needs a GPU")\n',
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n    n = len(text.splitlines())\n"
                '    if os.environ.get("ACCEPT_USE_GPU"):\n        from helpers import gpu\n\n'
                '        n = gpu.boost(n)\n    return {"v": n}\n',
            ),
        },
    )

    result = helpers.run_pivot(project, home, "repro")

    message = result.stdout + result.stderr
    assert result.returncode != 0, message
    assert "imports user module 'helpers'" in message
    assert "needs a GPU" not in message
    assert "StageParams" in message


def test_body_import_package_attribute_submodule_change_reruns_stage(
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "from . import lib\n",
            "helpers/lib.py": "VALUE = 1\n",
            "pipeline.py": _pipeline(
                "\nimport helpers.lib\n",
                "@stage\ndef count(text: str) -> dict:\n"
                "    import helpers\n"
                '    return {"v": helpers.lib.VALUE + len(text.splitlines())}\n',
            ),
        },
    )
    assert helpers.run_pivot(project, home, "repro").returncode == 0
    assert _count(project) == 4
    _edit(project, "helpers/lib.py", "VALUE = 1", "VALUE = 100")

    result = helpers.run_pivot(project, home, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == 103


def test_body_import_of_unloaded_submodule_through_loaded_package_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """`import pkg.sub` where discovery loaded `pkg` but not `pkg.sub` names an unloaded user
    module, so it is refused like any other unloaded user body import."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/lib.py": "VALUE = 1\n",
            "pipeline.py": _pipeline(
                "\nimport helpers\n",
                "@stage\ndef count(text: str) -> dict:\n"
                "    import helpers.lib\n"
                '    return {"v": helpers.lib.VALUE + len(text.splitlines())}\n',
            ),
        },
    )

    result = helpers.run_pivot(project, home, "repro")

    message = result.stdout + result.stderr
    assert result.returncode != 0, message
    assert "imports user module 'helpers.lib'" in message


def test_preloaded_package_attribute_of_unloaded_package_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """A stage that loads `helpers.lib` in its body so a later stage can read `helpers.lib`
    through `import helpers` is refused when discovery loaded neither module."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            **_LIB_VALUE,
            "pipeline.py": (
                "from pivot import Pipeline, stage\n\n\n"
                "@stage\ndef preload(text: str) -> dict:\n"
                "    import helpers.lib\n"
                '    return {"v": len(text.splitlines())}\n\n\n'
                "@stage\ndef count(value: dict) -> dict:\n"
                "    import helpers\n"
                '    return {"v": helpers.lib.VALUE + value["v"]}\n\n\n'
                'with Pipeline("t") as pipeline:\n'
                '    count(preload(pipeline.input("lines.txt", t=str)))\n'
            ),
        },
    )

    result = helpers.run_pivot(project, home, "repro")

    message = result.stdout + result.stderr
    assert result.returncode != 0, message
    assert "imports user module 'helpers" in message, message
    assert "import it at module level" in message
    assert not (project / "data" / "t" / "count.yaml").exists()


_LOADED_SUBMODULE_BODIES = {
    "import-submodule-without-attribute-read": (
        "@stage\ndef count(text: str) -> dict:\n"
        "    import helpers.lib\n"
        '    return {"v": len(text.splitlines())}\n',
        "t/count",
        3,
    ),
    "from-package-import-submodule": (
        "@stage\ndef count(text: str) -> dict:\n"
        "    from helpers import lib\n"
        '    return {"v": lib.VALUE + len(text.splitlines())}\n',
        "t/count",
        103,
    ),
    "preload-stage-then-package-attribute": (
        "@stage\ndef preload(text: str) -> dict:\n"
        "    import helpers.lib\n"
        '    return {"v": helpers.lib.VALUE + len(text.splitlines())}\n\n\n'
        "@stage\ndef count(value: dict) -> dict:\n"
        "    import helpers\n"
        '    return {"v": helpers.lib.VALUE + value["v"]}\n',
        "t/preload",
        203,
    ),
}


@pytest.mark.parametrize("shape", sorted(_LOADED_SUBMODULE_BODIES))
def test_static_body_import_of_loaded_submodule_is_reused(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """A static body import of a user submodule discovery loaded is covered by the fingerprint,
    with its package: the stages run without a warning, stay cached on the next `repro`, and
    `count` shows an edit to the module it reads."""
    home = tmp_path / "home"
    body, first_stage, after_edit = _LOADED_SUBMODULE_BODIES[shape]
    wiring = (
        'count(preload(pipeline.input("lines.txt", t=str)))'
        if first_stage == "t/preload"
        else 'count(pipeline.input("lines.txt", t=str))'
    )
    project = _project(
        tmp_path,
        {
            **_LIB_VALUE,
            "pipeline.py": (
                "import helpers.lib\n\nfrom pivot import Pipeline, stage\n\n\n"
                + body
                + f'\n\nwith Pipeline("t") as pipeline:\n    {wiring}\n'
            ),
        },
    )
    configured = helpers.run_pivot(project, home, "config", "set", "core.max_workers", "1")
    assert configured.returncode == 0, configured.stdout + configured.stderr

    first = helpers.run_pivot(project, home, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    assert "does not cover" not in first.stdout + first.stderr
    assert _run_cache_entries(project) != []
    again = helpers.run_pivot(project, home, "repro", "--json")
    assert again.returncode == 0, again.stdout + again.stderr
    for stage_name in {first_stage, "t/count"}:
        assert f'"stage": "{stage_name}", "status": "cached"' in again.stdout, again.stdout
    _edit(project, *_EDIT_LIB_VALUE)

    edited = helpers.run_pivot(project, home, "repro")

    assert edited.returncode == 0, edited.stdout + edited.stderr
    assert "does not cover" not in edited.stdout + edited.stderr
    assert _count(project) == after_edit


def test_eval_lambda_is_hashed_by_its_code_object(tmp_path: pathlib.Path) -> None:
    """A lambda built by `eval` has no source; its code object is hashed, so its fingerprint is
    stable across runs and an edit to it reruns the stage."""
    home = tmp_path / "home"
    pipeline = _pipeline(
        '\nhelper = eval("lambda x: x + 1")\n',
        '@stage\ndef count(text: str) -> dict:\n    return {"v": helper(len(text.splitlines()))}\n',
    )
    project = _project(tmp_path, {"pipeline.py": pipeline})
    assert helpers.run_pivot(project, home, "repro").returncode == 0

    again = helpers.run_pivot(project, home, "repro", "--json")

    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    _edit(project, "pipeline.py", "x + 1", "x + 100")
    edited = helpers.run_pivot(project, home, "repro", "--json")
    assert '"stage": "t/count", "status": "ran"' in edited.stdout, edited.stdout
    assert _count(project) == 103


_WARNING_FIXES = {
    "import-module-through-dict": (
        "\nimport helpers.lib\n",
        'load = importlib.__dict__["import_module"]\n'
        '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}',
        'return {"v": helpers.lib.VALUE + len(text.splitlines())}',
    ),
    "resources-files": (
        "\nimport importlib.resources\n\nimport helpers\n",
        'data = importlib.resources.files("helpers").joinpath("table.txt").read_text()\n'
        '    return {"v": int(data) + len(text.splitlines())}',
        'data = importlib.resources.files(helpers).joinpath("table.txt").read_text()\n'
        '    return {"v": int(data) + len(text.splitlines())}',
    ),
}


@pytest.mark.parametrize("shape", sorted(_WARNING_FIXES))
def test_following_the_uncovered_import_warning_fix_clears_it(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """The warning's fix (import the module and read it by name in the stage's own code) clears
    it: a module-level import alone does not, and after the fix the stage is cached again."""
    home = tmp_path / "home"
    imports, as_written, fixed = _WARNING_FIXES[shape]

    def stage_body(body: str) -> str:
        return f"@stage\ndef count(text: str) -> dict:\n    {body}\n"

    project = _project(
        tmp_path,
        {
            **_LIB_VALUE,
            "helpers/table.txt": "1\n",
            "pipeline.py": _pipeline(imports, stage_body(as_written)),
        },
    )
    flagged = helpers.run_pivot(project, home, "repro")
    message = " ".join((flagged.stdout + flagged.stderr).split())
    assert flagged.returncode == 0, message
    assert message.count("imported user modules its fingerprint does not cover") == 1, message
    assert "read them by name in the stage's own code" in message, message
    assert "pipeline.py)" not in message, message
    (project / "pipeline.py").write_text(_pipeline(imports, stage_body(fixed)))

    first = helpers.run_pivot(project, home, "repro")
    again = helpers.run_pivot(project, home, "repro", "--json")

    assert first.returncode == 0, first.stdout + first.stderr
    assert "does not cover" not in " ".join((first.stdout + first.stderr).split())
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    if shape == "import-module-through-dict":
        _edit(project, *_EDIT_LIB_VALUE)
        assert helpers.run_pivot(project, home, "repro").returncode == 0
        assert _count(project) == 103


_FILELESS_LIBRARY_BODIES = {
    "xml-etree": (
        "import xml.etree.ElementTree as ET\n"
        '    return {"v": len(ET.fromstring("<a><b/></a>")) + len(text.splitlines())}'
    ),
    "networkx": (
        "import networkx\n"
        '    return {"v": networkx.path_graph(2).number_of_nodes() + len(text.splitlines())}'
    ),
}


@pytest.mark.parametrize("shape", sorted(_FILELESS_LIBRARY_BODIES))
def test_library_that_creates_fileless_modules_caches(tmp_path: pathlib.Path, shape: str) -> None:
    """A library that creates modules without a source file (`pyexpat.errors`) is not user
    code to the stage-import check: the stage caches and `pivot verify` passes, as on main."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n    "
                + _FILELESS_LIBRARY_BODIES[shape]
                + "\n",
            )
        },
    )

    first = helpers.run_pivot(project, home, "repro")
    again = helpers.run_pivot(project, home, "repro", "--json")
    verify = helpers.run_pivot(project, home, "verify")

    assert first.returncode == 0, first.stdout + first.stderr
    assert "does not cover" not in " ".join((first.stdout + first.stderr).split())
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    assert verify.returncode == 0, verify.stdout + verify.stderr


_TRANSITIVE_LAYOUTS = {
    "lib-imports-sibling": (
        {
            "helpers/__init__.py": "",
            "helpers/util.py": "SCALE = 1\n",
            "helpers/lib.py": "from helpers.util import SCALE\n\nVALUE = 1\n",
        },
        "from helpers.lib import VALUE\n",
        "VALUE",
    ),
    "package-init-reexports": (
        {
            "helpers/__init__.py": "from helpers import other  # noqa: F401\n",
            "helpers/other.py": "OTHER = 1\n",
            "helpers/lib.py": "VALUE = 1\n",
        },
        "import helpers.lib\n",
        "helpers.lib.VALUE",
    ),
}


def _transitive_pipeline(body_import: str, read: str, extra: str = "") -> str:
    return _pipeline(
        "\nimport helpers.lib\n",
        "@stage\ndef count(text: str) -> dict:\n"
        f"    {body_import}{extra}"
        f'    return {{"v": {read} + len(text.splitlines())}}\n',
    )


@pytest.mark.parametrize("shape", sorted(_TRANSITIVE_LAYOUTS))
def test_body_import_of_loaded_module_that_imports_user_modules_is_reused(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """A body import of a loaded module is walked like a module-level import even when that
    module imports other user modules: no warning, cached, and an edit reruns the stage."""
    home = tmp_path / "home"
    files, body_import, read = _TRANSITIVE_LAYOUTS[shape]
    project = _project(tmp_path, {**files, "pipeline.py": _transitive_pipeline(body_import, read)})
    configured = helpers.run_pivot(project, home, "config", "set", "core.max_workers", "1")
    assert configured.returncode == 0, configured.stdout + configured.stderr

    first = helpers.run_pivot(project, home, "repro")
    again = helpers.run_pivot(project, home, "repro", "--json")

    assert first.returncode == 0, first.stdout + first.stderr
    assert "does not cover" not in " ".join((first.stdout + first.stderr).split())
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    _edit(project, *_EDIT_LIB_VALUE)
    edited = helpers.run_pivot(project, home, "repro")
    assert edited.returncode == 0, edited.stdout + edited.stderr
    assert _count(project) == 103


def test_uncovered_dynamic_import_after_a_covered_import_still_warns(
    tmp_path: pathlib.Path,
) -> None:
    """The covered module's own import of a sibling is excused; a later runtime import of a
    third, uncovered user module in the same stage still makes the result not reusable."""
    home = tmp_path / "home"
    files, body_import, read = _TRANSITIVE_LAYOUTS["lib-imports-sibling"]
    project = _project(
        tmp_path,
        {
            **files,
            "late.py": "VALUE = 0\n",
            "pipeline.py": _transitive_pipeline(body_import, read, '    exec("import late")\n'),
        },
    )

    first = helpers.run_pivot(project, home, "repro")

    _assert_not_reusable(project, home, first)
    message = " ".join((first.stdout + first.stderr).split())
    assert "does not cover: late." in message, message
    assert "helpers.util" not in message, message


_UNHOOKED_REGISTRATIONS = {
    "exec-module": (
        "import importlib.util\nimport pathlib\nimport sys\n\n"
        "_path = pathlib.Path(__file__).resolve().parent.parent / 'plugin.py'\n"
        "_spec = importlib.util.spec_from_file_location('plugin', _path)\n"
        "_module = importlib.util.module_from_spec(_spec)\n"
        "sys.modules['plugin'] = _module\n"
        "_spec.loader.exec_module(_module)\n\n"
        "VALUE = 1\n"
    ),
    "entry-point-load": (
        "import importlib.metadata\n\n"
        "_factor = importlib.metadata.EntryPoint(name='p', value='plugin:FACTOR', group='g').load()"
        "\n\nVALUE = 1\n"
    ),
    "importlib-dunder-import": "import importlib\n\n_p = importlib.__import__('plugin')\nVALUE = 1\n",
}


@pytest.mark.parametrize("shape", sorted(_UNHOOKED_REGISTRATIONS))
def test_body_import_of_module_that_registers_a_module_without_the_hooks_is_reused(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """A loaded `helpers.lib` whose top level registers `plugin` without `__import__` or
    `importlib.import_module` (`exec_module`, `EntryPoint.load()`, `importlib.__import__`): every
    module that becomes resident during the stage's covered import of it is excused, as a
    module-level import would leave it. The stage warns nothing, caches, and reruns on an edit to
    `helpers/lib.py`."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/lib.py": _UNHOOKED_REGISTRATIONS[shape],
            "plugin.py": "FACTOR = 1\n",
            "pipeline.py": _transitive_pipeline("import helpers.lib\n", "helpers.lib.VALUE"),
        },
    )

    first = helpers.run_pivot(project, home, "repro")
    again = helpers.run_pivot(project, home, "repro", "--json")

    assert first.returncode == 0, first.stdout + first.stderr
    assert "does not cover" not in " ".join((first.stdout + first.stderr).split())
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    _edit(project, *_EDIT_LIB_VALUE)
    edited = helpers.run_pivot(project, home, "repro")
    assert edited.returncode == 0, edited.stdout + edited.stderr
    assert _count(project) == 103


_LATE_THREAD_IMPORTS = {
    "hooked": 'exec("import late")',
    "unhooked": 'importlib.__import__("late")',
}


@pytest.mark.parametrize("spelling", sorted(_LATE_THREAD_IMPORTS))
def test_thread_import_after_the_covered_import_returned_still_warns(
    tmp_path: pathlib.Path, spelling: str
) -> None:
    """A second thread the stage itself starts after its covered import returned is not a
    covered thread: its import of an uncovered user module still makes the result not reusable,
    whether or not the import goes through the hooks."""
    home = tmp_path / "home"
    files, body_import, read = _TRANSITIVE_LAYOUTS["lib-imports-sibling"]
    thread = (
        "    import threading\n\n"
        f"    loader = threading.Thread(target=lambda: {_LATE_THREAD_IMPORTS[spelling]})\n"
        "    loader.start()\n"
        "    loader.join()\n"
    )
    project = _project(
        tmp_path,
        {
            **files,
            "late.py": "VALUE = 0\n",
            "pipeline.py": _transitive_pipeline(body_import, read, thread),
        },
    )

    first = helpers.run_pivot(project, home, "repro")

    _assert_not_reusable(project, home, first)
    message = " ".join((first.stdout + first.stderr).split())
    assert "does not cover: late." in message, message
    assert "helpers.util" not in message, message


_THREADED_COVERED_IMPORTS = {
    "thread-dunder-import": (
        "import threading\n\n"
        "_t = threading.Thread(target=lambda: __import__('plugin'))\n"
        "_t.start()\n_t.join()\n\n"
        "VALUE = 1\n"
    ),
    "pool-import-module": (
        "import concurrent.futures\nimport importlib\n\n"
        "with concurrent.futures.ThreadPoolExecutor(2) as _pool:\n"
        "    _plugins = list(_pool.map(importlib.import_module, ['plugin']))\n\n"
        "VALUE = 1\n"
    ),
}


@pytest.mark.parametrize("shape", sorted(_THREADED_COVERED_IMPORTS))
def test_body_import_of_module_whose_import_thread_imports_a_user_module_is_reused(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """A loaded `helpers.lib` whose top level starts a thread that imports `plugin` through
    `__import__` or `importlib.import_module` and joins it: that thread is a covered thread, so
    its import belongs to the covered import, as a module-level import would leave it. The stage
    warns nothing, caches, and reruns on an edit to `helpers/lib.py`."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/lib.py": _THREADED_COVERED_IMPORTS[shape],
            "plugin.py": "FACTOR = 1\n",
            "pipeline.py": _transitive_pipeline("import helpers.lib\n", "helpers.lib.VALUE"),
        },
    )

    first = helpers.run_pivot(project, home, "repro")
    again = helpers.run_pivot(project, home, "repro", "--json")

    assert first.returncode == 0, first.stdout + first.stderr
    assert "does not cover" not in " ".join((first.stdout + first.stderr).split())
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    _edit(project, *_EDIT_LIB_VALUE)
    edited = helpers.run_pivot(project, home, "repro")
    assert edited.returncode == 0, edited.stdout + edited.stderr
    assert _count(project) == 103


_WARM_DEPENDENCY_PIPELINE = """\
import helpers.lib

from pivot import Pipeline, stage


@stage
def scaled(text: str) -> str:
    import helpers.util

    return text * helpers.util.SCALE


@stage
def count(text: str) -> dict:
    import helpers.lib

    return {"v": helpers.lib.VALUE + len(text.splitlines())}


with Pipeline("t") as pipeline:
    count(scaled(pipeline.input("lines.txt", t=str)))
"""


def test_covered_import_of_a_dependency_an_earlier_stage_loaded_is_reused_warm_or_cold(
    tmp_path: pathlib.Path,
) -> None:
    """`scaled` imports `helpers.util`; `count` then body-imports `helpers.lib`, whose top level
    imports `helpers.util`, in the same worker. That import belongs to the covered `helpers.lib`
    whether `helpers.util` is already resident (warm worker, first run) or not (a fresh worker
    when only `count` reruns): `count` warns nothing and caches either way."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/util.py": "SCALE = 1\n",
            "helpers/lib.py": "from helpers.util import SCALE\n\nVALUE = 1\n",
            "pipeline.py": _WARM_DEPENDENCY_PIPELINE,
        },
    )
    configured = helpers.run_pivot(project, home, "config", "set", "core.max_workers", "1")
    assert configured.returncode == 0, configured.stdout + configured.stderr

    warm = helpers.run_pivot(project, home, "repro")
    warm_again = helpers.run_pivot(project, home, "repro", "--json")
    _edit(project, *_EDIT_LIB_VALUE)
    cold = helpers.run_pivot(project, home, "repro", "--json")
    cold_again = helpers.run_pivot(project, home, "repro", "--json")

    for result in (warm, warm_again, cold, cold_again):
        assert result.returncode == 0, result.stdout + result.stderr
        assert "does not cover" not in " ".join((result.stdout + result.stderr).split())
    assert '"stage": "t/count", "status": "cached"' in warm_again.stdout, warm_again.stdout
    assert '"stage": "t/scaled", "status": "cached"' in cold.stdout, cold.stdout
    assert '"stage": "t/count", "status": "ran"' in cold.stdout, cold.stdout
    assert '"stage": "t/count", "status": "cached"' in cold_again.stdout, cold_again.stdout
    assert _count(project) == 103


_RAISING_FINDER = (
    "import sys\n\n\n"
    "class _Raising:\n"
    "    @classmethod\n"
    "    def find_spec(cls, fullname, path=None, target=None):\n"
    "        if fullname in ('optmod', 'winonly'):\n"
    "            raise RuntimeError('finder boom')\n"
    "        return None\n\n\n"
    "sys.meta_path.append(_Raising)\n"
)
_RAISING_FINDER_BODIES = {
    "caught-optional-import": (
        "    try:\n"
        "        import optmod\n"
        "    except Exception:\n"
        "        optmod = None\n"
        '    return {"v": (optmod is None) + len(text.splitlines())}\n',
        4,
    ),
    "untaken-branch": (
        "    if len(text) < 0:\n"
        "        import winonly\n\n"
        '        return {"v": winonly.VALUE}\n'
        '    return {"v": len(text.splitlines())}\n',
        3,
    ),
}


@pytest.mark.parametrize("shape", sorted(_RAISING_FINDER_BODIES))
def test_body_import_a_raising_meta_path_finder_rejects_runs_and_caches(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """A `sys.meta_path` finder that raises something other than ImportError for a name it is
    asked about does not refuse the stage: the definition-time lookup treats it as not found,
    and the stage runs and caches, as on main."""
    home = tmp_path / "home"
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_RAISING_FINDER)
    body, expected = _RAISING_FINDER_BODIES[shape]
    project = _project(
        tmp_path, {"pipeline.py": _pipeline("", "@stage\ndef count(text: str) -> dict:\n" + body)}
    )

    first = helpers.run_pivot(project, home, "repro", extra_env={"PYTHONPATH": str(site)})
    again = helpers.run_pivot(project, home, "repro", "--json", extra_env={"PYTHONPATH": str(site)})

    assert first.returncode == 0, first.stdout + first.stderr
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    assert _count(project) == expected


def test_dunder_import_with_cpython_keywords_runs_in_a_stage(tmp_path: pathlib.Path) -> None:
    """The worker's `__import__` takes CPython's keyword names (`globals=`, `locals=`)."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                "    scope = {}\n"
                "    exec(\"json = __import__('json', globals=globals(), locals=None)\", scope)\n"
                '    return {"v": len(scope["json"].dumps([1])) + len(text.splitlines())}\n',
            )
        },
    )

    result = helpers.run_pivot(project, home, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == 6


def test_unloaded_package_found_by_a_meta_path_finder_is_refused(tmp_path: pathlib.Path) -> None:
    """A package only a `sys.meta_path` finder locates (a setuptools PEP 660 editable install)
    is an unloaded user module like any other: its body import is refused at definition."""
    home = tmp_path / "home"
    package = tmp_path / "ext" / "src" / "extpkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "import importlib.util\nimport sys\n\n\n"
        "class _EditableFinder:\n"
        "    @classmethod\n"
        "    def find_spec(cls, fullname, path=None, target=None):\n"
        "        if fullname != 'extpkg':\n"
        "            return None\n"
        "        return importlib.util.spec_from_file_location(\n"
        f"            'extpkg', {str(package / '__init__.py')!r},\n"
        f"            submodule_search_locations=[{str(package)!r}],\n"
        "        )\n\n\n"
        "sys.meta_path.append(_EditableFinder)\n"
    )
    project = _project(
        tmp_path,
        {
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                "    import extpkg\n\n"
                '    return {"v": extpkg.VALUE + len(text.splitlines())}\n',
            )
        },
    )

    result = helpers.run_pivot(project, home, "repro", extra_env={"PYTHONPATH": str(site)})

    message = result.stdout + result.stderr
    assert result.returncode != 0, message
    assert "imports user module 'extpkg'" in message, message


_BOX = (
    "class Box:\n"
    "    def __init__(self, value):\n"
    "        self.value = value\n\n"
    "    def doubled(self):\n"
    "        return self.value * 2\n"
)


def _pickled_box_pipeline(annotation: str, body: str) -> str:
    make = (
        '    return {"box": Box(len(text.splitlines()))}\n'
        if annotation == "Box"
        else '    return {"box": {"b": Box(len(text.splitlines()))}}\n'
    )
    return (
        "from typing import Annotated, TypedDict\n\n"
        "from mypkg.models import Box\n"
        "from pivot import Pipeline, loaders, stage\n\n\n"
        f"class Made(TypedDict):\n    box: Annotated[{annotation}, loaders.Pickle()]\n\n\n"
        "@stage\ndef make(text: str) -> Made:\n" + make + "\n\n"
        f"@stage\ndef count(box: Annotated[{annotation}, loaders.Pickle()]) -> dict:\n"
        f'    return {{"v": {body}}}\n\n\n'
        'with Pipeline("t") as pipeline:\n'
        '    count(make(pipeline.input("lines.txt", t=str))["box"])\n'
    )


@pytest.mark.parametrize(
    ("annotation", "body", "warns"),
    [
        pytest.param("dict", 'box["b"].doubled()', True, id="plain-dict"),
        pytest.param("Box", "box.doubled()", False, id="annotation-names-class"),
    ],
)
def test_user_class_reached_only_through_a_loaded_input(
    tmp_path: pathlib.Path, annotation: str, body: str, warns: bool
) -> None:
    """A user class a stage reaches only through a pickled input counts as a module the stage
    imported: the stage warns and reruns, so an edit to the class is seen. Naming the class in
    the parameter's annotation covers it: the stage caches, and an edit still reruns it."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "src/mypkg/__init__.py": "",
            "src/mypkg/models.py": _BOX,
            "pipeline.py": _pickled_box_pipeline(annotation, body),
        },
    )
    src = project / "src"

    first = helpers.run_pivot(project, home, "repro", extra_env={"PYTHONPATH": str(src)})
    again = helpers.run_pivot(project, home, "repro", "--json", extra_env={"PYTHONPATH": str(src)})

    message = " ".join((first.stdout + first.stderr).split())
    assert first.returncode == 0, message
    assert ("does not cover: mypkg, mypkg.models" in message) is warns, message
    count_status = "ran" if warns else "cached"
    assert f'"stage": "t/count", "status": "{count_status}"' in again.stdout, again.stdout
    _edit(project, "src/mypkg/models.py", "self.value * 2", "self.value * 200")
    edited = helpers.run_pivot(project, home, "repro", "--json", extra_env={"PYTHONPATH": str(src)})
    assert '"stage": "t/count", "status": "ran"' in edited.stdout, edited.stdout
    assert _count(project) == 600


_WRAP_TAIL = (
    "\n\n@stage\n@deco\ndef count(text: str) -> dict:\n"
    '    return {"v": len(text.splitlines())}\n\n\n'
    'with Pipeline("t") as pipeline:\n'
    '    count(pipeline.input("lines.txt", t=str))\n'
)


def _wrapper_pipeline(wrapper_body: str) -> str:
    return (
        "import functools\nimport importlib\n\nfrom pivot import Pipeline, stage\n\n\n"
        "def deco(fn):\n"
        "    @functools.wraps(fn)\n"
        "    def wrapper(text):\n" + wrapper_body + "\n    return wrapper" + _WRAP_TAIL
    )


_WRAPPER_BODIES_MAIN_RUNS = {
    "getattr-computed-name": {
        "pipeline.py": _wrapper_pipeline(
            '        name = "upper"\n        return fn(getattr(text, name)())\n'
        )
    },
    "import-module": {
        "pipeline.py": _wrapper_pipeline(
            '        importlib.import_module("json")\n        return fn(text)\n'
        )
    },
    "globals": {
        "pipeline.py": _wrapper_pipeline("        _ = len(globals())\n        return fn(text)\n")
    },
    "decorator-in-helper-module": {
        "helpers/__init__.py": "",
        "helpers/deco.py": (
            "import functools\n\nNAME = 'upper'\n\n\ndef deco(fn):\n"
            "    @functools.wraps(fn)\n"
            "    def wrapper(text):\n"
            "        return fn(getattr(text, NAME)())\n\n"
            "    return wrapper\n"
        ),
        "pipeline.py": (
            "from helpers.deco import deco\n\nfrom pivot import Pipeline, stage\n" + _WRAP_TAIL
        ),
    },
}


@pytest.mark.parametrize("shape", sorted(_WRAPPER_BODIES_MAIN_RUNS))
def test_dynamic_name_refusals_read_the_wrapped_function_source(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """As on main, the definition-time refusals for a functools.wraps stage read
    `inspect.getsource` of the stage, which follows `__wrapped__`: a computed `getattr`,
    `import_module` or `globals()` in the wrapper's own body runs and caches."""
    home = tmp_path / "home"
    project = _project(tmp_path, _WRAPPER_BODIES_MAIN_RUNS[shape])

    first = helpers.run_pivot(project, home, "repro")
    again = helpers.run_pivot(project, home, "repro", "--json")

    assert first.returncode == 0, first.stdout + first.stderr
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    assert _count(project) == 3


def test_dynamic_name_refusal_reads_a_function_reached_only_through_wrapped(
    tmp_path: pathlib.Path,
) -> None:
    """A wrapper that calls `wrapper.__wrapped__` never reaches its function through the walk;
    the refusal still reads the wrapped source, so a computed `getattr` there is refused at
    definition, as on main."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "pipeline.py": (
                "import functools\n\nfrom pivot import Pipeline, stage\n\n\n"
                "def deco(fn):\n"
                "    @functools.wraps(fn)\n"
                "    def wrapper(text):\n"
                "        return wrapper.__wrapped__(text)\n\n"
                "    return wrapper\n\n\n"
                "@stage\n@deco\ndef count(text: str) -> dict:\n"
                '    name = "upper"\n'
                '    return {"v": len(getattr(text, name)().splitlines())}\n\n\n'
                'with Pipeline("t") as pipeline:\n'
                '    count(pipeline.input("lines.txt", t=str))\n'
            )
        },
    )

    result = helpers.run_pivot(project, home, "repro")

    message = result.stdout + result.stderr
    assert result.returncode != 0, message
    assert "'count' uses getattr() with non-literal attribute" in message, message


_INNER_LOADER = """\
import dataclasses
import pathlib

from pivot import loaders


@dataclasses.dataclass(frozen=True)
class Inner(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return path.read_text().splitlines()[0] + "A"
"""

_HELD_INSTANCE_HEADER = """\
import dataclasses
import functools
import pathlib

from inner import Inner
from pivot import Pipeline, loaders, stage


@dataclasses.dataclass(frozen=True)
class Outer(loaders.Reader[str]):
    inner: loaders.Reader[str]

    def load(self, path: pathlib.Path) -> str:
        return self.inner.load(path)


@dataclasses.dataclass(frozen=True)
class Outer2(loaders.Reader[str]):
    scale: int
    inner: loaders.Reader[str]

    def load(self, path: pathlib.Path) -> str:
        return self.inner.load(path) * self.scale


@dataclasses.dataclass(frozen=True)
class Settings:
    reader: loaders.Reader[str]


@dataclasses.dataclass(frozen=True)
class ThroughSettings(loaders.Reader[str]):
    settings: Settings

    def load(self, path: pathlib.Path) -> str:
        return self.settings.reader.load(path)


def _read(reader: loaders.Reader[str], path: pathlib.Path) -> str:
    return reader.load(path)


READ = functools.partial(_read, Inner())
READ_KEYWORD = functools.partial(_read, reader=Inner())


"""


def _held_stage(source: str) -> str:
    return (
        '@stage\ndef use(text: str) -> str:\n    return text\n\n\nwith Pipeline("t") as pipeline:\n'
        f'    use(pipeline.input("lines.txt", format={source}))\n'
    )


def _partial_stage(call: str) -> str:
    return (
        f"@stage\ndef use(text: str) -> str:\n    return {call}\n\n\n"
        'with Pipeline("t") as pipeline:\n    use(pipeline.input("lines.txt", t=str))\n'
    )


# Where the pivot loader `Inner()` is held: in a loader's config field, its second field, a
# non-loader dataclass in a loader's field, and the arguments or keywords of a partial the stage
# reads as a global.
_HELD_INSTANCE_SHAPES = {
    "loader-field": _held_stage("Outer(Inner())"),
    "second-loader-field": _held_stage("Outer2(1, Inner())"),
    "dataclass-in-a-loader-field": _held_stage("ThroughSettings(Settings(Inner()))"),
    "stage-level-partial": _partial_stage('READ(pathlib.Path("data/raw/lines.txt"))'),
    "stage-level-partial-keyword": _partial_stage(
        'READ_KEYWORD(path=pathlib.Path("data/raw/lines.txt"))'
    ),
}


def _assert_an_edit_reruns_once(project: pathlib.Path, home: pathlib.Path, edited: str) -> None:
    output = project / "data" / "t" / "use.txt"
    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "Local Set Default Proof UsingA"
    _edit(project, edited, '"A"', '"B"')

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "Local Set Default Proof UsingB"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


@pytest.mark.parametrize("shape", sorted(_HELD_INSTANCE_SHAPES))
def test_an_edit_to_a_held_loaders_code_reruns_its_stage(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """The pivot loader `Inner()` is held where the walk only serializes it. `Inner.load` is in
    its own module, and no annotation names `Inner`. An edit to `Inner.load` alone reruns the
    stage, and the next run reuses the result."""
    project = _project(
        tmp_path,
        {
            "inner.py": _INNER_LOADER,
            "pipeline.py": _HELD_INSTANCE_HEADER + _HELD_INSTANCE_SHAPES[shape],
        },
    )
    _assert_an_edit_reruns_once(project, tmp_path / "home", "inner.py")


_IMPORTS_ITS_MODULE = """\
import dataclasses
import pathlib

from pivot import loaders


def helper(path: pathlib.Path) -> str:
    return path.read_text().splitlines()[0] + "A"


@dataclasses.dataclass(frozen=True)
class Inner(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        import inner

        return inner.helper(path)
"""


def test_an_edit_to_a_helper_a_held_loader_imports_in_its_body_reruns_its_stage(
    tmp_path: pathlib.Path,
) -> None:
    """`Inner.load` imports its own module in its body and calls `helper`. The held loader's walk
    covers `inner` only because it walks that import and hashes `helper`, as a top-level loader's
    walk does, so an edit to `helper` reruns the stage with the new output (main reran the stage
    on every run, marking the import uncovered)."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "inner.py": _IMPORTS_ITS_MODULE,
            "pipeline.py": _HELD_INSTANCE_HEADER + _held_stage("Outer(Inner())"),
        },
    )
    output = project / "data" / "t" / "use.txt"
    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "Local Set Default Proof UsingA"
    _edit(project, "inner.py", '"A"', '"B"')

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "Local Set Default Proof UsingB"


_BASE_LOADER = _INNER_LOADER.replace("class Inner(", "class Base(")
_INHERITS = "\n\n@dataclasses.dataclass(frozen=True)\nclass Inner(Base):\n    pass\n"
_MIXIN = (
    "import pathlib\n\n\nclass ReadMixin:\n    def load(self, path: pathlib.Path) -> str:\n"
    '        return path.read_text().splitlines()[0] + "A"\n'
)

# `Inner.load` is a base class's `load` (in another module or in Inner's), or a mixin's: each
# with the files that define it and the file the edit is in.
_INHERITED_LOADS = {
    "base-in-another-module": (
        {
            "base.py": _BASE_LOADER,
            "inner.py": "import dataclasses\n\nfrom base import Base\n" + _INHERITS,
        },
        "base.py",
    ),
    "base-in-the-same-module": ({"inner.py": _BASE_LOADER + _INHERITS}, "inner.py"),
    "mixin": (
        {
            "mixins.py": _MIXIN,
            "inner.py": (
                "import dataclasses\n\nfrom mixins import ReadMixin\nfrom pivot import loaders\n"
                "\n\n@dataclasses.dataclass(frozen=True)\n"
                "class Inner(ReadMixin, loaders.Reader[str]):\n    pass\n"
            ),
        },
        "mixins.py",
    ),
}


@pytest.mark.parametrize("held", [False, True], ids=["top-level", "held"])
@pytest.mark.parametrize("inheritance", sorted(_INHERITED_LOADS))
def test_an_edit_to_an_inherited_load_reruns_its_stage(
    tmp_path: pathlib.Path, inheritance: str, held: bool
) -> None:
    """`Inner` inherits its `load`. Read as the input's loader itself, or held as
    `Outer(Inner())`, an edit to the inherited `load` reruns the stage: a held loader gets a
    top-level one's fingerprint, inherited user methods included."""
    files, edited = _INHERITED_LOADS[inheritance]
    stage = _held_stage("Outer(Inner())" if held else "Inner()")
    project = _project(tmp_path, {**files, "pipeline.py": _HELD_INSTANCE_HEADER + stage})
    _assert_an_edit_reruns_once(project, tmp_path / "home", edited)


_CALLS_A_HELPER = """\
import dataclasses
import pathlib

from pivot import loaders


@dataclasses.dataclass(frozen=True)
class Inner(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return self.decorate(path.read_text().splitlines()[0])

    def decorate(self, text: str) -> str:
        return text + "A"
"""


@pytest.mark.parametrize("held", [False, True], ids=["top-level", "held"])
def test_an_edit_to_a_loaders_helper_method_reruns_its_stage(
    tmp_path: pathlib.Path, held: bool
) -> None:
    """`Inner.load` calls `self.decorate()`. Every user method of a loader's class is walked,
    top-level or held, so an edit to `decorate` alone reruns the stage."""
    stage = _held_stage("Outer(Inner())" if held else "Inner()")
    project = _project(
        tmp_path, {"inner.py": _CALLS_A_HELPER, "pipeline.py": _HELD_INSTANCE_HEADER + stage}
    )
    _assert_an_edit_reruns_once(project, tmp_path / "home", "inner.py")


_REACHED_MEMBER = """\
import dataclasses
import functools
import pathlib

from pivot import loaders

HELPERS


@dataclasses.dataclass(frozen=True)
class Inner(loaders.Reader[str]):
MEMBERS

    def load(self, path: pathlib.Path) -> str:
        return path.read_text().splitlines()[0] + READ
"""

_SUFFIX_METHOD = '    def _suffix(self) -> str:\n        return "A"\n'
_SUFFIX_CLASSMETHOD = '    @classmethod\n    def _suffix(cls) -> str:\n        return "A"\n'
_LOGGED = """\
def logged(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper
"""

# How `load` reaches the member that holds "A": (module helpers, class members, the expression
# `load` adds to the first line). The first six are static forms; the last two use `self` in a
# way no static form names, so every member is walked.
_MEMBER_REACHES = {
    "getattr-literal": ("", _SUFFIX_METHOD, 'getattr(self, "_suffix")()'),
    "type-of-self": ("", _SUFFIX_CLASSMETHOD, "type(self)._suffix()"),
    "self-class": ("", _SUFFIX_CLASSMETHOD, "self.__class__._suffix()"),
    "wrapped-helper": (
        _LOGGED,
        "    @logged\n    def _helper(self) -> str:\n        return self._suffix()\n\n"
        + _SUFFIX_METHOD,
        "self._helper()",
    ),
    "builtin-dunder": ("", '    def __str__(self) -> str:\n        return "A"\n', "str(self)"),
    "class-constant": ("", '    SUFFIX = "A"\n', "self.SUFFIX"),
    "self-given-to-a-function": (
        "def _apply(obj):\n    return obj._suffix()\n",
        _SUFFIX_METHOD,
        "_apply(self)",
    ),
    "operator-on-self": (
        "",
        '    def __add__(self, other: str) -> str:\n        return "A" + other\n',
        '(self + "")',
    ),
}


@pytest.mark.parametrize("reach", sorted(_MEMBER_REACHES))
def test_an_edit_to_a_member_load_reaches_reruns_its_stage(
    tmp_path: pathlib.Path, reach: str
) -> None:
    """`Inner.load` reaches the member holding "A" through `getattr` with a literal name,
    `type(self)`, `self.__class__`, a `functools.wraps` helper, a builtin that runs a dunder, a
    class constant, `self` given to a module function, or an operator on `self`. An edit to the
    member alone reruns the stage with the new output."""
    helpers, members, read = _MEMBER_REACHES[reach]
    inner = (
        _REACHED_MEMBER.replace("HELPERS", helpers)
        .replace("MEMBERS", members)
        .replace("READ", read)
    )
    project = _project(
        tmp_path, {"inner.py": inner, "pipeline.py": _HELD_INSTANCE_HEADER + _held_stage("Inner()")}
    )
    _assert_an_edit_reruns_once(project, tmp_path / "home", "inner.py")


_CYCLE = "def cycle(function):\n    function.__wrapped__ = function\n    return function\n"

# A loader whose `load`, a helper `load` reaches, or a member the fallback to every member
# walks (`load` applies an operator to `self`) is its own `__wrapped__`: (the inner.py text, the
# function the refusal names).
_LOOP_WRAPPED = {
    "reached-helper": (
        _REACHED_MEMBER.replace("HELPERS", _CYCLE)
        .replace("MEMBERS", '    @cycle\n    def _suffix(self) -> str:\n        return "A"\n')
        .replace("READ", "self._suffix()"),
        "Inner._suffix",
    ),
    "load": (
        _REACHED_MEMBER.replace("HELPERS", _CYCLE)
        .replace("MEMBERS", "")
        .replace("READ", '""')
        .replace("    def load(", "    @cycle\n    def load("),
        "Inner.load",
    ),
    "fallback-member": (
        _REACHED_MEMBER.replace("HELPERS", _CYCLE)
        .replace(
            "MEMBERS",
            '    @cycle\n    def _suffix(self) -> str:\n        return "A"\n\n'
            + "    def __add__(self, other: str) -> str:\n        return other\n",
        )
        .replace("READ", '(self + "")'),
        "Inner._suffix",
    ),
}


@pytest.mark.parametrize("looped", sorted(_LOOP_WRAPPED))
def test_a_loader_method_whose_wrapped_chain_loops_is_refused(
    tmp_path: pathlib.Path, looped: str
) -> None:
    """`Inner.load`, `_suffix`, which `load` reaches, or `_suffix` as a member the fallback to
    every member walks, is its own `__wrapped__` (its decorator sets it). `pivot repro` refuses
    the stage, naming that function, and ends (`helpers.run_pivot` stops a run after 300
    seconds)."""
    inner, name = _LOOP_WRAPPED[looped]
    project = _project(
        tmp_path, {"inner.py": inner, "pipeline.py": _HELD_INSTANCE_HEADER + _held_stage("Inner()")}
    )

    result = helpers.run_pivot(project, tmp_path / "home", "repro")

    assert result.returncode != 0, result.stdout
    assert f"'{name}' has a `__wrapped__` chain that loops" in result.stdout + result.stderr


_ROW_FACTORY = """\
import collections


def make_row(fields):
    return collections.namedtuple("Row", fields)
"""

_READS_ROWS = """\
import dataclasses
import pathlib

import ROWS
from pivot import loaders


@dataclasses.dataclass(frozen=True)
class Inner(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return path.read_text().splitlines()[0] + READ
"""

# A class with no source whose definition reads another file than its own module's: the files,
# the module `Inner.load` reads (ROWS), what it adds (READ), the edit, and the output after it.
_CLASSES_FROM_OTHER_FILES = {
    "factory-in-another-module": (
        {"shapes.py": _ROW_FACTORY, "rows.py": 'import shapes\n\nRow = shapes.make_row("a b")\n'},
        "rows",
        '",".join(rows.Row._fields)',
        ("rows.py", '"a b"', '"a c"'),
        "a,c",
    ),
    "two-factory-classes-one-name": (
        {
            "shapes.py": _ROW_FACTORY,
            "rows_a.py": 'import shapes\n\nRow = shapes.make_row("a b")\n',
            "rows_b.py": 'import shapes\n\nRow = shapes.make_row("c d")\n',
        },
        "rows_a, rows_b",
        '",".join(rows_a.Row._fields + rows_b.Row._fields)',
        ("rows_b.py", '"c d"', '"c e"'),
        "a,b,c,e",
    ),
    "default-from-a-config-module": (
        {
            "config.py": 'DEFAULT = "x"\n',
            "rows.py": "import collections\n\nimport config\n\n"
            'Row = collections.namedtuple("Row", "a b", defaults=(config.DEFAULT, 1))\n',
        },
        "rows",
        'rows.Row._field_defaults["a"]',
        ("config.py", '"x"', '"y"'),
        "y",
    ),
}


@pytest.mark.parametrize("shape", sorted(_CLASSES_FROM_OTHER_FILES))
def test_an_edit_to_another_file_a_class_with_no_source_reads_reruns_its_stage(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """`Inner.load` reads a class with no source whose definition reads another file than the
    module it is named in: a factory's arguments, one of two factory classes with one name, a
    default from a config module. Its hash is not cached by its module's file, so an edit to
    that other file reruns the stage with the new output, and the next run reuses it."""
    home = tmp_path / "home"
    files, rows, read, (edited, old, new), after = _CLASSES_FROM_OTHER_FILES[shape]
    inner = _READS_ROWS.replace("ROWS", rows).replace("READ", read)
    project = _project(
        tmp_path,
        {
            **files,
            "inner.py": inner,
            "pipeline.py": _HELD_INSTANCE_HEADER + _held_stage("Inner()"),
        },
    )
    output = project / "data" / "t" / "use.txt"
    assert _repro_statuses(project, home) == {"t/use": "ran"}
    _edit(project, edited, old, new)

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == f"Local Set Default Proof Using{after}"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


_READS_A_DEFAULT = """\
import dataclasses
import pathlib

from pivot import loaders


@dataclasses.dataclass(frozen=True)
class Inner(loaders.Reader[str]):
    suffix: str = "ok"

    def load(self, path: pathlib.Path) -> str:
        from config import Defaults

        return path.read_text().splitlines()[0] + Defaults().reader.suffix
"""

_DEFAULTS = """\
import dataclasses
import typing

from inner import Inner


@dataclasses.dataclass(frozen=True)
class Defaults:
    reader: typing.Any = Inner()
"""


@pytest.mark.parametrize("held", [False, True], ids=["top-level", "held"])
def test_a_loader_its_own_walk_reaches_again_runs_and_caches(
    tmp_path: pathlib.Path, held: bool
) -> None:
    """`Inner.load` imports `config.Defaults`, whose field default is an `Inner()`: the walk
    reaches a loader it is fingerprinting already, stops there, and the stage runs and then
    caches, as at main."""
    home = tmp_path / "home"
    pipeline = _HELD_INSTANCE_HEADER.replace(
        "from inner import Inner\n", "import config  # noqa: F401\nfrom inner import Inner\n"
    )
    project = _project(
        tmp_path,
        {
            "inner.py": _READS_A_DEFAULT,
            "config.py": _DEFAULTS,
            "pipeline.py": pipeline + _held_stage("Outer(Inner())" if held else "Inner()"),
        },
    )
    output = project / "data" / "t" / "use.txt"

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "Local Set Default Proof Usingok"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


_FACTORY = """\
import dataclasses
import pathlib

from pivot import loaders


def make_reader(suffix: str) -> type:
    @dataclasses.dataclass(frozen=True)
    class Made(loaders.Reader[str]):
        def load(self, path: pathlib.Path) -> str:
            return path.read_text().splitlines()[0] + suffix

    return Made
"""

_TWO_FACTORY_CLASSES = """\
import dataclasses
import pathlib

from factory import make_reader
from pivot import Pipeline, loaders, stage

First = make_reader("first")
Second = make_reader("A")


@dataclasses.dataclass(frozen=True)
class Both(loaders.Reader[str]):
    readers: tuple

    def load(self, path: pathlib.Path) -> str:
        return "|".join(reader.load(path) for reader in self.readers)


@dataclasses.dataclass(frozen=True)
class Wrap(loaders.Reader[str]):
    inner: loaders.Reader[str]

    def load(self, path: pathlib.Path) -> str:
        return self.inner.load(path)


@stage
def use(text: str) -> str:
    return text


with Pipeline("t") as pipeline:
    use(pipeline.input("lines.txt", format=FORMAT))
"""

# A loader holding two classes one factory made, which look alike in the config text, and an
# edit: to the suffix the second is made with, or a swap of the two.
_FACTORY_EDITS = {
    "edit-the-second": (
        "Both((First(), Second()))",
        ('make_reader("A")', 'make_reader("B")'),
        "first|B",
    ),
    "swap": (
        "Both((First(), Second()))",
        ("Both((First(), Second()))", "Both((Second(), First()))"),
        "A|first",
    ),
    "edit-the-second-wrapped": (
        "Both((Wrap(First()), Wrap(Second())))",
        ('make_reader("A")', 'make_reader("B")'),
        "first|B",
    ),
}


@pytest.mark.parametrize("shape", sorted(_FACTORY_EDITS))
def test_an_edit_to_one_of_two_factory_classes_reruns_its_stage(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """`First` and `Second` come from one factory, so they share a name and a canonical form.
    Each held loader is keyed by its position and fingerprinted on its own, so a new suffix for
    `Second`, held directly or inside a holder, or a swap of the two reruns the stage."""
    home = tmp_path / "home"
    held, (old, new), after = _FACTORY_EDITS[shape]
    pipeline = _TWO_FACTORY_CLASSES.replace("FORMAT", held)
    project = _project(tmp_path, {"factory.py": _FACTORY, "pipeline.py": pipeline})
    output = project / "data" / "t" / "use.txt"
    line = "Local Set Default Proof Using"
    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == f"{line}first|{line}A"
    _edit(project, "pipeline.py", old, new)

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "|".join(f"{line}{suffix}" for suffix in after.split("|"))
    assert _repro_statuses(project, home) == {"t/use": "cached"}


_CYCLIC_SET = """\
import dataclasses
from typing import Any
import pathlib

from pivot import Pipeline, loaders, stage


@dataclasses.dataclass(frozen=True)
class SetA(loaders.Reader[str]):
    tag: str
    other: Any = dataclasses.field(default=None, compare=False, hash=False)

    def load(self, path: pathlib.Path) -> str:
        return path.read_text().splitlines()[0] + self.tag


@dataclasses.dataclass(frozen=True)
class SetB(loaders.Reader[str]):
    tag: str
    other: Any = dataclasses.field(default=None, compare=False, hash=False)

    def load(self, path: pathlib.Path) -> str:
        return path.read_text().splitlines()[0] + self.tag


A = SetA("A")
B = SetB("B")
object.__setattr__(A, "other", B)
object.__setattr__(B, "other", A)


@dataclasses.dataclass(frozen=True, eq=False)
class SetReader(loaders.Reader[str]):
    readers: frozenset

    def load(self, path: pathlib.Path) -> str:
        return "|".join(sorted(reader.load(path) for reader in self.readers))


@stage
def use(text: str) -> str:
    return text


with Pipeline("t") as pipeline:
    use(pipeline.input("lines.txt", format=SetReader(frozenset((A, B)))))
"""


def _first_of_the_set(seed: str) -> str:
    """The member a two-member frozenset yields first under `PYTHONHASHSEED=seed`: the set
    `frozenset((A, B))` of `_CYCLIC_SET`, whose members hash as `("A",)` and `("B",)` do."""
    code = 'print(next(iter(frozenset((("A",), ("B",))))))'
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**helpers.clean_env(), "PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_a_set_of_loaders_in_a_cycle_is_reused_under_any_hash_seed(tmp_path: pathlib.Path) -> None:
    """`SetA("A")` and `SetB("B")` hold each other and sit in a frozenset, whose order follows
    the hash seed. A run under a seed that yields them in the other order reuses the first
    run's result, and so does a run under the first seed again."""
    home = tmp_path / "home"
    project = _project(tmp_path, {"pipeline.py": _CYCLIC_SET})
    seeds = dict[str, str]()
    for seed in map(str, range(32)):
        seeds.setdefault(_first_of_the_set(seed), seed)
    assert len(seeds) == 2, seeds
    first, other = seeds.values()

    assert _repro_statuses(project, home, seed=first) == {"t/use": "ran"}
    assert _repro_statuses(project, home, seed=other) == {"t/use": "cached"}
    assert _repro_statuses(project, home, seed=first) == {"t/use": "cached"}


_UNUSED_MEMBERS = """\
import dataclasses
import pathlib

from pivot import loaders

COLUMNS = ["a", "b"]
NAME = "n"


@dataclasses.dataclass(frozen=True)
class Inner(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return path.read_text().splitlines()[0] + "A"

    def describe(self) -> str:
        return ",".join(COLUMNS)

    @property
    def columns(self) -> list:
        return COLUMNS

    def name(self) -> str:
        return NAME
"""


@pytest.mark.parametrize("held", [False, True], ids=["top-level", "held"])
def test_a_member_no_loader_method_reaches_is_not_walked(
    tmp_path: pathlib.Path, held: bool
) -> None:
    """`Inner.load` reads none of `describe`, `columns` and `name`. The first two read a list
    global, which the walk would refuse, and the third a string. None of them is walked: the
    stage runs, and an edit to the string only they read reuses the result."""
    home = tmp_path / "home"
    stage = _held_stage("Outer(Inner())" if held else "Inner()")
    project = _project(
        tmp_path, {"inner.py": _UNUSED_MEMBERS, "pipeline.py": _HELD_INSTANCE_HEADER + stage}
    )

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    _edit(project, "inner.py", 'NAME = "n"', 'NAME = "m"')
    assert _repro_statuses(project, home) == {"t/use": "cached"}


_SOURCELESS_LOADER = """\
import collections
import dataclasses
import enum
import pathlib
import random

import pydantic

from pivot import loaders

DEFINITION


@dataclasses.dataclass(frozen=True)
class Inner(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return path.read_text().splitlines()[0] + KIND.__name__[:0] + "A"
"""

_SLOTS_BOX = """\
class Box:
    __slots__ = ("x",)

    def __init__(self, x: int) -> None:
        self.x = x


"""

# A class with no source of its own that `Inner.load` reads, an edit to what defines it, and
# the status a comment edit elsewhere in its module gives: `cached`, or `ran` for a class that
# holds a value its form cannot show (an `object()` sentinel, a `__slots__` object, a
# `random.Random`), which is then hashed with its module's source.
_SOURCELESS_KINDS = {
    "namedtuple": (
        'KIND = collections.namedtuple("Row", "left right")',
        ('"left right"', '"left middle"'),
        "cached",
    ),
    "make-dataclass": (
        'KIND = dataclasses.make_dataclass("Gen", [("x", int)])',
        ('("x", int)', '("y", int)'),
        "cached",
    ),
    "functional-enum": ('KIND = enum.Enum("Mode", "A B")', ('"A B"', '"A C"'), "cached"),
    "type-call": ('KIND = type("Made", (), {"x": 1})', ('{"x": 1}', '{"x": 2}'), "cached"),
    "create-model": (
        'KIND = pydantic.create_model("Model", x=(int, 1), y=(str, "p1"))',
        ('"p1"', '"p2"'),
        "cached",
    ),
    "namedtuple-sentinel-default": (
        '_MISSING = object()\nKIND = collections.namedtuple("Row", "a b c", defaults=(_MISSING, "s1"))',
        ('"s1"', '"s2"'),
        "ran",
    ),
    "dataclass-random-default": (
        'KIND = dataclasses.make_dataclass("Gen", [("rng", object, dataclasses.field('
        'default=random.Random(1))), ("y", str, dataclasses.field(default="d1"))])',
        ('"d1"', '"d2"'),
        "ran",
    ),
    "dataclass-random-seed": (
        'KIND = dataclasses.make_dataclass("Gen", [("rng", object, dataclasses.field('
        'default=random.Random(1))), ("y", str, dataclasses.field(default="d1"))])',
        ("random.Random(1)", "random.Random(2)"),
        "ran",
    ),
    "namedtuple-slots-default": (
        _SLOTS_BOX + 'KIND = collections.namedtuple("Row", "a box c", defaults=(Box(1), "s1"))',
        ('"s1"', '"s2"'),
        "ran",
    ),
}


@pytest.mark.parametrize("kind", sorted(_SOURCELESS_KINDS))
def test_a_class_with_no_source_is_hashed_the_same_in_every_process(
    tmp_path: pathlib.Path, kind: str
) -> None:
    """`Inner.load` reads a class with no source of its own. Its hash is what defines it, not
    its address, so a fresh state database reuses the result and an edit to any part of its
    definition reruns the stage. A comment edit elsewhere in its module reuses it too, unless
    the class holds a value its form cannot show; that class is hashed with its module's
    source, so the edit reruns the stage."""
    home = tmp_path / "home"
    definition, (old, new), after_comment = _SOURCELESS_KINDS[kind]
    project = _project(
        tmp_path,
        {
            "inner.py": _SOURCELESS_LOADER.replace("DEFINITION", definition),
            "pipeline.py": _HELD_INSTANCE_HEADER + _held_stage("Inner()"),
        },
    )

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    shutil.rmtree(project / ".pivot" / "state.lmdb")
    assert _repro_statuses(project, home) == {"t/use": "cached"}, "A new process should reuse it"
    _edit(project, "inner.py", old, new)
    assert _repro_statuses(project, home) == {"t/use": "ran"}, "A definition edit should rerun"
    _edit(project, "inner.py", "from pivot import loaders\n", "from pivot import loaders  # note\n")
    assert _repro_statuses(project, home) == {"t/use": after_comment}


_MAPPED_PARTIAL_STAGES = """\
from typing import Annotated

from pivot import directory


@stage
def plan(
    entries: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in entries.items()}


@stage
def cut(entry: dict[str, int]) -> str:
    return READ(pathlib.Path("data/raw/lines.txt"))


with Pipeline("t") as pipeline:
    planned = plan(pipeline.input("plan.yaml", t=dict[str, int]))
    cut(pipeline.map(planned))
"""


def test_an_edit_to_a_held_loaders_code_reruns_a_mapped_instance(tmp_path: pathlib.Path) -> None:
    """The mapped stage `t/cut` calls `READ`, a global `functools.partial(_read, Inner())`. Its
    instance `t/cut@a` is fingerprinted as any stage is: an edit to `Inner.load` alone reruns
    the instance, not its producer, and the next run reuses both."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {"inner.py": _INNER_LOADER, "pipeline.py": _HELD_INSTANCE_HEADER + _MAPPED_PARTIAL_STAGES},
    )
    (project / "data" / "raw" / "plan.yaml").write_text("a: 1\n")
    output = project / "data" / "t" / "cut@a.txt"
    assert _repro_statuses(project, home) == {"t/plan": "ran", "t/cut@a": "ran"}
    assert output.read_text() == "Local Set Default Proof UsingA"
    _edit(project, "inner.py", '"A"', '"B"')

    assert _repro_statuses(project, home) == {"t/plan": "cached", "t/cut@a": "ran"}
    assert output.read_text() == "Local Set Default Proof UsingB"
    assert _repro_statuses(project, home) == {"t/plan": "cached", "t/cut@a": "cached"}


_KIND_FIELDS_PIPELINE = """\
from typing import TypedDict

import tag  # noqa: F401 - sourceless, and read by no stage
from pivot import Pipeline, stage


class Kinds(TypedDict):
    func: str
    const: str
    default: str
    schema: str
    partial: str
    loader: str


class Other(TypedDict):
    other: str


@stage
def kinds(text: str) -> Kinds:
    return {"func": text, "const": text, "default": text, "schema": text, "partial": text,
            "loader": text}


@stage
def other(text: str) -> Other:
    return {"other": text}


with Pipeline("t") as pipeline:
    kinds(pipeline.input("lines.txt", t=str))
    other(pipeline.input("lines.txt", t=str))
"""


def test_an_output_field_named_like_a_code_kind_does_not_mark_a_module_its_tag_names(
    tmp_path: pathlib.Path,
) -> None:
    """`pipeline.py` imports `tag`, a user module with no source (only `tag.pyc`) that no stage
    reads. Stage `kinds` has output fields named like the kinds of entry that hash a module's
    code (`func`, `loader`, ...), whose tags are `out:t/kinds:<field>:tag`: none of them is a use
    of module `tag`, so the stage is reused like stage `other`, and its lock carries no mark."""
    home = tmp_path / "home"
    project = _project(tmp_path, {"pipeline.py": _KIND_FIELDS_PIPELINE})
    (tmp_path / "tag.py").write_text("VALUE = 1\n")
    py_compile.compile(str(tmp_path / "tag.py"), cfile=str(project / "tag.pyc"), doraise=True)

    assert _repro_statuses(project, home) == {"t/kinds": "ran", "t/other": "ran"}
    assert _repro_statuses(project, home) == {"t/kinds": "cached", "t/other": "cached"}

    lock = yaml.safe_load((project / ".pivot" / "stages" / "t" / "kinds.lock").read_text())
    marks = [key for key in lock["code_manifest"] if key.startswith("unreadable-source:")]
    assert marks == []


_SOURCELESS_LOADERS = """\
import dataclasses
import pathlib

from pivot import loaders


@dataclasses.dataclass(frozen=True)
class Inner(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return path.read_text().splitlines()[0]


@dataclasses.dataclass(frozen=True)
class Writes(loaders.Writer[str]):
    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(data)
"""

_SOURCELESS_LOADERS_PIPELINE = """\
import dataclasses
import pathlib
from typing import Annotated, TypedDict

from pivot import Pipeline, loaders, stage
from wlib import Inner, Writes


@dataclasses.dataclass(frozen=True)
class Outer(loaders.Reader[str]):
    inner: loaders.Reader[str]

    def load(self, path: pathlib.Path) -> str:
        return self.inner.load(path)


class Written(TypedDict):
    func: Annotated[str, Writes()]


@stage
def held(text: str) -> str:
    return text


@stage
def written(text: str) -> Written:
    return {"func": text}


with Pipeline("t") as pipeline:
    held(pipeline.input("lines.txt", format=Outer(Inner())))
    written(pipeline.input("other.txt", t=str))
"""


def test_a_loader_from_a_sourceless_module_is_a_use_of_it_held_or_on_an_output_field(
    tmp_path: pathlib.Path,
) -> None:
    """`wlib` has no source (only `wlib.pyc`). Stage `held` reads its input through
    `Outer(Inner())`, `Inner` from `wlib`; stage `written` writes output field `func` with
    `Writes()` from `wlib`. Each stage's manifest hashes `wlib`'s code, under the held loader's
    `held:<digest>:` prefix or the output field's `out:` prefix, so each stage uses an unreadable
    module: it carries the mark and is never reused."""
    home = tmp_path / "home"
    project = _project(tmp_path, {"pipeline.py": _SOURCELESS_LOADERS_PIPELINE})
    (project / "data" / "raw" / "other.txt").write_text("other\n")
    (tmp_path / "wlib.py").write_text(_SOURCELESS_LOADERS)
    py_compile.compile(str(tmp_path / "wlib.py"), cfile=str(project / "wlib.pyc"), doraise=True)

    for _ in range(2):
        assert _repro_statuses(project, home) == {"t/held": "ran", "t/written": "ran"}

    for stage in ("held", "written"):
        lock = yaml.safe_load((project / ".pivot" / "stages" / "t" / f"{stage}.lock").read_text())
        marks = [key for key in lock["code_manifest"] if key.startswith("unreadable-source:")]
        assert marks == ["unreadable-source:wlib"], (stage, lock["code_manifest"])
