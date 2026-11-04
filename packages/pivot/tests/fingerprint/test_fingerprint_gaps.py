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
# walks (`load` applies an operator to `self`) is its own `__wrapped__`; each returns its "A".
_LOOP_WRAPPED = {
    "reached-helper": (
        _REACHED_MEMBER.replace("HELPERS", _CYCLE)
        .replace("MEMBERS", '    @cycle\n    def _suffix(self) -> str:\n        return "A"\n')
        .replace("READ", "self._suffix()")
    ),
    "load": (
        _REACHED_MEMBER.replace("HELPERS", _CYCLE)
        .replace("MEMBERS", "")
        .replace("READ", '"A"')
        .replace("    def load(", "    @cycle\n    def load(")
    ),
    "fallback-member": (
        _REACHED_MEMBER.replace("HELPERS", _CYCLE)
        .replace(
            "MEMBERS",
            '    @cycle\n    def _suffix(self) -> str:\n        return "A"\n\n'
            + "    def __add__(self, other: str) -> str:\n        return other + self._suffix()\n",
        )
        .replace("READ", '(self + "")')
    ),
}


@pytest.mark.parametrize("looped", sorted(_LOOP_WRAPPED))
def test_a_loader_method_whose_wrapped_chain_loops_is_walked_from_its_own_code(
    tmp_path: pathlib.Path, looped: str
) -> None:
    """`Inner.load`, `_suffix`, which `load` reaches, or `_suffix` as a member the fallback to
    every member walks, is its own `__wrapped__` (its decorator sets it). The walk reads that
    function from its own code: `pivot repro` runs the stage and ends (`helpers.run_pivot`
    stops a run after 300 seconds), an edit to the function's body reruns it, and the next run
    reuses the result."""
    project = _project(
        tmp_path,
        {
            "inner.py": _LOOP_WRAPPED[looped],
            "pipeline.py": _HELD_INSTANCE_HEADER + _held_stage("Inner()"),
        },
    )
    _assert_an_edit_reruns_once(project, tmp_path / "home", "inner.py")


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
# the status a comment edit elsewhere in its module gives: `cached`, also for a class that holds
# an `object()` sentinel or a `__slots__` object, which its form shows by their type and slots.
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
        "cached",
    ),
    "namedtuple-slots-default": (
        _SLOTS_BOX + 'KIND = collections.namedtuple("Row", "a box c", defaults=(Box(1), "s1"))',
        ('"s1"', '"s2"'),
        "cached",
    ),
}


@pytest.mark.parametrize("kind", sorted(_SOURCELESS_KINDS))
def test_a_class_with_no_source_is_hashed_the_same_in_every_process(
    tmp_path: pathlib.Path, kind: str
) -> None:
    """`Inner.load` reads a class with no source of its own. Its hash is what defines it, not
    its address, so a fresh state database reuses the result, an edit to any part of its
    definition reruns the stage, and a comment edit elsewhere in its module reuses it."""
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


def test_a_dataclass_whose_class_level_default_keeps_its_state_in_c_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """`Inner.load` reads a dataclass `make_dataclass` made, whose `rng` default is a
    `random.Random`: its state is in C, so no hash shows a change to its seed. The stage is
    refused, naming the field, with the fix."""
    home = tmp_path / "home"
    definition = (
        'KIND = dataclasses.make_dataclass("Gen", [("rng", object, dataclasses.field('
        'default=random.Random(1))), ("y", str, dataclasses.field(default="d1"))])'
    )
    project = _project(
        tmp_path,
        {
            "inner.py": _SOURCELESS_LOADER.replace("DEFINITION", definition),
            "pipeline.py": _HELD_INSTANCE_HEADER + _held_stage("Inner()"),
        },
    )

    result = helpers.run_pivot(project, home, "repro", "--json")
    message = " ".join((result.stdout + result.stderr).split())
    assert result.returncode != 0, message
    assert "the default of field 'inner.Gen.rng' keeps state pivot cannot read" in message
    assert "default_factory" in message, message


_SOURCELESS_FROM_SETTINGS = """\
import collections
import dataclasses
import pathlib
import random

from pivot import loaders

import settings

DEFINITION


@dataclasses.dataclass(frozen=True)
class Inner(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return path.read_text().splitlines()[0] + str(VIEW)
"""


def _sourceless_from_settings(
    tmp_path: pathlib.Path, settings: str, definition: str, view: str
) -> pathlib.Path:
    """A project whose `Inner.load` reads `KIND`, a class with no source `inner.py` builds from
    a value of `settings.py`, and writes `view` of it after the input's first line."""
    inner = _SOURCELESS_FROM_SETTINGS.replace("DEFINITION", definition).replace("VIEW", view)
    return _project(
        tmp_path,
        {
            "settings.py": settings,
            "inner.py": inner,
            "pipeline.py": _HELD_INSTANCE_HEADER + _held_stage("Inner()"),
        },
    )


def test_an_edit_to_a_string_a_class_with_no_source_holds_reruns_its_stage(
    tmp_path: pathlib.Path,
) -> None:
    """`KIND`, a namedtuple with no source, takes the default `settings.HEX = "0xabcdef"`, which
    looks like a memory address. Its form keeps the string as it is, so an edit to
    `settings.py` alone reruns the stage with the new output, and the next run reuses it."""
    home = tmp_path / "home"
    project = _sourceless_from_settings(
        tmp_path,
        'HEX = "0xabcdef"\n',
        'KIND = collections.namedtuple("Row", "a b", defaults=(settings.HEX,))',
        "KIND(1).b",
    )
    output = project / "data" / "t" / "use.txt"

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "Local Set Default Proof Using0xabcdef"
    assert _repro_statuses(project, home) == {"t/use": "cached"}
    _edit(project, "settings.py", '"0xabcdef"', '"0x123456"')

    assert _repro_statuses(project, home) == {"t/use": "ran"}, "The edit should rerun"
    assert output.read_text() == "Local Set Default Proof Using0x123456"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


def test_a_class_with_no_source_holding_state_no_form_reads_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """`KIND`, a namedtuple with no source, takes the default `random.Random(settings.SEED)`:
    its state is in C, so no hash of the class shows an edit to `settings.SEED`. The stage is
    refused, naming the class, with the fix, and no output is written."""
    home = tmp_path / "home"
    project = _sourceless_from_settings(
        tmp_path,
        "SEED = 1\n",
        'KIND = collections.namedtuple("Row", "a rng", defaults=(random.Random(settings.SEED),))',
        "KIND(1).rng.randint(0, 9)",
    )

    result = helpers.run_pivot(project, home, "repro", "--json")
    message = " ".join((result.stdout + result.stderr).split())
    assert result.returncode != 0, f"The stage should be refused: {message}"
    assert "class 'inner.Row', which has no source of its own, holds a value" in message, message
    assert "StageParams" in message, message
    assert not (project / "data" / "t" / "use.txt").exists()


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


_THING = 'class Thing:\n    def suffix(self) -> str:\n        return "A"\n'

_CAPTURED_HEADER = """\
import dataclasses
import functools
import pathlib
from typing import Any

from pivot import Pipeline, StageParams, loaders, stage
from thing import Thing


def _tag(thing: Any, text: str) -> str:
    return text.splitlines()[0] + thing.suffix()


"""


def _captured_stage(call: str, source: str = 'pipeline.input("lines.txt", t=str)') -> str:
    return (
        f"@stage\ndef use(text: str) -> str:\n    return {call}\n\n\n"
        f'with Pipeline("t") as pipeline:\n    use({source})\n'
    )


_GLOBAL_PARTIAL = "TAG = functools.partial(_tag, {})\n\n\n" + _captured_stage("TAG(text)")

# Where the stage captures `Thing()` (a module global partial's argument or keyword, a loader's
# config field, the field default of a dataclass the stage reads, the field default of the
# StageParams default of a helper), and the other kinds of instance a global partial holds (a
# class whose `suffix` a base class in another module defines, a `str` subclass, an enum member,
# a dataclass, a pydantic model): each with its files and the file that defines `suffix`. No
# annotation names the class, so the walk reaches it only through the instance.
_CAPTURED_INSTANCES = {
    "global-partial": ({"pipeline.py": _GLOBAL_PARTIAL.format("Thing()")}, "thing.py"),
    "partial-keyword": (
        {
            "pipeline.py": "TAG = functools.partial(_tag, thing=Thing())\n\n\n"
            + _captured_stage("TAG(text=text)")
        },
        "thing.py",
    ),
    "loader-config-field": (
        {
            "pipeline.py": "@dataclasses.dataclass(frozen=True)\n"
            "class Tagged(loaders.Reader[str]):\n    thing: Any\n\n"
            "    def load(self, path: pathlib.Path) -> str:\n"
            "        return _tag(self.thing, path.read_text())\n\n\n"
            + _captured_stage("text", 'pipeline.input("lines.txt", format=Tagged(Thing()))')
        },
        "thing.py",
    ),
    "dataclass-default": (
        {
            "pipeline.py": "@dataclasses.dataclass(frozen=True)\n"
            "class Settings:\n    thing: Any = Thing()\n\n\n"
            + _captured_stage("_tag(Settings().thing, text)")
        },
        "thing.py",
    ),
    "stageparams-default": (
        {
            "pipeline.py": "class TagParams(StageParams):\n    thing: Any = Thing()\n\n\n"
            "def _tag_with(text: str, params: TagParams = TagParams()) -> str:\n"
            "    return _tag(params.thing, text)\n\n\n" + _captured_stage("_tag_with(text)")
        },
        "thing.py",
    ),
    "inherited-method": (
        {
            "base.py": _THING.replace("class Thing:", "class Base:"),
            "thing.py": "from base import Base\n\n\nclass Thing(Base):\n    pass\n",
            "pipeline.py": _GLOBAL_PARTIAL.format("Thing()"),
        },
        "base.py",
    ),
    "str-subclass": (
        {
            "thing.py": _THING.replace("class Thing:", "class Thing(str):"),
            "pipeline.py": _GLOBAL_PARTIAL.format('Thing("t")'),
        },
        "thing.py",
    ),
    "enum-member": (
        {
            "thing.py": "import enum\n\n\n"
            + _THING.replace("class Thing:", "class Thing(enum.Enum):\n    ONE = 1\n"),
            "pipeline.py": _GLOBAL_PARTIAL.format("Thing.ONE"),
        },
        "thing.py",
    ),
    "dataclass-instance": (
        {
            "thing.py": "import dataclasses\n\n\n@dataclasses.dataclass(frozen=True)\n"
            + _THING.replace("class Thing:", "class Thing:\n    x: int = 1\n"),
            "pipeline.py": _GLOBAL_PARTIAL.format("Thing()"),
        },
        "thing.py",
    ),
    "pydantic-model": (
        {
            "thing.py": "import pydantic\n\n\n"
            + _THING.replace("class Thing:", "class Thing(pydantic.BaseModel):\n    x: int = 1\n"),
            "pipeline.py": _GLOBAL_PARTIAL.format("Thing()"),
        },
        "thing.py",
    ),
}


@pytest.mark.parametrize("shape", sorted(_CAPTURED_INSTANCES))
def test_an_edit_to_a_method_of_a_captured_instance_reruns_its_stage(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """The stage captures an instance of a user class, which is not a pivot loader, where the
    walk only serializes it. Its class is walked as for a stage that names it, so an edit to
    the method the stage calls alone reruns the stage, and the next run reuses the result."""
    home = tmp_path / "home"
    files, edited = _CAPTURED_INSTANCES[shape]
    pipeline = _CAPTURED_HEADER + files["pipeline.py"]
    project = _project(tmp_path, {"thing.py": _THING, **files, "pipeline.py": pipeline})
    output = project / "data" / "t" / "use.txt"
    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "Local Set Default Proof UsingA"
    _edit(project, edited, '"A"', '"B"')

    assert _repro_statuses(project, home) == {"t/use": "ran"}, "The method edit should rerun"
    assert output.read_text() == "Local Set Default Proof UsingB"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


_NOISE_HEADER = """\
import collections
import dataclasses
import datetime
import decimal
import enum
import fractions
import functools
import pathlib
from typing import Any

import pydantic

from pivot import Pipeline, stage

Row = collections.namedtuple("Row", "x y")
Gen = dataclasses.make_dataclass("Gen", [("x", int)], frozen=True)
Mode = enum.Enum("Mode", "A B")
Made = type("Made", (), {"x": 1})
Model = pydantic.create_model("Model", x=(int, 1))


def _take(value: Any, text: str) -> str:
    return text.splitlines()[0] + str(value)[:0]

"""

# Instances of classes with no source of their own, and of library classes, that a global
# partial holds.
_UNEDITED_INSTANCES = {
    "namedtuple": "Row(1, 2)",
    "make_dataclass": "Gen(1)",
    "functional_enum": "Mode.A",
    "type_call": "Made()",
    "create_model": "Model()",
    "decimal": 'decimal.Decimal("1.5")',
    "fraction": "fractions.Fraction(1, 3)",
    "date": "datetime.date(2020, 1, 1)",
    "path": 'pathlib.PurePosixPath("a/b")',
}


def test_a_captured_instance_with_no_source_or_from_a_library_reruns_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """Each stage calls a global partial holding an instance of a class with no source of its
    own (hashed by what defines it) or of a library class (not walked). Under a new hash seed, a
    comment edit in the module that defines them and a fresh state database, every stage reuses
    its first result."""
    home = tmp_path / "home"
    stages = "".join(
        f"HOLD_{name} = functools.partial(_take, {expression})\n\n\n"
        f"@stage\ndef use_{name}(text: str) -> str:\n    return HOLD_{name}(text)\n\n\n"
        for name, expression in _UNEDITED_INSTANCES.items()
    )
    calls = "".join(
        f'    use_{name}(pipeline.input("lines.txt", t=str))\n' for name in _UNEDITED_INSTANCES
    )
    pipeline = _NOISE_HEADER + "\n" + stages + f'with Pipeline("t") as pipeline:\n{calls}'
    project = _project(tmp_path, {"pipeline.py": pipeline})
    ran = {f"t/use_{name}": "ran" for name in _UNEDITED_INSTANCES}
    cached = dict.fromkeys(ran, "cached")

    assert _repro_statuses(project, home, seed="1") == ran
    assert _repro_statuses(project, home, seed="2") == cached, "A new hash seed should reuse it"
    _edit(project, "pipeline.py", "\n\ndef _take(", "\n\n# A comment.\ndef _take(")
    assert _repro_statuses(project, home, seed="3") == cached, "A comment edit should reuse it"
    shutil.rmtree(project / ".pivot" / "state.lmdb")
    assert _repro_statuses(project, home, seed="4") == cached, "A fresh state should reuse it"


_OPAQUE_DEFAULTS = '''\
import dataclasses
from typing import Any, ClassVar

_MISSING = object()


class Slotted:
    __slots__ = ("x",)

    def __init__(self) -> None:
        self.x = 1


@dataclasses.dataclass(frozen=True)
class Sentinel:
    """A sentinel class default the instance overrides."""

    token: Any = _MISSING

    def value(self) -> str:
        return str(self.token)


@dataclasses.dataclass(frozen=True)
class SlotsDefault:
    """A `__slots__` object class default the instance overrides."""

    box: Any = Slotted()

    def value(self) -> str:
        return str(self.box)


@dataclasses.dataclass(frozen=True)
class WithClassVar:
    """A ClassVar that holds an object; not a field of the instance."""

    marker: ClassVar[Any] = object()
    x: int = 1

    def value(self) -> str:
        return str(self.x)
'''

_OPAQUE_DEFAULTS_PIPELINE = """\
import functools
from typing import Any

from defaults import Sentinel, SlotsDefault, WithClassVar
from pivot import Pipeline, stage


def _take(thing: Any, text: str) -> str:
    return text.splitlines()[0] + thing.value()


SENTINEL = functools.partial(_take, Sentinel("explicit"))
SLOTS = functools.partial(_take, SlotsDefault("explicit"))
CLASSVAR = functools.partial(_take, WithClassVar())


@stage
def use_sentinel(text: str) -> str:
    return SENTINEL(text)


@stage
def use_slots(text: str) -> str:
    return SLOTS(text)


@stage
def use_classvar(text: str) -> str:
    return CLASSVAR(text)


with Pipeline("t") as pipeline:
    use_sentinel(pipeline.input("lines.txt", t=str))
    use_slots(pipeline.input("lines.txt", t=str))
    use_classvar(pipeline.input("lines.txt", t=str))
"""


def test_a_captured_dataclass_with_class_defaults_no_form_shows_reruns_nothing_new(
    tmp_path: pathlib.Path,
) -> None:
    """Each stage captures a dataclass instance whose class keeps a default no repr shows: an
    `object()` sentinel or a `__slots__` object the instance overrides, or a `ClassVar` object.
    The class walk hashes such a default by its type and its slots, never by its address, so the
    same hash seed, a new one, a fresh state database and a comment edit in either module reuse
    the result."""
    home = tmp_path / "home"
    project = _project(
        tmp_path, {"defaults.py": _OPAQUE_DEFAULTS, "pipeline.py": _OPAQUE_DEFAULTS_PIPELINE}
    )
    names = ("t/use_sentinel", "t/use_slots", "t/use_classvar")
    ran, cached = dict.fromkeys(names, "ran"), dict.fromkeys(names, "cached")

    assert _repro_statuses(project, home, seed="1") == ran
    assert _repro_statuses(project, home, seed="1") == cached, "The same seed should reuse it"
    assert _repro_statuses(project, home, seed="2") == cached, "A new hash seed should reuse it"
    shutil.rmtree(project / ".pivot" / "state.lmdb")
    assert _repro_statuses(project, home, seed="3") == cached, "A fresh state should reuse it"
    _edit(project, "pipeline.py", "\n\ndef _take(", "\n\n# A comment.\ndef _take(")
    assert _repro_statuses(project, home, seed="4") == cached, "A comment edit should reuse it"
    _edit(project, "defaults.py", "import dataclasses\n", "import dataclasses  # note\n")
    assert _repro_statuses(project, home, seed="5") == cached, "A comment edit should reuse it"


_FACTORY_MODULE = """\
def make(suffix: str) -> type:
    class Made:
        def suffix(self) -> str:
            return suffix

    return Made
"""

# The same factory made with `type()`: its classes have no source and no `<locals>` in their
# qualified name, and its module's name does not resolve to them.
_TYPE_FACTORY_MODULE = """\
def _suffix_of(suffix: str):
    def suffix_method(self) -> str:
        return suffix

    return suffix_method


def make(suffix: str) -> type:
    return type("Made", (), {"suffix": _suffix_of(suffix)})
"""

_TWO_HOLDERS_HEADER = """\
import dataclasses
import functools
from typing import Any

from factory import make
from pivot import Pipeline, StageParams, stage


def _take(thing: Any, text: str) -> str:
    return text.splitlines()[0] + thing.suffix()


First = make("A")
Second = make("Z")


"""

# Two values of one stage that hold instances of two classes `factory.make` made, at the same
# position in each: two partials, two field defaults of one dataclass, and two helpers'
# StageParams defaults. The factory calls are module statements no walked code holds, so only
# the walks of the classes see an edit to their arguments.
_TWO_HOLDERS = {
    "two-partials": (
        "FIRST = functools.partial(_take, First())\n"
        "SECOND = functools.partial(_take, Second())\n\n\n"
        "@stage\ndef use(text: str) -> str:\n    return FIRST(text) + SECOND(text)\n"
    ),
    "two-fields": (
        "@dataclasses.dataclass(frozen=True)\nclass Pair:\n"
        "    first: Any = First()\n    second: Any = Second()\n\n\n"
        "@stage\ndef use(text: str) -> str:\n"
        "    return _take(Pair().first, text) + _take(Pair().second, text)\n"
    ),
    "two-params": (
        "class FirstParams(StageParams):\n    made: Any = First()\n\n\n"
        "class SecondParams(StageParams):\n    made: Any = Second()\n\n\n"
        "def _first(text: str, params: Any = FirstParams()) -> str:\n"
        "    return _take(params.made, text)\n\n\n"
        "def _second(text: str, params: Any = SecondParams()) -> str:\n"
        "    return _take(params.made, text)\n\n\n"
        "@stage\ndef use(text: str) -> str:\n    return _first(text) + _second(text)\n"
    ),
}


@pytest.mark.parametrize("shape", sorted(_TWO_HOLDERS))
def test_an_edit_to_one_of_two_factory_classes_two_values_hold_reruns_its_stage(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """Two values of the stage hold instances of two classes one factory made, at the same
    position in each. The first class's walk keeps its own key, so changing only the factory
    argument that makes it reruns the stage with the new output; the second never overwrites
    it."""
    home = tmp_path / "home"
    pipeline = (
        _TWO_HOLDERS_HEADER
        + _TWO_HOLDERS[shape]
        + '\n\nwith Pipeline("t") as pipeline:\n    use(pipeline.input("lines.txt", t=str))\n'
    )
    project = _project(tmp_path, {"factory.py": _FACTORY_MODULE, "pipeline.py": pipeline})
    output = project / "data" / "t" / "use.txt"
    line = "Local Set Default Proof Using"

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == f"{line}A{line}Z"
    assert _repro_statuses(project, home) == {"t/use": "cached"}
    _edit(project, "pipeline.py", 'make("A")', 'make("B")')

    assert _repro_statuses(project, home) == {"t/use": "ran"}, "The edit should rerun"
    assert output.read_text() == f"{line}B{line}Z"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


@pytest.mark.parametrize(
    "factory", [_FACTORY_MODULE, _TYPE_FACTORY_MODULE], ids=["class-statement", "type-call"]
)
@pytest.mark.parametrize("shape", sorted(_TWO_HOLDERS))
def test_swapping_two_factory_classes_between_two_values_reruns_its_stage(
    tmp_path: pathlib.Path, shape: str, factory: str
) -> None:
    """Two values of the stage hold instances of two classes one factory made, with a class
    statement or with `type()`. Swapping the factory arguments between them leaves the same two
    classes, now held by the other value: each is keyed by its identity inside a value, so the
    swap reruns the stage with the swapped output."""
    home = tmp_path / "home"
    pipeline = (
        _TWO_HOLDERS_HEADER
        + _TWO_HOLDERS[shape]
        + '\n\nwith Pipeline("t") as pipeline:\n    use(pipeline.input("lines.txt", t=str))\n'
    )
    project = _project(tmp_path, {"factory.py": factory, "pipeline.py": pipeline})
    output = project / "data" / "t" / "use.txt"
    line = "Local Set Default Proof Using"

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert _repro_statuses(project, home) == {"t/use": "cached"}
    _edit(project, "pipeline.py", 'make("A")', 'make("Q")')
    _edit(project, "pipeline.py", 'make("Z")', 'make("A")')
    _edit(project, "pipeline.py", 'make("Q")', 'make("Z")')

    assert _repro_statuses(project, home) == {"t/use": "ran"}, "The swap should rerun"
    assert output.read_text() == f"{line}Z{line}A"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


_NAMED_AND_HELD = """\
import dataclasses
import pathlib
from typing import Any

from factory import make
from pivot import Pipeline, loaders, stage

Made = make("A")


@dataclasses.dataclass(frozen=True)
class Reader(loaders.Reader[str]):
    made: Any = None

    def load(self, path: pathlib.Path) -> str:
        return path.read_text().splitlines()[0] + self.made.suffix()


@stage
def use(text: str) -> str:
    return text + Made().suffix()


with Pipeline("t") as pipeline:
    use(pipeline.input("lines.txt", format=Reader(Made())))
"""


def test_a_class_the_stage_names_and_its_loader_holds_is_cached_warm_and_cold(
    tmp_path: pathlib.Path,
) -> None:
    """The stage names `Made`, a class `factory.make` made (`<locals>` in its name), and its
    input's loader holds an instance of it. The stage and its loaders are walked apart, so the
    loader's entries are the same whether the stage's manifest comes from the cache (warm) or
    from a walk (cold): the stage runs once, then is cached warm, cold, and warm again."""
    home = tmp_path / "home"
    project = _project(tmp_path, {"factory.py": _FACTORY_MODULE, "pipeline.py": _NAMED_AND_HELD})

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert _repro_statuses(project, home) == {"t/use": "cached"}, "A warm run should reuse it"
    shutil.rmtree(project / ".pivot" / "state.lmdb")
    assert _repro_statuses(project, home) == {"t/use": "cached"}, "A cold run should reuse it"
    assert _repro_statuses(project, home) == {"t/use": "cached"}, "A warm run should reuse it"


_DEFAULT_FROM_ELSEWHERE_PIPELINE = """\
import functools
from typing import Any

from pivot import Pipeline, stage

import defaults


def _fresh(thing: Any, text: str) -> str:
    return text.splitlines()[0] + thing.fresh()


CAPTURED = functools.partial(_fresh, defaults.Box(None))


@stage
def named(text: str) -> str:
    return text.splitlines()[0] + defaults.Box().fresh()


@stage
def captured(text: str) -> str:
    return CAPTURED(text)


with Pipeline("t") as pipeline:
    named(pipeline.input("lines.txt", t=str))
    captured(pipeline.input("lines.txt", t=str))
"""

_DEFAULT_FROM_ELSEWHERE_BOX = """\
import dataclasses
from typing import Any

import settings


@dataclasses.dataclass(frozen=True)
class Box:
    value: Any = DEFAULT

    def fresh(self) -> str:
        return str(type(self)().valueVIEW)
"""

# A dataclass whose class-level default comes from another module: a string that looks like a
# memory address, and a `__slots__` object built from a value; each with the edit to
# `settings.py` that changes it, and the output before and after.
_DEFAULTS_FROM_ELSEWHERE = {
    "hex-string": (
        'HEX = "0xabcdef"\n',
        {"DEFAULT": "settings.HEX", "VIEW": ""},
        ('"0xabcdef"', '"0x123456"'),
        ("0xabcdef", "0x123456"),
    ),
    "slots-object": (
        'VALUE = 1\n\n\nclass Slotted:\n    __slots__ = ("x",)\n\n'
        "    def __init__(self, x: int) -> None:\n        self.x = x\n",
        {"DEFAULT": "settings.Slotted(settings.VALUE)", "VIEW": ".x"},
        ("VALUE = 1", "VALUE = 2"),
        ("1", "2"),
    ),
    "slots-callback": (
        'VALUE = "A"\n\n\nclass Callback:\n    __slots__ = ("callback",)\n\n'
        "    def __init__(self, callback) -> None:\n        self.callback = callback\n\n"
        "    def read(self) -> str:\n        return self.callback()\n",
        {"DEFAULT": "settings.Callback(lambda: settings.VALUE)", "VIEW": ".read()"},
        ('VALUE = "A"', 'VALUE = "B"'),
        ("A", "B"),
    ),
    "hex-bytes": (
        'DATA = b"0xabcdef"\n',
        {"DEFAULT": "settings.DATA", "VIEW": ".decode()"},
        ('b"0xabcdef"', 'b"0x123456"'),
        ("0xabcdef", "0x123456"),
    ),
    "hex-path": (
        'import pathlib\n\nPATH = pathlib.PurePosixPath("cache/0xabcdef")\n',
        {"DEFAULT": "settings.PATH", "VIEW": ""},
        ('"cache/0xabcdef"', '"cache/0x123456"'),
        ("cache/0xabcdef", "cache/0x123456"),
    ),
    "exception-attribute": (
        'ERROR = ValueError("bad")\nERROR.code = "A"\n',
        {"DEFAULT": "settings.ERROR", "VIEW": ".code"},
        ('ERROR.code = "A"', 'ERROR.code = "B"'),
        ("A", "B"),
    ),
    "identity-bytes-at": (
        'DATA = b"located at 0xabcdef"\n',
        {"DEFAULT": "settings.DATA", "VIEW": ".decode()"},
        ('b"located at 0xabcdef"', 'b"located at 0x123456"'),
        ("located at 0xabcdef", "located at 0x123456"),
    ),
    "identity-bytes-py": (
        'DATA = b"Py(0xabcdef)"\n',
        {"DEFAULT": "settings.DATA", "VIEW": ".decode()"},
        ('b"Py(0xabcdef)"', 'b"Py(0x123456)"'),
        ("Py(0xabcdef)", "Py(0x123456)"),
    ),
    "identity-bytes-object": (
        'DATA = b"<object at 0xabcdef>"\n',
        {"DEFAULT": "settings.DATA", "VIEW": ".decode()"},
        ('b"<object at 0xabcdef>"', 'b"<object at 0x123456>"'),
        ("<object at 0xabcdef>", "<object at 0x123456>"),
    ),
    "identity-path-at": (
        'import pathlib\n\nPATH = pathlib.PurePosixPath("cache/at 0xabcdef")\n',
        {"DEFAULT": "settings.PATH", "VIEW": ""},
        ('"cache/at 0xabcdef"', '"cache/at 0x123456"'),
        ("cache/at 0xabcdef", "cache/at 0x123456"),
    ),
    "classmethod-callback": (
        'VALUE = "A"\n\n\nclass ClassCallback:\n'
        "    @classmethod\n    def read(cls) -> str:\n        return VALUE\n\n\n"
        "class DictCallback:\n    def __init__(self, callback) -> None:\n"
        "        self.callback = callback\n\n    def read(self) -> str:\n        return self.callback()\n",
        {"DEFAULT": "settings.DictCallback(settings.ClassCallback.read)", "VIEW": ".read()"},
        ('VALUE = "A"', 'VALUE = "B"'),
        ("A", "B"),
    ),
    "os-exception": (
        'ERROR = FileNotFoundError(2, "missing", "A")\n',
        {"DEFAULT": "settings.ERROR", "VIEW": ".filename"},
        ('FileNotFoundError(2, "missing", "A")', 'FileNotFoundError(2, "missing", "B")'),
        ("A", "B"),
    ),
}


@pytest.mark.parametrize("shape", sorted(_DEFAULTS_FROM_ELSEWHERE))
def test_an_edit_to_a_class_default_taken_from_another_module_reruns_its_stages(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """`defaults.Box`'s class-level default is built from `settings.py`. A stage that names the
    class and one that captures an instance overriding the default (and reads a fresh instance's
    default) run once and reuse their result under the same and a new hash seed; an edit to
    `settings.py` that changes the default reruns both with the new output."""
    home = tmp_path / "home"
    settings_text, box_names, (old, new), (before, after) = _DEFAULTS_FROM_ELSEWHERE[shape]
    box = _DEFAULT_FROM_ELSEWHERE_BOX
    for name, text in box_names.items():
        box = box.replace(name, text)
    project = _project(
        tmp_path,
        {
            "settings.py": settings_text,
            "defaults.py": box,
            "pipeline.py": _DEFAULT_FROM_ELSEWHERE_PIPELINE,
        },
    )
    line = "Local Set Default Proof Using"
    stages = ("t/named", "t/captured")

    assert _repro_statuses(project, home, seed="1") == dict.fromkeys(stages, "ran")
    assert _repro_statuses(project, home, seed="1") == dict.fromkeys(stages, "cached")
    assert _repro_statuses(project, home, seed="2") == dict.fromkeys(stages, "cached")
    for name in ("named", "captured"):
        assert (project / "data" / "t" / f"{name}.txt").read_text() == f"{line}{before}"
    _edit(project, "settings.py", old, new)

    assert _repro_statuses(project, home, seed="1") == dict.fromkeys(stages, "ran"), (
        "The edit to the default should rerun"
    )
    for name in ("named", "captured"):
        assert (project / "data" / "t" / f"{name}.txt").read_text() == f"{line}{after}"
    assert _repro_statuses(project, home, seed="2") == dict.fromkeys(stages, "cached")


_TYPE_SUBCLASS_DEFAULTS = """\
import dataclasses
from typing import Any

import settings


@dataclasses.dataclass(frozen=True)
class Base:
    marker: Any = settings.LABEL


Derived = type("Derived", (Base,), {})
"""

_TYPE_SUBCLASS_PIPELINE = """\
import functools
from typing import Any

from pivot import Pipeline, stage

import defaults


def _fresh(value: Any, text: str) -> str:
    return text.splitlines()[0] + str(type(value)().marker)


CAPTURED = functools.partial(_fresh, defaults.Derived("explicit"))


@stage
def captured(text: str) -> str:
    return CAPTURED(text)


@stage
def named(text: str) -> str:
    return text.splitlines()[0] + str(defaults.Derived().marker)


with Pipeline("t") as pipeline:
    captured(pipeline.input("lines.txt", t=str))
    named(pipeline.input("lines.txt", t=str))
"""


def test_an_edit_to_the_default_a_dataclass_subclass_without_a_statement_takes_reruns(
    tmp_path: pathlib.Path,
) -> None:
    """`defaults.Derived`, made by `type()` with an empty namespace, takes the `marker` default
    of its dataclass base, which comes from `settings.LABEL`. The walk of `Derived` resolves its
    defaults over its MRO, so an edit to `settings.py` alone reruns the stage that names
    `Derived` and the one that captures an instance of it, with the new output."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "settings.py": 'LABEL = "A"\n',
            "defaults.py": _TYPE_SUBCLASS_DEFAULTS,
            "pipeline.py": _TYPE_SUBCLASS_PIPELINE,
        },
    )
    line = "Local Set Default Proof Using"
    stages = ("t/named", "t/captured")

    assert _repro_statuses(project, home) == dict.fromkeys(stages, "ran")
    assert _repro_statuses(project, home) == dict.fromkeys(stages, "cached")
    _edit(project, "settings.py", '"A"', '"B"')

    assert _repro_statuses(project, home) == dict.fromkeys(stages, "ran"), "The edit should rerun"
    for name in ("named", "captured"):
        assert (project / "data" / "t" / f"{name}.txt").read_text() == f"{line}B"
    assert _repro_statuses(project, home) == dict.fromkeys(stages, "cached")


_RANDOM_READER_DEFAULT = """\
import dataclasses
import pathlib
import random
from typing import Any

from pivot import loaders


@dataclasses.dataclass(frozen=True)
class RandomReader(loaders.Reader[str]):
    rng: Any

    def load(self, path: pathlib.Path) -> str:
        return str(self.rng.random())


@dataclasses.dataclass(frozen=True)
class Box:
    reader: Any = RandomReader(random.Random(1))
"""

_RANDOM_READER_PIPELINE = """\
from pivot import Pipeline, stage

import defaults


@stage
def use(text: str) -> str:
    return text.splitlines()[0] + defaults.Box().reader.load(None)


with Pipeline("t") as pipeline:
    use(pipeline.input("lines.txt", t=str))
"""


def test_a_loader_in_a_default_whose_config_holds_unreadable_state_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """`defaults.Box`'s default is a reader whose config holds a `random.Random`. A run records
    the field's read first, which keeps the loader's form for the stage; the kept form still
    reports the generator's state, so the stage is refused, naming the field, and writes no
    output."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {"defaults.py": _RANDOM_READER_DEFAULT, "pipeline.py": _RANDOM_READER_PIPELINE},
    )

    result = helpers.run_pivot(project, home, "repro", "--json")
    message = " ".join((result.stdout + result.stderr).split())
    assert result.returncode != 0, f"The stage should be refused: {message}"
    assert "the default of field 'defaults.Box.reader' keeps state pivot cannot read" in message
    assert not (project / "data" / "t" / "use.txt").exists()


def test_swapping_later_instances_of_two_factory_classes_in_one_list_reruns_its_stage(
    tmp_path: pathlib.Path,
) -> None:
    """A partial holds `[First(), Second(), First(), Second()]`, two classes one factory made.
    Swapping the last two leaves each class's first instance where it was; each class's walk
    is keyed by every position of its instances, so the swap reruns the stage with the new
    output."""
    home = tmp_path / "home"
    pipeline = (
        _TWO_HOLDERS_HEADER
        + "def _take_all(things: Any, text: str) -> str:\n"
        + '    return text.splitlines()[0] + "".join(thing.suffix() for thing in things)\n\n\n'
        + "MANY = functools.partial(_take_all, [First(), Second(), First(), Second()])\n\n\n"
        + "@stage\ndef use(text: str) -> str:\n    return MANY(text)\n\n\n"
        + 'with Pipeline("t") as pipeline:\n    use(pipeline.input("lines.txt", t=str))\n'
    )
    project = _project(tmp_path, {"factory.py": _FACTORY_MODULE, "pipeline.py": pipeline})
    output = project / "data" / "t" / "use.txt"

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "Local Set Default Proof UsingAZAZ"
    assert _repro_statuses(project, home) == {"t/use": "cached"}
    _edit(project, "pipeline.py", "First(), Second()])", "Second(), First()])")

    assert _repro_statuses(project, home) == {"t/use": "ran"}, "The swap should rerun"
    assert output.read_text() == "Local Set Default Proof UsingAZZA"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


def test_replacing_a_factory_class_at_an_equal_set_rank_reruns_its_stage(
    tmp_path: pathlib.Path,
) -> None:
    """A partial holds identity-distinct instances with equal forms in a frozenset. Changing
    `[First, First, Second]` to `[First, Second, Second]` keeps every instance at rank `s0`.
    The held class keys keep each rank's occurrence count, so the stage reruns with the new
    output."""
    home = tmp_path / "home"
    pipeline = (
        _TWO_HOLDERS_HEADER
        + "def _take_all(things: Any, text: str) -> str:\n"
        + '    return text.splitlines()[0] + "".join(sorted(thing.suffix() for thing in things))\n\n\n'
        + "MANY = functools.partial(_take_all, frozenset((First(), First(), Second())))\n\n\n"
        + "@stage\ndef use(text: str) -> str:\n    return MANY(text)\n\n\n"
        + 'with Pipeline("t") as pipeline:\n    use(pipeline.input("lines.txt", t=str))\n'
    )
    project = _project(tmp_path, {"factory.py": _FACTORY_MODULE, "pipeline.py": pipeline})
    output = project / "data" / "t" / "use.txt"

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "Local Set Default Proof UsingAAZ"
    assert _repro_statuses(project, home) == {"t/use": "cached"}
    _edit(project, "pipeline.py", "First(), First(), Second()", "First(), Second(), Second()")

    assert _repro_statuses(project, home) == {"t/use": "ran"}, "The multiplicity edit should rerun"
    assert output.read_text() == "Local Set Default Proof UsingAZZ"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


_CALLBACK_CYCLE = """\
import dataclasses
from typing import Any, ClassVar


class Callback:
    __slots__ = ("callback",)

    def __init__(self, callback) -> None:
        self.callback = callback

    def read(self) -> str:
        return self.callback()


def class_callback() -> str:
    return CycleBox.marker


@dataclasses.dataclass(frozen=True)
class CycleBox:
    marker: ClassVar[str] = "safe"
    value: Any = Callback(class_callback)
"""

_CALLBACK_CYCLE_PIPELINE = """\
from pivot import Pipeline, stage

import cycle


@stage
def use(text: str) -> str:
    return text.splitlines()[0] + cycle.CycleBox().value.read()


with Pipeline("t") as pipeline:
    use(pipeline.input("lines.txt", t=str))
"""


def test_a_callback_that_reads_its_enclosing_class_runs_and_caches(
    tmp_path: pathlib.Path,
) -> None:
    """A callback in a dataclass default reads that dataclass's class variable. The fingerprint
    sees the class again as a back-reference, so `repro` runs once and caches instead of
    recursing."""
    home = tmp_path / "home"
    project = _project(
        tmp_path, {"cycle.py": _CALLBACK_CYCLE, "pipeline.py": _CALLBACK_CYCLE_PIPELINE}
    )

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert (project / "data" / "t" / "use.txt").read_text() == "Local Set Default Proof Usingsafe"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


def test_a_callback_whose_wrapped_chain_loops_runs_and_caches(
    tmp_path: pathlib.Path,
) -> None:
    """A callback in a dataclass default has `__wrapped__` pointing to itself. The walk reads it
    from its own code, so `repro` runs the stage, with no `ValueError` from `inspect.unwrap`, and
    the next run reuses the result."""
    home = tmp_path / "home"
    looping = _CALLBACK_CYCLE.replace(
        "def class_callback() -> str:\n    return CycleBox.marker\n",
        'def class_callback() -> str:\n    return "safe"\n\n\nclass_callback.__wrapped__ = class_callback\n',
    )
    project = _project(tmp_path, {"cycle.py": looping, "pipeline.py": _CALLBACK_CYCLE_PIPELINE})

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert (project / "data" / "t" / "use.txt").read_text() == "Local Set Default Proof Usingsafe"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


_CONFIG_HELD = """\
import json
import pathlib

VALUE = json.loads((pathlib.Path(__file__).parent / "config.json").read_text())["v"]


class Held:
    def value(self) -> str:
        return str(VALUE)
"""

_CONFIG_PIPELINE = """\
import functools
from typing import Any

from pivot import Pipeline, stage

import held


def _read(thing: Any, text: str) -> str:
    return text.splitlines()[0] + thing.value()


HELD = functools.partial(_read, held.Held())


@stage
def use(text: str) -> str:
    return HELD(text)


with Pipeline("t") as pipeline:
    use(pipeline.input("lines.txt", t=str))
"""


def test_a_refusal_found_on_a_cache_hit_refuses_the_stage_warm_and_fresh(
    tmp_path: pathlib.Path,
) -> None:
    """A captured class's method reads `VALUE`, which `held.py` loads from `config.json`, so no
    source edit changes it. While it is a string the stage runs and caches. Once it is a list,
    the walk refuses it: on the warm path, where the manifest cache's check walks the class
    again and refuses, the stage walk that follows refuses too (the refused walk left nothing
    in a memo), and so it does on the fresh path, with no state database. The output stays the
    one the safe value made."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "held.py": _CONFIG_HELD,
            "config.json": '{"v": "safe"}\n',
            "pipeline.py": _CONFIG_PIPELINE,
        },
    )
    output = project / "data" / "t" / "use.txt"
    line = "Local Set Default Proof Using"

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert _repro_statuses(project, home) == {"t/use": "cached"}
    _edit(project, "config.json", '"safe"', '["unsafe-A"]')
    for path in ("warm", "fresh"):
        if path == "fresh":
            shutil.rmtree(project / ".pivot" / "state.lmdb")
        result = helpers.run_pivot(project, home, "repro", "--json")
        message = " ".join((result.stdout + result.stderr).split())
        assert result.returncode != 0, f"The {path} path should refuse: {message}"
        assert "captures non-scalar variable 'VALUE'" in message, message
        assert output.read_text() == f"{line}safe"


# ---------------------------------------------------------------------------------------------
# Enum values, type expressions, deep classes and values, user code the walk must not run, and a
# StageParams `model_config` a method reads
# ---------------------------------------------------------------------------------------------

_ENUM_FROM_SETTINGS = """\
import dataclasses
import enum

import settings


class Choice(enum.Enum):
    PICK = settings.VALUE


@dataclasses.dataclass(frozen=True)
class Holder:
    choice: Choice = Choice.PICK
"""

_ENUM_PIPELINE = """\
import functools

from pivot import Pipeline, stage

import choices


def _value(choice, text: str) -> str:
    return text.splitlines()[0] + choice.value


HELD = functools.partial(_value, choices.Choice.PICK)


@stage
def use(text: str) -> str:
    return {body}


with Pipeline("t") as pipeline:
    use(pipeline.input("lines.txt", t=str))
"""

# Where the stage meets `Choice.PICK`: held by a partial it calls, or as the class-level default
# of a dataclass it names.
_ENUM_BODIES = {
    "partial-holds-the-member": "HELD(text)",
    "default-of-a-named-dataclass": "_value(choices.Holder().choice, text)",
}


@pytest.mark.parametrize("body", sorted(_ENUM_BODIES))
def test_an_edit_to_the_value_of_an_enum_member_a_value_holds_reruns_its_stage(
    tmp_path: pathlib.Path, body: str
) -> None:
    """`Choice.PICK` takes its value from `settings.VALUE`. The class statement holds the
    expression, not the value, so the member is formed with its value: an edit to `settings.py`
    reruns the stage once, and the next run reuses it."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "settings.py": 'VALUE = "A"\n',
            "choices.py": _ENUM_FROM_SETTINGS,
            "pipeline.py": _ENUM_PIPELINE.format(body=_ENUM_BODIES[body]),
        },
    )
    output = project / "data" / "t" / "use.txt"

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "Local Set Default Proof UsingA"
    assert _repro_statuses(project, home) == {"t/use": "cached"}
    _edit(project, "settings.py", '"A"', '"B"')
    assert _repro_statuses(project, home) == {"t/use": "ran"}, "The value edit should rerun"
    assert output.read_text() == "Local Set Default Proof UsingB"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


_TYPE_EXPRESSIONS_PIPELINE = """\
import dataclasses
import functools
import pathlib
import typing
from typing import Any

from pivot import Pipeline, loaders, stage


def _take(value: Any, text: str) -> str:
    return text.splitlines()[0]


GENERIC = functools.partial(_take, list[int])
OPTIONAL = functools.partial(_take, typing.Optional[str])
UNION = functools.partial(_take, str | None)
Made = dataclasses.make_dataclass(
    "Made", [("items", list[str]), ("note", typing.Optional[str], None)]
)
Row = typing.NamedTuple("Row", [("x", list[str])])
Rec = typing.TypedDict("Rec", {"x": list[str]})


@dataclasses.dataclass(frozen=True)
class TypedReader(loaders.Reader[str]):
    schema: Any = None

    def load(self, path: pathlib.Path) -> str:
        return path.read_text().splitlines()[0]


@stage
def generic(text: str) -> str:
    return GENERIC(text)


@stage
def optional(text: str) -> str:
    return OPTIONAL(text)


@stage
def union(text: str) -> str:
    return UNION(text)


@stage
def made(text: str) -> str:
    return text.splitlines()[0] + str(len(Made(["a"]).items))


@stage
def row(text: str) -> str:
    return text.splitlines()[0] + str(len(Row(["a"]).x))


@stage
def rec(text: str) -> str:
    return text.splitlines()[0] + str(len(Rec(x=["a"])["x"]))


@stage
def configured(text: str) -> str:
    return text


with Pipeline("t") as pipeline:
    for each in (generic, optional, union, made, row, rec):
        each(pipeline.input("lines.txt", t=str))
    configured(pipeline.input("typed.txt", format=TypedReader(list[str])))
"""


def test_a_type_expression_inside_a_value_runs_and_caches(tmp_path: pathlib.Path) -> None:
    """`list[int]`, `Optional[str]` and `str | None` held by partials, the field types of a
    `make_dataclass`, call-made `NamedTuple` and `TypedDict` class, and a loader config field
    holding `list[str]` are formed by their origin and arguments, not refused as proxies: every
    stage runs, and the next run reuses each."""
    home = tmp_path / "home"
    project = _project(tmp_path, {"pipeline.py": _TYPE_EXPRESSIONS_PIPELINE})
    (project / "data" / "raw" / "typed.txt").write_text(_LINES)
    stages = ("configured", "generic", "made", "optional", "rec", "row", "union")

    assert _repro_statuses(project, home) == {f"t/{name}": "ran" for name in stages}
    assert _repro_statuses(project, home) == {f"t/{name}": "cached" for name in stages}


_DEEP = """\
class Node:
    def __init__(self, nxt) -> None:
        self.nxt = nxt


def _chain(depth: int) -> type:
    cls = type("C0", (), {"marker": "A"})
    for index in range(1, depth):
        cls = type(f"C{index}", (cls,), {})
    return cls


def _linked(depth: int) -> object:
    head = None
    for _ in range(depth):
        head = Node(head)
    return head


def _lists(depth: int) -> object:
    value = []
    for _ in range(depth):
        value = [value]
    return value


def _dicts(depth: int) -> object:
    value = {}
    for _ in range(depth):
        value = {"k": value}
    return value


DEEP_MRO = _chain(1100)
LINKED = _linked(1000)
LISTS = _lists(1000)
DICTS = _dicts(1000)
"""

_DEEP_PIPELINE = """\
import functools
from typing import Any

from pivot import Pipeline, stage

import deep


def _take(value: Any, text: str) -> str:
    return text.splitlines()[0]


MRO = functools.partial(_take, deep.DEEP_MRO())
LINKED = functools.partial(_take, deep.LINKED)
LISTS = functools.partial(_take, deep.LISTS)
DICTS = functools.partial(_take, deep.DICTS)


@stage
def captured_mro(text: str) -> str:
    return MRO(text)


@stage
def named_mro(text: str) -> str:
    return text.splitlines()[0] + deep.DEEP_MRO().marker


@stage
def linked(text: str) -> str:
    return LINKED(text)


@stage
def lists(text: str) -> str:
    return LISTS(text)


@stage
def dicts(text: str) -> str:
    return DICTS(text)


with Pipeline("t") as pipeline:
    for each in (captured_mro, named_mro, linked, lists, dicts):
        each(pipeline.input("lines.txt", t=str))
"""


def test_classes_and_values_deeper_than_the_recursion_limit_run_and_cache(
    tmp_path: pathlib.Path,
) -> None:
    """A class made by `type()` on a chain of 1,100 bases, held as an instance or named by the
    stage, and a linked list of 1,000 objects, lists nested 1,000 deep and dicts nested 1,000 deep,
    each held by a partial: class identities nest no deeper than a fixed limit and values are
    formed on an explicit stack, so each stage runs, and the next run reuses it."""
    home = tmp_path / "home"
    project = _project(tmp_path, {"deep.py": _DEEP, "pipeline.py": _DEEP_PIPELINE})
    stages = ("captured_mro", "dicts", "linked", "lists", "named_mro")

    assert _repro_statuses(project, home) == {f"t/{name}": "ran" for name in stages}
    assert _repro_statuses(project, home) == {f"t/{name}": "cached" for name in stages}


# Each hook writes its name to `hooks.log` once the module has built its values, so a hook the
# fingerprint runs leaves the file behind.
_HOOK_LOG = """\
_ARMED = False


def _ran(name):
    if _ARMED:
        with (pathlib.Path(__file__).parent / "hooks.log").open("a") as log:
            log.write(name + "\\n")
"""

_HOOKS_ACCEPTED = (
    "import functools\nimport pathlib\nimport types\n\nimport numpy\n\n"
    + _HOOK_LOG
    + """

class Key:
    def __hash__(self):
        _ran("Key.__hash__")
        return 1

    def __eq__(self, other):
        _ran("Key.__eq__")
        return self is other


class KeyDict(dict):
    pass


class KeySet(set):
    pass


class KeyFrozenSet(frozenset):
    pass


class ItemsDict(dict):
    def items(self):
        _ran("ItemsDict.items")
        return dict.items(self)

    def keys(self):
        _ran("ItemsDict.keys")
        return dict.keys(self)

    def __iter__(self):
        _ran("ItemsDict.__iter__")
        return dict.__iter__(self)

    def __getitem__(self, key):
        _ran("ItemsDict.__getitem__")
        return dict.__getitem__(self, key)


class Scalar(numpy.float64):
    @property
    def dtype(self):
        _ran("Scalar.dtype")
        return numpy.dtype("float64")

    def tobytes(self, order="C"):
        _ran("Scalar.tobytes")
        return numpy.float64.tobytes(self, order)


@functools.lru_cache(maxsize=8)
def cached(value):
    return value * 2


DICT = KeyDict({Key(): 1})
SET = KeySet([Key(), Key()])
FROZEN = KeyFrozenSet([Key(), Key()])
PROXY = types.MappingProxyType(ItemsDict(a=1))
SCALAR = Scalar(1.5)
CACHED = cached
_ARMED = True
"""
)

_HOOKS_ACCEPTED_PIPELINE = """\
import functools
from typing import Any

from pivot import Pipeline, stage

import hooks


def _tag(thing: Any, text: str) -> str:
    return text.splitlines()[0] + type(thing).__name__


DICT = functools.partial(_tag, hooks.DICT)
SET = functools.partial(_tag, hooks.SET)
FROZEN = functools.partial(_tag, hooks.FROZEN)
PROXY = functools.partial(_tag, hooks.PROXY)
SCALAR = functools.partial(_tag, hooks.SCALAR)
CACHED = functools.partial(_tag, hooks.CACHED)


@stage
def dict_subclass(text: str) -> str:
    return DICT(text)


@stage
def set_subclass(text: str) -> str:
    return SET(text)


@stage
def frozenset_subclass(text: str) -> str:
    return FROZEN(text)


@stage
def mapping_proxy(text: str) -> str:
    return PROXY(text)


@stage
def numpy_scalar(text: str) -> str:
    return SCALAR(text)


@stage
def lru_cache(text: str) -> str:
    return CACHED(text)


with Pipeline("t") as pipeline:
    for each in (dict_subclass, set_subclass, frozenset_subclass, mapping_proxy, numpy_scalar,
                 lru_cache):
        each(pipeline.input("lines.txt", t=str))
"""


def test_a_value_is_formed_without_running_the_code_of_its_members_or_subclass(
    tmp_path: pathlib.Path,
) -> None:
    """Partials hold a dict, set and frozenset subclass whose members' `__hash__` and `__eq__`
    are user code, a `MappingProxyType` over a dict subclass whose `items`, `keys`,
    `__iter__` and `__getitem__` are, a numpy scalar subclass whose `dtype` and `tobytes` are,
    and an `lru_cache` wrapper. Each is formed through the built-in types' own C methods and
    descriptors and the cache's own closure. The first `repro` sends the stages to workers, which
    pickle their values and so may run the hooks; its log is removed. After `pivot fingerprint
    reset`, `pivot status` fingerprints the stages cold, and the next `repro` checks their cached
    manifests: no hook runs in either (`hooks.log` is not written), and every stage is cached."""
    home = tmp_path / "home"
    project = _project(
        tmp_path, {"hooks.py": _HOOKS_ACCEPTED, "pipeline.py": _HOOKS_ACCEPTED_PIPELINE}
    )
    stages = (
        "dict_subclass",
        "frozenset_subclass",
        "lru_cache",
        "mapping_proxy",
        "numpy_scalar",
        "set_subclass",
    )
    log = project / "hooks.log"

    assert _repro_statuses(project, home) == {f"t/{name}": "ran" for name in stages}
    log.unlink(missing_ok=True)
    for command in (("fingerprint", "reset"), ("status", "--json")):
        result = helpers.run_pivot(project, home, *command)
        assert result.returncode == 0, result.stdout + result.stderr
    assert not log.exists(), f"Fingerprinting ran user code: {log.read_text().split()}"
    assert _repro_statuses(project, home) == {f"t/{name}": "cached" for name in stages}
    assert not log.exists(), f"A cache hit ran user code: {log.read_text().split()}"


_HOOKS_REFUSED = (
    "import functools\nimport pathlib\n\n"
    + _HOOK_LOG
    + """

def _double(value):
    return value * 2


REPLACED = functools.lru_cache(maxsize=8)(_double)


def _parameters():
    _ran("cache_parameters")
    return {"maxsize": 8, "typed": False}


REPLACED.cache_parameters = _parameters


class Descriptor:
    def __get__(self, instance, owner=None):
        _ran("Descriptor.__get__")
        return self

    def __call__(self):
        return "d"


class Forwards:
    def __getattr__(self, name):
        _ran("Forwards.__getattr__")
        raise AttributeError(name)


class Described:
    DESC = Descriptor()

    def unused(self) -> str:
        return self.DESC()


class Forwarded:
    FWD = Forwards()

    def unused(self) -> str:
        return str(self.FWD)


class Meta(type):
    def __getattribute__(cls, name):
        _ran("Meta.__getattribute__")
        return type.__getattribute__(cls, name)


class Hooked(metaclass=Meta):
    def suffix(self) -> str:
        return "A"


_ARMED = True
"""
)

_HOOKS_REFUSED_PIPELINE = """\
import functools
from typing import Any

from pivot import Pipeline, stage

import hooks


def _tag(thing: Any, text: str) -> str:
    return text


HELD = {held}


@stage
def use(text: str) -> str:
    return {body}


with Pipeline("t") as pipeline:
    use(pipeline.input("lines.txt", t=str))
"""

_HELD_CALL = "HELD(text)"
# What a partial holds (or `None` when the stage names the class), the stage's body, and what
# the refusal says. Reading each value would run user code: a replaced `cache_parameters`, a
# descriptor's `__get__` and a class attribute's `__getattr__` that a method reads through
# `self`, and a metaclass `__getattribute__`.
_REFUSED_HOOKS = {
    "replaced-cache-parameters": (
        "functools.partial(_tag, hooks.REPLACED)",
        _HELD_CALL,
        "'pipeline.HELD.args[0]' keeps state Pivot cannot read (type: functools._lru_cache_wrapper)",
    ),
    "descriptor-read-through-self": (
        "functools.partial(_tag, hooks.Described())",
        _HELD_CALL,
        "captures non-scalar variable 'hooks.Described.DESC' (type: Descriptor)",
    ),
    "forwarder-read-through-self": (
        "functools.partial(_tag, hooks.Forwarded())",
        _HELD_CALL,
        "captures non-scalar variable 'hooks.Forwarded.FWD' (type: Forwards)",
    ),
    "metaclass-hook-held": (
        "functools.partial(_tag, hooks.Hooked())",
        _HELD_CALL,
        "class 'hooks.Hooked' has the metaclass 'hooks.Meta'",
    ),
    "metaclass-hook-named": (
        "None",
        "text + hooks.Hooked().suffix()",
        "class 'hooks.Hooked' has the metaclass 'hooks.Meta'",
    ),
}


@pytest.mark.parametrize("shape", sorted(_REFUSED_HOOKS))
def test_a_value_whose_reading_would_run_user_code_is_refused_by_name(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """Where a value cannot be read through C descriptors (an `lru_cache` wrapper whose
    `cache_parameters` was replaced, a class attribute a method reads through `self` that is a
    descriptor or answers lookups in `__getattr__`, a class whose metaclass defines
    `__getattribute__`), the stage is refused with a message naming it, and the code is not run
    (`hooks.log` is not written)."""
    held, body, expected = _REFUSED_HOOKS[shape]
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "hooks.py": _HOOKS_REFUSED,
            "pipeline.py": _HOOKS_REFUSED_PIPELINE.format(held=held, body=body),
        },
    )

    result = helpers.run_pivot(project, home, "repro", "--json")
    message = " ".join((result.stdout + result.stderr).split())
    assert result.returncode != 0, f"The stage should be refused: {message}"
    assert expected in message, message
    log = project / "hooks.log"
    assert not log.exists(), f"The fingerprint ran user code: {log.read_text().split()}"


_CONFIG_PARAMS = """\
import pydantic

from pivot import StageParams

import settings


class Params(StageParams):
    model_config = pydantic.ConfigDict(title=settings.TITLE)
    scale: int = 2

    def title(self) -> str:
        return str(self.model_config.get("title"))
"""

_CONFIG_PARAMS_PIPELINE = """\
from pivot import Pipeline, stage

import params


@stage
def use(text: str, settings: params.Params = params.Params()) -> str:
    return text.splitlines()[0] + settings.title()


with Pipeline("t") as pipeline:
    use(pipeline.input("lines.txt", t=str))
"""


def test_an_edit_to_the_model_config_a_stage_params_method_reads_reruns_its_stage(
    tmp_path: pathlib.Path,
) -> None:
    """A `StageParams` method reads `self.model_config`, whose `title` comes from
    `settings.TITLE`. The schema hash leaves the title out, and the config is a dict, which a
    class attribute read through `self` would refuse; it is recorded by its form instead: the stage
    runs and caches, an edit to `settings.py` reruns it, and the next run reuses it."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "settings.py": 'TITLE = "T1"\n',
            "params.py": _CONFIG_PARAMS,
            "pipeline.py": _CONFIG_PARAMS_PIPELINE,
        },
    )
    output = project / "data" / "t" / "use.txt"

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert output.read_text() == "Local Set Default Proof UsingT1"
    assert _repro_statuses(project, home) == {"t/use": "cached"}
    _edit(project, "settings.py", '"T1"', '"T2"')
    assert _repro_statuses(project, home) == {"t/use": "ran"}, "The config edit should rerun"
    assert output.read_text() == "Local Set Default Proof UsingT2"
    assert _repro_statuses(project, home) == {"t/use": "cached"}


_WRAPPED_PAIR = """\
def first():
    return first.__wrapped__()


def second():
    return "G" + settings.VALUE


first.__wrapped__ = second
second.__wrapped__ = first
VALUE = first
"""

_SLOTTED_NUMPY_SCALAR = """\
import numpy


class Scalar(numpy.float64):
    __slots__ = ("tag",)

    def __new__(cls, value, tag):
        item = super().__new__(cls, value)
        item.tag = tag
        return item

    def __reduce__(self):
        return Scalar, (float(self), self.tag)


VALUE = Scalar(1, settings.VALUE)
"""

_ROUND_SIX_VALUE_PIPELINE = """\
import functools

from pivot import Pipeline, stage

import values


def read(value, text):
    return {read}


HELD = functools.partial(read, values.VALUE)


@stage
def use(text: str) -> str:
    return HELD(text)


with Pipeline("t") as pipeline:
    use(pipeline.input("lines.txt", t=str))
"""


@pytest.mark.parametrize(
    ("definition", "read", "before", "after"),
    [
        pytest.param(_WRAPPED_PAIR, "value()", "GA", "GB", id="wrapped-pair"),
        pytest.param(_SLOTTED_NUMPY_SCALAR, "value.tag", "A", "B", id="numpy-slot"),
    ],
)
def test_an_edit_behind_a_wrapped_cycle_or_in_a_numpy_slot_reruns_its_stage(
    tmp_path: pathlib.Path, definition: str, read: str, before: str, after: str
) -> None:
    """A cache hit must follow every Python wrapper and form a scalar subclass's slots."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "settings.py": 'VALUE = "A"\n',
            "values.py": "import settings\n\n" + definition,
            "pipeline.py": _ROUND_SIX_VALUE_PIPELINE.format(read=read),
        },
    )
    output = project / "data/t/use.txt"
    assert _repro_statuses(project, home, "1") == {"t/use": "ran"}
    assert output.read_text() == before
    assert _repro_statuses(project, home, "1") == {"t/use": "cached"}
    _edit(project, "settings.py", '"A"', '"B"')
    assert _repro_statuses(project, home, "7") == {"t/use": "ran"}
    assert output.read_text() == after
    assert _repro_statuses(project, home, "7") == {"t/use": "cached"}
    reset = helpers.run_pivot(project, home, "fingerprint", "reset")
    assert reset.returncode == 0, reset.stdout + reset.stderr
    assert _repro_statuses(project, home, "1") == {"t/use": "cached"}


@pytest.mark.parametrize("slotted", [False, True], ids=["method", "slot"])
def test_a_user_generic_alias_subclass_is_refused_instead_of_losing_its_code_or_state(
    tmp_path: pathlib.Path, slotted: bool
) -> None:
    home = tmp_path / "home"
    definition = (
        '    __slots__ = ("tag",)\n\n'
        '    def __call__(self):\n        return object.__getattribute__(self, "tag")\n'
        if slotted
        else "    def __call__(self):\n        return settings.VALUE\n"
    )
    project = _project(
        tmp_path,
        {
            "settings.py": 'VALUE = "A"\n',
            "values.py": "import types\nimport settings\n\nclass Alias(types.GenericAlias):\n"
            + definition
            + "\nVALUE = Alias(list, (int,))\n"
            + ("VALUE.tag = settings.VALUE\n" if slotted else ""),
            "pipeline.py": _ROUND_SIX_VALUE_PIPELINE.format(read="value()"),
        },
    )

    refused = helpers.run_pivot(project, home, "status" if slotted else "repro")

    message = refused.stdout + refused.stderr
    assert refused.returncode != 0, message
    assert "Alias" in message and "StageParams" in message, message
    assert "state" in message, message


@pytest.mark.parametrize("factory", [False, True], ids=["type-call", "class-statement"])
def test_a_long_ring_of_captured_classes_runs_and_caches(
    tmp_path: pathlib.Path, factory: bool
) -> None:
    make = (
        "def make():\n    class Made:\n        pass\n    return Made\n"
        if factory
        else 'def make():\n    return type("Made", (), {})\n'
    )
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "values.py": make
            + "\nclasses = [make() for _ in range(200)]\n"
            + "for i, cls in enumerate(classes):\n    cls.peer = classes[(i + 1) % len(classes)]\n"
            + "VALUE = classes[0]()\n",
            "pipeline.py": _ROUND_SIX_VALUE_PIPELINE.format(read='"safe"'),
        },
    )

    assert _repro_statuses(project, home, "1") == {"t/use": "ran"}
    assert (project / "data/t/use.txt").read_text() == "safe"
    assert _repro_statuses(project, home, "7") == {"t/use": "cached"}


@pytest.mark.parametrize("container", ["ordered", "subclass", "proxy"])
def test_reading_ordered_dict_order_runs_no_user_key_hash(
    tmp_path: pathlib.Path, container: str
) -> None:
    values = {
        "ordered": "collections.OrderedDict([(Key(), 1)])",
        "subclass": "OrderedSub([(Key(), 1)])",
        "proxy": "types.MappingProxyType(collections.OrderedDict([(Key(), 1)]))",
    }
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "values.py": "import collections\nimport pathlib\nimport types\n"
            + _HOOK_LOG
            + "\nclass Key:\n"
            + '    def __hash__(self):\n        _ran("hash")\n        return 1\n'
            + "\nclass OrderedSub(collections.OrderedDict):\n    pass\n"
            + f"\nVALUE = {values[container]}\n_ARMED = True\n",
            "pipeline.py": _ROUND_SIX_VALUE_PIPELINE.format(read='"safe"'),
        },
    )

    result = helpers.run_pivot(project, home, "status")

    message = result.stdout + result.stderr
    assert not (project / "hooks.log").exists(), message
    assert result.returncode != 0, message
    refused_type = "OrderedSub" if container == "subclass" else "OrderedDict"
    assert refused_type in message and "StageParams" in message, message


def test_a_top_level_loader_with_a_metaclass_attribute_hook_is_refused_before_class_reads(
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "values.py": "import dataclasses\nimport pathlib\nfrom pivot import loaders\n"
            + _HOOK_LOG
            + "\nclass Meta(type(loaders.Text)):\n"
            + "    def __getattribute__(cls, name):\n"
            + '        _ran("metaclass:" + name)\n'
            + "        return type.__getattribute__(cls, name)\n"
            + "\n@dataclasses.dataclass(frozen=True)\n"
            + "class Reader(loaders.Text, metaclass=Meta):\n    pass\n"
            + "\nVALUE = Reader()\n",
            "pipeline.py": "from pivot import Pipeline, stage\nimport values\n"
            + "\n@stage\ndef use(text: str) -> str:\n    return text\n"
            + '\nwith Pipeline("t") as pipeline:\n'
            + '    use(pipeline.input("lines.txt", format=values.VALUE))\n'
            + "\nvalues._ARMED = True\n",
        },
    )

    refused = helpers.run_pivot(project, home, "repro")

    message = refused.stdout + refused.stderr
    assert refused.returncode != 0, message
    assert "Reader" in message and "Meta" in message and "StageParams" in message, message
    assert not (project / "hooks.log").exists(), message


def test_schema_fingerprinting_releases_transient_models_after_its_pass(
    tmp_path: pathlib.Path,
) -> None:
    """A long-lived process must not retain a model through a cached schema or its validator."""
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "values.py": "import gc\nimport weakref\nimport pydantic\n"
            + "from pivot import fingerprint\n"
            + "\ndef retained():\n    refs = []\n"
            + "    for i in range(3):\n"
            + "        with fingerprint.fingerprint_pass():\n"
            + '            model = pydantic.create_model("Transient", value=(int, i))\n'
            + "            refs.append(weakref.ref(model))\n"
            + "            fingerprint._schema_hash(model)\n"
            + "        del model\n"
            + "    gc.collect()\n"
            + "    return str(sum(ref() is not None for ref in refs))\n",
            "pipeline.py": "from pivot import Pipeline, stage\nimport values\n"
            + "\n@stage\ndef use(text: str) -> str:\n    return values.retained()\n"
            + '\nwith Pipeline("t") as pipeline:\n'
            + '    use(pipeline.input("lines.txt", t=str))\n',
        },
    )

    assert _repro_statuses(project, home) == {"t/use": "ran"}
    assert (project / "data/t/use.txt").read_text() == "0"


@pytest.mark.parametrize(
    ("definition", "read", "expected"),
    [
        pytest.param(
            "import collections.abc\nVALUE = collections.abc.Callable[[int], str]\n",
            "text.splitlines()[0]",
            "Local Set Default Proof Using",
            id="stdlib-callable-alias",
        ),
        pytest.param(
            "import json\n\ndef encode(value):\n    return json.dumps(value)\n"
            "encode.__wrapped__ = json.dumps\nVALUE = encode\n",
            "value(text.splitlines()[0])",
            '"Local Set Default Proof Using"',
            id="library-wrapped-target",
        ),
    ],
)
def test_a_stdlib_alias_or_library_wrapped_target_runs_and_caches(
    tmp_path: pathlib.Path, definition: str, read: str, expected: str
) -> None:
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "values.py": definition,
            "pipeline.py": _ROUND_SIX_VALUE_PIPELINE.format(read=read),
        },
    )

    assert _repro_statuses(project, home, "1") == {"t/use": "ran"}
    assert (project / "data/t/use.txt").read_text() == expected
    assert _repro_statuses(project, home, "7") == {"t/use": "cached"}


@pytest.mark.parametrize("wrapper", ["cache", "lru_cache", "partial"])
def test_a_mixed_python_and_c_wrapper_cycle_tracks_each_function(
    tmp_path: pathlib.Path, wrapper: str
) -> None:
    constructors = {
        "cache": "functools.cache(second)",
        "lru_cache": "functools.lru_cache(maxsize=8)(second)",
        "partial": "functools.partial(second)",
    }
    definition = (
        "import functools\nimport settings\n\n"
        "def first():\n    return first.__wrapped__()\n\n"
        'def second():\n    return "G" + settings.VALUE\n\n'
        f"CACHED = {constructors[wrapper]}\n"
        'CACHED.__name__ = CACHED.__qualname__ = "CACHED"\n'
        "first.__wrapped__ = CACHED\nsecond.__wrapped__ = first\n"
        + ("CACHED.__wrapped__ = first\n" if wrapper == "partial" else "")
        + "VALUE = first\n"
    )
    home = tmp_path / "home"
    project = _project(
        tmp_path,
        {
            "settings.py": 'VALUE = "A"\n',
            "values.py": definition,
            "pipeline.py": _ROUND_SIX_VALUE_PIPELINE.format(read="value()"),
        },
    )
    output = project / "data/t/use.txt"

    assert _repro_statuses(project, home, "1") == {"t/use": "ran"}
    assert output.read_text() == "GA"
    assert _repro_statuses(project, home, "1") == {"t/use": "cached"}
    _edit(project, "settings.py", '"A"', '"B"')
    assert _repro_statuses(project, home, "7") == {"t/use": "ran"}
    assert output.read_text() == "GB"
    assert _repro_statuses(project, home, "7") == {"t/use": "cached"}
