"""A module the program that runs Pivot holds must be the module a stage's worker imports.

A worker imports by name each module discovery did not send by value, with the project root
first on its `sys.path`. A program that runs Pivot in its own process (a notebook, a script, a
test) can hold a module of that name from another file, which `import` in `pipeline.py` returns
from `sys.modules`; the stage would be fingerprinted from that module while its worker runs the
other one, and an edit to the project's file would read `cached`. Fingerprinting fails instead
(`loaded_source.check_worker_imports`): a cheap prediction, confirmed in a new interpreter
started as a worker starts before it raises.

Each test names its modules uniquely: the test process puts `packages/pivot/tests` (with its own
`helpers.py`) on `sys.path`.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import multiprocessing.process
import os
import pathlib
import re
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from tests import helpers

from pivot import discovery, exceptions, executor, fingerprint, loaded_source, project

if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Generator

    from pytest_mock import MockerFixture

    from pivot.registry import PipelineLike

_PLAIN = "def plain() -> str:\n    return __file__\n"
_ANOTHER_FILE = "in this process, but a worker would import"


def _pipeline(imports: str, call: str) -> str:
    return (
        f"{imports}\nfrom pivot import Pipeline, stage\n\n\n@stage\ndef use() -> str:\n"
        + f'    return {call}\n\n\nwith Pipeline("p") as pipeline:\n    use()\n'
    )


def _held(name: str, path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load `path` as module `name` into `sys.modules`, as the program that runs Pivot did."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PLAIN)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _held_namespace(
    name: str, directory: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> types.ModuleType:
    """A namespace package `name` whose one portion is `directory`, held in `sys.modules`."""
    directory.mkdir(parents=True, exist_ok=True)
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = [str(directory)]
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _resolved(path: pathlib.Path) -> str:
    return re.escape(str(path.resolve()))


def _walk(pipeline: PipelineLike) -> dict[str, str]:
    """What the walk records for stage `p/use`, entries a lock leaves out included."""
    return fingerprint.get_stage_fingerprint(pipeline.get_stage("p/use")["func"])


def _discover(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str], pipeline: str
) -> PipelineLike:
    root = tmp_path / "project"
    for rel_path, text in {**files, "pipeline.py": pipeline}.items():
        (root / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (root / rel_path).write_text(text)
    (root / ".git").mkdir(exist_ok=True)
    (root / ".pivot").mkdir(exist_ok=True)
    monkeypatch.setattr(project, "_project_root_cache", root)
    loaded = discovery.discover_pipeline(project_root=root)
    assert loaded is not None
    return loaded


@pytest.fixture
def hold_modules() -> Generator[Callable[[], None]]:
    """For a test that calls `check_worker_imports` itself: a call takes what `sys.modules` holds
    then as what the program held when its import session started, as a discovery does."""
    yield lambda: loaded_source.note_held_modules(list(sys.modules.items()))
    loaded_source.note_held_modules(())


# Modules the program holds from another file --------------------------------------------------


@pytest.mark.parametrize(
    ("held_file", "root_file", "worker"),
    [
        pytest.param("program/wi_a.py", "wi_a.py", "{root}/wi_a.py", id="outside-the-root"),
        pytest.param(
            "project/notebooks/wi_a.py", "wi_a.py", "{root}/wi_a.py", id="notebook-under-the-root"
        ),
        pytest.param(
            "program/wi_a/__init__.py",
            "wi_a/notes.txt",
            "a namespace package in {root}/wi_a",
            id="root-namespace-dir",
        ),
    ],
)
def test_a_module_the_program_holds_from_another_file_fails_the_stages_fingerprint(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    held_file: str,
    root_file: str,
    worker: str,
) -> None:
    """The program imported its own `wi_a` before it ran the pipeline, whose `import wi_a`
    returns it. A worker would import the root's `wi_a` (a namespace directory there is the only
    `wi_a` on its path). The fingerprint fails, naming both."""
    _held("wi_a", tmp_path / held_file, monkeypatch)
    pipeline = _discover(
        tmp_path, monkeypatch, {root_file: _PLAIN}, _pipeline("import wi_a", "wi_a.plain()")
    )

    with pytest.raises(exceptions.PivotError, match=_ANOTHER_FILE) as raised:
        pipeline.ensure_fingerprint("p/use")

    message = str(raised.value)
    root = (tmp_path / "project").resolve()
    assert f"module 'wi_a' is {(tmp_path / held_file).resolve()} in this process" in message
    assert f"a worker would import {worker.format(root=root)}" in message
    assert "rename one of the two modules" in message


_SUB_FILES = {"wi_pkg/__init__.py": "", "wi_pkg/sub.py": _PLAIN}


@pytest.mark.parametrize(
    ("held", "imports", "call"),
    [
        pytest.param(
            ("wi_pkg", "wi_pkg.sub"),
            "from wi_pkg import sub",
            "sub.plain()",
            id="package-and-submodule",
        ),
        pytest.param(
            ("wi_pkg.sub",), "from wi_pkg.sub import plain", "plain()", id="submodule-only"
        ),
    ],
)
def test_a_submodule_the_program_holds_from_another_file_fails_the_stages_fingerprint(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    held: tuple[str, ...],
    imports: str,
    call: str,
) -> None:
    """The program holds `wi_pkg.sub` from its own package, with or without that package. The
    project's `wi_pkg` has a `sub` too, which a worker would import by name: with only the
    submodule held, `from wi_pkg.sub import plain` takes it from `sys.modules` without importing
    the project's package, so nothing sends it by value."""
    files = {"wi_pkg": "program/wi_pkg/__init__.py", "wi_pkg.sub": "program/wi_pkg/sub.py"}
    for name in held:
        _held(name, tmp_path / files[name], monkeypatch)
    pipeline = _discover(tmp_path, monkeypatch, _SUB_FILES, _pipeline(imports, call))

    with pytest.raises(exceptions.PivotError, match=r"module 'wi_pkg\.sub' is .+sub\.py"):
        pipeline.ensure_fingerprint("p/use")


def test_a_submodule_the_program_holds_under_a_project_package_goes_to_the_worker_by_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The program holds only `wi_pkg.sub`, so `pipeline.py` imports the project's `wi_pkg`,
    which discovery sends by value with its submodules: the worker runs the program's `sub` that
    the fingerprint describes, and imports nothing by that name."""
    _held("wi_pkg.sub", tmp_path / "program" / "wi_pkg" / "sub.py", monkeypatch)
    pipeline = _discover(
        tmp_path, monkeypatch, _SUB_FILES, _pipeline("from wi_pkg import sub", "sub.plain()")
    )

    assert "module:wi_pkg.sub" in pipeline.ensure_fingerprint("p/use")


@pytest.mark.parametrize(
    ("imports", "call"),
    [
        pytest.param("import wi_lib", "wi_lib.plain()", id="module-global"),
        pytest.param("from wi_lib import plain", "plain()", id="library-callable"),
    ],
)
def test_a_library_the_program_holds_with_a_project_modules_name_fails_the_stages_fingerprint(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, imports: str, call: str
) -> None:
    """A library the program imported (from a `site-packages` directory) is recorded by name,
    read as a module global or as the module that defines a callable; a worker would import the
    project's `wi_lib.py` in its place."""
    held = tmp_path / "lib" / "site-packages" / "wi_lib.py"
    _held("wi_lib", held, monkeypatch)
    pipeline = _discover(tmp_path, monkeypatch, {"wi_lib.py": _PLAIN}, _pipeline(imports, call))

    with pytest.raises(exceptions.PivotError, match=f"module 'wi_lib' is {_resolved(held)}"):
        pipeline.ensure_fingerprint("p/use")


_BODY_IMPORT_PIPELINE = (
    "from pivot import Pipeline, stage\n\n\n@stage\ndef use() -> str:\n    import wi_lib\n\n"
    + '    return wi_lib.plain()\n\n\nwith Pipeline("p") as pipeline:\n    use()\n'
)


def test_a_library_the_program_holds_and_a_stage_imports_in_its_body_fails_the_fingerprint(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage imports `wi_lib` in its body: the worker runs that import with the root first
    on its path and gets the project's `wi_lib.py`, not the library the program holds."""
    held = tmp_path / "lib" / "site-packages" / "wi_lib.py"
    _held("wi_lib", held, monkeypatch)
    pipeline = _discover(tmp_path, monkeypatch, {"wi_lib.py": _PLAIN}, _BODY_IMPORT_PIPELINE)

    with pytest.raises(exceptions.PivotError, match=f"module 'wi_lib' is {_resolved(held)}"):
        pipeline.ensure_fingerprint("p/use")


def test_a_library_module_read_as_a_global_is_recorded_by_name_outside_the_lock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its code is library code, not hashed, but a worker imports it by name: the walk names it
    (`byname:`), so the check and the manifest cache see it. The stage's fingerprint, which its
    lock records, does not, so locks written before this entry existed stay current. The root
    has no `wi_lib`, so a worker imports the program's, and nothing fails."""
    site = tmp_path / "lib" / "site-packages"
    _held("wi_lib", site / "wi_lib.py", monkeypatch)
    monkeypatch.syspath_prepend(str(site))
    pipeline = _discover(tmp_path, monkeypatch, {}, _pipeline("import wi_lib", "wi_lib.plain()"))

    recorded = pipeline.ensure_fingerprint("p/use")
    walked = _walk(pipeline)

    assert walked.get("byname:wi_lib") == "", walked
    assert not any(key.startswith("func:wi_lib") for key in walked), walked
    assert not any("byname:" in key for key in recorded), recorded


def test_a_namespace_child_the_program_holds_from_a_library_fails_the_stages_fingerprint(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The program holds `wi_shared.library` from a namespace package in `site-packages`; the
    project's `wi_shared` namespace directory has its own `library.py`. The module that defines
    `plain` is compared, not only the top-level namespace."""
    site = tmp_path / "lib" / "site-packages"
    _held_namespace("wi_shared", site / "wi_shared", monkeypatch)
    _held("wi_shared.library", site / "wi_shared" / "library.py", monkeypatch)
    pipeline = _discover(
        tmp_path,
        monkeypatch,
        {"wi_shared/library.py": _PLAIN},
        _pipeline("from wi_shared.library import plain", "plain()"),
    )
    held = _resolved(site / "wi_shared" / "library.py")

    with pytest.raises(exceptions.PivotError, match=rf"module 'wi_shared\.library' is {held}"):
        pipeline.ensure_fingerprint("p/use")


# Modules a worker imports from the same place --------------------------------------------------


def test_a_module_the_program_holds_from_the_projects_file_fingerprints(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """The program imported the project's own `wi_own.py`: a worker gets the same code."""
    confirm = mocker.spy(loaded_source, "_confirm")
    _held("wi_own", tmp_path / "project" / "wi_own.py", monkeypatch)
    pipeline = _discover(tmp_path, monkeypatch, {}, _pipeline("import wi_own", "wi_own.plain()"))

    assert "module:wi_own" in pipeline.ensure_fingerprint("p/use")
    assert confirm.call_count == 0


def test_a_built_in_module_a_root_file_names_needs_no_new_interpreter(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """`time` is built into the interpreter: a root `time.py` cannot shadow it, in a worker or
    here, so the prediction says so without starting an interpreter."""
    confirm = mocker.spy(loaded_source, "_confirm")
    pipeline = _discover(
        tmp_path,
        monkeypatch,
        {"time.py": _PLAIN},
        _pipeline("from time import gmtime", "str(gmtime(0).tm_year)"),
    )

    pipeline.ensure_fingerprint("p/use")
    assert "byname:time" in _walk(pipeline)
    assert confirm.call_count == 0


def test_a_frozen_module_needs_no_new_interpreter(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """`os` is frozen into this interpreter (as in a release build): the frozen importer, which
    comes before any file, finds it in a worker as here."""
    confirm = mocker.spy(loaded_source, "_confirm")
    pipeline = _discover(tmp_path, monkeypatch, {}, _pipeline("import os", "os.sep"))

    pipeline.ensure_fingerprint("p/use")
    assert "byname:os" in _walk(pipeline)
    assert confirm.call_count == 0


_FRONT_FINDER = """\
import importlib.abc
import importlib.util
import sys


class WiFront(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "wi_front":
            return importlib.util.spec_from_file_location("wi_front", {served!r})
        return None


sys.meta_path.insert(0, WiFront())
"""


def test_a_module_a_startup_finder_before_path_finder_serves_needs_no_new_interpreter(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """A finder at the front of `sys.meta_path`, installed at startup in every interpreter (here
    by a `sitecustomize`; meson-python's editable loader uses a `.pth` file), serves `wi_front`
    from outside the root. The prediction asks the finders in import order, so it finds the
    module a worker gets, though the root has a `wi_front.py`."""
    startup = tmp_path / "startup"
    served = startup / "served" / "wi_front.py"
    served.parent.mkdir(parents=True)
    served.write_text(_PLAIN)
    finder = _FRONT_FINDER.format(served=str(served))
    (startup / "sitecustomize.py").write_text(finder)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(startup), os.environ.get("PYTHONPATH", "")])
    )
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    exec(compile(finder, str(startup / "sitecustomize.py"), "exec"), {})  # as at startup
    monkeypatch.delitem(sys.modules, "wi_front", raising=False)
    importlib.import_module("wi_front")
    confirm = mocker.spy(loaded_source, "_confirm")
    pipeline = _discover(
        tmp_path,
        monkeypatch,
        {"wi_front.py": _PLAIN},
        _pipeline("import wi_front", "wi_front.plain()"),
    )

    assert "module:wi_front" in pipeline.ensure_fingerprint("p/use")
    assert confirm.call_count == 0


def test_a_module_the_program_reaches_through_a_symlink_is_its_real_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """The program imported `wi_link.py` through a symlink to its directory, which is on
    `sys.path` by its real name: the same file a worker imports, compared by resolved path."""
    confirm = mocker.spy(loaded_source, "_confirm")
    program = tmp_path / "program"
    program.mkdir()
    (tmp_path / "link").symlink_to(program, target_is_directory=True)
    _held("wi_link", tmp_path / "link" / "wi_link.py", monkeypatch)
    monkeypatch.syspath_prepend(str(program))
    pipeline = _discover(tmp_path, monkeypatch, {}, _pipeline("import wi_link", "wi_link.plain()"))

    assert "module:wi_link" in pipeline.ensure_fingerprint("p/use")
    assert confirm.call_count == 0


def test_a_held_module_replaced_during_the_session_is_not_the_programs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The program held `wi_swap` from a directory on `sys.path`, and `pipeline.py` bound it.
    Then `sys.modules` gets another file under that name. The fingerprint describes the module
    the pipeline bound, which is the file a worker imports, so nothing fails."""
    program = tmp_path / "program"
    _held("wi_swap", program / "wi_swap.py", monkeypatch)
    monkeypatch.syspath_prepend(str(program))
    pipeline = _discover(tmp_path, monkeypatch, {}, _pipeline("import wi_swap", "wi_swap.plain()"))
    _held("wi_swap", tmp_path / "other" / "wi_swap.py", monkeypatch)

    assert "module:wi_swap" in pipeline.ensure_fingerprint("p/use")


def test_the_programs_main_module_is_never_compared(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, hold_modules: Callable[[], None]
) -> None:
    """A worker receives the main module's functions by value and never imports it by name, so
    a `__main__.py` first on `sys.path` is not what it runs."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "__main__.py").write_text(_PLAIN)
    monkeypatch.syspath_prepend(str(root))
    _held("__main__", tmp_path / "program" / "run.py", monkeypatch)
    hold_modules()

    loaded_source.check_worker_imports({"module:__main__": ""})


# The comparison, one rule at a time ------------------------------------------------------------


@pytest.mark.parametrize(
    ("held_portion", "raises"),
    [
        pytest.param("program/wi_ns", True, id="other-directory"),
        pytest.param("project/wi_ns", False, id="same-directory"),
    ],
)
def test_namespace_packages_compare_by_their_directories(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    hold_modules: Callable[[], None],
    held_portion: str,
    raises: bool,
) -> None:
    """Two namespace packages are the same module only with the same directories."""
    (tmp_path / "project" / "wi_ns").mkdir(parents=True)
    monkeypatch.syspath_prepend(str(tmp_path / "project"))
    _held_namespace("wi_ns", tmp_path / held_portion, monkeypatch)
    hold_modules()

    if not raises:
        loaded_source.check_worker_imports({"module:wi_ns": ""})
        return
    held = _resolved(tmp_path / held_portion)
    with pytest.raises(exceptions.StageDefinitionError, match=f"namespace package in {held} in"):
        loaded_source.check_worker_imports({"module:wi_ns": ""})


def test_a_loaders_module_named_by_name_is_compared(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, hold_modules: Callable[[], None]
) -> None:
    """An input's or output's loader records its modules under `dep:`/`out:`; they run in the
    worker too."""
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "wi_loader.py").write_text(_PLAIN)
    monkeypatch.syspath_prepend(str(tmp_path / "project"))
    _held("wi_loader", tmp_path / "program" / "wi_loader.py", monkeypatch)
    hold_modules()

    with pytest.raises(exceptions.StageDefinitionError, match="module 'wi_loader'"):
        loaded_source.check_worker_imports({"dep:raw:byname:wi_loader": ""})


def test_an_empty_path_entry_is_the_directory_the_process_started_in(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, hold_modules: Callable[[], None]
) -> None:
    """Loky gives a worker `""` as the directory Pivot's process started in, not the directory it
    is in now, which has another `wi_cwd.py`."""
    started = tmp_path / "started"
    _held("wi_cwd", started / "wi_cwd.py", monkeypatch)
    (tmp_path / "now").mkdir()
    (tmp_path / "now" / "wi_cwd.py").write_text(_PLAIN)
    (tmp_path / "project").mkdir()
    monkeypatch.setattr(multiprocessing.process, "ORIGINAL_DIR", str(started))
    monkeypatch.chdir(tmp_path / "now")
    monkeypatch.setattr(sys, "path", [str(tmp_path / "project"), "", *sys.path])
    hold_modules()

    loaded_source.check_worker_imports({"byname:wi_cwd": ""})


def test_a_changed_sys_path_is_resolved_again(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, hold_modules: Callable[[], None]
) -> None:
    """The program puts a directory with another `wi_moved.py` first on `sys.path` after a check
    found nothing wrong: the next check resolves the module on the new path."""
    _held("wi_moved", tmp_path / "program" / "wi_moved.py", monkeypatch)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "wi_moved.py").write_text(_PLAIN)
    monkeypatch.setattr(sys, "path", [str(tmp_path / "program"), *sys.path])
    hold_modules()
    loaded_source.check_worker_imports({"byname:wi_moved": ""})
    sys.path.insert(0, str(tmp_path / "other"))

    other = _resolved(tmp_path / "other" / "wi_moved.py")
    with pytest.raises(exceptions.StageDefinitionError, match=f"would import {other}"):
        loaded_source.check_worker_imports({"byname:wi_moved": ""})


# Through a worker -----------------------------------------------------------------------------


_SITECUSTOMIZE = "import wi_startup  # noqa: F401 - in every interpreter of this environment\n"


def test_a_module_every_interpreter_imports_at_startup_is_the_one_its_worker_runs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """A `sitecustomize` imports `wi_startup` in every interpreter of the environment, workers
    included, before anything sets their `sys.path`, so the root's own `wi_startup.py` is never
    what they import. The prediction places it at the root; the new interpreter finds the
    startup module, so nothing fails, and the worker runs that file."""
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    held = startup / "wi_startup.py"
    _held("wi_startup", held, monkeypatch)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(startup), os.environ.get("PYTHONPATH", "")])
    )
    confirm = mocker.spy(loaded_source, "_confirm")
    _discover(
        tmp_path,
        monkeypatch,
        {"wi_startup.py": _PLAIN},
        _pipeline("import wi_startup", "wi_startup.plain()"),
    )

    results = executor.run()

    assert results["p/use"]["status"] == "ran", results
    output = tmp_path / "project" / "data" / "p" / "use.txt"
    assert pathlib.Path(output.read_text()).resolve() == held.resolve()
    assert confirm.call_count == 1


def test_a_module_from_another_file_fails_the_run_before_a_worker_runs_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run fails at fingerprinting: no worker runs the project's file under a fingerprint of
    the program's."""
    _held("wi_run", tmp_path / "program" / "wi_run.py", monkeypatch)
    _discover(
        tmp_path, monkeypatch, {"wi_run.py": _PLAIN}, _pipeline("import wi_run", "wi_run.plain()")
    )

    with pytest.raises(exceptions.PivotError, match=_ANOTHER_FILE):
        executor.run()

    assert not (tmp_path / "project" / "data" / "p" / "use.txt").exists()


# The `pivot` command ----------------------------------------------------------------------------


def _pivot(root: pathlib.Path, *args: str, env: dict[str, str] | None = None) -> None:
    proc = helpers.run_pivot(root, root.parent / "home", *args, extra_env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _cli_project(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _pivot(root, "init")
    for rel_path, text in files.items():
        (root / rel_path).write_text(text)
    return root


def test_a_root_file_named_like_a_built_in_module_runs_the_built_in_one(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot repro` with a root `time.py` runs the built-in `time`, here and in the worker."""
    root = _cli_project(
        tmp_path,
        {
            "time.py": _PLAIN,
            "pipeline.py": _pipeline("from time import gmtime", "str(gmtime(0).tm_year)"),
        },
    )

    _pivot(root, "repro")

    assert (root / "data" / "p" / "use.txt").read_text() == "1970"


def test_a_sitecustomize_module_with_a_root_files_name_runs_in_the_cli(
    tmp_path: pathlib.Path,
) -> None:
    """With a `sitecustomize` that imports an outside `wi_startup` in every interpreter, `pivot
    repro` and `pivot run` fingerprint that module, and their worker runs it."""
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    (startup / "wi_startup.py").write_text(_PLAIN)
    root = _cli_project(
        tmp_path,
        {
            "wi_startup.py": _PLAIN,
            "pipeline.py": _pipeline("import wi_startup", "wi_startup.plain()"),
        },
    )
    env = {"PYTHONPATH": os.pathsep.join([str(startup), os.environ.get("PYTHONPATH", "")])}
    output = root / "data" / "p" / "use.txt"

    _pivot(root, "repro", env=env)
    assert pathlib.Path(output.read_text()).resolve() == (startup / "wi_startup.py").resolve()
    output.unlink()
    _pivot(root, "run", "p/use", "--force", env=env)
    assert pathlib.Path(output.read_text()).resolve() == (startup / "wi_startup.py").resolve()
