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

import email.mime.text
import importlib.machinery
import importlib.util
import json
import multiprocessing.process
import os
import pathlib
import re
import subprocess
import sys
import uuid
from typing import TYPE_CHECKING

import pytest
import yaml
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


def _held(
    name: str, path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, text: str = _PLAIN
) -> types.ModuleType:
    """Load `path` (holding `text`) as module `name` into `sys.modules`, as the program that runs
    Pivot did."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
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


def test_a_partial_over_a_library_callable_the_program_holds_fails_the_fingerprint(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage calls a `functools.partial` over `wi_lib.plain`: cloudpickle sends the partial
    with `plain` by name, so the worker imports `wi_lib`, the project's file in its place."""
    held = tmp_path / "lib" / "site-packages" / "wi_lib.py"
    _held("wi_lib", held, monkeypatch)
    pipeline = _discover(
        tmp_path,
        monkeypatch,
        {"wi_lib.py": _PLAIN},
        _pipeline(
            "import functools\n\nfrom wi_lib import plain\n\nPLAIN = functools.partial(plain)",
            "PLAIN()",
        ),
    )

    with pytest.raises(exceptions.PivotError, match=f"module 'wi_lib' is {_resolved(held)}"):
        pipeline.ensure_fingerprint("p/use")


def test_a_library_submodule_a_stage_imports_from_a_project_package_is_recorded_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The project's `wi_up/__init__.py` adds a `site-packages` directory to its `__path__`, and
    the stage runs `from wi_up import sub`, a library module there that nothing loaded. The
    worker imports `wi_up.sub` by name, so the walk names it (`byname:`)."""
    site = tmp_path / "lib" / "site-packages" / "wi_up"
    site.mkdir(parents=True)
    (site / "sub.py").write_text(_PLAIN)
    pipeline = _discover(
        tmp_path,
        monkeypatch,
        {"wi_up/__init__.py": f"__path__.append({str(site)!r})\n"},
        "import wi_up\nfrom pivot import Pipeline, stage\n\n\n@stage\ndef use() -> str:\n"
        + "    from wi_up import sub\n\n    return sub.plain()\n\n\n"
        + 'with Pipeline("p") as pipeline:\n    use()\n',
    )

    assert _walk(pipeline).get("byname:wi_up.sub") == "", _walk(pipeline)


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


def test_a_pipeline_fingerprinted_after_a_later_discovery_found_nothing_is_still_compared(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The program loaded its own `wi_late` from `program/` by file, and `other/` on its
    `sys.path` has another `wi_late.py`. It discovers the project, then a directory with no
    pipeline, which ends the project's import session, and then fingerprints the project's stage:
    a worker would import `other/wi_late.py`, so the fingerprint still fails."""
    _held("wi_late", tmp_path / "program" / "wi_late.py", monkeypatch)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "wi_late.py").write_text(_PLAIN)
    monkeypatch.syspath_prepend(str(tmp_path / "other"))
    pipeline = _discover(tmp_path, monkeypatch, {}, _pipeline("import wi_late", "wi_late.plain()"))
    (tmp_path / "empty").mkdir()
    assert discovery.discover_pipeline(project_root=tmp_path / "empty") is None

    other = _resolved(tmp_path / "other" / "wi_late.py")
    with pytest.raises(exceptions.PivotError, match=f"would import {other}"):
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


def test_a_library_submodule_a_worker_imports_from_the_same_file_needs_no_new_interpreter(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """The stage reads `email.mime.text`, which this process holds from the standard library. A
    worker imports it from the same file, found in its package's directories, so the prediction
    agrees without starting an interpreter."""
    assert pathlib.Path(email.mime.text.__file__).name == "text.py"
    confirm = mocker.spy(loaded_source, "_confirm")
    pipeline = _discover(
        tmp_path,
        monkeypatch,
        {},
        _pipeline("from email.mime import text as mime_text", "mime_text.__name__"),
    )

    pipeline.ensure_fingerprint("p/use")

    assert "byname:email.mime.text" in _walk(pipeline)
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

    loaded_source.check_worker_imports({"module:__main__": ""}, ())


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
        loaded_source.check_worker_imports({"module:wi_ns": ""}, ())
        return
    held = _resolved(tmp_path / held_portion)
    with pytest.raises(exceptions.StageDefinitionError, match=f"namespace package in {held} in"):
        loaded_source.check_worker_imports({"module:wi_ns": ""}, ())


_LOADER_READS_A_MODULE = """\
import dataclasses
import pathlib
from typing import Annotated, TypedDict

import wi_loader
from pivot import Pipeline, loaders, stage


@dataclasses.dataclass(frozen=True)
class Located(loaders.Loader[str, str]):
    def load(self, path: pathlib.Path) -> str:
        return wi_loader.plain()

    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(wi_loader.plain())


class Fields(TypedDict):
    FIELD: Annotated[str, OUTPUT]


@stage
def use(raw: str) -> RETURNS:
    return VALUE


with Pipeline("p") as pipeline:
    use(pipeline.input("raw.txt", format=INPUT))
"""


@pytest.mark.parametrize(
    ("held_in", "input_loader", "output_loader", "field"),
    [
        pytest.param("lib/site-packages", "Located()", "loaders.Text()", None, id="input"),
        pytest.param("lib/site-packages", "loaders.Text()", "Located()", None, id="output"),
        *(
            pytest.param("program", "loaders.Text()", "Located()", field, id=f"user-{field}")
            for field in ("plain", "byname", "module")
        ),
    ],
)
def test_a_loaders_module_named_by_name_is_compared(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    held_in: str,
    input_loader: str,
    output_loader: str,
    field: str | None,
) -> None:
    """An input's or output's loader reads module `wi_loader`, which the program holds from
    another file than the root's `wi_loader.py`: a library from `site-packages` (a `byname:`
    entry), or its own user module (a `module:` entry under the output's prefix, which the
    stage's output keys place, for a field named `plain`, `byname` or `module` alike). The worker
    runs the loader too, so the module is compared like one the stage reads."""
    held_file = tmp_path / held_in / "wi_loader.py"
    _held("wi_loader", held_file, monkeypatch)
    returns = "Annotated[str, OUTPUT]" if field is None else "Fields"
    value = "raw" if field is None else f"{{{field!r}: raw}}"
    pipeline = _discover(
        tmp_path,
        monkeypatch,
        {"wi_loader.py": _PLAIN},
        _LOADER_READS_A_MODULE.replace("RETURNS", returns)
        .replace("VALUE", value)
        .replace("FIELD", field or "unused")
        .replace("INPUT", input_loader)
        .replace("OUTPUT", output_loader),
    )
    held = _resolved(held_file)

    with pytest.raises(exceptions.PivotError, match=rf"module 'wi_loader' is {held}"):
        pipeline.ensure_fingerprint("p/use")


_INNER_LOADER = """\
import dataclasses
import pathlib

from pivot import loaders


@dataclasses.dataclass(frozen=True)
class Inner(loaders.Reader[str]):
    def load(self, path: pathlib.Path) -> str:
        return __file__
"""

_LOADER_HOLDS_AN_INSTANCE = """\
import dataclasses
import pathlib

import wi_inst
from pivot import Pipeline, loaders, stage


@dataclasses.dataclass(frozen=True)
class Outer(loaders.Reader[str]):
    inner: loaders.Reader[str]

    def load(self, path: pathlib.Path) -> str:
        return self.inner.load(path)


@stage
def use(raw: str) -> str:
    return raw


with Pipeline("p") as pipeline:
    use(pipeline.input("raw.txt", format=Outer(wi_inst.Inner())))
"""


def test_the_module_of_a_loader_another_loader_holds_is_compared(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The input's loader holds the loader `wi_inst.Inner()`, which the program holds from its
    own `wi_inst.py` while the root has another. A worker runs `Inner.load`, importing `wi_inst`
    by name, so the module the held loader's walk covers is compared like one the stage reads."""
    held = tmp_path / "program" / "wi_inst.py"
    _held("wi_inst", held, monkeypatch, _INNER_LOADER)
    pipeline = _discover(
        tmp_path, monkeypatch, {"wi_inst.py": _INNER_LOADER}, _LOADER_HOLDS_AN_INSTANCE
    )

    with pytest.raises(exceptions.PivotError, match=rf"module 'wi_inst' is {_resolved(held)}"):
        pipeline.ensure_fingerprint("p/use")


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

    loaded_source.check_worker_imports({"byname:wi_cwd": ""}, ())


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
    loaded_source.check_worker_imports({"byname:wi_moved": ""}, ())
    sys.path.insert(0, str(tmp_path / "other"))

    other = _resolved(tmp_path / "other" / "wi_moved.py")
    with pytest.raises(exceptions.StageDefinitionError, match=f"would import {other}"):
        loaded_source.check_worker_imports({"byname:wi_moved": ""}, ())


def test_a_relative_path_entry_is_predicted_in_the_working_directory_of_the_check(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, hold_modules: Callable[[], None]
) -> None:
    """`sys.path` has the relative entry `rel`, and the program holds `first/rel/wi_rl.py`. A check
    in `first` finds nothing wrong; after a `chdir` to `second`, a new worker would start there
    and import `second/rel/wi_rl.py`, though this process's finder for `rel` still searches
    `first/rel`."""
    for directory in ("first", "second"):
        (tmp_path / directory / "rel").mkdir(parents=True)
        (tmp_path / directory / "rel" / "wi_rl.py").write_text(_PLAIN)
    _held("wi_rl", tmp_path / "first" / "rel" / "wi_rl.py", monkeypatch)
    monkeypatch.setattr(sys, "path_importer_cache", dict(sys.path_importer_cache))
    monkeypatch.setattr(sys, "path", ["rel", *sys.path])
    monkeypatch.chdir(tmp_path / "first")
    hold_modules()
    # The program's own imports made the finder for `rel` here, in `first`.
    assert importlib.machinery.PathFinder.find_spec("wi_rl", ["rel"]) is not None
    loaded_source.check_worker_imports({"byname:wi_rl": ""}, ())
    monkeypatch.chdir(tmp_path / "second")

    second = _resolved(tmp_path / "second" / "rel" / "wi_rl.py")
    with pytest.raises(exceptions.StageDefinitionError, match=f"would import {second}"):
        loaded_source.check_worker_imports({"byname:wi_rl": ""}, ())


def test_a_path_object_in_sys_path_is_skipped_as_the_import_system_skips_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`PathFinder` skips an entry of `sys.path` that is not a `str`, in a worker as here: with
    `pathlib.Path(program)` before `other`, a worker imports `other/wi_pobj.py`, not the
    program's `wi_pobj`."""
    program = tmp_path / "program"
    _held("wi_pobj", program / "wi_pobj.py", monkeypatch)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "wi_pobj.py").write_text(_PLAIN)
    monkeypatch.setattr(sys, "path", [program, str(tmp_path / "other"), *sys.path])
    pipeline = _discover(tmp_path, monkeypatch, {}, _pipeline("import wi_pobj", "wi_pobj.plain()"))

    other = _resolved(tmp_path / "other" / "wi_pobj.py")
    with pytest.raises(exceptions.PivotError, match=f"would import {other}"):
        pipeline.ensure_fingerprint("p/use")


def test_a_confirmation_is_kept_per_sys_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The confirming interpreter's answer for `wi_ck` is kept for the `sys.path` it searched; a
    new `sys.path` asks again."""
    monkeypatch.setattr(loaded_source, "_confirmed", {})
    base = list(sys.path)
    reports = list[object]()
    for directory in ("first", "second"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "wi_ck.py").write_text(_PLAIN)
        monkeypatch.setattr(sys, "path", [str(tmp_path / directory), *base])
        results, failure = loaded_source._confirm(["wi_ck"])
        assert failure is None
        reports.append(results["wi_ck"])

    assert reports == [
        ("file", str((tmp_path / directory / "wi_ck.py").resolve()))
        for directory in ("first", "second")
    ]


def test_a_confirmation_that_cannot_run_fails_on_the_prediction(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the confirming interpreter fails, the prediction decides, and the error says the
    confirmation did not run."""
    monkeypatch.setattr(loaded_source, "_CONFIRM_PROGRAM", "raise SystemExit(3)\n")
    _held("wi_fail", tmp_path / "program" / "wi_fail.py", monkeypatch)
    pipeline = _discover(
        tmp_path,
        monkeypatch,
        {"wi_fail.py": _PLAIN},
        _pipeline("import wi_fail", "wi_fail.plain()"),
    )

    with pytest.raises(exceptions.PivotError, match=r"could not start an interpreter.+\(exit 3"):
        pipeline.ensure_fingerprint("p/use")


def test_a_lock_keeps_every_entry_of_a_dependency_or_output_named_byname() -> None:
    """Only a top-level `byname:` key is a by-name entry: a loader's join the stage's own without a
    prefix. Each `dep:` and `out:` entry is the lock's, also of a dependency parameter or an
    output field named `byname`."""
    named_byname = {
        "dep:byname:loader:pipeline.Reads:load": "2",
        "out:p/use:byname:tag": "data",
        "out:p/use:byname:loader:pipeline.Writes:save": "3",
    }
    manifest = {"self:use": "1", "byname:json": "", **named_byname}

    assert fingerprint.without_by_name(manifest) == {"self:use": "1", **named_byname}
    assert fingerprint.by_name_modules(manifest, {"p/use:byname"}) == {"json"}


# Through a worker -----------------------------------------------------------------------------


_SITECUSTOMIZE = (
    "import wi_startup  # noqa: F401 - in every interpreter of this environment\n"
    + 'print("sitecustomize: wi_startup imported")  # startup output on stdout\n'
)


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


def test_the_suspects_of_a_session_are_confirmed_in_one_interpreter(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """Three stages each read a module a `sitecustomize` imports, each shadowed by a root file.
    The first stage's check confirms every held suspect at once, so the session starts one
    interpreter, not one per stage."""
    names = ("wi_s0", "wi_s1", "wi_s2")
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "sitecustomize.py").write_text(
        "".join(f"import {name}  # noqa: F401\n" for name in names)
    )
    for name in names:
        _held(name, startup / f"{name}.py", monkeypatch)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(startup), os.environ.get("PYTHONPATH", "")])
    )
    stages = "".join(
        f"@stage\ndef use_{name}() -> str:\n    return {name}.plain()\n\n\n" for name in names
    )
    calls = "".join(f"    use_{name}()\n" for name in names)
    pipeline = _discover(
        tmp_path,
        monkeypatch,
        {f"{name}.py": _PLAIN for name in names},
        f"import {', '.join(names)}\nfrom pivot import Pipeline, stage\n\n\n{stages}"
        + f'with Pipeline("p") as pipeline:\n{calls}',
    )
    run = mocker.spy(subprocess, "run")

    for name in names:
        pipeline.ensure_fingerprint(f"p/use_{name}")

    started = [
        call for call in run.call_args_list if loaded_source._CONFIRM_PROGRAM in call.args[0]
    ]
    assert len(started) == 1, started


def test_the_check_is_reported_with_the_stages_other_refusals(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage reads a module-level list and the program's `wi_ag`: one error names both, as
    every fingerprinting refusal of a stage is reported together."""
    _held("wi_ag", tmp_path / "program" / "wi_ag.py", monkeypatch)
    pipeline = _discover(
        tmp_path,
        monkeypatch,
        {"wi_ag.py": _PLAIN},
        _pipeline("import wi_ag\n\nITEMS = [1, 2]", "wi_ag.plain() + str(ITEMS)"),
    )

    with pytest.raises(exceptions.PivotError) as raised:
        pipeline.ensure_fingerprint("p/use")

    assert "captures non-scalar variable 'ITEMS'" in str(raised.value)
    assert _ANOTHER_FILE in str(raised.value)


def test_a_relative_startup_path_names_a_directory_under_the_working_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`PYTHONPATH=startup` is relative: a worker starts in this process's working directory and
    finds `startup/sitecustomize.py` there, which imports `wi_rel`. The confirming interpreter
    starts there too, so nothing fails."""
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "sitecustomize.py").write_text("import wi_rel  # noqa: F401\n")
    _held("wi_rel", startup / "wi_rel.py", monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["startup", os.environ.get("PYTHONPATH", "")]))
    pipeline = _discover(
        tmp_path, monkeypatch, {"wi_rel.py": _PLAIN}, _pipeline("import wi_rel", "wi_rel.plain()")
    )

    assert "module:wi_rel" in pipeline.ensure_fingerprint("p/use")


def test_a_module_a_worker_imports_before_it_gets_its_path_is_the_one_it_uses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker imports `uuid` from the standard library while loky starts it, before it gets the
    `sys.path` with the root first, so the root's `uuid.py` is not what the stage uses there.
    The confirming interpreter starts the same way, so nothing fails."""
    assert pathlib.Path(uuid.__file__).name == "uuid.py"
    pipeline = _discover(
        tmp_path,
        monkeypatch,
        {"uuid.py": _PLAIN},
        _pipeline("import uuid", "str(uuid.NAMESPACE_DNS)"),
    )

    pipeline.ensure_fingerprint("p/use")

    assert "byname:uuid" in _walk(pipeline)


# Mapped stages (Pipeline.map) ------------------------------------------------------------------


def _mapped_pipeline(module: str, *, mapped: bool = True, static: bool = True) -> str:
    """The producer `p/plan` writes a directory; with `mapped`, `p/cut` is mapped over it, and
    with `static`, the static stage `p/use` reads all of it. Both read module `module` as a
    global."""
    return (
        f"from typing import Annotated\n\nimport {module}\n"
        + "from pivot import Pipeline, directory, loaders, stage\n\n\n"
        + "@stage\ndef plan(\n    entries: dict[str, int],\n"
        + ") -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:\n"
        + '    return {f"{name}.yaml": {"value": value} for name, value in entries.items()}\n\n\n'
        + f"@stage\ndef cut(entry: dict[str, int]) -> str:\n    return {module}.plain()\n\n\n"
        + "@stage\ndef use(entries: dict[str, dict[str, int]]) -> str:\n"
        + f"    return {module}.plain()\n\n\n"
        + 'with Pipeline("p") as pipeline:\n'
        + '    planned = plan(pipeline.input("plan.yaml", t=dict[str, int]))\n'
        + ("    cut(pipeline.map(planned))\n" if mapped else "")
        + ("    use(planned)\n" if static else "")
    )


_PLAN = {"data/raw/plan.yaml": "a: 1\n"}


def test_an_expanded_instance_whose_module_is_another_file_fails_as_a_static_stage_does(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The program holds its own `wi_mp`, and the root has another. The static stage `p/use`
    and the instance `p/cut@a` both read it as a global, and each fingerprint fails with the
    same refusal: an instance is fingerprinted as any stage is."""
    held = tmp_path / "program" / "wi_mp.py"
    _held("wi_mp", held, monkeypatch)
    pipeline = _discover(
        tmp_path, monkeypatch, {"wi_mp.py": _PLAIN, **_PLAN}, _mapped_pipeline("wi_mp")
    )
    assert pipeline.expand_template("p/cut", [("a", "a.yaml")]) == ["p/cut@a"]

    with pytest.raises(exceptions.PivotError, match=_ANOTHER_FILE) as static:
        pipeline.ensure_fingerprint("p/use")
    with pytest.raises(exceptions.PivotError, match=_ANOTHER_FILE) as instance:
        pipeline.ensure_fingerprint("p/cut@a")

    assert f"module 'wi_mp' is {held.resolve()} in this process" in str(instance.value)
    assert str(instance.value) == str(static.value).replace("'p/use'", "'p/cut@a'")


def test_an_expanded_instance_records_its_by_name_modules_outside_the_lock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The instance reads the library `wi_ml`, which the root has no file for, so its worker
    imports the program's: the walk names it (`byname:`), and the instance's fingerprint, which
    its lock records, does not, as for a static stage."""
    site = tmp_path / "lib" / "site-packages"
    _held("wi_ml", site / "wi_ml.py", monkeypatch)
    monkeypatch.syspath_prepend(str(site))
    pipeline = _discover(tmp_path, monkeypatch, _PLAN, _mapped_pipeline("wi_ml"))
    pipeline.expand_template("p/cut", [("a", "a.yaml")])

    walked = fingerprint.get_stage_fingerprint(pipeline.get_stage("p/cut@a")["func"])
    recorded = {name: pipeline.ensure_fingerprint(name) for name in ("p/use", "p/cut@a")}

    assert walked.get("byname:wi_ml") == "", walked
    for name, entries in recorded.items():
        assert not any("byname:" in key for key in entries), (name, entries)


_MAPPED_OUTPUT_LOADER = """\
import dataclasses
import pathlib
from typing import Annotated

import wi_mq
from pivot import Pipeline, directory, loaders, stage


@dataclasses.dataclass(frozen=True)
class Located(loaders.Loader[str, str]):
    def load(self, path: pathlib.Path) -> str:
        return wi_mq.plain()

    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(wi_mq.plain())


@stage
def plan(
    entries: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in entries.items()}


@stage
def cut(entry: dict[str, int]) -> Annotated[str, Located()]:
    return str(entry["value"])


with Pipeline("p") as pipeline:
    planned = plan(pipeline.input("plan.yaml", t=dict[str, int]))
    cut(pipeline.map(planned))
"""


def test_an_expanded_instances_output_loader_module_is_compared(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output loader of the mapped stage `p/cut` reads the project's `wi_mq`, which the
    program holds from its own file: the instance's fingerprint fails as a static stage's does.
    The loader's entries are placed under the instance's own output key (`out:p/cut@a:`), so the
    check reads them only when the instance's output keys are the ones passed (#546)."""
    held = tmp_path / "program" / "wi_mq.py"
    _held("wi_mq", held, monkeypatch)
    pipeline = _discover(
        tmp_path, monkeypatch, {"wi_mq.py": _PLAIN, **_PLAN}, _MAPPED_OUTPUT_LOADER
    )
    assert pipeline.expand_template("p/cut", [("a", "a.yaml")]) == ["p/cut@a"]

    with pytest.raises(exceptions.PivotError, match=_ANOTHER_FILE) as instance:
        pipeline.ensure_fingerprint("p/cut@a")

    assert f"module 'wi_mq' is {held.resolve()} in this process" in str(instance.value)


@pytest.mark.parametrize(
    ("mapped", "stage", "output"),
    [
        pytest.param(False, "p/use", "use.txt", id="static-stage"),
        pytest.param(True, "p/cut@a", "cut@a.txt", id="mapped-instance"),
    ],
)
def test_a_stage_after_its_producer_whose_module_is_another_file_fails_the_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mapped: bool, stage: str, output: str
) -> None:
    """The producer runs; the run then reaches the stage that reads the program's `wi_mr`, the
    static `p/use` or the instance `p/cut@a` the run adds, and fails at its fingerprint the same
    way: no worker runs the root's `wi_mr` under a fingerprint of the program's."""
    _held("wi_mr", tmp_path / "program" / "wi_mr.py", monkeypatch)
    _discover(
        tmp_path,
        monkeypatch,
        {"wi_mr.py": _PLAIN, **_PLAN},
        _mapped_pipeline("wi_mr", mapped=mapped, static=not mapped),
    )
    data = tmp_path / "project" / "data" / "p"

    # The engine's task group wraps the refusal of a stage that starts after another finishes
    refusal = pytest.RaisesExc(exceptions.PivotError, match=f"Stage '{stage}'.*{_ANOTHER_FILE}")
    with pytest.RaisesGroup(refusal, allow_unwrapped=True):
        executor.run()

    assert (data / "plan" / "a.yaml").exists()
    assert not (data / output).exists()


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


_NAMED_BYNAME = """\
import dataclasses
import json
import pathlib
from typing import Annotated, TypedDict

from pivot import Pipeline, loaders, stage


@dataclasses.dataclass(frozen=True)
class Reads(loaders.Loader[dict, dict]):
    def load(self, path: pathlib.Path) -> dict:
        return {"read": 1, **json.loads(path.read_text())}

    def save(self, data: dict, path: pathlib.Path) -> None:
        path.write_text(json.dumps(data))


@dataclasses.dataclass(frozen=True)
class Writes(loaders.Loader[dict, dict]):
    def load(self, path: pathlib.Path) -> dict:
        return json.loads(path.read_text())

    def save(self, data: dict, path: pathlib.Path) -> None:
        path.write_text(json.dumps({"written": 1, **data}))


class Out(TypedDict):
    byname: Annotated[dict, Writes()]


@stage
def use(byname: dict) -> Out:
    return {"byname": byname}


with Pipeline("p") as pipeline:
    use(pipeline.input("raw.json", format=Reads()))
"""


@pytest.mark.parametrize(
    ("edit", "edited"),
    [
        pytest.param(
            ('"read": 1', '"read": 2'), {"written": 1, "read": 2, "x": 0}, id="dependency"
        ),
        pytest.param(
            ('"written": 1', '"written": 2'), {"written": 2, "read": 1, "x": 0}, id="output"
        ),
    ],
)
def test_a_dependency_and_an_output_named_byname_rerun_when_their_loader_changes(
    tmp_path: pathlib.Path, edit: tuple[str, str], edited: dict[str, int]
) -> None:
    """Stage `p/use` reads dependency parameter `byname` and writes output field `byname`. The
    lock records the entries of each one's loader, which spell `byname` after `dep:` and the
    output's name, so an edit to either loader reruns the stage."""
    root = _cli_project(tmp_path, {"pipeline.py": _NAMED_BYNAME})
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw" / "raw.json").write_text('{"x": 0}')
    _pivot(root, "repro")
    (output,) = [path for path in (root / "data" / "p").rglob("*") if path.is_file()]
    assert json.loads(output.read_text()) == {"written": 1, "read": 1, "x": 0}
    lock = yaml.safe_load((root / ".pivot" / "stages" / "p" / "use.lock").read_text())
    recorded = {key for key in lock["code_manifest"] if "byname" in key}
    assert recorded == {
        "wire:byname",
        "wire:byname:path",
        "dep:byname:func:pipeline.Reads.load",
        "dep:byname:func:pipeline.Reads.save",
        "dep:byname:module:pipeline",
        "dep:byname:loader:pipeline.Reads:load",
        "dep:byname:loader:pipeline.Reads:save",
        "out:p/use:byname:tag",
        "out:p/use:byname:func:pipeline.Writes.load",
        "out:p/use:byname:func:pipeline.Writes.save",
        "out:p/use:byname:module:pipeline",
        "out:p/use:byname:loader:pipeline.Writes:load",
        "out:p/use:byname:loader:pipeline.Writes:save",
    }, recorded

    (root / "pipeline.py").write_text(_NAMED_BYNAME.replace(*edit))
    _pivot(root, "repro")

    assert json.loads(output.read_text()) == edited


def test_a_lock_records_no_byname_entry_of_a_user_loader_that_reads_a_library(
    tmp_path: pathlib.Path,
) -> None:
    """Stage `p/use` reads dependency `plain` and writes output field `plain`, each through a
    user loader whose code reads library module `json` (a `byname:json` entry of its walk). The
    lock records each loader's entries under its prefix but no `byname:` entry, under a prefix
    or not: a lock written before those entries existed stays current."""
    root = _cli_project(tmp_path, {"pipeline.py": _NAMED_BYNAME.replace("byname", "plain")})
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw" / "raw.json").write_text('{"x": 0}')

    _pivot(root, "repro")

    lock = yaml.safe_load((root / ".pivot" / "stages" / "p" / "use.lock").read_text())
    recorded = set(lock["code_manifest"])
    assert {
        "dep:plain:loader:pipeline.Reads:load",
        "out:p/use:plain:loader:pipeline.Writes:save",
    } <= recorded, recorded
    assert [key for key in recorded if "byname" in key] == [], recorded
