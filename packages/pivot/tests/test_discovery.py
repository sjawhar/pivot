# pyright: reportMissingImports=false
from __future__ import annotations

import _imp
import contextlib
import importlib
import importlib.machinery
import importlib.util
import json
import linecache
import os
import pathlib
import py_compile
import subprocess
import sys
import types as _stdlib_types
from typing import TYPE_CHECKING

import cloudpickle
import pytest

import pivot_tui
from pivot import discovery, executor, fingerprint, loaded_source, project
from pivot.storage import state
from tests import conftest

if TYPE_CHECKING:
    from collections.abc import Generator


def test_find_config_in_dir_ignores_pivot_yaml(tmp_path: pathlib.Path) -> None:
    (tmp_path / "pivot.yaml").write_text("stages:\n  train:\n    python: train.py\n")

    assert discovery.find_config_in_dir(tmp_path) is None


def test_find_config_in_dir_returns_pipeline_py(tmp_path: pathlib.Path) -> None:
    pipeline_path = tmp_path / "pipeline.py"
    pipeline_path.write_text("pipeline = None\n")

    assert discovery.find_config_in_dir(tmp_path) == pipeline_path


def _write_pipeline_file(tmp_path: pathlib.Path, content: str) -> pathlib.Path:
    pipeline_path = tmp_path / "pipeline.py"
    pipeline_path.write_text(content)
    return pipeline_path


def test_load_pipeline_from_module_requires_protocol_methods(tmp_path: pathlib.Path) -> None:
    pipeline_path = _write_pipeline_file(
        tmp_path,
        "pipeline = object()\n",
    )

    with pytest.raises(discovery.DiscoveryError) as excinfo:
        discovery._load_pipeline_from_module(pipeline_path)

    message = str(excinfo.value)
    assert "missing required methods" in message
    for attr in [
        "name",
        "list_stages",
        "get_stage",
        "build_dag",
        "ensure_fingerprint",
        "snapshot",
    ]:
        assert attr in message


def test_load_pipeline_from_module_rejects_partial_pipeline_like(
    tmp_path: pathlib.Path,
) -> None:
    pipeline_path = _write_pipeline_file(
        tmp_path,
        """
class FakePipeline:
    name = "fake"

    def list_stages(self):
        return []

    def get_stage(self, name):
        raise KeyError(name)

    def build_dag(self):
        raise NotImplementedError

    def ensure_fingerprint(self, name):
        return {}

    def snapshot(self):
        return {}


pipeline = FakePipeline()
""",
    )

    with pytest.raises(discovery.DiscoveryError) as excinfo:
        discovery._load_pipeline_from_module(pipeline_path)

    message = str(excinfo.value)
    assert "missing required methods" in message


def test_discover_pipeline_fingerprints_helper_defined_in_pipeline(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline, stage


def normalize(value: str) -> str:
    return value.strip()


@stage
def preprocess(value: str) -> str:
    return normalize(value)


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    preprocess(pipeline.input("value.txt", t=str))
"""
    )

    pipeline = discovery.discover_pipeline(project_root=tmp_path)

    assert pipeline is not None
    assert "func:pipeline.normalize" in pipeline.ensure_fingerprint("test/preprocess")


def test_discover_root_pipeline_imports_sibling_stages_module(tmp_path: pathlib.Path) -> None:
    (tmp_path / "stages.py").write_text(
        """
from pivot.compose import stage


@stage
def clean(value: str) -> str:
    return value.strip()
"""
    )
    (tmp_path / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline
from stages import clean


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    clean(pipeline.input("value.txt", t=str))
"""
    )

    pipeline = discovery.discover_pipeline(project_root=tmp_path)

    assert pipeline is not None
    assert pipeline.list_stages() == ["test/clean"]


@pytest.mark.usefixtures("set_project_root")
def test_discover_subdirectory_pipeline_imports_package_helper(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report"
    report.mkdir()
    (report / "helpers.py").write_text(
        """
def clean(value: str) -> str:
    return value.strip()
"""
    )
    (report / "pipeline.py").write_text(
        """


from pivot.compose import Pipeline, stage
from report.helpers import clean


@stage
def preprocess(value: str) -> str:
    return clean(value)


with Pipeline("report") as pipeline:
    preprocess(pipeline.input("value.txt", t=str))
"""
    )
    monkeypatch.chdir(report)

    pipeline = discovery.discover_pipeline(project_root=tmp_path)

    assert pipeline is not None
    assert pipeline.list_stages() == ["report/preprocess"]
    assert "func:report.helpers.clean" in pipeline.ensure_fingerprint("report/preprocess")


@pytest.mark.usefixtures("set_project_root")
def test_discover_all_reuses_pipeline_imported_by_another_pipeline(
    tmp_path: pathlib.Path,
) -> None:
    prep = tmp_path / "prep"
    prep.mkdir()
    (prep / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline, stage

with (pathlib.Path(__file__).with_name("loads.txt")).open("a") as loads:
    loads.write("prep\\n")


@stage
def kept(value: str) -> str:
    return value


with Pipeline("prep") as pipeline:
    kept_handle = kept(pipeline.input("value.txt", t=str))
"""
    )
    report = tmp_path / "report"
    report.mkdir()
    (report / "pipeline.py").write_text(
        """


from pivot.compose import Pipeline, stage
from prep.pipeline import kept_handle


@stage
def consume(value: str) -> str:
    return value


with Pipeline("report") as pipeline:
    consume(kept_handle)
"""
    )

    pipeline = discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True)

    assert pipeline is not None
    assert set(pipeline.list_stages()) == {"prep/kept", "report/consume"}
    assert (prep / "loads.txt").read_text().splitlines() == ["prep"]


def test_discover_pipeline_replaces_modules_and_import_root_between_projects(
    tmp_path: pathlib.Path,
) -> None:
    """A later discovery does not reuse a same-named module from an earlier project."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root, value in ((first, "first"), (second, "second")):
        root.mkdir()
        (root / "stages.py").write_text(
            f"""
from pivot.compose import stage


@stage
def clean(value: str) -> str:
    return "{value}"
"""
        )
        (root / "pipeline.py").write_text(
            """
import pathlib

from pivot.compose import Pipeline
from stages import clean


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    clean(pipeline.input("value.txt", t=str))
"""
        )

    first_pipeline = discovery.discover_pipeline(project_root=first)
    second_pipeline = discovery.discover_pipeline(project_root=second)

    assert first_pipeline is not None
    assert second_pipeline is not None
    module_file = getattr(sys.modules["stages"], "__file__", None)
    assert isinstance(module_file, str)
    assert pathlib.Path(module_file).resolve() == (second / "stages.py")
    assert str(first) not in sys.path
    assert str(second) == sys.path[0]


_SHOUT_PIPELINE = """
from pivot.compose import Pipeline, stage
from shout_helpers import shout


@stage
def loud(text: str) -> str:
    return shout(text)


with Pipeline("first") as pipeline:
    loud(pipeline.input("words.txt", t=str))
"""


def _helper_fingerprinted_project(root: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `root` the current project, discover its pipeline and fingerprint its stage, as the
    engine does after discovery: the hashes and manifest that walk computes wait for a flush."""
    root.mkdir()
    (root / ".pivot").mkdir()
    (root / "shout_helpers.py").write_text(
        "def shout(text: str) -> str:\n    return text.upper()\n"
    )
    (root / "pipeline.py").write_text(_SHOUT_PIPELINE)
    monkeypatch.setattr(project, "_project_root_cache", root)
    pipeline = discovery.discover_pipeline(project_root=root)
    assert pipeline is not None
    pipeline.ensure_fingerprint("first/loud")


def _helper_fingerprint_cache_keys(root: pathlib.Path) -> list[bytes]:
    """The function-hash (`fp:`) and stage-manifest (`sm:`) keys in `root`'s StateDB."""
    if not (root / ".pivot" / "state.lmdb").exists():
        return []
    with state.StateDB(root / ".pivot", readonly=True) as db:
        return [key for prefix in (b"fp:", b"sm:") for key, _ in db.iter_prefix(prefix)]


def test_discovery_writes_no_fingerprint_cache_entry_queued_in_another_project(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hashes and manifests queued in one project never reach a later project's StateDB.

    A process pointed at several projects in turn, as a test process is, flushes the pending
    cache writes at each discovery. Those queued in the first project name its files, so
    discovery in the second drops them and writes nothing there, no write lock included.
    """
    _helper_fingerprinted_project(tmp_path / "first", monkeypatch)
    second = tmp_path / "second"
    second.mkdir()
    (second / ".pivot").mkdir()
    (second / "pipeline.py").write_text(
        "from pivot.compose import Pipeline\n\nwith Pipeline('second') as pipeline:\n    pass\n"
    )
    monkeypatch.setattr(project, "_project_root_cache", second)

    assert discovery.discover_pipeline(project_root=second) is not None

    assert not list(second.rglob("*.lock")), "a write lock in a project nothing was queued for"
    assert _helper_fingerprint_cache_keys(second) == []


def test_flush_writes_fingerprint_cache_entries_into_the_project_that_queued_them(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writes queued after discovery reach the StateDB of the project that queued them."""
    first = tmp_path / "first"
    _helper_fingerprinted_project(first, monkeypatch)

    fingerprint.flush_ast_hash_cache()
    fingerprint.flush_manifest_cache()

    names = {key.split(b"\x00")[0] for key in _helper_fingerprint_cache_keys(first)}
    assert {b"fp:shout_helpers.py", b"sm:first/loud"} <= names


def _write_sibling_stage_project(
    root: pathlib.Path, value: str, *, broken: bool = False
) -> pathlib.Path:
    root.mkdir()
    (root / "stages.py").write_text(f'VALUE = "{value}"\n')
    pipeline_path = root / "pipeline.py"
    pipeline_path.write_text(
        f"""
import pathlib

from pivot.compose import Pipeline, stage
from stages import VALUE


@stage
def build() -> str:
    return VALUE


with Pipeline("{root.name}") as pipeline:
    build()
"""
        + ("\nraise RuntimeError('broken pipeline')\n" if broken else "")
    )
    return pipeline_path


def test_failed_pipeline_import_rolls_back_sibling_modules(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed = _write_sibling_stage_project(tmp_path / "failed", "a", broken=True)
    monkeypatch.setattr(project, "_project_root_cache", failed.parent)
    with pytest.raises(discovery.DiscoveryError, match="broken pipeline"):
        discovery.discover_pipeline(project_root=failed.parent)

    valid = _write_sibling_stage_project(tmp_path / "valid", "b")
    monkeypatch.setattr(project, "_project_root_cache", valid.parent)
    pipeline = discovery.discover_pipeline(project_root=valid.parent)

    assert pipeline is not None
    assert pipeline.get_stage("valid/build")["func"]() == "b"
    module_file = getattr(sys.modules["stages"], "__file__", None)
    assert pathlib.Path(str(module_file)).resolve() == valid.parent / "stages.py"


def test_direct_pipeline_load_starts_fresh_import_session(tmp_path: pathlib.Path) -> None:
    first = _write_sibling_stage_project(tmp_path / "first", "a")
    second = _write_sibling_stage_project(tmp_path / "second", "b")

    first_pipeline = discovery.load_pipeline_from_path(first)
    second_pipeline = discovery.load_pipeline_from_path(second)

    assert first_pipeline is not None
    assert second_pipeline is not None
    assert second_pipeline.get_stage("second/build")["func"]() == "b"


def _helper_rewrite_in_one_second(path: pathlib.Path, text: str) -> None:
    """Rewrite ``path`` with ``text``, of the same size, as if its previous write, that write's
    cached bytecode and this write all fell in one whole second: CPython checks cached bytecode
    only by the source's size and whole-second mtime."""
    old = path.stat()
    cached = pathlib.Path(importlib.util.cache_from_source(str(path)))
    assert cached.is_file(), "the first discovery caches the file's bytecode"
    assert len(text.encode()) == old.st_size
    path.write_text(text)
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    os.utime(cached, ns=(old.st_atime_ns, old.st_mtime_ns))


@pytest.fixture
def sibling_project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A project whose pipeline.py imports ``VALUE = "alpha"`` from its sibling stages.py."""
    root = _write_sibling_stage_project(tmp_path / "project", "alpha").parent
    monkeypatch.setattr(project, "_project_root_cache", root)
    return root


def test_discovery_loads_a_same_size_edit_of_pipeline_py_made_in_the_same_second(
    sibling_project: pathlib.Path,
) -> None:
    """An edit to pipeline.py that keeps its size, made in the second its previous version was
    written and compiled, is loaded, not the previous version's cached bytecode."""
    pipeline_path = sibling_project / "pipeline.py"
    first = discovery.discover_pipeline(project_root=sibling_project)
    assert first is not None
    assert first.list_stages() == ["project/build"]

    _helper_rewrite_in_one_second(
        pipeline_path, pipeline_path.read_text().replace("build", "built")
    )
    second = discovery.discover_pipeline(project_root=sibling_project)

    assert second is not None
    assert second.list_stages() == ["project/built"]


def test_discovery_loads_a_same_size_edit_of_an_imported_module_made_in_the_same_second(
    sibling_project: pathlib.Path,
) -> None:
    """The same holds for a project module pipeline.py imports."""
    first = discovery.discover_pipeline(project_root=sibling_project)
    assert first is not None
    assert first.get_stage("project/build")["func"]() == "alpha"

    _helper_rewrite_in_one_second(sibling_project / "stages.py", 'VALUE = "bravo"\n')
    second = discovery.discover_pipeline(project_root=sibling_project)

    assert second is not None
    assert second.get_stage("project/build")["func"]() == "bravo"


@pytest.mark.parametrize("cache_case", ["pycache", "read-only-pycache", "dont-write-bytecode"])
def test_discovery_loads_an_edit_over_bytecode_another_tool_cached_in_the_same_second(
    sibling_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, cache_case: str
) -> None:
    """Bytecode checked by mtime that an older pivot or a plain ``python`` import cached does not
    hide a same-size edit made in its second either, even when that bytecode cannot be replaced:
    ``__pycache__`` is read-only, or Python is told not to write bytecode."""
    stages = sibling_project / "stages.py"
    py_compile.compile(str(stages), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    _helper_rewrite_in_one_second(stages, 'VALUE = "bravo"\n')
    pycache = sibling_project / "__pycache__"
    if cache_case == "read-only-pycache":
        pycache.chmod(0o555)
    if cache_case == "dont-write-bytecode":
        monkeypatch.setattr(sys, "dont_write_bytecode", True)
    try:
        pipeline = discovery.discover_pipeline(project_root=sibling_project)
    finally:
        pycache.chmod(0o755)

    assert pipeline is not None
    assert pipeline.get_stage("project/build")["func"]() == "bravo"


def test_discovery_writes_no_project_bytecode_when_python_is_told_not_to(
    sibling_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sys.dont_write_bytecode`` (``PYTHONDONTWRITEBYTECODE``, ``python -B``) keeps project
    directories free of ``__pycache__``, as it does for any import, and the mtime-checked
    bytecode another tool left is removed, so no other process (a worker importing a stage's
    module by name) can read it."""
    stages = sibling_project / "stages.py"
    py_compile.compile(str(stages), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    left = pathlib.Path(importlib.util.cache_from_source(str(stages)))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    pipeline = discovery.discover_pipeline(project_root=sibling_project)

    assert pipeline is not None
    assert pipeline.get_stage("project/build")["func"]() == "alpha"
    assert not left.exists()
    assert list((sibling_project / "__pycache__").iterdir()) == []


def test_discovery_compiles_from_source_when_hash_based_bytecode_is_never_checked(
    sibling_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``--check-hash-based-pycs never`` CPython trusts checked-hash bytecode without
    checking it, so project code is compiled from source: a same-size edit in the second of
    the cached version is still seen."""
    assert discovery.discover_pipeline(project_root=sibling_project) is not None
    _helper_rewrite_in_one_second(sibling_project / "stages.py", 'VALUE = "bravo"\n')
    monkeypatch.setattr(_imp, "check_hash_based_pycs", "never")

    pipeline = discovery.discover_pipeline(project_root=sibling_project)

    assert pipeline is not None
    assert pipeline.get_stage("project/build")["func"]() == "bravo"


def test_never_checked_mode_still_replaces_bytecode_another_tool_cached(
    sibling_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``--check-hash-based-pycs never`` discovery compiles from source, and it still
    replaces the mtime-checked bytecode another tool cached: pivot's workers do not inherit the
    flag, and another process importing the module by name, as a worker does, reads that
    bytecode."""
    stages = sibling_project / "stages.py"
    py_compile.compile(str(stages), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    _helper_rewrite_in_one_second(stages, 'VALUE = "bravo"\n')
    monkeypatch.setattr(_imp, "check_hash_based_pycs", "never")

    pipeline = discovery.discover_pipeline(project_root=sibling_project)

    assert pipeline is not None
    assert pipeline.get_stage("project/build")["func"]() == "bravo"
    assert discovery._is_checked_hash_pyc(importlib.util.cache_from_source(str(stages)))
    other_process = subprocess.run(
        [sys.executable, "-c", "import stages; print(stages.VALUE)"],
        cwd=sibling_project,
        capture_output=True,
        text=True,
        check=True,
    )
    assert other_process.stdout.strip() == "bravo"


def test_stage_fingerprint_is_of_the_helper_source_discovery_loaded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage's fingerprint describes its helper as discovery loaded it, the code a worker
    receives by value, even when the helper file is edited before the fingerprint is computed."""
    root = tmp_path / "project"
    root.mkdir()
    helper = root / "edited_helper.py"
    alpha = 'def value() -> str:\n    return "alpha"\n'
    bravo = 'def value() -> str:\n    return "bravo, edited after discovery"\n'
    (root / "pipeline.py").write_text(
        "import edited_helper\n"
        + "from pivot.compose import Pipeline, stage\n\n\n"
        + "@stage\n"
        + "def build() -> str:\n"
        + "    return edited_helper.value()\n\n\n"
        + 'with Pipeline("project") as pipeline:\n'
        + "    build()\n"
    )
    monkeypatch.setattr(project, "_project_root_cache", root)

    def discover_and_fingerprint(text: str, *, then_edit_to: str | None = None) -> dict[str, str]:
        helper.write_text(text)
        pipeline = discovery.discover_pipeline(project_root=root)
        assert pipeline is not None
        if then_edit_to is not None:
            helper.write_text(then_edit_to)
        return pipeline.ensure_fingerprint("project/build")

    as_alpha = discover_and_fingerprint(alpha)
    as_bravo = discover_and_fingerprint(bravo)
    assert as_alpha != as_bravo, "the fingerprint tracks the helper's body"

    assert discover_and_fingerprint(alpha, then_edit_to=bravo) == as_alpha


def test_a_later_discovery_forgets_the_earlier_projects_loaded_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovering project B ends project A's import session: A's recorded source and the
    `linecache` lines pinned for it go, so a later load of A's edited helper outside a session is
    fingerprinted from the edited text, not the text A's session compiled."""
    for name in ("a", "b"):
        root = tmp_path / name
        (root / ".pivot").mkdir(parents=True)
        (root / f"pinned_{name}.py").write_text('def value():\n    return "v1"\n')
        (root / "pipeline.py").write_text(
            f"import pinned_{name}\nfrom pivot.compose import Pipeline, stage\n\n\n@stage\n"
            + f"def s() -> str:\n    return pinned_{name}.value()\n\n\n"
            + f'with Pipeline("{name}") as pipeline:\n    s()\n'
        )
    helper = (tmp_path / "a" / "pinned_a.py").resolve()
    monkeypatch.setattr(project, "_project_root_cache", tmp_path / "a")
    first = discovery.discover_pipeline(project_root=tmp_path / "a")
    assert first is not None
    first.ensure_fingerprint("a/s")
    assert loaded_source.digest(str(helper)) is not None
    assert str(helper) in linecache.cache, "fingerprinting pinned the loaded lines"

    monkeypatch.setattr(project, "_project_root_cache", tmp_path / "b")
    assert discovery.discover_pipeline(project_root=tmp_path / "b") is not None

    assert loaded_source.digest(str(helper)) is None
    assert str(helper) not in linecache.cache
    helper.write_text('def value():\n    return "v2, edited"\n')
    spec = importlib.util.spec_from_file_location("pinned_a_again", helper)
    assert spec is not None and spec.loader is not None
    edited = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edited)
    assert "v2, edited" in loaded_source.getsource(edited.value)


_MODULES_BY_KIND = {
    "nested package": "pkg.sub.mod",
    "namespace package": "nspkg.inner.leaf",
    "vendored directory": "vendored_mod",
    "module outside the root": "outside_mod",
    "library inside the root": "inside_lib_mod",
    "stdlib": "colorsys",
}


@pytest.fixture
def import_layout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[pathlib.Path]:
    """A project whose pipeline.py imports one module of each kind ``_MODULES_BY_KIND`` names.
    The modules it imports and its sys.path entries are removed afterwards."""
    root = tmp_path / "project"
    site = root / ".venv" / "lib" / "python3" / "site-packages"
    files = {
        root / "pkg" / "__init__.py": "",
        root / "pkg" / "sub" / "__init__.py": "",
        root / "pkg" / "sub" / "mod.py": "VALUE = 1\n",
        root / "nspkg" / "inner" / "leaf.py": "VALUE = 2\n",
        root / "vendor" / "vendored_mod.py": "VALUE = 3\n",
        tmp_path / "outside" / "outside_mod.py": "VALUE = 4\n",
        site / "inside_lib_mod.py": "VALUE = 5\n",
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for entry in (root / "vendor", tmp_path / "outside", site):
        monkeypatch.syspath_prepend(str(entry))
    stdlib_module = sys.modules.pop(_MODULES_BY_KIND["stdlib"], None)
    imports = "\n".join(f"import {name}" for name in _MODULES_BY_KIND.values())
    (root / "pipeline.py").write_text(
        f"{imports}\n\nfrom pivot.compose import Pipeline\n\n"
        + "with Pipeline('layout') as pipeline:\n    pass\n"
    )
    monkeypatch.setattr(project, "_project_root_cache", root)
    try:
        yield root
    finally:
        for name in _MODULES_BY_KIND.values():
            sys.modules.pop(name, None)
        if stdlib_module is not None:
            sys.modules[_MODULES_BY_KIND["stdlib"]] = stdlib_module


def test_discovery_loads_user_code_and_only_user_code_through_the_checked_loader(
    import_layout: pathlib.Path,
) -> None:
    """Every module fingerprinting treats as user code, under the root or outside it, loads
    through ``_ProjectSourceLoader``; a library inside the root and the stdlib keep CPython's
    loader, and their bytecode is not rewritten as checked-hash bytecode."""
    assert discovery.discover_pipeline(project_root=import_layout) is not None

    loaders = {
        kind: type(sys.modules[name].__loader__).__name__ for kind, name in _MODULES_BY_KIND.items()
    }
    assert loaders == {
        "nested package": "_ProjectSourceLoader",
        "namespace package": "_ProjectSourceLoader",
        "vendored directory": "_ProjectSourceLoader",
        "module outside the root": "_ProjectSourceLoader",
        "library inside the root": "SourceFileLoader",
        "stdlib": "SourceFileLoader",
    }
    for kind in ("library inside the root", "stdlib"):
        source = sys.modules[_MODULES_BY_KIND[kind]].__file__
        assert source is not None
        assert not discovery._is_checked_hash_pyc(importlib.util.cache_from_source(source)), kind


@pytest.mark.parametrize("dont_write_bytecode", [False, True], ids=["write", "dont-write"])
def test_library_package_symlinked_into_the_root_keeps_its_bytecode(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, dont_write_bytecode: bool
) -> None:
    """A library package symlinked into the root is found on the root's import path, but its
    files resolve into site-packages: it loads as CPython loads it, and the bytecode its
    install cached there is neither rewritten as checked-hash bytecode nor removed."""
    package = tmp_path / "env" / "lib" / "python3.13" / "site-packages" / "symlinked_lib"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text('VALUE = "lib"\n')
    py_compile.compile(str(init), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    library_bytecode = pathlib.Path(importlib.util.cache_from_source(str(init)))
    installed = library_bytecode.read_bytes()
    root = tmp_path / "project"
    root.mkdir()
    (root / "symlinked_lib").symlink_to(package, target_is_directory=True)
    (root / "pipeline.py").write_text(
        "import symlinked_lib\n\nfrom pivot.compose import Pipeline\n\n"
        + "with Pipeline('layout') as pipeline:\n    pass\n"
    )
    monkeypatch.setattr(project, "_project_root_cache", root)
    monkeypatch.setattr(sys, "dont_write_bytecode", dont_write_bytecode)
    try:
        assert discovery.discover_pipeline(project_root=root) is not None
        assert sys.modules["symlinked_lib"].VALUE == "lib"
    finally:
        sys.modules.pop("symlinked_lib", None)

    assert library_bytecode.is_file(), "the library's bytecode is not removed"
    assert library_bytecode.read_bytes() == installed, "nor rewritten"


def test_user_code_rule_skips_libraries_the_stdlib_and_pivot(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directories the loader takes are the ones fingerprinting treats as user code: not
    the interpreter's prefix (the stdlib, even with the interpreter installed under the root),
    not site-packages, not pivot's own packages (pivot, and pivot_tui, which pivot's workspace
    installs from its source checkout outside site-packages), and under the root not the
    tooling directories discovery skips."""
    root = tmp_path / "project"
    env = root / "env"
    monkeypatch.setattr(fingerprint, "_STDLIB_PATHS", (*fingerprint._STDLIB_PATHS, env))
    pivot_dir = pathlib.Path(discovery.__file__).resolve().parent
    tui_dir = pathlib.Path(pivot_tui.__file__).resolve().parent

    holds = {
        str(path): discovery._holds_user_code(str(path), root)
        for path in (
            root,
            root / "pkg" / "sub",
            tmp_path / "outside",
            env / "lib" / "python3.13",
            root / ".venv" / "lib" / "python3.13" / "site-packages",
            root / "node_modules" / "x",
            pathlib.Path(json.__file__).resolve().parent,
            pathlib.Path(cloudpickle.__file__).resolve().parent,
            pivot_dir,
            pivot_dir / "storage",
            tui_dir,
            tui_dir / "widgets",
        )
    }

    assert [path for path, holds_user in holds.items() if holds_user] == [
        str(root),
        str(root / "pkg" / "sub"),
        str(tmp_path / "outside"),
    ]


def _helper_pivot_tui_modules() -> dict[str, _stdlib_types.ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "pivot_tui" or name.startswith("pivot_tui.")
    }


@pytest.mark.parametrize("dont_write_bytecode", [False, True], ids=["write", "dont-write"])
def test_pivot_tui_imported_during_a_session_keeps_its_bytecode(
    sibling_project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, dont_write_bytecode: bool
) -> None:
    """pivot_tui is pivot's own package, not user code, also when it is installed from a source
    checkout outside site-packages, as pivot's workspace installs it: imported while a
    discovery's session lasts, it loads as CPython loads it, and the bytecode cached for it is
    neither rewritten as checked-hash bytecode nor removed."""
    tui_dir = pathlib.Path(pivot_tui.__file__).resolve().parent
    assert "site-packages" not in tui_dir.parts, "pivot_tui is installed from its checkout"
    cached = list[pathlib.Path]()
    for source in (tui_dir / "__init__.py", tui_dir / "stats.py"):
        py_compile.compile(str(source), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        cached.append(pathlib.Path(importlib.util.cache_from_source(str(source))))
    monkeypatch.setattr(sys, "dont_write_bytecode", dont_write_bytecode)
    assert discovery.discover_pipeline(project_root=sibling_project) is not None
    loaded = _helper_pivot_tui_modules()
    for name in loaded:
        del sys.modules[name]
    try:
        loader = importlib.import_module("pivot_tui.stats").__loader__
    finally:
        for name in _helper_pivot_tui_modules():
            del sys.modules[name]
        sys.modules.update(loaded)

    for bytecode in cached:
        assert bytecode.is_file(), f"{bytecode} is not removed"
        assert not discovery._is_checked_hash_pyc(str(bytecode)), f"{bytecode} is not rewritten"
    assert type(loader) is importlib.machinery.SourceFileLoader


def test_session_end_gives_user_code_cpythons_loader_back(
    import_layout: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A discovery installs the loader only while it imports. Once it returns, sys.path_hooks,
    sys.meta_path and spec_from_file_location are as they were and an import from a user-code
    directory gets CPython's loader, while the modules the discovery imported stay loaded until
    the next discovery ends its session."""
    root = import_layout
    empty = tmp_path / "empty"
    empty.mkdir()
    assert discovery.discover_pipeline(project_root=empty) is None, "ends any earlier session"
    hooks_before = list(sys.path_hooks)
    meta_path_before = list(sys.meta_path)
    assert discovery.discover_pipeline(project_root=root) is not None
    assert sys.path_hooks == hooks_before
    assert sys.meta_path == meta_path_before
    assert importlib.util.spec_from_file_location is discovery._original_spec_from_file_location
    assert discovery.import_session_root() == root.resolve()
    nested = sys.modules[_MODULES_BY_KIND["nested package"]]
    assert type(nested.__loader__).__name__ == "_ProjectSourceLoader", "imported in the discovery"
    (tmp_path / "outside" / "late_mod.py").write_text("VALUE = 6\n")
    importlib.invalidate_caches()
    try:
        assert type(importlib.import_module("late_mod").__loader__) is (
            importlib.machinery.SourceFileLoader
        ), "a module imported after the discovery returned gets CPython's loader"
        del sys.modules["late_mod"]

        assert discovery.discover_pipeline(project_root=empty) is None

        assert sys.path_hooks == hooks_before
        assert _MODULES_BY_KIND["nested package"] not in sys.modules, "the session ended"
        assert type(importlib.import_module("late_mod").__loader__) is (
            importlib.machinery.SourceFileLoader
        )
    finally:
        sys.modules.pop("late_mod", None)


def _helper_pivot_finders() -> list[str]:
    """The import path entries whose cached finder loads source with Pivot's checked loader."""
    return [
        entry
        for entry, finder in sys.path_importer_cache.items()
        if isinstance(finder, importlib.machinery.FileFinder)
        and any(loader is discovery._ProjectSourceLoader for _, loader in vars(finder)["_loaders"])
    ]


_BROKEN = "raise ValueError('broken pipeline')\n"
_EXITS = "import sys\nsys.exit(3)\n"
_INTERRUPTED = "raise KeyboardInterrupt\n"


@pytest.mark.parametrize(
    ("load", "body", "raises"),
    [
        pytest.param("discover", "", None, id="discover-loaded"),
        pytest.param("discover", _BROKEN, discovery.DiscoveryError, id="discover-error"),
        pytest.param("discover", _EXITS, discovery.DiscoveryError, id="discover-exit"),
        pytest.param("discover", _INTERRUPTED, KeyboardInterrupt, id="discover-interrupt"),
        pytest.param("path", "", None, id="path-loaded"),
        # load_pipeline_from_path logs an Exception from pipeline.py and returns None.
        pytest.param("path", _BROKEN, None, id="path-error"),
        pytest.param("path", _EXITS, SystemExit, id="path-exit"),
        pytest.param("path", _INTERRUPTED, KeyboardInterrupt, id="path-interrupt"),
    ],
)
def test_a_finished_discovery_leaves_no_import_hook_installed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    load: str,
    body: str,
    raises: type[BaseException] | None,
) -> None:
    """A discovery that returns, fails or is interrupted leaves sys.meta_path, sys.path_hooks,
    spec_from_file_location and the finders cached for user-code directories as it found them,
    so a later import in the process, or a later test, is not recorded or loaded by Pivot."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "helper.py").write_text("VALUE = 1\n")
    (root / "pipeline.py").write_text(
        "import helper\n\nfrom pivot.compose import Pipeline\n\n"
        + body
        + "with Pipeline('hooks') as pipeline:\n    pass\n"
    )
    monkeypatch.setattr(project, "_project_root_cache", root)
    meta_path_before = list(sys.meta_path)
    hooks_before = list(sys.path_hooks)
    pivot_finders_before = _helper_pivot_finders()
    try:
        with pytest.raises(raises) if raises is not None else contextlib.nullcontext():
            if load == "discover":
                assert discovery.discover_pipeline(project_root=root) is not None
            else:
                pipeline = discovery.load_pipeline_from_path(root / "pipeline.py")
                assert (pipeline is None) == (body == _BROKEN)

        assert sys.meta_path == meta_path_before
        assert sys.path_hooks == hooks_before
        assert importlib.util.spec_from_file_location is (
            discovery._original_spec_from_file_location
        )
        assert _helper_pivot_finders() == pivot_finders_before
    finally:
        sys.modules.pop("helper", None)


def test_a_module_that_replaces_itself_in_sys_modules_keeps_its_record(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module can put another module object under its name while it loads (a ModuleType
    subclass, for module-level properties). Discovery's loader recorded the bytes it compiled for
    the name; the end of discovery links that record to the object ``sys.modules`` now holds
    (``loaded_source.record_user_modules``), so the module is not judged again as one no recorder
    saw compile, which would make every stage that uses it untrusted."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "self_replacing.py").write_text(
        "import sys\nimport types\n\n\nclass _Module(types.ModuleType):\n    pass\n\n\n"
        + "VALUE = 1\n_replacement = _Module(__name__)\n"
        + "_replacement.__dict__.update(globals())\nsys.modules[__name__] = _replacement\n"
    )
    (root / "pipeline.py").write_text(
        "import self_replacing\n\nfrom pivot.compose import Pipeline\n\n"
        + "with Pipeline('replacing') as pipeline:\n    pass\n"
    )
    monkeypatch.setattr(project, "_project_root_cache", root)
    try:
        assert discovery.discover_pipeline(project_root=root) is not None
        assert type(sys.modules["self_replacing"]).__name__ == "_Module"

        path, digest, reason = loaded_source._record_fields(
            loaded_source.digests()["self_replacing"]
        )
        assert (path, reason) == (str((root / "self_replacing.py").resolve()), "")
        assert digest, "recorded with the bytes discovery's loader compiled"
    finally:
        sys.modules.pop("self_replacing", None)


def test_discovery_reuses_the_bytecode_of_an_unchanged_project_file(
    sibling_project: pathlib.Path,
) -> None:
    """A project file that has not changed since discovery cached its bytecode is loaded from
    that bytecode, not compiled and cached again."""
    assert discovery.discover_pipeline(project_root=sibling_project) is not None
    cached = pathlib.Path(importlib.util.cache_from_source(str(sibling_project / "stages.py")))
    before = cached.stat()

    pipeline = discovery.discover_pipeline(project_root=sibling_project)

    assert pipeline is not None
    assert pipeline.get_stage("project/build")["func"]() == "alpha"
    after = cached.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


@pytest.mark.usefixtures("set_project_root")
def test_discover_all_skips_ordinary_broken_sibling(tmp_path: pathlib.Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    (valid / "pipeline.py").write_text(
        """


from pivot.compose import Pipeline, stage


@stage
def build() -> str:
    return "ok"


with Pipeline("valid") as pipeline:
    build()
"""
    )
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "pipeline.py").write_text("raise RuntimeError('broken sibling')\n")

    pipeline = discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True)

    assert pipeline is not None
    assert pipeline.list_stages() == ["valid/build"]


def test_subdirectory_pipeline_displaces_and_restores_stdlib_parent(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "project"
    collision = root / "json"
    collision.mkdir(parents=True)
    (collision / "pipeline.py").write_text(
        """
import pathlib

from pivot.compose import Pipeline, stage


@stage
def build() -> str:
    return "project"


with Pipeline("project", root=pathlib.Path(__file__).parents[1]) as pipeline:
    build()
"""
    )

    pipeline = discovery.discover_pipeline(project_root=root, all_pipelines=True)

    assert pipeline is not None
    assert pipeline.list_stages() == ["project/build"]
    assert sys.modules["json"] is not json

    assert discovery.discover_pipeline(project_root=tmp_path / "empty") is None
    assert sys.modules["json"] is json


def test_discovery_restores_displaced_pipeline_module_and_caller_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foreign = _stdlib_types.ModuleType("pipeline")
    monkeypatch.setitem(sys.modules, "pipeline", foreign)
    root = _write_sibling_stage_project(tmp_path / "project", "project").parent
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setattr(project, "_project_root_cache", root)

    pipeline = discovery.discover_pipeline(project_root=root)

    assert pipeline is not None
    assert sys.modules["pipeline"] is not foreign
    assert discovery.discover_pipeline(project_root=tmp_path / "empty") is None
    assert sys.modules["pipeline"] is foreign
    assert sys.path.count(str(root)) == 1


def test_discovery_keeps_project_module_preimported_by_caller(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_sibling_stage_project(tmp_path / "project", "caller").parent
    monkeypatch.setattr(project, "_project_root_cache", root)
    # The caller's own import outlives discovery by design, so this test must remove it itself.
    with conftest.stage_module_isolation(root):
        stages = importlib.import_module("stages")

        pipeline = discovery.discover_pipeline(project_root=root)

        assert pipeline is not None
        assert sys.modules["stages"] is stages
        assert discovery.discover_pipeline(project_root=tmp_path / "empty") is None
        assert sys.modules["stages"] is stages


_PLAIN_PIPELINE = """\
import helpers
from pivot import Pipeline, stage


@stage
def use_plain(seed: str) -> str:
    return helpers.plain()


with Pipeline("p") as pipeline:
    use_plain(pipeline.input("seed.txt"))
"""


def _load_program_module(name: str, path: pathlib.Path) -> _stdlib_types.ModuleType:
    """The module a program that runs Pivot imported from ``path`` under ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_programs_own_helpers_module_does_not_hide_an_edit_to_the_projects(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A program that runs Pivot in its own process (`executor.run`, as a notebook does) has
    imported its own `helpers`, which has no `plain`, from outside the project. The stage runs
    the project's `helpers.py` in its worker, so that is the module its lock must describe: an
    edit to it reruns the stage instead of leaving plain-v1 cached. When the session ends, the
    program has its own module back."""
    program = tmp_path / "program"
    program.mkdir()
    (program / "helpers.py").write_text('def greet() -> str:\n    return "the program\'s"\n')
    programs_helpers = _load_program_module("helpers", program / "helpers.py")
    monkeypatch.setitem(sys.modules, "helpers", programs_helpers)
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    (root / ".pivot").mkdir()
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "seed.txt").write_text("seed\n")
    helpers = root / "helpers.py"
    helpers.write_text('def plain() -> str:\n    return "plain-v1"\n')
    (root / "pipeline.py").write_text(_PLAIN_PIPELINE)
    monkeypatch.setattr(project, "_project_root_cache", root)
    output = root / "data" / "p" / "use_plain.txt"

    first = executor.run()
    assert first["p/use_plain"]["status"] == "ran", first
    assert output.read_text() == "plain-v1"
    helpers.write_text(helpers.read_text().replace("-v1", "-v2"))
    second = executor.run()

    assert second["p/use_plain"]["status"] == "ran", second
    assert output.read_text() == "plain-v2"
    assert discovery.discover_pipeline(project_root=tmp_path / "empty") is None
    assert sys.modules["helpers"] is programs_helpers


def test_discovery_takes_over_a_programs_package_and_its_submodule_for_the_session(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The program holds its own `helpers` package and `helpers.text`; the project's `helpers`
    package has a `text` module too. Project code reads the project's while the session lasts,
    and the program gets both of its modules back when the session ends."""
    program = tmp_path / "program" / "helpers"
    program.mkdir(parents=True)
    (program / "__init__.py").write_text("")
    (program / "text.py").write_text("VALUE = 'program'\n")
    package = _load_program_module("helpers", program / "__init__.py")
    text = _load_program_module("helpers.text", program / "text.py")
    monkeypatch.setitem(sys.modules, "helpers", package)
    monkeypatch.setitem(sys.modules, "helpers.text", text)
    root = tmp_path / "project"
    (root / "helpers").mkdir(parents=True)
    (root / "helpers" / "__init__.py").write_text("")
    (root / "helpers" / "text.py").write_text("VALUE = 'project'\n")
    (root / "pipeline.py").write_text(
        "from pivot.compose import Pipeline, stage\nfrom helpers.text import VALUE\n\n\n"
        + "@stage\ndef build() -> str:\n    return VALUE\n\n\n"
        + 'with Pipeline("project") as pipeline:\n    build()\n'
    )
    monkeypatch.setattr(project, "_project_root_cache", root)

    pipeline = discovery.discover_pipeline(project_root=root)

    assert pipeline is not None
    assert pipeline.get_stage("project/build")["func"]() == "project"
    assert discovery.discover_pipeline(project_root=tmp_path / "empty") is None
    assert sys.modules["helpers"] is package
    assert sys.modules["helpers.text"] is text


def test_discovery_keeps_a_library_the_root_has_a_module_for(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pivot and its workers hold their libraries: a root `json.py` displaces none of them."""
    root = _write_sibling_stage_project(tmp_path / "project", "project").parent
    (root / "json.py").write_text("VALUE = 'project'\n")
    monkeypatch.setattr(project, "_project_root_cache", root)

    assert discovery.discover_pipeline(project_root=root) is not None

    assert sys.modules["json"] is json


@pytest.mark.parametrize(
    ("name", "root_file"),
    [
        pytest.param("__main__", "__main__.py", id="main-module"),
        pytest.param("helpers", "helpers/notes.txt", id="namespace-portion"),
    ],
)
def test_discovery_keeps_a_programs_module_the_root_does_not_take_over(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, name: str, root_file: str
) -> None:
    """The program's main module stays, whatever the root holds, and a directory without
    `__init__.py` is a namespace portion, which a module found on `sys.path` wins over."""
    program = tmp_path / "program"
    program.mkdir()
    (program / "held.py").write_text("VALUE = 'program'\n")
    held = _load_program_module(name, program / "held.py")
    monkeypatch.setitem(sys.modules, name, held)
    root = _write_sibling_stage_project(tmp_path / "project", "project").parent
    (root / root_file).parent.mkdir(parents=True, exist_ok=True)
    (root / root_file).write_text("VALUE = 'project'\n")
    monkeypatch.setattr(project, "_project_root_cache", root)

    assert discovery.discover_pipeline(project_root=root) is not None

    assert sys.modules[name] is held


def test_discover_pipeline_missing_pipeline_variable_names_instance(
    tmp_path: pathlib.Path,
) -> None:
    """A Pipeline assigned to the wrong module variable names that variable in the error."""
    (tmp_path / "pipeline.py").write_text(
        "import pathlib\nfrom pivot.compose import Pipeline\n"
        + "wrong_name = Pipeline('wrong', root=pathlib.Path('.'))\n"
    )

    with pytest.raises(
        discovery.DiscoveryError, match="Found Pipeline instance named 'wrong_name'"
    ):
        discovery.discover_pipeline(project_root=tmp_path)


@pytest.mark.usefixtures("set_project_root")
def test_discover_all_rolls_back_pickle_registration_for_broken_sibling(
    tmp_path: pathlib.Path,
) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "broken_helper.py").write_text("VALUE = 'broken'\n")
    (broken / "pipeline.py").write_text(
        """
from broken.broken_helper import VALUE

raise RuntimeError(VALUE)
"""
    )
    valid = tmp_path / "valid"
    valid.mkdir()
    (valid / "pipeline.py").write_text(
        """


from pivot.compose import Pipeline, stage


@stage
def build() -> str:
    return "valid"


with Pipeline("valid") as pipeline:
    build()
"""
    )

    pipeline = discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True)

    assert pipeline is not None
    assert pipeline.list_stages() == ["valid/build"]
    assert "broken.broken_helper" not in sys.modules
    assert "broken.broken_helper" not in cloudpickle.list_registry_pickle_by_value()


def test_second_discovery_keeps_libraries_from_venv_inside_root(
    venv_inside_root_project: pathlib.Path,
) -> None:
    """Site-packages under the root are libraries, not project modules: discovery keeps them."""
    first = discovery.discover_pipeline(project_root=venv_inside_root_project)
    second = discovery.discover_pipeline(project_root=venv_inside_root_project)

    assert first is not None
    assert second is not None, "the second discovery should load pipeline.py again"
    assert second.list_stages() == ["venv/train"]
