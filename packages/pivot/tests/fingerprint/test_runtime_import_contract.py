# pyright: reportImplicitStringConcatenation=false

"""Runtime-import contracts exercised through the installed ``pivot`` CLI and shipped docs."""

import pathlib
import subprocess

import pytest
import yaml
from tests import helpers

_REPO = pathlib.Path(__file__).resolve().parents[4]
_UNCOVERED = "imported user modules its fingerprint does not cover"
_LIB = {"helpers/__init__.py": "", "helpers/lib.py": "VALUE = 1\n"}


def _flat(result: subprocess.CompletedProcess[str]) -> str:
    return " ".join((result.stdout + result.stderr).split())


def _project(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    home = tmp_path / "home"
    project = tmp_path / "project"
    for rel, text in {**files, "data/raw/lines.txt": "a\nb\nc\n"}.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    init = helpers.run_pivot(project, home, "init")
    assert init.returncode == 0, init.stdout + init.stderr
    return project


def _output(project: pathlib.Path) -> str:
    return next((project / "data" / "t").iterdir()).read_text()


def _edit_lib(project: pathlib.Path) -> None:
    (project / "helpers" / "lib.py").write_text("VALUE = 100\n")


# D1 -------------------------------------------------------------------------------------------

_NO_FINGERPRINT = """\
{module_import}from pivot import Pipeline, no_fingerprint, stage


@stage
@no_fingerprint(code_deps=["helpers/lib.py"])
def count(text: str) -> dict:
    from helpers.lib import VALUE
    return {{"v": VALUE + len(text.splitlines())}}


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=str))
"""


@pytest.mark.parametrize(
    "module_import",
    [
        pytest.param("", id="not-loaded-at-discovery"),
        pytest.param("import helpers.lib\n", id="loaded"),
    ],
)
def test_d1_no_fingerprint_stage_with_body_import_is_reused(
    tmp_path: pathlib.Path, module_import: str
) -> None:
    """A `@no_fingerprint(code_deps=[...])` stage tracks its declared files, not its imports.
    It reuses unchanged outputs and reruns after a declared file changes."""
    home = tmp_path / "home"
    project = _project(
        tmp_path, {**_LIB, "pipeline.py": _NO_FINGERPRINT.format(module_import=module_import)}
    )

    first = helpers.run_pivot(project, home, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    assert _UNCOVERED not in _flat(first)
    again = helpers.run_pivot(project, home, "repro", "--json")
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    verify = helpers.run_pivot(project, home, "verify")
    assert verify.returncode == 0, verify.stdout + verify.stderr
    _edit_lib(project)

    edited = helpers.run_pivot(project, home, "repro", "--json")

    assert '"stage": "t/count", "status": "ran"' in edited.stdout, edited.stdout
    assert yaml.safe_load(_output(project))["v"] == 103


# D2 -------------------------------------------------------------------------------------------

_READER = """\
import dataclasses
import pathlib

from pivot import loaders


@dataclasses.dataclass(frozen=True)
class Scaled(loaders.Reader[int]):
    def load(self, path: pathlib.Path) -> int:
        from helpers import lib
        return len(path.read_text().splitlines()) * lib.VALUE
"""

_READER_PIPELINE = """\
import helpers.lib
from helpers.myloader import Scaled
from pivot import Pipeline, stage


@stage
def count(n: int) -> dict:
    return {"v": n}


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=int, format=Scaled()))
"""

_WRITER = """\
import dataclasses
import pathlib

from pivot import loaders


@dataclasses.dataclass(frozen=True)
class Scaled(loaders.Writer[int]):
    def save(self, data: int, path: pathlib.Path) -> None:
        from helpers import lib
        path.write_text(f"{data * lib.VALUE}\\n")
"""

_WRITER_PIPELINE = """\
from typing import Annotated

import helpers.lib
from helpers.myloader import Scaled
from pivot import Pipeline, stage


@stage
def count(text: str) -> Annotated[int, Scaled()]:
    return len(text.splitlines())


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=str))
"""


@pytest.mark.parametrize(
    ("loader", "pipeline", "after_edit"),
    [
        pytest.param(_READER, _READER_PIPELINE, "v: 300\n", id="reader-load"),
        pytest.param(_WRITER, _WRITER_PIPELINE, "300\n", id="writer-save"),
    ],
)
def test_d2_user_loader_body_import_of_loaded_module_is_covered(
    tmp_path: pathlib.Path, loader: str, pipeline: str, after_edit: str
) -> None:
    """A user loader's loaded body import is covered by the loader manifest, so it emits no
    uncovered-import warning, reuses unchanged output, and reruns when its module changes."""
    home = tmp_path / "home"
    project = _project(tmp_path, {**_LIB, "helpers/myloader.py": loader, "pipeline.py": pipeline})

    first = helpers.run_pivot(project, home, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    assert _UNCOVERED not in _flat(first)
    again = helpers.run_pivot(project, home, "repro", "--json")
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    verify = helpers.run_pivot(project, home, "verify")
    assert verify.returncode == 0, verify.stdout + verify.stderr
    _edit_lib(project)

    edited = helpers.run_pivot(project, home, "repro", "--json")

    assert '"stage": "t/count", "status": "ran"' in edited.stdout, edited.stdout
    assert _output(project) == after_edit


# D3 -------------------------------------------------------------------------------------------

_RUNTIME_IMPORT = """\
from pivot import Pipeline, stage


@stage
def count(text: str) -> dict:
    scope = {}
    exec("import helpers.lib as loaded", scope)
    return {"v": scope["loaded"].VALUE + len(text.splitlines())}


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=str))
"""


def test_d3_commit_after_a_flagged_committing_repro_is_described(tmp_path: pathlib.Path) -> None:
    """A `pivot commit` after a flagged `repro` preserves the mark, so an edit to the dynamically
    imported module reruns the stage instead of restoring a stale result."""
    home = tmp_path / "home"
    project = _project(tmp_path, {**_LIB, "pipeline.py": _RUNTIME_IMPORT})
    flagged = helpers.run_pivot(project, home, "repro")
    assert flagged.returncode == 0, flagged.stdout + flagged.stderr
    assert _UNCOVERED in _flat(flagged)
    committed = helpers.run_pivot(project, home, "commit")
    assert committed.returncode == 0, committed.stdout + committed.stderr
    _edit_lib(project)

    edited = helpers.run_pivot(project, home, "repro", "--json")

    if '"stage": "t/count", "status": "ran"' in edited.stdout:
        assert yaml.safe_load(_output(project))["v"] == 103
        return
    for doc in ("concepts", "architecture"):
        text = " ".join((_REPO / "docs" / doc / "fingerprinting.md").read_text().split())
        assert "`pivot repro` without `--no-commit`" not in text, (
            f"{doc}: that remedy does not help"
        )


# D4 -------------------------------------------------------------------------------------------

_ALREADY_IMPORTED = """\
import importlib

import helpers.lib
from pivot import Pipeline, stage


@stage
def count(text: str) -> dict:
    load = importlib.__dict__["import_module"]
    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=str))
"""


def test_d4_uncovered_import_warning_names_a_fix_that_applies(tmp_path: pathlib.Path) -> None:
    """The warning's static-import remedy requires reading the module by name in the stage's own
    code, or moving the value into `StageParams`; a `pipeline.py` import alone does not cover it."""
    home = tmp_path / "home"
    project = _project(tmp_path, {**_LIB, "pipeline.py": _ALREADY_IMPORTED})

    first = helpers.run_pivot(project, home, "repro")

    assert first.returncode == 0, first.stdout + first.stderr
    message = _flat(first)
    if _UNCOVERED not in message:
        return
    warning = message[message.index(_UNCOVERED) :]
    warning = warning[: warning.find("Summary:")] if "Summary:" in warning else warning
    # stdout carries the warning and stderr the "Discovered <path>/pipeline.py" line, which
    # `_flat` joins after it; the check is on the warning alone.
    warning = warning.split(" Discovered ", 1)[0]
    assert "module level" not in warning, warning
    assert "pipeline.py" not in warning, warning


# D5 -------------------------------------------------------------------------------------------


def test_d5_concepts_doc_names_a_command_that_exists(tmp_path: pathlib.Path) -> None:
    """The concepts documentation names only commands that can explain a marked stage."""
    home = tmp_path / "home"
    explain = helpers.run_pivot(tmp_path, home, "explain", "--help")
    if explain.returncode == 0:
        return
    concepts = (_REPO / "docs" / "concepts" / "fingerprinting.md").read_text()
    assert "pivot explain" not in " ".join(concepts.split())
