# pyright: reportImplicitStringConcatenation=false

"""Red tests from the PR #496 round-seven acceptance recheck (ACCEPTANCE_R7.md D1-D5).

Place in ``packages/pivot/tests/fingerprint/``. Each test drives the installed ``pivot`` CLI in a
fresh project, or reads the shipped docs, and fails at PR head 51d44ec1.
"""

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

_PIVOT = pathlib.Path(sys.executable).parent / "pivot"
_REPO = pathlib.Path(__file__).resolve().parents[4]
_UNCOVERED = "imported user modules its fingerprint does not cover"
_LIB = {"helpers/__init__.py": "", "helpers/lib.py": "VALUE = 1\n"}


def _run(project: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PIVOT_UNSAFE_FINGERPRINTING")
    }
    return subprocess.run(
        [str(_PIVOT), *args], cwd=project, env=env, capture_output=True, text=True, timeout=300
    )


def _flat(result: subprocess.CompletedProcess[str]) -> str:
    return " ".join((result.stdout + result.stderr).split())


def _project(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    project = tmp_path / "project"
    for rel, text in {**files, "data/raw/lines.txt": "a\nb\nc\n"}.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    init = _run(project, "init")
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
    """`@no_fingerprint(code_deps=[...])` hashes files, not the AST, so its manifest has no
    `module:` keys. Main caches this stage and reruns it when `helpers/lib.py` changes. At
    51d44ec1 it warns and reruns on every `repro`, and `pivot verify` fails."""
    project = _project(
        tmp_path, {**_LIB, "pipeline.py": _NO_FINGERPRINT.format(module_import=module_import)}
    )

    first = _run(project, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    assert _UNCOVERED not in _flat(first)
    again = _run(project, "repro", "--json")
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    verify = _run(project, "verify")
    assert verify.returncode == 0, verify.stdout + verify.stderr
    _edit_lib(project)

    edited = _run(project, "repro", "--json")

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
    """A user loader's `load`/`save` body-imports a module discovery loaded. The static walk
    covers it (the lock holds `dep:<name>:module:helpers.lib` / `out:<key>:module:...`), so the
    runtime check must not call it uncovered: no warning, cached on the next run, rerun on edit.
    At 51d44ec1 every run warns, files no run-cache entry, and reruns, and `verify` fails."""
    project = _project(tmp_path, {**_LIB, "helpers/myloader.py": loader, "pipeline.py": pipeline})

    first = _run(project, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    assert _UNCOVERED not in _flat(first)
    again = _run(project, "repro", "--json")
    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    verify = _run(project, "verify")
    assert verify.returncode == 0, verify.stdout + verify.stderr
    _edit_lib(project)

    edited = _run(project, "repro", "--json")

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
    """The docs scope the commit boundary to `repro --no-commit` and give "run `pivot repro`
    without `--no-commit`" as the remedy. A routine `pivot commit` after that remedy records the
    flagged stage as current all the same (the stage is always stale, and `commit` with no
    arguments commits every stale stage), and the edit is then missed. Either `commit` must
    leave the flagged stage stale, or the docs must stop presenting that remedy as sufficient."""
    project = _project(tmp_path, {**_LIB, "pipeline.py": _RUNTIME_IMPORT})
    flagged = _run(project, "repro")
    assert flagged.returncode == 0, flagged.stdout + flagged.stderr
    assert _UNCOVERED in _flat(flagged)
    committed = _run(project, "commit")
    assert committed.returncode == 0, committed.stdout + committed.stderr
    _edit_lib(project)

    edited = _run(project, "repro", "--json")

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
    """The warning says "Fix: import them at module level (or in pipeline.py)". Here pipeline.py
    already imports `helpers.lib` at module level, and the warning still fires: the named fix is
    already applied and cannot clear it. A fix that does is a static import the stage's own code
    makes (walked under contract item 2), or StageParams."""
    project = _project(tmp_path, {**_LIB, "pipeline.py": _ALREADY_IMPORTED})

    first = _run(project, "repro")

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
    """The concepts doc says `pivot status` and `pivot explain` show the reason; there is no
    `pivot explain` command (`pivot status --explain` and `pivot repro --explain` show it)."""
    explain = _run(tmp_path, "explain", "--help")
    if explain.returncode == 0:
        return
    concepts = (_REPO / "docs" / "concepts" / "fingerprinting.md").read_text()
    assert "pivot explain" not in " ".join(concepts.split())
