"""A stage's lock describes the code its worker ran, when user code is edited during a run.

Discovery loads the project's code before any stage runs. A worker receives a project module's
functions by value, as loaded, and imports any other user module from its file. These tests edit
code after discovery (from an upstream stage) and check, through the installed ``pivot`` CLI, that
the next run never reports an output that edited code would change as cached.
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap
from typing import cast

import pytest

_PIVOT = pathlib.Path(sys.executable).parent / "pivot"
_EDITED = "imported user modules that changed after pivot loaded them"
_HELPERS = 'def plain():\n    return "plain-v1"\n\n\ndef other():\n    return "other-v1"\n'

# `edit_helper` rewrites the helpers file after discovery loaded it; `use_plain` runs after it.
_EDIT_THEN_USE = """\
import pathlib

import helpers
from pivot import Pipeline, stage

HELPERS = {helpers_file!r}


@stage
def edit_helper(seed: str) -> str:
    path = pathlib.Path(HELPERS)
    path.write_text(path.read_text().replace("-v1", "-v2"))
    return "ready"


@stage
def use_plain(ready: str) -> str:
    return helpers.plain()


with Pipeline("p") as pipeline:
    use_plain(edit_helper(pipeline.input("seed.txt")))
"""

# `edit_pipeline` rewrites the `answer` stage's own source after discovery loaded it.
_EDIT_STAGE = """\
import pathlib

from pivot import Pipeline, no_fingerprint, stage


@stage
def edit_pipeline(seed: str) -> str:
    path = pathlib.Path(__file__)
    path.write_text(path.read_text().replace("answer" + "-v1", "answer" + "-v2"))
    return "ready"


@stage{no_fingerprint}
def answer(ready: str) -> str:
    return "answer-v1"


with Pipeline("p") as pipeline:
    answer(edit_pipeline(pipeline.input("seed.txt")))
"""


def _run(
    project: pathlib.Path, *args: str, pythonpath: pathlib.Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PIVOT_UNSAFE_FINGERPRINTING", "PYTHONPATH")
    }
    env["COLUMNS"] = "1000"
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    result = subprocess.run(
        [str(_PIVOT), *args], cwd=project, env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _repro(project: pathlib.Path, pythonpath: pathlib.Path | None = None) -> str:
    """Run `pivot repro`; return what it printed, warnings included, on one line."""
    result = _run(project, "repro", pythonpath=pythonpath)
    return " ".join((result.stdout + result.stderr).split())


def _repro_json(
    project: pathlib.Path, pythonpath: pathlib.Path | None = None
) -> dict[str, tuple[str, str]]:
    """Run `pivot repro --json`; return each stage's (status, reason)."""
    result = _run(project, "repro", "--json", pythonpath=pythonpath)
    stages = dict[str, tuple[str, str]]()
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        fields = cast("dict[str, object]", event)
        if "stage" in fields and "status" in fields:
            stages[str(fields["stage"])] = (str(fields["status"]), str(fields.get("reason", "")))
    return stages


def _project(
    tmp_path: pathlib.Path, pipeline: str, *, helpers_in_root: bool = True
) -> tuple[pathlib.Path, pathlib.Path | None]:
    """A project whose `helpers.py` sits in its root or, on PYTHONPATH, outside it; returns the
    project and the PYTHONPATH directory."""
    project = tmp_path / "project"
    helpers_dir = project if helpers_in_root else tmp_path / "ext"
    helpers_dir.mkdir(parents=True)
    project.mkdir(exist_ok=True)
    helpers_file = helpers_dir / "helpers.py"
    helpers_file.write_text(_HELPERS)
    (project / "pipeline.py").write_text(pipeline.format(helpers_file=str(helpers_file)))
    (project / "data" / "raw").mkdir(parents=True)
    (project / "data" / "raw" / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    pythonpath = None if helpers_in_root else helpers_dir
    _run(project, "init", pythonpath=pythonpath)
    return project, pythonpath


def _output(project: pathlib.Path, stage: str) -> str:
    return (project / "data" / "p" / f"{stage}.txt").read_text()


def _lock(project: pathlib.Path, stage: str) -> str:
    return (project / ".pivot" / "stages" / "p" / f"{stage}.lock").read_text()


def test_in_root_helper_edited_by_an_upstream_stage_is_not_cached_stale(
    tmp_path: pathlib.Path,
) -> None:
    """The worker runs the helper as discovery loaded it, and the lock records that version, so
    the next run sees the edit and reruns the stage instead of reporting the old output."""
    project, _ = _project(tmp_path, _EDIT_THEN_USE)

    first = _repro(project)
    assert _output(project, "use_plain") == "plain-v1", "the worker runs the loaded helper"
    assert _EDITED not in first, "a helper sent by value runs as loaded: nothing to warn about"
    assert "edited-source:" not in _lock(project, "use_plain")

    second = _repro_json(project)
    assert second["p/use_plain"] == ("ran", "Code changed"), second
    assert _output(project, "use_plain") == "plain-v2"

    third = _repro_json(project)
    assert third["p/use_plain"][0] == "cached", third


@pytest.mark.parametrize(
    "no_fingerprint", ["", "\n@no_fingerprint()"], ids=["fingerprinted", "no-fingerprint"]
)
def test_stage_edited_by_an_upstream_stage_is_not_cached_stale(
    tmp_path: pathlib.Path, no_fingerprint: str
) -> None:
    """A stage's own source in `pipeline.py`, edited after discovery, is recorded as loaded: by
    its AST, or by its file's hash for a `@no_fingerprint` stage."""
    project, _ = _project(tmp_path, _EDIT_STAGE.replace("{no_fingerprint}", no_fingerprint))

    _repro(project)
    assert _output(project, "answer") == "answer-v1"

    second = _repro_json(project)
    assert second["p/answer"] == ("ran", "Code changed"), second
    assert _output(project, "answer") == "answer-v2"


def test_out_of_root_helper_a_warm_worker_holds_is_not_cached_stale(
    tmp_path: pathlib.Path,
) -> None:
    """One worker imports the PYTHONPATH helper for `early`, before `edit_helper` edits it, and
    runs `use_plain` with the module it holds. The lock records that version, not the file's."""
    pipeline = """\
import pathlib

import helpers
from pivot import Pipeline, stage

HELPERS = {helpers_file!r}


@stage
def early(seed: str) -> str:
    return helpers.other()


@stage
def edit_helper(first: str) -> str:
    path = pathlib.Path(HELPERS)
    path.write_text(path.read_text().replace("-v1", "-v2"))
    return "ready"


@stage
def use_plain(ready: str) -> str:
    return helpers.plain()


with Pipeline("p") as pipeline:
    use_plain(edit_helper(early(pipeline.input("seed.txt"))))
"""
    project, pythonpath = _project(tmp_path, pipeline, helpers_in_root=False)
    _run(project, "config", "set", "core.max_workers", "1", pythonpath=pythonpath)

    first = _repro(project, pythonpath)
    assert _output(project, "use_plain") == "plain-v1", "the warm worker holds the old module"
    assert _EDITED not in first, "the worker ran the code pivot loaded: nothing to warn about"
    assert "edited-source:" not in _lock(project, "use_plain")

    second = _repro_json(project, pythonpath)
    assert second["p/use_plain"] == ("ran", "Code changed"), second
    assert _output(project, "use_plain") == "plain-v2"


def test_out_of_root_helper_imported_after_its_edit_is_not_reused(tmp_path: pathlib.Path) -> None:
    """The worker imports the PYTHONPATH helper only after `edit_helper` edited it, so it runs
    code the coordinator's fingerprint does not describe: the stage warns, its lock is marked,
    and the next run reruns it."""
    project, pythonpath = _project(tmp_path, _EDIT_THEN_USE, helpers_in_root=False)

    first = _repro(project, pythonpath)
    assert _output(project, "use_plain") == "plain-v2", "the worker imports the edited file"
    assert f"Stage 'p/use_plain' {_EDITED}: helpers." in first, first
    assert "edited-source:helpers" in _lock(project, "use_plain")

    second = _repro_json(project, pythonpath)
    assert second["p/use_plain"] == (
        "ran",
        "Previous run imported modules edited after pivot loaded them: helpers",
    ), second
    assert "edited-source:" not in _lock(project, "use_plain")

    third = _repro_json(project, pythonpath)
    assert third["p/use_plain"][0] == "cached", third


@pytest.mark.parametrize("helpers_in_root", [True, False], ids=["in-root", "out-of-root"])
def test_unedited_project_is_cached_on_the_next_run(
    tmp_path: pathlib.Path, helpers_in_root: bool
) -> None:
    """Without an edit nothing warns and every stage is cached on the next run: the workers
    import the content the coordinator loaded, by module and by a stage body's import."""
    pipeline = textwrap.dedent(
        """\
        import helpers
        from pivot import Pipeline, stage


        @stage
        def first(seed: str) -> str:
            return helpers.other()


        @stage
        def body_import(ready: str) -> str:
            import helpers

            return helpers.plain()


        @stage
        def use_plain(ready: str) -> str:
            return helpers.plain()


        with Pipeline("p") as pipeline:
            ready = first(pipeline.input("seed.txt"))
            body_import(ready)
            use_plain(ready)
        """
    )
    project, pythonpath = _project(tmp_path, pipeline, helpers_in_root=helpers_in_root)

    first = _repro(project, pythonpath)
    assert "3 ran" in first, first
    assert _EDITED not in first
    for stage in ("first", "body_import", "use_plain"):
        assert "edited-source:" not in _lock(project, stage)

    second = _repro_json(project, pythonpath)
    assert set(second) == {"p/first", "p/body_import", "p/use_plain"}, second
    assert {status for status, _ in second.values()} == {"cached"}, second
