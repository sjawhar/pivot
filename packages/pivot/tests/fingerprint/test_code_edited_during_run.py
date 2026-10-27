"""A stage's lock describes the code its worker ran, when user code is edited during a run.

Discovery loads the project's code before any stage runs. A worker receives a project module's
functions by value, as loaded, and imports any other user module from its file. These tests edit
code after discovery (from an upstream stage) and check, through the installed ``pivot`` CLI, that
the next run never reports an output that edited code would change as cached.
"""

import contextlib
import json
import os
import pathlib
import py_compile
import shutil
import signal
import subprocess
import sys
import sysconfig
import textwrap
import time
import zipfile
from collections.abc import Generator, Sequence
from typing import cast

import pytest
from tests.conftest import send_rpc

from pivot import loaded_source
from pivot.storage import state

_PIVOT = pathlib.Path(sys.executable).parent / "pivot"
# The command that runs pivot: the installed CLI unless a test names another interpreter.
_PIVOT_COMMAND = (str(_PIVOT),)
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
    project: pathlib.Path,
    *args: str,
    pythonpath: pathlib.Path | None = None,
    pivot: Sequence[str] = _PIVOT_COMMAND,
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
        [*pivot, *args], cwd=project, env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _repro(
    project: pathlib.Path,
    pythonpath: pathlib.Path | None = None,
    *,
    pivot: Sequence[str] = _PIVOT_COMMAND,
) -> str:
    """Run `pivot repro`; return what it printed, warnings included, on one line."""
    result = _run(project, "repro", pythonpath=pythonpath, pivot=pivot)
    return " ".join((result.stdout + result.stderr).split())


def _repro_json(
    project: pathlib.Path,
    pythonpath: pathlib.Path | None = None,
    *,
    pivot: Sequence[str] = _PIVOT_COMMAND,
) -> dict[str, tuple[str, str]]:
    """Run `pivot repro --json`; return each stage's (status, reason)."""
    result = _run(project, "repro", "--json", pythonpath=pythonpath, pivot=pivot)
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


_EDIT_USE_THEN_UNRELATED = """\
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


@stage
def unrelated(value: str) -> str:
    return "unrelated-fixed"


with Pipeline("p") as pipeline:
    unrelated(use_plain(edit_helper(pipeline.input("seed.txt"))))
"""


def test_stage_that_never_touches_the_edited_module_is_not_marked(tmp_path: pathlib.Path) -> None:
    """One worker imports the edited PYTHONPATH helper for `use_plain`, then runs `unrelated`,
    which never touches helpers. Only `use_plain` ran code its fingerprint does not describe:
    `unrelated` must not warn that it imported helpers, must not carry the mark, and must be
    cached on the next run, as it is at main."""
    project, pythonpath = _project(tmp_path, _EDIT_USE_THEN_UNRELATED, helpers_in_root=False)
    _run(project, "config", "set", "core.max_workers", "1", pythonpath=pythonpath)

    first = _repro(project, pythonpath)
    assert f"Stage 'p/use_plain' {_EDITED}: helpers." in first, first
    assert f"Stage 'p/unrelated' {_EDITED}" not in first, first
    assert "edited-source:" not in _lock(project, "unrelated")

    second = _repro_json(project, pythonpath)
    assert second["p/use_plain"][0] == "ran", second
    assert second["p/unrelated"][0] == "cached", second


def test_no_commit_then_commit_does_not_cache_the_loaded_helper_output(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot repro --no-commit` runs `use_plain` with the helper discovery loaded (plain-v1)
    after `edit_helper` edited helpers.py. `pivot commit` must not then record that output
    against the edited file: the next run reruns the stage, as it does without --no-commit,
    instead of reporting plain-v1 as cached while helpers.py returns plain-v2."""
    project, _ = _project(tmp_path, _EDIT_THEN_USE)

    _run(project, "repro", "--no-commit")
    assert _output(project, "use_plain") == "plain-v1", "the worker runs the loaded helper"

    # A fix may refuse the commit for this stage instead of recording it; both are sound.
    subprocess.run(
        [str(_PIVOT), "commit"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=300,
        env={k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")},
    )

    after = _repro_json(project)
    assert after["p/use_plain"][0] != "cached", after
    assert _output(project, "use_plain") == "plain-v2"


def test_no_commit_record_commits_the_loaded_manifest_when_outputs_match(
    tmp_path: pathlib.Path,
) -> None:
    """A successful `pivot commit` writes the manifest the matching no-commit run left: v1 for
    `helpers.plain`, not the v2 file it sees in its fresh process. The next repro then reruns."""
    project, _ = _project(tmp_path, _EDIT_THEN_USE)

    _run(project, "repro", "--no-commit")
    _run(project, "commit")

    assert "func:helpers.plain: 09b7aba4b5d8001a" in _lock(project, "use_plain")
    after = _repro_json(project)
    assert after["p/use_plain"] == ("ran", "Code changed"), after


def test_no_commit_then_commit_keeps_the_edited_source_mark(tmp_path: pathlib.Path) -> None:
    """The worker imported the out-of-root helper after its edit. `pivot commit` records the
    outputs with the manifest that run left, the mark included, so the next run reruns, and the
    warning says what `pivot commit` does instead of claiming nothing reuses the result."""
    project, pythonpath = _project(tmp_path, _EDIT_THEN_USE, helpers_in_root=False)

    result = _run(project, "repro", "--no-commit", pythonpath=pythonpath)
    printed = " ".join((result.stdout + result.stderr).split())
    assert f"Stage 'p/use_plain' {_EDITED}: helpers." in printed, printed
    assert "`pivot commit` keeps that when it records these outputs." in printed, printed

    _run(project, "commit", pythonpath=pythonpath)
    assert "edited-source:helpers" in _lock(project, "use_plain")

    after = _repro_json(project, pythonpath)
    assert after["p/use_plain"] == (
        "ran",
        "Previous run imported modules edited after pivot loaded them: helpers",
    ), after


def test_commit_of_outputs_edited_after_a_no_commit_run_trusts_the_user(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot commit` is the "trust me" path: when the outputs on disk are not the ones the
    --no-commit run wrote (the user edited them), it records them against the current code."""
    project, _ = _project(tmp_path, _EDIT_THEN_USE)
    _run(project, "repro", "--no-commit")
    (project / "data" / "p" / "use_plain.txt").write_text("plain-hand-edited")

    _run(project, "commit")

    after = _repro_json(project)
    assert after["p/use_plain"][0] == "cached", after
    assert _output(project, "use_plain") == "plain-hand-edited"


def test_no_commit_warning_does_not_say_the_result_will_not_be_reused(
    tmp_path: pathlib.Path,
) -> None:
    """Under --no-commit no lock is written, and `pivot commit` records the outputs on disk as
    current. The edited-source warning must not tell the user the result "will not be reused";
    like the uncovered-import warning, it must say what `pivot commit` will do."""
    project, pythonpath = _project(tmp_path, _EDIT_THEN_USE, helpers_in_root=False)

    result = _run(project, "repro", "--no-commit", pythonpath=pythonpath)
    printed = " ".join((result.stdout + result.stderr).split())
    assert f"Stage 'p/use_plain' {_EDITED}: helpers." in printed, printed
    assert "will not be reused" not in printed or "pivot commit" in printed, printed


def test_marked_run_files_no_run_cache_entry(tmp_path: pathlib.Path) -> None:
    """Run 1 runs the edited out-of-root helper (plain-v2) under the v1 fingerprint and marks
    its lock. After the user restores v1, the run cache must not hand back plain-v2."""
    project, pythonpath = _project(tmp_path, _EDIT_THEN_USE, helpers_in_root=False)
    assert pythonpath is not None
    _repro(project, pythonpath)
    assert _output(project, "use_plain") == "plain-v2"
    _repro(project, pythonpath)

    (pythonpath / "helpers.py").write_text(_HELPERS)
    third = _repro_json(project, pythonpath)

    assert third["p/use_plain"][0] == "ran", third
    assert _output(project, "use_plain") == "plain-v1"


_EDGE_TAIL = """
@stage
def edit_helper(first: str) -> str:
    {edit}
    return "ready"


@stage
def use_plain(ready: str) -> str:
    return helpers.plain()


with Pipeline("p") as pipeline:
    use_plain(edit_helper(early(pipeline.input("seed.txt"))))
"""


def _edge_project(
    tmp_path: pathlib.Path, header: str, early: str, edit: str, pythonpath: pathlib.Path | None
) -> pathlib.Path:
    """A project whose stage `use_plain` calls a helper loaded some other way than through
    discovery's loader; `edit_helper` changes the helper after discovery loaded it."""
    project = tmp_path / "project"
    (project / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (project / "data" / "raw" / "seed.txt").write_text("seed\n")
    (project / "pipeline.py").write_text(header + early + _EDGE_TAIL.format(edit=edit))
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    _run(project, "init", pythonpath=pythonpath)
    return project


_EARLY_OTHER = "\n\n@stage\ndef early(seed: str) -> str:\n    return helpers.other()\n\n"
_EARLY_SEED = "\n\n@stage\ndef early(seed: str) -> str:\n    return seed\n\n"


def _assert_loaded_code_recorded(project: pathlib.Path, pythonpath: pathlib.Path | None) -> None:
    """Run 1 ran the loaded helper (plain-v1) and its lock records that code; the next run sees
    the edit and reruns."""
    _repro(project, pythonpath)
    assert _output(project, "use_plain") == "plain-v1"
    second = _repro_json(project, pythonpath)
    assert second["p/use_plain"] == ("ran", "Code changed"), second
    assert _output(project, "use_plain") == "plain-v2"


def test_helper_from_a_meta_path_finder_edited_mid_run_is_not_cached_stale(
    tmp_path: pathlib.Path,
) -> None:
    """An editable install's finder (setuptools' flat layout) loads the helper with CPython's
    loader; one worker imports it before the edit. The coordinator records it from that loader."""
    hooked = tmp_path / "hooked"
    hooked.mkdir()
    helper = hooked / "helpers.py"
    helper.write_text(_HELPERS)
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "import importlib.abc\nimport importlib.util\nimport sys\n\n\n"
        + "class _Finder(importlib.abc.MetaPathFinder):\n"
        + "    def find_spec(self, fullname, path=None, target=None):\n"
        + "        if fullname == 'helpers':\n"
        + f"            return importlib.util.spec_from_file_location(fullname, {str(helper)!r})\n"
        + "        return None\n\n\nsys.meta_path.append(_Finder())\n"
    )
    project = _edge_project(
        tmp_path,
        "import pathlib\n\nimport helpers\nfrom pivot import Pipeline, stage\n",
        _EARLY_OTHER,
        f"p = pathlib.Path({str(helper)!r}); p.write_text(p.read_text().replace('-v1', '-v2'))",
        site,
    )
    _run(project, "config", "set", "core.max_workers", "1", pythonpath=site)

    _assert_loaded_code_recorded(project, site)


def test_in_root_helper_loaded_by_spec_from_file_location_is_not_cached_stale(
    tmp_path: pathlib.Path,
) -> None:
    helper = tmp_path / "project" / "helpers.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(_HELPERS)
    header = (
        "import importlib.util\nimport pathlib\nimport sys\n\nfrom pivot import Pipeline, stage\n\n"
        + '_spec = importlib.util.spec_from_file_location("helpers", '
        + 'pathlib.Path(__file__).parent / "helpers.py")\n'
        + "helpers = importlib.util.module_from_spec(_spec)\n"
        + 'sys.modules["helpers"] = helpers\n'
        + "_spec.loader.exec_module(helpers)\n"
    )
    project = _edge_project(
        tmp_path,
        header,
        _EARLY_SEED,
        f"p = pathlib.Path({str(helper)!r}); p.write_text(p.read_text().replace('-v1', '-v2'))",
        None,
    )

    _assert_loaded_code_recorded(project, None)


def test_helper_in_a_zip_edited_mid_run_is_not_cached_stale(tmp_path: pathlib.Path) -> None:
    """A zip on PYTHONPATH; one worker imports the helper before the edit rewrites the zip."""
    archive = tmp_path / "ext" / "helpers.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("helpers.py", _HELPERS)
    edit = (
        f"z = zipfile.ZipFile({str(archive)!r}); t = z.read('helpers.py').decode(); z.close(); "
        + f"z = zipfile.ZipFile({str(archive)!r}, 'w'); "
        + "z.writestr('helpers.py', t.replace('-v1', '-v2')); z.close()"
    )
    early = (
        "\n\n@stage\ndef early(seed: str) -> str:\n    import helpers\n\n"
        + "    return helpers.__name__\n\n"
    )
    project = _edge_project(
        tmp_path,
        "import zipfile\n\nimport helpers\nfrom pivot import Pipeline, stage\n",
        early,
        edit,
        archive,
    )
    _run(project, "config", "set", "core.max_workers", "1", pythonpath=archive)

    _assert_loaded_code_recorded(project, archive)


def test_stage_using_a_sourceless_module_is_never_reused(tmp_path: pathlib.Path) -> None:
    """A `.pyc` with no source on PYTHONPATH, replaced mid-run: no one can read which code ran,
    so the stage warns, its lock is marked, it reruns on every run, and the run cache never
    restores its output, even after the user restores the first version."""
    ext = tmp_path / "ext"
    ext.mkdir()
    built = tmp_path / "built"
    built.mkdir()
    for version in ("v1", "v2"):
        source = built / f"helpers_{version}.py"
        source.write_text(_HELPERS.replace("-v1", f"-{version}"))
        py_compile.compile(
            str(source),
            cfile=str(built / f"helpers_{version}.pyc"),
            dfile=str(ext / "helpers.py"),
            doraise=True,
        )
    helper = ext / "helpers.pyc"
    shutil.copy(built / "helpers_v1.pyc", helper)
    project = _edge_project(
        tmp_path,
        "import shutil\n\nimport helpers\nfrom pivot import Pipeline, stage\n",
        _EARLY_SEED,
        f"shutil.copy({str(built / 'helpers_v2.pyc')!r}, {str(helper)!r})",
        ext,
    )
    unreadable = "uses user modules whose source pivot cannot read: helpers."

    first = _repro(project, ext)
    assert f"Stage 'p/use_plain' {unreadable}" in first, first
    assert "unreadable-source:helpers" in _lock(project, "use_plain")
    second = _repro_json(project, ext)
    assert second["p/use_plain"] == (
        "ran",
        "Previous run used modules whose source pivot cannot read: helpers",
    ), second

    shutil.copy(built / "helpers_v1.pyc", helper)
    third = _repro_json(project, ext)
    assert third["p/use_plain"][0] == "ran", third
    assert _output(project, "use_plain") == "plain-v1"


def test_new_package_shadowing_the_helper_mid_run_is_compared_by_name(
    tmp_path: pathlib.Path,
) -> None:
    """`edit_helper` adds `helpers/__init__.py` beside the out-of-root `helpers.py`, so the
    worker imports another file for the same module name. The stage ran other code: it is
    marked and reruns."""
    pipeline = _EDIT_THEN_USE.replace(
        'path = pathlib.Path(HELPERS)\n    path.write_text(path.read_text().replace("-v1", "-v2"))',
        'package = pathlib.Path(HELPERS).parent / "helpers"\n    package.mkdir()\n'
        + '    (package / "__init__.py").write_text(\'def plain():\\n    return "plain-pkg"\\n\')',
    )
    project, pythonpath = _project(tmp_path, pipeline, helpers_in_root=False)

    first = _repro(project, pythonpath)
    assert _output(project, "use_plain") == "plain-pkg"
    assert f"Stage 'p/use_plain' {_EDITED}: helpers." in first, first
    second = _repro_json(project, pythonpath)
    assert second["p/use_plain"][0] == "ran", second


@pytest.mark.parametrize("helpers_in_root", [True, False], ids=["in-root", "out-of-root"])
def test_helper_imported_before_discovery_edited_mid_run_is_not_cached_stale(
    tmp_path: pathlib.Path, helpers_in_root: bool
) -> None:
    """`sitecustomize` imports the helper when each interpreter starts, before Pivot installs
    its recorder (a `.pth` import line does the same). `edit_helper` then edits it mid-run, and
    `use_plain` runs the helper its worker loaded at start (plain-v1). The next run must not
    report that output as cached while helpers.py returns plain-v2."""
    project, pythonpath = _project(tmp_path, _EDIT_THEN_USE, helpers_in_root=helpers_in_root)
    helpers_dir = project if pythonpath is None else pythonpath
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        f"import sys\nsys.path.insert(0, {str(helpers_dir)!r})\nimport helpers\n"
        + f"sys.path.remove({str(helpers_dir)!r})\n"
    )
    _run(project, "config", "set", "core.max_workers", "1", pythonpath=site)

    _repro(project, site)
    second = _repro_json(project, site)

    assert second["p/use_plain"][0] != "cached" or _output(project, "use_plain") == "plain-v2", (
        second,
        _output(project, "use_plain"),
    )


def _venv_with_pth(tmp_path: pathlib.Path, line: str) -> list[str]:
    """A virtual environment over this one whose site-packages holds a `.pth` file with the
    import line `line`, which `site` runs when each of its interpreters starts, a worker's
    included. Returns its `pivot` command."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    python = venv / "bin" / "python"
    site_packages = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    parent = sysconfig.get_path("purelib")
    (pathlib.Path(site_packages) / "_parent.pth").write_text(
        f"import site; site.addsitedir({parent!r})\n"
    )
    (pathlib.Path(site_packages) / "preload.pth").write_text(f"{line}\n")
    return [str(python), "-c", "from pivot.cli import _console_main; _console_main()"]


@pytest.mark.parametrize("helpers_in_root", [True, False], ids=["in-root", "out-of-root"])
def test_helper_a_pth_line_imports_at_startup_edited_mid_run_is_not_cached_stale(
    tmp_path: pathlib.Path, helpers_in_root: bool
) -> None:
    """A `.pth` import line in site-packages imports the helper when each interpreter starts,
    the coordinator and every worker, before Pivot installs its recorder. `edit_helper` edits it
    mid-run: the next run must not report `use_plain`'s plain-v1 as cached."""
    project, pythonpath = _project(tmp_path, _EDIT_THEN_USE, helpers_in_root=helpers_in_root)
    helpers_dir = str(project if pythonpath is None else pythonpath)
    pivot = _venv_with_pth(
        tmp_path,
        f"import sys; sys.path.insert(0, {helpers_dir!r}); import helpers; "
        + f"sys.path.remove({helpers_dir!r})",
    )
    _run(project, "config", "set", "core.max_workers", "1", pivot=pivot)

    _repro(project, pivot=pivot)
    second = _repro_json(project, pivot=pivot)

    assert second["p/use_plain"][0] != "cached" or _output(project, "use_plain") == "plain-v2", (
        second,
        _output(project, "use_plain"),
    )


def test_helper_imported_before_discovery_and_left_unchanged_is_reused(
    tmp_path: pathlib.Path,
) -> None:
    """A helper `sitecustomize` imports before Pivot installs its recorder, unchanged for longer
    than the 2 s margin when the run starts: its file and bytecode show it holds the file's code,
    so Pivot records it, warns about nothing, and reuses the stage that uses it."""
    pipeline = (
        "import helpers\nfrom pivot import Pipeline, stage\n\n\n@stage\n"
        + "def use_plain(seed: str) -> str:\n    return helpers.plain()\n\n\n"
        + 'with Pipeline("p") as pipeline:\n    use_plain(pipeline.input("seed.txt"))\n'
    )
    project, _ = _project(tmp_path, pipeline)
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        f"import sys\nsys.path.insert(0, {str(project)!r})\nimport helpers\n"
        + f"sys.path.remove({str(project)!r})\n"
    )
    _run(project, "config", "set", "core.max_workers", "1", pythonpath=site)
    # A file changed less than the margin before a process started is unreadable in it.
    time.sleep(loaded_source._CHANGE_TIME_SLACK_NS / 1e9)

    first = _repro(project, site)
    second = _repro_json(project, site)

    assert "did not see user module" not in first, first
    assert "unreadable-source:" not in _lock(project, "use_plain")
    assert second["p/use_plain"][0] == "cached", second


def test_helper_imported_before_discovery_and_edited_before_it_is_never_reused(
    tmp_path: pathlib.Path,
) -> None:
    """`sitecustomize` imports the helper, then edits it, before Pivot starts: no file Pivot can
    read shows which code the module holds, so a stage that uses it warns why and is never
    reused."""
    pipeline = (
        "import helpers\nfrom pivot import Pipeline, stage\n\n\n@stage\n"
        + "def use_plain(seed: str) -> str:\n    return helpers.plain()\n\n\n"
        + 'with Pipeline("p") as pipeline:\n    use_plain(pipeline.input("seed.txt"))\n'
    )
    project, _ = _project(tmp_path, pipeline)
    helpers_file = project / "helpers.py"
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        f"import pathlib\nimport sys\nsys.path.insert(0, {str(project)!r})\nimport helpers\n"
        + f"sys.path.remove({str(project)!r})\n"
        + f"pathlib.Path({str(helpers_file)!r}).write_text({_HELPERS!r})\n"
    )
    _run(project, "config", "set", "core.max_workers", "1", pythonpath=site)

    first = _repro(project, site)
    second = _repro_json(project, site)

    assert "Pivot did not see user module 'helpers' compile" in first, first
    assert "because its file changed after this process started" in first, first
    assert "unreadable-source:helpers" in _lock(project, "use_plain")
    assert second["p/use_plain"][0] == "ran", second


@contextlib.contextmanager
def _whole_second_directory(tmp_path: pathlib.Path) -> Generator[pathlib.Path]:
    """A directory on an ext4 file system made with 128-byte inodes, which keep whole-second
    timestamps, mounted from an image under `tmp_path`. Skips the test when it cannot be made."""
    search = os.pathsep.join((os.environ.get("PATH", ""), "/usr/sbin", "/sbin"))
    mkfs = shutil.which("mkfs.ext4", path=search)
    if mkfs is None or shutil.which("sudo") is None:
        pytest.skip("needs mkfs.ext4 and sudo to mount a file system with whole-second times")
    image = tmp_path / "whole-second.ext4"
    with image.open("wb") as file:
        file.truncate(64 * 1024 * 1024)
    made = subprocess.run(
        [mkfs, "-q", "-F", "-I", "128", str(image)], capture_output=True, text=True, check=False
    )
    if made.returncode != 0:
        pytest.skip(f"mkfs.ext4 -I 128 failed: {made.stderr.strip()}")
    mount = tmp_path / "mnt"
    mount.mkdir()
    mounted = subprocess.run(
        ["sudo", "-n", "mount", "-o", "loop", str(image), str(mount)],
        capture_output=True,
        text=True,
        check=False,
    )
    if mounted.returncode != 0:
        pytest.skip(f"cannot mount the image: {mounted.stderr.strip()}")
    try:
        subprocess.run(
            ["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", str(mount)],
            capture_output=True,
            check=True,
        )
        probe = mount / "probe"
        probe.write_text("probe")
        if probe.stat().st_ctime_ns % 1_000_000_000:
            pytest.skip("the image keeps sub-second change times")
        yield mount
    finally:
        subprocess.run(["sudo", "-n", "umount", "-l", str(mount)], capture_output=True, check=False)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="mounts an ext4 image")
@pytest.mark.skipif(
    not os.environ.get("CI"), reason="mounts a file system image with sudo; runs where CI is set"
)
def test_helper_edited_in_the_second_its_process_started_on_a_whole_second_file_system(
    tmp_path: pathlib.Path,
) -> None:
    """On ext4 made with 128-byte inodes a change time keeps whole seconds, so an edit made in the
    second its process started reads as up to 1 s older than it was. `sitecustomize` imports the
    in-root helper when the `pivot` process starts and then, once, edits it, 0.1 to 0.3 s into
    that second: the run uses the loaded plain-v1. No later run may report `use_plain` cached
    while helpers.py returns plain-v2."""
    pipeline = (
        "import helpers\nfrom pivot import Pipeline, stage\n\n\n@stage\n"
        + "def use_plain(seed: str) -> str:\n    return helpers.plain()\n\n\n"
        + 'with Pipeline("p") as pipeline:\n    use_plain(pipeline.input("seed.txt"))\n'
    )
    with _whole_second_directory(tmp_path) as root:
        project, _ = _project(root, pipeline)
        helpers_file = project / "helpers.py"
        flag = tmp_path / "edit.flag"
        site = tmp_path / "site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(
            f"import pathlib\nimport sys\nsys.path.insert(0, {str(project)!r})\nimport helpers\n"
            + f"sys.path.remove({str(project)!r})\nflag = pathlib.Path({str(flag)!r})\n"
            + "if flag.exists():\n    flag.unlink()\n"
            + f"    path = pathlib.Path({str(helpers_file)!r})\n"
            + "    path.write_text(path.read_text().replace('-v1', '-v2'))\n"
        )
        _run(project, "config", "set", "core.max_workers", "1", pythonpath=site)
        _repro(project, site)
        flag.touch()
        # Start the run 0.1 to 0.3 s into a second, so the edit lands in the second it started.
        while not 0.1 <= time.time() % 1 <= 0.3:
            time.sleep(0.002)
        _repro(project, site)
        assert not flag.exists(), "sitecustomize did not edit the helper"
        later = list[tuple[str, str]]()
        for _ in range(2):
            stages = _repro_json(project, site)
            later.append((stages["p/use_plain"][0], _output(project, "use_plain")))

    assert all(status != "cached" or output == "plain-v2" for status, output in later), later
    assert later[-1][1] == "plain-v2", later


_FOUR_STAGES = """\
from pivot import Pipeline, stage


@stage
def a(seed: str) -> str:
    return "a:" + seed


@stage
def b(seed: str) -> str:
    return "b:" + seed


@stage
def c(seed: str) -> str:
    return "c:" + seed


@stage
def d(seed: str) -> str:
    return "d:" + seed


with Pipeline("p") as pipeline:
    seed = pipeline.input("seed.txt")
    a(seed)
    b(seed)
    c(seed)
    d(seed)
"""

# Wraps the worker's check of a stage's modules, in every process this PYTHONPATH reaches, to log
# the process and whether it holds the coordinator's records when it checks.
_LOG_EACH_CHECK = """\
import os

if os.environ.get("JUDGE_LOG"):
    from pivot import loaded_source
    from pivot.executor import worker

    _checked = worker._untrusted_modules

    def _logged(*args, **kwargs):
        with open(os.environ["JUDGE_LOG"], "a") as log:
            log.write(f"{os.getpid()} {loaded_source.is_worker()}\\n")
        return _checked(*args, **kwargs)

    worker._untrusted_modules = _logged
"""


def _wait_for_checks(log: pathlib.Path, count: int, proc: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 120
    while not (log.exists() and len(log.read_text().splitlines()) >= count):
        assert proc.poll() is None, f"pivot exited with {proc.returncode} before {count} checks"
        assert time.monotonic() < deadline, f"fewer than {count} checks: {log.read_text()!r}"
        time.sleep(0.1)


@pytest.mark.parametrize(
    ("surface", "workers"),
    [("repro", 1), ("repro", 4), ("run", 4), ("watch", 4), ("rpc", 4)],
    ids=["repro-1-worker", "repro-4-workers", "run", "watch", "rpc-warm-pool"],
)
def test_every_worker_that_checks_a_stage_holds_the_coordinators_records(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, surface: str, workers: int
) -> None:
    """A worker judges the modules no recorder saw compile after each stage only while it holds
    the coordinator's records (`loaded_source.is_worker`), which a process that runs a stage
    itself does not. Every pool Pivot starts runs `core._init_worker`, which sets them: one worker
    and many, `pivot run`, `--watch`, and the RPC server, whose second run uses the warm pool."""
    project, _ = _project(tmp_path, _FOUR_STAGES)
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_LOG_EACH_CHECK)
    _run(project, "config", "set", "core.max_workers", str(workers))
    log = tmp_path / "checks.log"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PIVOT_UNSAFE_FINGERPRINTING", "PYTHONPATH")
    }
    env.update(COLUMNS="1000", PYTHONPATH=str(site), JUDGE_LOG=str(log))
    command = {
        "repro": ["repro"],
        "run": ["run", "p/a", "p/b", "p/c", "p/d"],
        "watch": ["repro", "--watch", "--json"],
        "rpc": ["repro", "--watch", "--serve"],
    }[surface]
    expected = 8 if surface == "rpc" else 4

    with (tmp_path / "pivot.out").open("wb") as out:
        proc = subprocess.Popen(
            [str(_PIVOT), *command], cwd=project, env=env, stdout=out, stderr=subprocess.STDOUT
        )
    try:
        if surface in ("repro", "run"):
            assert proc.wait(timeout=300) == 0, (tmp_path / "pivot.out").read_text()
        else:
            _wait_for_checks(log, 4, proc)
            if surface == "rpc":
                # Relative to the project: a long temporary path exceeds the socket path limit.
                monkeypatch.chdir(project)
                sock = pathlib.Path(".pivot", "agent.sock")
                deadline = time.monotonic() + 60
                reply = send_rpc(sock, "run", {"force": True})
                while "result" not in reply and time.monotonic() < deadline:
                    time.sleep(0.2)  # refused until the first run has finished
                    reply = send_rpc(sock, "run", {"force": True})
                assert "result" in reply, reply
                _wait_for_checks(log, expected, proc)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM if surface == "rpc" else signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    checks = [line.split() for line in log.read_text().splitlines()]
    assert len(checks) >= expected, checks
    assert all(holds == "True" for _, holds in checks), checks
    assert all(int(pid) != proc.pid for pid, _ in checks), (proc.pid, checks)


def test_a_lock_a_later_run_writes_retires_the_no_commit_record(tmp_path: pathlib.Path) -> None:
    """A normal run after `pivot repro --no-commit` writes a lock, which supersedes the record the
    --no-commit run left. The user then puts that run's output (plain-v1) back by hand and runs
    `pivot commit`: the lock is current, so commit leaves it and the next run restores the
    output the lock records. A record left behind would make commit match it against plain-v1
    and write that old run's manifest (helper v1), and the next run would rerun the stage."""
    project, _ = _project(tmp_path, _EDIT_THEN_USE)
    _run(project, "repro", "--no-commit")
    assert _output(project, "use_plain") == "plain-v1"
    _repro(project)
    assert _output(project, "use_plain") == "plain-v2"
    output = project / "data" / "p" / "use_plain.txt"
    output.unlink()  # a committed output is read-only
    output.write_text("plain-v1")

    _run(project, "commit")

    assert "func:helpers.plain: 09b7aba4b5d8001a" not in _lock(project, "use_plain")
    after = _repro_json(project)
    assert after["p/use_plain"][0] == "cached", after
    assert _output(project, "use_plain") == "plain-v2"


def test_commit_reads_a_no_commit_record_without_code_as_no_record(tmp_path: pathlib.Path) -> None:
    """A Pivot that keeps `.pvt` files in directory outputs leaves a `nocommit:` record with only
    its own field (`pvt_members`). It holds no code a run ran, so `pivot commit` acts as with no
    record: it leaves a current lock alone, and after an edit records the outputs against the
    code on disk, which retires the record."""
    pipeline = (
        "import helpers\nfrom pivot import Pipeline, stage\n\n\n@stage\n"
        + "def use_plain(seed: str) -> str:\n    return helpers.plain()\n\n\n"
        + 'with Pipeline("p") as pipeline:\n    use_plain(pipeline.input("seed.txt"))\n'
    )
    project, _ = _project(tmp_path, pipeline)
    _repro(project)
    lock_before = _lock(project, "use_plain")
    with state.StateDB(project / ".pivot") as db:
        db.put_raw(
            b"nocommit:p/use_plain",
            json.dumps({"pvt_members": {"data/p/out/a.pvt": "77aa"}}).encode(),
        )

    unchanged = _run(project, "commit")

    assert "Nothing to commit" in unchanged.stdout, unchanged.stdout
    assert _lock(project, "use_plain") == lock_before
    (project / "helpers.py").write_text(_HELPERS.replace("-v1", "-v2"))
    edited = _run(project, "commit")
    assert "Committed 1 stage(s)" in edited.stdout, edited.stdout
    assert _lock(project, "use_plain") != lock_before
    with state.StateDB(project / ".pivot", readonly=True) as db:
        assert db.get_no_commit_record("p/use_plain") is None


_EXT_STAGE = (
    'from pivot import stage\n\n\n@stage\ndef use_ext(ready: str) -> str:\n    return "ext-v1"\n'
)


def test_stage_defined_out_of_root_edited_mid_run_is_never_cached_stale(
    tmp_path: pathlib.Path,
) -> None:
    """A stage defined in a module on PYTHONPATH reaches the worker by reference, and `edit_ext`
    edits that module mid-run. Whichever version the worker ran, no later run, before or after
    the file is put back, reports the stage cached with another version's output."""
    ext = tmp_path / "ext"
    ext.mkdir()
    ext_file = ext / "stages_ext.py"
    ext_file.write_text(_EXT_STAGE)
    project = tmp_path / "project"
    (project / "data" / "raw").mkdir(parents=True)
    (project / "data" / "raw" / "seed.txt").write_text("seed\n")
    (project / "pipeline.py").write_text(
        "import pathlib\n\nfrom pivot import Pipeline, stage\nfrom stages_ext import use_ext\n\n"
        + f"EXT = {str(ext_file)!r}\n\n\n@stage\ndef edit_ext(seed: str) -> str:\n"
        + "    path = pathlib.Path(EXT)\n"
        + "    path.write_text(path.read_text().replace('-v1', '-v2'))\n    return 'ready'\n\n\n"
        + 'with Pipeline("p") as pipeline:\n    use_ext(edit_ext(pipeline.input("seed.txt")))\n'
    )
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    _run(project, "init", pythonpath=ext)
    _run(project, "config", "set", "core.max_workers", "1", pythonpath=ext)
    _repro(project, ext)

    for on_disk in ("ext-v2", "ext-v1"):
        if on_disk == "ext-v1":
            ext_file.write_text(_EXT_STAGE)
        status = _repro_json(project, ext)["p/use_ext"][0]
        assert status != "cached" or _output(project, "use_ext") == on_disk, (
            on_disk,
            status,
            _output(project, "use_ext"),
        )


_REEXPORT_THEN_IMPORT = """\
import pathlib

import helpers
from pivot import Pipeline, stage

HELPERS = {helpers_file!r}


@stage
def edit_helper(seed: str) -> str:
    path = pathlib.Path(HELPERS)
    path.write_text(path.read_text().replace("plain_a as", "plain_b as"))
    return "ready"


@stage
def early(ready: str) -> str:
    import helpers

    return helpers.__name__


@stage
def use_plain(name: str) -> str:
    import helpers

    return helpers.plain()


with Pipeline("p") as pipeline:
    use_plain(early(edit_helper(pipeline.input("seed.txt"))))
"""


def test_module_a_stage_imports_while_it_runs_edited_mid_run_is_marked(
    tmp_path: pathlib.Path,
) -> None:
    """`helpers` re-exports `impl.plain_a` as `plain`; `edit_helper` makes it re-export
    `plain_b`. On one worker `early` imports the edited `helpers` first, so when `use_plain`
    imports it at run time it is already loaded, and `use_plain` runs plain-b. The module it
    imported while it ran counts as used, so its lock is marked; unmarked, putting the file back
    would make the lock match again and serve plain-b as cached."""
    project, pythonpath = _project(tmp_path, _REEXPORT_THEN_IMPORT, helpers_in_root=False)
    assert pythonpath is not None
    helpers_file = pythonpath / "helpers.py"
    reexport = "from impl import plain_a as plain\n"
    helpers_file.write_text(reexport)
    (pythonpath / "impl.py").write_text(
        'def plain_a():\n    return "plain-a"\n\n\ndef plain_b():\n    return "plain-b"\n'
    )
    _run(project, "config", "set", "core.max_workers", "1", pythonpath=pythonpath)

    first = _repro(project, pythonpath)
    assert _output(project, "use_plain") == "plain-b", "the stage ran the edited re-export"
    assert f"Stage 'p/use_plain' {_EDITED}: helpers." in first, first

    helpers_file.write_text(reexport)
    second = _repro_json(project, pythonpath)
    assert second["p/use_plain"][0] != "cached" or _output(project, "use_plain") == "plain-a", (
        second,
        _output(project, "use_plain"),
    )
