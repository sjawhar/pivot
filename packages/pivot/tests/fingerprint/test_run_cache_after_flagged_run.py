# pyright: reportImplicitStringConcatenation=false

"""A lock the stage-import check marked is never satisfied from the run cache, through the CLI.

An entry filed for the same input hash by an earlier run (0.5.2, or `pivot commit`) would restore
an output the check never saw over the flagged run's output. A marked lock never consults the
run cache, at any input hash, and a flagged run or a `pivot commit` that carries the mark deletes
any entry at its own input hash.
"""

import json
import pathlib
import subprocess

import pytest
import yaml
from tests import helpers

from pivot.storage import state

_LIB = {"helpers/__init__.py": "", "helpers/lib.py": "VALUE = 1\n"}
_LINES = "Local Set Default Proof Using\nSet Default Proof Using\nLemma a\n"
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

# The lock `pivot repro` at main 19fd78cb (release 0.5.2) writes for `_RUNTIME_IMPORT` with
# `_LINES` as its input (VM pivot-slices, python 3.13.16, sha256 742157a1...). It is the same
# fixture `test_runtime_import_limits.py` uses.
_LOCK_0_5_2 = """\
schema_version: 2
code_manifest:
  dep:text:loader:pivot.loaders.Text:empty: f346543fc28fb29c
  dep:text:loader:pivot.loaders.Text:load: 69b78870ffe7932d
  dep:text:loader:pivot.loaders.Text:save: f187f23051043f60
  out:t/count:loader:pivot.loaders.YAML:config: cfc61af456ae42e3
  out:t/count:loader:pivot.loaders.YAML:empty: a66d45b00a1a3a57
  out:t/count:loader:pivot.loaders.YAML:load: 4a31bd3a307ad2f0
  out:t/count:loader:pivot.loaders.YAML:save: 5240a272b83ad042
  self:count: 44514e69896aabf0
  wire:text: lines.txt
  wire:text:path: data/raw/lines.txt
params: {}
deps:
- producer: lines.txt
  key: null
  hash: 133a2387bf822f09
outs:
- key: null
  hash: 2ad953167adbd9db
  path: data/t/count.yaml
"""


def _ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


def _project(tmp_path: pathlib.Path) -> pathlib.Path:
    home = tmp_path / "home"
    project = tmp_path / "project"
    for rel, text in {**_LIB, "pipeline.py": _RUNTIME_IMPORT, "data/raw/lines.txt": _LINES}.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    _ok(helpers.run_pivot(project, home, "init"))
    return project


def _value(project: pathlib.Path) -> object:
    return yaml.safe_load((project / "data" / "t" / "count.yaml").read_text())["v"]


def _edit_lib(project: pathlib.Path) -> None:
    (project / "helpers" / "lib.py").write_text("VALUE = 100\n")


def _leave_0_5_2_state(project: pathlib.Path, home: pathlib.Path) -> None:
    """Leave the project as a `pivot repro` at 0.5.2 leaves it: its lock, and its run-cache entry.

    One run supplies the output object and the run history; then the lock is replaced by 0.5.2's
    bytes and the run-cache entry 0.5.2 files for this input is written. 0.5.2 wrote, for this
    project, key `runcache:t/count:6656b60bc9776467` and value
    `{"output_hashes":[{"hash":"2ad953167adbd9db","path":"t/count"}],"run_id":"<its run>"}`; the
    input hash is read back from this version's run history, which records the same one.
    """
    _ok(helpers.run_pivot(project, home, "repro"))
    assert _value(project) == 4
    lock = project / ".pivot" / "stages" / "t" / "count.lock"
    fixture = yaml.safe_load(_LOCK_0_5_2)
    written = yaml.safe_load(lock.read_text())
    assert written["deps"] == fixture["deps"], "input bytes differ from the 0.5.2 fixture"
    assert written["outs"] == fixture["outs"], "output bytes differ from the 0.5.2 fixture"
    lock.write_text(_LOCK_0_5_2)
    with state.StateDB(project / ".pivot") as db:
        (run,) = db.list_runs(limit=10)
        input_hash = run["stages"]["t/count"]["input_hash"]
        assert input_hash is not None
        entry = json.dumps(
            {
                "output_hashes": [{"hash": fixture["outs"][0]["hash"], "path": "t/count"}],
                "run_id": run["run_id"],
            },
            separators=(",", ":"),
        )
        db.put_raw(f"runcache:t/count:{input_hash}".encode(), entry.encode())
    _ok(helpers.run_pivot(project, home, "repro"))
    assert _value(project) == 4, "the 0.5.2 lock is reused, as the docs state"


def test_d1_forced_run_after_an_edit_is_not_undone_by_a_0_5_2_run_cache_entry(
    tmp_path: pathlib.Path,
) -> None:
    """After a forced recheck marks a dynamically importing stage, a later `repro` must not
    restore the same-input run-cache entry that a prior Pivot version filed."""
    home = tmp_path / "home"
    project = _project(tmp_path)
    _leave_0_5_2_state(project, home)
    _edit_lib(project)

    _ok(helpers.run_pivot(project, home, "repro", "--force", "t/count"))
    assert _value(project) == 103
    _ok(helpers.run_pivot(project, home, "repro"))

    assert _value(project) == 103, "the next repro restored the 0.5.2 output over the forced run's"


def test_d1_stage_a_forced_run_marked_reruns_on_the_next_edit(tmp_path: pathlib.Path) -> None:
    """A forced recheck marks the dynamically importing stage, so a later module edit reruns it
    instead of restoring an older run-cache entry."""
    home = tmp_path / "home"
    project = _project(tmp_path)
    _leave_0_5_2_state(project, home)

    _ok(helpers.run_pivot(project, home, "repro", "--force", "t/count"))
    _edit_lib(project)
    _ok(helpers.run_pivot(project, home, "repro"))

    assert _value(project) == 103, (
        "the edit after the forced re-check was served from the run cache"
    )


def test_d1_forced_run_after_no_commit_and_commit_is_not_undone(tmp_path: pathlib.Path) -> None:
    """A forced flagged run after `repro --no-commit` and `pivot commit` must not be undone by the
    run-cache entry that commit filed."""
    home = tmp_path / "home"
    project = _project(tmp_path)
    _ok(helpers.run_pivot(project, home, "repro", "--no-commit"))
    _ok(helpers.run_pivot(project, home, "commit"))
    _edit_lib(project)

    _ok(helpers.run_pivot(project, home, "repro", "--force", "t/count"))
    assert _value(project) == 103
    _ok(helpers.run_pivot(project, home, "repro"))

    assert _value(project) == 103, (
        "the next repro restored the committed output over the forced run's"
    )


@pytest.mark.parametrize("repeat", [1, 2])
def test_d1_control_without_a_prior_entry_the_flagged_stage_reruns(
    tmp_path: pathlib.Path, repeat: int
) -> None:
    """Without an older run-cache entry, a flagged stage reruns on every `repro` after an edit."""
    home = tmp_path / "home"
    project = _project(tmp_path)
    _ok(helpers.run_pivot(project, home, "repro"))
    _edit_lib(project)
    for _ in range(repeat):
        _ok(helpers.run_pivot(project, home, "repro"))
    assert _value(project) == 103


def _latest_run_cache_entry(project: pathlib.Path) -> object:
    """The run-cache entry at the input hash the latest run of `t/count` recorded."""
    with state.StateDB(project / ".pivot", readonly=True) as db:
        run = db.list_runs(limit=1)[0]
        input_hash = run["stages"]["t/count"]["input_hash"]
        assert input_hash is not None
        return db.lookup_run_cache("t/count", input_hash)


def test_flagged_run_deletes_the_run_cache_entry_at_its_input_hash(
    tmp_path: pathlib.Path,
) -> None:
    """A forced run the stage-import check flags removes the entry 0.5.2 filed at its own input
    hash, so no later lookup without the mark can find that older result."""
    home = tmp_path / "home"
    project = _project(tmp_path)
    _leave_0_5_2_state(project, home)
    assert _latest_run_cache_entry(project) is not None, "the 0.5.2 entry was not set up"

    _ok(helpers.run_pivot(project, home, "repro", "--force", "t/count"))

    assert _latest_run_cache_entry(project) is None


def test_status_explain_reports_a_marked_stage_as_will_run_not_run_cache(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot status --explain` on a stage whose lock carries the uncovered-import mark, with a
    0.5.2 run-cache entry at its input hash: the stage will run with the uncovered-import reason,
    never as cached from the run cache, and `pivot verify` fails it with that reason; `repro`
    then runs it."""
    home = tmp_path / "home"
    project = _project(tmp_path)
    _leave_0_5_2_state(project, home)
    _ok(helpers.run_pivot(project, home, "repro", "--force", "t/count"))
    _edit_lib(project)

    explain = helpers.run_pivot(project, home, "status", "--explain", "t/count")
    verify = helpers.run_pivot(project, home, "verify")

    _ok(explain)
    text = " ".join((explain.stdout + explain.stderr).split())
    assert "Status: WILL RUN" in text, text
    assert "Previous run imported modules its fingerprint does not cover" in text, text
    assert "run cache" not in text, text
    verified = " ".join((verify.stdout + verify.stderr).split())
    assert verify.returncode == 1, verified
    assert "t/count: failed" in verified, verified
    assert "Previous run imported modules its fingerprint does not cover" in verified, verified
    rerun = helpers.run_pivot(project, home, "repro", "--json")
    _ok(rerun)
    assert '"stage": "t/count", "status": "ran"' in rerun.stdout, rerun.stdout
    assert _value(project) == 103


def _set_lines(project: pathlib.Path, text: str) -> None:
    (project / "data" / "raw" / "lines.txt").write_text(text)


def _run_cache_keys(project: pathlib.Path) -> list[bytes]:
    with state.StateDB(project / ".pivot", readonly=True) as db:
        return [key for key, _ in db.iter_prefix(b"runcache:t/count:")]


def test_marked_lock_is_not_satisfied_from_an_entry_at_another_input_hash(
    tmp_path: pathlib.Path,
) -> None:
    """0.5.2 filed an entry for input Y. The input changes to X, the imported module is edited,
    and `repro --force` at X marks the lock (deleting only X's entry). Back at Y, the marked lock
    must not be satisfied from Y's entry: the stage runs and gives 103, not 0.5.2's 4."""
    home = tmp_path / "home"
    project = _project(tmp_path)
    _leave_0_5_2_state(project, home)
    _set_lines(project, _LINES + "Lemma b\n")
    _edit_lib(project)
    _ok(helpers.run_pivot(project, home, "repro", "--force", "t/count"))
    assert _value(project) == 104
    assert _run_cache_keys(project), "the entry 0.5.2 filed for Y is gone"

    _set_lines(project, _LINES)
    _ok(helpers.run_pivot(project, home, "repro"))

    assert _value(project) == 103, "the marked lock was satisfied from Y's 0.5.2 entry"


def test_commit_carrying_the_mark_deletes_the_entry_at_its_input_hash(
    tmp_path: pathlib.Path,
) -> None:
    """0.5.2 filed an entry for input Y. A forced run at X marks the lock; back at Y,
    `repro --no-commit` then `pivot commit` records Y's outputs and carries the mark, so it
    deletes the entry at Y's input hash: no run-cache entry for the stage remains."""
    home = tmp_path / "home"
    project = _project(tmp_path)
    _leave_0_5_2_state(project, home)
    _set_lines(project, _LINES + "Lemma b\n")
    _ok(helpers.run_pivot(project, home, "repro", "--force", "t/count"))
    _set_lines(project, _LINES)
    _ok(helpers.run_pivot(project, home, "repro", "--no-commit"))
    assert _run_cache_keys(project), "the entry 0.5.2 filed for Y is gone before the commit"

    _ok(helpers.run_pivot(project, home, "commit"))

    assert _run_cache_keys(project) == []
    lock = (project / ".pivot" / "stages" / "t" / "count.lock").read_text()
    assert "uncovered-import:" in lock
