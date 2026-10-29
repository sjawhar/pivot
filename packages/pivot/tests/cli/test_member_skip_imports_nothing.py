# pyright: reportUnknownArgumentType=false, reportArgumentType=false
"""Deciding which ``*.pvt`` files are a stage's own members imports no other pipeline.

A command loads its own pipeline once. A ``*.pvt`` outside that pipeline's outputs (here a
tracked input in ``data/raw``, the usual place for one) is decided from recorded state, so:

- ``pivot repro`` never imports the project's code a second time: a helper edited as the command
  starts leaves the lock describing the code the stage ran, and the next run reruns it;
- a pipeline the command does not run is never imported: one that cannot be combined with the
  loaded one fails nothing, and one that prints when imported writes nothing into
  ``status --json``'s stdout.

The tests are the review's of sjawhar/pivot#540 at 14484bae, kept as the reviewer wrote them.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli
from pivot.storage import track

if TYPE_CHECKING:
    from click.testing import CliRunner

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_HELPERS = 'def plain():\n    return "plain-v1"\n'
_PIPELINE = """\
import helpers
from pivot import Pipeline, stage


@stage
def use_plain(seed: str) -> str:
    return helpers.plain()


with Pipeline("p") as pipeline:
    use_plain(pipeline.input("seed.txt"))
"""


def _repro_json(root: pathlib.Path) -> dict[str, tuple[str, str]]:
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    proc = subprocess.run(
        [_PIVOT, "repro", "--json"], cwd=root, env=env, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    stages = dict[str, tuple[str, str]]()
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and "stage" in event and "status" in event:
            stages[str(event["stage"])] = (str(event["status"]), str(event.get("reason", "")))
    return stages


def test_a_helper_edited_as_repro_starts_is_not_cached_stale_with_a_tracked_input(
    runner: CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage runs the helper discovery loaded (plain-v1). Its lock must describe that code,
    so the next run sees the edit (plain-v2) and reruns the stage instead of reporting plain-v1
    as cached."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        helpers = root / "helpers.py"
        helpers.write_text(_HELPERS)
        (root / "pipeline.py").write_text(_PIPELINE)
        seed = root / "data" / "raw" / "seed.txt"
        seed.parent.mkdir(parents=True)
        seed.write_text("seed\n")
        tracked = runner.invoke(cli.cli, ["track", "data/raw/seed.txt"])
        assert tracked.exit_code == 0, tracked.output

        real = track.discover_pvt_files_and_update_memo

        def edit_then_walk(*args: object, **kwargs: object) -> dict[str, track.PvtData]:
            helpers.write_text(helpers.read_text().replace("-v1", "-v2"))
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(track, "discover_pvt_files_and_update_memo", edit_then_walk)
        first = runner.invoke(cli.cli, ["repro"])
        monkeypatch.setattr(track, "discover_pvt_files_and_update_memo", real)

        assert first.exit_code == 0, first.output
        output = root / "data" / "p" / "use_plain.txt"
        assert output.read_text() == "plain-v1", "the stage runs the helper discovery loaded"
        assert "plain-v2" in helpers.read_text()

        second = _repro_json(root)

        assert second["p/use_plain"][0] == "ran", second
        assert output.read_text() == "plain-v2"


_ROOT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
"""


def _cli(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    return subprocess.run(
        [_PIVOT, *args], cwd=root, env=env, capture_output=True, text=True, timeout=300
    )


def _root_with_a_tracked_input(
    tmp_path: pathlib.Path, other_pipeline: str, root_pipeline: str = _ROOT_PIPELINE
) -> pathlib.Path:
    """Pipeline `a` at the root, run once, with one tracked file in data/raw; another
    `pipeline.py` in `other/` that the commands run from the root never load on main."""
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    assert _cli(root, "init").returncode == 0
    (root / "pipeline.py").write_text(root_pipeline)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "kept.txt").write_text("kept")
    (raw / "note.txt").write_text("b's note")
    (root / "a").mkdir()
    (root / "a" / "note.txt").write_text("a's note")
    for args in (("track", "data/raw/kept.txt"), ("repro",)):
        proc = _cli(root, *args)
        assert proc.returncode == 0, proc.stdout + proc.stderr
    (root / "other").mkdir()
    (root / "other" / "pipeline.py").write_text(other_pipeline)
    return root


_SETS_ROOT = (
    "import pathlib\n\nfrom pivot import Pipeline, stage\n\n\n@stage\ndef t() -> str:\n"
    + '    return "t"\n\n\nwith Pipeline("b", root=pathlib.Path(__file__).parent) as pipeline:\n'
    + "    t()\n"
)

_NOTE_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def use(note: str) -> str:
    return note


with Pipeline("{name}") as pipeline:
    use(pipeline.input("note.txt"{path}))
"""


@pytest.mark.parametrize(
    ("other_pipeline", "root_pipeline"),
    [
        pytest.param(_SETS_ROOT, _ROOT_PIPELINE, id="other-sets-root"),
        pytest.param(
            _NOTE_PIPELINE.format(name="b", path=""),
            _NOTE_PIPELINE.format(name="a", path=', path="a/note.txt"'),
            id="input-name-bound-to-two-paths",
        ),
    ],
)
def test_commands_of_one_pipeline_run_beside_another_they_cannot_combine_with(
    tmp_path: pathlib.Path, other_pipeline: str, root_pipeline: str
) -> None:
    """`other/pipeline.py` cannot be combined with the root pipeline: it sets `root=`, which
    discovery refuses, or it binds the input name `note.txt` to another path than `a` does (each
    pipeline is valid on its own). Commands run from the root load pipeline `a` only, and main
    ran them; they must not fail over a pipeline they do not run."""
    root = _root_with_a_tracked_input(tmp_path, other_pipeline, root_pipeline)
    (root / "data" / "raw" / "kept.txt").unlink()

    results = {
        " ".join(args): _cli(root, *args)
        for args in (("status",), ("checkout", "--only-missing"), ("repro",), ("verify",))
    }

    for command, proc in results.items():
        assert proc.returncode == 0, f"{command}: {proc.stdout}{proc.stderr}"
    assert (root / "data" / "raw" / "kept.txt").read_text() == "kept"


def test_status_json_stays_one_document_beside_a_pipeline_that_prints_when_imported(
    tmp_path: pathlib.Path,
) -> None:
    """`status --json` writes one JSON document to stdout. A pipeline the command does not run
    prints when imported; its text must not reach stdout."""
    root = _root_with_a_tracked_input(
        tmp_path,
        'print("hello from b")\n' + _ROOT_PIPELINE.replace('Pipeline("a")', 'Pipeline("b")'),
    )

    proc = _cli(root, "status", "--tracked-only", "--json")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    document = json.loads(proc.stdout)
    assert [info["path"] for info in document["tracked_files"]] == ["data/raw/kept.txt"]
