"""A backslash in a file or directory name is an ordinary character.

Pivot runs on POSIX only, where ``/`` is the one path separator. Each test tracks a file whose
path holds a backslash and checks that a command finds the file at that path. The restore tests
also check that nothing is written where reading each backslash as a separator would put it.

The setup tracks each file from its own directory by its bare name. ``status``, ``repro`` and
``checkout`` without a target then get the path from the ``.pvt`` file alone; ``track`` from the
root, ``checkout <path>``, ``push <path>`` and ``pull <path>`` also get it on the command line.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import shutil
from typing import TYPE_CHECKING

import boto3
import pytest

from conftest import isolated_pivot_dir
from pivot import cli, config
from pivot.storage import track

if TYPE_CHECKING:
    import click.testing

# One stage copies the tracked input at DATA_PATH to data/test/copy.txt.
_COPY_PIPELINE = """\
from pivot.compose import Pipeline, stage


@stage
def copy(data: str) -> str:
    return data


with Pipeline("test") as pipeline:
    copy(pipeline.input("data.csv", path=DATA_PATH, t=str))
"""

_CONTENT = "tracked\n"

# (directory that holds the project, tracked path relative to the project root)
_LAYOUTS = [
    pytest.param("", "dir\\with\\backslash/f.csv", id="backslash-in-directory-name"),
    pytest.param("", "a\\b.csv", id="backslash-in-file-name"),
    pytest.param("project\\parent", "f.csv", id="backslash-in-project-root"),
]


def _helper_write(cwd: pathlib.Path, rel_path: str) -> pathlib.Path:
    """Write pipeline.py, whose stage reads ``rel_path``, and the file at ``rel_path``."""
    (cwd / "pipeline.py").write_text(_COPY_PIPELINE.replace("DATA_PATH", repr(rel_path)))
    data = cwd / rel_path
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(_CONTENT)
    return data


def _helper_track(
    runner: click.testing.CliRunner, cwd: pathlib.Path, rel_path: str
) -> pathlib.Path:
    """Write the project, then ``pivot track`` the file from its own directory by name."""
    data = _helper_write(cwd, rel_path)
    with contextlib.chdir(data.parent):
        result = runner.invoke(cli.cli, ["track", data.name])
    assert result.exit_code == 0, result.output
    return data


def _helper_split(data: pathlib.Path) -> pathlib.Path:
    """Where ``data`` lands when each backslash in its path is read as a separator."""
    return pathlib.Path(str(data).replace("\\", "/"))


def _helper_project_parent(tmp_path: pathlib.Path, parent: str) -> pathlib.Path:
    project_parent = tmp_path / parent
    project_parent.mkdir(exist_ok=True)
    return project_parent


@pytest.mark.parametrize(("parent", "rel_path"), _LAYOUTS)
@pytest.mark.parametrize(
    "target", [pytest.param("{rel}", id="relative"), pytest.param("{abs}", id="absolute")]
)
def test_track_from_the_project_root_takes_a_path_holding_a_backslash(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    parent: str,
    rel_path: str,
    target: str,
) -> None:
    with isolated_pivot_dir(runner, _helper_project_parent(tmp_path, parent)) as cwd:
        data = _helper_write(cwd, rel_path)
        result = runner.invoke(cli.cli, ["track", target.format(rel=rel_path, abs=data)])
        sidecar = track.read_pvt_file(data.with_name(f"{data.name}.pvt"))

    assert result.exit_code == 0, result.output
    assert sidecar is not None
    assert sidecar["path"] == data.name


@pytest.mark.parametrize(("parent", "rel_path"), _LAYOUTS)
def test_status_lists_a_tracked_path_holding_a_backslash_as_clean(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, parent: str, rel_path: str
) -> None:
    with isolated_pivot_dir(runner, _helper_project_parent(tmp_path, parent)) as cwd:
        _helper_track(runner, cwd, rel_path)
        result = runner.invoke(cli.cli, ["--quiet", "status", "--tracked-only", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["tracked_files"] == [
        {"path": rel_path, "status": "clean", "size": len(_CONTENT)}
    ]


@pytest.mark.parametrize(("parent", "rel_path"), _LAYOUTS)
def test_repro_reads_a_tracked_input_whose_path_holds_a_backslash(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, parent: str, rel_path: str
) -> None:
    with isolated_pivot_dir(runner, _helper_project_parent(tmp_path, parent)) as cwd:
        _helper_track(runner, cwd, rel_path)
        result = runner.invoke(cli.cli, ["repro"])
        output = cwd / "data" / "test" / "copy.txt"
        copied = output.read_text() if output.exists() else None

    assert result.exit_code == 0, result.output
    assert copied == _CONTENT


@pytest.mark.parametrize(("parent", "rel_path"), _LAYOUTS)
@pytest.mark.parametrize(
    "targets", [pytest.param((), id="all"), pytest.param(("{rel}",), id="target")]
)
def test_checkout_restores_a_tracked_path_holding_a_backslash_in_place(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    parent: str,
    rel_path: str,
    targets: tuple[str, ...],
) -> None:
    with isolated_pivot_dir(runner, _helper_project_parent(tmp_path, parent)) as cwd:
        data = _helper_track(runner, cwd, rel_path)
        data.unlink()
        result = runner.invoke(
            cli.cli, ["checkout", *(target.format(rel=rel_path) for target in targets)]
        )
        restored = data.read_text() if data.exists() else None
        split_written = _helper_split(data).exists()

    assert result.exit_code == 0, result.output
    assert restored == _CONTENT
    assert not split_written, "checkout wrote the file where a backslash read as '/' puts it"


@pytest.mark.parametrize(("parent", "rel_path"), _LAYOUTS)
def test_repro_checkout_missing_restores_a_tracked_input_whose_path_holds_a_backslash(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, parent: str, rel_path: str
) -> None:
    with isolated_pivot_dir(runner, _helper_project_parent(tmp_path, parent)) as cwd:
        data = _helper_track(runner, cwd, rel_path)
        data.unlink()
        result = runner.invoke(cli.cli, ["repro", "--checkout-missing"])
        restored = data.read_text() if data.exists() else None
        split_written = _helper_split(data).exists()

    assert result.exit_code == 0, result.output
    assert restored == _CONTENT
    assert not split_written, "repro wrote the file where a backslash read as '/' puts it"


@pytest.mark.usefixtures("moto_patch_session")
@pytest.mark.parametrize(("parent", "rel_path"), _LAYOUTS)
def test_push_and_pull_a_tracked_path_holding_a_backslash_by_target(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, parent: str, rel_path: str
) -> None:
    bucket = f"pivot-backslash-{tmp_path.name[-8:].lower().replace('_', '-')}"
    boto3.Session(region_name="us-east-1").client("s3").create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
    with isolated_pivot_dir(runner, _helper_project_parent(tmp_path, parent)) as cwd:
        (cwd / ".pivot" / "config.yaml").write_text(f"remotes:\n  origin: s3://{bucket}/c\n")
        data = _helper_track(runner, cwd, rel_path)
        push = runner.invoke(cli.cli, ["push", rel_path])
        data.unlink()
        shutil.rmtree(config.get_cache_dir() / "files")
        pull = runner.invoke(cli.cli, ["pull", rel_path])
        restored = data.read_text() if data.exists() else None
        split_written = _helper_split(data).exists()

    assert push.exit_code == 0, push.output
    assert "Pushed to 'origin': 1 transferred" in push.output
    assert pull.exit_code == 0, pull.output
    assert restored == _CONTENT
    assert not split_written, "pull wrote the file where a backslash read as '/' puts it"


def test_checkout_keeps_a_sidecar_path_holding_backslashes_inside_the_project(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """``path: ..\\..\\escaped.csv`` names one file beside its sidecar.

    Read with each backslash as a separator, it climbs two directories out of the project.
    """
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "seed.csv").write_text(_CONTENT)
        assert runner.invoke(cli.cli, ["track", "seed.csv"]).exit_code == 0
        seed = track.read_pvt_file(cwd / "seed.csv.pvt")
        assert seed is not None
        (cwd / "seed.csv.pvt").unlink()
        (cwd / "seed.csv").unlink()
        (cwd / "sub").mkdir()
        track.write_pvt_file(
            cwd / "sub" / "x.pvt",
            track.PvtData(path="..\\..\\escaped.csv", hash=seed["hash"], size=seed["size"]),
        )

        result = runner.invoke(cli.cli, ["checkout"])
        inside = cwd / "sub" / "..\\..\\escaped.csv"
        restored = inside.read_text() if inside.exists() else None
        escaped = (cwd.parent / "escaped.csv").exists()

    assert result.exit_code == 0, result.output
    assert restored == _CONTENT
    assert not escaped, "checkout wrote a file outside the project"
