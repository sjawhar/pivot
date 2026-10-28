"""Every .pvt sidecar that `pivot track` accepts stays visible to the commands that read them.

A file tracked before a pipeline declares the output around it leaves its sidecar inside that
output, and `pivot track` accepts paths under any directory named `.git` or `.pivot` other than the
project's own state. A later command that loads another pipeline, or that walks the project, must
still find those sidecars: list them in status, refuse to run while one is missing, restore them on
checkout and select them on fetch and pull. Rows D1 and D2 of the PR #513 audit
(PR513_ASTRA_DEEP.md) name each case.
"""

from __future__ import annotations

import json
import shutil
import uuid
from typing import TYPE_CHECKING

import boto3
import pytest
import yaml

from pivot import cli
from pivot.storage import track
from pivot.types import TrackedFileStatus
from tests.conftest import isolated_pivot_dir

if TYPE_CHECKING:
    import pathlib

    import click.testing


_ROOT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
"""

_ROOT_PIPELINE_INCLUDING_B = """\
from other.pipeline import pipeline as b
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
    pipeline.include(b)
"""

# Pipeline b's stage writes a directory output at data/b/tree/
_OTHER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {"g.txt": "g"}


with Pipeline("b") as pipeline:
    tree()
"""

# Stage p/use reads the tracked file sub/.pivot/raw.txt
_HIDDEN_INPUT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def use(raw: str) -> str:
    return raw + "-used"


with Pipeline("p") as pipeline:
    use(pipeline.input("raw.txt", path="sub/.pivot/raw.txt"))
"""


def _helper_write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _helper_ok(runner: click.testing.CliRunner, *args: str) -> str:
    result = runner.invoke(cli.cli, list(args))
    assert result.exit_code == 0, f"pivot {' '.join(args)} failed:\n{result.output}"
    return result.output


def _helper_tracked_status(runner: click.testing.CliRunner, *flags: str) -> dict[str, str]:
    """Map each tracked file `pivot status --tracked-only --json` lists to its status."""
    result = runner.invoke(cli.cli, ["status", "--tracked-only", "--json", *flags])
    assert result.exit_code == 0, result.output
    return {entry["path"]: entry["status"] for entry in json.loads(result.stdout)["tracked_files"]}


@pytest.fixture
def bucket_url(moto_patch_session: None) -> str:
    """A fresh moto S3 bucket, as the URL of a Pivot remote under its ``cache/`` prefix."""
    name = f"pivot-pvt-walk-{uuid.uuid4().hex[:8]}"
    boto3.Session(region_name="us-east-1").client("s3").create_bucket(  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
        Bucket=name
    )
    return f"s3://{name}/cache"


# =============================================================================
# D1: a sidecar inside another pipeline's output
# =============================================================================

_LOADS_B = pytest.mark.parametrize(
    ("root_after_track", "flags"),
    [
        pytest.param(_ROOT_PIPELINE, ["--all"], id="all-pipelines"),
        pytest.param(_ROOT_PIPELINE_INCLUDING_B, [], id="root-includes-b"),
    ],
)


def _helper_missing_file_inside_b(
    runner: click.testing.CliRunner, cwd: pathlib.Path, root_after_track: str
) -> pathlib.Path:
    """Track data/b/tree/f.txt before b exists, delete it, then write b and root_after_track.

    Pipeline b, in other/pipeline.py, declares data/b/tree/ as its stage's directory output.
    `pivot track` refuses a path inside any pipeline's declared output, so b is written after it.
    """
    _helper_write(cwd / "pipeline.py", _ROOT_PIPELINE)
    tracked = cwd / "data" / "b" / "tree" / "f.txt"
    _helper_write(tracked, "tracked bytes")
    track_output = _helper_ok(runner, "track", "data/b/tree/f.txt")
    assert "Tracked: data/b/tree/f.txt" in track_output, track_output
    tracked.unlink()
    _helper_write(cwd / "other" / "pipeline.py", _OTHER_PIPELINE)
    _helper_write(cwd / "pipeline.py", root_after_track)
    return tracked


@_LOADS_B
def test_status_lists_a_missing_file_tracked_inside_another_pipelines_output(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    root_after_track: str,
    flags: list[str],
) -> None:
    """Audit D1, second-pipeline track, status row: loading b keeps the file in status.

    `data/b/tree/f.txt` was tracked before pipeline b declared that directory output. Loading b
    too (`--all`, or `a` including b) must not hide that file.
    """
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_missing_file_inside_b(runner, cwd, root_after_track)

        assert _helper_tracked_status(runner, *flags).get("data/b/tree/f.txt") == (
            TrackedFileStatus.MISSING
        ), "status must list the tracked file inside b's output as missing"


@_LOADS_B
def test_repro_refuses_a_missing_file_tracked_inside_another_pipelines_output(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    root_after_track: str,
    flags: list[str],
) -> None:
    """Audit D1, second-pipeline track, repro row: the run stops on the missing tracked file."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        tracked = _helper_missing_file_inside_b(runner, cwd, root_after_track)

        repro = runner.invoke(cli.cli, ["repro", *flags])

        assert repro.exit_code != 0, f"repro must refuse a missing tracked file:\n{repro.output}"
        assert "The following tracked files are missing" in repro.output, repro.output
        assert "data/b/tree/f.txt" in repro.output, repro.output
        assert track.get_pvt_path(tracked).is_file(), "the refused run keeps the sidecar"


@_LOADS_B
def test_checkout_restores_a_missing_file_tracked_inside_another_pipelines_output(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    root_after_track: str,
    flags: list[str],
) -> None:
    """Audit D1, second-pipeline track, checkout row: checkout restores the file from cache."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        tracked = _helper_missing_file_inside_b(runner, cwd, root_after_track)

        checkout_output = _helper_ok(runner, "checkout", *flags)

        assert "Restored 1 file(s)" in checkout_output, checkout_output
        assert tracked.is_file(), "checkout must restore the tracked file"
        assert tracked.read_text() == "tracked bytes"


# =============================================================================
# D2: a sidecar under a directory named .git or .pivot that is not pivot's state
# =============================================================================


def test_tracking_a_file_under_a_nested_pivot_dir_again_needs_force(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Audit D2, repeated track: a second track without --force is refused, sidecar unchanged."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_write(cwd / "pipeline.py", _ROOT_PIPELINE)
        data = cwd / "sub" / ".pivot" / "raw.txt"
        _helper_write(data, "first")
        _helper_ok(runner, "track", "sub/.pivot/raw.txt")
        pvt_path = track.get_pvt_path(data)
        sidecar_before = pvt_path.read_bytes()
        data.write_text("second, longer")

        again = runner.invoke(cli.cli, ["track", "sub/.pivot/raw.txt"])

        assert again.exit_code == 1, f"a second track without --force must fail:\n{again.output}"
        assert "is already tracked. Use --force to update." in again.output, again.output
        assert pvt_path.read_bytes() == sidecar_before, "the sidecar must not change"


_NESTED_STATE_NAMED_PATHS = pytest.mark.parametrize(
    "rel_path",
    [
        pytest.param("sub/.pivot/u.txt", id="nested-pivot"),
        pytest.param("child/.git/user.txt", id="nested-git"),
    ],
)


def _helper_missing_tracked_file(
    runner: click.testing.CliRunner, cwd: pathlib.Path, rel_path: str
) -> pathlib.Path:
    """Track rel_path in a project with a one-stage root pipeline, then delete it."""
    _helper_write(cwd / "pipeline.py", _ROOT_PIPELINE)
    data = cwd / rel_path
    _helper_write(data, "user bytes")
    _helper_ok(runner, "track", rel_path)
    data.unlink()
    return data


@_NESTED_STATE_NAMED_PATHS
def test_status_lists_a_missing_file_tracked_under_a_nested_state_named_dir(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, rel_path: str
) -> None:
    """Audit D2, status row: a missing file under a nested `.pivot`/`.git` is listed."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_missing_tracked_file(runner, cwd, rel_path)

        assert _helper_tracked_status(runner).get(rel_path) == TrackedFileStatus.MISSING, (
            f"status must list {rel_path} as missing"
        )


@_NESTED_STATE_NAMED_PATHS
def test_checkout_restores_a_file_tracked_under_a_nested_state_named_dir(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, rel_path: str
) -> None:
    """Audit D2, checkout row: a missing file under a nested `.pivot`/`.git` is restored."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        data = _helper_missing_tracked_file(runner, cwd, rel_path)

        checkout_output = _helper_ok(runner, "checkout")

        assert "Restored 1 file(s)" in checkout_output, checkout_output
        assert data.is_file(), "checkout must restore the tracked file"
        assert data.read_text() == "user bytes"


def _helper_missing_file_under_pivot_with_moved_state(
    runner: click.testing.CliRunner, cwd: pathlib.Path
) -> pathlib.Path:
    """With `core.state_dir: .state`, track .pivot/u.txt, a user file, run, then delete it.

    The run opens the state database, which shows pivot uses .state as its state directory.
    """
    _helper_write(cwd / ".pivot" / "config.yaml", yaml.safe_dump({"core": {"state_dir": ".state"}}))
    _helper_write(cwd / "pipeline.py", _ROOT_PIPELINE)
    data = cwd / ".pivot" / "u.txt"
    _helper_write(data, "user bytes")
    _helper_ok(runner, "track", ".pivot/u.txt")
    _helper_ok(runner, "repro")
    assert (cwd / ".state" / "state.lmdb").is_dir(), (
        "pivot's state database must live in the configured .state"
    )
    data.unlink()
    return data


def test_status_lists_a_missing_file_tracked_under_pivot_dir_when_state_moved(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Audit D2, hidden imports and state configuration: with the state in .state, a missing
    tracked .pivot/u.txt is listed by status."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_missing_file_under_pivot_with_moved_state(runner, cwd)

        assert _helper_tracked_status(runner).get(".pivot/u.txt") == TrackedFileStatus.MISSING, (
            "status must list .pivot/u.txt as missing"
        )


def test_checkout_restores_a_file_tracked_under_pivot_dir_when_state_moved(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Audit D2, hidden imports and state configuration: with the state in .state, checkout
    restores a missing tracked .pivot/u.txt."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        data = _helper_missing_file_under_pivot_with_moved_state(runner, cwd)

        checkout_output = _helper_ok(runner, "checkout")

        assert "Restored 1 file(s)" in checkout_output, checkout_output
        assert data.is_file(), "checkout must restore the tracked file"
        assert data.read_text() == "user bytes"


def _helper_pushed_then_cleared(
    runner: click.testing.CliRunner, cwd: pathlib.Path, bucket_url: str
) -> pathlib.Path:
    """Run p/use on tracked sub/.pivot/raw.txt, push, then delete the local cache and the input.

    Returns the input's path. Locks, sidecar and the stage output stay in the workspace.
    """
    _helper_write(
        cwd / ".pivot" / "config.yaml", yaml.safe_dump({"remotes": {"origin": bucket_url}})
    )
    _helper_write(cwd / "pipeline.py", _HIDDEN_INPUT_PIPELINE)
    data = cwd / "sub" / ".pivot" / "raw.txt"
    _helper_write(data, "raw")
    _helper_ok(runner, "track", "sub/.pivot/raw.txt")
    _helper_ok(runner, "repro")
    assert (cwd / "data" / "p" / "use.txt").read_text() == "raw-used"
    push_output = _helper_ok(runner, "push")
    assert "Pushed to 'origin': 2 transferred" in push_output, push_output
    shutil.rmtree(cwd / ".pivot" / "cache")
    data.unlink()
    return data


def test_fetch_dry_run_counts_a_tracked_input_under_a_nested_pivot_dir(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, bucket_url: str
) -> None:
    """Audit D2, remote row `fetch --dry-run`: the stage's tracked input is selected too."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_pushed_then_cleared(runner, cwd, bucket_url)

        fetch_output = _helper_ok(runner, "fetch", "--dry-run", "p/use")

        assert "Would fetch 2 file(s) from 'origin'" in fetch_output, (
            f"fetch must select the tracked input and the output:\n{fetch_output}"
        )


def test_pull_of_a_tracked_file_under_a_nested_pivot_dir_restores_it(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, bucket_url: str
) -> None:
    """Audit D2, remote row `pull <file>`: exits 0 and restores the tracked input."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        data = _helper_pushed_then_cleared(runner, cwd, bucket_url)

        pull_output = _helper_ok(runner, "pull", "sub/.pivot/raw.txt")

        assert "Restored 1 file(s)" in pull_output, pull_output
        assert data.is_file(), "pull <file> must restore the tracked input"
        assert data.read_text() == "raw"


def test_pull_all_restores_a_tracked_input_under_a_nested_pivot_dir(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, bucket_url: str
) -> None:
    """Audit D2, remote row `pull --all`: the tracked input is present afterwards."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        data = _helper_pushed_then_cleared(runner, cwd, bucket_url)

        _helper_ok(runner, "pull", "--all")

        assert data.is_file(), "pull --all must restore the tracked input"
        assert data.read_text() == "raw"
        assert (cwd / "data" / "p" / "use.txt").read_text() == "raw-used"
