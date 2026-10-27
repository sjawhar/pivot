"""What a restore may replace, and what Pivot records before it writes.

A restore requested through a stage is that stage's output restore, whatever else the path is.
Pivot records a stage's `*.pvt` member as the stage's before it writes the file, on every path
that writes one. A forced checkout checks every target before it changes any.
"""

from __future__ import annotations

import contextlib
import stat
from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli, project, types
from pivot.remote import storage as remote_storage
from pivot.storage import track

if TYPE_CHECKING:
    import pathlib

    from click.testing import CliRunner
    from pytest_mock import MockerFixture

    from conftest import GitRepo


_A_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
"""

_FILE_OUTPUT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def single() -> str:
    return "stage-written"


with Pipeline("b") as pipeline:
    single()
"""

_DIRECTORY_OUTPUT_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {"g.txt": "g"}


with Pipeline("b") as pipeline:
    tree()
"""

_PVT_MEMBER_SOMETIMES_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, StageParams, directory, stage


class TreeParams(StageParams):
    n: int = 1


@stage
def tree(params: TreeParams) -> Annotated[dict[str, str], directory]:
    out = {"value.txt": str(params.n)}
    if params.n == 2:
        out["notes.pvt"] = "ordinary stage output"
    return out


with Pipeline("b") as pipeline:
    tree(params=TreeParams())
"""


@pytest.mark.parametrize(
    ("pipeline", "output", "stage"),
    [
        pytest.param(_FILE_OUTPUT_PIPELINE, "data/b/single.txt", "b/single", id="file-output"),
        pytest.param(_DIRECTORY_OUTPUT_PIPELINE, "data/b/tree", "b/tree", id="directory-output"),
    ],
)
def test_forced_checkout_of_a_stage_target_keeps_a_tracked_file_at_its_output_path(
    runner: CliRunner, tmp_path: pathlib.Path, pipeline: str, output: str, stage: str
) -> None:
    """A tracked file sits at the path a stage declares as its output (tracked before the stage
    existed; the stage then ran over it, as main allowed, and the user restored the tracked file
    with `checkout <path> --force`). `repro --force`, `checkout --all --force` and
    `get --force` refuse to replace it. `checkout <stage> --force` (and `pull <stage> --force`,
    which runs it) restores the stage's output, not the tracked path's own content: it must
    refuse too, and keep the tracked file and its sidecar."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_A_PIPELINE)
        tracked = root / output
        tracked.parent.mkdir(parents=True)
        tracked.write_text("user bytes")
        result = runner.invoke(cli.cli, ["track", output])
        assert result.exit_code == 0, result.output
        sidecar = track.get_pvt_path(tracked)
        saved_sidecar = sidecar.read_bytes()
        # The stage runs over the path while the sidecar is away (main's behavior), then the user
        # puts the sidecar back and restores the tracked file
        sidecar.unlink()
        (root / "pipeline.py").write_text(pipeline)
        ran = runner.invoke(cli.cli, ["repro"])
        assert ran.exit_code == 0, ran.output
        sidecar.write_bytes(saved_sidecar)
        restored = runner.invoke(cli.cli, ["checkout", output, "--force"])
        assert restored.exit_code == 0, restored.output
        assert tracked.read_text() == "user bytes"

        result = runner.invoke(cli.cli, ["checkout", stage, "--force"])

        assert result.exit_code != 0, result.output
        assert tracked.is_file(), result.output
        assert tracked.read_text() == "user bytes"
        assert sidecar.read_bytes() == saved_sidecar


@pytest.mark.parametrize(
    "then",
    [
        pytest.param(["status"], id="status"),
        pytest.param(["commit"], id="commit"),
        pytest.param(["checkout", "--force"], id="checkout-force"),
        pytest.param(["repro"], id="repro-lock-params"),
        pytest.param(["repro", "--force"], id="repro-force-new-params"),
    ],
)
def test_pvt_member_a_no_commit_run_restored_from_the_run_cache_is_the_stages(
    runner: CliRunner, tmp_path: pathlib.Path, then: list[str]
) -> None:
    """No user file anywhere. A `repro --no-commit` run that the run cache serves writes the
    cached output, with the stage's own `notes.pvt` member, into the workspace; no lock and no
    `--no-commit` record lists it. It is still the stage's: the next command warns about nothing
    and succeeds, as at main."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_SOMETIMES_PIPELINE)
        params = root / "params.yaml"
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        params.write_text("b/tree:\n  n: 2\n")
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        params.unlink()
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        params.write_text("b/tree:\n  n: 2\n")
        pending = runner.invoke(cli.cli, ["repro", "--no-commit"])
        assert pending.exit_code == 0, pending.output
        notes = root / "data" / "b" / "tree" / "notes.pvt"
        assert notes.read_text() == "ordinary stage output"
        if then == ["repro"]:
            params.unlink()
        elif then == ["repro", "--force"]:
            params.write_text("b/tree:\n  n: 3\n")

        result = runner.invoke(cli.cli, then)

        assert result.exit_code == 0, result.output
        assert "outside every stage output" not in result.output
        assert "Refusing" not in result.output


@pytest.mark.parametrize(
    "then",
    [
        pytest.param(["status"], id="status"),
        pytest.param(["repro"], id="repro-lock-params"),
        pytest.param(["checkout", "--force"], id="checkout-force"),
        pytest.param(["repro", "--force"], id="repro-force-new-params"),
    ],
)
def test_pvt_member_get_restored_from_a_revision_is_the_stages(
    runner: CliRunner,
    git_repo: GitRepo,
    monkeypatch: pytest.MonkeyPatch,
    then: list[str],
) -> None:
    """No user file anywhere. `pivot get --rev R <stage> --force` writes R's output, with the
    stage's own `notes.pvt` member, into the workspace; the current lock does not list it and no
    `--no-commit` run wrote it. It is still the stage's: the next command warns about nothing
    and succeeds, as at main."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    (repo_path / "pipeline.py").write_text(_PVT_MEMBER_SOMETIMES_PIPELINE)
    params = repo_path / "params.yaml"
    assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
    params.write_text("b/tree:\n  n: 2\n")
    assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
    sha = commit("n=2")
    params.unlink()
    assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
    restored = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "b/tree", "--force"])
    assert restored.exit_code == 0, restored.output
    notes = repo_path / "data" / "b" / "tree" / "notes.pvt"
    assert notes.read_text() == "ordinary stage output"
    if then == ["repro", "--force"]:
        params.write_text("b/tree:\n  n: 3\n")

    result = runner.invoke(cli.cli, then)

    assert result.exit_code == 0, result.output
    assert "outside every stage output" not in result.output
    assert "Refusing" not in result.output


def test_forced_checkout_all_changes_nothing_when_it_refuses_a_target(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """`checkout --all --force` refuses a stage output that holds a tracked file saved inside it
    (saved before the check existed). A command that refuses removes nothing: the other target,
    a tracked file with a local edit, stays as it is (`get --force` already refuses all of its
    targets when one is refused)."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_DIRECTORY_OUTPUT_PIPELINE)
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        # Saved while no pipeline declared data/b/tree (main's track allowed it in any state)
        (root / "pipeline.py").write_text(_A_PIPELINE)
        saved = root / "data" / "b" / "tree" / "f.txt"
        saved.write_text("saved")
        raw = root / "data" / "raw" / "in.txt"
        raw.parent.mkdir(parents=True)
        raw.write_text("raw v1")
        tracked = runner.invoke(cli.cli, ["track", "data/b/tree/f.txt", "data/raw/in.txt"])
        assert tracked.exit_code == 0, tracked.output
        (root / "pipeline.py").write_text(_DIRECTORY_OUTPUT_PIPELINE)
        raw.unlink()
        raw.write_text("local edit")

        result = runner.invoke(cli.cli, ["checkout", "--all", "--force"])

        assert result.exit_code != 0, result.output
        assert saved.read_text() == "saved"
        assert track.get_pvt_path(saved).is_file()
        assert raw.read_text() == "local edit", result.output


@contextlib.contextmanager
def _unreadable(path: pathlib.Path):
    """Make ``path`` mode 0 for the body, then restore it so tmp cleanup can remove it."""
    original = path.stat().st_mode
    path.chmod(0)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            path.chmod(original | stat.S_IRWXU)


def test_forced_get_keeps_a_tracked_file_under_an_unreadable_directory(
    runner: CliRunner, git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pivot get --rev R b/tree --force` restores R's output over the current output path. A
    subdirectory the user locked to mode 0 may hide a tracked file, so Pivot refuses, names the
    directory it cannot read, and keeps the file."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    (repo_path / "pipeline.py").write_text(_DIRECTORY_OUTPUT_PIPELINE)
    assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
    sha = commit("tree output")
    # Track a file inside the output while the stage is not registered (main allowed it).
    (repo_path / "pipeline.py").write_text(_A_PIPELINE)
    secret_dir = repo_path / "data" / "b" / "tree" / "sekret"
    secret_dir.mkdir(parents=True)
    tracked = secret_dir / "h.txt"
    tracked.write_text("user secret bytes")
    assert runner.invoke(cli.cli, ["track", str(tracked.relative_to(repo_path))]).exit_code == 0
    sidecar = track.get_pvt_path(tracked)
    assert sidecar.is_file()
    (repo_path / "pipeline.py").write_text(_DIRECTORY_OUTPUT_PIPELINE)

    with _unreadable(secret_dir):
        result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "b/tree", "--force"])

    assert tracked.is_file(), result.output
    assert tracked.read_text() == "user secret bytes"
    assert sidecar.is_file()
    assert result.exit_code != 0, result.output


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(["pull", "b/tree", "--force"], id="pull-stage-force"),
        pytest.param(["pull", "--force"], id="pull-all-force"),
    ],
)
def test_forced_pull_keeps_a_tracked_file_under_an_unreadable_directory(
    runner: CliRunner, tmp_path: pathlib.Path, mocker: MockerFixture, command: list[str]
) -> None:
    """`pivot pull --force` fetches, then restores through checkout. A subdirectory the user
    locked to mode 0 may hide a tracked file, so the checkout step refuses and keeps it."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_DIRECTORY_OUTPUT_PIPELINE)
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        (root / "pipeline.py").write_text(_A_PIPELINE)
        secret_dir = root / "data" / "b" / "tree" / "sekret"
        secret_dir.mkdir(parents=True)
        tracked = secret_dir / "h.txt"
        tracked.write_text("user secret bytes")
        assert runner.invoke(cli.cli, ["track", str(tracked.relative_to(root))]).exit_code == 0
        sidecar = track.get_pvt_path(tracked)
        assert sidecar.is_file()
        (root / "pipeline.py").write_text(_DIRECTORY_OUTPUT_PIPELINE)
        # The cache already holds the output, so the fetch step is a no-op; only the checkout
        # step runs, and that is where the ownership check must refuse.
        mocker.patch(
            "pivot.remote.sync.create_remote_from_name",
            return_value=(mocker.Mock(spec=remote_storage.S3Remote), "origin"),
        )
        mocker.patch(
            "pivot.remote.sync.pull",
            return_value=types.TransferSummary(transferred=0, skipped=0, failed=0, errors=[]),
        )

        with _unreadable(secret_dir):
            result = runner.invoke(cli.cli, command)

        assert tracked.is_file(), result.output
        assert tracked.read_text() == "user secret bytes"
        assert sidecar.is_file()
        assert result.exit_code != 0, result.output


def test_forced_checkout_all_leaves_a_sibling_untouched_when_a_directory_is_unreadable(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """`checkout --all --force` refuses a stage output when a subdirectory inside it is unreadable
    (it may hide a tracked file). A command that refuses removes nothing: a sibling target, a
    tracked file with a local edit, keeps that edit."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_DIRECTORY_OUTPUT_PIPELINE)
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        # Both tracked while no pipeline declared data/b/tree (main's track allowed it).
        (root / "pipeline.py").write_text(_A_PIPELINE)
        secret_dir = root / "data" / "b" / "tree" / "sekret"
        secret_dir.mkdir(parents=True)
        tracked = secret_dir / "h.txt"
        tracked.write_text("user secret bytes")
        raw = root / "data" / "raw" / "in.txt"
        raw.parent.mkdir(parents=True)
        raw.write_text("raw v1")
        result = runner.invoke(cli.cli, ["track", "data/b/tree/sekret/h.txt", "data/raw/in.txt"])
        assert result.exit_code == 0, result.output
        (root / "pipeline.py").write_text(_DIRECTORY_OUTPUT_PIPELINE)
        raw.unlink()
        raw.write_text("local edit")

        with _unreadable(secret_dir):
            result = runner.invoke(cli.cli, ["checkout", "--all", "--force"])

        assert result.exit_code != 0, result.output
        assert tracked.read_text() == "user secret bytes"
        assert track.get_pvt_path(tracked).is_file()
        assert raw.read_text() == "local edit", result.output
