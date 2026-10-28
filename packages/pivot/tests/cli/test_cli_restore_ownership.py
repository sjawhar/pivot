"""What a restore may replace, and what Pivot records before it writes.

A restore requested through a stage is that stage's output restore, whatever else the path is.
Pivot records a stage's `*.pvt` member as the stage's before it writes the file, on every path
that writes one. A forced checkout checks every target before it changes any, and checks each
target again just before it replaces it.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import shutil
import stat
import subprocess
from typing import TYPE_CHECKING, Any

import pytest

from conftest import isolated_pivot_dir
from pivot import cli, project, types
from pivot.cli import checkout as checkout_mod
from pivot.remote import storage as remote_storage
from pivot.storage import cache, track

if TYPE_CHECKING:
    import os

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


_TWO_TREES_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def one() -> Annotated[dict[str, str], directory]:
    return {"g.txt": "one"}


@stage
def two() -> Annotated[dict[str, str], directory]:
    return {"g.txt": "two"}


with Pipeline("b") as pipeline:
    one()
    two()
"""


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(["checkout", "--force"], id="checkout-force"),
        pytest.param(["checkout", "--all", "--force"], id="checkout-all-force"),
        pytest.param(["pull", "--force"], id="pull-force"),
        pytest.param(["get", "--rev", "HEAD", "b/one", "b/two", "--force"], id="get-force"),
    ],
)
def test_forced_restore_keeps_a_tracked_file_that_lands_in_an_output_while_it_runs(
    runner: CliRunner,
    git_repo: GitRepo,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    command: list[str],
) -> None:
    """A tracked file and its sidecar are copied into one output after the command checked every
    target, while it restores the other output. The restore checks each target again, under its
    restore lock, just before it replaces it: it refuses that target, names the sidecar, exits 1
    and keeps both files. The other output is still restored."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    (repo_path / "pipeline.py").write_text(_TWO_TREES_PIPELINE)
    assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
    user = repo_path / "data" / "raw" / "u.txt"
    user.parent.mkdir(parents=True)
    user.write_text("user bytes")
    assert runner.invoke(cli.cli, ["track", "data/raw/u.txt"]).exit_code == 0
    user_sidecar = track.get_pvt_path(user).read_bytes()
    commit("two trees")
    outputs = [(repo_path / "data" / "b" / name).resolve() for name in ("one", "two")]
    mocker.patch(
        "pivot.remote.sync.create_remote_from_name",
        return_value=(mocker.Mock(spec=remote_storage.S3Remote), "origin"),
    )
    mocker.patch(
        "pivot.remote.sync.pull",
        return_value=types.TransferSummary(transferred=0, skipped=0, failed=0, errors=[]),
    )
    # One restore at a time, so the second output's restore starts after the first one's
    monkeypatch.setattr(checkout_mod, "MAX_CONCURRENT_RESTORES", 1)
    landed = list[pathlib.Path]()
    restore_from_cache = cache.restore_from_cache

    def restore_while_a_file_lands(path: pathlib.Path, *args: Any, **kwargs: Any) -> bool:
        if not landed and path.resolve() in outputs:
            other = outputs[1] if path.resolve() == outputs[0] else outputs[0]
            shutil.copy2(user, other / "u.txt")
            shutil.copy2(track.get_pvt_path(user), other / "u.txt.pvt")
            landed.append(other)
        return restore_from_cache(path, *args, **kwargs)

    monkeypatch.setattr(cache, "restore_from_cache", restore_while_a_file_lands)

    result = runner.invoke(cli.cli, command)

    assert landed, result.output
    other = landed[0]
    assert (other / "u.txt").read_text() == "user bytes", result.output
    assert (other / "u.txt.pvt").read_bytes() == user_sidecar
    assert result.exit_code != 0, result.output
    assert f"data/b/{other.name}/u.txt.pvt" in " ".join(result.output.split())
    restored = outputs[0] if other == outputs[1] else outputs[1]
    assert (restored / "g.txt").read_text() == restored.name


_GET_BOTH_TREES = ["get", "--rev", "HEAD", "b/one", "b/two", "--force"]


@pytest.mark.parametrize(
    ("command", "sidecar"),
    [
        pytest.param(["checkout", "--force"], "u.txt.pvt", id="checkout-force"),
        pytest.param(["checkout", "--all", "--force"], "u.txt.pvt", id="checkout-all-force"),
        pytest.param(["pull", "--force"], "u.txt.pvt", id="pull-force"),
        pytest.param(_GET_BOTH_TREES, "u.txt.pvt", id="get-force"),
        pytest.param(["checkout", "--force"], "u\x1b[2J.txt.pvt", id="checkout-force-escape"),
        pytest.param(_GET_BOTH_TREES, "u\x1b[2J.txt.pvt", id="get-force-escape"),
    ],
)
def test_forced_restore_keeps_a_tracked_file_that_lands_in_an_output_during_its_own_restore(
    runner: CliRunner,
    git_repo: GitRepo,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    command: list[str],
    sidecar: str,
) -> None:
    """A tracked file and its sidecar are copied into an output after its restore checked it
    again, while that restore builds its replacement from the cache. The swap checks the old
    output once it has moved it aside: it refuses that target, names the sidecar, exits 1 and
    puts the output back with both files in it. The other output is still restored. A control
    character in the sidecar's name is printed escaped, never raw."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    (repo_path / "pipeline.py").write_text(_TWO_TREES_PIPELINE)
    assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
    user = repo_path / "data" / "raw" / "u.txt"
    user.parent.mkdir(parents=True)
    user.write_text("user bytes")
    assert runner.invoke(cli.cli, ["track", "data/raw/u.txt"]).exit_code == 0
    user_sidecar = track.get_pvt_path(user).read_bytes()
    commit("two trees")
    outputs = [(repo_path / "data" / "b" / name).resolve() for name in ("one", "two")]
    mocker.patch(
        "pivot.remote.sync.create_remote_from_name",
        return_value=(mocker.Mock(spec=remote_storage.S3Remote), "origin"),
    )
    mocker.patch(
        "pivot.remote.sync.pull",
        return_value=types.TransferSummary(transferred=0, skipped=0, failed=0, errors=[]),
    )
    monkeypatch.setattr(checkout_mod, "MAX_CONCURRENT_RESTORES", 1)
    landed = list[pathlib.Path]()
    checkout_with_fallback = cache._checkout_with_fallback

    def stage_while_a_file_lands(path: pathlib.Path, *args: Any, **kwargs: Any) -> os.stat_result:
        # A directory restore stages each file under a temporary name, naming the output file it
        # becomes; the first one staged lands the user's file in that same output.
        display = kwargs.get("display_path")
        if not landed and display is not None:
            (output,) = (out for out in outputs if display.resolve().is_relative_to(out))
            shutil.copy2(user, output / "u.txt")
            shutil.copy2(track.get_pvt_path(user), output / sidecar)
            landed.append(output)
        return checkout_with_fallback(path, *args, **kwargs)

    monkeypatch.setattr(cache, "_checkout_with_fallback", stage_while_a_file_lands)

    result = runner.invoke(cli.cli, command)

    assert landed, result.output
    output = landed[0]
    assert (output / "u.txt").read_text() == "user bytes", result.output
    assert (output / sidecar).read_bytes() == user_sidecar
    assert (output / "g.txt").read_text() == output.name, "the output went back unchanged"
    assert result.exit_code != 0, result.output
    shown = sidecar.replace("\x1b", "\\x1b")
    assert f"data/b/{output.name}/{shown}" in " ".join(result.output.split())
    assert "\x1b" not in result.output
    restored = outputs[0] if output == outputs[1] else outputs[1]
    assert (restored / "g.txt").read_text() == restored.name
    assert not [p.name for p in output.parent.iterdir() if p.name.startswith(".pivot_")]


_PATH_ONLY_DIRECTORY_PIPELINE = """\
import pathlib
import tempfile
from typing import Annotated

from pivot import Pipeline, loaders, stage

NOTES = {notes!r}


@stage
def build() -> Annotated[pathlib.Path, loaders.PathOnly()]:
    out = pathlib.Path(tempfile.mkdtemp()) / "build"
    (out / "sub").mkdir(parents=True)
    (out / "a.txt").write_text("a")
    (out / "sub" / "a.txt").write_text("a")
    (out / "notes.pvt").write_text(NOTES)
    (out / "sub" / "deep.pvt").write_text(NOTES)
    return out


with Pipeline("p") as pipeline:
    build()
"""


@pytest.mark.parametrize(
    "notes",
    [
        pytest.param("ordinary stage output", id="not-a-sidecar"),
        pytest.param("path: a.txt\nhash: 0123456789abcdef\nsize: 1\n", id="parses-as-sidecar"),
    ],
)
def test_path_only_directory_output_holding_its_own_pvt_files_runs_as_at_main(
    runner: CliRunner, tmp_path: pathlib.Path, notes: str
) -> None:
    """A stage returns a directory it built outside the project (`PathOnly`, the documented
    directory artifact), holding its own `*.pvt` files. They are the stage's, whatever the
    output's tag: `repro`, `repro --force`, `status`, `repro --force --no-commit` with
    `commit`, and `checkout <stage> --force` all succeed and keep them, as at main."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PATH_ONLY_DIRECTORY_PIPELINE.format(notes=notes))

        results = [
            runner.invoke(cli.cli, command)
            for command in (
                ["repro"],
                ["repro", "--force"],
                ["status"],
                ["repro", "--force", "--no-commit"],
                ["commit", "p/build"],
                ["checkout", "p/build", "--force"],
            )
        ]

        for result in results:
            assert result.exit_code == 0, result.output
            assert "Refusing" not in result.output, result.output
            assert "Warning:" not in result.output, result.output
        (output,) = (root / "data" / "p").glob("build*")
        assert (output / "notes.pvt").read_text() == notes
        assert (output / "sub" / "deep.pvt").read_text() == notes


_KILLED_MOVE_PIPELINE = """\
import os
import pathlib
import signal
import tempfile
from typing import Annotated

from pivot import Pipeline, directory, stage

KILL_MARK = {kill_mark!r}


@stage
def bundle() -> Annotated[dict[str, pathlib.Path], directory]:
    if os.path.exists(KILL_MARK):
        # The worker dies as SIGKILL or the OOM killer would stop it: while it moves the member
        # it staged, and has recorded, into the output
        replace = pathlib.Path.replace

        def replace_or_die(self, target):
            if ".pivot-staged" in self.name:
                os.kill(os.getpid(), signal.SIGKILL)
            return replace(self, target)

        pathlib.Path.replace = replace_or_die
    scratch = pathlib.Path(tempfile.mkdtemp())
    (scratch / "docs.d").mkdir()
    (scratch / "docs.d" / "notes.pvt").write_text("ordinary stage output")
    (scratch / "a.txt").write_text("a")
    return {{"docs.d": scratch / "docs.d", "a.txt": scratch / "a.txt"}}


with Pipeline("b") as pipeline:
    bundle()
"""


def test_worker_killed_while_moving_a_staged_member_leaves_nothing_that_blocks_the_output(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A directory output's `PathOnly` member is a directory holding a `*.pvt`. The worker stages
    it, records the `*.pvt` as the stage's, and is killed while it moves it into the output. The
    staged copy it leaves is beside the output, not in it, so the next run and a forced checkout
    succeed and keep the stage's file, rather than refuse it as a user's sidecar."""
    kill_mark = tmp_path / "kill-the-worker"
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_KILLED_MOVE_PIPELINE.format(kill_mark=str(kill_mark)))
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output
        (output,) = (root / "data" / "b").glob("bundle*")

        kill_mark.touch()
        killed = runner.invoke(cli.cli, ["repro", "--force"])
        kill_mark.unlink()

        assert killed.exit_code != 0, killed.output
        assert list((root / "data" / "b").rglob("*.pivot-staged*")), "no staged copy was left"
        assert not list(output.rglob("*.pivot-staged*")), "the staged copy is inside the output"
        results = [
            runner.invoke(cli.cli, command)
            for command in (["repro", "--force"], ["checkout", "--force"])
        ]
        for result in results:
            assert result.exit_code == 0, result.output
            assert "Refusing" not in result.output, result.output
        assert (output / "docs.d" / "notes.pvt").read_text() == "ordinary stage output"
        assert (output / "a.txt").read_text() == "a"


_SIDECAR_SHAPED_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, StageParams, directory, stage


class TreeParams(StageParams):
    n: int = 1


@stage
def tree(params: TreeParams) -> Annotated[dict[str, str], directory]:
    return {
        "value.txt": str(params.n),
        "fake.pvt": f"path: value.txt\\nhash: abcdef{params.n:010x}\\nsize: 1\\n",
    }


with Pipeline("b") as pipeline:
    tree(params=TreeParams())
"""


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(["checkout", "b/tree", "--force"], id="checkout-force"),
        pytest.param(["repro"], id="repro"),
    ],
)
def test_stage_sidecar_shaped_member_stays_the_stages_after_its_lock_changes_through_git(
    runner: CliRunner, git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    """A stage writes `fake.pvt`, which parses as a sidecar. The user checks out an older lock
    from git: it lists `fake.pvt` with the older content, and the workspace holds the newer
    content the stage wrote, which Pivot cached. That file is still the stage's: restoring the
    older lock replaces it, as at main."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    (repo_path / "pipeline.py").write_text(_SIDECAR_SHAPED_MEMBER_PIPELINE)
    assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
    fake = repo_path / "data" / "b" / "tree" / "fake.pvt"
    first = fake.read_text()
    assert track.read_pvt_file(fake) is not None
    sha = commit("n=1")
    params = repo_path / "params.yaml"
    params.write_text("b/tree:\n  n: 2\n")
    assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
    assert fake.read_text() != first
    commit("n=2")
    params.unlink()
    lock_path = ".pivot/stages/b/tree.lock"
    subprocess.run(["git", "checkout", sha, "--", lock_path], cwd=repo_path, check=True)

    result = runner.invoke(cli.cli, command)

    assert result.exit_code == 0, result.output
    assert "Refusing" not in result.output, result.output
    assert fake.read_text() == first


@pytest.mark.parametrize("value_present", [False, True], ids=["value-deleted", "value-present"])
def test_targetless_forced_checkout_restores_an_output_whose_own_member_is_sidecar_shaped(
    runner: CliRunner, tmp_path: pathlib.Path, value_present: bool
) -> None:
    """`fake.pvt` parses as a sidecar naming `value.txt`, with a hash no cache holds, and the
    stage's lock lists it: it is the stage's own member, not a tracked file. A checkout with no
    target restores `b/tree` whole, as `checkout b/tree` does, and restores nothing `fake.pvt`
    names."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_SIDECAR_SHAPED_MEMBER_PIPELINE)
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        tree = root / "data" / "b" / "tree"
        fake = tree / "fake.pvt"
        written = fake.read_text()
        assert track.read_pvt_file(fake) is not None
        if not value_present:
            (tree / "value.txt").unlink()

        result = runner.invoke(cli.cli, ["checkout", "--force"])

        assert result.exit_code == 0, result.output
        assert "Restored 2 file(s)" in result.output, result.output
        assert "Missing" not in result.output, result.output
        assert (tree / "value.txt").read_text() == "1"
        assert fake.read_text() == written


def test_repro_and_status_read_no_tracked_file_from_a_stages_own_sidecar_shaped_member(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """The repro pre-flight and `status` discover tracked files as checkout does: `fake.pvt`,
    the stage's own member, names no tracked file, so a deleted `value.txt` is no missing tracked
    file, and status lists none."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_SIDECAR_SHAPED_MEMBER_PIPELINE)
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        tree = root / "data" / "b" / "tree"
        (tree / "value.txt").unlink()

        status = runner.invoke(cli.cli, ["status", "--tracked-only", "--json"])
        result = runner.invoke(cli.cli, ["repro"])

        assert status.exit_code == 0, status.output
        assert json.loads(status.stdout)["tracked_files"] == [], status.stdout
        assert result.exit_code == 0, result.output
        assert (tree / "value.txt").read_text() == "1"


def test_track_force_refuses_a_path_only_a_stages_own_sidecar_shaped_member_names(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """`fake.pvt`, the stage's own member, names `value.txt` but tracks nothing: `track --force`
    refuses `value.txt` as a path inside the stage's output, and writes no sidecar there."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_SIDECAR_SHAPED_MEMBER_PIPELINE)
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        tree = root / "data" / "b" / "tree"

        result = runner.invoke(cli.cli, ["track", "--force", "data/b/tree/value.txt"])

        assert result.exit_code != 0, result.output
        assert "Cannot track 'data/b/tree/value.txt'" in result.output, result.output
        assert not track.get_pvt_path(tree / "value.txt").exists()


def test_targetless_forced_checkout_still_refuses_a_users_tracked_file_beside_a_member(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A user's tracked file and its `.pvt`, copied into the output beside the stage's own
    `fake.pvt`, are no member: discovery still finds the file, and the forced checkout refuses
    the output, naming the sidecar, and changes nothing."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_SIDECAR_SHAPED_MEMBER_PIPELINE)
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        raw = root / "data" / "raw" / "saved.txt"
        raw.parent.mkdir(parents=True)
        raw.write_text("user bytes")
        assert runner.invoke(cli.cli, ["track", "data/raw/saved.txt"]).exit_code == 0
        tree = root / "data" / "b" / "tree"
        shutil.copy2(raw, tree / "saved.txt")
        shutil.copy2(track.get_pvt_path(raw), tree / "saved.txt.pvt")

        status = runner.invoke(cli.cli, ["status", "--tracked-only", "--json"])
        result = runner.invoke(cli.cli, ["checkout", "--force"])

        assert status.exit_code == 0, status.output
        tracked = [info["path"] for info in json.loads(status.stdout)["tracked_files"]]
        assert sorted(tracked) == ["data/b/tree/saved.txt", "data/raw/saved.txt"]
        assert result.exit_code != 0, result.output
        assert "Refusing to replace" in result.output, result.output
        assert "data/b/tree/saved.txt.pvt" in result.output, result.output
        assert (tree / "saved.txt").read_text() == "user bytes"
        assert (tree / "saved.txt.pvt").is_file()
        assert (tree / "value.txt").read_text() == "1"


def test_run_cache_restore_after_the_lock_changes_through_git_keeps_the_stages_pvt(
    runner: CliRunner, git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage writes `notes.pvt` only with `n: 2`. The user checks out the `n: 1` lock from git,
    which lists no `notes.pvt`, keeps `n: 2`, and edits `value.txt`: `repro` restores the run
    cache's `n: 2` output over the directory, and that hash lists the `notes.pvt` the workspace
    holds. The file is the stage's (the hash the restore writes lists it), so the restore
    succeeds and keeps it."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    (repo_path / "pipeline.py").write_text(_PVT_MEMBER_SOMETIMES_PIPELINE)
    assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
    sha = commit("n=1")
    (repo_path / "params.yaml").write_text("b/tree:\n  n: 2\n")
    assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
    tree = repo_path / "data" / "b" / "tree"
    notes = tree / "notes.pvt"
    assert notes.read_text() == "ordinary stage output"
    commit("n=2")
    lock_path = ".pivot/stages/b/tree.lock"
    subprocess.run(["git", "checkout", sha, "--", lock_path], cwd=repo_path, check=True)
    (tree / "value.txt").unlink()
    (tree / "value.txt").write_text("edited")

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, result.output
    assert "Refusing" not in result.output, result.output
    assert notes.read_text() == "ordinary stage output"
    assert (tree / "value.txt").read_text() == "2"


def test_forced_get_restores_the_other_targets_beside_one_whose_pvt_is_not_valid(
    runner: CliRunner, git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pivot get --force` only reports a target whose `.pvt` at the revision is not valid; it
    never replaces that path. A tracked file inside the path therefore refuses nothing: the
    other target is restored, the invalid one is reported, and the tracked file is kept."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    (repo_path / "pipeline.py").write_text(_A_PIPELINE)
    other = repo_path / "other.txt"
    other.write_text("v1")
    tracked = repo_path / "data" / "d" / "f.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("user bytes")
    assert runner.invoke(cli.cli, ["track", "data/d/f.txt"]).exit_code == 0
    invalid = repo_path / "data" / "d.pvt"
    invalid.write_text("path: d\nhash: 1234567890123456\nsize: 1\n")
    sha = commit("v1")
    invalid.unlink()
    other.write_text("local edit")

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "data/d", "other.txt", "--force"])

    assert "Refusing" not in result.output, result.output
    assert "not a valid .pvt file" in result.output, result.output
    assert other.read_text() == "v1", result.output
    assert tracked.read_text() == "user bytes"
    assert track.get_pvt_path(tracked).is_file()
