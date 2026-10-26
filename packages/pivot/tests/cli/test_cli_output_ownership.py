"""Output ownership protections across independently declared pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli, project
from pivot.storage import cache, track

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from click.testing import CliRunner

    from conftest import GitRepo


_ROOT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
"""

_OTHER_DIRECTORY_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {"generated.txt": "generated"}


with Pipeline("b") as pipeline:
    tree()
"""


def _write_other_pipeline(root: pathlib.Path) -> None:
    other = root / "other"
    other.mkdir()
    (other / "pipeline.py").write_text(_OTHER_DIRECTORY_PIPELINE)


def test_track_refuses_file_in_output_of_unloaded_pipeline(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A root pipeline cannot hide another declared pipeline's output from `pivot track`."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_ROOT_PIPELINE)
        _write_other_pipeline(root)
        tracked = root / "data" / "b" / "tree" / "saved.txt"
        tracked.parent.mkdir(parents=True)
        tracked.write_text("user input")

        result = runner.invoke(cli.cli, ["track", "data/b/tree/saved.txt"])

        output = " ".join(result.output.split())
        assert result.exit_code != 0, result.output
        assert (
            "Cannot track 'data/b/tree/saved.txt': it is inside output 'data/b/tree' of stage "
            + "'b/tree', and that stage replaces its output when it runs. Move it outside every "
            + "stage output (for example under data/raw/) and track it there."
        ) in output
        assert not tracked.with_suffix(".txt.pvt").exists()


def test_track_fails_closed_when_another_pipeline_cannot_load(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """An un-loadable pipeline cannot make output-overlap admission incomplete."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_ROOT_PIPELINE)
        other = root / "other"
        other.mkdir()
        (other / "pipeline.py").write_text('raise RuntimeError("broken pipeline")\n')
        candidate = root / "data" / "b" / "tree" / "saved.txt"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("user input")

        result = runner.invoke(cli.cli, ["track", "data/b/tree/saved.txt"])

        assert result.exit_code != 0, result.output
        assert "Failed to load pipeline" in result.output
        assert not candidate.with_suffix(".txt.pvt").exists()


def test_status_and_repro_preserve_saved_input_in_later_declared_output(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A file tracked before another pipeline declared its directory: repro --all warns once,
    fails the stage naming the file and its sidecar, and removes neither; status warns once."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_ROOT_PIPELINE)
        tracked = root / "data" / "b" / "tree" / "saved.txt"
        tracked.parent.mkdir(parents=True)
        tracked.write_text("user input")
        tracked_result = runner.invoke(cli.cli, ["track", "data/b/tree/saved.txt"])
        assert tracked_result.exit_code == 0, tracked_result.output
        pvt_path = tracked.with_suffix(".txt.pvt")
        assert pvt_path.exists()

        _write_other_pipeline(root)

        repro_result = runner.invoke(cli.cli, ["repro", "--all"])

        # The console wraps long lines; compare words, not line breaks
        repro_output = " ".join(repro_result.output.split())
        assert repro_result.exit_code != 0, repro_result.output
        assert repro_output.count("Warning: tracked path") == 1, repro_result.output
        assert "Refusing to replace a stage output" in repro_output, repro_result.output
        assert "Traceback" not in repro_output, "a refusal is a message, not a crash"
        assert "data/b/tree/saved.txt.pvt" in repro_output
        assert tracked.read_text() == "user input"
        assert pvt_path.exists()

        status_result = runner.invoke(cli.cli, ["status", "--all", "--tracked-only"])
        status_output = " ".join(status_result.output.split())
        assert status_result.exit_code == 0, status_result.output
        assert status_output.count("Warning: tracked path") == 1, status_result.output
        assert (
            "Warning: tracked path 'data/b/tree/saved.txt' (sidecar 'data/b/tree/saved.txt.pvt') "
            + "is inside output 'data/b/tree' of stage 'b/tree'. Move the tracked path and its "
            + "sidecar outside every stage output"
        ) in status_output

        # The saved file can still be updated where it is, with a warning
        tracked.write_text("user input v2")
        retrack_result = runner.invoke(cli.cli, ["track", "--force", "data/b/tree/saved.txt"])
        assert retrack_result.exit_code == 0, retrack_result.output
        assert "is tracked inside a stage output: it is inside output 'data/b/tree'" in " ".join(
            retrack_result.output.split()
        )
        assert "Tracked: data/b/tree/saved.txt" in retrack_result.output
        pvt_data = track.read_pvt_file(pvt_path)
        assert pvt_data is not None
        assert pvt_data["hash"] == cache.hash_file(tracked)[0]


def test_repro_keeps_unreadable_sidecar_inside_directory_output(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A `.pvt` Pivot cannot parse (a newer format, a damaged file) is the user's all the same."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_ROOT_PIPELINE)
        _write_other_pipeline(root)
        sidecar = root / "data" / "b" / "tree" / "future.txt.pvt"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text("format: 99\n")

        result = runner.invoke(cli.cli, ["repro", "--all"])

        output = " ".join(result.output.split())
        assert result.exit_code != 0, result.output
        assert (
            "sidecar 'data/b/tree/future.txt.pvt', which this version of Pivot cannot read, is "
            + "inside output 'data/b/tree' of stage 'b/tree'"
        ) in output
        assert sidecar.read_text() == "format: 99\n"


@pytest.mark.parametrize(
    "sidecar_rel",
    [
        pytest.param("data/b/tree.pvt", id="output-level"),
        pytest.param("data/b.pvt", id="parent-level"),
    ],
)
def test_repro_keeps_unreadable_sidecar_beside_output_or_parent(
    runner: CliRunner, tmp_path: pathlib.Path, sidecar_rel: str
) -> None:
    """A sidecar Pivot cannot parse, beside the output or beside a directory that holds it, may
    track that path: the stage refuses to replace the output, and the sidecar is kept."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_ROOT_PIPELINE)
        _write_other_pipeline(root)
        sidecar = root / sidecar_rel
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("format: 99\n")

        result = runner.invoke(cli.cli, ["repro", "--all"])

        output = " ".join(result.output.split())
        assert result.exit_code != 0, result.output
        assert f"sidecar '{sidecar_rel}', which this version of Pivot cannot read" in output
        assert "Refusing to replace a stage output" in output
        assert sidecar.read_text() == "format: 99\n"
        assert not (root / "data" / "b" / "tree" / "generated.txt").exists()


_PVT_MEMBER_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, StageParams, directory, stage


class TreeParams(StageParams):
    n: int = 1


@stage
def tree(params: TreeParams) -> Annotated[dict[str, str], directory]:
    return {"notes.pvt": f"ordinary stage output {params.n}", "value.txt": "v"}


with Pipeline("b") as pipeline:
    tree(params=TreeParams())
"""


@pytest.mark.parametrize("recorded", [False, True], ids=["new-member", "recorded-member"])
def test_stage_member_path_holding_a_user_sidecar_is_refused(
    runner: CliRunner, tmp_path: pathlib.Path, recorded: bool
) -> None:
    """A stage returns `notes.pvt`, but a user's sidecar with other content sits at that path:
    whether or not the lock records a `notes.pvt` of the stage's, the stage refuses to replace
    the output and the user's sidecar is kept."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_PIPELINE)
        tree = root / "data" / "b" / "tree"
        if recorded:
            first = runner.invoke(cli.cli, ["repro"])
            assert first.exit_code == 0, first.output
            (tree / "notes.pvt").unlink()
        tree.mkdir(parents=True, exist_ok=True)
        (tree / "notes").write_text("user data")
        user_sidecar = tree / "notes.pvt"
        track.write_pvt_file(
            user_sidecar,
            track.PvtData(path="notes", hash=cache.hash_file(tree / "notes")[0], size=9),
        )
        before = user_sidecar.read_bytes()

        result = runner.invoke(cli.cli, ["repro", "--force"])

        output = " ".join(result.output.split())
        assert result.exit_code != 0, result.output
        assert "tracked path 'data/b/tree/notes' (sidecar 'data/b/tree/notes.pvt')" in output
        assert user_sidecar.read_bytes() == before
        assert (tree / "notes").read_text() == "user data"


def test_checkout_without_force_restores_other_files_beside_saved_input(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """`checkout` and `checkout --only-missing` never replace an existing output, so a tracked
    file saved inside one makes neither refuse for ownership: each restores every other file and
    reports that output as main does. `checkout --force` refuses only for that output."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_ROOT_PIPELINE)
        _write_other_pipeline(root)
        outside = root / "outside.txt"
        outside.write_text("outside")
        assert runner.invoke(cli.cli, ["track", "outside.txt"]).exit_code == 0
        ran = runner.invoke(cli.cli, ["repro", "--all"])
        assert ran.exit_code == 0, ran.output
        # Saved input: tracked while no pipeline declared data/b/tree (as before this check)
        other = root / "other" / "pipeline.py"
        other.rename(root / "other" / "pipeline.py.off")
        saved = root / "data" / "b" / "tree" / "saved.txt"
        saved.write_text("user input")
        tracked_saved = runner.invoke(cli.cli, ["track", "data/b/tree/saved.txt"])
        assert tracked_saved.exit_code == 0, tracked_saved.output
        (root / "other" / "pipeline.py.off").rename(other)
        sidecar = track.get_pvt_path(saved)
        alive = root / "data" / "a" / "alive.txt"

        for flags, local_message in (
            ([], "already exists with different content"),
            (["--only-missing"], "local changes that are not in the cache"),
        ):
            outside.unlink()
            alive.unlink()
            result = runner.invoke(cli.cli, ["checkout", "--all", *flags])
            output = " ".join(result.output.split())
            assert local_message in output, result.output
            assert "Refusing" not in output, result.output
            assert outside.read_text() == "outside"
            assert alive.read_text() == "alive"
            assert saved.read_text() == "user input"
            assert sidecar.exists()

        outside.unlink()
        forced = runner.invoke(cli.cli, ["checkout", "--all", "--force"])
        forced_output = " ".join(forced.output.split())
        assert forced.exit_code != 0, forced.output
        assert "Refusing to replace a stage output" in forced_output
        assert "data/b/tree/saved.txt.pvt" in forced_output
        assert outside.read_text() == "outside"
        assert saved.read_text() == "user input"
        assert sidecar.exists()


_TREE_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {"g.txt": "g"}


with Pipeline("b") as pipeline:
    tree()
"""

_UNREADABLE = "format: 99\nnot a sidecar this Pivot can read\n"


def test_pivot_commit_does_not_give_an_unreadable_user_sidecar_to_the_stage(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A `.pvt` inside a stage output that this Pivot cannot read may be a user's sidecar (for
    example a newer format). `repro --force` refuses to remove it. `pivot commit` must not record
    it as the stage's own file, so a later forced run still keeps it."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_TREE_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output
        sidecar = root / "data" / "b" / "tree" / "future.pvt"
        sidecar.write_text(_UNREADABLE)
        refused = runner.invoke(cli.cli, ["repro", "--force"])
        assert refused.exit_code != 0, refused.output
        assert sidecar.read_text() == _UNREADABLE

        committed = runner.invoke(cli.cli, ["commit", "b/tree"])
        forced = runner.invoke(cli.cli, ["repro", "--force"])

        assert sidecar.is_file(), committed.output + forced.output
        assert sidecar.read_text() == _UNREADABLE


def test_pivot_commit_after_a_no_commit_run_refuses_a_sidecar_that_run_did_not_write(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """`pivot commit` after `repro --no-commit` claims the `*.pvt` members that run wrote and
    nothing else: an unreadable `.pvt` the user put in the output after the run makes the commit
    refuse, naming it, and a later forced run keeps it."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output
        (root / "params.yaml").write_text("b/tree:\n  n: 2\n")
        pending = runner.invoke(cli.cli, ["repro", "--no-commit"])
        assert pending.exit_code == 0, pending.output
        sidecar = root / "data" / "b" / "tree" / "future.pvt"
        sidecar.write_text(_UNREADABLE)

        committed = runner.invoke(cli.cli, ["commit"])
        forced = runner.invoke(cli.cli, ["repro", "--force"])

        committed_output = " ".join(committed.output.split())
        assert committed.exit_code != 0, committed.output
        assert (
            "sidecar 'data/b/tree/future.pvt', which this version of Pivot cannot read, is inside "
            + "output 'data/b/tree' of stage 'b/tree'"
        ) in committed_output
        assert forced.exit_code != 0, forced.output
        assert sidecar.read_text() == _UNREADABLE
        assert (root / "data" / "b" / "tree" / "notes.pvt").read_text() == "ordinary stage output 2"


@pytest.mark.parametrize("targets", [["data/b/tree"], ["--all"]], ids=["target", "all"])
def test_forced_checkout_keeps_an_unreadable_sidecar_beside_a_stage_output(
    runner: CliRunner, tmp_path: pathlib.Path, targets: list[str]
) -> None:
    """`checkout --force` checks a stage output as the stage's own write does: a sidecar beside
    the output that Pivot cannot read may track that path, so the checkout refuses and keeps it.
    (A tracked path's own restore does not look beside the path.)"""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_TREE_PIPELINE)
        ran = runner.invoke(cli.cli, ["repro"])
        assert ran.exit_code == 0, ran.output
        sidecar = root / "data" / "b" / "tree.pvt"
        sidecar.write_text(_UNREADABLE)

        result = runner.invoke(cli.cli, ["checkout", *targets, "--force"])

        output = " ".join(result.output.split())
        assert result.exit_code != 0, result.output
        assert (
            "sidecar 'data/b/tree.pvt', which this version of Pivot cannot read, may track stage "
            + "output 'data/b/tree' itself"
        ) in output
        assert sidecar.read_text() == _UNREADABLE


def test_only_missing_swap_of_a_tracked_directory_keeps_a_tracked_file_inside_it(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """`checkout --only-missing` replaces a tracked directory whole when the cache holds every
    file that differs or is extra. A tracked file and its sidecar inside it are another user's
    files even when the cache happens to hold the sidecar's bytes: the swap refuses and keeps
    them."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_ROOT_PIPELINE)
        tracked_dir = root / "data" / "raw" / "dir"
        tracked_dir.mkdir(parents=True)
        (tracked_dir / "keep.txt").write_text("keep")
        assert runner.invoke(cli.cli, ["track", "data/raw/dir"]).exit_code == 0
        tracked_dir.chmod(0o755)
        saved = tracked_dir / "n.txt"
        saved.write_text("saved")
        assert runner.invoke(cli.cli, ["track", "data/raw/dir/n.txt"]).exit_code == 0
        sidecar = track.get_pvt_path(saved)
        # Another tracked file with the sidecar's bytes puts that content in the cache
        twin = root / "data" / "raw" / "twin.txt"
        twin.write_bytes(sidecar.read_bytes())
        assert runner.invoke(cli.cli, ["track", "data/raw/twin.txt"]).exit_code == 0

        result = runner.invoke(cli.cli, ["checkout", "data/raw/dir", "--only-missing"])

        output = " ".join(result.output.split())
        assert result.exit_code != 0, result.output
        assert (
            "tracked path 'data/raw/dir/n.txt' (sidecar 'data/raw/dir/n.txt.pvt') is inside "
            + "restored path 'data/raw/dir'"
        ) in output
        assert saved.read_text() == "saved"
        assert sidecar.is_file()


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


def test_user_sidecar_at_a_path_a_no_commit_run_wrote_stays_the_users(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A `--no-commit` run's record of the `*.pvt` members it wrote outlives a later run that
    restores the lock's output. A `.pvt` the user then writes at such a path holds other content,
    so it is not the stage's: a forced run refuses and keeps it."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_SOMETIMES_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output
        params = root / "params.yaml"
        params.write_text("b/tree:\n  n: 2\n")
        pending = runner.invoke(cli.cli, ["repro", "--no-commit"])
        assert pending.exit_code == 0, pending.output
        params.unlink()
        restored = runner.invoke(cli.cli, ["repro"])
        assert restored.exit_code == 0, restored.output
        notes = root / "data" / "b" / "tree" / "notes.pvt"
        assert not notes.exists()
        notes.write_text(_UNREADABLE)

        forced = runner.invoke(cli.cli, ["repro", "--force"])

        assert forced.exit_code != 0, forced.output
        assert notes.read_text() == _UNREADABLE


def _make_own_restore_state(
    runner: CliRunner, repo_path: pathlib.Path, commit: Callable[[str], str], shape: str
) -> tuple[str, str, pathlib.Path]:
    """Build one of the recheck's states with this build's own `track`, before pipeline `b`
    declares `data/b/tree`. Return the restored tracked path, the revision that records it and
    the tracked file saved beneath it."""
    (repo_path / "pipeline.py").write_text(_ROOT_PIPELINE)
    (repo_path / "outside.txt").write_text("outside")
    target = {"file": "data/b/tree", "directory": "data/b", "nested": "data/raw/dir"}[shape]
    target_path = repo_path / target
    if shape == "file":
        target_path.parent.mkdir(parents=True)
        target_path.write_text("tracked file")
    else:
        target_path.mkdir(parents=True)
        (target_path / "keep.txt").write_text("keep")
    tracked = runner.invoke(cli.cli, ["track", target, "outside.txt"])
    assert tracked.exit_code == 0, tracked.output
    sha = commit("v1")
    if shape == "file":
        # Pipeline b ran and made data/b/tree its directory output; the file's sidecar stayed
        target_path.unlink()
        target_path.mkdir()
        (target_path / "g.txt").write_text("g")
        saved = target_path / "f.txt"
    elif shape == "directory":
        # Pipeline b ran and wrote its directory output data/b/tree inside data/b
        target_path.chmod(0o755)
        (target_path / "tree").mkdir()
        (target_path / "tree" / "g.txt").write_text("g")
        saved = target_path / "tree" / "f.txt"
    else:
        target_path.chmod(0o755)
        saved = target_path / "n.txt"
    saved.write_text("saved")
    saved_track = runner.invoke(cli.cli, ["track", str(saved.relative_to(repo_path))])
    assert saved_track.exit_code == 0, saved_track.output
    if shape != "nested":
        (repo_path / "other").mkdir()
        (repo_path / "other" / "pipeline.py").write_text(_TREE_PIPELINE)
    return target, sha, saved


@pytest.mark.parametrize("command", ["checkout", "checkout-all", "get"])
@pytest.mark.parametrize("shape", ["file", "directory", "nested"])
def test_forced_restore_of_a_tracked_path_keeps_other_tracked_files_beneath_it(
    runner: CliRunner,
    git_repo: GitRepo,
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
    command: str,
) -> None:
    """A forced restore of a tracked path may replace that path's own content and nothing else.
    `checkout --force` (which `pull --force` runs), `checkout --all --force` and `get --force`
    refuse, before removing anything, to restore it over a tracked file and sidecar saved beneath
    it, in each state: a tracked file whose path became a directory output (file), a tracked
    directory holding a directory output (directory), and a tracked directory holding another
    tracked file (nested). `checkout --all --force` still restores each target it does not
    refuse."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    target, sha, saved = _make_own_restore_state(runner, repo_path, commit, shape)
    outside = repo_path / "outside.txt"
    outside.unlink()

    if command == "checkout":
        result = runner.invoke(cli.cli, ["checkout", target, "--force"])
    elif command == "checkout-all":
        result = runner.invoke(cli.cli, ["checkout", "--all", "--force"])
    else:
        result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], target, "--force"])

    output = " ".join(result.output.split())
    assert result.exit_code != 0, result.output
    assert (
        f"tracked path '{saved.relative_to(repo_path)}' "
        + f"(sidecar '{saved.relative_to(repo_path)}.pvt') is inside restored path '{target}'"
    ) in output
    assert saved.read_text() == "saved"
    assert track.get_pvt_path(saved).is_file()
    if command == "checkout-all":
        assert outside.read_text() == "outside"


@pytest.mark.parametrize("command", ["checkout", "get"])
def test_restoring_a_tracked_directory_keeps_a_tracked_file_saved_inside_it(
    runner: CliRunner,
    git_repo: GitRepo,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """`data/b` is a tracked directory, and pipeline `b` declares the output `data/b/tree` inside
    it. `data/b/tree/f.txt` was tracked before that pipeline existed. A forced restore of `data/b`
    is data/b's own restore, but it is not f.txt's: Pivot must not delete f.txt or its sidecar
    (round-2 rule D1: the guard protects a tracked file from every write that is not that tracked
    file's own restore)."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    (repo_path / "pipeline.py").write_text(_ROOT_PIPELINE)
    keep = repo_path / "data" / "b" / "keep.txt"
    keep.parent.mkdir(parents=True)
    keep.write_text("keep")
    tracked = runner.invoke(cli.cli, ["track", "data/b"])
    assert tracked.exit_code == 0, tracked.output
    sha = commit("v1")
    keep.parent.chmod(0o755)
    saved = repo_path / "data" / "b" / "tree" / "f.txt"
    saved.parent.mkdir()
    saved.write_text("saved")
    saved_track = runner.invoke(cli.cli, ["track", "data/b/tree/f.txt"])
    assert saved_track.exit_code == 0, saved_track.output
    (repo_path / "other").mkdir()
    (repo_path / "other" / "pipeline.py").write_text(_TREE_PIPELINE)

    if command == "checkout":
        result = runner.invoke(cli.cli, ["checkout", "data/b", "--force"])
    else:
        result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "data/b", "--force"])

    assert saved.is_file(), result.output
    assert saved.read_text() == "saved", result.output
    assert (saved.parent / "f.txt.pvt").is_file(), result.output
