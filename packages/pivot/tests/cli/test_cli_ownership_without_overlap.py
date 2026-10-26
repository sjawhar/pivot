"""Projects where nothing overlaps a stage output keep working as before the ownership guard.

No case here has a tracked file, an import or a `.pvt` sidecar inside a stage output. The guard
against removing a user's files must refuse nothing in such a project.
"""

from __future__ import annotations

import pathlib
import re
from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli, project
from pivot.storage import state, store

if TYPE_CHECKING:
    import click.testing

    from conftest import GitRepo


_ROOT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
"""

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

# The stage writes notes.pvt only when n is 2, so only the run with n == 2 records it.
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


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_get_force_restores_an_earlier_version_of_a_tracked_input(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """`pivot get --rev R <tracked path> --force` restores R's version of a tracked file or
    directory that no stage declares: restoring a tracked path to its own place is its own
    restore, not the replacement of a stage output, and no message may call it one."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    target = pathlib.Path("data/raw/in.txt" if kind == "file" else "data/raw/dir")
    member = repo_path / (target if kind == "file" else target / "x.txt")
    member.parent.mkdir(parents=True)
    member.write_text("v1")
    tracked = runner.invoke(cli.cli, ["track", str(target)])
    assert tracked.exit_code == 0, tracked.output
    sha = commit("v1")
    member.parent.chmod(0o755)
    member.unlink()
    member.write_text("v2")
    retracked = runner.invoke(cli.cli, ["track", "--force", str(target)])
    assert retracked.exit_code == 0, retracked.output

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "--force", str(target)])

    assert result.exit_code == 0, result.output
    assert "stage output" not in result.output
    assert member.read_text() == "v1"


def test_get_relative_output_is_relative_to_the_working_directory(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative `pivot get -o NAME` names a path relative to the working directory: run in a
    subdirectory it writes NAME there, never replaces a same-named user file at the project
    root, and `-o ../NAME` lands one directory up, inside the project."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache").mkdir(parents=True)
    (repo_path / "file.txt").write_text("from revision")
    (repo_path / "notes.txt").write_text("user notes at the project root")
    sha = commit("initial")
    sub = repo_path / "sub"
    sub.mkdir()
    escaped = repo_path.parent / "pr521-escaped-copy.txt"
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(sub)

    forced = runner.invoke(
        cli.cli, ["get", "--rev", sha[:7], "file.txt", "-o", "notes.txt", "--force"]
    )
    up = runner.invoke(
        cli.cli, ["get", "--rev", sha[:7], "file.txt", "-o", "../pr521-escaped-copy.txt"]
    )

    assert forced.exit_code == 0, forced.output
    assert up.exit_code == 0, up.output
    assert (repo_path / "notes.txt").read_text() == "user notes at the project root"
    assert (sub / "notes.txt").read_text() == "from revision"
    assert (repo_path / "pr521-escaped-copy.txt").read_text() == "from revision"
    assert not escaped.exists(), "a relative -o must not write outside the project"


@pytest.mark.parametrize(
    "module_source",
    [
        pytest.param("def make_pipeline(steps):\n    return list(steps)\n", id="no-pipeline-var"),
        pytest.param('pipeline = ["clean", "train"]\n', id="non-pivot-pipeline-var"),
    ],
)
def test_track_accepts_input_when_a_pipeline_py_is_not_a_pivot_pipeline(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, module_source: str
) -> None:
    """A module named pipeline.py that loads but defines no Pivot pipeline declares no output,
    so it does not stop `pivot track` of an input outside every output."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_ROOT_PIPELINE)
        helper = root / "src" / "mlproj" / "pipeline.py"
        helper.parent.mkdir(parents=True)
        helper.write_text(module_source)
        data = root / "data" / "raw" / "in.txt"
        data.parent.mkdir(parents=True)
        data.write_text("raw input")

        result = runner.invoke(cli.cli, ["track", "data/raw/in.txt"])

        assert result.exit_code == 0, result.output
        assert (root / "data" / "raw" / "in.txt.pvt").exists()


def test_pvt_named_member_survives_checkout_and_run_cache_restore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A `*.pvt` directory member the stage itself writes is the stage's: `checkout`,
    `checkout --force` and a run-cache restore replace it, and no warning tells the user to
    move it."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output

        checkout = runner.invoke(cli.cli, ["checkout"])
        forced = runner.invoke(cli.cli, ["checkout", "--force"])
        (root / "params.yaml").write_text("b/tree:\n  n: 2\n")
        changed = runner.invoke(cli.cli, ["repro"])
        (root / "params.yaml").unlink()
        restored = runner.invoke(cli.cli, ["repro"])

        assert checkout.exit_code == 0, checkout.output
        assert forced.exit_code == 0, forced.output
        assert changed.exit_code == 0, changed.output
        assert restored.exit_code == 0, restored.output
        outputs = " ".join((checkout.output + changed.output + restored.output).split())
        assert "outside every stage output" not in outputs
        tree = root / "data" / "b" / "tree"
        assert (tree / "notes.pvt").read_text() == "ordinary stage output 1"


def test_pvt_named_member_commits_after_a_no_commit_run(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """`pivot commit` after `repro --no-commit` records the stage's own `*.pvt` member as its
    output."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_PIPELINE)

        ran = runner.invoke(cli.cli, ["repro", "--no-commit"])
        committed = runner.invoke(cli.cli, ["commit"])

        assert ran.exit_code == 0, ran.output
        assert committed.exit_code == 0, committed.output
        assert "Refusing" not in committed.output


def test_get_force_restores_a_stage_output_holding_its_own_pvt_member(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pivot get --rev R <stage> --force` restores a stage output whose `*.pvt` member the
    stage wrote, as at main (kills mutant R04: get --force passes no members)."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    (repo_path / "pipeline.py").write_text(_PVT_MEMBER_PIPELINE)
    first = runner.invoke(cli.cli, ["repro"])
    assert first.exit_code == 0, first.output
    sha = commit("v1")
    (repo_path / "params.yaml").write_text("b/tree:\n  n: 2\n")
    second = runner.invoke(cli.cli, ["repro"])
    assert second.exit_code == 0, second.output

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "b/tree", "--force"])

    assert result.exit_code == 0, result.output
    notes = repo_path / "data" / "b" / "tree" / "notes.pvt"
    assert notes.read_text() == "ordinary stage output 1"


def test_stage_own_pvt_member_draws_no_saved_input_warning(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """`status` and `repro` warn about a user's sidecar inside an output, never about the
    stage's own `*.pvt` member (kills mutant R05: the warnings pass no members). The tester's
    round-1 check looks for "overlaps stage output", which the round-2 warning no longer says;
    this one looks for the remedy every round-2 ownership warning gives. ("Skipping invalid .pvt
    file" about the member is older than this PR and is not checked here.)"""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output

        status = runner.invoke(cli.cli, ["status"])
        again = runner.invoke(cli.cli, ["repro", "--force"])

        assert status.exit_code == 0, status.output
        assert again.exit_code == 0, again.output
        assert "outside every stage output" not in status.output + again.output


def test_run_cache_restore_replaces_a_pvt_member_only_the_current_lock_records(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A run-cache restore of an older output removes the stage's own `*.pvt` member that only
    the current lock records, as at main (kills mutant R06: the run-cache restore drops the
    lock's hashes)."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_SOMETIMES_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output
        (root / "params.yaml").write_text("b/tree:\n  n: 2\n")
        second = runner.invoke(cli.cli, ["repro"])
        assert second.exit_code == 0, second.output
        assert (root / "data" / "b" / "tree" / "notes.pvt").is_file()
        (root / "params.yaml").unlink()

        restored = runner.invoke(cli.cli, ["repro"])

        assert restored.exit_code == 0, restored.output
        assert not (root / "data" / "b" / "tree" / "notes.pvt").exists()
        assert (root / "data" / "b" / "tree" / "value.txt").read_text() == "1"


def test_run_after_a_no_commit_run_replaces_the_stage_pvt_member(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """After `repro --no-commit` writes new content to the stage's own `*.pvt` member, the next
    run replaces it, as at main (kills mutant R15: an unparsed `.pvt` at a recorded path is the
    user's unless its content was recorded)."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output
        (root / "params.yaml").write_text("b/tree:\n  n: 2\n")
        pending = runner.invoke(cli.cli, ["repro", "--no-commit"])
        assert pending.exit_code == 0, pending.output

        again = runner.invoke(cli.cli, ["repro"])

        assert again.exit_code == 0, again.output
        notes = root / "data" / "b" / "tree" / "notes.pvt"
        assert notes.read_text() == "ordinary stage output 2"


def test_pvt_member_only_a_no_commit_run_wrote_is_the_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A `*.pvt` member that only a `repro --no-commit` run wrote (no lock records it) is the
    stage's, as at main: `status` and a second `--no-commit` run draw no warning to move it,
    `checkout --force` and a run that restores the lock's output replace it, and `pivot commit`
    after another `--no-commit` run records it."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_SOMETIMES_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output
        params = root / "params.yaml"
        params.write_text("b/tree:\n  n: 2\n")
        pending = runner.invoke(cli.cli, ["repro", "--no-commit"])
        assert pending.exit_code == 0, pending.output
        notes = root / "data" / "b" / "tree" / "notes.pvt"
        value = root / "data" / "b" / "tree" / "value.txt"
        assert notes.is_file()

        status = runner.invoke(cli.cli, ["status"])
        pending_again = runner.invoke(cli.cli, ["repro", "--no-commit", "--force"])
        assert status.exit_code == 0, status.output
        assert pending_again.exit_code == 0, pending_again.output
        assert "outside every stage output" not in status.output + pending_again.output

        checkout = runner.invoke(cli.cli, ["checkout", "--force"])
        assert checkout.exit_code == 0, checkout.output
        assert not notes.exists()
        assert value.read_text() == "1"

        again = runner.invoke(cli.cli, ["repro", "--no-commit"])
        assert again.exit_code == 0, again.output
        params.unlink()
        restored = runner.invoke(cli.cli, ["repro"])
        assert restored.exit_code == 0, restored.output
        assert not notes.exists()
        assert value.read_text() == "1"

        params.write_text("b/tree:\n  n: 2\n")
        rerun = runner.invoke(cli.cli, ["repro", "--no-commit"])
        committed = runner.invoke(cli.cli, ["commit"])
        assert rerun.exit_code == 0, rerun.output
        assert committed.exit_code == 0, committed.output
        assert "Refusing" not in committed.output
        assert notes.read_text() == "ordinary stage output"


def test_run_after_a_no_commit_run_replaces_a_pvt_member_it_no_longer_returns(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A run that returns no `notes.pvt`, after a `--no-commit` run wrote one that no lock
    records, replaces the output and the stage's old member with it, as at main."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_SOMETIMES_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output
        params = root / "params.yaml"
        params.write_text("b/tree:\n  n: 2\n")
        pending = runner.invoke(cli.cli, ["repro", "--no-commit"])
        assert pending.exit_code == 0, pending.output
        params.write_text("b/tree:\n  n: 3\n")

        ran = runner.invoke(cli.cli, ["repro"])

        assert ran.exit_code == 0, ran.output
        assert not (root / "data" / "b" / "tree" / "notes.pvt").exists()
        assert (root / "data" / "b" / "tree" / "value.txt").read_text() == "3"


def test_get_force_after_a_no_commit_run_replaces_its_pvt_member(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pivot get --rev R <stage> --force` after a `--no-commit` run restores R's output over the
    `*.pvt` member that run wrote, as at main."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    monkeypatch.chdir(repo_path)
    (repo_path / "pipeline.py").write_text(_PVT_MEMBER_SOMETIMES_PIPELINE)
    first = runner.invoke(cli.cli, ["repro"])
    assert first.exit_code == 0, first.output
    sha = commit("v1")
    (repo_path / "params.yaml").write_text("b/tree:\n  n: 2\n")
    pending = runner.invoke(cli.cli, ["repro", "--no-commit"])
    assert pending.exit_code == 0, pending.output

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "b/tree", "--force"])

    assert result.exit_code == 0, result.output
    assert not (repo_path / "data" / "b" / "tree" / "notes.pvt").exists()
    assert (repo_path / "data" / "b" / "tree" / "value.txt").read_text() == "1"


def test_run_replaces_a_stage_pvt_member_edited_to_unreadable_content(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """The stage's own `*.pvt` member at the path its lock records, edited by hand to content
    Pivot cannot parse and nothing records, is still the stage's: the next forced run replaces
    it, as at main (kills mutant R15: an unparsed `.pvt` at a recorded path is the user's unless
    its content was recorded)."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output
        notes = root / "data" / "b" / "tree" / "notes.pvt"
        notes.unlink()
        notes.write_text("edited by hand\n")

        forced = runner.invoke(cli.cli, ["repro", "--force"])

        assert forced.exit_code == 0, forced.output
        assert notes.read_text() == "ordinary stage output 1"


def test_pivot_commit_keeps_the_pvt_member_the_lock_records(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """`pivot commit` of a stage whose lock records its own `*.pvt` member, with no
    `--no-commit` record, commits it as main does (kills mutant N08: commit passes only the
    record's members)."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output

        committed = runner.invoke(cli.cli, ["commit", "b/tree"])

        assert committed.exit_code == 0, committed.output
        assert "Refusing" not in committed.output


def test_a_written_lock_ends_the_no_commit_record(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """`repro --no-commit` keeps a record of the `*.pvt` members it wrote; `pivot commit` and a
    run that writes the lock remove it."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PVT_MEMBER_PIPELINE)

        def recorded_members() -> list[str] | None:
            with state.StateDB(root / ".pivot", readonly=True) as db:
                record = db.get_no_commit_record("b/tree")
            if record is None:
                return None
            return sorted(record["pvt_members"]) if "pvt_members" in record else []

        assert runner.invoke(cli.cli, ["repro", "--no-commit"]).exit_code == 0
        after_first = recorded_members()
        assert runner.invoke(cli.cli, ["commit"]).exit_code == 0
        after_commit = recorded_members()
        (root / "params.yaml").write_text("b/tree:\n  n: 2\n")
        assert runner.invoke(cli.cli, ["repro", "--no-commit"]).exit_code == 0
        after_second = recorded_members()
        assert runner.invoke(cli.cli, ["repro"]).exit_code == 0
        after_run = recorded_members()

        assert after_first == ["data/b/tree/notes.pvt"]
        assert after_commit is None
        assert after_second == ["data/b/tree/notes.pvt"]
        assert after_run is None


def test_storage_agents_md_names_only_existing_workspace_store_methods() -> None:
    """storage/AGENTS.md tells the next author which methods guard an output's replacement;
    each `WorkspaceStore.<name>` it cites exists."""
    text = (pathlib.Path(store.__file__).parent / "AGENTS.md").read_text()
    cited = set(re.findall(r"WorkspaceStore\.(\w+)", text))
    missing = sorted(name for name in cited if not hasattr(store.WorkspaceStore, name))
    assert not missing, f"storage/AGENTS.md cites missing WorkspaceStore methods: {missing}"
