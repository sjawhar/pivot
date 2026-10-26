"""Output ownership protections across independently declared pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli
from pivot.storage import cache, track

if TYPE_CHECKING:
    import pathlib

    from click.testing import CliRunner


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
