"""Output ownership protections across independently declared pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

        assert result.exit_code != 0, result.output
        assert "data/b/tree" in result.output
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
            "tracked path 'data/b/tree/saved.txt' (sidecar 'data/b/tree/saved.txt.pvt') "
            + "overlaps stage output 'data/b/tree'. Move both outside the output"
        ) in status_output

        # The saved file can still be updated where it is, with a warning
        tracked.write_text("user input v2")
        retrack_result = runner.invoke(cli.cli, ["track", "--force", "data/b/tree/saved.txt"])
        assert retrack_result.exit_code == 0, retrack_result.output
        assert "overlaps with stage output" in retrack_result.output
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
        assert "sidecar 'data/b/tree/future.txt.pvt' overlaps stage output 'data/b/tree'" in output
        assert sidecar.read_text() == "format: 99\n"
