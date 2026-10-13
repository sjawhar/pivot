"""Execution with tracked (.pvt) file and directory inputs."""

from __future__ import annotations

import pathlib  # noqa: TC003 - compose resolves postponed stage annotations at runtime
from typing import TYPE_CHECKING

import pytest

from pivot import cli, exceptions, executor, project
from pivot.storage import track
from tests import helpers

if TYPE_CHECKING:
    from collections.abc import Callable

    import click.testing

    from pivot.registry import PipelineLike as Pipeline


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set a real project root for CLI track/checkout plus direct executor calls."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "get_project_root", lambda: tmp_path)
    return tmp_path


def _process_data(data: pathlib.Path) -> str:
    return f"processed: {len(data.read_text())} bytes"


def _process_simple(data: pathlib.Path) -> str:
    _ = data
    return "done"


def _process_data_content(data: pathlib.Path) -> str:
    return f"processed: {data.read_text()}"


def _count_images(images: pathlib.Path) -> str:
    return str(len(list(images.iterdir())))


def _process_images(images: pathlib.Path) -> str:
    _ = images
    return "done"


def _process_mixed(data: pathlib.Path, config: pathlib.Path) -> str:
    return f"{data.read_text()}|{config.read_text()}"


def _process_uppercase(data: pathlib.Path) -> str:
    return data.read_text().upper()


def _output(stage: str) -> pathlib.Path:
    return helpers.get_output_path(stage)


def _register_data_stage(func: Callable[..., object], *, name: str = "process") -> None:
    helpers.register_test_stage(func, name=name, deps={"data": "data.csv"})


def test_run_succeeds_with_existing_tracked_file(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path
) -> None:
    """Pipeline runs successfully when a tracked input exists."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("col1,col2\n1,2\n")
    track.write_pvt_file(
        pipeline_dir / "data.csv.pvt", {"path": "data.csv", "hash": "placeholder", "size": 100}
    )

    _register_data_stage(_process_data)
    results = executor.run(pipeline=test_pipeline)

    assert results["test/process"]["status"] == "ran"
    assert _output("process").read_text() == "processed: 14 bytes"


def test_run_fails_when_tracked_file_missing(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path
) -> None:
    """Pipeline fails with a helpful error if a tracked input is absent."""
    track.write_pvt_file(
        pipeline_dir / "data.csv.pvt", {"path": "data.csv", "hash": "abc123", "size": 100}
    )
    _register_data_stage(_process_simple)

    with pytest.raises(exceptions.TrackedFileMissingError, match=r"checkout|restore|missing"):
        executor.run(pipeline=test_pipeline)


def test_run_succeeds_with_hash_mismatch(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """A tracked input hash mismatch warns but still permits execution."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("original content")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    data_file.write_text("modified content that changes the hash")

    _register_data_stage(_process_simple)
    results = executor.run(pipeline=test_pipeline)

    assert results["test/process"]["status"] == "ran"
    assert _output("process").read_text() == "done"


def test_tracked_file_change_triggers_downstream_rerun(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """Re-tracking a changed file re-executes its dependent stage."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("original")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    _register_data_stage(_process_data_content)

    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"
    assert _output("process").read_text() == "processed: original"

    data_file.write_text("modified")
    assert runner.invoke(cli.cli, ["track", "--force", "data.csv"]).exit_code == 0

    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"
    assert _output("process").read_text() == "processed: modified"


def test_unchanged_tracked_file_allows_skip(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """An unchanged tracked input permits a cache skip."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("content")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    _register_data_stage(_process_data_content)

    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"
    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "cached"


def test_tracked_directory_change_triggers_rerun(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """Re-tracking an added directory file re-executes the dependent stage."""
    data_dir = pipeline_dir / "images"
    data_dir.mkdir()
    (data_dir / "a.jpg").write_bytes(b"image1")
    (data_dir / "b.jpg").write_bytes(b"image2")
    assert runner.invoke(cli.cli, ["track", "images"]).exit_code == 0
    helpers.register_test_stage(_count_images, name="count_images", deps={"images": "images"})

    assert executor.run(pipeline=test_pipeline)["test/count_images"]["status"] == "ran"
    assert _output("count_images").read_text() == "2"

    (data_dir / "c.jpg").write_bytes(b"image3")
    assert runner.invoke(cli.cli, ["track", "--force", "images"]).exit_code == 0

    assert executor.run(pipeline=test_pipeline)["test/count_images"]["status"] == "ran"
    assert _output("count_images").read_text() == "3"


def test_run_fails_when_tracked_directory_missing(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path
) -> None:
    """A missing tracked directory raises the tracked-file restore error."""
    track.write_pvt_file(
        pipeline_dir / "images.pvt",
        {
            "path": "images",
            "hash": "abc123",
            "size": 1000,
            "num_files": 2,
            "manifest": [
                {"relpath": "a.jpg", "hash": "h1", "size": 500, "isexec": False},
                {"relpath": "b.jpg", "hash": "h2", "size": 500, "isexec": False},
            ],
        },
    )
    helpers.register_test_stage(_process_images, name="process", deps={"images": "images"})

    with pytest.raises(exceptions.TrackedFileMissingError, match=r"checkout|restore|missing"):
        executor.run(pipeline=test_pipeline)


def test_mixed_tracked_and_regular_dependencies(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """A stage can depend on both a tracked and regular file."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("tracked data")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    config_file = pipeline_dir / "config.txt"
    config_file.write_text("setting=1")

    helpers.register_test_stage(
        _process_mixed,
        name="process",
        deps={"data": "data.csv", "config": "config.txt"},
    )
    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"

    config_file.write_text("setting=2")
    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"


def test_checkout_then_run_succeeds(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """checkout restores a tracked input that a compositional stage then consumes."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("important data")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    data_file.unlink()
    assert runner.invoke(cli.cli, ["checkout", "data.csv"]).exit_code == 0
    assert data_file.exists()

    _register_data_stage(_process_uppercase)
    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"
    assert _output("process").read_text() == "IMPORTANT DATA"


def test_checkout_missing_restores_tracked_input_before_running(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """run(checkout_missing=True) restores a deleted tracked input from cache, then uses it."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("cached data")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    data_file.unlink()
    _register_data_stage(_process_uppercase)

    results = executor.run(pipeline=test_pipeline, checkout_missing=True)

    assert results["test/process"]["status"] == "ran"
    assert data_file.read_text() == "cached data", "the tracked input is back in the workspace"
    assert _output("process").read_text() == "CACHED DATA"


def test_checkout_missing_reports_tracked_input_absent_from_cache(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path
) -> None:
    """A missing tracked input that is not cached fails, suggesting a pull, and nothing runs."""
    track.write_pvt_file(
        pipeline_dir / "data.csv.pvt",
        {"path": "data.csv", "hash": "0123456789abcdef", "size": 10},
    )
    _register_data_stage(_process_simple)

    with pytest.raises(exceptions.TrackedFileMissingError) as exc_info:
        executor.run(pipeline=test_pipeline, checkout_missing=True)

    assert "not in local cache" in str(exc_info.value)
    assert exc_info.value.get_suggestion() == "Run 'pivot pull' to fetch from remote storage"
    assert not _output("process").exists()
