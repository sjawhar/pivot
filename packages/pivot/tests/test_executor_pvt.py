"""Execution regressions for tracked directory inputs."""

from __future__ import annotations

import pathlib

import click.testing

from pivot import cli, compose, executor
from pivot.storage import store as store_mod


@compose.stage
def _count_images(images: pathlib.Path) -> str:
    return str(len(list(images.iterdir())))


def _register_count_images(pipeline: compose.Pipeline, *, input_path: str = "images") -> pathlib.Path:
    with compose.Pipeline(pipeline.name, root=pipeline.root) as builder:
        images = builder.input("images", t=pathlib.Path, path=input_path)
        _count_images(images)
    pipeline.include(builder)
    stage = pipeline.get_stage(f"{pipeline.name}/_count_images")
    workspace = store_mod.WorkspaceStore(
        project_root=pipeline.root,
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
    )
    return workspace.resolve_display_path(stage["outs"][0])


def test_tracked_directory_change_triggers_rerun(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Re-tracking an added directory member re-executes its dependent stage."""
    images = mock_discovery.root / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"image1")
    (images / "b.jpg").write_bytes(b"image2")
    assert runner.invoke(cli.cli, ["track", "images"]).exit_code == 0
    output_path = _register_count_images(mock_discovery)

    assert executor.run(pipeline=mock_discovery)["test/_count_images"]["status"] == "ran"
    assert output_path.read_text() == "2"

    (images / "c.jpg").write_bytes(b"image3")
    assert runner.invoke(cli.cli, ["track", "--force", "images"]).exit_code == 0

    assert executor.run(pipeline=mock_discovery)["test/_count_images"]["status"] == "ran"
    assert output_path.read_text() == "3"


def test_directory_dependency_hashed_and_runs(mock_discovery: compose.Pipeline) -> None:
    """A plain directory input is content-hashed and supplied as a Path."""
    directory = mock_discovery.root / "directory"
    directory.mkdir()
    (directory / "file.txt").write_text("content")
    output_path = _register_count_images(mock_discovery, input_path="directory")

    results = executor.run(pipeline=mock_discovery)

    assert results["test/_count_images"]["status"] == "ran"
    assert output_path.read_text() == "1"
