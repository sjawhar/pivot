"""checkout/pull must never overwrite local edits inside an existing directory without --force."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import cli, config, project
from pivot.storage import cache, track

if TYPE_CHECKING:
    import pathlib

    import click.testing

    from pivot.registry import PipelineLike


def _helper_track_directory(root: pathlib.Path) -> pathlib.Path:
    """Create data/ext with two files, cache them, and write data/ext.pvt (as `pivot track`)."""
    data_dir = root / "data" / "ext"
    data_dir.mkdir(parents=True)
    (data_dir / "a.txt").write_text("one\n")
    (data_dir / "b.txt").write_text("two\n")
    output_hash = cache.save_to_cache(
        data_dir, config.get_cache_dir() / "files", checkout_mode=config.CheckoutMode.COPY
    )
    assert "manifest" in output_hash
    track.write_pvt_file(
        track.get_pvt_path(data_dir),
        track.PvtData(
            path="ext",
            hash=output_hash["hash"],
            size=8,
            num_files=2,
            manifest=output_hash["manifest"],
        ),
    )
    return data_dir


def test_checkout_only_missing_keeps_edited_file_in_directory(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    data_dir = _helper_track_directory(project.get_project_root())
    (data_dir / "a.txt").write_text("one EDITED\n")
    (data_dir / "b.txt").unlink()

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code == 0, result.output
    assert (data_dir / "a.txt").read_text() == "one EDITED\n", "Edited file must not be replaced"
    assert (data_dir / "b.txt").read_text() == "two\n", "Missing file should be restored"


def test_checkout_only_missing_keeps_untracked_file_in_directory(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    data_dir = _helper_track_directory(project.get_project_root())
    (data_dir / "notes.txt").write_text("mine\n")

    result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data/ext"])

    assert result.exit_code == 0, result.output
    assert (data_dir / "notes.txt").read_text() == "mine\n", "Extra file must survive"


def test_checkout_default_refuses_edited_directory_and_force_overwrites(
    mock_discovery: PipelineLike, runner: click.testing.CliRunner
) -> None:
    data_dir = _helper_track_directory(project.get_project_root())
    (data_dir / "a.txt").write_text("one EDITED\n")

    refused = runner.invoke(cli.cli, ["checkout", "data/ext"])

    assert refused.exit_code != 0
    assert "--force" in refused.output, "Error should say how to override"
    assert (data_dir / "a.txt").read_text() == "one EDITED\n"

    forced = runner.invoke(cli.cli, ["checkout", "--force", "data/ext"])

    assert forced.exit_code == 0, forced.output
    assert (data_dir / "a.txt").read_text() == "one\n"
