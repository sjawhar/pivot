from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli
from pivot.storage import track

if TYPE_CHECKING:
    import pathlib

    import click.testing


@pytest.mark.parametrize("document", ["path: payload\n", "path: [unclosed\n", "{}\n"])
def test_track_force_repairs_its_own_unreadable_sidecar(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, document: str
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        payload = root / "payload"
        payload.mkdir()
        (payload / "a.txt").write_text("retained data")
        sidecar = root / "payload.pvt"
        sidecar.write_text(document)

        repaired = runner.invoke(cli.cli, ["track", "--force", "payload"])

        assert repaired.exit_code == 0, repaired.output
        metadata = track.read_pvt_file(sidecar)
        assert metadata is not None
        assert "nfiles" in metadata
        assert metadata["nfiles"] == 1
        assert (payload / "a.txt").read_text() == "retained data"
        verified = runner.invoke(cli.cli, ["checkout", "--all", "--force"])
        assert verified.exit_code == 0, verified.output
        assert (payload / "a.txt").read_text() == "retained data"


@pytest.mark.parametrize(
    "document",
    [
        "format: 99\n",
        "path: future\nhash: abcdef\nsize: 0\nfuture_field: true\n",
    ],
)
@pytest.mark.parametrize("command", [["checkout", "--all", "--force"], ["verify"]])
def test_foreign_sidecar_remains_inert_for_unrelated_restoration_and_verification(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    document: str,
    command: list[str],
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(
            "from pivot import Pipeline, stage\n\n@stage\ndef alive() -> str:\n"
            + "    return 'alive'\n\nwith Pipeline('test') as pipeline:\n    alive()\n"
        )
        reproduced = runner.invoke(cli.cli, ["repro"])
        assert reproduced.exit_code == 0, reproduced.output
        sidecar = root / "future.pvt"
        sidecar.write_text(document)

        result = runner.invoke(cli.cli, command)

        assert result.exit_code == 0, result.output
        assert sidecar.read_text() == document
        assert not (root / "future").exists()


@pytest.mark.parametrize("command", [["checkout", "--all", "--force"], ["verify"]])
def test_tracked_pvt_file_is_payload_not_unreadable_metadata(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, command: list[str]
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(
            "from pivot import Pipeline, stage\n\n@stage\ndef alive() -> str:\n"
            + "    return 'alive'\n\nwith Pipeline('test') as pipeline:\n    alive()\n"
        )
        reproduced = runner.invoke(cli.cli, ["repro"])
        assert reproduced.exit_code == 0, reproduced.output
        payload = root / "notes.pvt"
        payload.write_text("ordinary text")
        tracked = runner.invoke(cli.cli, ["track", "notes.pvt"])
        assert tracked.exit_code == 0, tracked.output

        result = runner.invoke(cli.cli, command)

        assert result.exit_code == 0, result.output
        assert payload.read_text() == "ordinary text"
