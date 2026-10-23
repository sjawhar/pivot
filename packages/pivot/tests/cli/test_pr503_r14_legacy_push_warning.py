"""Push dry-run reports a legacy cache scan once."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot import cli, project
from pivot.remote import sync as transfer

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture

_HASH = "abc1234567890def"


def test_push_dry_run_warns_for_a_legacy_cache(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = pathlib.Path.cwd()
        (root / ".git").mkdir()
        (root / ".pivot").mkdir()
        (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
        cached = root / ".pivot" / "cache" / "files" / _HASH[:2] / _HASH[2:]
        cached.parent.mkdir(parents=True)
        cached.write_text("cached\n")
        mocker.patch.object(project, "_project_root_cache", None)
        mocker.patch.object(
            transfer, "create_remote_from_name", return_value=(mocker.MagicMock(), "origin")
        )

        result = runner.invoke(cli.cli, ["push", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Would push 1 file(s) to 'origin'" in result.output
    assert result.output.count("Pivot 0.5.2 stored locks and cache") == 1


def test_push_dry_run_stays_silent_for_a_current_cache(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = pathlib.Path.cwd()
        (root / ".git").mkdir()
        (root / ".pivot").mkdir()
        (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
        cached = root / "altstate" / "cache" / "files" / _HASH[:2] / _HASH[2:]
        cached.parent.mkdir(parents=True)
        cached.write_text("cached\n")
        mocker.patch.object(project, "_project_root_cache", None)
        mocker.patch.object(
            transfer, "create_remote_from_name", return_value=(mocker.MagicMock(), "origin")
        )

        result = runner.invoke(cli.cli, ["push", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Would push 1 file(s) to 'origin'" in result.output
    assert "Pivot 0.5.2 stored locks and cache" not in result.output
