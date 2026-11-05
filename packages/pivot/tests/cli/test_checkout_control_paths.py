from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli, config
from pivot.storage import cache, track

if TYPE_CHECKING:
    import pathlib

    import click.testing


def _helper_sidecar(root: pathlib.Path, target: str) -> None:
    source = root / "payload.txt"
    source.write_text("cached replacement\n")
    record = cache.save_to_cache(
        source, config.get_cache_dir() / "files", checkout_mode=cache.CheckoutMode.COPY
    )
    source.unlink()
    path = root / target
    path.parent.mkdir(parents=True, exist_ok=True)
    track.write_pvt_file(
        track.get_pvt_path(path),
        track.PvtData(path=path.name, hash=record["hash"], size=19),
    )


@pytest.mark.parametrize("control", [".git", ".pivot"])
@pytest.mark.parametrize("target_form", ["all", "path", "sidecar"])
def test_forced_checkout_refuses_control_directory_before_restoring_any_file(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, control: str, target_form: str
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        marker = root / control / "keep.txt"
        marker.write_text("metadata must survive\n")
        _helper_sidecar(root, "a.txt")
        _helper_sidecar(root, control)
        targets = {"all": [], "path": [control], "sidecar": [f"{control}.pvt"]}

        result = runner.invoke(cli.cli, ["checkout", "--force", *targets[target_form]])

        assert marker.is_file(), result.output
        assert marker.read_text() == "metadata must survive\n", result.output
        assert result.exit_code == 1, result.output
        assert control in result.output and "Refusing" in result.output, result.output
        assert "Restored" not in result.output, result.output
        assert not (root / "a.txt").exists(), "A refused preflight must restore nothing"


@pytest.mark.parametrize("options", [[], ["--force"], ["--only-missing"]])
@pytest.mark.parametrize("target", [".git/HEAD", ".pivot/config.yaml"])
def test_checkout_refuses_files_inside_control_directories(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    options: list[str],
    target: str,
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        _helper_sidecar(root, target)

        result = runner.invoke(cli.cli, ["checkout", *options, target])

        assert result.exit_code == 1, result.output
        assert target in result.output and "Refusing" in result.output, result.output
        assert not (root / target).exists(), result.output


@pytest.mark.parametrize(
    ("configuration", "target", "control"),
    [
        ("core:\n  state_dir: state\n", "state", "state"),
        ("core:\n  state_dir: data/state\n", "data", "data/state"),
        ("cache:\n  dir: objects\n", "objects", "objects"),
    ],
)
def test_forced_checkout_preserves_configured_control_directories(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    configuration: str,
    target: str,
    control: str,
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / ".pivot" / "config.yaml").write_text(configuration)
        marker = root / control / "keep.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("metadata must survive\n")
        _helper_sidecar(root, target)

        result = runner.invoke(cli.cli, ["checkout", "--force", target])

        assert marker.is_file(), result.output
        assert marker.read_text() == "metadata must survive\n", result.output
        assert result.exit_code == 1, result.output
        assert target in result.output and "Refusing" in result.output, result.output


def test_checkout_refuses_writes_through_an_alias_into_git_metadata(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "alias").symlink_to(root / ".git", target_is_directory=True)
        _helper_sidecar(root, "HEAD")
        sidecar = root / "HEAD.pvt"
        record = track.read_pvt_file(sidecar)
        assert record is not None
        record["path"] = "alias/HEAD"
        track.write_pvt_file(sidecar, record)

        result = runner.invoke(cli.cli, ["checkout", "--force"])

        assert result.exit_code == 1, result.output
        assert "alias/HEAD" in result.output and ".git" in result.output, result.output
        assert not (root / ".git" / "HEAD").exists(), result.output


def test_checkout_still_restores_an_ordinary_dot_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        hidden = root / ".hidden"
        hidden.mkdir()
        (hidden / "file.txt").write_text("cached data\n")
        result = runner.invoke(cli.cli, ["track", ".hidden"])
        assert result.exit_code == 0, result.output
        cache.remove_output(hidden)

        result = runner.invoke(cli.cli, ["checkout", "--force", ".hidden"])

        assert result.exit_code == 0, result.output
        assert (hidden / "file.txt").read_text() == "cached data\n", result.output


def test_checkout_keeps_data_beside_cache_owned_directories(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / ".pivot" / "config.yaml").write_text("cache:\n  dir: shared\n")
        _helper_sidecar(root, "shared/data/file.txt")

        result = runner.invoke(cli.cli, ["checkout", "--force", "shared/data/file.txt"])

        assert result.exit_code == 0, result.output
        assert (root / "shared/data/file.txt").read_text() == "cached replacement\n"


@pytest.mark.parametrize("state_dir", [".", "state"])
def test_checkout_keeps_data_beside_state_owned_entries(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, state_dir: str
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {state_dir}\n")
        target = "data/file.txt" if state_dir == "." else "state/data/file.txt"
        _helper_sidecar(root, target)

        result = runner.invoke(cli.cli, ["checkout", "--force", target])

        assert result.exit_code == 0, result.output
        assert (root / target).read_text() == "cached replacement\n"
