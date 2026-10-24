"""A project Pivot 0.5.2 laid out keeps working after `core.state_dir` is set: its locks and cache
are read where 0.5.2 wrote them."""

from __future__ import annotations

import pathlib
import shutil
from typing import TYPE_CHECKING

import pytest
import yaml

from pivot import cli, project
from pivot.remote import sync as transfer
from tests import helpers

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture

# A project built by Pivot 0.5.2 remains readable after core.state_dir is set.

_LAYOUT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def test_uses_legacy_locks_and_cache_after_state_dir_is_configured(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert helpers.run_cli(root, home, "git", *args).returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_LAYOUT_PIPELINE)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
    assert helpers.run_cli(root, home, "git", "add", "-A").returncode == 0
    assert helpers.run_cli(root, home, "git", "commit", "-qm", "base").returncode == 0

    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")

    verify = helpers.run_pivot(root, home, "verify")
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "Pivot 0.5.2 stored locks and cache" in verify.stderr

    output = root / "data" / "test" / "process.txt"
    output.unlink()
    checkout = helpers.run_pivot(root, home, "checkout", "test/process")
    assert checkout.returncode == 0, checkout.stdout + checkout.stderr
    assert output.read_text() == "INPUT\n"


# Commands on a project that Pivot 0.5.2 laid out, after core.state_dir is set.
#
# Pivot 0.5.2 (`init`, `config set core.state_dir`, `repro`, then `git add -A` and a commit) left
# locks, cache and run history in `.pivot`; in the configured state directory only `locks/` and
# `state.lmdb`, with no `.gitignore`, so a relative one is committed with `.pivot/config.yaml`.
# This test writes that layout. The default `.pivot` layout is byte-identical to 0.5.2's, so the
# head builds it. Each case (none, a relative `altstate`, an absolute directory outside the
# project) then runs one command per copy. Every command keeps 0.5.2's exit code (0). Configured
# layouts read the legacy locks and cache and print the migration warning once; the unconfigured
# layout prints none.

_MATRIX_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


_WARNING = "Pivot 0.5.2 stored locks and cache"


# name -> core.state_dir value ({outside} is a directory beside the project).
_CASES: dict[str, str | None] = {
    "none": None,
    "altstate": "altstate",
    "abs-outside": "{outside}/st",
}


# Commands, with whether each reads stage locks or cache objects.
_COMMANDS: dict[str, tuple[tuple[str, ...], bool]] = {
    "status": (("status",), True),
    "verify": (("verify",), True),
    "list": (("list",), False),
    "checkout-deleted": (("checkout", "test/process"), True),
    "get-rev": (("get", "--rev", "HEAD", "test/process", "-o", "restored.txt"), True),
    "repro": (("repro",), True),
    "params-diff": (("params", "diff"), True),
    "config-get-cache": (("config", "get", "cache.dir"), False),
}


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    directory = tmp_path_factory.mktemp("legacy-base")
    home = directory / "home"
    home.mkdir()
    root = directory / "project"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert helpers.run_cli(root, home, "git", *args).returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_MATRIX_PIPELINE)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
    assert helpers.run_cli(root, home, "git", "add", "-A").returncode == 0
    assert helpers.run_cli(root, home, "git", "commit", "-qm", "legacy layout").returncode == 0
    return directory


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("command", list(_COMMANDS))
def test_legacy_layout_command_matrix(
    base: pathlib.Path, tmp_path: pathlib.Path, case: str, command: str
) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(base, copy, symlinks=True)
    (copy / "outside").mkdir()
    root = copy / "project"
    state_dir = _CASES[case]
    if state_dir is not None:
        value = state_dir.format(outside=copy / "outside")
        (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")
        # 0.5.2 wrote only its state database and artifact locks to the configured directory.
        configured = root / value
        (configured / "locks").mkdir(parents=True)
        shutil.copytree(root / ".pivot" / "state.lmdb", configured / "state.lmdb")
        assert helpers.run_cli(root, copy / "home", "git", "add", "-A").returncode == 0
        assert (
            helpers.run_cli(root, copy / "home", "git", "commit", "-qm", "configured").returncode
            == 0
        )
    if command == "checkout-deleted":
        (root / "data" / "test" / "process.txt").unlink()
    args, reads_legacy = _COMMANDS[command]

    result = helpers.run_pivot(root, copy / "home", *args)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Traceback" not in output, output
    if state_dir is not None and reads_legacy:
        assert result.stderr.count(_WARNING) == 1, result.stderr
    else:
        assert _WARNING not in result.stderr, result.stderr
    match command:
        case "status":
            assert "1 cached" in result.stdout, result.stdout
        case "repro":
            # 0.5.2 skipped the stage from its committed lock ("0 ran, 1 cached").
            assert "test/process cached" in result.stdout, result.stdout
            assert "0 ran, 1 cached" in output, output
        case "checkout-deleted":
            assert (root / "data" / "test" / "process.txt").read_text() == "INPUT\n"
        case "get-rev":
            assert (root / "restored.txt").read_text() == "INPUT\n"
        case _:
            pass


# `diff` reads the legacy cache after core.state_dir changes.

_DIFF_PIPELINE = """\
from typing import Annotated

import pandas

from pivot import Pipeline, loaders, stage


@stage
def process(value: str) -> Annotated[pandas.DataFrame, loaders.CSV()]:
    return pandas.DataFrame({"id": [1], "value": [value.strip()]})


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


_METRIC_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, metric, stage


@stage
def process(value: str) -> Annotated[dict, metric]:
    return {"score": int(value.strip())}


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def test_data_diff_reads_a_legacy_cache_object(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert helpers.run_cli(root, home, "git", *args).returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("old\n")
    (root / "pipeline.py").write_text(_DIFF_PIPELINE)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
    assert helpers.run_cli(root, home, "git", "add", "-A").returncode == 0
    assert helpers.run_cli(root, home, "git", "commit", "-qm", "legacy layout").returncode == 0
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    output = root / "data" / "test" / "process.csv"
    output.unlink()
    output.write_text("id,value\n1,new\n")

    diff = helpers.run_pivot(root, home, "diff", "--no-tui", "data/test/process.csv")

    assert diff.returncode == 0, diff.stdout + diff.stderr
    assert "Rows: 1 -> 1" in diff.stdout
    assert "~1 modified" in diff.stdout
    assert diff.stderr.count("Pivot 0.5.2 stored locks and cache") == 1, diff.stderr


def test_metrics_diff_reads_a_legacy_cache_object(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert helpers.run_cli(root, home, "git", *args).returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("1\n")
    (root / "pipeline.py").write_text(_METRIC_PIPELINE)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
    # The metric is not committed, so the old side can come only from the cache, as 0.5.2 read it.
    (root / ".gitignore").write_text("/metrics/\n")
    assert helpers.run_cli(root, home, "git", "add", "-A").returncode == 0
    assert helpers.run_cli(root, home, "git", "commit", "-qm", "legacy layout").returncode == 0
    lock_data = yaml.safe_load((root / ".pivot" / "stages" / "test" / "process.lock").read_text())
    file_hash = lock_data["outs"][0]["hash"]
    cached = root / ".pivot" / "cache" / "files" / file_hash[:2] / file_hash[2:]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text("score: 1\n")
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    output = root / "metrics" / "test" / "process.yaml"
    output.write_text("score: 2\n")

    diff = helpers.run_pivot(root, home, "metrics", "diff", "metrics/test/process.yaml")

    assert diff.returncode == 0, diff.stdout + diff.stderr
    assert "modified" in diff.stdout.lower(), diff.stdout
    assert "added" not in diff.stdout.lower(), diff.stdout
    assert diff.stderr.count("Pivot 0.5.2 stored locks and cache") == 1, diff.stderr


def test_metrics_diff_does_not_read_a_symlinked_legacy_cache_object(
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert helpers.run_cli(root, home, "git", *args).returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("1\n")
    (root / "pipeline.py").write_text(_METRIC_PIPELINE)
    assert helpers.run_pivot(root, home, "repro").returncode == 0
    (root / ".gitignore").write_text("/metrics/\n")
    assert helpers.run_cli(root, home, "git", "add", "-A").returncode == 0
    assert helpers.run_cli(root, home, "git", "commit", "-qm", "metric base").returncode == 0
    lock_data = yaml.safe_load((root / ".pivot" / "stages" / "test" / "process.lock").read_text())
    file_hash = lock_data["outs"][0]["hash"]
    cache_object = root / ".pivot" / "cache" / "files" / file_hash[:2] / file_hash[2:]
    cache_object.parent.mkdir(parents=True, exist_ok=True)
    cache_object.unlink(missing_ok=True)
    private_metric = root / "private-metrics.json"
    private_metric.write_text('{"score": 999}\n')
    cache_object.symlink_to(private_metric)
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    (root / "metrics" / "test" / "process.yaml").write_text("score: 2\n")

    diff = helpers.run_pivot(root, home, "metrics", "diff", "metrics/test/process.yaml")

    assert diff.returncode == 0, diff.stdout + diff.stderr
    assert "999" not in diff.stdout


# Push dry-run reports a legacy cache scan once.

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
