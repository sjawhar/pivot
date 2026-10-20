"""Installed-CLI contracts for configured state directories, matching Pivot 0.5.2 where it applies."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def _run(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def _ok(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("layout", ["transition-relative", "transition-absolute", "preconfigured"])
def test_configured_state_dir_keeps_config_lock_git_ignored(
    tmp_path: pathlib.Path, layout: str
) -> None:
    """The restored config.yaml.lock keeps 0.5.2's contract: it exists and git ignores it."""
    root = tmp_path / "project"
    root.mkdir()
    _ok(_run(root, "git", "init", "-q"))
    if layout == "preconfigured":
        (root / ".pivot").mkdir()
        (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: state\n")
        _ok(_run(root, _PIVOT, "init", "--force"))
    else:
        _ok(_run(root, _PIVOT, "init"))
        value = "state" if layout == "transition-relative" else str(tmp_path / "external-state")
        _ok(_run(root, _PIVOT, "config", "set", "core.state_dir", value))
    _ok(_run(root, _PIVOT, "config", "set", "remotes.origin", "s3://unused-bucket/cache"))

    assert (root / ".pivot" / "config.yaml.lock").is_file()
    ignored = _run(root, "git", "check-ignore", "-q", ".pivot/config.yaml.lock")
    assert ignored.returncode == 0, ".pivot/config.yaml.lock is not ignored by git"
    status = _run(root, "git", "status", "--porcelain", "--untracked-files=all", ".pivot")
    untracked = sorted(line[3:] for line in status.stdout.splitlines())
    assert ".pivot/config.yaml.lock" not in untracked, untracked


@pytest.mark.parametrize("target_kind", ["existing-file", "dangling-symlink", "file-parent"])
def test_config_set_state_dir_refusal_is_a_single_error_line(
    tmp_path: pathlib.Path, target_kind: str
) -> None:
    """The installed CLI refuses with one `Error:` line, exit 1, and no traceback."""
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, "git", "init", "-q").returncode == 0
    assert _run(root, _PIVOT, "init").returncode == 0
    assert _run(root, _PIVOT, "config", "set", "core.max_workers", "2").returncode == 0
    config_path = root / ".pivot" / "config.yaml"
    before = config_path.read_bytes()
    if target_kind == "existing-file":
        (root / "state").write_text("not a directory\n")
        value = "state"
    elif target_kind == "dangling-symlink":
        (root / "state").symlink_to(root / "missing")
        value = "state"
    else:
        (root / "parent").write_text("not a directory\n")
        value = "parent/state"

    result = _run(root, _PIVOT, "config", "set", "core.state_dir", value)

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert config_path.read_bytes() == before
    assert "Traceback" not in output, output
    lines = output.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("Error: core.state_dir"), output


def test_config_set_refuses_state_dir_under_dangling_symlink(tmp_path: pathlib.Path) -> None:
    """A parent that is a dangling symlink is a parent that is not a directory: refuse before writing."""
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, "git", "init", "-q").returncode == 0
    assert _run(root, _PIVOT, "init").returncode == 0
    assert _run(root, _PIVOT, "config", "set", "core.max_workers", "2").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    config_path = root / ".pivot" / "config.yaml"
    before = config_path.read_bytes()
    (root / "link").symlink_to(root / "missing-parent")

    result = _run(root, _PIVOT, "config", "set", "core.state_dir", "link/state")
    status = _run(root, _PIVOT, "status")

    output = result.stdout + result.stderr
    assert config_path.read_bytes() == before, config_path.read_text()
    assert status.returncode == 0, status.stdout + status.stderr
    assert result.returncode == 1, output


@pytest.mark.parametrize("spelling", ["relative", "absolute"])
def test_list_and_status_all_keep_project_pipelines_under_ancestor_state_dir(
    tmp_path: pathlib.Path, spelling: str
) -> None:
    """Like 0.5.2, the pre-round-6 tree, and plain `pivot list`, --all finds the root pipeline."""
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, "git", "init", "-q").returncode == 0
    assert _run(root, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    value = ".." if spelling == "relative" else str(tmp_path)
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")

    plain = _run(root, _PIVOT, "list")
    listed = _run(root, _PIVOT, "list", "--all")
    status = _run(root, _PIVOT, "status", "--all")

    assert plain.returncode == 0 and "test/process" in plain.stdout, plain.stdout + plain.stderr
    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert "test/process" in listed.stdout, listed.stdout + listed.stderr
    assert status.returncode == 0, status.stdout + status.stderr
    assert "test/process" in status.stdout, status.stdout + status.stderr


def test_default_init_force_warns_like_pivot_052(tmp_path: pathlib.Path) -> None:
    """Without core.state_dir, overwriting a custom ignore file prints 0.5.2's exact warning."""
    root = tmp_path / "project"
    root.mkdir()
    assert _run(root, "git", "init", "-q").returncode == 0
    assert _run(root, _PIVOT, "init").returncode == 0
    (root / ".pivot" / ".gitignore").write_text("custom\n")

    result = _run(root, _PIVOT, "init", "--force")

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == "Warning: Overwriting existing .pivot/.gitignore\n", result.stderr
