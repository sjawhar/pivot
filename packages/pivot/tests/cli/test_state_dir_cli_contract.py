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


def _init_project(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, bytes]:
    root = tmp_path / "project"
    root.mkdir()
    _ok(_run(root, "git", "init", "-q"))
    _ok(_run(root, _PIVOT, "init"))
    _ok(_run(root, _PIVOT, "config", "set", "core.max_workers", "2"))
    config_path = root / ".pivot" / "config.yaml"
    return root, config_path, config_path.read_bytes()


@pytest.mark.parametrize("target_kind", ["regular-file", "dangling-symlink"])
def test_config_set_normalizes_parent_segments_before_the_preflight(
    tmp_path: pathlib.Path, target_kind: str
) -> None:
    """`missing/../state` names `state`, so an unusable `state` is refused before writing."""
    root, config_path, before = _init_project(tmp_path)
    if target_kind == "regular-file":
        (root / "state").write_text("not a directory\n")
    else:
        (root / "state").symlink_to(root / "missing-target")

    result = _run(root, _PIVOT, "config", "set", "core.state_dir", "missing/../state")

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert config_path.read_bytes() == before
    assert output.strip().startswith("Error: core.state_dir"), output


def test_config_set_repairs_a_hand_edited_invalid_state_dir(tmp_path: pathlib.Path) -> None:
    """Like 0.5.2, `config set` overwrites an unusable value written by hand."""
    root = tmp_path / "project"
    root.mkdir()
    _ok(_run(root, "git", "init", "-q"))
    _ok(_run(root, _PIVOT, "init"))
    config_path = root / ".pivot" / "config.yaml"
    config_path.write_text("core:\n  state_dir: ''\n")

    result = _run(root, _PIVOT, "config", "set", "core.state_dir", "altstate")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "state_dir: altstate" in config_path.read_text()
    assert "nothing was moved" in output, output
    assert (root / "altstate" / ".gitignore").is_file()


def test_config_unset_state_dir_reports_the_transition_like_set(tmp_path: pathlib.Path) -> None:
    """Unsetting core.state_dir transitions back to `.pivot` with the same notice as set."""
    root = tmp_path / "project"
    root.mkdir()
    _ok(_run(root, "git", "init", "-q"))
    _ok(_run(root, _PIVOT, "init"))
    _ok(_run(root, _PIVOT, "config", "set", "core.state_dir", "altstate"))
    (root / "altstate" / "state.lmdb").mkdir()

    result = _run(root, _PIVOT, "config", "unset", "core.state_dir")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"Existing state at {root / 'altstate'} is no longer used." in output, output
    assert "Removed core.state_dir from local config" in output, output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_config_set_refuses_a_state_dir_under_a_read_only_parent(tmp_path: pathlib.Path) -> None:
    """A target whose nearest existing ancestor is not writable is refused before writing."""
    import os

    if os.geteuid() == 0:
        pytest.skip("root ignores directory write permission")
    root, config_path, before = _init_project(tmp_path)
    read_only = root / "read-only"
    read_only.mkdir()
    read_only.chmod(0o555)
    try:
        result = _run(root, _PIVOT, "config", "set", "core.state_dir", "read-only/state")
    finally:
        read_only.chmod(0o755)

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert config_path.read_bytes() == before
    assert output.strip().startswith("Error: core.state_dir"), output


def _run_with_home(
    cwd: pathlib.Path, home: pathlib.Path, *args: str
) -> subprocess.CompletedProcess[str]:
    import os

    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def _project_with_global_state_dir(
    tmp_path: pathlib.Path, global_state_dir: str
) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text(f"core:\n  state_dir: {global_state_dir}\n")
    root = tmp_path / "project"
    root.mkdir()
    _ok(_run_with_home(root, home, "git", "init", "-q"))
    _ok(_run_with_home(root, home, _PIVOT, "init"))
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    return root, home


def test_global_relative_state_dir_is_rooted_at_each_project(tmp_path: pathlib.Path) -> None:
    """Like 0.5.2, a global relative core.state_dir holds that project's state under its root."""
    root, home = _project_with_global_state_dir(tmp_path, "globalstate")

    repro = _run_with_home(root, home, _PIVOT, "repro")
    reported = _run_with_home(root, home, _PIVOT, "config", "get", "core.state_dir")

    assert repro.returncode == 0, repro.stdout + repro.stderr
    assert (root / "globalstate" / "stages" / "test" / "process.lock").is_file()
    assert (root / "globalstate" / "state.lmdb").is_dir()
    assert not (root / ".pivot" / "state.lmdb").exists()
    assert reported.stdout.strip() == "core.state_dir = globalstate (global)", reported.stdout


def test_local_state_dir_overrides_the_global_one(tmp_path: pathlib.Path) -> None:
    """A project's own core.state_dir wins over the global value."""
    root, home = _project_with_global_state_dir(tmp_path, "globalstate")
    _ok(_run_with_home(root, home, _PIVOT, "config", "set", "core.state_dir", "localstate"))

    repro = _run_with_home(root, home, _PIVOT, "repro")

    assert repro.returncode == 0, repro.stdout + repro.stderr
    assert (root / "localstate" / "stages" / "test" / "process.lock").is_file()
    assert not (root / "globalstate" / "stages" / "test").exists()
    assert not (root / "globalstate" / "state.lmdb").exists()
