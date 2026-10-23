"""Installed-CLI contracts for configured state directories, matching Pivot 0.5.2 where it applies."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
import yaml

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


def _write_global_state_dir(home: pathlib.Path, value: str) -> pathlib.Path:
    global_config = home / ".config" / "pivot" / "config.yaml"
    global_config.parent.mkdir(parents=True, exist_ok=True)
    global_config.write_text(f"core:\n  state_dir: {value}\n")
    return global_config


def _project(home: pathlib.Path, root: pathlib.Path, input_text: str) -> pathlib.Path:
    root.mkdir()
    _ok(_run_with_home(root, home, "git", "init", "-q"))
    _ok(_run_with_home(root, home, _PIVOT, "init"))
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text(input_text)
    (root / "pipeline.py").write_text(_PIPELINE)
    return root


def _ignored_warning(global_config: pathlib.Path, root: pathlib.Path) -> str:
    return (
        f"core.state_dir in {global_config} is ignored; set it in {root / '.pivot' / 'config.yaml'}"
    )


def _stage_states(home: pathlib.Path, root: pathlib.Path) -> str:
    status = _run_with_home(root, home, _PIVOT, "status")
    assert status.returncode == 0, status.stdout + status.stderr
    return status.stdout


def test_projects_sharing_a_global_state_dir_keep_correct_skip_decisions(
    tmp_path: pathlib.Path,
) -> None:
    """A global core.state_dir is ignored, so two projects never share lock files."""
    home = tmp_path / "home"
    global_config = _write_global_state_dir(home, str(tmp_path / "shared"))
    a = _project(home, tmp_path / "a", "input-a\n")
    b = _project(home, tmp_path / "b", "input-b\n")

    for root in (a, b):
        repro = _run_with_home(root, home, _PIVOT, "repro")
        assert repro.returncode == 0, repro.stdout + repro.stderr
        assert _ignored_warning(global_config, root) in repro.stderr, repro.stderr

    assert "1 cached, 0 stale" in _stage_states(home, a)
    assert "1 cached, 0 stale" in _stage_states(home, b)

    (b / "data" / "raw" / "input.txt").write_text("input-b-changed\n")
    _ok(_run_with_home(b, home, _PIVOT, "repro"))

    assert "1 cached, 0 stale" in _stage_states(home, a)
    assert "1 cached, 0 stale" in _stage_states(home, b)
    for root in (a, b):
        assert (root / ".pivot" / "stages" / "test" / "process.lock").is_file()
    assert not (tmp_path / "shared" / "stages").exists()


def test_config_get_and_list_show_a_global_state_dir_as_ignored(tmp_path: pathlib.Path) -> None:
    """The reporting surfaces never present an ignored global value as in effect."""
    home = tmp_path / "home"
    global_config = _write_global_state_dir(home, "globalstate")
    root = _project(home, tmp_path / "project", "input\n")

    got = _run_with_home(root, home, _PIVOT, "config", "get", "core.state_dir")
    listed = _run_with_home(root, home, _PIVOT, "config", "list")

    assert got.returncode == 0, got.stdout + got.stderr
    assert got.stdout.strip() == "core.state_dir = globalstate (global, ignored)", got.stdout
    assert _ignored_warning(global_config, root) in got.stderr, got.stderr
    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert "core.state_dir = globalstate (global, ignored)" in listed.stdout, listed.stdout


def test_config_set_global_state_dir_is_refused(tmp_path: pathlib.Path) -> None:
    """A new global write is refused with the same one-line message, leaving no global file."""
    home = tmp_path / "home"
    root = _project(home, tmp_path / "project", "input\n")
    global_config = home / ".config" / "pivot" / "config.yaml"

    result = _run_with_home(
        root, home, _PIVOT, "config", "set", "--global", "core.state_dir", "shared"
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert output.strip() == f"Error: {_ignored_warning(global_config, root)}", output
    assert not global_config.exists()


def test_config_list_marks_a_hand_edited_empty_state_dir(tmp_path: pathlib.Path) -> None:
    """N2: with ``state_dir: ''``, config list exits 0 as 0.5.2 does, and marks that row invalid.

    A non-string value (e.g. a list) already fails 0.5.2's schema validation, so it is not here.
    """
    root = tmp_path / "project"
    root.mkdir()
    _ok(_run(root, "git", "init", "-q"))
    _ok(_run(root, _PIVOT, "init"))
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: ''\n")

    listed = _run(root, _PIVOT, "config", "list")

    output = listed.stdout + listed.stderr
    assert listed.returncode == 0, output
    assert "Traceback" not in output, output
    assert "cache.dir = .pivot/cache (default)" in listed.stdout, listed.stdout
    invalid_rows = [line for line in listed.stdout.splitlines() if "invalid" in line]
    assert len(invalid_rows) == 1, listed.stdout
    assert invalid_rows[0].startswith("core.state_dir = "), listed.stdout


def test_warning_names_a_shadowed_global_state_dir(tmp_path: pathlib.Path) -> None:
    """N3: when the project sets its own value, the global one is reported as shadowed."""
    home = tmp_path / "home"
    global_config = _write_global_state_dir(home, "globalstate")
    root = _project(home, tmp_path / "project", "input\n")
    _ok(_run_with_home(root, home, _PIVOT, "config", "set", "core.state_dir", "localstate"))

    status = _run_with_home(root, home, _PIVOT, "status")

    expected = (
        f"core.state_dir in {global_config} is shadowed by "
        f"{root / '.pivot' / 'config.yaml'} and ignored"
    )
    assert status.returncode == 0, status.stdout + status.stderr
    assert status.stderr.splitlines().count(expected) == 1, status.stderr
    assert "set it in" not in status.stderr, status.stderr


def test_cache_dir_completion_help_matches_pivot_052() -> None:
    """N5: the completion description for cache.dir is 0.5.2's text."""
    from pivot.config import models

    assert models.CONFIG_KEY_DESCRIPTIONS["cache.dir"] == "Local cache directory"


_UNKNOWN_STATE_DIR = ". The state directory is unknown; Pivot uses the default .pivot."


def _project_with_invalid_local_config(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    home = tmp_path / "home"
    root = _project(home, tmp_path / "project", "input\n")
    config_path = root / ".pivot" / "config.yaml"
    config_path.write_text("core: [unclosed\n")
    return root, f"Invalid YAML in {config_path}:"


@pytest.mark.parametrize(
    "args",
    [("list",), ("list", "--all"), ("dag",)],
    ids=lambda args: "-".join(args),
)
def test_list_and_dag_warn_on_an_invalid_local_config(
    tmp_path: pathlib.Path, args: tuple[str, ...]
) -> None:
    """Like 0.5.2, listing never needs the config; the unknown state directory is warned once."""
    root, invalid_yaml = _project_with_invalid_local_config(tmp_path)

    result = _run_with_home(root, tmp_path / "home", _PIVOT, *args)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "test/process" in result.stdout, result.stdout
    assert result.stderr.count(invalid_yaml) == 1, result.stderr
    assert result.stderr.count(_UNKNOWN_STATE_DIR) == 1, result.stderr
    assert "Invalid YAML" not in result.stdout, result.stdout


def test_list_json_stays_parseable_with_an_invalid_local_config(tmp_path: pathlib.Path) -> None:
    """The warning goes to stderr only, so `list --json` output still parses."""
    import json

    root, invalid_yaml = _project_with_invalid_local_config(tmp_path)

    result = _run_with_home(root, tmp_path / "home", _PIVOT, "list", "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    assert [stage["name"] for stage in json.loads(result.stdout)["stages"]] == ["test/process"]
    assert result.stderr.count(invalid_yaml) == 1, result.stderr


@pytest.mark.parametrize("command", ["status", "repro"])
def test_state_commands_fail_on_an_invalid_local_config_as_on_052(
    tmp_path: pathlib.Path, command: str
) -> None:
    """Every command that uses state still exits 1 with 0.5.2's one `Error:` line."""
    root, invalid_yaml = _project_with_invalid_local_config(tmp_path)

    result = _run_with_home(root, tmp_path / "home", _PIVOT, command)

    assert result.returncode == 1, result.stdout + result.stderr
    errors = [line for line in result.stderr.splitlines() if line.startswith("Error")]
    assert errors and errors[0].startswith(f"Error: {invalid_yaml}"), result.stderr
    assert not (root / ".pivot" / "state.lmdb").exists()


@pytest.mark.parametrize("layout", ["configured", "legacy"])
@pytest.mark.parametrize("object_kind", ["symlink", "directory"])
def test_push_dry_run_ignores_a_cache_object_that_is_not_a_regular_file(
    tmp_path: pathlib.Path, layout: str, object_kind: str
) -> None:
    """A symlink or directory at a cache object path is absent, in the configured and 0.5.2 caches."""
    home = tmp_path / "home"
    home.mkdir()
    root = _project(home, tmp_path / "project", "input\n")
    if layout == "configured":
        _ok(_run_with_home(root, home, _PIVOT, "config", "set", "core.state_dir", "altstate"))
    _ok(_run_with_home(root, home, _PIVOT, "repro"))
    if layout == "legacy":
        _ok(_run_with_home(root, home, _PIVOT, "config", "set", "core.state_dir", "altstate"))
    _ok(_run_with_home(root, home, _PIVOT, "config", "set", "remotes.origin", "s3://cache-objects"))

    state = root / "altstate" if layout == "configured" else root / ".pivot"
    lock = yaml.safe_load((state / "stages" / "test" / "process.lock").read_text())
    file_hash = lock["outs"][0]["hash"]
    cache_object = state / "cache" / "files" / file_hash[:2] / file_hash[2:]
    cache_object.unlink()
    marker = "CACHE-OBJECT-TARGET"
    if object_kind == "symlink":
        target = root / "outside-cache-object"
        target.write_text(f"{marker}\n")
        cache_object.symlink_to(target)
    else:
        cache_object.mkdir()

    result = _run_with_home(root, home, _PIVOT, "push", "--dry-run", "--remote", "origin")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert marker not in output, output
    assert "No files to push" in result.stdout, output


_METRIC_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, metric, stage


@stage
def process(value: str) -> Annotated[dict, metric]:
    return {"score": int(value.strip())}


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""


def _break_cache_prefix(prefix: pathlib.Path, prefix_kind: str) -> None:
    """Replace a cache prefix directory with a regular file or a symlink loop."""
    import shutil

    if prefix.is_dir() and not prefix.is_symlink():
        shutil.rmtree(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if prefix_kind == "file":
        prefix.write_text("not a cache prefix directory\n")
    else:
        prefix.symlink_to(prefix.name)


@pytest.mark.parametrize("prefix_kind", ["file", "symlink-loop"])
def test_metrics_diff_treats_an_unusable_cache_prefix_as_missing(
    tmp_path: pathlib.Path, prefix_kind: str
) -> None:
    """0.5.2 and main exit 0 and read the committed metric from git when the prefix is not a directory."""
    home = tmp_path / "home"
    root = _project(home, tmp_path / "project", "1\n")
    (root / "pipeline.py").write_text(_METRIC_PIPELINE)
    for args in (
        ("git", "config", "user.email", "t@example.com"),
        ("git", "config", "user.name", "t"),
        (_PIVOT, "repro"),
        ("git", "add", "-A"),
        ("git", "commit", "-qm", "base"),
    ):
        _ok(_run_with_home(root, home, *args))
    lock = yaml.safe_load((root / ".pivot" / "stages" / "test" / "process.lock").read_text())
    _break_cache_prefix(
        root / ".pivot" / "cache" / "files" / lock["outs"][0]["hash"][:2], prefix_kind
    )
    (root / "metrics" / "test" / "process.yaml").write_text("score: 2\n")

    result = _run_with_home(root, home, _PIVOT, "metrics", "diff", "metrics/test/process.yaml")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "modified" in result.stdout, output
    assert "Error" not in output, output


@pytest.mark.parametrize("prefix_kind", ["file", "symlink-loop"])
def test_checkout_reports_a_missing_object_when_the_cache_prefix_is_unusable(
    tmp_path: pathlib.Path, prefix_kind: str
) -> None:
    """0.5.2 and main say the object is missing and name `pivot pull`, not a raw errno."""
    home = tmp_path / "home"
    root = _project(home, tmp_path / "project", "input\n")
    _ok(_run_with_home(root, home, _PIVOT, "repro"))
    lock = yaml.safe_load((root / ".pivot" / "stages" / "test" / "process.lock").read_text())
    _break_cache_prefix(
        root / ".pivot" / "cache" / "files" / lock["outs"][0]["hash"][:2], prefix_kind
    )
    (root / "data" / "test" / "process.txt").unlink()

    result = _run_with_home(root, home, _PIVOT, "checkout", "data/test/process.txt")

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "Missing 1 file(s)" in output, output
    assert "pivot pull" in output, output
    assert "Errno" not in output, output
    assert "Traceback" not in output, output


def test_metrics_diff_treats_a_cache_dir_that_is_a_file_as_missing(tmp_path: pathlib.Path) -> None:
    """Main exits 0 when PIVOT_CACHE_DIR names a regular file; the committed metric is read from git."""
    import os

    home = tmp_path / "home"
    root = _project(home, tmp_path / "project", "1\n")
    (root / "pipeline.py").write_text(_METRIC_PIPELINE)
    for args in (
        ("git", "config", "user.email", "t@example.com"),
        ("git", "config", "user.name", "t"),
        (_PIVOT, "repro"),
        ("git", "add", "-A"),
        ("git", "commit", "-qm", "base"),
    ):
        _ok(_run_with_home(root, home, *args))
    cache_file = tmp_path / "cache-is-a-file"
    cache_file.write_text("not a directory\n")
    (root / "metrics" / "test" / "process.yaml").write_text("score: 2\n")

    result = subprocess.run(
        [_PIVOT, "metrics", "diff", "metrics/test/process.yaml"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME": str(home), "PIVOT_CACHE_DIR": str(cache_file)},
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "modified" in result.stdout, output


@pytest.mark.parametrize("command", ["fetch", "pull"])
def test_fetch_and_pull_download_under_an_empty_state_dir(
    tmp_path: pathlib.Path, moto_services: dict[str, str], command: str
) -> None:
    """0.5.2 exits 0 with `1 transferred` under a hand-edited `core.state_dir: ''`; so does the head."""
    import os

    import boto3
    import xxhash

    s3 = boto3.client(
        "s3",
        endpoint_url=moto_services["s3"],
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    )
    bucket = f"empty-state-dir-{command}"
    s3.create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
    content = b"remote object\n"
    digest = xxhash.xxh64(content).hexdigest()
    s3.put_object(  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
        Bucket=bucket, Key=f"cache/files/{digest[:2]}/{digest[2:]}", Body=content
    )

    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    _ok(_run_with_home(root, home, "git", "init", "-q"))
    _ok(_run_with_home(root, home, _PIVOT, "init"))
    (root / ".pivot" / "config.yaml").write_text(
        f"core:\n  state_dir: ''\nremotes:\n  origin: s3://{bucket}/cache\ndefault_remote: origin\n"
    )
    (root / "pipeline.py").write_text("from pivot import Pipeline\n\npipeline = Pipeline('test')\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIVOT", "AWS"))}
    env.update(
        HOME=str(home),
        AWS_ENDPOINT_URL=moto_services["s3"],
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )

    result = subprocess.run(
        [_PIVOT, command], cwd=root, capture_output=True, text=True, check=False, env=env
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 transferred" in output, output
