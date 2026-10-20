from __future__ import annotations

import asyncio
import json
import pathlib
import signal
import socket
import subprocess
import time
from typing import TYPE_CHECKING

import pydantic
import pytest
import yaml

from conftest import GitRepo, init_git_repo
from pivot import cli, compose, exceptions, import_artifact, parameters, project, stage_def
from pivot.cli import completion
from pivot.config import io as config_io
from pivot.show import common as show_common

if TYPE_CHECKING:
    import click.testing


@compose.stage
def writes_state(input_file: str) -> str:
    return input_file.upper()


class _SecretParams(stage_def.StageParams):
    token: pydantic.SecretStr


def _configure_state_dir(root: pathlib.Path) -> pathlib.Path:
    pivot_dir = root / ".pivot"
    pivot_dir.mkdir(exist_ok=True)
    (pivot_dir / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    config_io.clear_config_cache()
    return root / "altstate"


def _pivot_entries(root: pathlib.Path) -> list[str]:
    return sorted(path.name for path in (root / ".pivot").iterdir())


def _lock_text() -> str:
    return yaml.safe_dump({"code_manifest": {}, "params": {}, "deps": [], "outs": []})


def test_repro_keeps_all_state_in_the_configured_state_dir(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """A configured state dir holds repro state while .pivot keeps its config marker."""
    root = mock_discovery.root
    state_dir = _configure_state_dir(root)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "input.txt").write_text("data")
    with mock_discovery:
        writes_state(mock_discovery.input("input.txt", t=str))

    initialized = runner.invoke(cli.cli, ["init", "--force"])
    assert initialized.exit_code == 0, initialized.output
    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, result.output
    assert _pivot_entries(root) == ["config.yaml"]
    assert (state_dir / "stages" / "test" / "writes_state.lock").is_file()
    assert (state_dir / "state.lmdb").is_dir()
    assert (state_dir / "locks").is_dir()
    assert (state_dir / ".gitignore").is_file()


def test_init_writes_gitignore_to_the_configured_state_dir(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """init preserves the marker config and writes runtime ignores beside configured state."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = pathlib.Path.cwd()
        _configure_state_dir(root)
        project._project_root_cache = None

        result = runner.invoke(cli.cli, ["init", "--force"])

        assert result.exit_code == 0, result.output
        assert _pivot_entries(root) == ["config.yaml"]
        assert (root / "altstate" / ".gitignore").is_file()


def test_secret_key_uses_the_configured_state_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Secret digests keep both their key and ignore entry under configured state."""
    state_dir = _configure_state_dir(tmp_path)
    monkeypatch.delenv("PIVOT_SECRET_KEY", raising=False)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    parameters.get_effective_params(
        _SecretParams(token=pydantic.SecretStr("secret")), "test/key", None
    )

    assert (state_dir / "secret.key").is_file()
    assert "/secret.key" in (state_dir / ".gitignore").read_text()
    assert _pivot_entries(tmp_path) == ["config.yaml"]


def test_plots_show_writes_the_default_gallery_to_the_configured_state_dir(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """The default plot gallery belongs to configured state rather than the marker directory."""
    root = mock_discovery.root
    state_dir = _configure_state_dir(root)
    (root / "plot.png").write_bytes(b"plot")

    result = runner.invoke(cli.cli, ["plots", "show", "plot.png"])

    assert result.exit_code == 0, result.output
    assert (state_dir / "plots" / "index.html").is_file()
    assert _pivot_entries(root) == ["config.yaml"]


def test_plots_show_reports_an_absolute_configured_state_dir(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """The default gallery can live outside the project root."""
    root = mock_discovery.root
    state_dir = root.parent / "outside-state"
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {state_dir}\n")
    config_io.clear_config_cache()
    (root / "plot.png").write_bytes(b"plot")

    result = runner.invoke(cli.cli, ["plots", "show", "plot.png"])

    assert result.exit_code == 0, result.output
    assert (state_dir / "plots" / "index.html").is_file()


def test_completion_uses_the_configured_state_directory_cache(tmp_path: pathlib.Path) -> None:
    """Completion's fast path reads the cache written under configured state."""
    state_dir = _configure_state_dir(tmp_path)
    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text("# pipeline\n")

    completion._write_stages_cache(
        tmp_path, "pipeline.py", pipeline_file.stat().st_mtime, ["test/run"]
    )

    assert completion._get_stages_from_cache(tmp_path) == ["test/run"]
    assert (state_dir / "cache" / "stages.cache").is_file()
    assert _pivot_entries(tmp_path) == ["config.yaml"]


def test_show_reads_configured_state_dir_locks_from_git_head(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """show reads committed locks from the configured directory at HEAD."""
    root, commit = git_repo
    state_dir = _configure_state_dir(root)
    lock_file = state_dir / "stages" / "test" / "writes_state.lock"
    lock_file.parent.mkdir(parents=True)
    lock_file.write_text(_lock_text())
    commit("configured state lock")
    monkeypatch.setattr(project, "_project_root_cache", root)
    config_io.clear_config_cache()

    locks = show_common.read_lock_files_from_head(["test/writes_state"])

    assert locks["test/writes_state"] is not None


def test_show_refuses_a_state_dir_outside_the_project(
    git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git-head lock lookup must not escape the repository through core.state_dir."""
    root, _ = git_repo
    (root / ".pivot").mkdir()
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: ../outside-state\n")
    monkeypatch.setattr(project, "_project_root_cache", root)
    config_io.clear_config_cache()

    with pytest.raises(exceptions.ConfigError, match="outside the project"):
        show_common.read_lock_files_from_head(["test/writes_state"])


def test_import_reads_locks_from_the_source_configured_state_dir(tmp_path: pathlib.Path) -> None:
    """Import resolves source locks from that revision's configured state directory."""
    source = tmp_path / "source"
    source.mkdir()
    init_git_repo(source)
    subprocess.run(
        ["git", "config", "uploadArchive.allowUnreachable", "true"],
        cwd=source,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "checkout", "-b", "main"], cwd=source, check=False, capture_output=True)
    (source / ".pivot").mkdir()
    (source / ".pivot" / "config.yaml").write_text(
        "core:\n  state_dir: altstate\nremotes:\n  origin: s3://source-cache\n"
    )
    lock_file = source / "altstate" / "stages" / "test" / "writes_state.lock"
    lock_file.parent.mkdir(parents=True)
    lock_file.write_text(
        yaml.safe_dump(
            {
                "code_manifest": {},
                "params": {},
                "deps": [],
                "outs": [{"key": None, "tag": "data", "hash": "0123456789abcdef"}],
            }
        )
    )
    subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "configured state lock"],
        cwd=source,
        check=True,
        capture_output=True,
    )

    resolved = asyncio.run(
        import_artifact.resolve_remote_path(str(source), "test/writes_state", "main", None)
    )

    assert resolved["stage"] == "test/writes_state"
    assert resolved["remote_url"] == "s3://source-cache"


def test_serve_mode_uses_the_configured_state_directory_socket(tmp_path: pathlib.Path) -> None:
    """The real serve command binds its user-visible socket under configured state."""
    (tmp_path / ".pivot").mkdir()
    (tmp_path / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / "pipeline.py").write_text("""\
from pivot.compose import Pipeline, stage


pipeline = Pipeline("test")


@stage
def noop() -> None:
    pass


with pipeline:
    noop()
""")
    socket_path = tmp_path / "altstate" / "agent.sock"
    proc = subprocess.Popen(
        ["uv", "run", "pivot", "repro", "--watch", "--serve"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        for _ in range(100):
            if socket_path.exists():
                break
            time.sleep(0.1)
        else:
            stdout, stderr = proc.communicate(timeout=1)
            pytest.fail(
                f"Socket not created within 10s.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
            )

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect(str(socket_path))
            sock.sendall(b'{"jsonrpc":"2.0","method":"status","id":1}\n')
            response = json.loads(sock.recv(1024).decode().strip())

        assert "error" not in response, response
        assert "result" in response
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
