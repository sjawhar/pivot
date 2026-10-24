from __future__ import annotations

import asyncio
import json
import logging
import os
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
from pivot import cli, compose, config, exceptions, import_artifact, parameters, project, stage_def
from pivot import state_dir as state_dir_mod
from pivot.cli import completion
from pivot.config import io as config_io
from pivot.show import common as show_common
from pivot.storage import restore
from tests import helpers

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture


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


def _write_acceptance_pipeline(root: pathlib.Path) -> None:
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("data")
    (root / "pipeline.py").write_text("""\
from pivot.compose import Pipeline, stage


@stage
def upper(value: str) -> str:
    return value.upper()


with Pipeline("state") as pipeline:
    upper(pipeline.input("input.txt", t=str))
""")


def test_configuring_state_dir_after_init_keeps_marker_gitignore(
    tmp_path: pathlib.Path,
) -> None:
    """The documented init/config/repro sequence relocates future state without moving old state."""
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()

    initialized = helpers.run_pivot(root, home, "init")
    configured = helpers.run_pivot(root, home, "config", "set", "core.state_dir", "altstate")
    _write_acceptance_pipeline(root)
    repro = helpers.run_pivot(root, home, "repro")

    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert repro.returncode == 0, repro.stdout + repro.stderr
    assert _pivot_entries(root) == [".gitignore", "config.yaml", "config.yaml.lock", "stages"]
    assert (root / ".pivot" / ".gitignore").read_text() == state_dir_mod.STATE_GITIGNORE_CONTENT
    assert (root / "altstate" / ".gitignore").is_file()
    assert "no longer used" in configured.stdout + configured.stderr


def test_repro_after_hand_editing_state_dir_creates_its_gitignore(tmp_path: pathlib.Path) -> None:
    """A hand-edited state setting creates the configured directory's ignore file on repro."""
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()

    initialized = helpers.run_pivot(root, home, "init")
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    _write_acceptance_pipeline(root)
    repro = helpers.run_pivot(root, home, "repro")

    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    assert repro.returncode == 0, repro.stdout + repro.stderr
    assert (root / "altstate" / ".gitignore").is_file()


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
    assert _pivot_entries(root) == [".gitignore", "config.yaml"]
    assert (state_dir / "stages" / "test" / "writes_state.lock").is_file()
    assert (state_dir / "state.lmdb").is_dir()
    assert (state_dir / "locks").is_dir()
    assert (state_dir / ".gitignore").is_file()


def test_each_in_process_invocation_reads_an_edited_state_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """An embedder's next CLI invocation follows a `core.state_dir` edited after the last one.

    The merged config and the resolved state directory each last one invocation, so the second
    `status` opens its state database and locates its cache under the edited directory.
    """
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".pivot").mkdir()
    _write_acceptance_pipeline(root)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PIVOT_CACHE_DIR", raising=False)
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", None)
    located_cache = mocker.spy(config, "locate_cache_dir")

    cli.cli.main(["status"], standalone_mode=False)
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    cli.cli.main(["status"], standalone_mode=False)

    assert (root / "altstate" / "state.lmdb").is_dir()
    assert located_cache.spy_return == (root / "altstate" / "cache").resolve()


@pytest.mark.parametrize(
    ("state_dir_name", "symlink"),
    [
        pytest.param("outside-state", False, id="absolute-state-dir"),
        pytest.param("state-link", True, id="in-project-state-symlink"),
    ],
)
def test_init_force_preserves_external_state_dir_gitignore(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    state_dir_name: str,
    symlink: bool,
) -> None:
    """Force init never overwrites an ignore file outside the project."""
    root = tmp_path / "project"
    root.mkdir()
    external_state_dir = tmp_path / "outside-state"
    external_state_dir.mkdir()
    external_ignore = external_state_dir / ".gitignore"
    external_ignore.write_text("external content\n")
    (root / ".pivot").mkdir()
    if symlink:
        (root / state_dir_name).symlink_to(external_state_dir, target_is_directory=True)
        configured_state_dir = state_dir_name
    else:
        configured_state_dir = str(external_state_dir)
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {configured_state_dir}\n")
    monkeypatch.chdir(root)
    project._project_root_cache = None
    config_io.clear_config_cache()

    result = runner.invoke(cli.cli, ["init", "--force"])

    assert result.exit_code == 0, result.output
    assert external_ignore.read_text() == "external content\n"


def test_init_force_preserves_project_root_gitignore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root state directory is not Pivot-owned for force-overwrite purposes."""
    root = tmp_path / "project"
    root.mkdir()
    root_ignore = root / ".gitignore"
    root_ignore.write_text("repository content\n")
    (root / ".pivot").mkdir()
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: .\n")
    monkeypatch.chdir(root)
    project._project_root_cache = None
    config_io.clear_config_cache()

    result = runner.invoke(cli.cli, ["init", "--force"])

    assert result.exit_code == 0, result.output
    assert root_ignore.read_text() == "repository content\n"


def test_init_force_preserves_symlinked_internal_state_gitignore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlinked ignore file is not Pivot-owned even inside the project."""
    root = tmp_path / "project"
    root.mkdir()
    target = tmp_path / "outside-ignore"
    target.write_text("external content\n")
    state_dir = root / "state"
    state_dir.mkdir()
    (state_dir / ".gitignore").symlink_to(target)
    (root / ".pivot").mkdir()
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: state\n")
    monkeypatch.chdir(root)
    project._project_root_cache = None
    config_io.clear_config_cache()

    result = runner.invoke(cli.cli, ["init", "--force"])

    assert result.exit_code == 0, result.output
    assert target.read_text() == "external content\n"


@pytest.mark.parametrize(
    ("configured_state_dir", "symlink_to_git"),
    [
        pytest.param("state", False, id="configured-directory"),
        pytest.param(".git", False, id="git-directory"),
        pytest.param("state-link", True, id="symlink-to-git"),
    ],
)
def test_init_force_preserves_existing_configured_state_ignore(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_state_dir: str,
    symlink_to_git: bool,
) -> None:
    """Configured state never grants force-overwrite ownership of its ignore file."""
    root = tmp_path / "project"
    root.mkdir()
    git_dir = root / ".git"
    git_dir.mkdir()
    if symlink_to_git:
        (root / "state-link").symlink_to(git_dir, target_is_directory=True)
        state_dir = root / "state-link"
    else:
        state_dir = root / configured_state_dir
        state_dir.mkdir(exist_ok=True)
    state_ignore = state_dir / ".gitignore"
    state_ignore.write_text("existing content\n")
    (root / ".pivot").mkdir()
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {configured_state_dir}\n")
    monkeypatch.chdir(root)
    project._project_root_cache = None
    config_io.clear_config_cache()

    result = runner.invoke(cli.cli, ["init", "--force"])

    assert result.exit_code == 0, result.output
    assert state_ignore.read_text() == "existing content\n"
    # The notice names the file actually kept: the resolver follows ``state-link``.
    kept = pathlib.Path(os.path.realpath(state_ignore))
    assert f"Leaving {kept}; Pivot does not own it." in result.output
    created = result.output.split("Created:\n", 1)[1].split("\n\nNext steps:", 1)[0]
    assert f"  {kept.relative_to(root)}" not in created


def test_init_force_overwrites_default_state_ignore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """The unset state directory preserves main's force-overwrite behavior."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        initial = runner.invoke(cli.cli, ["init"])
        assert initial.exit_code == 0, initial.output
        ignore_path = pathlib.Path(".pivot/.gitignore")
        ignore_path.write_text("custom content\n")

        result = runner.invoke(cli.cli, ["init", "--force"])

        assert result.exit_code == 0, result.output
        assert ignore_path.read_text() == state_dir_mod.STATE_GITIGNORE_CONTENT


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
        assert _pivot_entries(root) == [".gitignore", "config.yaml"]
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
    """Completion's fast path reads the cache written under configured state.

    Listing stages never creates the configured directory, so nothing is cached until a
    command that uses the state directory has created it.
    """
    state_dir = _configure_state_dir(tmp_path)
    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text("# pipeline\n")
    mtime = pipeline_file.stat().st_mtime

    completion._write_stages_cache(tmp_path, "pipeline.py", mtime, ["test/run"])

    assert not state_dir.exists()
    assert completion._get_stages_from_cache(tmp_path) is None

    state_dir.mkdir()
    completion._write_stages_cache(tmp_path, "pipeline.py", mtime, ["test/run"])

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


@pytest.mark.parametrize(
    ("configured_state_dir", "symlink"),
    [
        pytest.param("nested/../actual-state", False, id="parent-segment"),
        pytest.param("state-link", True, id="in-project-symlink"),
    ],
)
def test_revision_lock_read_normalizes_configured_state_dir(
    git_repo: GitRepo,
    monkeypatch: pytest.MonkeyPatch,
    configured_state_dir: str,
    symlink: bool,
) -> None:
    """Revision reads use the Git path of the resolved in-project state directory."""
    root, commit = git_repo
    (root / ".pivot").mkdir()
    actual_state_dir = root / "actual-state"
    lock_file = actual_state_dir / "stages" / "test" / "writes_state.lock"
    lock_file.parent.mkdir(parents=True)
    lock_file.write_text(_lock_text())
    if symlink:
        (root / "state-link").symlink_to(actual_state_dir, target_is_directory=True)
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {configured_state_dir}\n")
    commit("configured revision lock")
    monkeypatch.setattr(project, "_project_root_cache", root)
    config_io.clear_config_cache()

    lock_data = restore.get_lock_data_from_revision(
        "test/writes_state",
        "HEAD",
        state_dir_mod.get_state_dir(),
    )

    assert lock_data is not None


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


@pytest.mark.parametrize(
    ("configured_state_dir", "symlink", "error"),
    [
        pytest.param("nested/../actual-state", False, None, id="parent-segment"),
        pytest.param(".", False, None, id="repository-root"),
        pytest.param("state/..", False, None, id="root-after-parent"),
        # The symlink is reported and 0.5.2's .pivot is read, which this source lacks.
        pytest.param("state-link", True, "no .pivot/stages directory", id="state-symlink"),
    ],
)
def test_import_normalizes_or_ignores_source_state_dir(
    tmp_path: pathlib.Path,
    configured_state_dir: str,
    symlink: bool,
    error: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A local git import normalizes source paths; a symlinked one falls back to `.pivot`."""
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
    actual_state_dir = (
        source if configured_state_dir in {".", "state/.."} else source / "actual-state"
    )
    lock_file = actual_state_dir / "stages" / "test" / "writes_state.lock"
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
    if symlink:
        (source / "state-link").symlink_to(actual_state_dir, target_is_directory=True)
    (source / ".pivot" / "config.yaml").write_text(
        "core:\n"
        + f"  state_dir: {configured_state_dir}\n"
        + "remotes:\n"
        + "  origin: s3://source-cache\n"
    )
    subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "configured source state"],
        cwd=source,
        check=True,
        capture_output=True,
    )

    if error is not None:
        with (
            caplog.at_level(logging.WARNING, logger="pivot.import_artifact"),
            pytest.raises(exceptions.RemoteError, match=error),
        ):
            asyncio.run(
                import_artifact.resolve_remote_path(str(source), "test/writes_state", "main", None)
            )
        assert "'state-link' is a symlink; reading its locks from the default .pivot" in (
            caplog.text
        )
    else:
        resolved = asyncio.run(
            import_artifact.resolve_remote_path(str(source), "test/writes_state", "main", None)
        )
        assert resolved["stage"] == "test/writes_state"


@pytest.mark.parametrize(
    "value",
    ["", 5, ["a"], "/abs/state", "../outside"],
    ids=["empty", "integer", "list", "absolute", "escaping"],
)
def test_import_ignores_a_source_state_dir_that_is_not_a_repo_path(
    value: object, caplog: pytest.LogCaptureFixture
) -> None:
    """0.5.2 read every source's locks from `.pivot`; an unusable value is reported, not refused."""
    with caplog.at_level(logging.WARNING, logger="pivot.import_artifact"):
        state_dir = import_artifact._remote_state_dir("/source", {"core": {"state_dir": value}})

    assert state_dir == ".pivot"
    assert f"Ignoring core.state_dir in .pivot/config.yaml of /source: {value!r}" in caplog.text


def test_import_reads_a_source_config_with_a_duplicate_key(tmp_path: pathlib.Path) -> None:
    """PyYAML, which 0.5.2 used, keeps the last duplicate; the import still writes its `.pvt`."""
    home = tmp_path / "home"
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
        "core:\n  max_workers: 1\n  max_workers: 2\nremotes:\n  origin: s3://source-cache\n"
    )
    lock_file = source / ".pivot" / "stages" / "test" / "writes_state.lock"
    lock_file.parent.mkdir(parents=True)
    lock_file.write_text(
        yaml.safe_dump(
            {
                "code_manifest": {},
                "params": {},
                "deps": [],
                "outs": [{"key": None, "tag": "data", "hash": "0123456789abcdef", "size": 5}],
            }
        )
    )
    subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "duplicate key"], cwd=source, check=True, capture_output=True
    )
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    init_git_repo(consumer)
    assert helpers.run_pivot(consumer, home, "init").returncode == 0

    result = helpers.run_pivot(
        consumer,
        home,
        "import",
        str(source),
        "test/writes_state",
        "--out",
        "got.txt",
        "--no-download",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert (consumer / "got.txt.pvt").is_file()
    assert "Traceback" not in output
    warning = f"Ignoring core.state_dir in .pivot/config.yaml of {source}: "
    assert result.stderr.count(warning) == 1, result.stderr


_YAML_1_1_SOURCE_CONFIG = (
    "core:\n  state_dir: on\nremotes:\n  on: s3://source-cache\ndefault_remote: on\n"
)


def test_import_reads_a_source_state_dir_named_like_a_yaml_1_1_scalar(
    tmp_path: pathlib.Path,
) -> None:
    """`config set core.state_dir on` keeps state in `on/`; import reads that source's locks."""
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
    (source / ".pivot" / "config.yaml").write_text(_YAML_1_1_SOURCE_CONFIG)
    lock_file = source / "on" / "stages" / "test" / "writes_state.lock"
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
        ["git", "commit", "-m", "state in on/"], cwd=source, check=True, capture_output=True
    )

    resolved = asyncio.run(
        import_artifact.resolve_remote_path(str(source), "test/writes_state", "main", None)
    )

    assert resolved["stage"] == "test/writes_state"
    assert resolved["remote_url"] == "s3://source-cache"


def test_import_reads_the_same_source_remotes_as_pivot_052(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """Remotes keep 0.5.2's parse: `default_remote: on` is not a name there, so it is ignored."""
    mocker.patch.object(
        import_artifact,
        "_read_remote_file",
        autospec=True,
        return_value=_YAML_1_1_SOURCE_CONFIG.encode(),
    )

    with caplog.at_level(logging.WARNING, logger="pivot.import_artifact"):
        url, source_config = asyncio.run(
            import_artifact.read_remote_config("/source", "main", None)
        )

    assert url == "s3://source-cache"
    assert "default_remote in /source must be a remote name, got True; ignoring it" in caplog.text
    assert import_artifact._remote_state_dir("/source", source_config) == "on"


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
        [helpers.PIVOT, "repro", "--watch", "--serve"],
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
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def test_import_uses_default_state_dir_when_source_core_omits_state_dir(
    tmp_path: pathlib.Path,
) -> None:
    """A source with other core settings keeps its default `.pivot/stages` lock location."""
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
    lock_file = source / ".pivot" / "stages" / "test" / "writes_state.lock"
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
    (source / ".pivot" / "config.yaml").write_text(
        "core:\n  max_workers: 1\nremotes:\n  origin: s3://source-cache\n"
    )
    subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "default source state"],
        cwd=source,
        check=True,
        capture_output=True,
    )

    resolved = asyncio.run(
        import_artifact.resolve_remote_path(str(source), "test/writes_state", "main", None)
    )

    assert resolved["stage"] == "test/writes_state"
