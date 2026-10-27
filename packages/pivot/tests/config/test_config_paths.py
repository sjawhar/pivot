import os
import pathlib
import pickle

import pydantic
import pytest

from pivot import config, config_files, exceptions, project
from pivot import state_dir as state_dir_mod
from pivot.config import io as config_io
from pivot.exceptions import ConfigError


@pytest.fixture
def project_root(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up a project root with .pivot directory."""
    pivot_dir = tmp_path / ".pivot"
    pivot_dir.mkdir()
    monkeypatch.setattr(project, "_project_root_cache", None)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_config(project_root: pathlib.Path, content: str) -> None:
    """Helper to write config file."""
    config_file = project_root / ".pivot" / "config.yaml"
    config_file.write_text(content)
    config_io.clear_config_cache()


def test_ensure_state_dir_does_not_create_a_project_marker(tmp_path: pathlib.Path) -> None:
    """A pipeline constructed outside a project must not turn its parent into a project."""
    result = state_dir_mod.ensure_state_dir(tmp_path)

    assert result == tmp_path / ".pivot"
    assert not result.exists()


# --- cache.dir tests ---


def test_cache_dir_from_config(project_root: pathlib.Path) -> None:
    """Config cache.dir is respected."""
    _write_config(project_root, "cache:\n  dir: custom-cache\n")

    result = config.get_cache_dir()
    assert result == project_root / "custom-cache"


def test_cache_dir_absolute_path(project_root: pathlib.Path) -> None:
    """Absolute paths in config are used as-is."""
    absolute_cache = project_root / "absolute-cache"
    _write_config(project_root, f"cache:\n  dir: {absolute_cache}\n")

    result = config.get_cache_dir()
    assert result == absolute_cache


def test_cache_dir_relative_path(project_root: pathlib.Path) -> None:
    """Relative paths resolved from project root."""
    _write_config(project_root, "cache:\n  dir: data/my-cache\n")

    result = config.get_cache_dir()
    assert result == project_root / "data" / "my-cache"


def test_cache_dir_default_when_no_config(project_root: pathlib.Path) -> None:
    """Default .pivot/cache is used when no config exists."""
    # No config file written

    result = config.get_cache_dir()
    assert result == project_root / ".pivot" / "cache"


def test_cache_dir_default_uses_the_configured_state_dir(project_root: pathlib.Path) -> None:
    """The implicit cache follows core.state_dir."""
    _write_config(project_root, "core:\n  state_dir: alternate-state\n")

    result = config.get_cache_dir()

    assert result == project_root / "alternate-state" / "cache"


# --- PIVOT_CACHE_DIR env var tests ---


def test_cache_dir_env_var_overrides_config(
    project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PIVOT_CACHE_DIR env var takes precedence over config file."""
    _write_config(project_root, "cache:\n  dir: config-cache\n")
    monkeypatch.setenv("PIVOT_CACHE_DIR", "env-cache")

    result = config.get_cache_dir()
    assert result == project_root / "env-cache"


def test_cache_dir_env_var_absolute_path(
    project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absolute path in PIVOT_CACHE_DIR is used as-is."""
    absolute_cache = project_root / "absolute-env-cache"
    monkeypatch.setenv("PIVOT_CACHE_DIR", str(absolute_cache))

    result = config.get_cache_dir()
    assert result == absolute_cache


def test_cache_dir_env_var_relative_path(
    project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative path in PIVOT_CACHE_DIR resolved against project root."""
    monkeypatch.setenv("PIVOT_CACHE_DIR", "data/env-cache")

    result = config.get_cache_dir()
    assert result == project_root / "data" / "env-cache"


def test_cache_dir_env_var_empty_uses_config(
    project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty PIVOT_CACHE_DIR treated as unset, falls back to config."""
    _write_config(project_root, "cache:\n  dir: config-cache\n")
    monkeypatch.setenv("PIVOT_CACHE_DIR", "")

    result = config.get_cache_dir()
    assert result == project_root / "config-cache"


def test_cache_dir_env_var_whitespace_only_uses_config(
    project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whitespace-only PIVOT_CACHE_DIR treated as unset, falls back to config."""
    _write_config(project_root, "cache:\n  dir: config-cache\n")
    monkeypatch.setenv("PIVOT_CACHE_DIR", "   ")

    result = config.get_cache_dir()
    assert result == project_root / "config-cache"


def test_cache_dir_env_var_without_config(
    project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PIVOT_CACHE_DIR works even when no config file exists."""
    monkeypatch.setenv("PIVOT_CACHE_DIR", "env-only-cache")

    result = config.get_cache_dir()
    assert result == project_root / "env-only-cache"


# --- core.state_dir tests ---


def test_state_dir_from_config(project_root: pathlib.Path) -> None:
    """Config core.state_dir is respected."""
    _write_config(project_root, "core:\n  state_dir: custom-state\n")

    result = config.get_state_dir()
    assert result == project_root / "custom-state"


def test_state_dir_absolute_path(project_root: pathlib.Path) -> None:
    """Absolute state_dir paths in config are used as-is."""
    absolute_state = project_root / "absolute-state"
    _write_config(project_root, f"core:\n  state_dir: {absolute_state}\n")

    result = config.get_state_dir()
    assert result == absolute_state


def test_state_dir_default_when_no_config(project_root: pathlib.Path) -> None:
    """Default .pivot is used when no config exists."""
    # No config file written

    result = config.get_state_dir()
    assert result == project_root / ".pivot"


def test_state_dir_for_reads_the_project_configuration(project_root: pathlib.Path) -> None:
    """The project resolver is the source of truth for state location."""
    _write_config(project_root, "core:\n  state_dir: custom-state\n")

    assert state_dir_mod.state_dir_for(project_root) == project_root / "custom-state"


def test_state_dir_lookup_outside_invocation_reads_changed_bytes_with_same_timestamp(
    project_root: pathlib.Path,
) -> None:
    """Outside a state-directory record every lookup reads the file again; neither its size
    nor its mtime is trusted."""
    config_file = project_root / ".pivot" / "config.yaml"
    config_file.write_text("core:\n  state_dir: alpha\n")
    stat = config_file.stat()
    assert state_dir_mod.state_dir_for(project_root) == project_root / "alpha"

    config_file.write_text("core:\n  state_dir: betaa\n")
    os.utime(config_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert config_file.stat().st_size == stat.st_size
    assert config_file.stat().st_mtime_ns == stat.st_mtime_ns
    assert state_dir_mod.state_dir_for(project_root) == project_root / "betaa"


def test_state_dir_lookup_outside_invocation_never_reuses_an_answer_from_other_bytes(
    project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config rewritten while a lookup reads it decides that lookup only.

    The first lookup's parse sees the file rewritten (`betaa`), and the file then returns to
    its first bytes (`alpha`). Nothing outside a record remembers an answer, so the next lookup
    reads `alpha` again; an answer filed under the bytes read before the rewrite would be
    `betaa`.
    """
    config_file = project_root / ".pivot" / "config.yaml"
    first = "core:\n  state_dir: alpha\n"
    config_file.write_text(first)
    read = config_files.read_config_file_value
    rewrites = ["core:\n  state_dir: betaa\n"]

    def _read_while_the_file_is_rewritten(path: pathlib.Path, key: str) -> tuple[bool, object]:
        if not rewrites:
            return read(path, key)
        path.write_text(rewrites.pop())
        try:
            return read(path, key)
        finally:
            path.write_text(first)

    monkeypatch.setattr(config_files, "read_config_file_value", _read_while_the_file_is_rewritten)

    assert state_dir_mod.state_dir_for(project_root) == project_root / "betaa"
    assert config_file.read_text() == first
    assert state_dir_mod.state_dir_for(project_root) == project_root / "alpha"


@pytest.mark.parametrize(
    ("value", "error"),
    [("''", exceptions.EmptyStateDirError), ("5", ConfigError), ("[unclosed", ConfigError)],
    ids=["empty", "integer", "invalid-yaml"],
)
def test_a_record_keeps_an_unusable_value_and_reads_the_config_once(
    project_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    error: type[ConfigError],
) -> None:
    """Inside one record a value that cannot be used is read once: every later lookup raises
    the same error again, and every fallback (`''` to `.pivot`) answers from the record."""
    (project_root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")
    read = config_files.read_config_file_value
    reads = list[pathlib.Path]()

    def _counted(path: pathlib.Path, key: str) -> tuple[bool, object]:
        reads.append(path)
        return read(path, key)

    monkeypatch.setattr(config_files, "read_config_file_value", _counted)
    token = state_dir_mod.begin_invocation(report=False)
    try:
        messages = set[str]()
        for _ in range(3):
            with pytest.raises(ConfigError) as caught:
                state_dir_mod.state_dir_for(project_root)
            assert caught.type is error
            messages.add(str(caught.value))
        default = project_root / ".pivot"
        assert state_dir_mod.state_dir_or_default(project_root) == default
        if error is exceptions.EmptyStateDirError:
            assert state_dir_mod.locate_state_dir(project_root) == default
    finally:
        state_dir_mod.end_invocation(token)

    assert len(messages) == 1, messages
    assert reads == [project_root / ".pivot" / "config.yaml"]


def test_a_record_keeps_the_failure_it_decided_for_a_path_that_cannot_be_a_directory(
    project_root: pathlib.Path,
) -> None:
    """A record decides once, with the value, that a regular file where the state directory
    goes fails a command that needs the configured directory: it gets the path, before and
    after the file goes, and nothing creates it. Its stages' directory stays `.pivot`."""
    (project_root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    (project_root / "s1").write_text("a regular file\n")
    default = project_root / ".pivot"
    token = state_dir_mod.begin_invocation(report=False)
    try:
        resolution = state_dir_mod.resolution_for(project_root)
        assert isinstance(resolution, exceptions.StateDirTargetError)
        assert not resolution.fallback
        assert state_dir_mod.state_dir_or_default(project_root) == default
        assert state_dir_mod.locate_state_dir(project_root) == project_root / "s1"
        (project_root / "s1").unlink()
        assert state_dir_mod.state_dir_or_default(project_root) == default
        assert state_dir_mod.locate_state_dir(project_root) == project_root / "s1"
        assert state_dir_mod.ensure_state_dir(project_root) == project_root / "s1"
        assert state_dir_mod.state_dir_for(project_root) == project_root / "s1"
    finally:
        state_dir_mod.end_invocation(token)

    assert not (project_root / "s1").exists()
    assert state_dir_mod.state_dir_or_default(project_root) == project_root / "s1"


def test_a_record_falls_back_for_a_path_that_still_holds_a_writable_database(
    project_root: pathlib.Path,
) -> None:
    """A state directory without write permission whose `state.lmdb` can still be written
    falls back to `.pivot` for every reader, after it is made writable again too."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")
    (project_root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: s1\n")
    s1 = project_root / "s1"
    (s1 / "state.lmdb").mkdir(parents=True)
    s1.chmod(0o555)
    default = project_root / ".pivot"
    token = state_dir_mod.begin_invocation(report=False)
    try:
        resolution = state_dir_mod.resolution_for(project_root)
        assert isinstance(resolution, exceptions.StateDirTargetError)
        assert resolution.fallback
        assert state_dir_mod.state_dir_or_default(project_root) == default
        assert state_dir_mod.locate_state_dir(project_root) == default
        assert state_dir_mod.ensure_state_dir(project_root) == default
        s1.chmod(0o755)
        assert state_dir_mod.locate_state_dir(project_root) == default
        assert state_dir_mod.state_dir_or_default(project_root) == default
    finally:
        s1.chmod(0o755)
        state_dir_mod.end_invocation(token)


@pytest.mark.parametrize("fallback", [False, True])
def test_a_kept_path_outcome_survives_pickling(fallback: bool) -> None:
    """A worker receives its Pivot process's outcome for an unusable path whole."""
    error = exceptions.StateDirTargetError("why", pathlib.Path("/project/s1"), fallback)
    copied = pickle.loads(pickle.dumps(error))

    assert (type(copied), str(copied), copied.path, copied.fallback) == (
        exceptions.StateDirTargetError,
        "why",
        pathlib.Path("/project/s1"),
        fallback,
    )


def test_a_pipeline_keeps_the_fallback_for_a_path_that_cannot_be_a_directory() -> None:
    """A pipeline's stages record `.pivot` for such a path, so its runs use `.pivot` too."""
    path = pathlib.Path("/project/s1")
    failing = exceptions.StateDirTargetError("why", path, False)

    kept = state_dir_mod.kept_by_pipeline(failing)

    assert isinstance(kept, exceptions.StateDirTargetError)
    assert (str(kept), kept.path, kept.fallback) == ("why", path, True)
    usable = pathlib.Path("/project/state")
    empty = exceptions.EmptyStateDirError("empty")
    assert state_dir_mod.kept_by_pipeline(usable) is usable
    assert state_dir_mod.kept_by_pipeline(empty) is empty


# --- Error handling tests ---


def test_invalid_yaml_raises_config_error(project_root: pathlib.Path) -> None:
    """Malformed YAML raises ConfigError."""
    _write_config(project_root, "cache:\n  dir: [invalid\n")  # Unclosed bracket

    with pytest.raises(ConfigError):
        config.get_cache_dir()


def test_invalid_type_raises_validation_error(project_root: pathlib.Path) -> None:
    """Wrong type for config value raises ValidationError."""
    _write_config(project_root, "cache:\n  dir:\n    - list\n    - not\n    - string\n")

    with pytest.raises(pydantic.ValidationError):
        config.get_cache_dir()
