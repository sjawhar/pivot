import pathlib

import pydantic
import pytest

from pivot import config, project
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
