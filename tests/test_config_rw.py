import pathlib
from typing import Any

import pytest

from pivot import project
from pivot.config import io

# --- parse_dotted_key tests ---


def test_parse_dotted_key_simple() -> None:
    assert io.parse_dotted_key("cache.dir") == ["cache", "dir"]


def test_parse_dotted_key_nested() -> None:
    assert io.parse_dotted_key("cache.checkout_mode") == ["cache", "checkout_mode"]


def test_parse_dotted_key_top_level() -> None:
    assert io.parse_dotted_key("default_remote") == ["default_remote"]


def test_parse_dotted_key_remotes() -> None:
    assert io.parse_dotted_key("remotes.origin") == ["remotes", "origin"]


# --- get_nested / set_nested / unset_nested tests ---


def test_get_nested_returns_value() -> None:
    data = {"cache": {"dir": "/tmp/cache"}}
    result = io.get_nested(data, ["cache", "dir"])
    assert result == "/tmp/cache"


def test_get_nested_returns_sentinel_for_missing_key() -> None:
    data = {"cache": {"dir": "/tmp/cache"}}
    result = io.get_nested(data, ["cache", "nonexistent"])
    assert result is io._NOT_FOUND


def test_get_nested_returns_sentinel_for_missing_parent() -> None:
    data = {"cache": {"dir": "/tmp/cache"}}
    result = io.get_nested(data, ["nonexistent", "key"])
    assert result is io._NOT_FOUND


def test_get_nested_top_level() -> None:
    data = {"default_remote": "origin"}
    result = io.get_nested(data, ["default_remote"])
    assert result == "origin"


def test_set_nested_creates_value() -> None:
    data: dict[str, object] = {}
    io.set_nested(data, ["cache", "dir"], "/tmp/cache")
    assert data == {"cache": {"dir": "/tmp/cache"}}


def test_set_nested_creates_intermediate_dicts() -> None:
    data: dict[str, object] = {}
    io.set_nested(data, ["a", "b", "c"], "value")
    assert data == {"a": {"b": {"c": "value"}}}


def test_set_nested_overwrites_existing() -> None:
    data: dict[str, Any] = {"cache": {"dir": "/old"}}
    io.set_nested(data, ["cache", "dir"], "/new")
    assert data["cache"]["dir"] == "/new"


def test_set_nested_top_level() -> None:
    data: dict[str, object] = {}
    io.set_nested(data, ["default_remote"], "origin")
    assert data == {"default_remote": "origin"}


def test_unset_nested_removes_key() -> None:
    data: dict[str, object] = {"cache": {"dir": "/tmp", "mode": "hardlink"}}
    removed = io.unset_nested(data, ["cache", "dir"])
    assert removed is True
    assert data == {"cache": {"mode": "hardlink"}}


def test_unset_nested_removes_empty_parents() -> None:
    data: dict[str, object] = {"cache": {"dir": "/tmp"}}
    removed = io.unset_nested(data, ["cache", "dir"])
    assert removed is True
    assert data == {}


def test_unset_nested_returns_false_for_missing() -> None:
    data: dict[str, object] = {"cache": {"dir": "/tmp"}}
    removed = io.unset_nested(data, ["cache", "nonexistent"])
    assert removed is False


def test_unset_nested_top_level() -> None:
    data: dict[str, object] = {"default_remote": "origin", "other": "value"}
    removed = io.unset_nested(data, ["default_remote"])
    assert removed is True
    assert data == {"other": "value"}


# --- load_config_file tests ---


def test_load_config_file_missing_returns_empty(tmp_path: pathlib.Path) -> None:
    result = io.load_config_file(tmp_path / "nonexistent.yaml")
    assert result == {}


def test_load_config_file_parses_yaml(tmp_path: pathlib.Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("cache:\n  dir: /tmp/cache\n")

    result = io.load_config_file(config_file)

    assert result == {"cache": {"dir": "/tmp/cache"}}


def test_load_config_file_empty_returns_empty(tmp_path: pathlib.Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")

    result = io.load_config_file(config_file)

    assert result == {}


# --- edit_config tests (replaces save_config_file tests) ---


def test_edit_config_creates_file(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()

    with io.edit_config(models.ConfigScope.LOCAL) as data:
        data["cache"] = {"dir": "/tmp"}

    config_file = tmp_path / ".pivot" / "config.yaml"
    assert config_file.exists()
    content = config_file.read_text()
    assert "cache:" in content
    assert "dir:" in content


def test_edit_config_creates_parent_dirs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    monkeypatch.setattr(
        io, "get_local_config_path", lambda: tmp_path / "nested" / "dir" / "config.yaml"
    )

    with io.edit_config(models.ConfigScope.LOCAL) as data:
        data["key"] = "value"

    config_file = tmp_path / "nested" / "dir" / "config.yaml"
    assert config_file.exists()


def test_edit_config_preserves_comments(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()

    config_file = tmp_path / ".pivot" / "config.yaml"
    config_file.write_text("# Important comment\ncache:\n  dir: /old\n")

    with io.edit_config(models.ConfigScope.LOCAL) as data:
        data["cache"]["dir"] = "/new"

    content = config_file.read_text()
    assert "# Important comment" in content
    assert "/new" in content


# --- get_merged_config tests ---


def test_get_merged_config_returns_defaults_when_no_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(io, "get_global_config_path", lambda: tmp_path / "global.yaml")

    result = io.get_merged_config()

    assert result.cache.dir == ".pivot/cache"
    assert result.core.max_workers == -2


def test_get_merged_config_local_overrides_global(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()

    global_file = tmp_path / "global.yaml"
    global_file.write_text("core:\n  max_workers: 4\n")
    monkeypatch.setattr(io, "get_global_config_path", lambda: global_file)

    local_file = tmp_path / ".pivot" / "config.yaml"
    local_file.write_text("core:\n  max_workers: 8\n")

    result = io.get_merged_config()

    assert result.core.max_workers == 8


def test_get_merged_config_global_overrides_defaults(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()

    global_file = tmp_path / "global.yaml"
    global_file.write_text("display:\n  precision: 3\n")
    monkeypatch.setattr(io, "get_global_config_path", lambda: global_file)

    result = io.get_merged_config()

    assert result.display.precision == 3


def test_get_merged_config_merges_sections(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()

    global_file = tmp_path / "global.yaml"
    global_file.write_text("cache:\n  dir: /global/cache\n")
    monkeypatch.setattr(io, "get_global_config_path", lambda: global_file)

    local_file = tmp_path / ".pivot" / "config.yaml"
    local_file.write_text("core:\n  max_workers: 4\n")

    result = io.get_merged_config()

    assert result.cache.dir == "/global/cache"
    assert result.core.max_workers == 4


# --- get_config_value tests ---


def test_get_config_value_returns_local_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(io, "get_global_config_path", lambda: tmp_path / "global.yaml")

    local_file = tmp_path / ".pivot" / "config.yaml"
    local_file.write_text("cache:\n  dir: /local/cache\n")

    value, source = io.get_config_value("cache.dir")

    assert value == "/local/cache"
    assert source == models.ConfigSource.LOCAL


def test_get_config_value_returns_global_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()

    global_file = tmp_path / "global.yaml"
    global_file.write_text("display:\n  precision: 3\n")
    monkeypatch.setattr(io, "get_global_config_path", lambda: global_file)

    value, source = io.get_config_value("display.precision")

    assert value == 3
    assert source == models.ConfigSource.GLOBAL


def test_get_config_value_returns_default_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(io, "get_global_config_path", lambda: tmp_path / "global.yaml")

    value, source = io.get_config_value("core.max_workers")

    assert value == -2
    assert source == models.ConfigSource.DEFAULT


def test_get_config_value_returns_none_for_unknown(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(io, "get_global_config_path", lambda: tmp_path / "global.yaml")

    value, source = io.get_config_value("unknown.key")

    assert value is None
    assert source == models.ConfigSource.UNKNOWN


# --- set_config_value tests ---


def test_set_config_value_creates_local_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(io, "get_global_config_path", lambda: tmp_path / "global.yaml")

    io.set_config_value("cache.dir", "/custom/cache", scope=models.ConfigScope.LOCAL)

    local_file = tmp_path / ".pivot" / "config.yaml"
    assert local_file.exists()
    content = local_file.read_text()
    assert "/custom/cache" in content


def test_set_config_value_creates_global_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()

    global_file = tmp_path / ".config" / "pivot" / "config.yaml"
    monkeypatch.setattr(io, "get_global_config_path", lambda: global_file)

    io.set_config_value("display.precision", 3, scope=models.ConfigScope.GLOBAL)

    assert global_file.exists()


def test_set_config_value_updates_existing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(io, "get_global_config_path", lambda: tmp_path / "global.yaml")

    local_file = tmp_path / ".pivot" / "config.yaml"
    local_file.write_text("cache:\n  dir: /old\n")

    io.set_config_value("cache.dir", "/new", scope=models.ConfigScope.LOCAL)

    content = local_file.read_text()
    assert "/new" in content
    assert "/old" not in content


def test_set_config_value_sets_remotes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(io, "get_global_config_path", lambda: tmp_path / "global.yaml")

    io.set_config_value("remotes.origin", "s3://my-bucket/cache", scope=models.ConfigScope.LOCAL)

    local_file = tmp_path / ".pivot" / "config.yaml"
    content = local_file.read_text()
    assert "origin" in content
    assert "s3://my-bucket/cache" in content


# --- unset_config_value tests ---


def test_unset_config_value_removes_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(io, "get_global_config_path", lambda: tmp_path / "global.yaml")

    local_file = tmp_path / ".pivot" / "config.yaml"
    local_file.write_text("cache:\n  dir: /custom\n")

    removed = io.unset_config_value("cache.dir", scope=models.ConfigScope.LOCAL)

    assert removed is True
    content = local_file.read_text()
    assert "/custom" not in content


def test_unset_config_value_returns_false_for_missing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(io, "get_global_config_path", lambda: tmp_path / "global.yaml")

    removed = io.unset_config_value("cache.dir", scope=models.ConfigScope.LOCAL)

    assert removed is False


def test_unset_config_value_cleans_empty_sections(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot.config import models

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    monkeypatch.setattr(io, "get_global_config_path", lambda: tmp_path / "global.yaml")

    local_file = tmp_path / ".pivot" / "config.yaml"
    local_file.write_text("cache:\n  dir: /custom\nother: value\n")

    io.unset_config_value("cache.dir", scope=models.ConfigScope.LOCAL)

    content = local_file.read_text()
    assert "cache:" not in content
    assert "other: value" in content


# --- deep_merge tests ---


def test_deep_merge_combines_dicts() -> None:
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"d": 3}, "e": 4}

    result = io.deep_merge(base, override)

    assert result == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}


def test_deep_merge_override_wins() -> None:
    base = {"a": 1}
    override = {"a": 2}

    result = io.deep_merge(base, override)

    assert result == {"a": 2}


def test_deep_merge_does_not_modify_base() -> None:
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}

    io.deep_merge(base, override)

    assert base == {"a": {"b": 1}}
