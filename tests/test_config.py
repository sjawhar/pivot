import pathlib

import pytest

from pivot import config

# --- load_config tests ---


def test_load_config_missing_file_returns_empty(tmp_path: pathlib.Path) -> None:
    result = config.load_config(tmp_path / "nonexistent.yaml")
    assert result == config.PivotConfig()


def test_load_config_parses_checkout_mode(tmp_path: pathlib.Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("cache:\n  checkout_mode:\n    - symlink\n    - copy\n")

    result = config.load_config(config_file)

    assert "cache" in result
    assert "checkout_mode" in result["cache"]
    assert result["cache"]["checkout_mode"] == ["symlink", "copy"]


def test_load_config_empty_file_returns_empty(tmp_path: pathlib.Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")

    result = config.load_config(config_file)

    assert result == config.PivotConfig()


def test_load_config_invalid_yaml_returns_empty(tmp_path: pathlib.Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [")

    result = config.load_config(config_file)

    assert result == config.PivotConfig()


def test_load_config_non_dict_returns_empty(tmp_path: pathlib.Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- just a list")

    result = config.load_config(config_file)

    assert result == config.PivotConfig()


def test_load_config_cache_non_dict_ignored(tmp_path: pathlib.Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("cache: not_a_dict")

    result = config.load_config(config_file)

    assert "cache" not in result


def test_load_config_checkout_mode_non_list_ignored(tmp_path: pathlib.Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("cache:\n  checkout_mode: symlink\n")

    result = config.load_config(config_file)

    assert "cache" not in result


def test_load_config_cached_per_process(tmp_path: pathlib.Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("cache:\n  checkout_mode:\n    - hardlink\n")

    first = config.load_config(config_file)

    # Modify file - should not affect cached result
    config_file.write_text("cache:\n  checkout_mode:\n    - copy\n")

    second = config.load_config(config_file)

    assert first is second, "Should return same cached instance"
    assert "cache" in first
    assert "checkout_mode" in first["cache"]
    assert first["cache"]["checkout_mode"] == ["hardlink"]


def test_load_config_uses_project_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot import project

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    config_file = tmp_path / ".pivot" / "config.yaml"
    config_file.write_text("cache:\n  checkout_mode:\n    - copy\n")

    result = config.load_config()

    assert "cache" in result
    assert "checkout_mode" in result["cache"]
    assert result["cache"]["checkout_mode"] == ["copy"]


# --- get_checkout_mode_order tests ---


def test_get_checkout_mode_order_default() -> None:
    empty_config = config.PivotConfig()

    result = config.get_checkout_mode_order(empty_config)

    assert result == ["hardlink", "symlink", "copy"]


def test_get_checkout_mode_order_from_config() -> None:
    test_config = config.PivotConfig(
        cache=config.CacheConfig(checkout_mode=["symlink", "hardlink"])
    )

    result = config.get_checkout_mode_order(test_config)

    assert result == ["symlink", "hardlink"]


def test_get_checkout_mode_order_invalid_mode_skipped() -> None:
    test_config = config.PivotConfig(
        cache=config.CacheConfig(checkout_mode=["hardlink", "invalid_mode", "copy"])
    )

    result = config.get_checkout_mode_order(test_config)

    assert result == ["hardlink", "copy"], "Invalid mode should be skipped"


def test_get_checkout_mode_order_all_invalid_returns_default() -> None:
    test_config = config.PivotConfig(
        cache=config.CacheConfig(checkout_mode=["invalid1", "invalid2"])
    )

    result = config.get_checkout_mode_order(test_config)

    assert result == ["hardlink", "symlink", "copy"], "Should fallback to defaults"


def test_get_checkout_mode_order_empty_list_returns_default() -> None:
    test_config = config.PivotConfig(cache=config.CacheConfig(checkout_mode=[]))

    result = config.get_checkout_mode_order(test_config)

    assert result == ["hardlink", "symlink", "copy"]


def test_get_checkout_mode_order_loads_config_when_none(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pivot import project

    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    (tmp_path / ".pivot").mkdir()
    config_file = tmp_path / ".pivot" / "config.yaml"
    config_file.write_text("cache:\n  checkout_mode:\n    - copy\n    - symlink\n")

    result = config.get_checkout_mode_order()

    assert result == ["copy", "symlink"]


def test_get_checkout_mode_order_returns_copy() -> None:
    test_config = config.PivotConfig(cache=config.CacheConfig(checkout_mode=["hardlink"]))

    result1 = config.get_checkout_mode_order(test_config)
    result2 = config.get_checkout_mode_order(test_config)

    assert result1 == result2
    assert result1 is not result2, "Should return a copy, not the same list"


# --- Config cache behavior tests ---


def test_config_cache_can_be_invalidated(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify config caching and invalidation via monkeypatch."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("cache:\n  checkout_mode:\n    - hardlink\n")
    first = config.load_config(config_file)

    monkeypatch.setattr(config, "_config_cache", None)
    config_file.write_text("cache:\n  checkout_mode:\n    - copy\n")

    second = config.load_config(config_file)

    assert "cache" in first
    assert "checkout_mode" in first["cache"]
    assert first["cache"]["checkout_mode"] == ["hardlink"]
    assert "cache" in second
    assert "checkout_mode" in second["cache"]
    assert second["cache"]["checkout_mode"] == ["copy"]
    assert first is not second
