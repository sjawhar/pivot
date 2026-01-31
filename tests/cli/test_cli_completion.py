from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import click
import pytest

from helpers import register_test_stage
from pivot.cli import completion

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from pivot.pipeline.pipeline import Pipeline


def _noop() -> None:
    """Module-level no-op function for stage registration in tests."""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ctx() -> mock.MagicMock:
    """Create a mock Click context."""
    return mock.MagicMock(spec=click.Context)


@pytest.fixture
def mock_param() -> mock.MagicMock:
    """Create a mock Click parameter."""
    return mock.MagicMock(spec=click.Parameter)


# =============================================================================
# _get_stages_fast tests
# =============================================================================


def test_get_stages_fast_simple_yaml(tmp_path: Path, mocker: MockerFixture) -> None:
    """Fast path extracts names from simple pivot.yaml."""
    yaml_content = """
stages:
  preprocess:
    python: stages.preprocess
  train:
    python: stages.train
"""
    (tmp_path / "pivot.yaml").write_text(yaml_content)
    (tmp_path / ".git").mkdir()

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_fast()
    assert result is not None
    assert set(result) == {"preprocess", "train"}


def test_get_stages_fast_matrix_yaml(tmp_path: Path, mocker: MockerFixture) -> None:
    """Fast path expands matrix configurations."""
    yaml_content = """
stages:
  train:
    python: stages.train
    matrix:
      model: [bert, gpt]
      dataset: [swe, human]
"""
    (tmp_path / "pivot.yaml").write_text(yaml_content)
    (tmp_path / ".git").mkdir()

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_fast()
    assert result is not None
    assert set(result) == {
        "train@bert_swe",
        "train@bert_human",
        "train@gpt_swe",
        "train@gpt_human",
    }


def test_get_stages_fast_mixed_simple_and_matrix(tmp_path: Path, mocker: MockerFixture) -> None:
    """Fast path handles mix of simple and matrix stages."""
    yaml_content = """
stages:
  preprocess:
    python: stages.preprocess
  train:
    python: stages.train
    matrix:
      model: [bert, gpt]
"""
    (tmp_path / "pivot.yaml").write_text(yaml_content)
    (tmp_path / ".git").mkdir()

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_fast()
    assert result is not None
    assert set(result) == {"preprocess", "train@bert", "train@gpt"}


def test_get_stages_fast_returns_none_when_no_project_root(mocker: MockerFixture) -> None:
    """Returns None when no project root found."""
    mocker.patch.object(completion, "_find_project_root_fast", return_value=None)

    result = completion._get_stages_fast()
    assert result is None


def test_get_stages_fast_returns_none_when_no_yaml(tmp_path: Path, mocker: MockerFixture) -> None:
    """Returns None when pivot.yaml doesn't exist."""
    (tmp_path / ".git").mkdir()

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_fast()
    assert result is None


def test_get_stages_fast_returns_none_on_variants(tmp_path: Path, mocker: MockerFixture) -> None:
    """Returns None when config has variants (needs fallback)."""
    yaml_content = """
stages:
  train:
    python: stages.train
    variants: stages.get_variants
"""
    (tmp_path / "pivot.yaml").write_text(yaml_content)
    (tmp_path / ".git").mkdir()

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_fast()
    assert result is None


def test_get_stages_fast_returns_none_on_yaml_error(tmp_path: Path, mocker: MockerFixture) -> None:
    """Returns None on YAML parse error."""
    (tmp_path / "pivot.yaml").write_text("invalid: yaml: [[[")
    (tmp_path / ".git").mkdir()

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_fast()
    assert result is None


def test_get_stages_fast_pivot_yml_alternative(tmp_path: Path, mocker: MockerFixture) -> None:
    """Fast path works with pivot.yml (alternative extension)."""
    yaml_content = """
stages:
  test:
    python: stages.test
"""
    (tmp_path / "pivot.yml").write_text(yaml_content)
    (tmp_path / ".git").mkdir()

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_fast()
    assert result is not None
    assert result == ["test"]


# =============================================================================
# _find_project_root_fast tests
# =============================================================================


def test_find_project_root_fast_finds_git_marker(tmp_path: Path, mocker: MockerFixture) -> None:
    """Finds project root via .git marker."""
    (tmp_path / ".git").mkdir()
    mocker.patch("pathlib.Path.cwd", return_value=tmp_path)

    result = completion._find_project_root_fast()
    assert result == tmp_path


def test_find_project_root_fast_finds_pivot_marker(tmp_path: Path, mocker: MockerFixture) -> None:
    """Finds project root via .pivot marker."""
    (tmp_path / ".pivot").mkdir()
    mocker.patch("pathlib.Path.cwd", return_value=tmp_path)

    result = completion._find_project_root_fast()
    assert result == tmp_path


def test_find_project_root_fast_walks_up(tmp_path: Path, mocker: MockerFixture) -> None:
    """Walks up directory tree to find marker."""
    (tmp_path / ".git").mkdir()
    subdir = tmp_path / "src" / "pkg"
    subdir.mkdir(parents=True)
    mocker.patch("pathlib.Path.cwd", return_value=subdir)

    result = completion._find_project_root_fast()
    assert result == tmp_path


def test_find_project_root_fast_returns_none_when_no_marker(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Returns None when no marker found."""
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    mocker.patch("pathlib.Path.cwd", return_value=isolated)

    result = completion._find_project_root_fast()
    assert result is None or hasattr(result, "exists")


# =============================================================================
# _get_stages_full tests
# =============================================================================


def test_get_stages_full_returns_registered_stages(mock_discovery: Pipeline) -> None:
    """Returns stages from registry after registration."""
    _ = mock_discovery

    # Register real stages (autouse fixture clears registry between tests)
    register_test_stage(_noop, name="stage1")
    register_test_stage(_noop, name="stage2")

    result = completion._get_stages_full()

    assert set(result) == {"stage1", "stage2"}


# =============================================================================
# complete_stages tests - use real YAML files instead of mocking
# =============================================================================


def test_complete_stages_filters_by_prefix(
    tmp_path: Path,
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
    mocker: MockerFixture,
) -> None:
    """Filters stage names by incomplete prefix."""
    yaml_content = """
stages:
  train:
    python: stages.train
  test:
    python: stages.test
  preprocess:
    python: stages.preprocess
"""
    (tmp_path / "pivot.yaml").write_text(yaml_content)
    (tmp_path / ".git").mkdir()
    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion.complete_stages(mock_ctx, mock_param, "tr")
    assert result == ["train"]


def test_complete_stages_empty_prefix_returns_all(
    tmp_path: Path,
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
    mocker: MockerFixture,
) -> None:
    """Empty prefix returns all stages."""
    yaml_content = """
stages:
  train:
    python: stages.train
  test:
    python: stages.test
"""
    (tmp_path / "pivot.yaml").write_text(yaml_content)
    (tmp_path / ".git").mkdir()
    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion.complete_stages(mock_ctx, mock_param, "")
    assert set(result) == {"train", "test"}


def test_complete_stages_falls_back_to_registry(
    mock_discovery: Pipeline,
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
    mocker: MockerFixture,
) -> None:
    """Falls back to registry when fast path returns None (no YAML)."""
    _ = mock_discovery

    # No YAML file, fast path returns None
    mocker.patch.object(completion, "_find_project_root_fast", return_value=None)

    # Register stages directly
    register_test_stage(_noop, name="fallback_stage")

    result = completion.complete_stages(mock_ctx, mock_param, "")
    assert "fallback_stage" in result


def test_complete_stages_returns_empty_on_exception(
    mock_ctx: mock.MagicMock, mock_param: mock.MagicMock, mocker: MockerFixture
) -> None:
    """Returns empty list if exception occurs."""
    mocker.patch.object(completion, "_get_stages_fast", side_effect=Exception("boom"))

    result = completion.complete_stages(mock_ctx, mock_param, "")
    assert result == []


def test_complete_stages_matrix_stage_completion(
    tmp_path: Path,
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
    mocker: MockerFixture,
) -> None:
    """Completes matrix stage names correctly."""
    yaml_content = """
stages:
  train:
    python: stages.train
    matrix:
      model: [bert, gpt]
      dataset: [swe]
  preprocess:
    python: stages.preprocess
"""
    (tmp_path / "pivot.yaml").write_text(yaml_content)
    (tmp_path / ".git").mkdir()
    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion.complete_stages(mock_ctx, mock_param, "train@b")
    assert result == ["train@bert_swe"]


def test_complete_stages_case_sensitive(
    tmp_path: Path,
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
    mocker: MockerFixture,
) -> None:
    """Completion is case-sensitive."""
    yaml_content = """
stages:
  Train:
    python: stages.Train
  train:
    python: stages.train
  TRAIN:
    python: stages.TRAIN
"""
    (tmp_path / "pivot.yaml").write_text(yaml_content)
    (tmp_path / ".git").mkdir()
    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion.complete_stages(mock_ctx, mock_param, "tr")
    assert result == ["train"]


# =============================================================================
# complete_targets tests
# =============================================================================


def test_complete_targets_includes_stage_names(
    tmp_path: Path,
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
    mocker: MockerFixture,
) -> None:
    """Target completion includes stage names."""
    yaml_content = """
stages:
  train:
    python: stages.train
  test:
    python: stages.test
"""
    (tmp_path / "pivot.yaml").write_text(yaml_content)
    (tmp_path / ".git").mkdir()
    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion.complete_targets(mock_ctx, mock_param, "tr")
    assert "train" in result


def test_complete_targets_filters_by_prefix(
    tmp_path: Path,
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
    mocker: MockerFixture,
) -> None:
    """Filters targets by incomplete prefix."""
    yaml_content = """
stages:
  train:
    python: stages.train
  test:
    python: stages.test
  deploy:
    python: stages.deploy
"""
    (tmp_path / "pivot.yaml").write_text(yaml_content)
    (tmp_path / ".git").mkdir()
    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion.complete_targets(mock_ctx, mock_param, "t")
    assert set(result) == {"train", "test"}


# =============================================================================
# complete_config_keys tests
# =============================================================================


def test_complete_config_keys_returns_static_keys(
    mock_ctx: click.Context,
    mock_param: click.Parameter,
) -> None:
    """Returns static config keys with descriptions."""
    result = completion.complete_config_keys(mock_ctx, mock_param, "")
    values = [item.value for item in result]
    assert "cache.dir" in values
    assert "core.max_workers" in values
    assert "display.precision" in values
    # Verify descriptions are present
    cache_dir = next(item for item in result if item.value == "cache.dir")
    assert cache_dir.help is not None


def test_complete_config_keys_filters_by_prefix(
    mock_ctx: click.Context,
    mock_param: click.Parameter,
) -> None:
    """Filters keys by incomplete prefix."""
    result = completion.complete_config_keys(mock_ctx, mock_param, "cache")
    values = [item.value for item in result]
    assert "cache.dir" in values
    assert "cache.checkout_mode" in values
    assert "core.max_workers" not in values


def test_complete_config_keys_includes_local_remotes(
    tmp_path: Path,
    mock_ctx: click.Context,
    mock_param: click.Parameter,
    mocker: MockerFixture,
) -> None:
    """Includes existing remote names from local config."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("remotes:\n  origin: s3://bucket\n")
    mocker.patch.object(
        completion,
        "_get_config_paths",
        return_value=[tmp_path / "nonexistent.yaml", config_path],
    )

    result = completion.complete_config_keys(mock_ctx, mock_param, "remotes")
    values = [item.value for item in result]
    assert "remotes.origin" in values


def test_complete_config_keys_includes_global_remotes(
    tmp_path: Path,
    mock_ctx: click.Context,
    mock_param: click.Parameter,
    mocker: MockerFixture,
) -> None:
    """Includes existing remote names from global config."""
    global_config = tmp_path / "global.yaml"
    global_config.write_text("remotes:\n  backup: s3://backup-bucket\n")
    mocker.patch.object(
        completion,
        "_get_config_paths",
        return_value=[global_config],
    )

    result = completion.complete_config_keys(mock_ctx, mock_param, "remotes")
    values = [item.value for item in result]
    assert "remotes.backup" in values


def test_complete_config_keys_merges_both_configs(
    tmp_path: Path,
    mock_ctx: click.Context,
    mock_param: click.Parameter,
    mocker: MockerFixture,
) -> None:
    """Merges remotes from both global and local configs."""
    global_config = tmp_path / "global.yaml"
    global_config.write_text("remotes:\n  backup: s3://backup\n")
    local_config = tmp_path / "local.yaml"
    local_config.write_text("remotes:\n  origin: s3://origin\n")
    mocker.patch.object(
        completion,
        "_get_config_paths",
        return_value=[global_config, local_config],
    )

    result = completion.complete_config_keys(mock_ctx, mock_param, "remotes")
    values = [item.value for item in result]
    assert "remotes.backup" in values
    assert "remotes.origin" in values


def test_complete_config_keys_filters_invalid_remote_names(
    tmp_path: Path,
    mock_ctx: click.Context,
    mock_param: click.Parameter,
    mocker: MockerFixture,
) -> None:
    """Filters out remote names with invalid characters."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "remotes:\n  valid-name: s3://a\n  'has.dot': s3://b\n  'has space': s3://c\n"
    )
    mocker.patch.object(
        completion,
        "_get_config_paths",
        return_value=[config_path],
    )

    result = completion.complete_config_keys(mock_ctx, mock_param, "remotes")
    values = [item.value for item in result]
    assert "remotes.valid-name" in values
    assert "remotes.has.dot" not in values
    assert "remotes.has space" not in values


def test_complete_config_keys_returns_empty_on_exception(
    mock_ctx: click.Context,
    mock_param: click.Parameter,
    mocker: MockerFixture,
) -> None:
    """Returns empty list if exception occurs."""
    mocker.patch.object(completion, "_get_config_keys", side_effect=Exception("boom"))
    result = completion.complete_config_keys(mock_ctx, mock_param, "")
    assert result == []


def test_complete_config_keys_handles_malformed_yaml(
    tmp_path: Path,
    mock_ctx: click.Context,
    mock_param: click.Parameter,
    mocker: MockerFixture,
) -> None:
    """Handles malformed YAML gracefully, returns static keys."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("invalid: yaml: [[[")
    mocker.patch.object(
        completion,
        "_get_config_paths",
        return_value=[config_path],
    )

    result = completion.complete_config_keys(mock_ctx, mock_param, "cache")
    values = [item.value for item in result]
    # Should still return static keys
    assert "cache.dir" in values


def test_complete_config_keys_handles_remotes_not_dict(
    tmp_path: Path,
    mock_ctx: click.Context,
    mock_param: click.Parameter,
    mocker: MockerFixture,
) -> None:
    """Handles remotes: null or remotes: 'string' gracefully."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("remotes: null\n")
    mocker.patch.object(
        completion,
        "_get_config_paths",
        return_value=[config_path],
    )

    result = completion.complete_config_keys(mock_ctx, mock_param, "")
    # Should not crash, returns static keys
    values = [item.value for item in result]
    assert "cache.dir" in values
