from __future__ import annotations

import pathlib
import time
from typing import TYPE_CHECKING
from unittest import mock

import click
import pytest

from helpers import register_test_stage
from pivot import discovery
from pivot.cli import completion
from pivot.pipeline import pipeline as pipeline_mod

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

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_fast()
    assert result is None


def test_get_stages_fast_returns_none_on_yaml_error(tmp_path: Path, mocker: MockerFixture) -> None:
    """Returns None on YAML parse error."""
    (tmp_path / "pivot.yaml").write_text("invalid: yaml: [[[")

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
    (tmp_path / ".pivot").mkdir()
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
    assert result is None


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
    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion.complete_stages(mock_ctx, mock_param, "tr")
    assert result == ["train"]


# =============================================================================
# complete_targets tests
# =============================================================================


def test_complete_targets_is_alias_for_complete_stages() -> None:
    """complete_targets is a direct alias for complete_stages."""
    assert completion.complete_targets is completion.complete_stages


# =============================================================================
# complete_config_keys tests
# =============================================================================


def test_complete_config_keys_returns_static_keys(
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
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
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
) -> None:
    """Filters keys by incomplete prefix."""
    result = completion.complete_config_keys(mock_ctx, mock_param, "cache")
    values = [item.value for item in result]
    assert "cache.dir" in values
    assert "cache.checkout_mode" in values
    assert "core.max_workers" not in values


def test_complete_config_keys_includes_local_remotes(
    tmp_path: Path,
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
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
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
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
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
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
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
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
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
    mocker: MockerFixture,
) -> None:
    """Returns empty list if exception occurs."""
    mocker.patch.object(completion, "_get_config_keys", side_effect=Exception("boom"))
    result = completion.complete_config_keys(mock_ctx, mock_param, "")
    assert result == []


def test_complete_config_keys_handles_malformed_yaml(
    tmp_path: Path,
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
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
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
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


# =============================================================================
# _detect_config_file tests
# =============================================================================


def test_detect_config_file_finds_pivot_yaml(tmp_path: Path) -> None:
    """Detects pivot.yaml and returns its mtime."""
    config = tmp_path / "pivot.yaml"
    config.write_text("stages: {}")

    result = completion._detect_config_file(tmp_path)

    assert result is not None
    name, mtime = result
    assert name == "pivot.yaml"
    assert mtime == config.stat().st_mtime


def test_detect_config_file_finds_pivot_yml(tmp_path: Path) -> None:
    """Detects pivot.yml when pivot.yaml doesn't exist."""
    config = tmp_path / "pivot.yml"
    config.write_text("stages: {}")

    result = completion._detect_config_file(tmp_path)

    assert result is not None
    name, mtime = result
    assert name == "pivot.yml"
    assert mtime == config.stat().st_mtime


def test_detect_config_file_finds_pipeline_py(tmp_path: Path) -> None:
    """Detects pipeline.py when no YAML exists."""
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")

    result = completion._detect_config_file(tmp_path)

    assert result is not None
    name, mtime = result
    assert name == "pipeline.py"
    assert mtime == config.stat().st_mtime


def test_detect_config_file_prefers_yaml_over_pipeline(tmp_path: Path) -> None:
    """pivot.yaml takes precedence over pipeline.py."""
    (tmp_path / "pivot.yaml").write_text("stages: {}")
    (tmp_path / "pipeline.py").write_text("# pipeline")

    result = completion._detect_config_file(tmp_path)

    assert result is not None
    name, _ = result
    assert name == "pivot.yaml"


def test_detect_config_file_returns_none_when_nothing_found(tmp_path: Path) -> None:
    """Returns None when no config file exists."""
    result = completion._detect_config_file(tmp_path)
    assert result is None


# =============================================================================
# _get_stages_from_cache tests
# =============================================================================


def test_get_stages_from_cache_reads_valid_cache(tmp_path: Path) -> None:
    """Reads stage names from a valid cache file."""
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")
    mtime = config.stat().st_mtime

    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "stages.cache").write_text(f"v1\npipeline.py:{mtime}\ntrain\ntest\nevaluate\n")

    result = completion._get_stages_from_cache(tmp_path)

    assert result == ["train", "test", "evaluate"]


def test_get_stages_from_cache_returns_none_when_no_cache(tmp_path: Path) -> None:
    """Returns None when cache file doesn't exist."""
    result = completion._get_stages_from_cache(tmp_path)
    assert result is None


def test_get_stages_from_cache_returns_none_on_version_mismatch(tmp_path: Path) -> None:
    """Returns None when cache version doesn't match."""
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")
    mtime = config.stat().st_mtime

    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "stages.cache").write_text(f"v2\npipeline.py:{mtime}\ntrain\n")

    result = completion._get_stages_from_cache(tmp_path)
    assert result is None


def test_get_stages_from_cache_returns_none_on_mtime_mismatch(tmp_path: Path) -> None:
    """Returns None when config file mtime doesn't match."""
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")

    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    # Use wrong mtime
    (cache_dir / "stages.cache").write_text("v1\npipeline.py:0.0\ntrain\n")

    result = completion._get_stages_from_cache(tmp_path)
    assert result is None


def test_get_stages_from_cache_returns_none_when_config_deleted(tmp_path: Path) -> None:
    """Returns None when config file no longer exists."""
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "stages.cache").write_text("v1\npipeline.py:1234567890.0\ntrain\n")

    result = completion._get_stages_from_cache(tmp_path)
    assert result is None


def test_get_stages_from_cache_returns_none_on_malformed_header(tmp_path: Path) -> None:
    """Returns None when header line is malformed (no colon)."""
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")

    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "stages.cache").write_text("v1\nmalformed_no_colon\ntrain\n")

    result = completion._get_stages_from_cache(tmp_path)
    assert result is None


def test_get_stages_from_cache_returns_none_on_too_few_lines(tmp_path: Path) -> None:
    """Returns None when cache has fewer than 2 lines."""
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "stages.cache").write_text("v1\n")

    result = completion._get_stages_from_cache(tmp_path)
    assert result is None


def test_get_stages_from_cache_handles_empty_stages(tmp_path: Path) -> None:
    """Handles cache with no stages (empty pipeline) written by _write_stages_cache."""
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")
    mtime = config.stat().st_mtime

    # Use the actual writer to create the cache (tests roundtrip correctness)
    completion._write_stages_cache(tmp_path, "pipeline.py", mtime, [])

    result = completion._get_stages_from_cache(tmp_path)

    # Empty list of stages - writer produces trailing blank line, reader filters it
    assert result == []


# =============================================================================
# _write_stages_cache tests
# =============================================================================


def test_write_stages_cache_creates_cache_file(tmp_path: Path) -> None:
    """Creates cache file with correct format."""
    (tmp_path / ".pivot").mkdir()

    completion._write_stages_cache(tmp_path, "pipeline.py", 1234567890.5, ["train", "test"])

    cache_path = tmp_path / ".pivot" / "cache" / "stages.cache"
    assert cache_path.exists()
    content = cache_path.read_text()
    lines = content.splitlines()
    assert lines[0] == "v1"
    assert lines[1] == "pipeline.py:1234567890.5"
    assert lines[2] == "train"
    assert lines[3] == "test"


def test_write_stages_cache_creates_cache_dir(tmp_path: Path) -> None:
    """Creates .pivot/cache directory if it doesn't exist."""
    # Neither .pivot nor .pivot/cache exist
    completion._write_stages_cache(tmp_path, "pipeline.py", 1234567890.0, ["stage1"])

    assert (tmp_path / ".pivot" / "cache" / "stages.cache").exists()


def test_write_stages_cache_overwrites_existing(tmp_path: Path) -> None:
    """Overwrites existing cache file."""
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "stages.cache"
    cache_path.write_text("old content")

    completion._write_stages_cache(tmp_path, "pipeline.py", 999.0, ["new_stage"])

    content = cache_path.read_text()
    assert "new_stage" in content
    assert "old content" not in content


def test_write_stages_cache_handles_empty_stages(tmp_path: Path) -> None:
    """Handles empty stages list."""
    (tmp_path / ".pivot").mkdir()

    completion._write_stages_cache(tmp_path, "pipeline.py", 1234567890.0, [])

    cache_path = tmp_path / ".pivot" / "cache" / "stages.cache"
    content = cache_path.read_text()
    # Format: "v1\npipeline.py:mtime\n\n" - the join of empty list is ""
    # So content ends with "\n\n" which gives 3 lines when split
    assert content == "v1\npipeline.py:1234567890.0\n\n"


# =============================================================================
# Stage cache integration tests
# =============================================================================


def test_get_stages_fast_returns_cached_stages(tmp_path: Path, mocker: MockerFixture) -> None:
    """Fast path returns cached stages when cache is valid."""
    # Create pipeline.py and its cache
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")
    mtime = config.stat().st_mtime

    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "stages.cache").write_text(f"v1\npipeline.py:{mtime}\ncached_stage\n")

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_fast()

    assert result == ["cached_stage"]


def test_get_stages_fast_ignores_stale_cache(tmp_path: Path, mocker: MockerFixture) -> None:
    """Fast path ignores cache when mtime doesn't match."""
    # Create pipeline.py
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")

    # Create cache with wrong mtime
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "stages.cache").write_text("v1\npipeline.py:0.0\nstale_stage\n")

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    # Fast path should return None (no yaml, cache invalid)
    result = completion._get_stages_fast()

    assert result is None  # Falls through to None (no YAML file)


def test_get_stages_full_writes_cache(
    mock_discovery: Pipeline, tmp_path: Path, mocker: MockerFixture
) -> None:
    """Full discovery writes cache after successful discovery."""
    _ = mock_discovery

    # Create pipeline.py config file
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")

    # Register stages
    register_test_stage(_noop, name="discovered_stage")

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_full()

    # Stage should be discovered
    assert "discovered_stage" in result

    # Cache should be written
    cache_path = tmp_path / ".pivot" / "cache" / "stages.cache"
    assert cache_path.exists()
    content = cache_path.read_text()
    assert "discovered_stage" in content


def test_get_stages_full_skips_cache_on_mtime_change(tmp_path: Path, mocker: MockerFixture) -> None:
    """Full discovery skips cache write if config file changed during discovery."""

    # Create pipeline.py
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    # Create a mock pipeline that modifies the file during discovery
    mock_pipeline = mocker.MagicMock(spec=pipeline_mod.Pipeline)
    mock_pipeline.list_stages.return_value = ["stage1"]

    def discover_and_modify() -> pipeline_mod.Pipeline:
        # Simulate file change during discovery
        time.sleep(0.01)  # Ensure mtime changes
        config.write_text("# modified pipeline")
        return mock_pipeline

    mocker.patch.object(discovery, "discover_pipeline", side_effect=discover_and_modify)

    completion._get_stages_full()

    # Cache should NOT be written due to mtime change
    cache_path = tmp_path / ".pivot" / "cache" / "stages.cache"
    assert not cache_path.exists()


def test_complete_stages_uses_cache(
    tmp_path: Path,
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
    mocker: MockerFixture,
) -> None:
    """Complete stages uses cache when available."""
    # Create pipeline.py with cache
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")
    mtime = config.stat().st_mtime

    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "stages.cache").write_text(f"v1\npipeline.py:{mtime}\ncached_train\ncached_test\n")

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion.complete_stages(mock_ctx, mock_param, "cached_t")

    assert set(result) == {"cached_train", "cached_test"}


# =============================================================================
# _get_stages_full additional tests
# =============================================================================


def test_get_stages_full_returns_empty_when_no_pipeline(mocker: MockerFixture) -> None:
    """Returns empty list when discover_pipeline() finds no pipeline."""
    mocker.patch.object(discovery, "discover_pipeline", return_value=None)
    mocker.patch.object(completion, "_find_project_root_fast", return_value=None)

    result = completion._get_stages_full()

    assert result == []


def test_get_stages_full_handles_cache_write_oserror(
    mock_discovery: Pipeline, tmp_path: Path, mocker: MockerFixture
) -> None:
    """Cache write failure is silently ignored, stages still returned."""
    _ = mock_discovery

    # Create a config file so _detect_config_file succeeds
    config = tmp_path / "pipeline.py"
    config.write_text("# pipeline")

    register_test_stage(_noop, name="resilient_stage")

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)
    # Make the cache write fail (e.g. read-only filesystem)
    mocker.patch.object(
        completion, "_write_stages_cache", side_effect=OSError("read-only filesystem")
    )

    result = completion._get_stages_full()

    # Stages should still be returned despite cache write failure
    assert "resilient_stage" in result


# =============================================================================
# _write_stages_cache error handling tests
# =============================================================================


def test_write_stages_cache_cleans_up_temp_on_rename_failure(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Cleans up temp file when rename fails, then re-raises."""
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)

    # Make rename fail to trigger the OSError cleanup path
    original_rename = pathlib.Path.rename
    rename_called = False

    def failing_rename(self: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        nonlocal rename_called
        # Only fail for .tmp files (the temp file created by mkstemp)
        if self.suffix == ".tmp":
            rename_called = True
            raise OSError("simulated rename failure")
        return original_rename(self, target)

    mocker.patch.object(pathlib.Path, "rename", failing_rename)

    with pytest.raises(OSError, match="simulated rename failure"):
        completion._write_stages_cache(tmp_path, "pipeline.py", 1234567890.0, ["stage1"])

    assert rename_called, "rename should have been called"
    # Verify no .tmp files left behind in cache dir
    tmp_files = list(cache_dir.glob("*.tmp"))
    assert tmp_files == [], f"Temp files should be cleaned up, found: {tmp_files}"


# =============================================================================
# _get_config_keys edge case tests
# =============================================================================


def test_get_config_keys_skips_empty_yaml_file(
    tmp_path: Path,
    mock_ctx: mock.MagicMock,
    mock_param: mock.MagicMock,
    mocker: MockerFixture,
) -> None:
    """Skips config files with empty/non-dict YAML content (e.g. just a comment)."""
    config_path = tmp_path / "config.yaml"
    # yaml.safe_load("# just a comment") returns None, not a dict
    config_path.write_text("# just a comment\n")
    mocker.patch.object(
        completion,
        "_get_config_paths",
        return_value=[config_path],
    )

    result = completion.complete_config_keys(mock_ctx, mock_param, "cache")
    values = [item.value for item in result]
    # Should still return static keys without crashing
    assert "cache.dir" in values


# =============================================================================
# Cache write-then-read roundtrip tests
# =============================================================================


def test_cache_roundtrip_preserves_stages(tmp_path: Path) -> None:
    """Writing then reading cache preserves the exact stage list."""
    config = tmp_path / "pivot.yaml"
    config.write_text("stages: {}")
    mtime = config.stat().st_mtime

    stages = ["train", "test@bert_swe", "preprocess", "evaluate@gpt_human"]
    completion._write_stages_cache(tmp_path, "pivot.yaml", mtime, stages)
    result = completion._get_stages_from_cache(tmp_path)

    assert result == stages, "Roundtrip should preserve exact stage list and order"
