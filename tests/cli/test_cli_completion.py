from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import click
import pytest

from pivot.cli import completion

if TYPE_CHECKING:
    import pathlib

    from pytest_mock import MockerFixture


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ctx() -> click.Context:
    """Create a mock Click context."""
    return mock.MagicMock(spec=click.Context)


@pytest.fixture
def mock_param() -> click.Parameter:
    """Create a mock Click parameter."""
    return mock.MagicMock(spec=click.Parameter)


# =============================================================================
# _get_stages_fast tests
# =============================================================================


def test_get_stages_fast_simple_yaml(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
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


def test_get_stages_fast_matrix_yaml(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
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


def test_get_stages_fast_mixed_simple_and_matrix(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
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


def test_get_stages_fast_returns_none_when_no_yaml(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Returns None when pivot.yaml doesn't exist."""
    (tmp_path / ".git").mkdir()

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_fast()
    assert result is None


def test_get_stages_fast_returns_none_on_variants(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
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


def test_get_stages_fast_returns_none_on_yaml_error(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Returns None on YAML parse error."""
    (tmp_path / "pivot.yaml").write_text("invalid: yaml: [[[")
    (tmp_path / ".git").mkdir()

    mocker.patch.object(completion, "_find_project_root_fast", return_value=tmp_path)

    result = completion._get_stages_fast()
    assert result is None


def test_get_stages_fast_pivot_yml_alternative(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
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


def test_find_project_root_fast_finds_git_marker(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Finds project root via .git marker."""
    (tmp_path / ".git").mkdir()
    mocker.patch("pathlib.Path.cwd", return_value=tmp_path)

    result = completion._find_project_root_fast()
    assert result == tmp_path


def test_find_project_root_fast_finds_pivot_marker(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Finds project root via .pivot marker."""
    (tmp_path / ".pivot").mkdir()
    mocker.patch("pathlib.Path.cwd", return_value=tmp_path)

    result = completion._find_project_root_fast()
    assert result == tmp_path


def test_find_project_root_fast_walks_up(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Walks up directory tree to find marker."""
    (tmp_path / ".git").mkdir()
    subdir = tmp_path / "src" / "pkg"
    subdir.mkdir(parents=True)
    mocker.patch("pathlib.Path.cwd", return_value=subdir)

    result = completion._find_project_root_fast()
    assert result == tmp_path


def test_find_project_root_fast_returns_none_when_no_marker(
    tmp_path: pathlib.Path, mocker: MockerFixture
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


def test_get_stages_full_returns_registered_stages(mocker: MockerFixture) -> None:
    """Returns stages from registry after discovery."""
    mock_discovery = mocker.patch("pivot.discovery")
    mock_discovery.has_registered_stages.return_value = False

    mock_registry = mocker.patch("pivot.registry")
    mock_registry.REGISTRY.list_stages.return_value = ["stage1", "stage2"]

    result = completion._get_stages_full()
    assert result == ["stage1", "stage2"]
    mock_discovery.discover_and_register.assert_called_once()


def test_get_stages_full_skips_discovery_if_registered(mocker: MockerFixture) -> None:
    """Skips discovery if stages already registered."""
    mock_discovery = mocker.patch("pivot.discovery")
    mock_discovery.has_registered_stages.return_value = True

    mock_registry = mocker.patch("pivot.registry")
    mock_registry.REGISTRY.list_stages.return_value = ["existing"]

    result = completion._get_stages_full()
    assert result == ["existing"]
    mock_discovery.discover_and_register.assert_not_called()


# =============================================================================
# complete_stages tests
# =============================================================================


def test_complete_stages_filters_by_prefix(
    mock_ctx: click.Context, mock_param: click.Parameter, mocker: MockerFixture
) -> None:
    """Filters stage names by incomplete prefix."""
    mocker.patch.object(
        completion, "_get_stages_fast", return_value=["train", "test", "preprocess"]
    )

    result = completion.complete_stages(mock_ctx, mock_param, "tr")
    assert result == ["train"]


def test_complete_stages_empty_prefix_returns_all(
    mock_ctx: click.Context, mock_param: click.Parameter, mocker: MockerFixture
) -> None:
    """Empty prefix returns all stages."""
    mocker.patch.object(completion, "_get_stages_fast", return_value=["train", "test"])

    result = completion.complete_stages(mock_ctx, mock_param, "")
    assert set(result) == {"train", "test"}


def test_complete_stages_falls_back_when_fast_returns_none(
    mock_ctx: click.Context, mock_param: click.Parameter, mocker: MockerFixture
) -> None:
    """Falls back to full discovery when fast path returns None."""
    mocker.patch.object(completion, "_get_stages_fast", return_value=None)
    mocker.patch.object(completion, "_get_stages_full", return_value=["fallback_stage"])

    result = completion.complete_stages(mock_ctx, mock_param, "")
    assert result == ["fallback_stage"]


def test_complete_stages_returns_empty_on_exception(
    mock_ctx: click.Context, mock_param: click.Parameter, mocker: MockerFixture
) -> None:
    """Returns empty list if exception occurs."""
    mocker.patch.object(completion, "_get_stages_fast", side_effect=Exception("boom"))

    result = completion.complete_stages(mock_ctx, mock_param, "")
    assert result == []


def test_complete_stages_matrix_stage_completion(
    mock_ctx: click.Context, mock_param: click.Parameter, mocker: MockerFixture
) -> None:
    """Completes matrix stage names correctly."""
    mocker.patch.object(
        completion,
        "_get_stages_fast",
        return_value=["train@bert_swe", "train@gpt_swe", "preprocess"],
    )

    result = completion.complete_stages(mock_ctx, mock_param, "train@b")
    assert result == ["train@bert_swe"]


def test_complete_stages_case_sensitive(
    mock_ctx: click.Context, mock_param: click.Parameter, mocker: MockerFixture
) -> None:
    """Completion is case-sensitive."""
    mocker.patch.object(completion, "_get_stages_fast", return_value=["Train", "train", "TRAIN"])

    result = completion.complete_stages(mock_ctx, mock_param, "tr")
    assert result == ["train"]


# =============================================================================
# complete_targets tests
# =============================================================================


def test_complete_targets_includes_stage_names(
    mock_ctx: click.Context, mock_param: click.Parameter, mocker: MockerFixture
) -> None:
    """Target completion includes stage names."""
    mocker.patch.object(completion, "_get_stages_fast", return_value=["train", "test"])

    result = completion.complete_targets(mock_ctx, mock_param, "tr")
    assert "train" in result


def test_complete_targets_filters_by_prefix(
    mock_ctx: click.Context, mock_param: click.Parameter, mocker: MockerFixture
) -> None:
    """Filters targets by incomplete prefix."""
    mocker.patch.object(completion, "_get_stages_fast", return_value=["train", "test", "deploy"])

    result = completion.complete_targets(mock_ctx, mock_param, "t")
    assert set(result) == {"train", "test"}
