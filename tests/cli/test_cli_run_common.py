"""Tests for shared run/repro CLI helpers."""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import TYPE_CHECKING, Annotated, TypedDict

import click
import click.testing
import networkx as nx
import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from pivot.pipeline.pipeline import Pipeline

from helpers import register_test_stage
from pivot import loaders, outputs
from pivot.cli import _run_common

# =============================================================================
# Module-level TypedDicts and Stage Functions for annotation-based registration
# =============================================================================


class _StageAOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _StageBOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


class _StageCOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("c.txt", loaders.PathOnly())]


def _helper_stage_a() -> _StageAOutputs:
    return _StageAOutputs(output=pathlib.Path("a.txt"))


def _helper_stage_b(
    dep: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _StageBOutputs:
    _ = dep
    return _StageBOutputs(output=pathlib.Path("b.txt"))


def _helper_stage_c(
    dep: Annotated[pathlib.Path, outputs.Dep("b.txt", loaders.PathOnly())],
) -> _StageCOutputs:
    _ = dep
    return _StageCOutputs(output=pathlib.Path("c.txt"))


# =============================================================================
# suppress_stderr_logging Tests
# =============================================================================


def test_suppress_stderr_logging_removes_stderr_handlers() -> None:
    """suppress_stderr_logging removes StreamHandlers writing to stderr."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)

    with _run_common.suppress_stderr_logging():
        assert handler not in root.handlers

    assert handler in root.handlers
    root.removeHandler(handler)


def test_suppress_stderr_logging_removes_stdout_handlers() -> None:
    """suppress_stderr_logging removes StreamHandlers writing to stdout."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    root.addHandler(handler)

    with _run_common.suppress_stderr_logging():
        assert handler not in root.handlers

    assert handler in root.handlers
    root.removeHandler(handler)


def test_suppress_stderr_logging_preserves_file_handlers(tmp_path: pathlib.Path) -> None:
    """suppress_stderr_logging preserves handlers not writing to stdout/stderr."""
    root = logging.getLogger()
    log_file = tmp_path / "test.log"
    handler = logging.FileHandler(log_file)
    root.addHandler(handler)

    with _run_common.suppress_stderr_logging():
        assert handler in root.handlers

    assert handler in root.handlers
    root.removeHandler(handler)
    handler.close()


def test_suppress_stderr_logging_restores_handlers_on_exception() -> None:
    """suppress_stderr_logging restores handlers even when exception occurs."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)

    with pytest.raises(ValueError), _run_common.suppress_stderr_logging():
        assert handler not in root.handlers
        raise ValueError("test")

    assert handler in root.handlers
    root.removeHandler(handler)


# =============================================================================
# compute_dag_levels Tests
# =============================================================================


def test_compute_dag_levels_empty_graph() -> None:
    """compute_dag_levels returns empty dict for empty graph."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    levels = _run_common.compute_dag_levels(graph)
    assert levels == {}


def test_compute_dag_levels_single_node() -> None:
    """compute_dag_levels assigns level 0 to single node."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_node("a")
    levels = _run_common.compute_dag_levels(graph)
    assert levels == {"a": 0}


def test_compute_dag_levels_linear_chain() -> None:
    """compute_dag_levels assigns increasing levels for linear chain."""
    # a -> b -> c (c depends on b, b depends on a)
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_edge("c", "b")  # c depends on b
    graph.add_edge("b", "a")  # b depends on a
    levels = _run_common.compute_dag_levels(graph)
    assert levels["a"] == 0
    assert levels["b"] == 1
    assert levels["c"] == 2


def test_compute_dag_levels_parallel_nodes() -> None:
    """compute_dag_levels assigns same level to parallel nodes."""
    # a (level 0), b (level 0), c depends on both (level 1)
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_edge("c", "a")
    graph.add_edge("c", "b")
    levels = _run_common.compute_dag_levels(graph)
    assert levels["a"] == 0
    assert levels["b"] == 0
    assert levels["c"] == 1


def test_compute_dag_levels_diamond_graph() -> None:
    """compute_dag_levels handles diamond dependency pattern."""
    #     a (0)
    #    / \
    #   b   c  (both level 1)
    #    \ /
    #     d (2)
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_edge("b", "a")
    graph.add_edge("c", "a")
    graph.add_edge("d", "b")
    graph.add_edge("d", "c")
    levels = _run_common.compute_dag_levels(graph)
    assert levels["a"] == 0
    assert levels["b"] == 1
    assert levels["c"] == 1
    assert levels["d"] == 2


# =============================================================================
# sort_for_display Tests
# =============================================================================


def test_sort_for_display_preserves_dag_order() -> None:
    """sort_for_display respects DAG structure."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_edge("b", "a")  # b depends on a
    execution_order = ["a", "b"]
    result = _run_common.sort_for_display(execution_order, graph)
    assert result.index("a") < result.index("b")


def test_sort_for_display_groups_matrix_variants() -> None:
    """sort_for_display groups matrix variants together."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_node("train@small")
    graph.add_node("train@medium")
    graph.add_node("train@large")
    execution_order = ["train@large", "train@small", "train@medium"]
    result = _run_common.sort_for_display(execution_order, graph)
    # All variants should be grouped, sorted alphabetically
    assert result == ["train@large", "train@medium", "train@small"]


def test_sort_for_display_groups_variants_at_earliest_level() -> None:
    """sort_for_display groups variants at their earliest member's level."""
    # preprocess -> train@small (level 1)
    # preprocess -> train@large (level 1)
    # both train variants depend on preprocess
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_edge("train@small", "preprocess")
    graph.add_edge("train@large", "preprocess")
    execution_order = ["preprocess", "train@small", "train@large"]
    result = _run_common.sort_for_display(execution_order, graph)
    # preprocess first, then train variants grouped
    assert result[0] == "preprocess"
    assert set(result[1:]) == {"train@small", "train@large"}


def test_sort_for_display_handles_non_variant_stages() -> None:
    """sort_for_display handles stages without @ in name."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_edge("b", "a")
    graph.add_edge("c", "b")
    execution_order = ["a", "b", "c"]
    result = _run_common.sort_for_display(execution_order, graph)
    assert result == ["a", "b", "c"]


# =============================================================================
# ensure_stages_registered Tests
# =============================================================================


def test_ensure_stages_registered_does_nothing_when_stages_exist(
    mock_discovery: Pipeline,
) -> None:
    """ensure_stages_registered is a no-op when stages already registered."""
    register_test_stage(_helper_stage_a, name="stage_a")
    _run_common.ensure_stages_registered()  # Should not raise


def test_ensure_stages_registered_raises_on_discovery_error(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """ensure_stages_registered raises ClickException on discovery error."""
    from conftest import isolated_pivot_dir

    with isolated_pivot_dir(runner, tmp_path) as cwd:
        # Create invalid pivot.yaml
        (cwd / "pivot.yaml").write_text("stages:\n  - invalid: true\n")

        with pytest.raises(click.ClickException):
            _run_common.ensure_stages_registered()


# =============================================================================
# validate_tui_log Tests
# =============================================================================


def test_validate_tui_log_returns_none_for_none() -> None:
    """validate_tui_log returns None when tui_log is None."""
    result = _run_common.validate_tui_log(None, as_json=False, tui_flag=True)
    assert result is None


def test_validate_tui_log_raises_for_json(tmp_path: pathlib.Path) -> None:
    """validate_tui_log raises when used with --json."""
    tui_log = tmp_path / "test.jsonl"
    with pytest.raises(click.ClickException, match="cannot be used with --jsonl"):
        _run_common.validate_tui_log(tui_log, as_json=True, tui_flag=True)


def test_validate_tui_log_raises_without_tui_flag(tmp_path: pathlib.Path) -> None:
    """validate_tui_log raises when --tui flag is not set."""
    tui_log = tmp_path / "test.jsonl"
    with pytest.raises(click.ClickException, match="--tui-log requires --tui"):
        _run_common.validate_tui_log(tui_log, as_json=False, tui_flag=False)


def test_validate_tui_log_raises_for_dry_run(tmp_path: pathlib.Path) -> None:
    """validate_tui_log raises when used with --dry-run."""
    tui_log = tmp_path / "test.jsonl"
    with pytest.raises(click.ClickException, match="cannot be used with --dry-run"):
        _run_common.validate_tui_log(tui_log, as_json=False, tui_flag=True, dry_run=True)


def test_validate_tui_log_resolves_path(tmp_path: pathlib.Path) -> None:
    """validate_tui_log resolves and returns the path."""
    tui_log = tmp_path / "test.jsonl"
    result = _run_common.validate_tui_log(tui_log, as_json=False, tui_flag=True)
    assert result is not None
    assert result.is_absolute()
    assert result.exists()


def test_validate_tui_log_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    """validate_tui_log creates parent directories."""
    tui_log = tmp_path / "subdir" / "nested" / "test.jsonl"
    result = _run_common.validate_tui_log(tui_log, as_json=False, tui_flag=True)
    assert result is not None
    assert result.parent.exists()


def test_validate_tui_log_raises_on_unwritable_path(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """validate_tui_log raises ClickException when path is unwritable."""
    tui_log = tmp_path / "test.jsonl"
    mocker.patch.object(
        pathlib.Path, "touch", autospec=True, side_effect=OSError("Permission denied")
    )

    with pytest.raises(click.ClickException) as exc_info:
        _run_common.validate_tui_log(tui_log, as_json=False, tui_flag=True)

    assert "Cannot write to" in str(exc_info.value)
    assert "Permission denied" in str(exc_info.value)


# =============================================================================
# validate_show_output Tests
# =============================================================================


def test_validate_show_output_raises_for_tui() -> None:
    """validate_show_output raises when used with --tui."""
    with pytest.raises(
        click.ClickException, match="--show-output and --tui are mutually exclusive"
    ):
        _run_common.validate_show_output(
            show_output=True, tui_flag=True, as_json=False, quiet=False
        )


def test_validate_show_output_raises_for_json() -> None:
    """validate_show_output raises when used with --jsonl."""
    with pytest.raises(
        click.ClickException, match="--show-output and --jsonl are mutually exclusive"
    ):
        _run_common.validate_show_output(
            show_output=True, tui_flag=False, as_json=True, quiet=False
        )


def test_validate_show_output_raises_for_quiet() -> None:
    """validate_show_output raises when used with --quiet."""
    with pytest.raises(
        click.ClickException, match="--show-output and --quiet are mutually exclusive"
    ):
        _run_common.validate_show_output(
            show_output=True, tui_flag=False, as_json=False, quiet=True
        )


def test_validate_show_output_succeeds_when_valid() -> None:
    """validate_show_output returns None when all flags are compatible."""
    _run_common.validate_show_output(show_output=True, tui_flag=False, as_json=False, quiet=False)
    # Should not raise


def test_validate_show_output_succeeds_when_disabled() -> None:
    """validate_show_output returns None when show_output=False regardless of other flags."""
    _run_common.validate_show_output(show_output=False, tui_flag=True, as_json=True, quiet=True)
    # Should not raise


# =============================================================================
# TypedDict Types Tests
# =============================================================================


def test_dry_run_json_stage_output_has_required_fields() -> None:
    """DryRunJsonStageOutput has required fields."""
    output = _run_common.DryRunJsonStageOutput(would_run=True, reason="code changed")
    assert output["would_run"] is True
    assert output["reason"] == "code changed"


def test_dry_run_json_output_has_required_fields() -> None:
    """DryRunJsonOutput has required fields."""
    stage_output = _run_common.DryRunJsonStageOutput(would_run=True, reason="forced")
    output = _run_common.DryRunJsonOutput(stages={"train": stage_output})
    assert "train" in output["stages"]
    assert output["stages"]["train"]["would_run"] is True


# =============================================================================
# configure_result_collector Tests
# =============================================================================


def test_configure_result_collector_adds_sink(mocker: MockerFixture) -> None:
    """configure_result_collector adds ResultCollectorSink to engine."""
    from pivot.engine import engine

    mock_engine = mocker.MagicMock(spec=engine.Engine)
    result_sink = _run_common.configure_result_collector(mock_engine)

    mock_engine.add_sink.assert_called_once()
    # Check that the sink added is the one returned
    call_args = mock_engine.add_sink.call_args
    assert call_args[0][0] is result_sink


# =============================================================================
# configure_output_sink Tests
# =============================================================================


def test_configure_output_sink_json_mode(mocker: MockerFixture) -> None:
    """configure_output_sink adds JsonlSink when as_json=True."""
    from pivot.engine import engine

    mock_engine = mocker.MagicMock(spec=engine.Engine)
    callback = mocker.MagicMock()

    _run_common.configure_output_sink(
        mock_engine,
        quiet=False,
        as_json=True,
        tui=False,
        app=None,
        run_id=None,
        use_console=True,
        jsonl_callback=callback,
    )

    mock_engine.add_sink.assert_called_once()
    added_sink = mock_engine.add_sink.call_args[0][0]
    assert isinstance(added_sink, _run_common.JsonlSink)


def test_configure_output_sink_quiet_mode(mocker: MockerFixture) -> None:
    """configure_output_sink adds no sinks when quiet=True and not JSON."""
    from pivot.engine import engine

    mock_engine = mocker.MagicMock(spec=engine.Engine)

    _run_common.configure_output_sink(
        mock_engine,
        quiet=True,
        as_json=False,
        tui=False,
        app=None,
        run_id=None,
        use_console=True,
        jsonl_callback=None,
    )

    mock_engine.add_sink.assert_not_called()


def test_configure_output_sink_tui_mode(mocker: MockerFixture) -> None:
    """configure_output_sink adds TuiSink when tui=True."""
    from pivot.engine import engine, sinks

    mock_engine = mocker.MagicMock(spec=engine.Engine)
    mock_app = mocker.MagicMock()

    _run_common.configure_output_sink(
        mock_engine,
        quiet=False,
        as_json=False,
        tui=True,
        app=mock_app,
        run_id="test-run-123",
        use_console=False,
        jsonl_callback=None,
    )

    mock_engine.add_sink.assert_called_once()
    added_sink = mock_engine.add_sink.call_args[0][0]
    assert isinstance(added_sink, sinks.TuiSink)


def test_configure_output_sink_console_mode(mocker: MockerFixture) -> None:
    """configure_output_sink adds ConsoleSink when use_console=True."""
    from pivot.engine import engine, sinks

    mock_engine = mocker.MagicMock(spec=engine.Engine)

    _run_common.configure_output_sink(
        mock_engine,
        quiet=False,
        as_json=False,
        tui=False,
        app=None,
        run_id=None,
        use_console=True,
        jsonl_callback=None,
    )

    mock_engine.add_sink.assert_called_once()
    added_sink = mock_engine.add_sink.call_args[0][0]
    assert isinstance(added_sink, sinks.ConsoleSink)


def test_configure_output_sink_json_overrides_quiet(mocker: MockerFixture) -> None:
    """configure_output_sink adds JsonlSink even when quiet=True if as_json=True."""
    from pivot.engine import engine

    mock_engine = mocker.MagicMock(spec=engine.Engine)
    callback = mocker.MagicMock()

    _run_common.configure_output_sink(
        mock_engine,
        quiet=True,
        as_json=True,
        tui=False,
        app=None,
        run_id=None,
        use_console=False,
        jsonl_callback=callback,
    )

    mock_engine.add_sink.assert_called_once()
    added_sink = mock_engine.add_sink.call_args[0][0]
    assert isinstance(added_sink, _run_common.JsonlSink)


# =============================================================================
# convert_results Tests
# =============================================================================


def test_convert_results_converts_ran_status() -> None:
    """convert_results converts StageCompleted events to ExecutionSummary."""
    from pivot.engine.types import StageCompleted
    from pivot.types import StageStatus

    stage_results: dict[str, StageCompleted] = {
        "train": StageCompleted(
            type="stage_completed",
            stage="train",
            status=StageStatus.RAN,
            reason="inputs changed",
            duration_ms=1234.5,
            index=0,
            total=1,
            input_hash="abc123",
        )
    }

    summaries = _run_common.convert_results(stage_results)

    assert "train" in summaries
    assert summaries["train"]["status"] == StageStatus.RAN
    assert summaries["train"]["reason"] == "inputs changed"
    assert summaries["train"]["input_hash"] == "abc123"


def test_convert_results_handles_multiple_stages() -> None:
    """convert_results handles multiple stages with different statuses."""
    from pivot.engine.types import StageCompleted
    from pivot.types import StageStatus

    stage_results: dict[str, StageCompleted] = {
        "train": StageCompleted(
            type="stage_completed",
            stage="train",
            status=StageStatus.RAN,
            reason="code changed",
            duration_ms=100.0,
            index=0,
            total=2,
            input_hash=None,
        ),
        "evaluate": StageCompleted(
            type="stage_completed",
            stage="evaluate",
            status=StageStatus.SKIPPED,
            reason="unchanged",
            duration_ms=0.0,
            index=1,
            total=2,
            input_hash=None,
        ),
    }

    summaries = _run_common.convert_results(stage_results)

    assert len(summaries) == 2
    assert summaries["train"]["status"] == StageStatus.RAN
    assert summaries["evaluate"]["status"] == StageStatus.SKIPPED


def test_convert_results_empty_dict() -> None:
    """convert_results handles empty stage results."""
    summaries = _run_common.convert_results({})
    assert summaries == {}


# =============================================================================
# validate_display_mode Tests
# =============================================================================


def test_validate_display_mode_raises_for_tui_and_json() -> None:
    """validate_display_mode raises when --tui and --jsonl both set."""
    with pytest.raises(click.ClickException, match="--tui and --jsonl are mutually exclusive"):
        _run_common.validate_display_mode(tui_flag=True, as_json=True)


def test_validate_display_mode_accepts_tui_only() -> None:
    """validate_display_mode accepts --tui alone."""
    _run_common.validate_display_mode(tui_flag=True, as_json=False)
    # Should not raise


def test_validate_display_mode_accepts_json_only() -> None:
    """validate_display_mode accepts --jsonl alone."""
    _run_common.validate_display_mode(tui_flag=False, as_json=True)
    # Should not raise


def test_validate_display_mode_accepts_neither() -> None:
    """validate_display_mode accepts neither flag."""
    _run_common.validate_display_mode(tui_flag=False, as_json=False)
    # Should not raise


# =============================================================================
# resolve_on_error Tests
# =============================================================================


def test_resolve_on_error_default_is_fail() -> None:
    """resolve_on_error returns FAIL when neither flag is set."""
    from pivot.types import OnError

    result = _run_common.resolve_on_error(fail_fast=False, keep_going=False)
    assert result == OnError.FAIL


def test_resolve_on_error_fail_fast_returns_fail() -> None:
    """resolve_on_error returns FAIL when --fail-fast is set."""
    from pivot.types import OnError

    result = _run_common.resolve_on_error(fail_fast=True, keep_going=False)
    assert result == OnError.FAIL


def test_resolve_on_error_keep_going_returns_keep_going() -> None:
    """resolve_on_error returns KEEP_GOING when --keep-going is set."""
    from pivot.types import OnError

    result = _run_common.resolve_on_error(fail_fast=False, keep_going=True)
    assert result == OnError.KEEP_GOING


def test_resolve_on_error_both_flags_raises() -> None:
    """resolve_on_error raises when both flags are set."""
    with pytest.raises(click.ClickException, match="mutually exclusive"):
        _run_common.resolve_on_error(fail_fast=True, keep_going=True)
