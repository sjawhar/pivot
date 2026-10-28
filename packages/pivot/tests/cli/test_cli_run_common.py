"""Tests for shared run/repro CLI helpers."""

from __future__ import annotations

import concurrent.futures
import logging
import pathlib
import sys
import threading
from typing import TYPE_CHECKING, override

import click
import click.testing
import networkx as nx
import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

import pivot_tui.event_poller
import pivot_tui.rpc_client_impl
from conftest import isolated_pivot_dir
from pivot import cli, compose, exceptions
from pivot.cli import _run_common
from pivot.engine import engine as engine_mod

# =============================================================================
# Module-level stage functions (required for pickling)
# =============================================================================


@compose.stage
def _helper_stage_a() -> str:
    return "a"


class _FakeApp:
    posted: list[object]
    event_poller: object | None
    ran: bool
    return_code: int | None
    run_return_code: int

    def __init__(self, *, run_return_code: int = 0) -> None:
        self.posted = list[object]()
        self.event_poller = None
        self.ran = False
        self.return_code = None
        self.run_return_code = run_return_code

    def post_message(self, message: object) -> bool:
        self.posted.append(message)
        return True

    def set_event_poller(self, poller: object) -> None:
        self.event_poller = poller

    def run(self) -> None:
        self.ran = True
        self.return_code = self.run_return_code


class _FakeRpcPivotClient:
    instances: list[_FakeRpcPivotClient] = []
    failures_before_success: int = 0

    connect_attempts: int

    def __init__(self) -> None:
        self.connect_attempts = 0
        type(self).instances.append(self)

    async def connect(self, _socket_path: pathlib.Path) -> None:
        self.connect_attempts += 1
        if self.connect_attempts <= self.failures_before_success:
            raise OSError("connect failed")

    async def disconnect(self) -> None:
        return None


class _FakeEventPoller:
    oneshot: bool
    ran: bool

    def __init__(self, _client: object, _post_fn: object, *, oneshot: bool) -> None:
        self.oneshot = oneshot
        self.ran = False

    async def run(self) -> None:
        self.ran = True

    def stop(self) -> None:
        return None


class _DoneFuture(concurrent.futures.Future[str]):
    set_result_called: bool
    set_exception_called: bool

    def __init__(self) -> None:
        super().__init__()
        self.set_result_called = False
        self.set_exception_called = False

    @override
    def done(self) -> bool:
        return True

    @override
    def set_result(self, result: str) -> None:
        _ = result
        self.set_result_called = True

    @override
    def set_exception(self, exception: BaseException | None) -> None:
        _ = exception
        self.set_exception_called = True


# =============================================================================
# defer_console_logging Tests
# =============================================================================


@pytest.mark.parametrize("stream_name", ["stderr", "stdout"])
def test_defer_console_logging_prints_warnings_once_the_tui_exits(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture, stream_name: str
) -> None:
    """A console handler prints nothing while the TUI runs, then the held warnings and errors,
    in order; the lower levels it drops."""
    caplog.set_level(logging.INFO, logger="pivot.engine")
    root = logging.getLogger()
    handler = logging.StreamHandler(getattr(sys, stream_name))
    root.addHandler(handler)
    logger = logging.getLogger("pivot.engine")
    try:
        with _run_common.defer_console_logging():
            logger.warning("tracked file changed")
            logger.info("progress")
            logger.error("stage failed")
            during = capsys.readouterr()
        after = capsys.readouterr()
        logger.warning("after the TUI")
        later = capsys.readouterr()
    finally:
        root.removeHandler(handler)

    assert during.out + during.err == "", "nothing should reach the terminal while the TUI runs"
    printed = getattr(after, stream_name[3:])
    assert printed == "tracked file changed\nstage failed\n", printed
    assert getattr(later, stream_name[3:]) == "after the TUI\n", "the handler should print again"


def test_defer_console_logging_drops_the_tuis_own_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Records the TUI logs about itself describe a display that has closed, so they are
    dropped, not printed after it exits."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    try:
        with _run_common.defer_console_logging():
            logging.getLogger("pivot_tui.run").error("Failed to connect to engine socket")
            logging.getLogger("pivot.executor.core").warning("tracked file changed")
        after = capsys.readouterr()
    finally:
        root.removeHandler(handler)

    assert after.err == "tracked file changed\n", after.err


def test_defer_console_logging_prints_a_repeated_warning_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A watch session checks the tracked files on every run; its warning prints once."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    logger = logging.getLogger("pivot.executor.core")
    try:
        with _run_common.defer_console_logging():
            for _ in range(3):
                logger.warning("tracked file changed")
                logger.warning("other file changed")
        after = capsys.readouterr()
    finally:
        root.removeHandler(handler)

    assert after.err == "tracked file changed\nother file changed\n", after.err


def test_defer_console_logging_holds_a_malformed_record_without_raising(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warning whose arguments do not fit its format string raises nothing on the thread that
    logs it; the handler reports it once the TUI exits, as it would have at once."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    # Only this handler: pytest's own capture handler raises on a malformed record.
    monkeypatch.setattr(root, "handlers", [handler])
    monkeypatch.setattr(logging, "raiseExceptions", True)
    logger = logging.getLogger("pivot.engine")
    with _run_common.defer_console_logging():
        logger.warning("needs %d", "text")
        logger.warning("tracked file changed")
    after = capsys.readouterr()

    assert "--- Logging error ---" in after.err, after.err
    assert after.err.endswith("tracked file changed\n"), after.err


def test_defer_console_logging_prints_a_warning_logged_on_another_thread(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The engine logs on its own thread while the TUI runs on the main thread."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    engine_thread = threading.Thread(
        target=logging.getLogger("pivot.executor.core").warning, args=("tracked file changed",)
    )
    try:
        with _run_common.defer_console_logging():
            engine_thread.start()
            engine_thread.join()
            during = capsys.readouterr()
        after = capsys.readouterr()
    finally:
        root.removeHandler(handler)

    assert during.err == "", during.err
    assert after.err == "tracked file changed\n", after.err


def test_defer_console_logging_leaves_file_handlers_alone(tmp_path: pathlib.Path) -> None:
    """A handler that writes to a file keeps writing while the TUI runs."""
    root = logging.getLogger()
    log_file = tmp_path / "test.log"
    handler = logging.FileHandler(log_file)
    root.addHandler(handler)
    try:
        with _run_common.defer_console_logging():
            logging.getLogger("pivot.engine").warning("tracked file changed")
            during = log_file.read_text()
    finally:
        root.removeHandler(handler)
        handler.close()

    assert during == "tracked file changed\n", during


def test_defer_console_logging_prints_held_warnings_when_the_tui_raises(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A TUI that ends on an exception still gets its held warnings printed."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    try:
        with pytest.raises(ValueError), _run_common.defer_console_logging():
            logging.getLogger("pivot.engine").warning("tracked file changed")
            raise ValueError("test")
        after = capsys.readouterr()
    finally:
        root.removeHandler(handler)

    assert after.err == "tracked file changed\n", after.err


# =============================================================================
# deferral key Tests
# =============================================================================
def test_defer_console_logging_never_reuses_a_bad_record_key(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed record held after the TUI starts cannot collide with an earlier dead record."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    monkeypatch.setattr(root, "handlers", [handler])
    monkeypatch.setattr(logging, "raiseExceptions", True)
    logger = logging.getLogger("pivot.engine")
    with _run_common.defer_console_logging(hold_now=False) as console_logging:
        for _ in range(20):
            logger.warning("needs %d", "text")
        console_logging.start_holding()
        for _ in range(20):
            logger.warning("needs %d", "different text")
    after = capsys.readouterr()

    assert after.err.count("--- Logging error ---") == 40, after.err


def test_defer_console_logging_ignores_a_third_party_condition(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A third-party logger can use an unhashable extra named pivot_condition without changing
    Pivot's deferral or raising in the thread that logged it."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    try:
        with _run_common.defer_console_logging():
            logging.getLogger("third_party").warning("x", extra={"pivot_condition": []})
        after = capsys.readouterr()
    finally:
        root.removeHandler(handler)

    assert after.err == "x\n", after.err


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
    mock_discovery: compose.Pipeline,
) -> None:
    """ensure_stages_registered is a no-op when stages already registered."""
    with mock_discovery:
        _helper_stage_a()

    _run_common.ensure_stages_registered()  # Should not raise

    assert mock_discovery.list_stages() == ["test/_helper_stage_a"], (
        "Pipeline in context should be left untouched"
    )


def test_ensure_stages_registered_raises_on_discovery_error(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """ensure_stages_registered raises ClickException on discovery error."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        # pipeline.py that raises while loading
        (cwd / "pipeline.py").write_text("raise RuntimeError('broken pipeline')\n")

        with pytest.raises(click.ClickException, match="broken pipeline"):
            _run_common.ensure_stages_registered()


# =============================================================================
# validate_tui_log Tests
# =============================================================================


def test_validate_tui_log_returns_none_for_none() -> None:
    """validate_tui_log returns None when tui_log is None."""
    result = _run_common.validate_tui_log(None, as_json=False, tui_flag=True)
    assert result is None


def test_validate_tui_log_raises_for_jsonl(tmp_path: pathlib.Path) -> None:
    """validate_tui_log raises when used with --jsonl."""
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
        click.ClickException, match="--show-output and --jsonl/--json are mutually exclusive"
    ):
        _run_common.validate_show_output(
            show_output=True, tui_flag=False, as_json=True, quiet=False
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
        use_console=True,
        jsonl_callback=None,
    )

    mock_engine.add_sink.assert_not_called()


def test_configure_output_sink_picks_live_for_terminal(mocker: MockerFixture) -> None:
    """configure_output_sink adds LiveConsoleSink for terminal consoles."""
    from pivot.engine import engine, sinks

    mock_engine = mocker.MagicMock(spec=engine.Engine)
    mocker.patch(
        "rich.console.Console.is_terminal",
        new_callable=mocker.PropertyMock,
        return_value=True,
    )

    _run_common.configure_output_sink(
        mock_engine,
        quiet=False,
        as_json=False,
        use_console=True,
        jsonl_callback=None,
    )

    mock_engine.add_sink.assert_called_once()
    added_sink = mock_engine.add_sink.call_args[0][0]
    assert isinstance(added_sink, sinks.LiveConsoleSink), "Should add LiveConsoleSink"


def test_configure_output_sink_picks_static_for_non_terminal(mocker: MockerFixture) -> None:
    """configure_output_sink adds StaticConsoleSink for non-terminal consoles."""
    from pivot.engine import engine, sinks

    mock_engine = mocker.MagicMock(spec=engine.Engine)
    mocker.patch(
        "rich.console.Console.is_terminal",
        new_callable=mocker.PropertyMock,
        return_value=False,
    )

    _run_common.configure_output_sink(
        mock_engine,
        quiet=False,
        as_json=False,
        use_console=True,
        jsonl_callback=None,
    )

    mock_engine.add_sink.assert_called_once()
    added_sink = mock_engine.add_sink.call_args[0][0]
    assert isinstance(added_sink, sinks.StaticConsoleSink), "Should add StaticConsoleSink"


def test_configure_output_sink_json_overrides_quiet(mocker: MockerFixture) -> None:
    """configure_output_sink adds JsonlSink even when quiet=True if as_json=True."""
    from pivot.engine import engine

    mock_engine = mocker.MagicMock(spec=engine.Engine)
    callback = mocker.MagicMock()

    _run_common.configure_output_sink(
        mock_engine,
        quiet=True,
        as_json=True,
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
            seq=0,
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
            seq=0,
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
            seq=1,
            stage="evaluate",
            status=StageStatus.CACHED,
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
    assert summaries["evaluate"]["status"] == StageStatus.CACHED


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


@pytest.mark.parametrize("command", ["repro", "run"])
@pytest.mark.parametrize(
    ("values", "message"),
    [
        pytest.param(["gpu"], "must use NAME=POSITIVE_INTEGER", id="missing-capacity"),
        pytest.param(["gpu=0"], "capacity must be a positive integer", id="zero"),
        pytest.param(["gpu=1", "GPU=2"], "'gpu' was supplied more than once", id="repeated"),
    ],
)
def test_resources_option_refuses_bad_capacity(
    runner: click.testing.CliRunner, command: str, values: list[str], message: str
) -> None:
    """repro and run share one --resources option with the same parsing."""
    args = [command]
    for value in values:
        args += ["--resources", value]

    result = runner.invoke(cli.cli, args)

    assert result.exit_code == 2, result.output
    assert "--resources" in result.output, result.output
    assert message in result.output, result.output


@pytest.mark.parametrize("command", ["repro", "run"])
def test_resources_option_documented_on_run_commands(
    runner: click.testing.CliRunner, command: str
) -> None:
    result = runner.invoke(cli.cli, [command, "--help"])

    assert result.exit_code == 0, result.output
    assert "--resources NAME=CAPACITY" in result.output, result.output
    assert "Set a scheduler resource" in result.output, result.output


def test_run_tui_with_engine_retries_poller_connect(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    """run_tui_with_engine retries poller connect after failures."""
    app = _FakeApp()
    socket_path = tmp_path / "pivot.sock"
    result_future: concurrent.futures.Future[str] = concurrent.futures.Future()

    _FakeRpcPivotClient.instances = []
    _FakeRpcPivotClient.failures_before_success = 2

    # The poller stops retrying once the engine finishes or the TUI exits, so keep both
    # running until the poller has connected.
    async def _engine_fn(socket_ready: threading.Event) -> str:
        import anyio

        socket_ready.set()
        with anyio.fail_after(10):
            while app.event_poller is None:
                await anyio.sleep(0.01)
        return "ok"

    def _run_until_connected() -> None:
        import time

        app.ran = True
        deadline = time.monotonic() + 10
        while app.event_poller is None and time.monotonic() < deadline:
            time.sleep(0.01)
        app.return_code = 0

    app.run = _run_until_connected
    mocker.patch.object(_run_common, "_POLLER_CONNECT_RETRY_S", 0.01)
    mocker.patch.object(pivot_tui.rpc_client_impl, "RpcPivotClient", _FakeRpcPivotClient)
    mocker.patch.object(pivot_tui.event_poller, "EventPoller", _FakeEventPoller)

    _run_common.run_tui_with_engine(
        app,
        _engine_fn,
        socket_path,
        result_future=result_future,
    )

    assert app.ran is True
    assert _FakeRpcPivotClient.instances, "Expected poller client to be created"
    poller_client = _FakeRpcPivotClient.instances[0]
    assert poller_client.connect_attempts == 3, "Should retry poller connect"
    assert result_future.result(timeout=1) == "ok"


def test_run_tui_with_engine_skips_setting_done_future(tmp_path: pathlib.Path) -> None:
    """run_tui_with_engine does not set result on completed futures."""
    app = _FakeApp()
    socket_path = tmp_path / "pivot.sock"
    result_future = _DoneFuture()

    async def _engine_fn(socket_ready: threading.Event) -> str:
        socket_ready.set()
        return "ok"

    _FakeRpcPivotClient.instances = []
    _FakeRpcPivotClient.failures_before_success = 0

    async def _fake_sleep(_duration: float) -> None:
        return None

    with (
        pytest.MonkeyPatch.context() as monkeypatch,
    ):
        monkeypatch.setattr("pivot.cli._run_common.anyio.sleep", _fake_sleep, raising=True)
        monkeypatch.setattr(pivot_tui.rpc_client_impl, "RpcPivotClient", _FakeRpcPivotClient)
        monkeypatch.setattr(pivot_tui.event_poller, "EventPoller", _FakeEventPoller)

        _run_common.run_tui_with_engine(
            app,
            _engine_fn,
            socket_path,
            result_future=result_future,
        )

    assert result_future.set_result_called is False
    assert result_future.set_exception_called is False


def test_run_tui_with_engine_exits_nonzero_when_app_crashes(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    """A TUI that exits with a non-zero return code (unhandled exception) fails the command."""
    app = _FakeApp(run_return_code=1)

    async def _engine_fn(socket_ready: threading.Event) -> None:
        socket_ready.set()

    _FakeRpcPivotClient.instances = []
    _FakeRpcPivotClient.failures_before_success = 0
    mocker.patch.object(pivot_tui.rpc_client_impl, "RpcPivotClient", _FakeRpcPivotClient)
    mocker.patch.object(pivot_tui.event_poller, "EventPoller", _FakeEventPoller)

    with pytest.raises(SystemExit) as exc_info:
        _run_common.run_tui_with_engine(
            app,
            _engine_fn,
            tmp_path / "pivot.sock",
            oneshot=False,
            suppress_keyboard_interrupt=True,
        )

    assert exc_info.value.code == 1


def test_run_tui_with_engine_watch_engine_crash_exits_tui_and_fails(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    """A watch-mode engine that crashes ends the TUI and fails the command with its error."""
    import time

    import pivot_tui.run as tui_run

    app = _FakeApp()

    def _run_until_shutdown() -> None:
        app.ran = True
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if any(isinstance(m, tui_run.TuiShutdown) for m in app.posted):
                break
            time.sleep(0.01)
        app.return_code = 1

    app.run = _run_until_shutdown

    async def _engine_fn(socket_ready: threading.Event) -> None:
        socket_ready.set()
        raise RuntimeError("engine crashed")

    mocker.patch.object(pivot_tui.rpc_client_impl, "RpcPivotClient", _FakeRpcPivotClient)
    mocker.patch.object(pivot_tui.event_poller, "EventPoller", _FakeEventPoller)

    with pytest.raises(RuntimeError, match="engine crashed"):
        _run_common.run_tui_with_engine(
            app,
            _engine_fn,
            tmp_path / "pivot.sock",
            oneshot=False,
            suppress_keyboard_interrupt=True,
        )

    shutdowns = [m for m in app.posted if isinstance(m, tui_run.TuiShutdown)]
    assert len(shutdowns) == 1
    assert shutdowns[0].error is not None
    assert "engine crashed" in shutdowns[0].error


def test_run_tui_with_engine_watch_quit_while_engine_runs_succeeds(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    """Quitting the watch TUI while the engine is still running returns normally (exit 0)."""
    import threading

    import anyio

    app = _FakeApp()
    release_engine = threading.Event()
    engine_threads_before = set(threading.enumerate())

    async def _engine_fn(socket_ready: threading.Event) -> None:
        socket_ready.set()
        while not release_engine.is_set():
            await anyio.sleep(0.01)

    mocker.patch.object(pivot_tui.rpc_client_impl, "RpcPivotClient", _FakeRpcPivotClient)
    mocker.patch.object(pivot_tui.event_poller, "EventPoller", _FakeEventPoller)

    try:
        _run_common.run_tui_with_engine(
            app,
            _engine_fn,
            tmp_path / "pivot.sock",
            oneshot=False,
            suppress_keyboard_interrupt=True,
        )
    finally:
        release_engine.set()
        for thread in set(threading.enumerate()) - engine_threads_before:
            thread.join(timeout=5)

    assert app.ran is True


def test_run_tui_with_engine_prints_a_warning_once_when_logged_before_and_during_the_tui(
    mocker: MockerFixture, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A watch session checks the tracked files on every run. When the engine's first check
    logs its warning before the main thread starts the TUI, and a later run logs the same
    warning while the TUI runs, the user sees it once."""
    import anyio

    app = _FakeApp()
    tui_running = threading.Event()
    logged_during_tui = threading.Event()
    logger = logging.getLogger("pivot.executor.core")
    warning = "Tracked file 'x.txt' has changed since tracking."

    async def _engine_fn(socket_ready: threading.Event) -> str:
        logger.warning(warning)  # the startup run's check, before the TUI starts
        socket_ready.set()
        with anyio.fail_after(10):
            while not tui_running.is_set():
                await anyio.sleep(0.01)
        logger.warning(warning)  # a later run's check, while the TUI runs
        logged_during_tui.set()
        return "ok"

    def _run_until_logged() -> None:
        app.ran = True
        tui_running.set()
        assert logged_during_tui.wait(10), "the engine never logged while the TUI ran"
        app.return_code = 0

    app.run = _run_until_logged
    _FakeRpcPivotClient.instances = []
    _FakeRpcPivotClient.failures_before_success = 0
    mocker.patch.object(pivot_tui.rpc_client_impl, "RpcPivotClient", _FakeRpcPivotClient)
    mocker.patch.object(pivot_tui.event_poller, "EventPoller", _FakeEventPoller)
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    result_future: concurrent.futures.Future[str] = concurrent.futures.Future()
    try:
        _run_common.run_tui_with_engine(
            app, _engine_fn, tmp_path / "pivot.sock", result_future=result_future
        )
    finally:
        root.removeHandler(handler)

    printed = capsys.readouterr().err
    assert printed.count(warning) == 1, printed
    assert result_future.result(timeout=1) == "ok"


def test_run_tui_with_engine_stops_poller_when_engine_finishes_before_connect(
    mocker: MockerFixture, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A one-shot engine that finishes before the poller connects ends the poller thread.

    The engine removes its socket on exit, so a poller still retrying connect must stop
    when the engine (and TUI) finish, not keep retrying after the command returns.
    """
    import threading
    import time

    app = _FakeApp()
    result_future: concurrent.futures.Future[str] = concurrent.futures.Future()

    async def _engine_fn(socket_ready: threading.Event) -> str:
        socket_ready.set()
        return "ok"

    class _SlowFirstConnectClient:
        connect_calls: int = 0

        async def connect(self, _socket_path: pathlib.Path) -> None:
            type(self).connect_calls += 1
            if type(self).connect_calls == 1:
                time.sleep(3.0)
            raise OSError("socket removed: engine already finished")

        async def disconnect(self) -> None:
            return None

    mocker.patch.object(pivot_tui.rpc_client_impl, "RpcPivotClient", _SlowFirstConnectClient)
    mocker.patch.object(pivot_tui.event_poller, "EventPoller", _FakeEventPoller)
    threads_before = set(threading.enumerate())

    with caplog.at_level(logging.DEBUG, logger="pivot.cli._run_common"):
        _run_common.run_tui_with_engine(
            app,
            _engine_fn,
            tmp_path / "pivot.sock",
            result_future=result_future,
        )

    new_threads = [t for t in threading.enumerate() if t not in threads_before and t.is_alive()]
    assert new_threads == [], f"poller thread still alive after return: {new_threads}"
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert result_future.result(timeout=1) == "ok"


@pytest.mark.parametrize(
    "checks_before_tui",
    [
        pytest.param(False, id="every-check-while-the-tui-runs"),
        pytest.param(True, id="startup-check-before-the-tui-holds-output"),
    ],
)
def test_run_tui_with_engine_watch_prints_a_repeated_warning_once(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    checks_before_tui: bool,
) -> None:
    """A watch session checks the tracked files on every run. Its warning reaches the terminal
    once, whether the startup run's check comes before the TUI holds console output or after."""
    import time

    import anyio

    import pivot_tui.run as tui_run

    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    logger = logging.getLogger("pivot.executor.core")
    app = _FakeApp()
    tui_running = threading.Event()

    async def _engine_fn(socket_ready: threading.Event) -> None:
        if checks_before_tui:
            logger.warning("tracked file changed")  # the startup run's check
        socket_ready.set()
        while not tui_running.is_set():
            await anyio.sleep(0.01)
        if not checks_before_tui:
            logger.warning("tracked file changed")  # the startup run's check
        logger.warning("tracked file changed")  # a later run's check

    def _run_until_shutdown() -> None:
        app.ran = True
        tui_running.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if any(isinstance(m, tui_run.TuiShutdown) for m in app.posted):
                break
            time.sleep(0.01)
        app.return_code = 0

    app.run = _run_until_shutdown
    mocker.patch.object(pivot_tui.rpc_client_impl, "RpcPivotClient", _FakeRpcPivotClient)
    mocker.patch.object(pivot_tui.event_poller, "EventPoller", _FakeEventPoller)
    try:
        _run_common.run_tui_with_engine(
            app,
            _engine_fn,
            tmp_path / "pivot.sock",
            oneshot=False,
            suppress_keyboard_interrupt=True,
        )
    finally:
        root.removeHandler(handler)

    printed = capsys.readouterr().err
    assert printed.count("tracked file changed") == 1, printed


def test_defer_console_logging_prints_each_distinct_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stage that fails twice in a watch session with different errors gets both printed."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    logger = logging.getLogger("pivot.engine.engine")
    try:
        with _run_common.defer_console_logging():
            for error in (ValueError("first failure"), KeyError("second failure")):
                # As the engine logs a stage whose worker raised
                logger.error("Stage %s failed: %s", "demo/train", exceptions.describe_error(error))
        after = capsys.readouterr()
    finally:
        root.removeHandler(handler)

    assert "first failure" in after.err, after.err
    assert "second failure" in after.err, after.err


def test_defer_console_logging_holds_a_record_whose_argument_repr_raises(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An argument whose repr raises makes the logging call raise nothing, as without the TUI;
    the record prints once the TUI exits."""

    class _NoRepr:
        @override
        def __str__(self) -> str:
            return "value"

        @override
        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    try:
        with _run_common.defer_console_logging():
            logging.getLogger("pivot.engine").warning("got %s", _NoRepr())
        after = capsys.readouterr()
    finally:
        root.removeHandler(handler)

    assert after.err == "got value\n", after.err


@pytest.mark.parametrize(
    ("error_at_exit", "printed"),
    [
        pytest.param(None, [], id="fixed-before-exit"),
        pytest.param("boom", ["Pipeline invalid - waiting for fix"], id="still-invalid-at-exit"),
        pytest.param("other", [], id="replaced-by-another-error"),
    ],
)
def test_run_tui_with_engine_prints_a_pipeline_error_only_while_the_tui_shows_it(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    error_at_exit: str | None,
    printed: list[str],
) -> None:
    """A watch session whose pipeline broke and was fixed while the TUI ran prints no stale
    "Pipeline invalid" after exit; one the engine still reports when the TUI exits prints."""
    import anyio

    app = _FakeApp()
    tui_running = threading.Event()
    logged = threading.Event()
    logger = logging.getLogger(engine_mod.__name__)

    async def _engine_fn(socket_ready: threading.Event) -> None:
        socket_ready.set()
        with anyio.fail_after(10):
            while not tui_running.is_set():
                await anyio.sleep(0.01)
        logger.error(
            "Pipeline invalid - waiting for fix",
            extra={"pivot_condition": (engine_mod.PIPELINE_ERROR_CONDITION, "boom")},
        )
        logged.set()

    def _run_until_logged() -> None:
        app.ran = True
        tui_running.set()
        assert logged.wait(10), "the engine never logged while the TUI ran"
        app.return_code = 0

    app.run = _run_until_logged
    _FakeRpcPivotClient.instances = []
    _FakeRpcPivotClient.failures_before_success = 0
    mocker.patch.object(pivot_tui.rpc_client_impl, "RpcPivotClient", _FakeRpcPivotClient)
    mocker.patch.object(pivot_tui.event_poller, "EventPoller", _FakeEventPoller)
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    try:
        _run_common.run_tui_with_engine(
            app,
            _engine_fn,
            tmp_path / "pivot.sock",
            conditions=lambda: {engine_mod.PIPELINE_ERROR_CONDITION: error_at_exit},
            oneshot=False,
            suppress_keyboard_interrupt=True,
        )
    finally:
        root.removeHandler(handler)

    assert capsys.readouterr().err.splitlines() == printed


def test_run_tui_with_engine_prints_the_pipeline_error_the_tui_shows_after_a_second_break(
    mocker: MockerFixture, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A watch session whose pipeline broke one way and then another while the TUI ran, and
    is still broken the second way when the TUI exits, prints that error's lines once, as
    plain `pivot repro --watch` prints them for the second break."""
    import anyio

    app = _FakeApp()
    tui_running = threading.Event()
    logged = threading.Event()
    logger = logging.getLogger(engine_mod.__name__)
    error = "invalid syntax (pipeline.py, line 33)"

    def _log_break(message: str) -> None:
        condition = {"pivot_condition": (engine_mod.PIPELINE_ERROR_CONDITION, message)}
        logger.warning(f"Pipeline invalid: {message}", extra=condition)
        logger.error("Pipeline invalid - waiting for fix", extra=condition)

    async def _engine_fn(socket_ready: threading.Event) -> None:
        socket_ready.set()
        with anyio.fail_after(10):
            while not tui_running.is_set():
                await anyio.sleep(0.01)
        _log_break("invalid syntax (pipeline.py, line 29)")
        _log_break(error)
        logged.set()

    def _run_until_logged() -> None:
        app.ran = True
        tui_running.set()
        assert logged.wait(10), "the engine never logged while the TUI ran"
        app.return_code = 0

    app.run = _run_until_logged
    _FakeRpcPivotClient.instances = []
    _FakeRpcPivotClient.failures_before_success = 0
    mocker.patch.object(pivot_tui.rpc_client_impl, "RpcPivotClient", _FakeRpcPivotClient)
    mocker.patch.object(pivot_tui.event_poller, "EventPoller", _FakeEventPoller)
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    try:
        _run_common.run_tui_with_engine(
            app,
            _engine_fn,
            tmp_path / "pivot.sock",
            conditions=lambda: {engine_mod.PIPELINE_ERROR_CONDITION: error},
            oneshot=False,
            suppress_keyboard_interrupt=True,
        )
    finally:
        root.removeHandler(handler)

    assert capsys.readouterr().err.splitlines() == [
        f"Pipeline invalid: {error}",
        "Pipeline invalid - waiting for fix",
    ]


@pytest.mark.parametrize(
    "again_while_the_tui_runs",
    [
        pytest.param(False, id="logged-once"),
        pytest.param(True, id="logged-again-while-the-tui-runs"),
    ],
)
def test_run_tui_with_engine_prints_a_warning_the_engine_was_writing_as_the_tui_started(
    mocker: MockerFixture,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    again_while_the_tui_runs: bool,
) -> None:
    """The engine thread's warning passes the console handler's filters just before the TUI
    starts, and the handler writes it only after the TUI took over stderr (a descheduled thread
    on a busy machine). The user sees the warning once after the TUI exits, as for a warning
    logged while the TUI runs, and also when a watch session's later run logs it again."""
    import io

    app = _FakeApp()
    in_emit = threading.Event()
    tui_owns_stderr = threading.Event()
    logged = threading.Event()
    warning = "Tracked file 'x.txt' has changed since tracking."

    class _EngineWritesLate(logging.StreamHandler):  # pyright: ignore[reportMissingTypeArgument]
        """Writes to whatever sys.stderr is when it writes, as the CLI's console handler does."""

        @property
        @override
        def stream(self) -> object:
            return sys.stderr

        @stream.setter
        def stream(self, _value: object) -> None:
            pass

        @override
        def emit(self, record: logging.LogRecord) -> None:
            if threading.current_thread() is not threading.main_thread():
                in_emit.set()
                # Descheduled between the filters and the write: the TUI may start meanwhile.
                tui_owns_stderr.wait(2)
            super().emit(record)

    async def _engine_fn(socket_ready: threading.Event) -> str:
        socket_ready.set()
        engine_logger = logging.getLogger("pivot.executor.core")
        engine_logger.warning(warning)
        if again_while_the_tui_runs:
            engine_logger.warning(warning)  # a later watch run's check, while the TUI runs
        logged.set()
        return "ok"

    def _run_as_textual() -> None:
        # Textual sends sys.stderr to its own capture while the app runs, and drops what the
        # app does not route to a widget.
        app.ran = True
        terminal, sys.stderr = sys.stderr, io.StringIO()
        try:
            tui_owns_stderr.set()
            assert logged.wait(10), "the engine never logged"
        finally:
            sys.stderr = terminal
        app.return_code = 0

    app.run = _run_as_textual
    real_start = threading.Thread.start

    def _start_tui_once_the_engine_is_writing(thread: threading.Thread) -> None:
        # The main thread starts the event poller just before it starts the TUI.
        if thread.name == _run_common.POLLER_THREAD_NAME:
            assert in_emit.wait(10), "the engine never wrote its warning"
        real_start(thread)

    mocker.patch.object(threading.Thread, "start", _start_tui_once_the_engine_is_writing)
    _FakeRpcPivotClient.instances = []
    _FakeRpcPivotClient.failures_before_success = 0
    mocker.patch.object(pivot_tui.rpc_client_impl, "RpcPivotClient", _FakeRpcPivotClient)
    mocker.patch.object(pivot_tui.event_poller, "EventPoller", _FakeEventPoller)
    root = logging.getLogger()
    handler = _EngineWritesLate()
    root.addHandler(handler)
    result_future: concurrent.futures.Future[str] = concurrent.futures.Future()
    try:
        _run_common.run_tui_with_engine(
            app, _engine_fn, tmp_path / "pivot.sock", result_future=result_future
        )
    finally:
        root.removeHandler(handler)

    assert capsys.readouterr().err.count(warning) == 1
    assert result_future.result(timeout=1) == "ok"
