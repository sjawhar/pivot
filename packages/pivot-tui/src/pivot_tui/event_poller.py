"""Event poller that converts engine events to TUI messages."""

# pyright: reportMissingImports=false
# pyright: reportImplicitRelativeImport=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnusedCallResult=false
# pyright: reportUnnecessaryTypeIgnoreComment=false

from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, cast

import anyio
from anyio import to_thread

from pivot.types import (
    StageStatus,
    TuiLogMessage,
    TuiMessageType,
    TuiReloadMessage,
    TuiStatusMessage,
    TuiWatchMessage,
    WatchStatus,
)
from pivot_tui.rpc_client_impl import (
    RpcProtocolError,
    decode_output_summary,
    decode_stage_explanation,
)
from pivot_tui.types import TuiExpandedMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot_tui.client import EventBatch, PivotRpc
    from pivot_tui.types import TuiMessage

__all__ = ["EventPoller"]

_logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.1  # 100ms
_RETRY_INTERVAL = 1.0  # 1s backoff on disconnect

_EXPLANATION_REQUIRED_KEYS = frozenset(
    {
        "stage_name",
        "will_run",
        "is_forced",
        "reason",
        "code_changes",
        "param_changes",
        "dep_changes",
        "upstream_stale",
    }
)


class EventPoller:
    """Polls engine events and posts TUI messages.

    Continuously polls ``client.events_since(version)`` and converts raw event
    dicts into typed TUI messages (``TuiStatusMessage``, ``TuiLogMessage``,
    ``TuiWatchMessage``, ``TuiReloadMessage``, ``TuiExpandedMessage``), posting each via
    *post_fn*.
    """

    _client: PivotRpc
    _post_fn: Callable[[TuiMessage], object]
    _version: int
    _stop_event: threading.Event
    _oneshot: bool

    def __init__(
        self,
        client: PivotRpc,
        post_fn: Callable[[TuiMessage], object],
        *,
        oneshot: bool = False,
    ) -> None:
        self._client = client
        self._post_fn = post_fn
        self._version = 0
        self._stop_event = threading.Event()
        self._oneshot = oneshot

    async def run(self) -> None:
        """Poll for events until stopped.

        In watch mode (default), retries on transient connection errors.
        In oneshot mode, stops on fatal errors (EndOfStream, closed socket).
        """
        self._stop_event.clear()
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._stop_watcher, tg.cancel_scope)
            try:
                await self._poll_loop()
            finally:
                self._stop_event.set()
                tg.cancel_scope.cancel()

    def stop(self) -> None:
        """Signal the poller to stop immediately."""
        self._stop_event.set()

    async def _stop_watcher(self, cancel_scope: anyio.CancelScope) -> None:
        await to_thread.run_sync(self._stop_event.wait)
        cancel_scope.cancel()

    async def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                batch = await self._client.events_since(self._version)
                self._process_batch(batch)
            except (anyio.EndOfStream, anyio.ClosedResourceError):
                if self._oneshot:
                    _logger.warning("Event stream closed in oneshot mode, stopping poller")
                    return
                _logger.debug("Connection closed, retrying in %ss", _RETRY_INTERVAL)
                await anyio.sleep(_RETRY_INTERVAL)
                continue
            except (RpcProtocolError, json.JSONDecodeError) as exc:
                _logger.error("Fatal protocol error, stopping poller: %s", exc)
                return
            except Exception:
                if self._oneshot:
                    _logger.debug("Poll error in oneshot mode, stopping", exc_info=True)
                    return
                _logger.debug("Event poll failed, retrying in %ss", _RETRY_INTERVAL, exc_info=True)
                await anyio.sleep(_RETRY_INTERVAL)
                continue
            await anyio.sleep(_POLL_INTERVAL)

    def _process_batch(self, batch: EventBatch) -> None:
        """Convert events in a batch to TUI messages and post them."""
        for versioned in batch["events"]:
            event = versioned["event"]
            try:
                msg = _convert_event(event)
            except Exception:
                _logger.warning("Failed to convert event, skipping: %s", event, exc_info=True)
                self._version = versioned["version"]  # Advance past failed event too
                continue
            if msg is not None:
                self._post_fn(msg)
            self._version = versioned["version"]  # Advance after each event


def _convert_event(
    event: dict[str, object],
) -> TuiMessage | None:
    """Convert a raw event dict to the appropriate TUI message, or None."""
    event_type = event.get("type")
    match event_type:
        case "stage_started":
            return _make_started_message(event)
        case "stage_completed":
            return _make_completed_message(event)
        case "log_line":
            return _make_log_message(event)
        case "engine_state_changed":
            return _make_watch_message(event)
        case "pipeline_reloaded":
            return _make_reload_message(event)
        case "stages_expanded":
            return _make_expanded_message(event)
        case _:
            return None


def _make_started_message(event: dict[str, object]) -> TuiStatusMessage | None:
    stage = event["stage"]
    index = event["index"]
    total = event["total"]
    run_id = event.get("run_id", "")
    if not isinstance(stage, str):
        return None
    if not isinstance(index, int):
        return None
    if not isinstance(total, int):
        return None
    if not isinstance(run_id, str):
        return None
    msg = TuiStatusMessage(
        type=TuiMessageType.STATUS,
        stage=stage,
        index=index,
        total=total,
        status=StageStatus.IN_PROGRESS,
        reason="",
        elapsed=None,
        run_id=run_id,
    )
    explanation: object = event.get("explanation")
    if isinstance(explanation, dict) and explanation.keys() >= _EXPLANATION_REQUIRED_KEYS:
        try:
            msg["explanation"] = decode_stage_explanation(explanation)
        except (RpcProtocolError, KeyError, ValueError) as exc:
            _logger.warning("Invalid explanation payload, skipping: %s", exc)
    elif explanation is not None:
        _logger.warning("Invalid explanation payload, skipping: %s", type(explanation).__name__)
    return msg


def _make_completed_message(event: dict[str, object]) -> TuiStatusMessage | None:
    stage = event["stage"]
    index = event["index"]
    total = event["total"]
    status = event["status"]
    reason = event.get("reason", "")
    duration_ms = event.get("duration_ms", 0)
    run_id = event.get("run_id", "")
    if not isinstance(stage, str):
        return None
    if not isinstance(index, int):
        return None
    if not isinstance(total, int):
        return None
    if not isinstance(status, str):
        return None
    if not isinstance(reason, str):
        return None
    if not isinstance(duration_ms, int | float):
        return None
    if not isinstance(run_id, str):
        return None
    try:
        stage_status = StageStatus(status)
    except ValueError:
        _logger.warning("Unknown stage status %r, skipping event", status)
        return None
    msg = TuiStatusMessage(
        type=TuiMessageType.STATUS,
        stage=stage,
        index=index,
        total=total,
        status=stage_status,
        reason=reason,
        elapsed=duration_ms / 1000.0,
        run_id=run_id,
    )
    if "output_summary" in event:
        output_summary = event["output_summary"]
        msg["output_summary"] = (
            None if output_summary is None else decode_output_summary(output_summary)
        )
    return msg


def _make_log_message(event: dict[str, object]) -> TuiLogMessage | None:
    stage = event["stage"]
    line = event["line"]
    is_stderr = event.get("is_stderr", False)
    if not isinstance(stage, str):
        return None
    if not isinstance(line, str):
        return None
    if not isinstance(is_stderr, bool):
        return None
    return TuiLogMessage(
        type=TuiMessageType.LOG,
        stage=stage,
        line=line,
        is_stderr=is_stderr,
        timestamp=time.time(),
    )


def _make_watch_message(event: dict[str, object]) -> TuiWatchMessage | None:
    state = event["state"]
    if state == "active":
        return None
    if state != "idle":
        raise RpcProtocolError("'idle' or 'active' (state)", repr(state))
    return TuiWatchMessage(
        type=TuiMessageType.WATCH,
        status=WatchStatus.WAITING,
        message="Watching for changes…",
    )


def _make_reload_message(event: dict[str, object]) -> TuiReloadMessage | None:
    stages = event.get("stages")
    stages_added = event.get("stages_added")
    stages_removed = event.get("stages_removed")
    stages_modified = event.get("stages_modified")
    if not isinstance(stages, list):
        return None
    if not isinstance(stages_added, list):
        return None
    if not isinstance(stages_removed, list):
        return None
    if not isinstance(stages_modified, list):
        return None
    error = event["error"]
    if error is not None and not isinstance(error, str):
        raise RpcProtocolError("str | None (error)", type(error).__name__)
    return TuiReloadMessage(
        type=TuiMessageType.RELOAD,
        stages=cast("list[str]", stages),
        stages_added=cast("list[str]", stages_added),
        stages_removed=cast("list[str]", stages_removed),
        stages_modified=cast("list[str]", stages_modified),
        error=error,
    )


def _make_expanded_message(event: dict[str, object]) -> TuiExpandedMessage:
    template = event["template"]
    if not isinstance(template, str):
        raise RpcProtocolError("str (template)", type(template).__name__)
    stages = event["stages"]
    if not isinstance(stages, list):
        raise RpcProtocolError("list (stages)", type(stages).__name__)
    names = cast("list[object]", stages)
    for name in names:
        if not isinstance(name, str):
            raise RpcProtocolError("list[str] (stages)", type(name).__name__)
    return TuiExpandedMessage(
        type=TuiMessageType.EXPANDED, template=template, stages=cast("list[str]", names)
    )
