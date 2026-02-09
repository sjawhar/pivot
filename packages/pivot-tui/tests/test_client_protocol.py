from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportAny=false
import inspect
from typing import TYPE_CHECKING, Literal

from pivot.types import CodeChange, DepChange, ParamChange, StageExplanation
from pivot_tui.client import (
    CommitResult,
    EngineStatus,
    EventBatch,
    PivotClient,
    PivotRpc,
    StageInfoResult,
    VersionedEvent,
)

if TYPE_CHECKING:
    from pathlib import Path


class MockClient:
    async def connect(self, socket_path: Path) -> None:
        _ = socket_path
        return None

    async def disconnect(self) -> None:
        return None

    async def run(self, stages: list[str] | None = None, *, force: bool = False) -> bool:
        _ = (stages, force)
        return True

    async def cancel(self) -> bool:
        return True

    async def commit(self) -> CommitResult:
        return CommitResult(committed=list[str](), failed=list[str]())

    async def status(self) -> EngineStatus:
        return EngineStatus(state="idle", running=list[str](), pending=list[str]())

    async def stages(self) -> list[str]:
        return list[str]()

    async def stage_info(self, stage: str) -> StageInfoResult:
        return StageInfoResult(name=stage, deps=list[str](), outs=list[str]())

    async def explain(self, stage: str) -> StageExplanation:
        return StageExplanation(
            stage_name=stage,
            will_run=False,
            is_forced=False,
            reason="",
            code_changes=list[CodeChange](),
            param_changes=list[ParamChange](),
            dep_changes=list[DepChange](),
            upstream_stale=list[str](),
        )

    async def events_since(self, version: int) -> EventBatch:
        return EventBatch(version=version, events=list[VersionedEvent]())

    async def set_on_error(self, mode: Literal["fail", "keep_going"]) -> bool:
        _ = mode
        return True

    async def diff_output(
        self,
        path: str,
        old_hash: str | None,
        new_hash: str | None,
        *,
        max_rows: int = 50,
    ) -> dict[str, object]:
        _ = (path, old_hash, new_hash, max_rows)
        return dict[str, object]()


def test_mock_client_matches_protocol_shape() -> None:
    client: PivotClient = MockClient()

    for name in (
        "connect",
        "disconnect",
        "run",
        "cancel",
        "commit",
        "status",
        "stages",
        "stage_info",
        "explain",
        "events_since",
        "set_on_error",
        "diff_output",
    ):
        attribute = getattr(MockClient, name)
        assert inspect.iscoroutinefunction(attribute)

    assert client is not None


class MockRpc:
    """Implements only PivotRpc — no connect/disconnect."""

    async def run(self, stages: list[str] | None = None, *, force: bool = False) -> bool:
        _ = (stages, force)
        return True

    async def cancel(self) -> bool:
        return True

    async def commit(self) -> CommitResult:
        return CommitResult(committed=list[str](), failed=list[str]())

    async def status(self) -> EngineStatus:
        return EngineStatus(state="idle", running=list[str](), pending=list[str]())

    async def stages(self) -> list[str]:
        return list[str]()

    async def stage_info(self, stage: str) -> StageInfoResult:
        return StageInfoResult(name=stage, deps=list[str](), outs=list[str]())

    async def explain(self, stage: str) -> StageExplanation:
        return StageExplanation(
            stage_name=stage,
            will_run=False,
            is_forced=False,
            reason="",
            code_changes=list[CodeChange](),
            param_changes=list[ParamChange](),
            dep_changes=list[DepChange](),
            upstream_stale=list[str](),
        )

    async def events_since(self, version: int) -> EventBatch:
        return EventBatch(version=version, events=list[VersionedEvent]())

    async def set_on_error(self, mode: Literal["fail", "keep_going"]) -> bool:
        _ = mode
        return True

    async def diff_output(
        self,
        path: str,
        old_hash: str | None,
        new_hash: str | None,
        *,
        max_rows: int = 50,
    ) -> dict[str, object]:
        _ = (path, old_hash, new_hash, max_rows)
        return dict[str, object]()


def test_mock_rpc_satisfies_pivot_rpc_protocol() -> None:
    """MockRpc without connect/disconnect satisfies PivotRpc."""
    rpc: PivotRpc = MockRpc()

    for name in (
        "run",
        "cancel",
        "commit",
        "status",
        "stages",
        "stage_info",
        "explain",
        "events_since",
        "set_on_error",
        "diff_output",
    ):
        attribute = getattr(MockRpc, name)
        assert inspect.iscoroutinefunction(attribute), f"{name} should be async"

    assert rpc is not None, "MockRpc must satisfy PivotRpc protocol"
    assert not hasattr(rpc, "connect"), "MockRpc should not have connect"
    assert not hasattr(rpc, "disconnect"), "MockRpc should not have disconnect"


def test_pivot_client_is_also_pivot_rpc() -> None:
    """PivotClient satisfies PivotRpc (it extends it)."""
    client: PivotClient = MockClient()
    rpc: PivotRpc = client  # PivotClient is a subtype of PivotRpc
    assert rpc is not None


def test_response_typed_dicts_constructible() -> None:
    engine_status = EngineStatus(state="idle", running=["a"], pending=["b"])
    commit_result = CommitResult(committed=["a"], failed=["b"])
    stage_info = StageInfoResult(name="stage", deps=["dep"], outs=["out"])
    versioned_event = VersionedEvent(version=1, event={"type": "log_line"})
    event_batch = EventBatch(version=1, events=[versioned_event])
    explanation = StageExplanation(
        stage_name="stage",
        will_run=True,
        is_forced=False,
        reason="reason",
        code_changes=list[CodeChange](),
        param_changes=list[ParamChange](),
        dep_changes=list[DepChange](),
        upstream_stale=list[str](),
    )

    assert engine_status["state"] == "idle"
    assert commit_result["committed"] == ["a"]
    assert stage_info["name"] == "stage"
    assert versioned_event["version"] == 1
    assert event_batch["events"][0]["event"]["type"] == "log_line"
    assert explanation["stage_name"] == "stage"
