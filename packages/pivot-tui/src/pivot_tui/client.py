from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.types import StageExplanation


class EngineStatus(TypedDict):
    state: Literal["idle", "active"]
    running: list[str]
    pending: list[str]


class CommitResult(TypedDict):
    committed: list[str]
    failed: list[str]


class StageInfoResult(TypedDict):
    name: str
    deps: list[str]
    outs: list[str]


class VersionedEvent(TypedDict):
    version: int
    event: dict[str, object]


class EventBatch(TypedDict):
    version: int
    events: list[VersionedEvent]


class PivotRpc(Protocol):
    """RPC methods for engine interaction (no transport concern)."""

    async def run(self, stages: list[str] | None = None, *, force: bool = False) -> bool: ...

    async def cancel(self) -> bool: ...

    async def commit(self) -> CommitResult: ...

    async def status(self) -> EngineStatus: ...

    async def stages(self) -> list[str]: ...

    async def stage_info(self, stage: str) -> StageInfoResult: ...

    async def explain(self, stage: str) -> StageExplanation: ...

    async def events_since(self, version: int) -> EventBatch: ...

    async def set_on_error(self, mode: Literal["fail", "keep_going"]) -> bool: ...

    async def diff_output(
        self,
        path: str,
        old_hash: str | None,
        new_hash: str | None,
        *,
        max_rows: int = 50,
    ) -> dict[str, object]: ...


class PivotClient(PivotRpc, Protocol):
    """Full client with transport + RPC."""

    async def connect(self, socket_path: Path) -> None: ...

    async def disconnect(self) -> None: ...
