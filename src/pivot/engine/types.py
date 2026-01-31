"""Core type definitions for the engine module."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.types import StageStatus

__all__ = [
    "StageExecutionState",
    "NodeType",
    "EngineState",
    # Input events
    "DataArtifactChanged",
    "CodeOrConfigChanged",
    "RunRequested",
    "CancelRequested",
    "InputEvent",
    # Output events
    "EngineStateChanged",
    "PipelineReloaded",
    "StageStarted",
    "StageCompleted",
    "StageStateChanged",
    "LogLine",
    "OutputEvent",
    # Protocols
    "EventSource",
    "EventSink",
]


class StageExecutionState(IntEnum):
    """Execution state of a single stage.

    States progress forward (with exception of re-triggering after completion).
    IntEnum enables ordered comparisons: state >= PREPARING means execution began.
    """

    PENDING = 0  # Not yet considered
    BLOCKED = 1  # Waiting for upstream stages
    READY = 2  # Can run, waiting for worker
    PREPARING = 3  # Pivot clearing outputs
    RUNNING = 4  # Stage function executing
    COMPLETED = 5  # Terminal (ran/skipped/failed)


class NodeType(Enum):
    """Node type in the bipartite artifact-stage graph."""

    ARTIFACT = "artifact"
    STAGE = "stage"


class EngineState(Enum):
    """Top-level engine state."""

    IDLE = "idle"  # Not started
    ACTIVE = "active"  # Processing events and executing
    SHUTDOWN = "shutdown"  # Draining, no new stages started


# =============================================================================
# Input Events (triggers)
# =============================================================================


class DataArtifactChanged(TypedDict):
    """Dep/Out files changed on disk."""

    type: Literal["data_artifact_changed"]
    paths: list[str]


class CodeOrConfigChanged(TypedDict):
    """Python files or pivot.yaml/pipeline.py changed."""

    type: Literal["code_or_config_changed"]
    paths: list[str]


class RunRequested(TypedDict):
    """Explicit run request from CLI, RPC, or agent."""

    type: Literal["run_requested"]
    stages: list[str] | None  # None = all stages
    force: bool
    reason: str  # "cli", "agent:{run_id}", "watch:initial"


class CancelRequested(TypedDict):
    """Stop scheduling new stages, let running ones complete."""

    type: Literal["cancel_requested"]


InputEvent = DataArtifactChanged | CodeOrConfigChanged | RunRequested | CancelRequested


# =============================================================================
# Output Events (notifications)
# =============================================================================


class EngineStateChanged(TypedDict):
    """Engine transitioned to a new state."""

    type: Literal["engine_state_changed"]
    state: EngineState


class PipelineReloaded(TypedDict):
    """Registry was reloaded, DAG structure may have changed."""

    type: Literal["pipeline_reloaded"]
    stages_added: list[str]
    stages_removed: list[str]
    stages_modified: list[str]
    error: str | None


class StageStarted(TypedDict):
    """A stage began executing."""

    type: Literal["stage_started"]
    stage: str
    index: int
    total: int


class StageCompleted(TypedDict):
    """A stage finished (ran, skipped, or failed)."""

    type: Literal["stage_completed"]
    stage: str
    status: StageStatus
    reason: str
    duration_ms: float
    index: int
    total: int


class LogLine(TypedDict):
    """A line of output from a running stage."""

    type: Literal["log_line"]
    stage: str
    line: str
    is_stderr: bool


class StageStateChanged(TypedDict):
    """Emitted when a stage's execution state changes."""

    type: Literal["stage_state_changed"]
    stage: str
    state: StageExecutionState
    previous_state: StageExecutionState


OutputEvent = (
    EngineStateChanged
    | PipelineReloaded
    | StageStarted
    | StageCompleted
    | StageStateChanged
    | LogLine
)


# =============================================================================
# Protocols
# =============================================================================


class EventSource(Protocol):
    """Source that produces input events."""

    def start(self, submit: Callable[[InputEvent], None]) -> None:
        """Begin producing events. Call submit() for each event."""
        ...

    def stop(self) -> None:
        """Stop producing events."""
        ...


class EventSink(Protocol):
    """Sink that consumes output events."""

    def handle(self, event: OutputEvent) -> None:
        """Process an event. Must be non-blocking."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...
