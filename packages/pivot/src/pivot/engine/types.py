from __future__ import annotations

from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict

if TYPE_CHECKING:
    import pathlib

    from anyio.streams.memory import MemoryObjectSendStream

    from pivot.types import CompletionType, OnError, StageExplanation

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
    "OutputChangeSummary",
    "StageStarted",
    "StageCompleted",
    "StageStateChanged",
    "LogLine",
    "SinkState",
    "SinkStateChanged",
    "EngineDiagnostic",
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
    WAITING_ON_LOCK = 4  # Waiting for artifact lock
    RUNNING = 5  # Stage function executing
    COMPLETED = 6  # Terminal (ran/skipped/failed)


class NodeType(Enum):
    """Node type in the bipartite artifact-stage graph."""

    ARTIFACT = "artifact"
    STAGE = "stage"


class EngineState(Enum):
    """Top-level engine state."""

    IDLE = "idle"  # Not started
    ACTIVE = "active"  # Processing events and executing


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
    """Explicit run request from CLI, RPC, or agent.

    All fields are required. Callers must provide all fields with their
    default values when creating events.
    """

    type: Literal["run_requested"]
    stages: list[str] | None  # None = all stages
    force: bool
    reason: str  # "cli", "agent:{run_id}", "watch:initial"
    single_stage: bool
    parallel: bool
    max_workers: int | None
    no_commit: bool
    on_error: OnError
    cache_dir: pathlib.Path | None
    allow_uncached_incremental: bool
    checkout_missing: bool


class CancelRequested(TypedDict):
    """Stop scheduling new stages, let running ones complete."""

    type: Literal["cancel_requested"]


InputEvent = DataArtifactChanged | CodeOrConfigChanged | RunRequested | CancelRequested


# =============================================================================
# Output Events (notifications)
#
# seq and run_id are NotRequired to allow construction without them (e.g. in tests,
# or from code outside Engine). Engine.emit() always stamps both fields at runtime,
# so they are guaranteed present on any event that flows through the engine.
# =============================================================================


class EngineStateChanged(TypedDict):
    """Engine transitioned to a new state."""

    type: Literal["engine_state_changed"]
    seq: NotRequired[int]
    run_id: NotRequired[str]
    state: EngineState


class PipelineReloaded(TypedDict):
    """Registry was reloaded, DAG structure may have changed."""

    type: Literal["pipeline_reloaded"]
    seq: NotRequired[int]
    run_id: NotRequired[str]
    stages: list[str]  # All stages in topological order
    stages_added: list[str]
    stages_removed: list[str]
    stages_modified: list[str]
    error: str | None


class StageStarted(TypedDict):
    """A stage began executing."""

    type: Literal["stage_started"]
    seq: NotRequired[int]
    stage: str
    index: int
    total: int
    run_id: NotRequired[str]
    explanation: NotRequired[StageExplanation | None]


class OutputChangeSummary(TypedDict):
    """Summary of output changes for a single output file."""

    path: str
    change_type: str | None  # "added", "modified", "removed", "unchanged", or None if unknown
    output_type: str  # "out", "metric", "plot"
    old_hash: str | None
    new_hash: str | None


class StageCompleted(TypedDict):
    """A stage finished (ran, skipped, or failed)."""

    type: Literal["stage_completed"]
    seq: NotRequired[int]
    stage: str
    status: CompletionType
    reason: str
    duration_ms: float
    index: int
    total: int
    run_id: NotRequired[str]
    input_hash: str | None
    output_summary: NotRequired[list[OutputChangeSummary] | None]


class LogLine(TypedDict):
    """A line of output from a running stage."""

    type: Literal["log_line"]
    seq: NotRequired[int]
    run_id: NotRequired[str]
    stage: str
    line: str
    is_stderr: bool


class StageStateChanged(TypedDict):
    """Emitted when a stage's execution state changes."""

    type: Literal["stage_state_changed"]
    seq: NotRequired[int]
    run_id: NotRequired[str]
    stage: str
    state: StageExecutionState
    previous_state: StageExecutionState


class SinkState(Enum):
    """Sink supervision state."""

    ENABLED = "enabled"
    DISABLED = "disabled"


class SinkStateChanged(TypedDict):
    """Emitted when a sink is disabled or re-enabled by supervision."""

    type: Literal["sink_state_changed"]
    seq: NotRequired[int]
    run_id: NotRequired[str]
    sink_id: str
    state: SinkState
    reason: str
    failure_count: int
    backoff_s: float | None


class EngineDiagnostic(TypedDict):
    """Engine-level diagnostic for non-fatal operational issues.

    Emitted when the engine detects an anomaly that doesn't warrant stopping
    execution but should be visible to operators (e.g., deferred event loop
    guard tripped, unexpected state transitions).
    """

    type: Literal["engine_diagnostic"]
    seq: NotRequired[int]
    run_id: NotRequired[str]
    message: str
    detail: str


OutputEvent = (
    EngineStateChanged
    | PipelineReloaded
    | StageStarted
    | StageCompleted
    | StageStateChanged
    | LogLine
    | SinkStateChanged
    | EngineDiagnostic
)


# =============================================================================
# Protocols
# =============================================================================


class EventSource(Protocol):
    """Protocol for async event sources that push events to the engine."""

    async def run(self, send: MemoryObjectSendStream[InputEvent]) -> None:
        """Run the source, pushing events to the send channel.

        The source should run until cancelled via task group cancellation.
        """
        ...


class EventSink(Protocol):
    """Protocol for async event sinks that receive engine output."""

    async def handle(self, event: OutputEvent) -> None:
        """Handle a single output event. Should complete quickly.

        Events are dispatched to each sink via a dedicated bounded queue
        (1024 items), preserving per-sink ordering. Slow sinks do not block
        other sinks; if a sink's queue fills or it raises repeated exceptions
        or times out, it is automatically disabled by the supervision system
        and re-enabled after an exponential backoff.
        """
        ...

    async def close(self) -> None:
        """Clean up resources when engine shuts down."""
        ...
