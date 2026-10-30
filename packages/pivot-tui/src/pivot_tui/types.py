from __future__ import annotations

import collections
import dataclasses
from typing import Literal, NamedTuple, TypedDict

from pivot.types import (
    OutputChange,
    StageExplanation,
    StageStatus,
    TuiLogMessage,
    TuiMessageType,
    TuiReloadMessage,
    TuiStatusMessage,
    TuiWatchMessage,
    parse_stage_name,
)


class TuiExpandedMessage(TypedDict):
    """A run added a template's instances (``Pipeline.map``, the engine's ``stages_expanded``):
    one stage per file of the directory it is mapped over, ``<template>@<key>``, in key order."""

    type: Literal[TuiMessageType.EXPANDED]
    template: str
    stages: list[str]


type TuiMessage = (
    TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage | TuiExpandedMessage
)


class LogEntry(NamedTuple):
    """A single log line with metadata."""

    line: str
    is_stderr: bool
    timestamp: float


@dataclasses.dataclass
class ExecutionHistoryEntry:
    """Snapshot of a single stage execution for history navigation."""

    run_id: str
    stage_name: str
    timestamp: float
    duration: float | None
    status: StageStatus
    reason: str
    logs: list[LogEntry]
    input_snapshot: StageExplanation | None
    output_snapshot: list[OutputChange] | None


@dataclasses.dataclass
class PendingHistoryState:
    """Temporary state for a stage execution in progress, before finalization."""

    run_id: str
    timestamp: float
    # Bounded deque to prevent memory growth in watch mode with verbose stages
    logs: collections.deque[LogEntry] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=500)
    )
    input_snapshot: StageExplanation | None = None


@dataclasses.dataclass
class StageInfo:
    """Mutable state for a single stage."""

    name: str
    index: int
    total: int
    base_name: str = ""  # Part before @, or full name if no @
    variant: str = ""  # Part after @, or empty if no @
    status: StageStatus = StageStatus.READY
    reason: str = ""
    elapsed: float | None = None
    logs: collections.deque[LogEntry] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=1000)
    )
    history: collections.deque[ExecutionHistoryEntry] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=50)
    )
    # Live snapshots for display during execution (before lock files are updated)
    live_input_snapshot: StageExplanation | None = None
    live_output_snapshot: list[OutputChange] | None = None
    # A template (``Pipeline.map``) runs nothing itself: its row reads its latest expansion
    is_template: bool = False
    instances: list[str] | None = None  # None until it expands

    def __post_init__(self) -> None:
        """Compute base_name and variant from name."""
        self.base_name, self.variant = parse_stage_name(self.name)
