from __future__ import annotations

import collections
import dataclasses
from typing import TYPE_CHECKING, NamedTuple, Protocol

from pivot.types import OutputChange, StageExplanation, StageStatus, parse_stage_name

if TYPE_CHECKING:
    from pivot.registry import RegistryStageInfo


class StageDataProvider(Protocol):
    """Protocol for TUI to look up stage metadata without CLI context.

    Decouples the TUI from pivot.cli.helpers by defining the two
    operations the TUI actually needs from the registry.
    """

    def get_stage(self, name: str) -> RegistryStageInfo:
        """Look up stage metadata by name. Raises KeyError if not found."""
        ...

    def ensure_fingerprint(self, name: str) -> dict[str, str]:
        """Compute/return cached code fingerprint for a stage."""
        ...


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

    def __post_init__(self) -> None:
        """Compute base_name and variant from name."""
        self.base_name, self.variant = parse_stage_name(self.name)
