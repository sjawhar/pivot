"""Shared types for Pivot modules."""

from __future__ import annotations

import enum
from typing import Any, Literal, TypedDict


class StageStatus(enum.StrEnum):
    """Status of a stage in the execution plan."""

    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    RAN = "ran"


class StageDisplayStatus(enum.StrEnum):
    """Display status for stage progress output."""

    CHECKING = "checking"
    RUNNING = "running"
    WAITING = "waiting"


class OnError(enum.StrEnum):
    """Error handling mode."""

    FAIL = "fail"
    KEEP_GOING = "keep_going"
    IGNORE = "ignore"


class StageResult(TypedDict):
    """Result from executing a single stage."""

    status: Literal["ran", "skipped", "failed"]
    reason: str
    output_lines: list[tuple[str, bool]]


class LockData(TypedDict, total=False):
    """Data stored in stage lock files."""

    code_manifest: dict[str, str]
    params: dict[str, Any]
    dep_hashes: dict[str, str]


# Type alias for output queue messages: (stage_name, line, is_stderr) or None for shutdown
OutputMessage = tuple[str, str, bool] | None
