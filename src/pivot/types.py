from __future__ import annotations

import enum
from typing import Any, Literal, NotRequired, TypedDict


class StageStatus(enum.StrEnum):
    """Status of a stage in the execution plan."""

    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    RAN = "ran"
    UNKNOWN = "unknown"


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

    status: Literal[StageStatus.RAN, StageStatus.SKIPPED, StageStatus.FAILED]
    reason: str
    output_lines: list[tuple[str, bool]]


class FileHash(TypedDict):
    """Hash info for a single file."""

    hash: str


class DirManifestEntry(TypedDict):
    """Entry in directory manifest."""

    relpath: str
    hash: str
    size: int
    isexec: NotRequired[bool]


class DirHash(TypedDict):
    """Hash info for a directory with full manifest."""

    hash: str
    manifest: list[DirManifestEntry]


# Non-null hash for computed hashes (from hash_dependencies, save_to_cache)
HashInfo = FileHash | DirHash

# Nullable hash for lock file storage (may be None for uncached outputs)
OutputHash = FileHash | DirHash | None


class DepEntry(TypedDict):
    """Entry in deps list for lock file storage."""

    path: str
    hash: str
    size: NotRequired[int]
    manifest: NotRequired[list[DirManifestEntry]]  # For directories


class OutEntry(TypedDict):
    """Entry in outs list for lock file storage."""

    path: str
    hash: str | None  # None for uncached outputs
    size: NotRequired[int]
    manifest: NotRequired[list[DirManifestEntry]]  # For directories


class StorageLockData(TypedDict, total=False):
    """Storage format for lock files (list-based, relative paths)."""

    code_manifest: dict[str, str]
    params: dict[str, Any]
    deps: list[DepEntry]
    outs: list[OutEntry]


class LockData(TypedDict):
    """Internal representation of stage lock data (dict-based, absolute paths)."""

    code_manifest: dict[str, str]
    params: dict[str, Any]
    dep_hashes: dict[str, HashInfo]
    output_hashes: dict[str, OutputHash]


# Type alias for output queue messages: (stage_name, line, is_stderr) or None for shutdown
OutputMessage = tuple[str, str, bool] | None

# Type alias for change detection status
ChangeType = Literal["modified", "added", "removed"]

# Type alias for CLI output format
OutputFormat = Literal["json", "md"] | None


class CodeChange(TypedDict):
    """Change info for a code component in the fingerprint."""

    key: str  # e.g., "func:helper_a", "mod:utils.helper"
    old_hash: str | None
    new_hash: str | None
    change_type: ChangeType


class ParamChange(TypedDict):
    """Change info for a parameter value."""

    key: str
    old_value: Any
    new_value: Any
    change_type: ChangeType


class DepChange(TypedDict):
    """Change info for an input dependency file."""

    path: str
    old_hash: str | None
    new_hash: str | None
    change_type: ChangeType


class StageExplanation(TypedDict):
    """Detailed explanation of why a stage would run."""

    stage_name: str
    will_run: bool
    reason: str  # High-level: "Code changed", "No previous run", etc.
    code_changes: list[CodeChange]
    param_changes: list[ParamChange]
    dep_changes: list[DepChange]
