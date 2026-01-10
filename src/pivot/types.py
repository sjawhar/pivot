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

# Metric values are JSON primitives after flattening (nested dicts become dot-separated keys)
MetricValue = str | int | float | bool | None

# Flattened metric data: {key: value} where keys are dot-separated paths
MetricData = dict[str, MetricValue]


class OutputFormat(enum.StrEnum):
    """Output format for display commands."""

    JSON = "json"
    MD = "md"


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
    # Stored at execution time for --no-commit mode (used by commit to record correct generations)
    dep_generations: dict[str, int]


class LockData(TypedDict):
    """Internal representation of stage lock data (dict-based, absolute paths)."""

    code_manifest: dict[str, str]
    params: dict[str, Any]
    dep_hashes: dict[str, HashInfo]
    output_hashes: dict[str, OutputHash]
    # Stored at execution time for --no-commit mode (used by commit to record correct generations)
    dep_generations: NotRequired[dict[str, int]]


# Type alias for output queue messages: (stage_name, line, is_stderr) or None for shutdown
OutputMessage = tuple[str, str, bool] | None


class ChangeType(enum.StrEnum):
    """Status for change detection."""

    MODIFIED = "modified"
    ADDED = "added"
    REMOVED = "removed"


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
    is_forced: bool  # True if stage is forced to run regardless of changes
    reason: str  # High-level: "Code changed", "No previous run", "forced", etc.
    code_changes: list[CodeChange]
    param_changes: list[ParamChange]
    dep_changes: list[DepChange]


# Remote cache types


class TransferResult(TypedDict):
    """Result of a single file transfer to/from remote."""

    hash: str
    success: bool
    error: NotRequired[str]


class TransferSummary(TypedDict):
    """Summary of push/pull operation."""

    transferred: int
    skipped: int
    failed: int
    errors: list[str]


class RemoteStatus(TypedDict):
    """Status comparison between local and remote cache."""

    local_only: set[str]
    remote_only: set[str]
    common: set[str]


class RawPivotConfig(TypedDict, total=False):
    """Raw config file structure (.pivot/config.yaml)."""

    remotes: dict[str, str]  # {remote_name: s3_url}
    default_remote: str


# =============================================================================
# Status Types
# =============================================================================


class PipelineStatus(enum.StrEnum):
    """Pipeline stage status for pivot status output."""

    CACHED = "cached"
    STALE = "stale"


class TrackedFileStatus(enum.StrEnum):
    """Status of a tracked file."""

    CLEAN = "clean"
    MODIFIED = "modified"
    MISSING = "missing"


class PipelineStatusInfo(TypedDict):
    """Status info for a single stage in pivot status output."""

    name: str
    status: PipelineStatus
    reason: str
    upstream_stale: list[str]


class TrackedFileInfo(TypedDict):
    """Status of a tracked file from pivot track."""

    path: str
    status: TrackedFileStatus
    size: int


class RemoteSyncInfo(TypedDict):
    """Remote sync status for pivot status output."""

    name: str
    url: str
    push_count: int
    pull_count: int


class StatusOutput(TypedDict, total=False):
    """JSON output structure for pivot status command."""

    stages: list[PipelineStatusInfo]
    tracked_files: list[TrackedFileInfo]
    remote: RemoteSyncInfo
    suggestions: list[str]


# =============================================================================
# Data Diff Types
# =============================================================================


class DataFileFormat(enum.StrEnum):
    """Supported data file formats for pivot data diff."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    UNKNOWN = "unknown"


class SchemaChange(TypedDict):
    """Change info for a column in a data file schema."""

    column: str
    old_dtype: str | None
    new_dtype: str | None
    change_type: ChangeType


class RowChange(TypedDict):
    """Change info for a row in a data file."""

    key: str | int  # Key column value or row index
    change_type: ChangeType
    old_values: dict[str, Any] | None
    new_values: dict[str, Any] | None


class DataDiffResult(TypedDict):
    """Result of comparing two data files."""

    path: str
    old_rows: int | None
    new_rows: int | None
    old_cols: list[str] | None
    new_cols: list[str] | None
    schema_changes: list[SchemaChange]
    row_changes: list[RowChange]
    reorder_only: bool  # True if same content, different row order
    truncated: bool  # True if large file, showing sample
    summary_only: bool  # True if no row-level diff available


# =============================================================================
# TUI Message Types
# =============================================================================


class DisplayMode(enum.StrEnum):
    """Display mode for pivot run output."""

    TUI = "tui"
    PLAIN = "plain"


class TuiMessageType(enum.StrEnum):
    """Type of TUI message."""

    LOG = "log"
    STATUS = "status"
    REACTIVE = "reactive"
    RELOAD = "reload"


class ReactiveStatus(enum.StrEnum):
    """Status of the reactive engine."""

    WAITING = "waiting"
    RESTARTING = "restarting"
    DETECTING = "detecting"
    ERROR = "error"


class TuiLogMessage(TypedDict):
    """Log line from worker process for TUI display."""

    type: Literal[TuiMessageType.LOG]
    stage: str
    line: str
    is_stderr: bool


class TuiStatusMessage(TypedDict):
    """Stage status update for TUI display."""

    type: Literal[TuiMessageType.STATUS]
    stage: str
    index: int
    total: int
    status: StageStatus
    reason: str
    elapsed: float | None


class TuiReactiveMessage(TypedDict):
    """Reactive engine status update for TUI display."""

    type: Literal[TuiMessageType.REACTIVE]
    status: ReactiveStatus
    message: str


class TuiReloadMessage(TypedDict):
    """Registry reload notification for TUI display."""

    type: Literal[TuiMessageType.RELOAD]
    stages: list[str]


TuiMessage = TuiLogMessage | TuiStatusMessage | TuiReactiveMessage | TuiReloadMessage | None


# =============================================================================
# Reactive JSONL Event Types (for --json output)
# =============================================================================


class ReactiveEventType(enum.StrEnum):
    """Type of reactive JSONL event."""

    STATUS = "status"
    FILES_CHANGED = "files_changed"
    AFFECTED_STAGES = "affected_stages"
    EXECUTION_RESULT = "execution_result"


class ReactiveStatusEvent(TypedDict):
    """Status message event."""

    type: Literal[ReactiveEventType.STATUS]
    message: str
    is_error: bool


class ReactiveFilesChangedEvent(TypedDict):
    """File change detection event."""

    type: Literal[ReactiveEventType.FILES_CHANGED]
    paths: list[str]
    code_changed: bool


class ReactiveAffectedStagesEvent(TypedDict):
    """Affected stages event."""

    type: Literal[ReactiveEventType.AFFECTED_STAGES]
    stages: list[str]
    count: int


class ReactiveStageResult(TypedDict):
    """Result for a single stage in reactive execution."""

    status: str
    reason: str


class ReactiveExecutionResultEvent(TypedDict):
    """Execution result event."""

    type: Literal[ReactiveEventType.EXECUTION_RESULT]
    stages: dict[str, ReactiveStageResult]


ReactiveJsonEvent = (
    ReactiveStatusEvent
    | ReactiveFilesChangedEvent
    | ReactiveAffectedStagesEvent
    | ReactiveExecutionResultEvent
)
