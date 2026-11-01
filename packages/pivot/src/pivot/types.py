from __future__ import annotations

import dataclasses
import enum
import pathlib
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    NotRequired,
    Required,
    TypedDict,
    TypeGuard,
    cast,
)

if TYPE_CHECKING:
    from pivot import loaders as loaders_module
    from pivot.run_history import RunCacheEntry

# =============================================================================
# Execution Types
# =============================================================================
#
# These types define how stages execute and report their status.
#
# Stage Result Types (multiple types exist for different contexts):
#
#   StageResult       Worker → Executor communication. Contains output_lines
#                     for real-time streaming and optional timing metrics.
#                     Used internally by the ProcessPoolExecutor.
#
#   ExecutionSummary  Public API return from executor.run(). Simplified view
#                     with just status and reason per stage. Defined in
#                     executor/core.py to keep executor types together.
#
#   StageRunRecord    Historical record in LMDB. Stores input_hash for cache
#                     lookups and duration_ms for performance tracking.
#                     Defined in run_history.py.
#
#   WatchStageResult  JSON output for watch mode (--json flag). Uses string
#                     status for clean serialization. See Watch Types below.
#


class StageStatus(enum.StrEnum):
    """Status of a stage in the execution plan."""

    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CACHED = "cached"  # Was SKIPPED — stage up-to-date, outputs restored from cache
    BLOCKED = "blocked"  # Was SKIPPED — upstream dependency failed
    CANCELLED = "cancelled"  # Was SKIPPED — run cancelled by user
    FAILED = "failed"
    RAN = "ran"
    UNKNOWN = "unknown"


CompletionType = Literal[
    StageStatus.RAN,
    StageStatus.CACHED,
    StageStatus.BLOCKED,
    StageStatus.CANCELLED,
    StageStatus.FAILED,
]
"""Status values for stages that have finished execution."""

_COMPLETION_STATUSES: frozenset[StageStatus] = frozenset(
    {
        StageStatus.RAN,
        StageStatus.CACHED,
        StageStatus.BLOCKED,
        StageStatus.CANCELLED,
        StageStatus.FAILED,
    }
)


def ensure_completion_type(status: StageStatus) -> CompletionType:
    """Validate and narrow StageStatus to CompletionType.

    Raises ValueError if status is not a terminal completion status.
    """
    if status not in _COMPLETION_STATUSES:
        msg = f"Expected completion status, got '{status}'"
        raise ValueError(msg)
    return cast("CompletionType", status)


class DisplayCategory(enum.StrEnum):
    """Display category for stage results in UI.

    Maps (StageStatus, reason) pairs to consistent visual treatment across
    TUI and plain console modes.
    """

    PENDING = "pending"
    RUNNING = "running"
    WAITING_ON_LOCK = "waiting_on_lock"
    SUCCESS = "success"
    CACHED = "cached"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNKNOWN = "unknown"


# The worker logs every LossyOutputWarning as "Stage '<stage>' wrote <output> <phrase>: ...";
# the console report keys on the phrase to show those warnings uncapped.
LOSSY_OUTPUT_PHRASE = "in a form Pivot cannot read back"


def categorize_stage_result(status: StageStatus) -> DisplayCategory:
    """Map status to display category for consistent UI."""
    match status:
        case StageStatus.READY:
            return DisplayCategory.PENDING
        case StageStatus.IN_PROGRESS:
            return DisplayCategory.RUNNING
        case StageStatus.COMPLETED | StageStatus.RAN:
            return DisplayCategory.SUCCESS
        case StageStatus.FAILED:
            return DisplayCategory.FAILED
        case StageStatus.CACHED:
            return DisplayCategory.CACHED
        case StageStatus.BLOCKED:
            return DisplayCategory.BLOCKED
        case StageStatus.CANCELLED:
            return DisplayCategory.CANCELLED
        case StageStatus.UNKNOWN:
            return DisplayCategory.UNKNOWN


def parse_stage_name(name: str) -> tuple[str, str]:
    """Parse stage name into (base_name, variant). Returns (name, '') if no @."""
    if "@" in name:
        base, variant = name.split("@", 1)
        return (base, variant)
    return (name, "")


class OnError(enum.StrEnum):
    """Error handling mode."""

    FAIL = "fail"
    KEEP_GOING = "keep_going"


FileHashEntry = tuple[str, int, int, int, int, str]
"""A StateDB file-hash entry handed to a process that can write: (path, mtime_ns, size, dev,
inode, hash), the path absolute and the stat fields from ``stat()``, which follows symlinks."""


class NoCommitRecord(TypedDict, total=False):
    """What a stage's outputs hold that no lock records yet, kept under the StateDB key
    `nocommit:<stage>`. A written lock supersedes it, and its whole record is removed.

    Two features write the record, each its own keys. `code_manifest` is the manifest a
    `--no-commit` run's lock would have held, rerun marks included, `output_hashes` the outputs it
    wrote, and `reads` the objects it read by hash (`pivot.read_by_hash`), when it read any: the
    run writes these (`NoCommitRun`, `StateDB.apply_deferred_writes`), and `pivot commit` writes
    that manifest and those reads when the manifest and outputs are present and the outputs on
    disk still have these hashes. `pvt_members` maps each `*.pvt` file Pivot wrote in a directory
    output (path relative to the project root) to its content hash; the feature that keeps those
    files owns it.

    Every writer replaces only its own keys and keeps every other key, in one write transaction.
    The reader (`StateDB.get_no_commit_record`) validates each of these four keys on its own and
    returns every other key as stored, so a record that holds only one feature's keys reads as
    present, and a malformed key hides no other.
    """

    code_manifest: dict[str, str]
    output_hashes: dict[str, str]  # output identity key -> content hash
    pvt_members: dict[str, str]  # `*.pvt` path relative to the project root -> content hash
    reads: list[str]  # content hashes the run read, sorted, each once


class NoCommitRun(TypedDict):
    """The keys of `NoCommitRecord` a `--no-commit` run writes: the manifest and outputs always,
    never one alone, and `reads` when it read any. Applying it replaces all three."""

    code_manifest: dict[str, str]
    output_hashes: dict[str, str]  # output identity key -> content hash
    reads: NotRequired[list[str]]  # content hashes the run read, sorted, each once


class RemoteHashes(TypedDict):
    """The objects a run's reads fetched from one remote (`reads.Recording.fetched`): its name,
    its URL as `RemoteStorage.url` gives it, and the hashes, sorted, each once."""

    name: str
    url: str
    hashes: list[str]


class DeferredWrites(TypedDict, total=False):
    """Deferred StateDB writes from worker for coordinator to apply.

    Uses total=False so keys are only present when there's data to write.
    Stage name and output paths are passed separately by coordinator.
    """

    dep_generations: dict[str, int]  # {dep_path: generation}
    run_cache_input_hash: str
    run_cache_entry: RunCacheEntry
    # A run the stage-import check flagged: delete any run-cache entry at these input hashes (its
    # key, and the unmarked key an older pivot files under, run_history.compute_input_hash).
    run_cache_delete_input_hashes: list[str]
    increment_outputs: bool  # True → increment output generations; False/absent → skip
    file_hash_entries: list[FileHashEntry]
    no_commit_record: NoCommitRun  # a --no-commit run's keys of its `nocommit:` record
    clear_no_commit_record: bool  # a lock was written: drop the stage's --no-commit record
    # The objects the run's reads fetched, which the remote index records as `pivot pull` does
    remote_hashes: list[RemoteHashes]


class StageResult(TypedDict):
    """Result from executing a single stage."""

    status: CompletionType
    # One line the console prints in place of `reason`, which keeps its exact text for results
    # and JSONL (a traceback, for an unexpected error in a worker). Set only for a failure of
    # pivot's own I/O on state it owns (`stage_io`). Empty when a log line already
    # names the failure. Absent otherwise: the console prints `reason`, traceback included.
    display_reason: NotRequired[str]
    reason: str
    input_hash: str | None  # None only for early failures before dep hashing
    output_lines: list[tuple[str, bool]]
    metrics: NotRequired[list[tuple[str, float]]]  # (name, duration_ms) for cross-process
    deferred_writes: NotRequired[DeferredWrites]
    user_code_files: NotRequired[list[str]]  # Absolute source paths workers imported


# =============================================================================
# Hash Types
# =============================================================================
#
# Content-addressable hashes for files and directories. Used throughout
# the caching system to detect changes and identify cached artifacts.
#
#   FileHash          Simple {hash: str} for individual files.
#
#   DirHash           A directory as a lock records it: the hash of its listing, its total
#                     size and its file count. The members live in the listing
#                     (storage/listing.py), committed beside the lock and kept in the cache.
#
#   HashInfo          Union type for hashes (files or directories).
#


class FileHash(TypedDict):
    """Hash info for a single file."""

    hash: str


class DirManifestEntry(TypedDict):
    """One file of a directory listing."""

    relpath: str  # os.fsdecode form: a byte that is not UTF-8 is a surrogate escape
    hash: str
    size: int
    isexec: bool


class DirHash(TypedDict):
    """A directory: the hash of its listing (``storage/listing.py``), the sum of its files'
    sizes, and its file count."""

    hash: str
    size: int
    nfiles: int


HashInfo = FileHash | DirHash


def is_dir_hash(h: HashInfo) -> TypeGuard[DirHash]:
    """Type guard to narrow HashInfo to DirHash."""
    return "nfiles" in h


MetricValue = str | int | float | bool | None
MetricData = dict[str, MetricValue]


class OutputFormat(enum.StrEnum):
    """Output format for display commands."""

    JSON = "json"
    MD = "md"


# =============================================================================
# Lock File Types
# =============================================================================
#
# Per-stage .lock files track what was used in the last successful run.
# Comparing current state to lock data determines if a stage needs re-run.
#
# Two representations exist for different purposes:
#
#   StorageLockData   On-disk YAML format. Uses project-relative identity keys
#                     and list-based deps/outs (stable YAML output). A lock file holds a
#                     directory's hash under ``listing`` (lock._DIR_HASH_KEY); this type
#                     holds it under ``hash``, as it holds a file's.
#
#   LockData          In-memory format. Uses ArtifactIdentity keys
#                     and dict-based deps/outs (O(1) lookups by identity).
#
# Conversion happens at read/write time in storage/lock.py.
#


class ArtifactIdentity(NamedTuple):
    producer: str
    key: str | None


class ArtifactIdentityJson(TypedDict):
    producer: str
    key: str | None


def identity_to_json(identity: ArtifactIdentity) -> ArtifactIdentityJson:
    return ArtifactIdentityJson(producer=identity.producer, key=identity.key)


def identity_from_json(payload: ArtifactIdentityJson) -> ArtifactIdentity:
    return ArtifactIdentity(payload["producer"], payload["key"])


def identity_key(identity: ArtifactIdentity) -> str:
    if identity.key is None:
        return identity.producer
    return f"{identity.producer}:{identity.key}"


def identity_from_key(key: str) -> ArtifactIdentity:
    if ":" in key:
        producer, artifact_key = key.split(":", 1)
        return ArtifactIdentity(producer, artifact_key)
    return ArtifactIdentity(key, None)


class ArtifactTag(enum.StrEnum):
    INPUT = "input"
    DATA = "data"
    METRIC = "metric"
    PLOT = "plot"
    DIRECTORY = "directory"
    INCREMENTAL = "incremental"


@dataclasses.dataclass(eq=False)
class ArtifactRef:
    """One output a stage writes, or one artifact a stage reads.

    ``directory`` is set on a dependency that reads one file of a directory output: the
    directory output's identity. The ref's own identity is that identity with the file's path
    appended to the key (``stage:file.json``, or ``stage:output/file.json`` for a keyed
    output), so a lock records the hash of that file alone.
    """

    identity: ArtifactIdentity
    format: (
        loaders_module.Reader[Any] | loaders_module.Writer[Any] | loaders_module.Loader[Any, Any]
    )
    python_type: type
    tag: ArtifactTag
    directory: ArtifactIdentity | None = None


def source_identity(ref: ArtifactRef) -> ArtifactIdentity:
    """The artifact a dependency reads from: the directory output for one of its files.

    Every other ref is its own source. The DAG, artifact locks and watch mode treat a file of
    a directory as part of the directory its producer writes.
    """
    return ref.identity if ref.directory is None else ref.directory


def directory_file_identity(directory: ArtifactIdentity, path: str) -> ArtifactIdentity:
    """The identity of one file of a directory output: the file's path appended to the key.

    ``stage:path`` for a stage's return value, ``stage:output/path`` for a ``TypedDict`` field.
    """
    return ArtifactIdentity(
        directory.producer, path if directory.key is None else f"{directory.key}/{path}"
    )


def validate_artifact_identity(identity: ArtifactIdentity) -> None:
    if identity.producer.strip() == "":
        raise ValueError("Artifact producer cannot be empty")
    if pathlib.PurePath(identity.producer).is_absolute():
        raise ValueError(f"Artifact producer cannot be absolute: {identity.producer!r}")
    bad_chars = {"\\", "\0", ":"}
    if bad_chars & set(identity.producer):
        raise ValueError(
            f"Artifact producer cannot contain backslashes, null bytes, or colons: {identity.producer!r}"
        )
    for segment in identity.producer.split("/"):
        if segment in {".", ".."}:
            raise ValueError(
                f"Artifact producer cannot contain traversal segments: {identity.producer!r}"
            )

    if identity.key is None:
        return
    if pathlib.PurePath(identity.key).is_absolute():
        raise ValueError(f"Artifact key cannot be absolute: {identity.key!r}")
    if identity.key.strip() == "":
        raise ValueError("Artifact key cannot be empty")
    if bad_chars & set(identity.key):
        raise ValueError(
            f"Artifact key cannot contain backslashes, null bytes, or colons: {identity.key!r}"
        )
    for segment in identity.key.split("/"):
        if segment in {".", ".."}:
            raise ValueError(f"Artifact key cannot contain traversal segments: {identity.key!r}")


class DepEntry(TypedDict):
    """Entry in deps list for lock file storage. A directory carries ``size`` and ``nfiles``."""

    producer: str
    key: str | None
    hash: str
    size: NotRequired[int]
    nfiles: NotRequired[int]


class OutEntry(TypedDict):
    """Entry in outs list for lock file storage. A directory carries ``size`` and ``nfiles``."""

    key: str | None
    hash: str
    path: NotRequired[str]
    size: NotRequired[int]
    nfiles: NotRequired[int]


class StorageLockData(TypedDict):
    """Storage format for lock files (list-based, relative paths)."""

    schema_version: int
    code_manifest: dict[str, str]
    params: dict[str, Any]
    deps: list[DepEntry]
    outs: list[OutEntry]
    # The objects the run read by hash (``pivot.read_by_hash``), sorted, each once; absent when it
    # read none, so the lock of a stage that reads nothing by hash is what it was before.
    reads: NotRequired[list[str]]


class LockData(TypedDict):
    """Internal representation of stage lock data (dict-based, identity-keyed)."""

    code_manifest: dict[str, str]
    params: dict[str, Any]
    dep_hashes: dict[ArtifactIdentity, HashInfo]
    output_hashes: dict[ArtifactIdentity, HashInfo]
    output_paths: NotRequired[dict[ArtifactIdentity, str]]
    reads: NotRequired[list[str]]  # content hashes the run read by hash, sorted, each once
    # Present for a lock an older pivot wrote: its directories' listings were inline and are in
    # the cache only (``StageLock.read``); a listing file beside it is another lock's
    # (``lock.committed_listing``).
    older_format: NotRequired[bool]


class OutputMessageKind(enum.StrEnum):
    """Discriminant for worker→engine output messages."""

    LOG = "log"
    STATE = "state"


class LogMessage(TypedDict):
    """Log line from worker process."""

    kind: Literal[OutputMessageKind.LOG]
    stage: str
    line: str
    is_stderr: bool


class StateChange(TypedDict):
    """Execution state transition from worker process."""

    kind: Literal[OutputMessageKind.STATE]
    stage: str
    state: Literal["waiting_on_lock", "running"]


OutputMessage = LogMessage | StateChange | None


# =============================================================================
# Change Detection Types
# =============================================================================
#
# Used by `pivot explain` to show exactly what changed since the last run.
# Each change type tracks old/new values to help debug unexpected re-runs.
#


class ChangeType(enum.StrEnum):
    """Status for change detection."""

    MODIFIED = "modified"
    ADDED = "added"
    REMOVED = "removed"


class CodeChange(TypedDict):
    """Change info for a code component in the fingerprint."""

    key: str  # e.g., "func:helper_a", "const:settings.FACTOR"
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

    identity: ArtifactIdentity
    old_hash: str | None
    new_hash: str | None
    change_type: ChangeType


class OutputChange(TypedDict):
    """Change info for an output artifact."""

    identity: ArtifactIdentity
    old_hash: str | None
    new_hash: str | None
    change_type: ChangeType | None  # None means unchanged
    output_type: Literal["out", "metric", "plot"]


class Recovery(enum.StrEnum):
    """What brings a stale stage up to date without running it. Carried beside the reason
    (`ChangeDecision`, `StageExplanation`, `PipelineStatusInfo`) so `pivot status` chooses its
    `pivot pull` suggestion from it, never from the reason's text. Present only when it recovers
    every reason the stage is stale, its stale upstream stages included; absent when only a run
    does."""

    PULL_OUTPUTS = "pull_outputs"  # its lock is current; `pivot pull` restores missing outputs
    PULL_READS = "pull_reads"  # `pivot pull` fetches objects it read by hash that the cache lacks
    UPSTREAM = "upstream"  # current on its own: up to date once its stale upstream stages are


class UnavailableReads(TypedDict):
    """Why objects a lock or run-cache entry records its run read by hash are neither in the
    cache nor known to be on the default remote (`reads.unavailable`)."""

    reason: str
    pullable: bool  # a usable remote may hold them, so `pivot pull` fetches them


class ChangeDecision(TypedDict, total=False):
    """Result of skip detection check_stage().

    In fast mode (explain=False), detail fields are omitted (not present).
    In explain mode (explain=True), they contain the full diff information.
    `recovery` is present when the stage is changed only for a reason `pivot pull` repairs.
    """

    changed: Required[bool]
    reason: Required[str]
    code_changes: list[CodeChange]
    run_cache_allowed: Required[bool]
    param_changes: list[ParamChange]
    dep_changes: list[DepChange]
    recovery: Recovery


class StageExplanation(TypedDict, total=False):
    """Detailed explanation of why a stage would run."""

    stage_name: Required[str]
    will_run: Required[bool]
    is_forced: Required[bool]
    reason: Required[str]  # "Code changed", "No previous run", "forced", etc.
    code_changes: Required[list[CodeChange]]
    param_changes: Required[list[ParamChange]]
    dep_changes: Required[list[DepChange]]
    upstream_stale: Required[list[str]]
    recovery: Recovery  # what brings it up to date without running it, when something does


# =============================================================================
# Remote Cache Types
# =============================================================================


class TransferResult(TypedDict):
    """Result of a single file transfer to/from remote."""

    hash: str
    success: bool
    error: NotRequired[str]
    # Present when the transfer failed with RemoteUnreachableError: no connection to the remote
    # could be made, so every other transfer to it would fail the same way
    connection_failed: NotRequired[Literal[True]]


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

    remotes: dict[str, str]  # {remote_name: s3:// or gs:// url}
    default_remote: str


# =============================================================================
# Status Types
# =============================================================================


class PipelineStatus(enum.StrEnum):
    """Pipeline stage status for pivot status output."""

    CACHED = "cached"
    STALE = "stale"
    # A stage mapped over a directory output (``Pipeline.map``) whose producer is not current,
    # so the command cannot know its instances; its reason says what it waits for
    NOT_EXPANDED = "not_expanded"
    # An instance of a mapped stage whose key left its producer's listing: no stage now, its
    # lock, listings and outputs go in the next run that expands the mapped stage
    ORPHANED = "orphaned"


class TrackedFileStatus(enum.StrEnum):
    """Status of a tracked file."""

    CLEAN = "clean"
    MODIFIED = "modified"
    MISSING = "missing"


class ImportCheckStatus(enum.StrEnum):
    """Update check result for an imported artifact."""

    UP_TO_DATE = "up_to_date"
    UPDATE_AVAILABLE = "update_available"
    ERROR = "error"


class PipelineStatusInfo(TypedDict):
    """Status info for a single stage in pivot status output."""

    name: str
    status: PipelineStatus
    reason: str
    upstream_stale: list[str]
    recovery: NotRequired[Recovery]  # what brings it up to date without running it, if anything


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


class ImportStatusInfo(TypedDict):
    """One imported artifact's update check, for pivot status --check-imports output.

    The revisions are the first 8 characters of each commit. A field that does not apply to
    the status is None (null in JSON): both revisions for up_to_date and error, error for the
    others.
    """

    path: str
    status: ImportCheckStatus
    current_rev: str | None
    latest_rev: str | None
    error: str | None


class StatusOutput(TypedDict, total=False):
    """JSON output structure for pivot status command."""

    stages: list[PipelineStatusInfo]
    tracked_files: list[TrackedFileInfo]
    remote: RemoteSyncInfo
    imports: list[ImportStatusInfo]
    suggestions: list[str]


class DepChangeJson(TypedDict):
    """A DepChange as JSON: the identity as ``{producer, key}``, as the RPC sends it."""

    identity: ArtifactIdentityJson
    old_hash: str | None
    new_hash: str | None
    change_type: ChangeType


def dep_change_to_json(change: DepChange) -> DepChangeJson:
    return DepChangeJson(
        identity=identity_to_json(change["identity"]),
        old_hash=change["old_hash"],
        new_hash=change["new_hash"],
        change_type=change["change_type"],
    )


class ExplainStageJson(TypedDict):
    """JSON-serializable stage explanation for status --explain --json output."""

    name: str
    status: Literal["stale", "cached", "not_expanded", "orphaned"]
    reason: str
    will_run: bool
    is_forced: bool
    code_changes: list[CodeChange]
    param_changes: list[ParamChange]
    dep_changes: list[DepChangeJson]
    upstream_stale: list[str]
    recovery: NotRequired[Recovery]  # what brings it up to date without running it, if anything


class ExplainOutput(TypedDict, total=False):
    """JSON output structure for pivot status --explain command."""

    stages: list[ExplainStageJson]
    tracked_files: list[TrackedFileInfo]
    remote: RemoteSyncInfo
    imports: list[ImportStatusInfo]
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
    change_type: ChangeType
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
#
# Messages sent from executor/watch engine to the Textual TUI for display.
# Sent via multiprocessing.Queue for cross-process communication.
#
# Message flow:
#   Worker process → Queue → TUI event loop → Rich panel update
#


class TuiMessageType(enum.StrEnum):
    """Type of TUI message."""

    LOG = "log"
    STATUS = "status"
    WATCH = "watch"
    RELOAD = "reload"
    EXPANDED = "expanded"


class WatchStatus(enum.StrEnum):
    """Status of the watch engine."""

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
    timestamp: float


class TuiStatusMessage(TypedDict):
    """Stage status update for TUI display."""

    type: Literal[TuiMessageType.STATUS]
    stage: str
    index: int
    total: int
    status: StageStatus
    reason: str
    elapsed: float | None
    run_id: str
    explanation: NotRequired[StageExplanation | None]
    output_summary: NotRequired[list[OutputChange] | None]


class TuiWatchMessage(TypedDict):
    """Watch engine status update for TUI display."""

    type: Literal[TuiMessageType.WATCH]
    status: WatchStatus
    message: str


class TuiReloadMessage(TypedDict):
    """Registry reload notification for TUI display.

    ``error`` is set when the edited pipeline is invalid; ``stages`` then lists the stages of
    the pipeline still loaded.
    """

    type: Literal[TuiMessageType.RELOAD]
    stages: list[str]
    stages_added: list[str]
    stages_removed: list[str]
    stages_modified: list[str]
    error: str | None


# =============================================================================
# Watch Mode JSONL Events (--json output)
# =============================================================================
#
# Structured events emitted by `pivot run --watch --json` for programmatic
# consumption. Each line is a complete JSON object (JSONL format).
#
# Event sequence during watch:
#   1. WatchStatusEvent      "Watching for changes..."
#   2. WatchFilesChangedEvent   Files that triggered re-run
#   3. WatchAffectedStagesEvent Stages that will execute
#   4. WatchExecutionResultEvent   Results of each stage
#   ... (repeat from step 1)
#


class WatchEventType(enum.StrEnum):
    """Type of watch mode JSONL event."""

    STATUS = "status"
    FILES_CHANGED = "files_changed"
    AFFECTED_STAGES = "affected_stages"
    EXECUTION_RESULT = "execution_result"


class WatchStatusEvent(TypedDict):
    """Status message event."""

    type: Literal[WatchEventType.STATUS]
    message: str
    is_error: bool


class WatchFilesChangedEvent(TypedDict):
    """File change detection event."""

    type: Literal[WatchEventType.FILES_CHANGED]
    paths: list[str]
    code_changed: bool


class WatchAffectedStagesEvent(TypedDict):
    """Affected stages event."""

    type: Literal[WatchEventType.AFFECTED_STAGES]
    stages: list[str]
    count: int


class WatchStageResult(TypedDict):
    """Result for a single stage in watch mode execution."""

    status: str
    reason: str


class WatchExecutionResultEvent(TypedDict):
    """Execution result event."""

    type: Literal[WatchEventType.EXECUTION_RESULT]
    stages: dict[str, WatchStageResult]


WatchJsonEvent = (
    WatchStatusEvent | WatchFilesChangedEvent | WatchAffectedStagesEvent | WatchExecutionResultEvent
)


# =============================================================================
# Run Mode JSONL Events (--json output)
# =============================================================================
#
# Structured events emitted by `pivot run --json` for programmatic consumption.
# Each line is a complete JSON object (JSONL format) with flush=True.
#
# Event sequence during run:
#   1. SchemaVersionEvent      Schema version for forward compatibility
#   2. StageStartEvent         Before each stage executes
#   3. StageCompleteEvent      After each stage finishes
#   ... (repeat 2-3 for each stage)
#   StagesExpandedEvent        When a mapped stage (``Pipeline.map``) adds its instances
#   4. ExecutionResultEvent    Final summary
#


class RunEventType(enum.StrEnum):
    """Type of run mode JSONL event."""

    SCHEMA_VERSION = "schema_version"
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"
    STAGES_EXPANDED = "stages_expanded"
    EXECUTION_RESULT = "execution_result"


class SchemaVersionEvent(TypedDict):
    """Schema version event for forward compatibility."""

    type: Literal[RunEventType.SCHEMA_VERSION]
    version: int


class StageStartEvent(TypedDict):
    """Event emitted before a stage begins execution."""

    type: Literal[RunEventType.STAGE_START]
    stage: str
    index: int
    total: int
    timestamp: str  # ISO 8601 UTC


class StageCompleteEvent(TypedDict):
    """Event emitted after a stage finishes execution."""

    type: Literal[RunEventType.STAGE_COMPLETE]
    stage: str
    status: Literal[
        StageStatus.RAN,
        StageStatus.CACHED,
        StageStatus.BLOCKED,
        StageStatus.CANCELLED,
        StageStatus.FAILED,
    ]
    reason: str
    duration_ms: float
    timestamp: str  # ISO 8601 UTC


class StagesExpandedEvent(TypedDict):
    """A mapped stage's template (``Pipeline.map``) added its instances to the run: one stage
    per file of the directory it is mapped over, ``<template>@<key>``, in key order. ``removed``
    names each instance whose key left that directory's listing and which the run removed (its
    lock, listings and outputs), sorted; absent when it removed none."""

    type: Literal[RunEventType.STAGES_EXPANDED]
    template: str
    stages: list[str]
    removed: NotRequired[list[str]]


class ExecutionResultEvent(TypedDict):
    """Final summary event after pipeline completes."""

    type: Literal[RunEventType.EXECUTION_RESULT]
    ran: int
    skipped: int
    failed: int
    failed_stages: list[str]
    total_duration_ms: float
    timestamp: str  # ISO 8601 UTC


RunJsonEvent = (
    SchemaVersionEvent
    | StageStartEvent
    | StageCompleteEvent
    | StagesExpandedEvent
    | ExecutionResultEvent
)


# =============================================================================
# Agent RPC Types
# =============================================================================
#
# Types for the JSON-RPC agent control plane. Enables AI agents to control
# pipeline execution via Unix socket while researcher watches the TUI.
#


class AgentState(enum.StrEnum):
    """State of the agent server."""

    IDLE = "idle"
    WATCHING = "watching"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRunParams(TypedDict, total=False):
    """Parameters for agent run() RPC method."""

    stages: list[str]
    force: bool


class AgentRunStartResult(TypedDict):
    """Result returned immediately when run() is called."""

    run_id: str
    status: Literal["started"]
    stages_queued: list[str]


class AgentRunRejection(TypedDict, total=False):
    """Reason an agent run request was rejected."""

    reason: Required[Literal["not_ready", "queue_full", "busy"]]
    current_state: Required[str]
    current_run_id: str | None


class AgentStatusResult(TypedDict, total=False):
    """Result of status() RPC method."""

    state: Required[str]  # EngineState.value or AgentState.value
    run_id: str
    stages_completed: list[str]
    stages_pending: list[str]
    ran: int
    skipped: int
    failed: int


class AgentCancelResult(TypedDict, total=False):
    """Result of cancel() RPC method."""

    cancelled: bool
    stage_interrupted: str


class AgentStageInfo(TypedDict):
    """Stage info returned by stages() RPC method."""

    name: str
    deps: list[ArtifactIdentityJson]
    outs: list[ArtifactIdentityJson]


class AgentStagesResult(TypedDict):
    """Result of stages() RPC method."""

    stages: list[AgentStageInfo]


# A stage function; compose derives its deps from handle arguments and its outputs
# from the return annotation.
type StageFunc = Callable[..., Any]
