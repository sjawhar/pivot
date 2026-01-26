from difflib import get_close_matches
from typing import override

# Fuzzy matching constants
_FUZZY_CUTOFF = 0.6
_FUZZY_MIN_LENGTH = 3


def _fuzzy_suggest(query: str, candidates: list[str]) -> str | None:
    """Return best fuzzy match if found, else None."""
    if not candidates or len(query) < _FUZZY_MIN_LENGTH:
        return None
    matches = get_close_matches(query, candidates, n=1, cutoff=_FUZZY_CUTOFF)
    return matches[0] if matches else None


class PivotError(Exception):
    """Base exception for Pivot errors."""

    def format_user_message(self) -> str:
        """Format a user-friendly error message."""
        return str(self)

    def get_suggestion(self) -> str | None:
        """Return actionable suggestion for resolving the error."""
        return None


class ValidationError(PivotError):
    """Raised when stage validation fails."""

    pass


class StageDefinitionError(ValidationError):
    """Raised when a stage function has invalid annotations (return type, deps, etc)."""

    pass


class SecurityValidationError(PivotError):
    """Raised for security-sensitive validation failures (path traversal, injection attacks).

    Inherits from PivotError (not ValidationError) to ensure security errors
    are never accidentally caught by broad ValidationError handlers.
    """

    pass


class OutputDuplicationError(ValidationError):
    """Raised when two stages produce the same output."""

    pass


class OverlappingOutputPathsError(ValidationError):
    """Raised when output paths overlap (one is parent/child of another)."""

    pass


class InvalidPathError(ValidationError):
    """Raised when a path is invalid (e.g., resolves outside project root)."""

    pass


class DAGError(PivotError):
    """Base class for DAG-related errors."""

    pass


class CyclicGraphError(DAGError):
    """Raised when DAG contains cycles."""

    @override
    def get_suggestion(self) -> str:
        return "Check stage dependencies for circular references"


class DependencyNotFoundError(DAGError):
    """Raised when a dependency doesn't exist."""

    _stage: str
    _dep: str
    _available: list[str]

    def __init__(
        self,
        stage: str,
        dep: str,
        available_outputs: list[str] | None = None,
    ) -> None:
        self._stage = stage
        self._dep = dep
        self._available = available_outputs or []
        super().__init__(
            f"Stage '{stage}' depends on '{dep}' which is not produced by any stage and does not exist on disk"
        )

    @override
    def format_user_message(self) -> str:
        msg = str(self)
        if match := _fuzzy_suggest(self._dep, self._available):
            msg += f"\n  Did you mean: '{match}'?"
        return msg

    @override
    def get_suggestion(self) -> str:
        return "Ensure the file exists or is produced by another stage"

    @override
    def __reduce__(self) -> tuple[type, tuple[str, str, list[str]]]:
        return (self.__class__, (self._stage, self._dep, self._available))


class StageNotFoundError(DAGError):
    """Raised when a requested stage doesn't exist."""

    _unknown: list[str]
    _available: list[str]

    def __init__(
        self,
        unknown_stages: list[str],
        available_stages: list[str] | None = None,
    ) -> None:
        self._unknown = unknown_stages
        self._available = available_stages or []
        super().__init__(f"Unknown stage(s): {', '.join(unknown_stages)}")

    @override
    def format_user_message(self) -> str:
        msg = str(self)
        if self._available:
            # Limit to 3 suggestions to avoid overwhelming output
            suggestions = [
                f"'{u}' -> '{m}'"
                for u in self._unknown[:3]
                if (m := _fuzzy_suggest(u, self._available))
            ]
            if suggestions:
                msg += f"\n  Did you mean: {', '.join(suggestions)}?"
                if len(self._unknown) > 3:
                    msg += f"\n  (showing first 3 of {len(self._unknown)} unknown stages)"
        return msg

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot list' to see available stages"

    @override
    def __reduce__(self) -> tuple[type, tuple[list[str], list[str]]]:
        return (self.__class__, (self._unknown, self._available))


class StageAlreadyRunningError(PivotError):
    """Raised when a stage is already being executed by another process."""

    @override
    def get_suggestion(self) -> str:
        return "Wait for the other process to finish or remove stale lock files"


class ExecutionError(PivotError):
    """Raised when pipeline execution fails."""

    pass


class DVCCompatError(PivotError):
    """Base class for DVC compatibility errors."""

    pass


class ExportError(DVCCompatError):
    """Raised when stage export to DVC format fails."""

    pass


class DVCImportError(DVCCompatError):
    """Raised when dvc.yaml import fails."""

    pass


class CacheError(PivotError):
    """Base class for cache-related errors."""

    pass


class OutputMissingError(CacheError):
    """Raised when a stage did not produce a declared output."""

    pass


class CacheRestoreError(CacheError):
    """Raised when restoring outputs from cache fails."""

    pass


class UncachedIncrementalOutputError(CacheError):
    """Raised when an IncrementalOut file exists but is not in cache."""

    pass


class ParamsError(PivotError):
    """Raised when parameter validation or loading fails."""

    pass


class PlotsError(PivotError):
    """Raised when plot processing fails."""

    pass


class TrackedFileMissingError(CacheError):
    """Raised when .pvt tracked files are missing (user should run pivot checkout)."""

    _missing: list[str]
    _checkout_attempted: bool

    def __init__(self, missing_files: list[str], checkout_attempted: bool = False) -> None:
        self._missing = missing_files
        self._checkout_attempted = checkout_attempted
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        missing_list = "\n".join(f"  - {p}" for p in self._missing)
        if self._checkout_attempted:
            return (
                f"The following tracked files are missing:\n{missing_list}\n\n"
                + "These files are not in local cache."
            )
        return (
            f"The following tracked files are missing:\n{missing_list}\n\n"
            + "These files are tracked by Pivot but don't exist on disk."
        )

    @override
    def get_suggestion(self) -> str:
        if self._checkout_attempted:
            return "Run 'pivot pull' to fetch from remote storage"
        if len(self._missing) == 1:
            return f"Run 'pivot checkout {self._missing[0]}' to restore, or 'pivot checkout --only-missing' to restore all missing files"
        return "Run 'pivot checkout --only-missing' to restore missing files, or 'pivot run --checkout-missing' to restore and run"

    @override
    def __reduce__(self) -> tuple[type, tuple[list[str], bool]]:
        return (self.__class__, (self._missing, self._checkout_attempted))


class GetError(PivotError):
    """Base class for get command errors."""

    pass


class RevisionNotFoundError(GetError):
    """Raised when git revision cannot be resolved."""

    pass


class TargetNotFoundError(GetError):
    """Raised when target is not found at specified revision."""

    pass


class CacheMissError(GetError):
    """Raised when file cannot be retrieved from cache, git, or remote."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot pull' to fetch from remote, or re-run the stage to regenerate"


class RemoteError(PivotError):
    """Base class for remote storage errors."""

    pass


class RemoteNotFoundError(RemoteError):
    """Raised when a named remote doesn't exist in configuration."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot remote list' to see available remotes"


class RemoteConnectionError(RemoteError):
    """Raised when connection to remote storage fails."""

    @override
    def get_suggestion(self) -> str:
        return "Check network connection and remote credentials"


class InvalidRemoteURLError(RemoteError):
    """Raised when remote URL is malformed or uses unsupported scheme."""

    @override
    def get_suggestion(self) -> str:
        return "Use format: s3://bucket/path, gs://bucket/path, or /local/path"


class RemoteTransferError(RemoteError):
    """Raised when file transfer to/from remote fails."""

    @override
    def get_suggestion(self) -> str:
        return "Check network connection and remote permissions"


class RemoteFetchError(RemoteError):
    """Raised when fetching from remote fails."""

    @override
    def get_suggestion(self) -> str:
        return "Check network connection and verify the file exists on remote"


class RemoteNotConfiguredError(RemoteError):
    """Raised when no remote is configured."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot config set remotes.<name> <url>' to configure a remote"


class ConfigError(PivotError):
    """Base class for configuration errors."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when config value fails validation."""

    pass


class ConfigKeyError(ConfigError):
    """Raised when config key is unknown or invalid."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot config list' to see available config keys"


class InitError(PivotError):
    """Base class for initialization errors."""

    pass


class AlreadyInitializedError(InitError):
    """Raised when project is already initialized."""

    @override
    def get_suggestion(self) -> str:
        return "Use --force to reinitialize"
