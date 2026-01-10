from typing import override


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

    @override
    def get_suggestion(self) -> str:
        return "Ensure the file exists or is produced by another stage"


class StageNotFoundError(DAGError):
    """Raised when a requested stage doesn't exist."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot list' to see available stages"


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
    """Raised when a .pvt tracked file is missing (user should run pivot checkout)."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot checkout' to restore from cache"


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
