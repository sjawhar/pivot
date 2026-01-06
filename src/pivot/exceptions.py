class PivotError(Exception):
    """Base exception for Pivot errors."""

    pass


class ValidationError(PivotError):
    """Raised when stage validation fails."""

    pass


class OutputDuplicationError(ValidationError):
    """Raised when two stages produce the same output."""

    pass


class OverlappingOutputPathsError(ValidationError):
    """Raised when output paths overlap (one is parent/child of another)."""

    pass


class DAGError(PivotError):
    """Base class for DAG-related errors."""

    pass


class CyclicGraphError(DAGError):
    """Raised when DAG contains cycles."""

    pass


class DependencyNotFoundError(DAGError):
    """Raised when a dependency doesn't exist."""

    pass


class StageNotFoundError(DAGError):
    """Raised when a requested stage doesn't exist."""

    pass


class StageAlreadyRunningError(PivotError):
    """Raised when a stage is already being executed by another process."""

    pass


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


class PathTraversalError(CacheError):
    """Raised when a manifest contains path traversal attempt."""

    pass


class UncachedIncrementalOutputError(CacheError):
    """Raised when an IncrementalOut file exists but is not in cache."""

    pass
