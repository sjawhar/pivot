"""Pivot exceptions."""

import enum


class PivotError(Exception):
    """Base exception for Pivot errors."""

    pass


class ValidationError(PivotError):
    """Raised when stage validation fails."""

    pass


class ValidationMode(enum.StrEnum):
    """Validation strictness levels."""

    ERROR = "error"  # Raise exception on validation failure
    WARN = "warn"  # Log warning, allow registration


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
