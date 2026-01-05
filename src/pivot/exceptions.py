"""Pivot exceptions."""

import enum

__all__ = [
    "PivotError",
    "ValidationError",
    "ValidationMode",
    "OutputDuplicationError",
    "OverlappingOutputPathsError",
    "DAGError",
    "CyclicGraphError",
    "DependencyNotFoundError",
]


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
