import click

from pivot import exceptions
from pivot.cli import errors as cli_errors

# =============================================================================
# handle_pivot_error Tests
# =============================================================================


def test_handle_pivot_error_formats_message() -> None:
    """handle_pivot_error creates ClickException with formatted message."""
    error = exceptions.StageNotFoundError("Unknown stage(s): foo")

    result = cli_errors.handle_pivot_error(error)

    assert isinstance(result, click.ClickException)
    assert "Unknown stage(s): foo" in result.format_message()


def test_handle_pivot_error_includes_suggestion() -> None:
    """handle_pivot_error includes suggestion in message."""
    error = exceptions.StageNotFoundError("Unknown stage(s): foo")

    result = cli_errors.handle_pivot_error(error)

    assert "Tip:" in result.format_message()
    assert "pivot list" in result.format_message()


def test_handle_pivot_error_without_suggestion() -> None:
    """handle_pivot_error works for errors without suggestions."""
    error = exceptions.ValidationError("Some validation error")

    result = cli_errors.handle_pivot_error(error)

    assert isinstance(result, click.ClickException)
    assert "Some validation error" in result.format_message()
    assert "Tip:" not in result.format_message()


# =============================================================================
# Exception Suggestion Tests
# =============================================================================


def test_stage_not_found_error_suggestion() -> None:
    """StageNotFoundError provides suggestion to run pivot list."""
    error = exceptions.StageNotFoundError("Unknown stage")

    assert error.get_suggestion() == "Run 'pivot list' to see available stages"


def test_dependency_not_found_error_suggestion() -> None:
    """DependencyNotFoundError provides appropriate suggestion."""
    error = exceptions.DependencyNotFoundError("Missing dep")

    assert "file exists" in error.get_suggestion()
    assert "produced by another stage" in error.get_suggestion()


def test_cyclic_graph_error_suggestion() -> None:
    """CyclicGraphError provides suggestion about dependencies."""
    error = exceptions.CyclicGraphError("Cycle detected")

    assert "dependencies" in error.get_suggestion()
    assert "circular" in error.get_suggestion()


def test_cache_miss_error_suggestion() -> None:
    """CacheMissError provides suggestions for pull or regenerate."""
    error = exceptions.CacheMissError("Not in cache")

    assert "pivot pull" in error.get_suggestion()
    assert "re-run" in error.get_suggestion()


def test_tracked_file_missing_error_suggestion() -> None:
    """TrackedFileMissingError suggests running pivot checkout."""
    error = exceptions.TrackedFileMissingError("File missing")

    assert "pivot checkout" in error.get_suggestion()


def test_remote_not_configured_error_suggestion() -> None:
    """RemoteNotConfiguredError suggests adding a remote."""
    error = exceptions.RemoteNotConfiguredError("No remote")

    assert "pivot remote add" in error.get_suggestion()


def test_remote_not_found_error_suggestion() -> None:
    """RemoteNotFoundError suggests listing remotes."""
    error = exceptions.RemoteNotFoundError("Unknown remote")

    assert "pivot remote list" in error.get_suggestion()


def test_base_pivot_error_no_suggestion() -> None:
    """Base PivotError returns None for suggestion."""
    error = exceptions.PivotError("Base error")

    assert error.get_suggestion() is None


def test_validation_error_no_suggestion() -> None:
    """ValidationError inherits None suggestion from base."""
    error = exceptions.ValidationError("Validation failed")

    assert error.get_suggestion() is None


# =============================================================================
# format_user_message Tests
# =============================================================================


def test_format_user_message_returns_str() -> None:
    """format_user_message returns string representation of error."""
    error = exceptions.StageNotFoundError("Unknown stage(s): foo, bar")

    assert error.format_user_message() == "Unknown stage(s): foo, bar"


def test_format_user_message_base_error() -> None:
    """Base error format_user_message returns str(self)."""
    error = exceptions.PivotError("Test error message")

    assert error.format_user_message() == "Test error message"
