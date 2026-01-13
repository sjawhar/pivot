import pickle

import click
import pytest

from pivot import exceptions
from pivot.cli import errors as cli_errors

# =============================================================================
# handle_pivot_error Tests
# =============================================================================


def test_handle_pivot_error_formats_message() -> None:
    """handle_pivot_error creates ClickException with formatted message."""
    error = exceptions.StageNotFoundError(["foo"])

    result = cli_errors.handle_pivot_error(error)

    assert isinstance(result, click.ClickException)
    assert "Unknown stage(s): foo" in result.format_message()


def test_handle_pivot_error_includes_suggestion() -> None:
    """handle_pivot_error includes suggestion in message."""
    error = exceptions.StageNotFoundError(["foo"])

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
    """StageNotFoundError provides suggestion."""
    error = exceptions.StageNotFoundError(["unknown"])
    suggestion = error.get_suggestion()
    assert suggestion is not None
    assert "pivot list" in suggestion


def test_dependency_not_found_error_suggestion() -> None:
    """DependencyNotFoundError provides suggestion."""
    error = exceptions.DependencyNotFoundError(stage="test", dep="missing.csv")
    suggestion = error.get_suggestion()
    assert suggestion is not None
    assert "file exists" in suggestion or "produced by another stage" in suggestion


@pytest.mark.parametrize(
    ("exception_class", "message", "expected_substrings"),
    [
        pytest.param(
            exceptions.CyclicGraphError,
            "Cycle detected",
            ["dependencies", "circular"],
            id="cyclic_graph",
        ),
        pytest.param(
            exceptions.CacheMissError,
            "Not in cache",
            ["pivot pull", "re-run"],
            id="cache_miss",
        ),
        pytest.param(
            exceptions.TrackedFileMissingError,
            "File missing",
            ["pivot checkout"],
            id="tracked_file_missing",
        ),
        pytest.param(
            exceptions.RemoteNotConfiguredError,
            "No remote",
            ["pivot config set remotes."],
            id="remote_not_configured",
        ),
        pytest.param(
            exceptions.RemoteNotFoundError,
            "Unknown remote",
            ["pivot remote list"],
            id="remote_not_found",
        ),
    ],
)
def test_exception_suggestion(
    exception_class: type[exceptions.PivotError],
    message: str,
    expected_substrings: list[str],
) -> None:
    """Exception types provide appropriate suggestions."""
    error = exception_class(message)
    suggestion = error.get_suggestion()
    assert suggestion is not None
    for expected in expected_substrings:
        assert expected in suggestion, f"Expected '{expected}' in suggestion '{suggestion}'"


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
    error = exceptions.StageNotFoundError(["foo", "bar"])

    assert "Unknown stage(s): foo, bar" in error.format_user_message()


def test_format_user_message_base_error() -> None:
    """Base error format_user_message returns str(self)."""
    error = exceptions.PivotError("Test error message")

    assert error.format_user_message() == "Test error message"


# =============================================================================
# Fuzzy Matching Tests
# =============================================================================


def test_stage_not_found_fuzzy_suggestion_single_typo() -> None:
    """StageNotFoundError suggests similar stage name for typo."""
    error = exceptions.StageNotFoundError(
        ["preproces"], available_stages=["preprocess", "train", "evaluate"]
    )

    message = error.format_user_message()

    assert "Did you mean" in message
    assert "preprocess" in message


def test_stage_not_found_fuzzy_suggestion_multiple_unknowns() -> None:
    """StageNotFoundError suggests for multiple unknown stages."""
    error = exceptions.StageNotFoundError(
        ["preproces", "trian"],
        available_stages=["preprocess", "train", "evaluate"],
    )

    message = error.format_user_message()

    assert "Did you mean" in message
    assert "preprocess" in message
    assert "train" in message


@pytest.mark.parametrize(
    ("unknown_stages", "available_stages", "expected_message"),
    [
        pytest.param(
            ["xyz"],
            ["preprocess", "train", "evaluate"],
            "Unknown stage(s): xyz",
            id="no_similar_match",
        ),
        pytest.param(
            ["preproces"],
            None,
            "Unknown stage(s): preproces",
            id="no_available_stages",
        ),
    ],
)
def test_stage_not_found_no_fuzzy_suggestion(
    unknown_stages: list[str],
    available_stages: list[str] | None,
    expected_message: str,
) -> None:
    """StageNotFoundError omits 'Did you mean' when no similar stage or no available stages."""
    error = exceptions.StageNotFoundError(unknown_stages, available_stages=available_stages)

    message = error.format_user_message()

    assert "Did you mean" not in message
    assert expected_message in message


def test_stage_not_found_truncation_message() -> None:
    """StageNotFoundError indicates when suggestions are truncated."""
    error = exceptions.StageNotFoundError(
        ["preproces", "trian", "evaluat", "infer"],
        available_stages=["preprocess", "train", "evaluate", "inference"],
    )

    message = error.format_user_message()

    assert "showing first 3" in message
    assert "4 unknown stages" in message


def test_stage_not_found_no_truncation_message_for_three_or_fewer() -> None:
    """StageNotFoundError omits truncation message when 3 or fewer unknown stages."""
    error = exceptions.StageNotFoundError(
        ["preproces", "trian", "evaluat"],
        available_stages=["preprocess", "train", "evaluate"],
    )

    message = error.format_user_message()

    assert "Did you mean" in message
    assert "showing first" not in message


def test_dependency_not_found_fuzzy_suggestion() -> None:
    """DependencyNotFoundError suggests similar output path."""
    error = exceptions.DependencyNotFoundError(
        stage="train",
        dep="data/input.scv",
        available_outputs=["data/input.csv", "data/output.csv"],
    )

    message = error.format_user_message()

    assert "Did you mean" in message
    assert "data/input.csv" in message


def test_dependency_not_found_no_suggestion_short_dep() -> None:
    """DependencyNotFoundError skips suggestion for very short paths."""
    error = exceptions.DependencyNotFoundError(
        stage="train",
        dep="ab",
        available_outputs=["abc", "def"],
    )

    message = error.format_user_message()

    assert "Did you mean" not in message


@pytest.mark.parametrize(
    "original",
    [
        pytest.param(
            exceptions.StageNotFoundError(["preproces"], available_stages=["preprocess", "train"]),
            id="stage_not_found",
        ),
        pytest.param(
            exceptions.DependencyNotFoundError(
                stage="train", dep="data/input.scv", available_outputs=["data/input.csv"]
            ),
            id="dependency_not_found",
        ),
    ],
)
def test_fuzzy_error_pickling(original: exceptions.PivotError) -> None:
    """Fuzzy matching errors can be pickled and unpickled with suggestions intact."""
    unpickled = pickle.loads(pickle.dumps(original))

    assert str(unpickled) == str(original)
    assert "Did you mean" in unpickled.format_user_message()
