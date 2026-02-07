"""Tests for log level detection regex and style mapping."""

from __future__ import annotations

import pytest

from pivot_tui.widgets.logs import _LEVEL_STYLES, _LOG_LEVEL_PATTERN, _get_line_style

# =============================================================================
# Pattern Matching - Basic Formats
# =============================================================================


@pytest.mark.parametrize(
    "line,expected_level",
    [
        # Pivot format (colon)
        ("INFO: Loading data", "INFO"),
        ("WARNING: Deprecated", "WARNING"),
        ("ERROR: Failed", "ERROR"),
        ("DEBUG: Internal", "DEBUG"),
        ("CRITICAL: Fatal", "CRITICAL"),
        # Bracketed format
        ("[INFO] Processing", "INFO"),
        ("[WARNING] Check this", "WARNING"),
        # Dash format
        ("INFO - message", "INFO"),
        ("ERROR - failed", "ERROR"),
        # With timestamp
        ("2024-01-01 10:00:00 INFO Loading", "INFO"),
        ("[2024-01-01T10:00:00] ERROR Failed", "ERROR"),
        # Case insensitive
        ("info: lowercase", "INFO"),
        ("Info: mixed", "INFO"),
        # Aliases
        ("WARN: short form", "WARN"),
        ("FATAL: alias", "FATAL"),
    ],
)
def test_log_level_pattern_matches(line: str, expected_level: str) -> None:
    match = _LOG_LEVEL_PATTERN.match(line)
    assert match is not None
    assert match.group(1).upper() == expected_level.upper()


@pytest.mark.parametrize(
    "line",
    [
        "Just a regular line",
        "This is INFORMATION",  # not at start
        "WARNING_FILE.txt processed",  # no delimiter
        "",
        "   ",
    ],
)
def test_log_level_pattern_no_match(line: str) -> None:
    assert _LOG_LEVEL_PATTERN.match(line) is None


# =============================================================================
# Pattern Matching - ANSI Edge Cases
# =============================================================================


def test_log_level_pattern_with_ansi() -> None:
    """ANSI sequences at start should not prevent matching."""
    line = "\033[31mERROR: message\033[0m"
    match = _LOG_LEVEL_PATTERN.match(line)
    assert match is not None
    assert match.group(1).upper() == "ERROR"


def test_log_level_pattern_with_multiple_ansi() -> None:
    """Multiple ANSI sequences should not prevent matching."""
    line = "\033[1m\033[31m\033[4mERROR: message\033[0m"
    match = _LOG_LEVEL_PATTERN.match(line)
    assert match is not None, "Multiple ANSI codes should not block matching"
    assert match.group(1).upper() == "ERROR"


# =============================================================================
# Pattern Matching - Timestamp Edge Cases
# =============================================================================


@pytest.mark.parametrize(
    "line,expected_level",
    [
        # Unix timestamp
        ("1704088800 INFO message", "INFO"),
        # ISO format with timezone
        ("2024-01-01T10:00:00Z ERROR failed", "ERROR"),
        # Comma separators in timestamp
        ("2024,01,01 10:00:00 WARNING check", "WARNING"),
        # Timestamp with brackets
        ("[2024-01-01 10:00:00] DEBUG trace", "DEBUG"),
        # Multiple spaces after timestamp
        ("2024-01-01 10:00:00   INFO  spaced", "INFO"),
    ],
)
def test_log_level_pattern_timestamp_variants(line: str, expected_level: str) -> None:
    """Verify various timestamp formats don't prevent level detection."""
    match = _LOG_LEVEL_PATTERN.match(line)
    assert match is not None, f"Failed to match timestamp format: {line}"
    assert match.group(1).upper() == expected_level


# =============================================================================
# Pattern Matching - Delimiter Edge Cases
# =============================================================================


@pytest.mark.parametrize(
    "line",
    [
        "ERROR.txt processed",  # period after level
        "INFO_LOG started",  # underscore after level
        "WARNINGCODE",  # no delimiter at all
        "DEBUGfile.log",  # no space before filename
    ],
)
def test_log_level_pattern_invalid_delimiters(line: str) -> None:
    """Verify delimiter requirements prevent false positives."""
    match = _LOG_LEVEL_PATTERN.match(line)
    assert match is None, f"Should not match line without valid delimiter: {line}"


def test_log_level_pattern_bracketed_no_space() -> None:
    """Bracketed format should work with closing bracket as delimiter."""
    line = "[ERROR]Failed"
    match = _LOG_LEVEL_PATTERN.match(line)
    assert match is not None, "Closing bracket should count as valid delimiter"
    assert match.group(1).upper() == "ERROR"


def test_log_level_pattern_colon_only() -> None:
    """Level followed by colon only (no message) should match."""
    line = "INFO:"
    match = _LOG_LEVEL_PATTERN.match(line)
    assert match is not None, "Colon-only delimiter should be valid"
    assert match.group(1).upper() == "INFO"


# =============================================================================
# Pattern Matching - Case Sensitivity
# =============================================================================


@pytest.mark.parametrize(
    "level_variant",
    ["eRrOr", "WaRnInG", "dEbUg", "CrItIcAl", "iNfO"],
)
def test_log_level_pattern_mixed_case_variants(level_variant: str) -> None:
    """Pattern should handle any case combination (re.IGNORECASE flag)."""
    line = f"{level_variant}: message"
    match = _LOG_LEVEL_PATTERN.match(line)
    assert match is not None, f"Should match mixed-case variant: {level_variant}"
    assert match.group(1).upper() in _LEVEL_STYLES


# =============================================================================
# Style Dictionary Completeness
# =============================================================================


def test_level_styles_complete() -> None:
    """All matched levels should have a style defined."""
    levels = ["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"]
    for level in levels:
        assert level in _LEVEL_STYLES, f"Missing style for {level}"


def test_level_styles_values() -> None:
    """Verify expected style values."""
    assert _LEVEL_STYLES["DEBUG"] == "dim"
    assert _LEVEL_STYLES["INFO"] is None
    assert _LEVEL_STYLES["WARNING"] == "yellow"
    assert _LEVEL_STYLES["WARN"] == "yellow"
    assert _LEVEL_STYLES["ERROR"] == "red"
    assert _LEVEL_STYLES["CRITICAL"] == "red bold"
    assert _LEVEL_STYLES["FATAL"] == "red bold"


def test_level_styles_unknown_level() -> None:
    """Unknown levels should return None from style dict."""
    assert _LEVEL_STYLES.get("TRACE") is None
    assert _LEVEL_STYLES.get("VERBOSE") is None
    assert _LEVEL_STYLES.get("UNKNOWN") is None


# =============================================================================
# Style Selection Logic (Integration)
# =============================================================================


@pytest.mark.parametrize(
    "line,is_stderr,expected_style",
    [
        # Log levels override stderr coloring
        ("INFO: message", True, None),
        ("WARNING: message", True, "yellow"),
        ("ERROR: message", True, "red"),
        ("DEBUG: trace", False, "dim"),
        ("CRITICAL: crash", True, "red bold"),
        # Non-log stderr gets red fallback
        ("Some random stderr output", True, "red"),
        # Non-log stdout gets no style
        ("Some random stdout output", False, None),
        # Empty lines
        ("", True, "red"),
        ("", False, None),
    ],
)
def test_style_selection_logic(line: str, is_stderr: bool, expected_style: str | None) -> None:
    """Test the complete style selection logic that _write_line uses."""
    assert _get_line_style(line, is_stderr) == expected_style


def test_style_selection_with_ansi_stderr_fallback() -> None:
    """ANSI-prefixed non-log stderr should still get red fallback."""
    line = "\033[1mSome non-log output\033[0m"
    style = _get_line_style(line, is_stderr=True)
    assert style == "red", "Non-log stderr with ANSI should fallback to red"


def test_style_selection_level_takes_precedence() -> None:
    """Log level style should take precedence over stderr fallback."""
    line = "INFO: normal message"
    # Even on stderr, INFO gets no style (not red fallback)
    style = _get_line_style(line, is_stderr=True)
    assert style is None, "INFO style (None) should override stderr fallback"


def test_style_selection_multiline_traceback() -> None:
    """First line gets level style, continuation lines get stderr fallback.

    Each line is processed independently. Traceback continuation lines don't
    have a level prefix, so they fall back to stderr coloring (red).
    """
    # Simulates a Python traceback on stderr
    lines_and_expected = [
        ("ERROR: Failed to process data", "red"),
        ("Traceback (most recent call last):", "red"),
        ('  File "foo.py", line 10, in bar', "red"),
        ("    raise ValueError('bad input')", "red"),
        ("ValueError: bad input", "red"),
    ]
    for line, expected_style in lines_and_expected:
        style = _get_line_style(line, is_stderr=True)
        assert style == expected_style, f"Line '{line}' should have style '{expected_style}'"
