"""Unit tests for pivot.display: URL redaction, control escaping, strict JSON."""

from __future__ import annotations

import json
import math
import pathlib
import re
from typing import TYPE_CHECKING

import pytest

from pivot import display

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# =============================================================================
# redact_url_credentials
# =============================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "https://user:token@host/repo.git",
            "https://***@host/repo.git",
            id="user-and-password",
        ),
        pytest.param("https://token@host/repo.git", "https://***@host/repo.git", id="user-only"),
        pytest.param(
            "https://:token@host/repo.git", "https://***@host/repo.git", id="password-only"
        ),
        pytest.param("https://@host/repo.git", "https://***@host/repo.git", id="empty-user-info"),
        pytest.param("s3://key:secret@bucket/prefix", "s3://***@bucket/prefix", id="s3-remote-url"),
        pytest.param(
            "https://user:token@[2001:db8::1]:8443/repo.git",
            "https://***@[2001:db8::1]:8443/repo.git",
            id="ipv6-host-with-port",
        ),
        pytest.param(
            "https://[2001:db8::1]:8443/repo.git",
            "https://[2001:db8::1]:8443/repo.git",
            id="ipv6-host-without-user-info",
        ),
        pytest.param(
            "Cannot resolve ref 'main' from https://user:token@host/r.git",
            "Cannot resolve ref 'main' from https://***@host/r.git",
            id="url-inside-a-message",
        ),
        pytest.param(
            "https://user:token@a/x and https://other:pw@b/y",
            "https://***@a/x and https://***@b/y",
            id="two-urls-in-one-message",
        ),
        pytest.param(
            "ssh://git@github.com/org/repo.git",
            "ssh://***@github.com/org/repo.git",
            id="ssh-url-user-info",
        ),
    ],
)
def test_redact_url_credentials_replaces_user_info(text: str, expected: str) -> None:
    """Every user-info form becomes ``***``, and the rest of the URL is untouched."""
    assert display.redact_url_credentials(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("https://host/repo.git", id="no-user-info"),
        pytest.param("https://host/path/with@at/file", id="at-in-path"),
        pytest.param("https://host/a@b/c", id="at-in-first-path-segment"),
        pytest.param("https://host:8443/a@b", id="at-in-path-with-port"),
        pytest.param("https://host/repo.git?who=a@b", id="at-in-query"),
        pytest.param("https://host/repo.git#a@b", id="at-in-fragment"),
        pytest.param("git@github.com:org/repo.git", id="scp-like-ssh"),
        pytest.param("mail to a@b.example about https://host/x", id="bare-at-after-url"),
        pytest.param("s3://bucket/prefix", id="plain-s3"),
        pytest.param("", id="empty"),
    ],
)
def test_redact_url_credentials_leaves_other_text_byte_for_byte(text: str) -> None:
    """A URL with no user-info, and text that only looks like one, comes back unchanged.

    An scp-like ``git@host:repo`` is not a URL: it has no password field, and its user is the
    SSH login, so it is left as typed.
    """
    assert display.redact_url_credentials(text) == text


def test_redact_url_credentials_keeps_a_credential_whole_across_a_carriage_return() -> None:
    """A control character does not end the authority, so no part of the secret survives."""
    redacted = display.redact_url_credentials("https://user:se\rcret@host/x")
    assert "cret" not in redacted, redacted
    assert redacted == "https://***@host/x"


# =============================================================================
# escape_control
# =============================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("main\x1b[31m", "main\\x1b[31m", id="esc"),
        pytest.param("a\tb", "a\\x09b", id="tab"),
        pytest.param("a\nb", "a\\x0ab", id="newline"),
        pytest.param("a\rb", "a\\x0db", id="carriage-return"),
        pytest.param("a\x00b", "a\\x00b", id="nul"),
        pytest.param("a\x7fb", "a\\x7fb", id="del"),
        pytest.param("a\x9bb", "a\\x9bb", id="c1-csi"),
        pytest.param("plain text", "plain text", id="plain"),
        pytest.param("héllo ✓", "héllo ✓", id="non-ascii-kept"),
    ],
)
def test_escape_control_writes_each_control_visibly(text: str, expected: str) -> None:
    assert display.escape_control(text) == expected


def test_escape_control_keep_newlines_keeps_only_newlines() -> None:
    """A whole message keeps its own line breaks; every other control is still escaped."""
    assert display.escape_control("a\nb\tc\x1bd", keep_newlines=True) == "a\nb\\x09c\\x1bd"


def test_escape_control_output_holds_no_control_character() -> None:
    every_control = "".join(chr(code) for code in [*range(0x20), 0x7F, *range(0x80, 0xA0)])
    escaped = display.escape_control(every_control)
    assert re.search(r"[\x00-\x1f\x7f-\x9f]", escaped) is None, escaped


def test_safe_message_escapes_then_redacts() -> None:
    """A log line keeps its layout, loses its control characters and hides its credentials."""
    message = "Failed:\n  https://user:token@host/x\ttail\x1b[31m"
    assert display.safe_message(message) == "Failed:\n  https://***@host/x\\x09tail\\x1b[31m"


def test_escape_markup_escapes_controls_and_markup() -> None:
    assert display.escape_markup("a\x1b[1m[bold]") == "a\\x1b[1m\\[bold]"


# =============================================================================
# to_json
# =============================================================================


def _refuse(name: str) -> object:
    raise ValueError(f"non-standard JSON constant {name}")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(math.inf, '"Infinity"', id="inf"),
        pytest.param(-math.inf, '"-Infinity"', id="-inf"),
        pytest.param(math.nan, '"NaN"', id="nan"),
        pytest.param(0.5, "0.5", id="finite-float"),
        pytest.param(3, "3", id="int"),
        pytest.param(True, "true", id="bool-is-not-a-float"),
    ],
)
def test_to_json_writes_non_finite_floats_as_strings(value: object, expected: str) -> None:
    assert display.to_json(value) == expected


def test_to_json_converts_non_finite_floats_anywhere_in_the_document() -> None:
    document = display.to_json(
        {"stages": [{"param_changes": [{"old_value": math.inf, "new_value": math.nan}]}]}
    )
    assert json.loads(document, parse_constant=_refuse) == {
        "stages": [{"param_changes": [{"old_value": "Infinity", "new_value": "NaN"}]}]
    }


def test_to_json_converts_a_non_finite_float_key() -> None:
    assert display.to_json({math.inf: 1}) == '{"Infinity": 1}'


def test_to_json_converts_what_default_returns() -> None:
    assert display.to_json(pathlib.Path("x"), default=lambda _o: math.inf) == '"Infinity"'


def test_to_json_refuses_a_non_finite_float_that_reaches_json_dumps(
    mocker: MockerFixture,
) -> None:
    """``allow_nan=False`` is the backstop behind the conversion, so a future change that
    stops converting a value fails loudly instead of writing a bare ``Infinity``."""

    def identity(value: object) -> object:
        return value

    mocker.patch.object(display, "_finite", identity)

    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        display.to_json({"v": math.inf})


def test_to_json_indent_matches_json_dumps() -> None:
    assert display.to_json({"a": [1, 2]}, indent=2) == json.dumps({"a": [1, 2]}, indent=2)
