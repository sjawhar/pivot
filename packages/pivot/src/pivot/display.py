"""Make values safe to show: hide URL credentials, escape control characters, write strict JSON.

Every URL Pivot prints goes through ``redact_url_credentials``; every value read from a file
(``.pvt``, lock, config, remote listing) that text output prints goes through
``escape_control``, or ``escape_markup`` where Rich or Textual renders it; every log line and
error message the CLI prints goes through ``safe_message``; every ``--json`` document, JSONL
line and agent RPC message is encoded by ``to_json``.
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

# A scheme and "://", then the authority up to its last "@": the user-info. The authority ends
# at the first "/", "?", "#" or space, so an "@" in a path, query or fragment, or further
# along a message, is not user-info. Every other character, a control character included,
# stays inside the authority, so a carriage return in a credential cannot end it early and
# leave the rest of the secret shown. Text without URL user-info comes back unchanged.
_URL_USER_INFO = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*://)[^/?# ]*@")

# Unicode category Cc is exactly these code points: C0 controls, DEL and C1 controls.
CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_CONTROL_CHARACTER_BUT_NEWLINE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")


def redact_url_credentials(text: str) -> str:
    """Replace the user-info of every URL in ``text`` with ``***``.

    ``https://user:token@host/repo`` becomes ``https://***@host/repo``, and so do the user-only
    and password-only forms. ``text`` may be one URL or a message that holds URLs. An scp-like
    ``git@host:repo`` is not a URL and is left as is: it has no password field, and its user is
    the SSH login, not a secret.
    """
    return _URL_USER_INFO.sub(r"\1***@", text)


def escape_control(text: str, *, keep_newlines: bool = False) -> str:
    r"""Write each control character in ``text`` as visible ``\xNN`` text.

    Covers every C0 control (tab and newline included, so one value stays on one line), DEL and
    every C1 control, so a value read from a file cannot move the cursor or restyle the terminal.
    ``keep_newlines`` keeps each newline, for a whole message whose own line breaks are its
    layout; every other control is still escaped.
    """
    pattern = _CONTROL_CHARACTER_BUT_NEWLINE if keep_newlines else CONTROL_CHARACTER
    return pattern.sub(lambda match: f"\\x{ord(match.group()):02x}", text)


def safe_message(text: str) -> str:
    """``text``, a whole log line or error message, made safe to print to a terminal.

    Escapes every control character except the message's newlines, then hides URL credentials.
    """
    return redact_url_credentials(escape_control(text, keep_newlines=True))


def escape_markup(text: str) -> str:
    """``text`` ready for a Rich or Textual renderable: controls escaped, then markup escaped.

    Rich strips only five C0 codes (BEL, BS, VT, FF, CR) from a renderable, so ESC, DEL and
    the C1 controls would otherwise reach the terminal; ``rich.markup.escape`` covers only
    markup. One value stays on one line, as in plain text output.

    Rich is imported here, not at module scope: ``path_policy`` imports this module for one
    regex and every worker process imports ``path_policy`` through ``storage.lock``, so an
    eager import loads Rich's 28 modules (about 14 ms) in every worker and 22 ms on
    ``pivot --help``. Only a console or a TUI renderable needs Rich.
    """
    import rich.markup

    return rich.markup.escape(escape_control(text))


def to_json(
    value: object,
    *,
    indent: int | None = None,
    default: Callable[[object], object] | None = None,
    sort_keys: bool = False,
) -> str:
    """Encode ``value`` as one RFC 8259 JSON text.

    A non-finite float is written as the string ``"Infinity"``, ``"-Infinity"`` or ``"NaN"``:
    the names Python's ``json`` writes bare, quoted so that strict parsers (jq, Go,
    ``JSON.parse``) accept the text. ``default`` converts other objects as ``json.dumps`` does,
    and its results get the same treatment. ``sort_keys`` orders each object's keys, for a text
    two equal values must share. ``allow_nan=False`` refuses any non-finite float that gets
    past the conversion.
    """
    finite_default = None if default is None else _finite_default(default)
    return json.dumps(
        _finite(value),
        indent=indent,
        default=finite_default,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def _finite_default(default: Callable[[object], object]) -> Callable[[object], object]:
    return lambda obj: _finite(default(obj))


def _finite(value: object) -> object:
    """``value`` with each non-finite float, as a value or a key, replaced by its name."""
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {
            (_finite(key) if isinstance(key, float) else key): _finite(item)
            for key, item in mapping.items()
        }
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return [_finite(item) for item in items]
    return value
