"""Stage name display and resolution helpers.

In single-pipeline mode the ``{pipeline}/`` prefix is noise. These helpers
strip the prefix for display and resolve bare names typed by the user back
to their fully-qualified form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def display_stage_name(stage_name: str, strip_prefix: str | None) -> str:
    """Strip pipeline prefix for single-pipeline display.

    Returns *stage_name* unchanged when *strip_prefix* is ``None`` or the
    name doesn't start with ``{strip_prefix}/``.
    """
    if strip_prefix and stage_name.startswith(f"{strip_prefix}/"):
        return stage_name[len(strip_prefix) + 1 :]
    return stage_name


def resolve_stage_name(name: str, all_stages: Mapping[str, object]) -> str:
    """Resolve a possibly bare stage name to its prefixed form.

    If *name* is already a registered stage, return it.  Otherwise look
    for a unique stage ending with ``/{name}``.  Falls through to the
    original *name* when there is no match or the match is ambiguous
    (caller handles error).
    """
    if name in all_stages:
        return name
    matches = [s for s in all_stages if s.endswith(f"/{name}")]
    if len(matches) == 1:
        return matches[0]
    return name
