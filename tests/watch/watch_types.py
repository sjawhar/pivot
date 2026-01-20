"""Type definitions for watch tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from watchfiles import Change

    from pivot import ignore
    from pivot.watch import _watch_utils


class CreateFilterForStages(Protocol):
    """Protocol for the create_filter_for_stages fixture factory."""

    def __call__(
        self,
        stages: list[str],
        output_filter: _watch_utils.OutputFilter | None = None,
        ignore_filter: ignore.IgnoreFilter | None = None,
        watch_globs: list[str] | None = None,
    ) -> Callable[[Change, str], bool]: ...
