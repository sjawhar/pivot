from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot.watch import _watch_utils

if TYPE_CHECKING:
    from collections.abc import Callable

    from watchfiles import Change

    from pivot import ignore

    from .watch_types import CreateFilterForStages


@pytest.fixture
def create_filter_for_stages() -> CreateFilterForStages:
    """Factory fixture: create watch filter for given stages.

    If output_filter is provided, use it. Otherwise create one from stages.
    """

    def _create(
        stages: list[str],
        output_filter: _watch_utils.OutputFilter | None = None,
        ignore_filter: ignore.IgnoreFilter | None = None,
        watch_globs: list[str] | None = None,
    ) -> Callable[[Change, str], bool]:
        # Create output filter from stages if not provided
        if output_filter is None:
            output_filter = _watch_utils.OutputFilter(stages)
        return _watch_utils.create_watch_filter(
            watch_globs=watch_globs,
            ignore_filter=ignore_filter,
            output_filter=output_filter,
        )

    return _create
