"""Type stubs for loky's reusable executor internals Pivot uses."""

from typing import Any

from loky import ProcessPoolExecutor

_executor: ProcessPoolExecutor | None
_executor_kwargs: dict[str, Any] | None

class _ExecutorStorage:
    executor: ProcessPoolExecutor | None
    executor_kwargs: dict[str, Any] | None

_executor_storage: _ExecutorStorage
