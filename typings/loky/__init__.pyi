"""Type stubs for loky - a robust multiprocessing library."""

from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from contextlib import AbstractContextManager
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Any, Literal, TypeVar

_T = TypeVar("_T")

class _ExecutorFlags:
    shutdown: bool
    # The exception that broke the executor, or None.
    broken: BaseException | None

class ProcessPoolExecutor(Executor):
    """Process pool executor that can be used with concurrent.futures."""

    _flags: _ExecutorFlags
    _max_workers: int
    _pending_work_items: dict[int, object]
    _processes: dict[int, BaseProcess]
    # None once the executor has been shut down.
    _processes_management_lock: AbstractContextManager[object] | None
    # Call-queue capacity, fixed at creation: 2 * max(cpu_count, max_workers) + 1.
    queue_size: int

    def __init__(
        self,
        max_workers: int | None = None,
        job_reducers: Mapping[type, Callable[[Any], Any]] | None = None,
        result_reducers: Mapping[type, Callable[[Any], Any]] | None = None,
        timeout: float | None = None,
        context: BaseContext | None = None,
        initializer: Callable[..., object] | None = None,
        initargs: tuple[Any, ...] = (),
        env: Mapping[str, str] | None = None,
    ) -> None: ...
    def shutdown(  # pyright: ignore[reportImplicitOverride] - loky adds kill_workers
        self, wait: bool = True, *, kill_workers: bool = False, cancel_futures: bool = False
    ) -> None: ...

def get_reusable_executor(
    max_workers: int | None = None,
    context: BaseContext | None = None,
    timeout: float | None = 10,
    kill_workers: bool = False,
    reuse: bool | Literal["auto"] = "auto",
    job_reducers: Mapping[type, Callable[[Any], Any]] | None = None,
    result_reducers: Mapping[type, Callable[[Any], Any]] | None = None,
    initializer: Callable[..., object] | None = None,
    initargs: tuple[Any, ...] = (),
    env: Mapping[str, str] | None = None,
) -> ProcessPoolExecutor: ...
def cpu_count() -> int | None:
    """Return the number of CPUs available, respecting cgroup limits."""
    ...
