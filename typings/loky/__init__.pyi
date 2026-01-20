"""Type stubs for loky - a robust multiprocessing library."""

from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from multiprocessing.context import BaseContext
from typing import Any, TypeVar

_T = TypeVar("_T")

class ProcessPoolExecutor(Executor):
    """Process pool executor that can be used with concurrent.futures."""

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

def get_reusable_executor(
    max_workers: int | None = None,
    context: BaseContext | None = None,
    timeout: int = 10,
    kill_workers: bool = False,
    reuse: str = "auto",
    job_reducers: Mapping[type, Callable[[Any], Any]] | None = None,
    result_reducers: Mapping[type, Callable[[Any], Any]] | None = None,
    initializer: Callable[..., object] | None = None,
    initargs: tuple[Any, ...] = (),
    env: Mapping[str, str] | None = None,
) -> ProcessPoolExecutor: ...
def cpu_count() -> int | None:
    """Return the number of CPUs available, respecting cgroup limits."""
    ...
