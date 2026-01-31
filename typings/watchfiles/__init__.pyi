"""Type stubs for watchfiles."""

from collections.abc import AsyncGenerator, Callable, Generator
from enum import IntEnum
from pathlib import Path
from threading import Event as ThreadingEvent

import anyio

__all__ = (
    "watch",
    "awatch",
    "run_process",
    "arun_process",
    "Change",
    "BaseFilter",
    "DefaultFilter",
    "PythonFilter",
    "VERSION",
)

VERSION: str

class Change(IntEnum):
    added = 1
    modified = 2
    deleted = 3

    def raw_str(self) -> str: ...

type FileChange = tuple[Change, str]

# AnyEvent can be anyio.Event, asyncio.Event, or trio.Event
# We use anyio.Event as the primary type since that's what anyio uses
type AnyEvent = anyio.Event

class BaseFilter:
    def __call__(self, change: Change, path: str) -> bool: ...

class DefaultFilter(BaseFilter):
    def __init__(
        self,
        *,
        ignore_dirs: tuple[str, ...] | None = None,
        ignore_entity_patterns: tuple[str, ...] | None = None,
        ignore_paths: tuple[Path, ...] | None = None,
    ) -> None: ...

class PythonFilter(DefaultFilter):
    def __init__(
        self,
        *,
        extra_extensions: tuple[str, ...] = (),
        ignore_paths: tuple[Path, ...] | None = None,
    ) -> None: ...

def watch(
    *paths: Path | str,
    watch_filter: Callable[[Change, str], bool] | None = ...,
    debounce: int = 1600,
    step: int = 50,
    stop_event: ThreadingEvent | None = None,
    rust_timeout: int | None = None,
    yield_on_timeout: bool = False,
    debug: bool | None = None,
    raise_interrupt: bool | None = None,
    force_polling: bool | None = None,
    poll_delay_ms: int = 300,
    recursive: bool = True,
    ignore_permission_denied: bool | None = None,
) -> Generator[set[FileChange]]: ...
def awatch(
    *paths: Path | str,
    watch_filter: Callable[[Change, str], bool] | None = ...,
    debounce: int = 1600,
    step: int = 50,
    stop_event: anyio.Event | None = None,
    rust_timeout: int | None = None,
    yield_on_timeout: bool = False,
    debug: bool | None = None,
    raise_interrupt: bool | None = None,
    force_polling: bool | None = None,
    poll_delay_ms: int = 300,
    recursive: bool = True,
    ignore_permission_denied: bool | None = None,
) -> AsyncGenerator[set[FileChange]]: ...
def run_process(
    *paths: Path | str,
    target: Callable[..., None],
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
    watch_filter: Callable[[Change, str], bool] | None = ...,
    debounce: int = 1600,
    step: int = 50,
    stop_event: ThreadingEvent | None = None,
    rust_timeout: int | None = None,
    debug: bool | None = None,
    raise_interrupt: bool | None = None,
    force_polling: bool | None = None,
    poll_delay_ms: int = 300,
    recursive: bool = True,
    ignore_permission_denied: bool | None = None,
    sigint_timeout: int = 5,
    sigkill_timeout: int = 1,
) -> int: ...
async def arun_process(
    *paths: Path | str,
    target: Callable[..., None],
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
    watch_filter: Callable[[Change, str], bool] | None = ...,
    debounce: int = 1600,
    step: int = 50,
    rust_timeout: int | None = None,
    debug: bool | None = None,
    force_polling: bool | None = None,
    poll_delay_ms: int = 300,
    recursive: bool = True,
    ignore_permission_denied: bool | None = None,
    sigint_timeout: int = 5,
    sigkill_timeout: int = 1,
) -> int: ...
