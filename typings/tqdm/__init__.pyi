"""Minimal type stubs for tqdm progress bar library."""

from collections.abc import Iterable, Iterator
from typing import IO

class tqdm[T]:
    """Progress bar wrapper."""

    total: int | None
    n: int

    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        desc: str | None = None,
        total: int | None = None,
        leave: bool = True,
        file: IO[str] | None = None,
        ncols: int | None = None,
        mininterval: float = 0.1,
        maxinterval: float = 10.0,
        miniters: int | float | None = None,
        ascii: bool | str | None = None,
        disable: bool = False,
        unit: str = "it",
        unit_scale: bool | int | float = False,
        dynamic_ncols: bool = False,
        smoothing: float = 0.3,
        bar_format: str | None = None,
        initial: int = 0,
        position: int | None = None,
        postfix: dict[str, object] | None = None,
        unit_divisor: float = 1000,
        write_bytes: bool = False,
        lock_args: tuple[object, ...] | None = None,
        nrows: int | None = None,
        colour: str | None = None,
        delay: float = 0,
    ) -> None: ...
    def __iter__(self) -> Iterator[T]: ...
    def update(self, n: int = 1) -> bool | None: ...
    def close(self) -> None: ...
    def refresh(
        self, nolock: bool = False, lock_args: tuple[object, ...] | None = None
    ) -> bool: ...
    def reset(self, total: int | None = None) -> None: ...
    def set_description(self, desc: str | None = None, refresh: bool = True) -> None: ...
    def set_postfix(
        self, ordered_dict: dict[str, object] | None = None, refresh: bool = True, **kwargs: object
    ) -> None: ...
