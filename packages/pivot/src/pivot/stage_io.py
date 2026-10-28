"""Which calls of a running stage are pivot's own I/O on state pivot owns.

A stage that fails prints one line (``exceptions.describe_error``) only when the exception left
such a call: storing its outputs in the cache, opening the state database, reading or writing its
lock file, preparing its output path, and creating its output file and putting it in place in the
formats pivot writes itself. Each of those calls is marked where pivot makes it (``PIVOT_IO``,
``output_io``), so the call decides, never the exception's type and never its frames.

Any other failure keeps its traceback: one out of the stage function, out of a loader or writer
(pivot's own included), out of whatever code of the user's they run, and any failure nothing
marked. A format that encodes a value as it writes it (JSON, JSON Lines, pickle) runs that value's
own code inside the write, so only its file creation is marked; a format that hands pivot the
finished text (text, YAML, DataFrame JSON Lines) has its write marked too.

This module imports nothing of pivot's, so every layer can mark its own calls.
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator
    from types import TracebackType


class _StageIO(threading.local):
    """This thread's running stage: the exception that left pivot's own I/O, and the output path
    pivot is writing now."""

    error: BaseException | None = None
    output: pathlib.Path | None = None


_state = _StageIO()


class _PivotCall:
    """Brackets a call of pivot's own I/O on state it owns: an exception that leaves it is
    pivot's."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_val is not None:
            _state.error = exc_val


PIVOT_IO = _PivotCall()
_NOT_PIVOT_IO = contextlib.nullcontext()


def output_io(path: pathlib.Path) -> contextlib.AbstractContextManager[None]:
    """The bracket for a writer's own handling of the file at ``path``: creating it, renaming it
    into place, or writing text the writer has already encoded.

    It is pivot's own I/O only while pivot is writing that very output (``writing_output``), not
    when a stage function or a user's writer calls a pivot writer for a path of its own.
    """
    return PIVOT_IO if path == _state.output else _NOT_PIVOT_IO


@contextlib.contextmanager
def writing_output(path: pathlib.Path) -> Generator[None]:
    """Mark ``path`` as the output pivot is writing while the block runs."""
    _state.output = path
    try:
        yield
    finally:
        _state.output = None


def raised_by_pivot(error: BaseException) -> bool:
    """Whether ``error`` left pivot's own I/O on state it owns, in this thread's stage."""
    return _state.error is error


def clear() -> None:
    """Drop the stage's recorded failure, whose frames hold the stage's locals and inputs."""
    _state.error = None
