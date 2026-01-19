"""Minimal type stubs for pygtrie - only APIs used by pivot."""

from collections.abc import Generator, Iterable, Sequence
from typing import Generic, TypeVar

_V = TypeVar("_V")

class _Step(Generic[_V]):  # noqa: UP046
    """Result from prefix queries - provides access to matched key/value."""

    @property
    def key(self) -> tuple[str, ...]: ...
    @property
    def value(self) -> _V | None: ...
    @property
    def is_set(self) -> bool: ...
    @property
    def has_subtrie(self) -> bool: ...

class Trie(Generic[_V]):  # noqa: UP046
    """Trie (prefix tree) mapping path tuples to values.

    Keys are sequences (typically tuple[str, ...] from Path.parts).
    """

    def __init__(self, *args: Iterable[tuple[Sequence[str], _V]], **kwargs: _V) -> None: ...
    def __setitem__(self, key: Sequence[str], value: _V) -> None: ...
    def __getitem__(self, key: Sequence[str]) -> _V: ...
    def __contains__(self, key: object) -> bool: ...
    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def has_subtrie(self, key: Sequence[str]) -> bool:
        """Return True if key is a prefix of any key in the trie."""
        ...

    def values(self, prefix: Sequence[str] | None = None, shallow: bool = False) -> Generator[_V]:
        """Iterate over values, optionally filtering by prefix."""
        ...

    def keys(
        self, prefix: Sequence[str] | None = None, shallow: bool = False
    ) -> Generator[tuple[str, ...]]:
        """Iterate over keys, optionally filtering by prefix."""
        ...

    def shortest_prefix(self, key: Sequence[str]) -> _Step[_V] | None:
        """Return Step for shortest prefix of key in trie, or None."""
        ...

    def longest_prefix(self, key: Sequence[str]) -> _Step[_V] | None:
        """Return Step for longest prefix of key in trie, or None."""
        ...
