from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pivot import loaders as loaders_module


@dataclasses.dataclass(frozen=True)
class DirectoryOut[T]:  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
    """Directory output marker: one loader writing a set of files chosen at runtime.

    Stages cannot use it. ``compose`` treats it as unsupported ``Annotated`` metadata on a
    return type and raises ``StageDefinitionError``, naming the stage, the output key and
    ``DirectoryOut``. ``path`` must end with '/'.
    """

    path: str  # Must be str ending with '/', not list/tuple
    loader: loaders_module.Writer[T]
    cache: bool = True

    def __post_init__(self) -> None:
        if not self.path.endswith("/"):
            raise ValueError(f"DirectoryOut path must end with '/': {self.path!r}")
