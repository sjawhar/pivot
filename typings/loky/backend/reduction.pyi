"""Type stubs for loky's reduction module: the pickler loky pickles a worker's call with."""

from collections.abc import Callable, Mapping
from typing import Any

import cloudpickle
from _typeshed import SupportsWrite

# The class `set_loky_pickler` builds on the pickler it is set to (cloudpickle's by default) and
# `get_loky_pickler` returns; loky names it `CustomizablePickler` inside that function only.
class _CustomizablePickler(cloudpickle.Pickler):
    def __init__(
        self,
        writer: SupportsWrite[bytes],
        reducers: Mapping[type, Callable[[Any], Any]] | None = None,
        protocol: int | None = ...,
    ) -> None: ...

def get_loky_pickler() -> type[_CustomizablePickler]: ...
