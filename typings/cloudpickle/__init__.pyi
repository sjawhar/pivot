import pickle
from collections.abc import Callable
from types import ModuleType
from typing import Any, override

from _typeshed import SupportsWrite

def list_registry_pickle_by_value() -> set[str]: ...
def register_pickle_by_value(module: ModuleType) -> None: ...
def unregister_pickle_by_value(module: ModuleType) -> None: ...

class Pickler(pickle.Pickler):
    def __init__(
        self,
        file: SupportsWrite[bytes],
        protocol: int | None = None,
        buffer_callback: Callable[[pickle.PickleBuffer], Any] | None = None,
    ) -> None: ...
    # cloudpickle defines a method where typeshed's `pickle.Pickler` declares an attribute.
    @override
    def reducer_override(self, obj: Any) -> Any: ...  # pyright: ignore[reportIncompatibleMethodOverride]
