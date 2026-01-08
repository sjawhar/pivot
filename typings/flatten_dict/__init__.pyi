from typing import Any

def flatten(
    d: dict[str, Any],
    reducer: str = "tuple",
    inverse: bool = False,
    max_flatten_depth: int | None = None,
    enumerate_types: tuple[type, ...] = (),
    keep_empty_types: tuple[type, ...] = (),
) -> dict[str, Any]: ...
