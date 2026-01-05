"""Type stubs for dvc.output."""

from typing import Any

class Output:
    """DVC output representation."""

    fs_path: str
    use_cache: bool
    persist: bool
    plot: bool | dict[str, Any]
    metric: bool
