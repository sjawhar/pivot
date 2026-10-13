from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pivot import loaders as loaders_module


@dataclasses.dataclass(frozen=True)
class DirectoryOut[T]:  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
    """Directory output - dynamic set of files determined at runtime.

    Generic parameter T represents the value type stored in each file.
    DirectoryOut[T] stores dict[str, T] (the return type), but the loader writes T.

    Use for outputs where the number and paths of files are determined at runtime:

        class TaskOutputs(TypedDict):
            task_results: Annotated[
                dict[str, TaskMetrics],
                DirectoryOut("metrics/task_results/", YAML())
            ]

        def process_tasks(...) -> TaskOutputs:
            return {"task_results": {
                "task_a.yaml": TaskMetrics(accuracy=0.95),
                "task_b.yaml": TaskMetrics(accuracy=0.87),
            }}

    Keys are relative paths within the directory. Values are serialized by the loader.
    Path must end with '/' to enforce directory semantics.
    """

    path: str  # Must be str ending with '/', not list/tuple
    loader: loaders_module.Writer[T]
    cache: bool = True

    def __post_init__(self) -> None:
        if not self.path.endswith("/"):
            raise ValueError(f"DirectoryOut path must end with '/': {self.path!r}")
