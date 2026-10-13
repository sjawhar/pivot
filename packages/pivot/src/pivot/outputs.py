from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pivot import loaders as loaders_module

# Path type: single string, variable-length list, or fixed-length tuple
PathType = str | list[str] | tuple[str, ...]

# JSON-serializable type for Metric
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


# Loader factory functions for dataclass defaults (defined before classes that use them)


def _default_json_writer() -> loaders_module.Writer[JsonValue]:
    """Factory for default Metric writer."""
    from pivot import loaders

    return loaders.JSON()


@dataclasses.dataclass(frozen=True)
class Out[W]:  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
    """Unified output marker and storage.

    Used both as annotation marker AND registry storage.

    Use in TypedDict return type to declare file outputs:

        class ProcessOutputs(TypedDict):
            result: Annotated[dict[str, int], Out("output.json", JSON())]

        def process(
            params: MyParams,
        ) -> ProcessOutputs:
            return {"result": {"count": 42}}

    For single outputs, annotate the return type directly:

        def transform(
            params: MyParams,
        ) -> Annotated[DataFrame, Out("output.csv", CSV())]:
            return DataFrame()

    For multiple files per output key:

        class ShardOutputs(TypedDict):
            shards: Annotated[list[dict], Out(["a.json", "b.json"], JSON())]

    Loader is REQUIRED to prevent type lies (e.g., Out[DataFrame] with wrong loader).
    """

    path: PathType
    loader: loaders_module.Writer[W]
    cache: bool = True


@dataclasses.dataclass(frozen=True)
class Metric(Out[JsonValue]):
    """Metric output - JSON file with cache=False by default (git-tracked).

    Default loader: JSON(). Type: any JSON-serializable value.

    Use for metrics that should be version-controlled:

        class TrainOutputs(TypedDict):
            metrics: Annotated[dict, Metric("metrics/train.json")]

    Metrics are not cached by default (they're small and should be git-tracked).
    """

    loader: loaders_module.Writer[JsonValue] = dataclasses.field(
        default_factory=_default_json_writer
    )  # type: ignore[assignment]
    cache: bool = False


@dataclasses.dataclass(frozen=True)
class Plot[W](Out[W]):  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
    """Plot output with visualization options.

    Use for visualization outputs with explicit loader:

        # Manual file creation (user saves the plot)
        class TrainOutputs(TypedDict):
            loss_curve: Annotated[pathlib.Path, Plot("plots/loss.png", PathOnly())]

        # Automatic saving (Pivot saves the figure)
        class TrainOutputs(TypedDict):
            loss_curve: Annotated[Figure, Plot("plots/loss.png", MatplotlibFigure())]

    Extra attributes (x, y, template) are used for DVC-compatible plot configuration.
    """

    x: str | None = None
    y: str | None = None
    template: str | None = None


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


