from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Generic, TypeGuard, TypeVar

if TYPE_CHECKING:
    from pivot import loaders as loaders_module

T = TypeVar("T")

# Path type: single string, variable-length list, or fixed-length tuple
PathType = str | list[str] | tuple[str, ...]

# JSON-serializable type for Metric
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


# Loader factory functions for dataclass defaults (defined before classes that use them)


def _default_json_loader() -> loaders_module.Loader[JsonValue]:
    """Factory for default Metric loader."""
    from pivot import loaders

    return loaders.JSON()


@dataclasses.dataclass(frozen=True)
class Dep(Generic[T]):  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
    """Dependency marker for Annotated type hints.

    Use in function parameters to declare a file dependency:

        def process(
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
        ) -> ProcessOutputs:
            return {"result": {"count": len(data)}}

    For multiple files, use a list (variable-length) or tuple (fixed-length):

        def process(
            shards: Annotated[list[DataFrame], Dep(["a.csv", "b.csv"], CSV())],
        ) -> ProcessOutputs:
            return {"result": {"count": sum(len(df) for df in shards)}}

    Testing is natural - just pass the data directly:

        result = process(test_dataframe)
    """

    path: PathType
    loader: loaders_module.Loader[T]


@dataclasses.dataclass(frozen=True)
class PlaceholderDep(Generic[T]):  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
    """Dependency marker with no default path — must be overridden at registration.

    Use when a stage needs a dependency that has no sensible default.
    Registration fails if dep_path_overrides doesn't include this dependency.

        def compare(
            baseline: Annotated[DataFrame, PlaceholderDep(CSV())],
            experiment: Annotated[DataFrame, PlaceholderDep(CSV())],
        ) -> CompareOutputs:
            ...

        pipeline.register(
            compare,
            dep_path_overrides={
                "baseline": "model_a/results.csv",
                "experiment": "model_b/results.csv",
            },
        )
    """

    loader: loaders_module.Loader[T]


@dataclasses.dataclass(frozen=True)
class Out(Generic[T]):  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
    """Unified output marker and storage.

    Used both as annotation marker AND registry storage.

    Use in TypedDict return type to declare file outputs:

        class ProcessOutputs(TypedDict):
            result: Annotated[dict[str, int], Out("output.json", JSON())]

        def process(
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
        ) -> ProcessOutputs:
            return {"result": {"count": len(data)}}

    For single outputs, annotate the return type directly:

        def transform(
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
        ) -> Annotated[DataFrame, Out("output.csv", CSV())]:
            return data.dropna()

    For multiple files per output key:

        class ShardOutputs(TypedDict):
            shards: Annotated[list[dict], Out(["a.json", "b.json"], JSON())]

    Loader is REQUIRED to prevent type lies (e.g., Out[DataFrame] with wrong loader).
    """

    path: PathType
    loader: loaders_module.Loader[T]
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

    loader: loaders_module.Loader[JsonValue] = dataclasses.field(
        default_factory=_default_json_loader
    )  # type: ignore[assignment]
    cache: bool = False


@dataclasses.dataclass(frozen=True)
class Plot(Out[T]):  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
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
class IncrementalOut(Out[T]):
    """Incremental output - restored from cache before stage runs.

    Use for outputs that are incrementally updated across runs:

        class CacheOutputs(TypedDict):
            cache: Annotated[dict, IncrementalOut("cache.json", JSON())]

    Before stage execution, previous output is restored from cache as a writable
    copy. Stage can read, modify, and write back. Changes are re-cached after run.

    Loader is REQUIRED (inherited from Out).
    """


@dataclasses.dataclass(frozen=True)
class DirectoryOut(Out[dict[str, T]]):  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
    """Directory output - dynamic set of files determined at runtime.

    Generic parameter T represents the value type stored in each file.
    DirectoryOut[T] stores dict[str, T], matching Out[dict[str, T]].

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

    Loader is REQUIRED (inherited from Out).
    """

    def __post_init__(self) -> None:
        # path must be str for DirectoryOut (not list/tuple)
        if not isinstance(self.path, str):
            raise TypeError(f"DirectoryOut path must be a string, got {type(self.path).__name__}")
        if not self.path.endswith("/"):
            raise ValueError(f"DirectoryOut path must end with '/': {self.path!r}")


def is_directory_out(spec: Out[Any]) -> TypeGuard[DirectoryOut[Any]]:
    """Type guard to narrow Out to DirectoryOut, reducing need for casts."""
    return isinstance(spec, DirectoryOut)


# Type alias for compatibility - Out is now the base
BaseOut = Out[Any]

# Output spec can be string (converted to Out with default loader) or any Out subclass
OutSpec = str | Out[Any]


def normalize_out(out: OutSpec) -> Out[Any]:
    """Convert string or Out subclass to Out object.

    String paths are converted to Out with PathOnly loader.
    """
    if isinstance(out, str):
        from pivot import loaders

        return Out(path=out, loader=loaders.PathOnly())
    return out
