from __future__ import annotations

import dataclasses
import pathlib  # noqa: TC003 - used at runtime in _load_deps and _save_outs
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    TypeVar,
    overload,
    override,
)

import pydantic

if TYPE_CHECKING:
    from pivot import loaders


T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class DepSpec:
    """Specification for a dependency."""

    path: str
    loader: loaders.Loader[Any]
    descriptor: Dep[Any]


@dataclasses.dataclass(frozen=True)
class OutSpec:
    """Specification for an output."""

    path: str
    loader: loaders.Loader[Any]
    descriptor: Out[Any]


class Dep(Generic[T]):  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
    """Dependency descriptor with type-safe access.

    Use as a class field to declare a file dependency:

        class MyParams(StageDef):
            data: Dep[pandas.DataFrame] = Dep("input.csv", loaders.CSV())

    After calling _load_deps(), accessing the field returns typed data:

        params = MyParams()
        params._load_deps(project_root)
        df = params.data  # Type: pandas.DataFrame
    """

    __slots__: ClassVar[tuple[str, ...]] = ("path", "loader", "_name")

    path: str
    loader: loaders.Loader[T]
    _name: str

    def __init__(self, path: str, loader: loaders.Loader[T]) -> None:
        self.path = path
        self.loader = loader
        self._name = ""

    def __set_name__(self, owner: type[StageDef], name: str) -> None:
        self._name = name
        # Register this dep with the owner's _deps_specs
        # Copy parent specs to ensure inheritance works (child gets parent deps + its own)
        if "_deps_specs" not in owner.__dict__:
            parent_specs = getattr(owner, "_deps_specs", {})
            owner._deps_specs = dict(parent_specs)  # pyright: ignore[reportPrivateUsage] - descriptor needs to access class internals
        owner._deps_specs[name] = DepSpec(  # pyright: ignore[reportPrivateUsage] - descriptor needs to access class internals
            path=self.path, loader=self.loader, descriptor=self
        )

    @overload
    def __get__(self, obj: None, _owner: type[StageDef]) -> Dep[T]: ...

    @overload
    def __get__(self, obj: StageDef, _owner: type[StageDef]) -> T: ...

    def __get__(self, obj: StageDef | None, _owner: type[StageDef]) -> Dep[T] | T:
        if obj is None:
            return self
        if self._name not in obj._deps_data:  # pyright: ignore[reportPrivateUsage] - descriptor needs to access instance internals
            raise RuntimeError(f"Dependency '{self._name}' not loaded. Call _load_deps() first.")
        return obj._deps_data[self._name]  # pyright: ignore[reportPrivateUsage] - descriptor needs to access instance internals


class Out(Generic[T]):  # noqa: UP046 - basedpyright doesn't support PEP 695 syntax yet
    """Output descriptor with type-safe access.

    Use as a class field to declare a file output:

        class MyParams(StageDef):
            result: Out[dict[str, int]] = Out("output.json", loaders.JSON())

    Assign values and they're saved when _save_outs() is called:

        params = MyParams()
        params.result = {"count": 1}  # Type-checked assignment
        params._save_outs(project_root)
    """

    __slots__: ClassVar[tuple[str, ...]] = ("path", "loader", "_name")

    path: str
    loader: loaders.Loader[T]
    _name: str

    def __init__(self, path: str, loader: loaders.Loader[T]) -> None:
        self.path = path
        self.loader = loader
        self._name = ""

    def __set_name__(self, owner: type[StageDef], name: str) -> None:
        self._name = name
        # Register this out with the owner's _outs_specs
        # Copy parent specs to ensure inheritance works (child gets parent outs + its own)
        if "_outs_specs" not in owner.__dict__:
            parent_specs = getattr(owner, "_outs_specs", {})
            owner._outs_specs = dict(parent_specs)  # pyright: ignore[reportPrivateUsage] - descriptor needs to access class internals
        owner._outs_specs[name] = OutSpec(  # pyright: ignore[reportPrivateUsage] - descriptor needs to access class internals
            path=self.path, loader=self.loader, descriptor=self
        )

    @overload
    def __get__(self, obj: None, _owner: type[StageDef]) -> Out[T]: ...

    @overload
    def __get__(self, obj: StageDef, _owner: type[StageDef]) -> T: ...

    def __get__(self, obj: StageDef | None, _owner: type[StageDef]) -> Out[T] | T:
        if obj is None:
            return self
        if self._name not in obj._outs_data:  # pyright: ignore[reportPrivateUsage] - descriptor needs to access instance internals
            raise RuntimeError(
                f"Output '{self._name}' not yet assigned. "
                + f"Assign a value to params.{self._name} first."
            )
        return obj._outs_data[self._name]  # pyright: ignore[reportPrivateUsage] - descriptor needs to access instance internals

    def __set__(self, obj: StageDef, value: T) -> None:
        obj._outs_data[self._name] = value  # pyright: ignore[reportPrivateUsage] - descriptor needs to access instance internals


class StageDef(pydantic.BaseModel):
    """Base class for stage definitions with typed deps/outs.

    Use Dep and Out descriptors to declare typed file dependencies:

        class ProcessParams(StageDef):
            # Parameters (regular Pydantic fields)
            threshold: float = 0.5

            # Dependencies (Dep descriptors - type-safe!)
            data: Dep[pandas.DataFrame] = Dep("input.csv", loaders.CSV())

            # Outputs (Out descriptors - type-safe!)
            result: Out[dict[str, int]] = Out("output.json", loaders.JSON())

    In your stage function:

        @stage
        def process(params: ProcessParams) -> None:
            params._load_deps(project_root)

            df = params.data  # Type: pandas.DataFrame (type-checked!)
            params.result = {"count": len(df)}  # Type-checked assignment!

            params._save_outs(project_root)
    """

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
    )

    _deps_specs: ClassVar[dict[str, DepSpec]] = {}
    _outs_specs: ClassVar[dict[str, OutSpec]] = {}

    _deps_data: dict[str, Any] = pydantic.PrivateAttr(default_factory=dict)
    _outs_data: dict[str, Any] = pydantic.PrivateAttr(default_factory=dict)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Copy parent specs to child class to ensure inheritance works
        # (child gets parent deps/outs even if it doesn't define any new ones)
        if "_deps_specs" not in cls.__dict__:
            parent_specs = getattr(cls, "_deps_specs", {})
            cls._deps_specs = dict(parent_specs)
        if "_outs_specs" not in cls.__dict__:
            parent_specs = getattr(cls, "_outs_specs", {})
            cls._outs_specs = dict(parent_specs)

    @override
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize model, excluding Dep/Out descriptor fields."""
        descriptor_names = set(self._deps_specs.keys()) | set(self._outs_specs.keys())
        return super().model_dump(exclude=descriptor_names, **kwargs)

    @override
    def __getattribute__(self, name: str) -> Any:
        # Check if this is a Dep or Out using the specs registry
        # We can't rely on __dict__ because Pydantic moves descriptors
        cls = type(self)
        if name in cls._deps_specs:
            spec = cls._deps_specs[name]
            return spec.descriptor.__get__(self, cls)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType] - descriptor stored in dataclass
        if name in cls._outs_specs:
            spec = cls._outs_specs[name]
            return spec.descriptor.__get__(self, cls)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType] - descriptor stored in dataclass
        # Otherwise, use normal Pydantic attribute access
        return super().__getattribute__(name)

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        # Check if this is an Out using the specs registry
        cls = type(self)
        if name in cls._outs_specs:
            spec = cls._outs_specs[name]
            spec.descriptor.__set__(self, value)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] - descriptor stored in dataclass
            return
        # Otherwise, use normal Pydantic attribute access
        super().__setattr__(name, value)

    def _load_deps(self, project_root: pathlib.Path) -> None:
        """Load all dependencies from disk."""
        loaded_names: list[str] = []
        try:
            for name, spec in self._deps_specs.items():
                self._deps_data[name] = spec.loader.load(project_root / spec.path)
                loaded_names.append(name)
        except Exception:
            for loaded_name in loaded_names:
                self._deps_data.pop(loaded_name, None)
            raise

    def _clear_deps(self) -> None:
        """Clear all loaded dependency data."""
        self._deps_data.clear()

    def _clear_outs(self) -> None:
        """Clear all output data."""
        self._outs_data.clear()

    def _save_outs(self, project_root: pathlib.Path) -> None:
        """Save all outputs to disk."""
        for name in self._outs_specs:
            if name not in self._outs_data:
                msg = f"Output '{name}' was declared but never assigned. "
                msg += f"Assign a value to params.{name} before the stage returns."
                raise RuntimeError(msg)

        for name, spec in self._outs_specs.items():
            full_path = project_root / spec.path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            spec.loader.save(self._outs_data[name], full_path)

    @classmethod
    def get_deps_paths(cls) -> dict[str, str]:
        """Get dict of dependency names to paths."""
        return {name: spec.path for name, spec in cls._deps_specs.items()}

    @classmethod
    def get_outs_paths(cls) -> dict[str, str]:
        """Get dict of output names to paths."""
        return {name: spec.path for name, spec in cls._outs_specs.items()}
