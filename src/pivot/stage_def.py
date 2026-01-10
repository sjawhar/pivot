from __future__ import annotations

import pathlib  # noqa: TC003 - used at runtime in _load/_save methods
import sys
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    NamedTuple,
    Protocol,
    cast,
    get_origin,
    get_type_hints,
    overload,
    override,
)

import pydantic

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot import loaders


class DepSpec(NamedTuple):
    """Specification for a dependency."""

    path: str
    loader: loaders.Loader[Any]


class OutSpec(NamedTuple):
    """Specification for an output."""

    path: str
    loader: loaders.Loader[Any]


class _BaseDescriptor[T]:
    """Base descriptor for typed file access with WeakKeyDictionary storage."""

    name: str
    path: str
    loader: loaders.Loader[T]
    _data: weakref.WeakKeyDictionary[object, T]
    _error_template: str

    def __init__(
        self, name: str, path: str, loader: loaders.Loader[T], error_template: str
    ) -> None:
        self.name = name
        self.path = path
        self.loader = loader
        self._data = weakref.WeakKeyDictionary()
        self._error_template = error_template

    @overload
    def __get__(self, obj: None, owner: type[object]) -> _BaseDescriptor[T]: ...

    @overload
    def __get__(self, obj: object, owner: type[object]) -> T: ...

    def __get__(self, obj: object | None, owner: type[object]) -> _BaseDescriptor[T] | T:
        if obj is None:
            return self
        if obj not in self._data:
            raise RuntimeError(self._error_template.format(name=self.name))
        return self._data[obj]

    def _clear(self, obj: object) -> None:
        """Clear data for the given instance."""
        self._data.pop(obj, None)


class DepDescriptor[T](_BaseDescriptor[T]):
    """Descriptor for typed dependency access."""

    def __init__(self, name: str, path: str, loader: loaders.Loader[T]) -> None:
        super().__init__(
            name, path, loader, "Dependency '{name}' not loaded - ensure _load_deps() was called"
        )

    def _load(self, obj: object, project_root: pathlib.Path) -> None:
        """Load data for the given instance."""
        full_path = project_root / self.path
        data = self.loader.load(full_path)
        self._data[obj] = data


class OutDescriptor[T](_BaseDescriptor[T]):
    """Descriptor for typed output access."""

    def __init__(self, name: str, path: str, loader: loaders.Loader[T]) -> None:
        super().__init__(
            name, path, loader, "Output '{name}' not assigned - set it before calling _save_outs()"
        )

    def __set__(self, obj: object, value: T) -> None:
        self._data[obj] = value
        # Track assignment on the StageDef instance
        parent: StageDef | None = getattr(obj, "_parent_stage_def", None)
        if parent is not None:
            parent._assigned_outs.add(self.name)  # pyright: ignore[reportPrivateUsage] - internal tracking

    def _save(self, obj: object, project_root: pathlib.Path) -> None:
        """Save data for the given instance."""
        if obj not in self._data:
            raise RuntimeError(f"Output '{self.name}' was never assigned")
        full_path = project_root / self.path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        self.loader.save(self._data[obj], full_path)


def _create_loader_from_annotation(
    annotation: type[Any],
) -> loaders.Loader[Any]:
    """Create a loader instance from a type annotation like CSV[DataFrame]."""
    from pivot import loaders

    origin = get_origin(annotation)
    if origin is None:
        # Direct class reference (e.g., PathOnly without type param)
        if issubclass(annotation, loaders.Loader):
            return annotation()  # pyright: ignore[reportUnknownVariableType] - generic loader
        raise TypeError(f"Invalid loader annotation: {annotation}")

    # Generic type like CSV[DataFrame]
    if isinstance(origin, type) and issubclass(origin, loaders.Loader):
        return origin()  # pyright: ignore[reportUnknownVariableType] - generic loader

    raise TypeError(f"Invalid loader annotation: {annotation}")


def _process_nested_class(
    nested_cls: type[object],
    descriptor_cls: type[DepDescriptor[Any]] | type[OutDescriptor[Any]],
    spec_cls: type[DepSpec] | type[OutSpec],
    parent_globals: dict[str, Any],
) -> tuple[dict[str, DepSpec | OutSpec], type[object]]:
    """Process a nested deps/outs class and create descriptors."""
    specs: dict[str, DepSpec | OutSpec] = {}

    # Get annotations from the nested class, resolving string annotations
    try:
        annotations = get_type_hints(nested_cls, globalns=parent_globals)
    except NameError:
        # Forward reference couldn't be resolved - fall back to raw annotations
        raw_annotations = getattr(nested_cls, "__annotations__", {})
        # Check for unresolved string annotations
        for name, ann in raw_annotations.items():
            if isinstance(ann, str):
                msg = f"Cannot resolve type annotation '{ann}' for '{name}'. "
                msg += "Ensure all loader types are imported."
                raise TypeError(msg) from None
        annotations = raw_annotations

    # Create a new namespace class with descriptors
    namespace_dict: dict[str, DepDescriptor[Any] | OutDescriptor[Any] | Callable[..., None]] = {}

    for name, annotation in annotations.items():
        # Get default value (the path string)
        default = getattr(nested_cls, name, None)
        if default is None:
            raise ValueError(f"Missing default path for '{name}'")

        path = str(default)
        loader = _create_loader_from_annotation(annotation)

        specs[name] = spec_cls(path=path, loader=loader)
        descriptor = descriptor_cls(name=name, path=path, loader=loader)
        namespace_dict[name] = descriptor

    # Add __init__ that takes parent reference
    def namespace_init(self: _NamespaceInstance) -> None:
        self._parent_stage_def = None  # pyright: ignore[reportPrivateUsage] - protocol attribute

    namespace_dict["__init__"] = namespace_init

    # Create namespace class dynamically
    namespace_cls = type(f"{nested_cls.__name__}Namespace", (object,), namespace_dict)

    return specs, namespace_cls


class _NamespaceInstance(Protocol):
    """Protocol for namespace instances that track their parent StageDef."""

    _parent_stage_def: StageDef | None


class _NamespaceDescriptor:
    """Descriptor for accessing deps/outs namespace from class or instance."""

    _cls_attr: str
    _instance_attr: str
    _name: str

    def __init__(self, cls_attr: str, instance_attr: str, name: str) -> None:
        self._cls_attr = cls_attr
        self._instance_attr = instance_attr
        self._name = name

    @overload
    def __get__(self, obj: None, owner: type[StageDef]) -> type[object]: ...

    @overload
    def __get__(self, obj: StageDef, owner: type[StageDef]) -> _NamespaceInstance: ...

    def __get__(
        self, obj: StageDef | None, owner: type[StageDef]
    ) -> type[object] | _NamespaceInstance:
        if obj is None:
            # Class access - return the namespace class with descriptors
            namespace_cls: type[object] | None = getattr(owner, self._cls_attr)
            if namespace_cls is None:
                raise AttributeError(f"No {self._name} defined for this StageDef")
            return namespace_cls
        # Instance access - return the instance namespace
        namespace_instance: _NamespaceInstance | None = getattr(obj, self._instance_attr)
        if namespace_instance is None:
            raise AttributeError(f"No {self._name} defined for this StageDef")
        return namespace_instance


class StageDef(pydantic.BaseModel):
    """Base class for stage definitions with typed deps/outs."""

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=(_NamespaceDescriptor,),
    )

    _deps_specs: ClassVar[dict[str, DepSpec]] = {}
    _outs_specs: ClassVar[dict[str, OutSpec]] = {}
    _deps_namespace_cls: ClassVar[type[object] | None] = None
    _outs_namespace_cls: ClassVar[type[object] | None] = None

    _deps_instance: _NamespaceInstance | None = pydantic.PrivateAttr(default=None)
    _outs_instance: _NamespaceInstance | None = pydantic.PrivateAttr(default=None)
    _assigned_outs: set[str] = pydantic.PrivateAttr(default_factory=set)

    # Use parameterized descriptors for deps/outs access
    deps: ClassVar[_NamespaceDescriptor] = _NamespaceDescriptor(
        "_deps_namespace_cls", "_deps_instance", "dependencies"
    )
    outs: ClassVar[_NamespaceDescriptor] = _NamespaceDescriptor(
        "_outs_namespace_cls", "_outs_instance", "outputs"
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        module = sys.modules.get(cls.__module__)
        parent_globals = vars(module) if module else {}

        # Process nested deps class if defined
        nested_deps = cls.__dict__.get("deps")
        if nested_deps is not None and isinstance(nested_deps, type):
            specs, namespace_cls = _process_nested_class(
                nested_deps, DepDescriptor, DepSpec, parent_globals
            )
            cls._deps_specs = cast("dict[str, DepSpec]", specs)
            cls._deps_namespace_cls = namespace_cls
            delattr(cls, "deps")
        elif not hasattr(cls, "_deps_specs") or cls._deps_specs is StageDef._deps_specs:
            cls._deps_specs = {}
            cls._deps_namespace_cls = None

        # Process nested outs class if defined
        nested_outs = cls.__dict__.get("outs")
        if nested_outs is not None and isinstance(nested_outs, type):
            specs, namespace_cls = _process_nested_class(
                nested_outs, OutDescriptor, OutSpec, parent_globals
            )
            cls._outs_specs = cast("dict[str, OutSpec]", specs)
            cls._outs_namespace_cls = namespace_cls
            delattr(cls, "outs")
        elif not hasattr(cls, "_outs_specs") or cls._outs_specs is StageDef._outs_specs:
            cls._outs_specs = {}
            cls._outs_namespace_cls = None

    @override
    def model_post_init(self, context: Any) -> None:
        """Initialize deps and outs namespace instances after pydantic init."""
        if self._deps_namespace_cls is not None:
            self._deps_instance = self._deps_namespace_cls()  # pyright: ignore[reportAttributeAccessIssue] - dynamic namespace class
            self._deps_instance._parent_stage_def = self  # pyright: ignore[reportOptionalMemberAccess,reportPrivateUsage] - just assigned above, protocol attribute

        if self._outs_namespace_cls is not None:
            self._outs_instance = self._outs_namespace_cls()  # pyright: ignore[reportAttributeAccessIssue] - dynamic namespace class
            self._outs_instance._parent_stage_def = self  # pyright: ignore[reportOptionalMemberAccess,reportPrivateUsage] - just assigned above, protocol attribute

    def _load_deps(self, project_root: pathlib.Path) -> None:
        """Load all dependencies from disk."""
        if self._deps_namespace_cls is None or self._deps_instance is None:
            return

        loaded_names: list[str] = []
        try:
            for name in self._deps_specs:
                descriptor = cast("DepDescriptor[Any]", getattr(self._deps_namespace_cls, name))
                descriptor._load(self._deps_instance, project_root)  # pyright: ignore[reportPrivateUsage] - internal API
                loaded_names.append(name)
        except Exception:
            # Clear any partially loaded state
            for loaded_name in loaded_names:
                descriptor = cast(
                    "DepDescriptor[Any]", getattr(self._deps_namespace_cls, loaded_name)
                )
                descriptor._clear(self._deps_instance)  # pyright: ignore[reportPrivateUsage] - internal API
            raise

    def _clear_deps(self) -> None:
        """Clear all loaded dependency data."""
        if self._deps_namespace_cls is None or self._deps_instance is None:
            return
        for name in self._deps_specs:
            descriptor = cast("DepDescriptor[Any]", getattr(self._deps_namespace_cls, name))
            descriptor._clear(self._deps_instance)  # pyright: ignore[reportPrivateUsage] - internal API

    def _clear_outs(self) -> None:
        """Clear all output data."""
        if self._outs_namespace_cls is None or self._outs_instance is None:
            return
        for name in self._outs_specs:
            descriptor = cast("OutDescriptor[Any]", getattr(self._outs_namespace_cls, name))
            descriptor._clear(self._outs_instance)  # pyright: ignore[reportPrivateUsage] - internal API

    def _save_outs(self, project_root: pathlib.Path) -> None:
        """Save all outputs to disk."""
        if self._outs_namespace_cls is None or self._outs_instance is None:
            return

        # Check all outputs were assigned
        for name in self._outs_specs:
            if name not in self._assigned_outs:
                msg = f"Output '{name}' was declared but never assigned. "
                msg += f"Assign a value to params.outs.{name} before the stage function returns."
                raise RuntimeError(msg)

        for name in self._outs_specs:
            descriptor = cast("OutDescriptor[Any]", getattr(self._outs_namespace_cls, name))
            descriptor._save(self._outs_instance, project_root)  # pyright: ignore[reportPrivateUsage] - internal API

    @classmethod
    def get_deps_paths(cls) -> dict[str, str]:
        """Get dict of dependency names to paths."""
        return {name: spec.path for name, spec in cls._deps_specs.items()}

    @classmethod
    def get_outs_paths(cls) -> dict[str, str]:
        """Get dict of output names to paths."""
        return {name: spec.path for name, spec in cls._outs_specs.items()}
