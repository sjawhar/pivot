---
title: "refactor: Split Loader into Reader/Writer"
type: refactor
date: 2026-01-31
issue: https://github.com/sjawhar/pivot/issues/237
---

# refactor: Split Loader into Reader/Writer

## Overview

Split the `Loader[T]` ABC into separate `Reader[R]` and `Writer[W]` base classes, enabling asymmetric formats where write and read types differ (e.g., write `Figure` to PNG, read back as `np.ndarray`).

## Problem Statement

1. `MatplotlibFigure.load()` raises `NotImplementedError` - a runtime hack
2. Can't have loaders with different read/write types (asymmetric formats)
3. The type system should prevent misuse at type-check time

## Proposed Solution

### New Class Hierarchy

Using [PEP 696](https://peps.python.org/pep-0696/) type parameter defaults (Python 3.13+):

```python
@dataclasses.dataclass(frozen=True)
class Reader[R](abc.ABC):
    """Read-only - can load data from a file."""
    @abc.abstractmethod
    def load(self, path: Path) -> R: ...

@dataclasses.dataclass(frozen=True)
class Writer[W](abc.ABC):
    """Write-only - can save data to a file."""
    @abc.abstractmethod
    def save(self, data: W, path: Path) -> None: ...

@dataclasses.dataclass(frozen=True)
class Loader[W, R = W](Writer[W], Reader[R]):
    """Bidirectional - can save W and load R (defaults to same type)."""
    def empty(self) -> R:
        raise NotImplementedError(f"{type(self).__name__} doesn't support empty()")
```

The `R = W` default means symmetric loaders only need one type parameter.

### Example Usage

```python
# Symmetric loader - single type param uses default (R = W)
class CSV(Loader[DataFrame]):
    def save(self, data: DataFrame, path: Path) -> None: ...
    def load(self, path: Path) -> DataFrame: ...

# Asymmetric loader - explicit both types
class PngImage(Loader[Figure, np.ndarray]):
    """Write matplotlib Figure, read back as numpy array."""
    def save(self, data: Figure, path: Path) -> None:
        data.savefig(path)
        plt.close(data)

    def load(self, path: Path) -> np.ndarray:
        from PIL import Image
        return np.array(Image.open(path))

# Write-only (no Reader inheritance)
class MatplotlibFigure(Writer[Figure]):
    def save(self, data: Figure, path: Path) -> None: ...
```

**What we're NOT doing** (per review feedback):
- No PEP8 renames (`CSV` stays `CSV`, not `Csv`)
- No attribute renames (`.loader` stays `.loader`)
- No runtime validation (type checker handles it)

## Technical Approach

### Phase 1: Add ABCs and Update Loaders

**File:** `src/pivot/loaders.py`

1. Add `Reader[R]` ABC with `load() -> R`
2. Add `Writer[W]` ABC with `save(data: W, ...)`
3. Change `Loader[T]` to `Loader[W, R](Writer[W], Reader[R])`
4. Move `empty() -> R` to `Loader`
5. Update existing loaders (minimal changes due to default):

| Loader | Before | After | Notes |
|--------|--------|-------|-------|
| `CSV` | `Loader[DataFrame]` | `Loader[DataFrame]` | No change (default R=W) |
| `JSON` | `Loader[T]` | `Loader[T]` | No change |
| `YAML` | `Loader[T]` | `Loader[T]` | No change |
| `Text` | `Loader[str]` | `Loader[str]` | No change |
| `JSONL` | `Loader[list[dict]]` | `Loader[list[dict]]` | No change |
| `DataFrameJSONL` | `Loader[DataFrame]` | `Loader[DataFrame]` | No change |
| `Pickle` | `Loader[T]` | `Loader[T]` | No change |
| `PathOnly` | `Loader[Path]` | `Loader[Path]` | No change |
| `MatplotlibFigure` | `Loader[Figure]` | `Writer[Figure]` | Now write-only |

6. (Optional) Add `PngImage(Loader[Figure, np.ndarray])` for asymmetric example

### Phase 2: Update Type Annotations

**File:** `src/pivot/outputs.py`

Change the type of `.loader` attributes:

```python
@dataclasses.dataclass(frozen=True)
class Dep(Generic[R]):
    path: PathType
    loader: Reader[R]  # Only needs to read

@dataclasses.dataclass(frozen=True)
class Out(Generic[W]):
    path: PathType
    loader: Writer[W]  # Only needs to write
    cache: bool = True

@dataclasses.dataclass(frozen=True)
class PlaceholderDep(Generic[R]):
    path: PathType
    loader: Reader[R]
```

**IncrementalOut special case:**

```python
@dataclasses.dataclass(frozen=True)
class IncrementalOut(Generic[W, R]):
    """Incremental output - writes W, reads R."""
    path: PathType
    loader: Loader[W, R]  # Needs both read and write
    cache: bool = True
```

For symmetric use (most common), users write `IncrementalOut[T]` (R defaults to W).

**DirectoryOut (type inconsistency fix):**

`DirectoryOut` has an existing type inconsistency: it inherits from `Out[dict[str, T]]` (so `loader: Loader[dict[str, T]]`), but actually uses the loader to write individual values of type `T`:

```python
# Current (type lie)
class DirectoryOut(Out[dict[str, T]]):
    # Inherits loader: Loader[dict[str, T]] from Out
    # But code does: loader.save(value_of_type_T, path)  # Type mismatch!
```

**Fix: Extract common base class**

```python
@dataclasses.dataclass(frozen=True)
class BaseOut(Generic[T]):
    """Common fields for output specs. T is the stage return type."""
    path: PathType
    cache: bool = True

@dataclasses.dataclass(frozen=True)
class Out(BaseOut[T]):
    """Single file output. Loader writes T."""
    loader: Writer[T]

@dataclasses.dataclass(frozen=True)
class DirectoryOut(BaseOut[dict[str, T]]):
    """Directory output. Stage returns dict[str, T], loader writes individual T values."""
    loader: Writer[T]  # Writes T, not dict[str, T]

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.endswith("/"):
            raise ValueError(...)
```

This way:
- `Out[T]` has `loader: Writer[T]` - type-correct
- `DirectoryOut[T]` extends `BaseOut[dict[str, T]]` (stage returns dict) but has `loader: Writer[T]` (writes items)
- Shared `path` and `cache` fields in `BaseOut`

**Other Out subclasses:**
- `Metric(Out[JsonValue])` - stays as `Out` subclass, inherits `loader: Writer[JsonValue]`
- `Plot(Out[T])` - stays as `Out` subclass, inherits `loader: Writer[T]`
- `IncrementalOut` - needs `Loader[W, R]` (bidirectional), see below

### Phase 3: Update Fingerprinting

**File:** `src/pivot/fingerprint.py`

```python
def get_loader_fingerprint(loader: Writer[Any] | Reader[Any]) -> dict[str, str]:
    """Generate fingerprint manifest for a loader instance."""
    manifest = dict[str, str]()
    name = type(loader).__name__

    if isinstance(loader, Writer):
        manifest[f"loader:{name}:save"] = hash_function_ast(loader.save)

    if isinstance(loader, Reader):
        manifest[f"loader:{name}:load"] = hash_function_ast(loader.load)
        # Only fingerprint empty() if overridden (use MRO check)
        if isinstance(loader, Loader):
            for cls in type(loader).__mro__:
                if cls is Loader:
                    break
                if "empty" in cls.__dict__:
                    manifest[f"loader:{name}:empty"] = hash_function_ast(loader.empty)
                    break

    # Config hash from dataclass fields (unchanged)
    field_values = [f"{f.name}={getattr(loader, f.name)!r}" for f in dataclasses.fields(loader)]
    if field_values:
        manifest[f"loader:{name}:config"] = xxhash.xxh64(",".join(field_values).encode()).hexdigest()

    return manifest
```

### Phase 4: Update Tests

- Update loader tests for two type parameters
- Add test for asymmetric loader (`Loader[Figure, np.ndarray]`)
- Verify `MatplotlibFigure` with `Dep` fails type-check
- Test `IncrementalOut` with both symmetric and asymmetric loaders

## Acceptance Criteria

- [ ] `Reader[R]` ABC with `load() -> R`
- [ ] `Writer[W]` ABC with `save(data: W, ...)`
- [ ] `Loader[W, R](Writer[W], Reader[R])` with `empty() -> R`
- [ ] Existing symmetric loaders use `Loader[T]` (R defaults to W)
- [ ] `MatplotlibFigure` is `Writer[Figure]` (no `load()` method)
- [ ] `Dep.loader: Reader[R]` type annotation
- [ ] `Out.loader: Writer[W]` type annotation
- [ ] `IncrementalOut.loader: Loader[W, R]`
- [ ] `BaseOut` extracted with shared `path` and `cache` fields
- [ ] `Out` and `DirectoryOut` both inherit from `BaseOut`
- [ ] `DirectoryOut.loader: Writer[T]` (writes items, not dict)
- [ ] Fingerprinting handles all cases
- [ ] All tests pass
- [ ] `basedpyright .` passes

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Two type params for Loader | `Loader[W, R = W]` | Enables asymmetric formats, defaults to symmetric |
| Parameter order | Write first, Read second | Matches dataflow: save(W) → load() → R |
| PEP 696 default | `R = W` | Symmetric loaders need only one type param |
| Keep `.loader` attribute name | Yes | Type annotation provides constraint |
| PEP8 renames | No | Unrelated to core problem |
| Runtime validation | No | Type checker handles it |
| `empty()` returns `R` | Yes | `empty()` provides initial read value |

## Open Questions

### IncrementalOut Inheritance

`IncrementalOut` currently inherits from `Out`. With the new design:
- `Out[W]` has `loader: Writer[W]`
- `IncrementalOut[W, R]` has `loader: Loader[W, R]`

Options:
1. **Don't inherit** - Make `IncrementalOut` standalone (some duplication)
2. **Override type** - Works because `Loader[W, R]` is a subtype of `Writer[W]`

Recommendation: Option 2 if type checker accepts it, otherwise Option 1.

## Files Changed

```
src/pivot/loaders.py      # ABCs + all loader type param updates
src/pivot/outputs.py      # Type annotation changes
src/pivot/fingerprint.py  # isinstance-based fingerprinting
tests/                    # Test updates
```

## Migration & Breaking Changes

**This is a pre-alpha project. Breaking changes are acceptable.**

Most changes are backwards-compatible due to PEP 696 defaults:

```python
# Before AND After - no change needed
class MyLoader(Loader[DataFrame]): ...
```

**Breaking changes:**
- `DirectoryOut` no longer inherits from `Out` (uses `BaseOut` instead)
- `IncrementalOut` becomes `IncrementalOut[W, R]` (may require explicit type params)
- `MatplotlibFigure` is now `Writer[Figure]` (can't be used with `Dep` - but this was broken anyway)
- Lock files will be invalidated (full re-run on first use after upgrade)

These are acceptable for pre-alpha. No migration path or deprecation warnings needed.

## References

- **Issue:** [#237](https://github.com/sjawhar/pivot/issues/237)
- **Brainstorm:** `docs/brainstorms/2026-01-31-reader-writer-split-brainstorm.md`
- **Current loaders:** `src/pivot/loaders.py:22-328`
