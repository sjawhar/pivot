---
date: 2026-02-02
category: design-patterns
tags: [pathlib, path-resolution, symlinks, cross-platform, brainstorming]
module: pipeline
symptoms: [verbose-paths, stage-reuse, dvc-migration]
---

# Path Resolution Design Patterns in Python

## Problem

When designing a pipeline framework where stages can be reused across different pipeline directories, how should paths in annotations be resolved? The goal is to allow simple relative paths like `Dep("data/raw.csv")` that resolve differently depending on which pipeline registers the stage.

## Key Design Decisions

### 1. Implicit vs Explicit Path Prefix

**Implicit (chosen):** Infer prefix from `pipeline.root` (directory containing `pipeline.py`)
```python
# eval_pipeline/horizon/pipeline.py
pipeline = Pipeline("horizon")  # root inferred from __file__
Dep("data/raw.csv")  # → eval_pipeline/horizon/data/raw.csv
```

**Explicit alternative:** User specifies prefix manually
```python
pipeline = Pipeline("horizon", path_prefix="eval_pipeline/horizon/")
```

**Tradeoff:** Implicit follows DVC convention (familiar to target users) but adds "magic". Explicit is clearer but more verbose.

### 2. Where Resolution Happens

**Bad:** Threading `pipeline_root` through multiple layers (Pipeline → Registry → stage_def)
- Spreads the feature across many files
- Hard to understand and maintain

**Good:** Resolve entirely in `Pipeline.register()`
- Extract annotation paths, resolve them, pass as overrides
- Registry and stage_def remain unchanged
- Feature contained in one place

### 3. Symlinks: resolve() vs normpath()

**`Path.resolve()`:** Follows symlinks, converts to absolute
- If `data/` is a symlink to `/mnt/external/`, the resolved path is `/mnt/external/file.csv`
- Breaks if symlink target is outside project root

**`os.path.normpath()`:** Normalizes `..` components but preserves symlinks
- `data/file.csv` stays as the symlink path, not the target
- Allows symlinks pointing outside project root

**Chosen:** `os.path.normpath()` - store the symlink path, not the target.

### 4. Windows Path Handling

Python's `pathlib.Path` on Unix treats backslashes as literal characters, not separators.

**Solution:** Use `PureWindowsPath` which parses both `/` and `\` as separators on any platform:
```python
from pathlib import PureWindowsPath

def normalize_to_posix(path: str) -> str:
    return PureWindowsPath(path).as_posix()

normalize_to_posix("foo\\bar")     # → "foo/bar"
normalize_to_posix("foo/bar")      # → "foo/bar"
normalize_to_posix("foo\\bar/baz") # → "foo/bar/baz"
```

### 5. Path Formats at Different Layers

| Layer | Format | Why |
|-------|--------|-----|
| Annotations | Pipeline-relative | Simple, reusable |
| RegistryStageInfo | Project-relative | Consistent for DAG building |
| Lock files | Project-relative | Portable, git-trackable |
| Execution | Absolute | No ambiguity at runtime |

Resolution happens once at registration time. All downstream code sees project-relative paths.

## Code Patterns

### Normalize path with custom base (no symlink following)

```python
import os.path
from pathlib import Path, PureWindowsPath

def normalize_path(path: str, base: Path) -> Path:
    """Normalize path relative to base, preserving symlinks."""
    # Handle Windows paths
    posix_path = PureWindowsPath(path).as_posix()
    p = Path(posix_path)

    # Make absolute from base
    abs_path = p.absolute() if p.is_absolute() else (base / p).absolute()

    # Normalize .. without following symlinks
    return Path(os.path.normpath(abs_path))
```

### Convert to project-relative

```python
def to_project_relative(abs_path: Path, project_root: Path) -> str:
    """Convert absolute path to project-relative string."""
    try:
        return str(abs_path.relative_to(project_root))
    except ValueError:
        # Path is outside project root - return as-is or error
        return str(abs_path)
```

## Gotchas

### pathlib has no normpath equivalent

`Path.resolve()` normalizes AND follows symlinks. `Path.absolute()` makes absolute but doesn't normalize `..`. There's no built-in way to normalize without following symlinks.

**Workaround:** `Path(os.path.normpath(path.absolute()))`

### PurePosixPath doesn't parse backslashes

On Unix, `PurePosixPath("foo\\bar")` has one component `"foo\\bar"`, not two.

**Solution:** Always use `PureWindowsPath` for parsing user input, then `.as_posix()` for internal use.

## Prevention

When designing path resolution:
1. Decide early: implicit vs explicit prefix
2. Keep resolution in one layer, don't thread through multiple modules
3. Use `normpath` not `resolve` if symlinks should be preserved
4. Test with symlinks pointing outside project root
5. Test with Windows-style paths if accepting user input
