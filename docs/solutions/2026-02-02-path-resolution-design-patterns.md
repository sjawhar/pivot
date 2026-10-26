---
date: 2026-02-02
category: design-patterns
tags: [pathlib, path-resolution, symlinks, posix, backslash, brainstorming]
module: storage
symptoms: [symlink-escape, backslash-read-as-separator, input-bindings]
---

# Path Resolution Design Patterns in Python

## Problem

Pipeline code contains almost no paths: stage outputs are placed by `WorkspaceStore._resolve_output_path` (`storage/store.py`) at `data/<pipeline>/<stage>.<ext>`, and inputs are bound by `pipeline.input(name, path=...)` (default `data/raw/<name>`). The remaining questions are how those paths are normalized, where they are resolved, and how symlinks are treated.

## Key Design Decisions

### 1. Paths Are Relative to the Project Root

Input bindings and output locations are relative to the project root (the top-most directory containing `.pivot/`), not to the directory of the `pipeline.py` that declared them. Two pipelines in different directories that both call `pipeline.input("scores.csv")` read the same `data/raw/scores.csv`; binding the same input name to two different paths is an error (`Input binding conflict`).

### 2. Where Resolution Happens

**Bad:** Threading roots and prefixes through multiple layers (Pipeline → Registry → stage_def).

**Good:** One place maps an artifact identity to a path. `WorkspaceStore` resolves both input bindings and output identities; the registry, DAG and lock files only ever hold identities (`ArtifactIdentity(producer, key)`).

### 3. Symlinks: resolve() vs normpath()

**`Path.resolve()`:** Follows symlinks, converts to absolute
- If `data/` is a symlink to `/mnt/external/`, the resolved path is `/mnt/external/file.csv`
- Breaks if symlink target is outside project root

**`os.path.normpath()`:** Normalizes `..` components but preserves symlinks
- `data/file.csv` stays as the symlink path, not the target
- Allows symlinks pointing outside project root

**Chosen:** `os.path.normpath()` - store the symlink path, not the target.

### 4. A Backslash Is Part of a Name

Pivot runs on POSIX only (Linux and macOS), where `/` is the one path separator and a backslash is an ordinary character in a file or directory name. `pathlib.Path` on POSIX already treats it that way: `Path("a\\b.csv").parts == ("a\\b.csv",)`.

Never convert user input with `PureWindowsPath(...).as_posix()` or `.replace("\\", "/")`. Pivot once did both (in `project.normalize_path` and in the `.pivotignore` filter), so a tracked file under a directory named `dir\with\backslash` was looked up at `dir/with/backslash/f.csv`: `pivot status` listed it as missing, `pivot repro` refused to run, `pivot checkout` restored it into a new `dir/with/backslash/` tree, and `pivot track`, `push` and `pull` could not find it by name. A `.pivotignore` pattern follows gitignore syntax, where a backslash escapes the character after it (`a\\b.csv` matches the file `a\b.csv`).

### 5. Path Formats at Different Layers

| Layer | Format | Why |
|-------|--------|-----|
| `pipeline.input(path=...)` bindings | Project-relative | Simple, portable |
| RegistryStageInfo, DAG, lock files | Artifact identities, no paths | Paths are derived, not stored |
| Execution | Absolute (from `WorkspaceStore`) | No ambiguity at runtime |

## Code Patterns

### Normalize path with custom base (no symlink following)

```python
import os.path
from pathlib import Path

def normalize_path(path: str, base: Path) -> Path:
    """Normalize path relative to base, preserving symlinks."""
    p = Path(path)  # a backslash stays part of a name

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

### Test names that hold a backslash

A conversion that splits on a backslash goes unnoticed with ordinary names. Test with a directory and a file whose names hold one, and check that a command finds the file at that path and writes nothing at the path the split gives.

## Invariant: Registration vs Execution Path Validation

Path validation has two distinct phases with different responsibilities:

| Phase | Validates | May call | Must NOT call |
|-------|-----------|----------|---------------|
| **Registration** (loading pipeline.py) | Declared/logical paths — syntax, containment, normalization | `normpath`, `is_relative_to` | `.resolve()`, `.exists()`, `.is_symlink()` |
| **Execution** (running stages) | Filesystem reality — symlink escape, permissions, actual content | `.resolve()`, `.exists()` | N/A |

**Why this matters:** Output files may be symlinks to the cache directory (created by a previous run with symlink checkout mode). If the cache is on a different filesystem (e.g., cross-filesystem worktrees with shared cache), those symlinks point outside the project root. Registration must not reject them — the declared path `data/output.jsonl` is inside the project regardless of what the file currently symlinks to on disk.

**Execution-time defense-in-depth:** `stage_def.py:_validate_path_not_escaped()` checks symlink targets when writing outputs. This is the correct place for filesystem state validation.

## Prevention

When designing path resolution:
1. Keep paths out of the registry: store identities, derive paths in one place (`WorkspaceStore`)
2. Keep resolution in one layer, don't thread through multiple modules
3. Use `normpath` not `resolve` if symlinks should be preserved
4. Never call `.resolve()`, `.exists()`, or `.is_symlink()` on artifact paths at registration time — see invariant above
5. Test with symlinks pointing outside project root (especially cross-filesystem cache scenarios)
6. Test with names that hold a backslash, and never read one as a separator
