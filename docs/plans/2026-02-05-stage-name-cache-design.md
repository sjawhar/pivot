# Stage Name Cache for Fast Tab-Completion

## Problem

Tab-completion for `pivot repro <TAB>` takes ~2.5 seconds for `pipeline.py` pipelines because it triggers full discovery, which imports heavy dependencies (pandas, sklearn, etc.).

**Metrics breakdown (108-stage pipeline):**
- `discovery.load_module`: 2835ms total
  - Module imports (pandas, sklearn, etc.): ~2170ms (76%)
  - `registry.register` (includes fingerprinting): ~666ms (24%)

The existing completion system has a fast path for `pivot.yaml` (~10ms) but `pipeline.py` users always hit the slow fallback.

## Solution

Cache stage names to `.pivot/cache/stages.cache` after discovery. Tab-completion reads the cache when fresh, avoiding Python imports entirely.

**Target:** <50ms for cached tab-completion (actual: ~0.5ms for cache read + ~30-50ms Python startup).

## Cache Format

Location: `.pivot/cache/stages.cache`

```
v1
pipeline.py:1707123456.789
train
preprocess
evaluate
deploy
```

- Line 1: Version tag (`v1`) for future format changes (unknown versions = cache miss)
- Line 2: `<config_file>:<mtime>` — config path (relative) + mtime as float seconds
- Lines 3+: Stage names, one per line

**Why text format:**
- File is tiny (<1KB for hundreds of stages)
- Human-debuggable
- No parsing library dependencies
- Fast: ~0.5ms to read and parse

## Implementation

### Helper: detect config file

```python
def _detect_config_file(root: pathlib.Path) -> tuple[str, float] | None:
    """Detect config file and capture its mtime.

    Returns (relative_path, mtime) or None if no config found.
    """
    for name in ("pivot.yaml", "pivot.yml", "pipeline.py"):
        path = root / name
        if path.exists():
            return (name, path.stat().st_mtime)
    return None
```

### Reading the cache

```python
def _get_stages_from_cache(root: pathlib.Path) -> list[str] | None:
    """Read stage names from cache if fresh."""
    cache_path = root / ".pivot" / "cache" / "stages.cache"
    try:
        lines = cache_path.read_text().splitlines()
    except FileNotFoundError:
        return None

    if len(lines) < 2 or lines[0] != "v1":
        return None

    header = lines[1]
    sep_idx = header.rfind(":")
    if sep_idx == -1:
        return None

    config_file, mtime_str = header[:sep_idx], header[sep_idx + 1:]

    try:
        current_mtime = (root / config_file).stat().st_mtime
    except FileNotFoundError:
        return None

    if str(current_mtime) != mtime_str:
        return None

    return lines[2:]
```

### Writing the cache

```python
def _write_stages_cache(
    root: pathlib.Path, config_file: str, mtime: float, stages: list[str]
) -> None:
    """Write stage names to cache (atomic)."""
    cache_dir = root / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "stages.cache"
    tmp_path = cache_path.with_suffix(".tmp")

    content = f"v1\n{config_file}:{mtime}\n" + "\n".join(stages) + "\n"

    tmp_path.write_text(content)
    tmp_path.rename(cache_path)  # atomic on POSIX
```

### Integration with completion.py

Update `_get_stages_fast()` to check cache first:

```python
def _get_stages_fast() -> list[str] | None:
    root = _find_project_root_fast()
    if root is None:
        return None

    # Try cache first (works for both YAML and pipeline.py)
    if (stages := _get_stages_from_cache(root)) is not None:
        return stages

    # Existing YAML fast-path continues below...
```

Update `_get_stages_full()` to write cache after discovery.

**Important:** Capture mtime *before* discovery to avoid race condition where config
file changes during slow discovery (~2.5s). Only write cache if mtime unchanged.

```python
def _get_stages_full() -> list[str]:
    from pivot import discovery, project

    # Capture config file and mtime BEFORE discovery (race condition prevention)
    root = _find_project_root_fast()
    pre_discovery = _detect_config_file(root) if root else None

    pipeline = discovery.discover_pipeline()
    if pipeline is None:
        return []

    stages = pipeline.list_stages()

    # Write cache only if config file unchanged during discovery
    if root and pre_discovery:
        config_file, pre_mtime = pre_discovery
        try:
            post_mtime = (root / config_file).stat().st_mtime
            if pre_mtime == post_mtime:
                _write_stages_cache(root, config_file, pre_mtime, stages)
        except OSError:
            pass  # Cache write failure is non-fatal

    return stages
```

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| No `.pivot/` directory | Cache miss → full discovery → creates cache |
| Cache file corrupted/malformed | Cache miss → full discovery → overwrites cache |
| `pipeline.py` deleted | Cache mtime check fails → cache miss |
| `pipeline.py` edited | Mtime changes → cache miss → rediscover |
| New clone (different mtimes) | Cache miss → full discovery |
| Concurrent tab-completions | Atomic write prevents torn reads |

## Known Limitations

**Imported modules not tracked:** The cache only checks the entry point file's mtime (`pipeline.py`). If stages are defined in imported modules (e.g., `stages/train.py`), edits there won't invalidate the cache.

This is acceptable because:
- Tab-completion only needs stage *names*, not fingerprints
- Stage names rarely change in imported modules
- Full commands (`pivot repro`) still do proper discovery

**Manual workaround:** If you rename stages in imported modules and need to refresh completions:
```bash
touch pipeline.py           # Invalidates cache via mtime change
# or
rm .pivot/cache/stages.cache  # Direct cache removal
```

## Files to Change

1. `src/pivot/cli/completion.py` — Add cache read/write functions, integrate with existing completion

## Not in Scope

- Escape hatch CLI command (manual workaround sufficient)
- Caching fingerprints or other stage metadata
- Tracking imported module mtimes
- Faster full discovery (that's a separate effort)
