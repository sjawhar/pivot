# Shared Cache Directory Support

## Problem

Pivot stores `.running` files (PID-based execution locks) in the cache directory at `{cache_dir}/{stage}.running`. If two worktrees share a cache directory via config (`cache.dir = ~/.pivot-cache`), these locks prevent concurrent `pivot run` even for independent projects.

The content-addressable cache (`files/`) is safe to share—same content always hashes to the same location. But instance-specific state like execution locks should not be shared.

## Solution

Move `.running` files from `{cache_dir}/` to `{stages_dir}/` (`.pivot/stages/`).

### Before

```
.pivot/
├── state.db/
└── stages/
    └── train.lock        # Persistent stage metadata

.pivot/cache/             # Shared via cache.dir config
├── files/                # Content-addressable (safe to share)
└── train.running         # Execution lock (NOT safe to share)
```

### After

```
.pivot/
├── state.db/
└── stages/
    ├── train.lock        # Persistent stage metadata
    └── train.running     # Execution lock (moved here)

~/.pivot-cache/           # Shared via cache.dir config
└── files/                # Content-addressable only
```

## Changes

### `src/pivot/storage/lock.py`

Rename parameter in two functions:

```python
# Before
def execution_lock(stage_name: str, cache_dir: Path) -> Generator[Path]:
def acquire_execution_lock(stage_name: str, cache_dir: Path) -> Path:

# After
def execution_lock(stage_name: str, stages_dir: Path) -> Generator[Path]:
def acquire_execution_lock(stage_name: str, stages_dir: Path) -> Path:
```

### `src/pivot/executor/worker.py:211`

Update the single caller:

```python
# Before
with lock.execution_lock(stage_name, cache_dir):

# After
with lock.execution_lock(stage_name, lock.get_stages_dir(stage_info["state_dir"])):
```

## Not Changed

Restore temp files and locks (`.pivot_restore_*`) stay in the cache directory. They protect shared cache integrity during concurrent restores of the same hash—this coordination is correct behavior for a shared cache.
