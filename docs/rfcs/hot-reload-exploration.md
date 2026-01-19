# RFC: Hot Reload Exploration

> **GitHub Issue Title:** Explore Hot Reload for Faster Code Change Handling
> **Labels:** `enhancement`, `research`, `future`

## Summary

Explore whether `importlib.reload()` can be used to handle code changes faster than worker restart (~50ms vs ~300ms). This is a **research/exploration issue** to determine feasibility, not a commitment to implement.

## Context

The watch execution engine (see related issue) uses **worker restart** when Python code changes:

```python
loky.get_reusable_executor(kill_workers=True)
```

This is reliable (~99%) but has ~300ms latency. Hot reload via `importlib.reload()` would be ~50ms but has known reliability issues (~80%).

## Known Issues with Hot Reload

The following issues were identified during architecture review. **All must be solved** for hot reload to be production-ready.

### 1. Import Staleness (CRITICAL)

When module A imports module B, reloading B does not update A's reference:

```python
# utils.py
def helper():
    return "v1"

# stages.py
from utils import helper  # Captures reference to v1

# After utils.py reload:
# stages.py still has OLD helper reference!
```

**Impact:** Silent execution of old code despite reload appearing successful.

**Potential solutions:**
- [ ] Reload in topological order (deepest dependencies first)
- [ ] Reload all modules that import the changed module
- [ ] Build import dependency graph and reload transitively
- [ ] Force re-import by manipulating `sys.modules`

### 2. cloudpickle Caching (CRITICAL)

loky uses cloudpickle which caches pickled functions. After reload:
- Function object may have same `id()`
- cloudpickle serves cached pickle with OLD code
- Workers execute old code despite "fresh" workers

**Impact:** Silent execution of stale code.

**Potential solutions:**
- [ ] Use `loky.set_loky_pickler('pickle')` (loses some features)
- [ ] Create new function objects after reload (not just reload modules)
- [ ] Invalidate cloudpickle cache explicitly (if possible)
- [ ] Verify `func.__code__.co_code` changed after reload

### 3. Registry Rollback Non-Atomicity (CRITICAL)

If multi-module reload fails partway:
- Module A reloaded successfully (new code in `sys.modules`)
- Module B fails (SyntaxError)
- Registry rolled back to old state
- But `sys.modules` has new Module A

**Impact:** Registry fingerprints don't match actual code, causing skip/run decisions to be wrong.

**Potential solutions:**
- [ ] Cache module objects before reload, restore on failure
- [ ] Validate all modules can reload before committing any
- [ ] Reload in transaction: snapshot `sys.modules`, rollback on failure
- [ ] Accept inconsistency, force full fingerprint recompute

### 4. Module-Level Side Effects

```python
# stages.py
import expensive_model  # Downloads 2GB model on import

def inference():
    ...
```

Reload triggers the download again.

**Impact:** Unexpected expensive operations on reload.

**Potential solutions:**
- [ ] Document as limitation ("avoid expensive imports")
- [ ] Detect and warn about heavy module-level code
- [ ] No solution - accept as fundamental Python limitation

### 5. Class Instance Identity

```python
# Before reload
class Config:
    pass

config = Config()

# After reload
isinstance(config, Config)  # False! Different class object
```

**Impact:** Type checks fail, captured references become orphaned.

**Potential solutions:**
- [ ] Document limitation ("don't capture class references")
- [ ] Clear module dict before reload (removes old class definitions)
- [ ] No solution - accept as fundamental Python limitation

### 6. `importlib.reload()` Doesn't Clear Module Attributes

Deleted attributes remain in `module.__dict__`:

```python
# Initial: def helper(): ...
# After edit: (helper deleted)
# After reload: module.helper still exists!
```

**Impact:** Fingerprinting may include deleted functions.

**Potential solutions:**
- [ ] Clear module dict (except essentials) before reload
- [ ] Track attribute set before/after, warn on zombies

### 7. Circular Import Handling

If module A imports B and both change, reload order matters:
- Reload A first → re-executes `import B` → gets OLD B
- Need topological sort of import graph

**Impact:** Silent use of old code.

**Potential solutions:**
- [ ] Build import dependency graph
- [ ] Reload in topological order
- [ ] Multi-pass reload (reload all, then reload again)

### 8. C Extensions Cannot Be Reloaded

Modules with C extensions (numpy internals, Cython, etc.) often cannot be reloaded.

**Impact:** Reload fails or has no effect.

**Potential solutions:**
- [ ] Detect C extensions, fall back to worker restart
- [ ] Document limitation

### 9. `__main__` Module Cannot Be Reloaded

Stages defined in the main script cannot be hot-reloaded (no module path for reimport).

**Impact:** Reload fails silently.

**Potential solutions:**
- [ ] Detect and warn when stages are in `__main__`
- [ ] Document limitation

## Research Tasks

### Phase 1: Feasibility Assessment
- [ ] Benchmark reload latency vs worker restart
- [ ] Test import staleness with realistic pipelines
- [ ] Test cloudpickle caching behavior with loky
- [ ] Measure how often users hit the failure cases

### Phase 2: Prototype Solutions
- [ ] Implement import graph builder
- [ ] Implement topological reload
- [ ] Implement `sys.modules` snapshot/restore
- [ ] Test reliability on real-world pipelines

### Phase 3: Decision Point
Based on research:
- If reliability >= 95% AND latency improvement is significant → Implement
- If reliability < 95% OR latency improvement is marginal → Don't implement

## Alternative Approaches

### A. Hybrid: Hot Reload for Data, Restart for Code

Only use hot reload for non-Python changes (data files, params). Always restart workers for `.py` changes. This is conservative but avoids all the Python reload issues.

### B. Process Restart Instead of Worker Restart

Instead of `kill_workers=True`, exec a new coordinator process entirely:

```python
os.execv(sys.executable, [sys.executable] + sys.argv)
```

This is the nuclear option - fresh Python interpreter, all modules reimported, no reload issues. But loses all in-memory state.

### C. Incremental Compilation (Far Future)

Use tools like `mypyc` or Cython to compile stages. Changes trigger recompilation instead of reload. Very complex but potentially solves all issues.

## Success Criteria

Hot reload should only be implemented if:

1. **Reliability >= 95%** on realistic pipelines
2. **Latency improvement is noticeable** (>100ms saved)
3. **Failure modes are detectable** (we know when to fall back)
4. **User impact is minimal** (rare edge cases, clear errors)

## Not In Scope

- Distributed hot reload (multi-machine)
- Hot reload of C extensions
- Hot reload across Python version changes

## Related

- #109 - Watch Execution Engine (main issue)
- #102 - `--force` flag

## References

- [Python importlib.reload() docs](https://docs.python.org/3/library/importlib.html#importlib.reload)
- [cloudpickle caching behavior](https://github.com/cloudpipe/cloudpickle)
- [Hot reloading in Jupyter](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html)
- [Django autoreload implementation](https://github.com/django/django/blob/main/django/utils/autoreload.py)
