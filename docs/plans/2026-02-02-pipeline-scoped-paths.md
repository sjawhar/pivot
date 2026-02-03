# Pipeline-Scoped Path Resolution

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow stage annotations to use simple relative paths that resolve relative to the pipeline's root directory, enabling shared stage functions across pipelines without verbose path overrides.

**Architecture:** Paths in annotations are resolved relative to `pipeline.root` at registration time in `Pipeline.register()`, then passed to the registry as overrides. Cross-pipeline references use `../` to traverse up.

**Tech Stack:** Python pathlib, existing `pivot.project` utilities

**Future work:** Cross-pipeline dependency resolution with auto-discovery (#323)

---

## Background

Currently, annotation paths are project-relative:
```python
# In eval_pipeline/horizon/pipeline.py
Dep("eval_pipeline/horizon/data/raw.csv", CSV())  # Full path required
```

When reusing stages across pipelines, verbose `dep_path_overrides` are needed:
```python
# In eval_pipeline/ga_paper/pipeline.py
pipeline.register(
    shared_stage,
    dep_path_overrides={"data": "eval_pipeline/ga_paper/data/raw.csv"},
)
```

## Proposed Behavior

Paths resolve relative to the pipeline's root directory (inferred from the directory containing `pipeline.py`):

```python
# eval_pipeline/horizon/pipeline.py
pipeline = Pipeline("horizon")  # root inferred from __file__ → eval_pipeline/horizon/

def compute_weights(
    data: Annotated[DataFrame, Dep("data/raw.csv", CSV())]
) -> Annotated[DataFrame, Out("data/weights.csv", CSV())]:
    ...

pipeline.register(compute_weights)
# Dep → eval_pipeline/horizon/data/raw.csv (project-relative)
# Out → eval_pipeline/horizon/data/weights.csv (project-relative)
```

Cross-pipeline references use `../`:
```python
# eval_pipeline/ga_paper/pipeline.py
pipeline = Pipeline("ga_paper")

def analyze(
    weights: Annotated[DataFrame, Dep("../horizon/data/weights.csv", CSV())],
) -> ...:
    ...
```

Absolute paths are passed through unchanged:
```python
Dep("/shared/datasets/common.csv", CSV())  # Absolute: used as-is
```

Path overrides (`dep_path_overrides`, `out_path_overrides`) are also pipeline-relative:
```python
pipeline.register(
    shared_stage,
    dep_path_overrides={"data": "custom/input.csv"},  # → eval_pipeline/horizon/custom/input.csv
)
```

---

## Implementation

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Resolution location | `Pipeline.register()` | Keeps feature contained; registry/stage_def unchanged |
| Path validation timing | After normalization | Allows `../` in input; validates resolved project-relative path |
| Annotation extraction | Reuse existing `get_dep_specs_from_signature()` | Avoids duplicate parsing logic |
| Windows path support | Yes (`PureWindowsPath`) | Handles paths from Windows tools/configs |

### Critical: Path Validation Order

The existing `path_policy.validate_path_syntax()` rejects `..` traversal. To allow `../` in annotations while still validating the final path:

1. **Input:** `../horizon/data/weights.csv` (pipeline-relative, contains `..`)
2. **Normalize:** Resolve relative to pipeline root, collapse `..`
3. **Result:** `eval_pipeline/horizon/data/weights.csv` (project-relative, no `..`)
4. **Validate:** Run `path_policy` checks on the normalized result

This order allows `../` traversal while ensuring the final path is safe.

---

### Task 1: Extend `project.normalize_path()`

**Files:**
- Modify: `src/pivot/project.py`
- Test: `tests/test_project.py`

Add optional `base` parameter and Windows path normalization:

```python
def normalize_path(path: str | pathlib.Path, base: pathlib.Path | None = None) -> pathlib.Path:
    """Make path absolute from base (default: project root), preserving symlinks.

    Accepts both Unix (/) and Windows (\\) path separators.
    """
    from pathlib import PureWindowsPath

    if base is None:
        base = get_project_root()

    # Normalize Windows paths to POSIX (handles both \\ and /)
    p = pathlib.Path(PureWindowsPath(os.fspath(path)).as_posix())

    abs_path = p.absolute() if p.is_absolute() else (base / p).absolute()
    return pathlib.Path(os.path.normpath(abs_path))
```

**Tests:**
- `test_normalize_path_with_custom_base` - relative path resolved from custom base
- `test_normalize_path_windows_backslash` - `foo\\bar` → `foo/bar`
- `test_normalize_path_mixed_separators` - `foo\\bar/baz` → `foo/bar/baz`
- `test_normalize_path_preserves_symlinks` - symlink path kept, not target
- `test_normalize_path_collapses_dotdot` - `foo/../bar` → `bar`

---

### Task 2: Add path resolution to `Pipeline.register()`

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py`
- Test: `tests/pipeline/test_pipeline.py`

**Step 1:** Add `_resolve_path()` method:

```python
def _resolve_path(self, annotation_path: str) -> str:
    """Convert pipeline-relative path to project-relative.

    Validation happens AFTER normalization to allow ../ traversal.
    """
    from pivot import project
    from pivot.config import path_policy

    # Absolute paths: normalize but keep absolute
    if annotation_path.startswith("/") or annotation_path.startswith("\\"):
        resolved = project.normalize_path(annotation_path).as_posix()
    else:
        # Relative paths: resolve from pipeline root → project-relative
        abs_path = project.normalize_path(annotation_path, base=self.root)
        resolved = project.to_relative_path(abs_path)

    # Validate the RESOLVED path (after ../ is collapsed)
    if error := path_policy.validate_path_syntax(resolved):
        raise ValueError(f"Invalid path '{annotation_path}': {error}")

    return resolved


def _resolve_out_override(self, override: registry.OutOverrideInput) -> registry.OutOverride:
    """Resolve path in an output override, preserving other options.

    Handles both simple path strings and OutOverride dicts.
    """
    if isinstance(override, str | pathlib.Path):
        return registry.OutOverride(path=self._resolve_path(str(override)))

    # OutOverride dict: resolve path, preserve cache option
    result = registry.OutOverride(path=self._resolve_path(str(override["path"])))
    if "cache" in override:
        result["cache"] = override["cache"]
    return result
```

**Step 2:** Update `register()` to resolve paths:

```python
def register(
    self,
    func: StageFunc,
    *,
    name: str | None = None,
    params: registry.ParamsArg = None,
    mutex: list[str] | None = None,
    variant: str | None = None,
    dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
    out_path_overrides: Mapping[str, registry.OutOverrideInput] | None = None,
) -> None:
    """Register a stage with this pipeline.

    Paths in annotations and overrides are resolved relative to pipeline root.
    """
    from pivot import stage_def

    stage_name = name or func.__name__

    # 1. Extract annotation paths using existing functions
    dep_specs = stage_def.get_dep_specs_from_signature(func, None)

    # Handle both TypedDict returns and single-output returns
    out_specs = stage_def.get_output_specs_from_return(func, stage_name)
    if not out_specs:
        single_out = stage_def.get_single_output_spec_from_return(func)
        if single_out is not None:
            out_specs = {"return": single_out}

    # 2. Resolve ALL annotation paths relative to pipeline root
    resolved_deps = {name: self._resolve_path(spec.path) for name, spec in dep_specs.items()}
    resolved_outs = {name: self._resolve_out_override(spec.path) for name, spec in out_specs.items()}

    # 3. Apply explicit overrides (also pipeline-relative)
    if dep_path_overrides:
        for dep_name, path in dep_path_overrides.items():
            resolved_deps[dep_name] = self._resolve_path(str(path))
    if out_path_overrides:
        for out_name, override in out_path_overrides.items():
            resolved_outs[out_name] = self._resolve_out_override(override)

    # 4. Pass all as overrides to registry
    self._registry.register(
        func=func,
        name=name,
        params=params,
        mutex=mutex,
        variant=variant,
        dep_path_overrides=resolved_deps,
        out_path_overrides=resolved_outs,
        state_dir=self.state_dir,
    )
```

**Tests:**
- `test_register_resolves_relative_to_pipeline_root`
- `test_register_cross_pipeline_dep_with_dotdot`
- `test_register_absolute_path_unchanged`
- `test_register_windows_path_normalized`
- `test_register_overrides_are_pipeline_relative`
- `test_register_validates_after_normalization` - `../foo` allowed, but escaping project root rejected
- `test_register_single_output_stage` - non-TypedDict return type resolved correctly
- `test_register_out_override_with_cache_option` - `OutOverride` dict preserves `cache` flag

---

### Task 3: Integration tests

**Files:**
- Test: `tests/pipeline/test_pipeline.py`

**Tests:**
- `test_dag_connects_cross_pipeline_deps` - DAG edges correct when using `../`
- `test_dag_missing_dep_raises` - validation catches missing deps
- `test_included_pipeline_paths_resolve_correctly` - `include()` works with scoped paths
- `test_lockfile_contains_project_relative_paths` - lock files use project-relative

---

### Task 4: Update documentation

**Files:**
- Modify: `docs/reference/pipelines.md`

Add section on path resolution:
- Paths relative to pipeline root (inferred from file location, like DVC)
- Cross-pipeline references with `../`
- Absolute paths
- Windows path support
- Path overrides are also pipeline-relative

---

## Path Storage Summary

| Location | Format |
|----------|--------|
| Annotations (source) | Pipeline-relative (`data/raw.csv`) |
| RegistryStageInfo | Project-relative (`horizon/data/raw.csv`) |
| Lock files | Project-relative (`horizon/data/raw.csv`) |
| WorkerStageInfo | Project-relative (derived from `project_root`) |
| Execution | Absolute (resolved at runtime) |

## Symlink Handling

Symlinks are preserved, not followed (using `os.path.normpath`, not `Path.resolve()`):
- `horizon/data` → `/mnt/shared/data` (symlink)
- `Dep("data/file.csv")` stores `horizon/data/file.csv`
- NOT `/mnt/shared/data/file.csv`

This allows symlinks pointing outside the project root.

**Note:** Existing `path_policy.py` does resolve symlinks for escape validation. The plan's approach (preserve in storage) and existing validation (resolve for security) serve different purposes and are compatible.

## Interaction with `include()`

`Pipeline.include()` deep-copies `RegistryStageInfo` objects from the source pipeline. Since path resolution happens at registration time (before `include()` is called), included stages retain their already-resolved paths.

```python
# horizon/pipeline.py
horizon.register(make_weights)  # Dep("data/raw.csv") → horizon/data/raw.csv

# main/pipeline.py
main.include(horizon)  # Copies stage with paths already resolved
# make_weights still reads from horizon/data/raw.csv
```

Cross-pipeline references from stages in the including pipeline use `../` as normal:
```python
# main/pipeline.py
main.register(
    analyze,  # Dep("../horizon/weights.csv") → horizon/weights.csv
)
```

**Future consideration:** Path overrides at include time (see #322).

## Not In Scope

- Cross-pipeline DAG auto-discovery (see #323)
- `DirectoryOut` cross-pipeline semantics
- Caching implications (unchanged - paths are project-relative)
- Path overrides at `include()` time (tracked separately)
