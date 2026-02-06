---
module: Pipeline
date: 2026-02-06
problem_type: logic_error
component: pipeline_discovery
symptoms:
  - "Output index cache always writes '.' as pipeline directory"
  - "Tier 2 cache lookups always fail, falling through to expensive tier 3 full scan"
  - "All sub-pipelines sharing project root produce identical index entries"
root_cause: logic_error
resolution_type: code_fix
severity: high
tags: [pipeline, discovery, output-index, state-dir, shared-root, three-tier, pivot]
---

# Troubleshooting: Output Index Writes Wrong Pipeline Directory for Shared-Root Pipelines

## Problem

The `_write_output_index()` method in `Pipeline` derived the producing pipeline's directory from `state_dir.parent`. When multiple sub-pipelines share the same project root (via `root=project.get_project_root()`), they all share `state_dir = <project_root>/.pivot`, so `state_dir.parent` always resolves to the project root, and the index entry is always `.`.

## Environment

- Module: Pipeline (`src/pivot/pipeline/pipeline.py`)
- Method: `_write_output_index()`
- Python Version: 3.13+
- Date: 2026-02-06

## Symptoms

- Every output index file at `.pivot/cache/outputs/{dep_path}` contains `.` instead of the actual sub-pipeline directory (e.g., `eval_pipeline/difficulty`)
- Tier 2 (output index cache) lookups fail 100% of the time: index says `.` -> looks for `pipeline.py` in project root -> no root `pipeline.py` in split layout -> falls through to tier 3
- Tier 3 (full scan) fires on every dependency resolution, adding ~10-400ms per `build_dag()` call

## Root Cause: `state_dir` Is Shared

The `state_dir` property is derived from the pipeline's `root`:

```python
@property
def state_dir(self) -> pathlib.Path:
    return self._root / ".pivot"
```

When sub-pipelines use `root=project.get_project_root()`, all share the same `state_dir`. The original index writer assumed each pipeline had a unique `state_dir`:

```python
# Before (broken):
pipeline_dir = str(info["state_dir"].parent.relative_to(project_root))
# Always produces "." when state_dir is <project_root>/.pivot
```

## Solution

Derive the pipeline directory from the stage function's source file using `inspect.getfile()`, then walk up to find the nearest directory containing a pipeline config:

```python
# After (fixed):
def _find_pipeline_dir_for_stage(
    info: registry.RegistryStageInfo,
    project_root: pathlib.Path,
) -> str | None:
    try:
        source_file = pathlib.Path(inspect.getfile(info["func"])).resolve()
        current = source_file.parent
        project_root_resolved = project_root.resolve()
        while current.is_relative_to(project_root_resolved):
            if discovery.find_config_in_dir(current) is not None:
                return str(current.relative_to(project_root_resolved))
            if current == project_root_resolved:
                break
            current = current.parent
    except (TypeError, OSError, ValueError):
        pass
    # Fall back to state_dir derivation
    ...
```

This correctly identifies `eval_pipeline/difficulty` for a stage function defined in `eval_pipeline/difficulty/compute.py`, regardless of what `state_dir` is.

## Why This Works

1. Stage functions are always defined in source files within their pipeline's directory tree
2. `inspect.getfile()` gives the actual source file, not the shared state directory
3. Walking up finds the nearest `pipeline.py` or `pivot.yaml`, which is the pipeline's config
4. Falls back to `state_dir` derivation for functions that can't be inspected (builtins, exec'd code)

## Prevention

- **Don't use `state_dir` as a proxy for "pipeline directory"** — `state_dir` is about state storage, not source code location. These are different concepts that happen to coincide for single-pipeline projects but diverge for multi-pipeline setups.
- **Test with shared-root sub-pipelines** — the split pipeline architecture (code in `eval_pipeline/X/`, data in `data/X/`, shared project root) is the primary use case. Always include a test where `root=project.get_project_root()` is used by multiple pipelines.

## Related Issues

- See also: [exists-check-prevents-reresolution-20260206.md](./exists-check-prevents-reresolution-20260206.md) (same feature area, discovered in same session)
- **Critical Pattern #3** in [critical-patterns.md](../patterns/critical-patterns.md)
