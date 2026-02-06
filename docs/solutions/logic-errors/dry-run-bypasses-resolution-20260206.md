---
module: CLI
date: 2026-02-06
problem_type: logic_error
component: tooling
symptoms:
  - "pivot repro --dry-run shows only local stages, not cross-pipeline dependencies"
  - "Dry-run output differs from actual execution in multi-pipeline setups"
  - "Cross-pipeline stages missing from dry-run DAG visualization"
root_cause: logic_error
resolution_type: code_fix
severity: medium
tags: [cli, dry-run, resolution, discovery, dag, build-dag, pivot]
---

# Troubleshooting: --dry-run Bypasses External Dependency Resolution

## Problem

The `--dry-run` code path in the CLI called `get_all_stages()` and `build_graph()` directly, bypassing `build_dag()` which is the method that triggers `resolve_external_dependencies()`. This meant dry-run output only showed locally-registered stages, not cross-pipeline dependencies that would be resolved during actual execution.

## Environment

- Module: CLI (`src/pivot/cli/repro.py`)
- Functions: `_dry_run()`, `_output_explain()`
- Python Version: 3.13+
- Date: 2026-02-06

## Symptoms

- `pivot repro --dry-run` shows only stages from the current `pipeline.py`
- Stages from parent/sibling pipelines that would be included during execution are missing
- Dry-run reports fewer stages than actual `pivot repro`
- Misleading "all stages would skip" output when cross-pipeline deps haven't been resolved
- Problem is invisible in single-pipeline projects (no external deps to resolve)

## Root Cause: Parallel Code Paths

Two separate code paths existed for building the stage graph:

```python
# Normal execution path (correct):
pipeline.build_dag()  # calls resolve_external_dependencies() internally
# ... then executes stages

# Dry-run path (broken):
stages = cli_helpers.get_all_stages()  # reads registry directly
graph = engine_graph.build_graph(stages)  # builds graph from local-only stages
# ... displays what would run
```

The dry-run path was written before external dependency resolution existed and was never updated.

## Solution

Call `resolve_external_dependencies()` before `get_all_stages()` in both dry-run code paths:

```python
# After (fixed):
cli_helpers.resolve_external_dependencies()  # new: ensure cross-pipeline deps resolved
stages = cli_helpers.get_all_stages()
graph = engine_graph.build_graph(stages)
```

A `cli_helpers.resolve_external_dependencies()` helper was added to encapsulate the call.

## Why This Works

- `resolve_external_dependencies()` populates the registry with external stages before the graph is built
- The dry-run path now sees the same stages as the execution path
- Idempotent: if resolution already ran (e.g., during a previous `build_dag()` call), the `_external_deps_resolved` flag skips redundant work

## Prevention

- **All code paths that read the stage registry should go through `build_dag()`** (or at minimum call `resolve_external_dependencies()` first). Direct access to `get_all_stages()` skips resolution.
- **When adding new CLI commands or output modes**, verify they use the same graph construction path as the execution engine. A common pattern is to copy-paste the graph building code without the resolution step.
- **Test CLI output modes with multi-pipeline setups** — single-pipeline tests won't catch code paths that skip resolution since there are no external deps to resolve.

## Related Issues

- See also: [exists-check-prevents-reresolution-20260206.md](./exists-check-prevents-reresolution-20260206.md) (same feature area)
- See also: [output-index-state-dir-shared-root-20260206.md](./output-index-state-dir-shared-root-20260206.md) (same feature area)
