---
module: Pipeline
date: 2026-02-06
problem_type: logic_error
component: pipeline_discovery
symptoms:
  - "Producer stage not included in DAG when output file exists from previous run"
  - "Re-running pipeline after successful first run skips dependency resolution"
  - "Changes to producer stage not detected on subsequent runs"
root_cause: logic_error
resolution_type: code_fix
severity: high
tags: [pipeline, discovery, resolution, exists-check, skip-detection, dag, pivot]
---

# Troubleshooting: exists() Check Prevents Re-Resolution of External Dependencies

## Problem

`resolve_external_dependencies()` skipped dependency resolution for any output file that already existed on disk. After a successful first run, all output files exist, so the method skipped finding their producing stages entirely. This meant the producer stages were never included in the DAG on subsequent runs.

## Environment

- Module: Pipeline (`src/pivot/pipeline/pipeline.py`)
- Method: `resolve_external_dependencies()`, line ~367
- Python Version: 3.13+
- Date: 2026-02-06

## Symptoms

- First `pivot repro` works correctly (producer stages found and included)
- Second `pivot repro` skips producer stages (they're not in the DAG)
- Changes to producer source code don't trigger re-execution of consumers
- Debug logs show no resolution activity on warm runs
- Problem only visible in multi-pipeline setups where producers are in separate `pipeline.py` files

## Root Cause: Conflation of "File Exists" with "No Producer Needed"

The original resolution loop had:

```python
# Before (broken):
if dep_path in local_outputs or pathlib.Path(dep_path).exists():
    continue  # Skip if already resolved OR exists on disk
```

The `exists()` check was intended as an optimization: if a file exists and has no local producer, it's probably raw input data. But this conflates two distinct concepts:

1. **File has a producer in another pipeline** — should be included in the DAG
2. **File is raw input data** — genuinely has no producer, safe to skip

After a successful first run, ALL output files exist on disk, so the method treats every cross-pipeline dependency as "raw input data" and skips resolution entirely.

## Solution

Remove the `exists()` check. Only skip deps that are already satisfied by a locally-registered stage:

```python
# After (fixed):
if dep_path in local_outputs:
    continue  # Skip only if a local stage produces this
```

The three-tier discovery system handles the rest: if no pipeline produces the dep, it's genuinely external input and resolution simply doesn't find a producer (no error).

## Why This Works

- **`local_outputs`** tracks what THIS pipeline (including already-resolved external stages) produces — safe to skip
- **File existence** says nothing about whether a producer exists in another pipeline
- The resolution tiers (traverse-up, index cache, full scan) handle the "does a producer exist?" question
- If no producer is found, the dep is treated as external input (same end result as before, but correct)

## Prevention

- **Don't use filesystem state to short-circuit DAG construction** — the DAG should reflect the logical dependency graph, not the current state of output files on disk. Filesystem state is for skip detection (at execution time), not for graph construction.
- **Test with warm cache** — always test pipeline resolution with output files already present from a previous run. A cold-start test (no files on disk) can pass while warm-start behavior is broken.

## Related Issues

- See also: [output-index-state-dir-shared-root-20260206.md](./output-index-state-dir-shared-root-20260206.md) (same feature area, discovered in same session)
- **Critical Pattern #2** in [critical-patterns.md](../patterns/critical-patterns.md)
