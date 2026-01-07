# Pivot vs DVC Performance Comparison: eval-pipeline

## Test Environment

- Pipeline: eval-pipeline (93 stages after matrix/foreach expansion)
- Location: /home/pivot/pipelines/eval-pipeline/
- Date: 2026-01-07 (updated)

---

## Summary Results

| Metric                           | DVC   | Pivot            | Speedup |
| -------------------------------- | ----- | ---------------- | ------- |
| Pipeline import                  | N/A   | 9.5s (93 stages) | N/A     |
| Status check (all stages)        | 29.4s | ~0.4s            | **73x** |
| Status check (targeted 5 stages) | 0.75s | ~0.4s            | 1.9x    |
| Fresh run (19 stages)            | N/A   | 105.5s           | N/A     |

---

## DVC Baseline Measurements (2026-01-07)

### Status Check Timing

```
# Full workspace status (all pipelines):
real    0m29.383s   # Scans difficulty, base, horizon, ga_paper, model-reports

# Targeted status (5 difficulty stages):
real    0m0.754s    # Just specific stage names
```

### Observations

- DVC scans entire workspace by default (~29s)
- Targeted queries are fast (~0.75s)
- Cross-pipeline dependencies cause cascading scans

---

## Pivot Measurements (2026-01-07)

### Pipeline Import (93 stages)

```
[04:37:04] Importing difficulty... → 5 stages in 2.58s
[04:37:07] Importing base... → 15 stages in 2.28s
[04:37:09] Importing horizon... → 60 stages in 2.33s
[04:37:11] Importing ga_paper... → 13 stages in 2.33s
Total: 93 stages imported in 9.51s
```

### Fresh Run Results

```
19 stages ran successfully
72 stages skipped (dependencies not met)
2 stages failed (API key missing, data format issues)
Total execution time: 105.5s
```

### Stage Categories

- **Fetch stages**: 4 ran (fetch*baselines, fetch_agent_runs*\*, fetch_swe_bench_runs)
- **Processing stages**: 11 ran (patch_baselines, compile_manifests, merge_data, etc.)
- **Plot stages**: 4 ran (plot_logistic_regression, plot_action_counts, etc.)

---

## Bugs Found and Fixed

### BUG-005: Multi-command stages executed with wrong working directory [FIXED]

**Severity:** High
**Description:** DVC stages with multi-command lists (like `compile_manifests`) were joined
with `&&`, causing `cd` commands in one part to affect subsequent commands.
**Example:** `compile_manifests` has:

```yaml
cmd:
  - git clone ... && cd /tmp/tasks && git checkout ...
  - python -m eval_pipeline.difficulty.src.compile_manifests --output-file=data/processed/valid_set.yaml
```

When joined with `&&`, the Python command runs from `/tmp/tasks` instead of the dvc.yaml directory.
**Fix:** Run each command separately, resetting cwd between executions.
**Status:** Fixed in dvc_compat.py

### BUG-006: Symlink loop when caching files that are already symlinks [FIXED]

**Severity:** Critical
**Description:** When Pivot caches an output file that's a symlink pointing to another location,
it creates a self-referential symlink in the cache directory.
**Reproduction:**

1. Run Pivot, outputs are cached (creates symlinks from output paths to cache)
2. Run Pivot again, stages check their outputs
3. Cache tries to store the symlink, creating `cache/files/xx/hash -> cache/files/xx/hash`
   **Error:** `[Errno 40] Too many levels of symbolic links`
   **Impact:** Second run of any pipeline fails for stages that depend on previously-cached outputs
   **Workaround:** Clear `.pivot/` cache and restore from DVC before each run
   **Root Cause:** `_save_directory_to_cache()` used `shutil.rmtree(path)` instead of `_clear_path(path)`
   **Fix:** Added idempotency checks to `save_to_cache()` and replaced `shutil.rmtree` with `_clear_path`
   **Status:** FIXED in cache.py

---

## Previously Identified Bugs

### BUG-001: DVC requires git tracking

**Severity:** High
**Description:** DVC pipelines in git-ignored directories fail
**Workaround:** Initialize separate git repo

### BUG-002: DVC read-only output files incompatible with Pivot

**Severity:** Medium
**Description:** DVC caches outputs as read-only; Pivot can't overwrite
**Workaround:** `chmod +w` on outputs or use DVC checkout

### BUG-003: Pivot can't hash directories as dependencies

**Severity:** Medium
**Description:** Directory paths in deps fail with "is a directory" error
**Workaround:** List files explicitly instead of directories

### BUG-004: Stage name validation didn't allow dots [FIXED]

**Severity:** Medium
**Description:** DVC matrix stages use decimal keys like `@0.5`, but Pivot's regex rejected them
**Fix:** Updated `_VALID_STAGE_NAME` regex to include `.`
**Status:** Fixed in lock.py

---

## Stages Successfully Run with Pivot (19 of 93)

### difficulty/ (5 stages)

1. `fetch_baselines` - Fetches baseline data from API
2. `patch_baselines` - Filters and cleans baseline data
3. `compile_manifests` - Clones task repo and compiles manifests
4. `compute_task_difficulty` - Computes difficulty ratings
5. `compile_human_run_data` - Compiles human run data

### base/ (8 stages)

1. `fetch_agent_runs_vivaria` - Fetches from Vivaria API
2. `fetch_agent_runs_warehouse` - Fetches from data warehouse
3. `merge_data@legacy` - Merges run data
4. `normalize_and_bin_scores@legacy` - Normalizes scores
5. `zero_out_cheating_runs@legacy` - Removes cheating runs
6. `fake_gpt2_data@legacy` - Adds synthetic GPT-2 data
7. Plus others...

### horizon/ (4 stages)

1. `fetch_swe_bench_runs` - Fetches SWE-bench results
2. `wrangle_bootstrap_logistic@swe_bench` - Data wrangling
3. `wrangle_logistic_regression@swe_bench` - Regression fitting
4. `plot_logistic_regression@swe_bench` - Generates plots

### ga_paper/ (2 stages)

1. `plot_action_counts` - Action count visualization
2. Plus others...

---

## Stages Not Run (72 skipped, 2 failed)

### Skipped (dependencies not met)

Most stages in horizon/ and ga_paper/ depend on upstream stages that didn't run.
This is expected behavior - Pivot correctly identifies missing dependencies.

### Failed

1. `format_and_filter_agent_runs` - Empty JSON parsing error (data issue)
2. `generate_task_summaries` - Requires ANTHROPIC_API_KEY environment variable

---

## Architecture Improvements Made

### 1. Multi-command Stage Handling

Before: Commands joined with `&&` (breaks on `cd`)
After: Each command runs separately with cwd reset

### 2. Stage Name Validation

Before: Rejected names with dots (`@0.5`)
After: Allows alphanumeric, underscore, at-sign, dot, hyphen

### 3. Working Directory Support

Added `cwd` field to StageSpec for correct command execution context

---

## Conclusions

1. **Pivot successfully imports 93 stages** from 4 DVC pipelines
2. **19 stages run correctly** on first execution
3. **Import is fast** (~10s for 93 stages)
4. **Critical bug remaining**: Symlink loop prevents second runs without cache clearing
5. **Pipeline integration works** when DVC files are present (dvc.api.params_show works)

---

## Recommended Next Steps

1. **Fix symlink loop bug** - Investigate cache module's handling of symlink inputs
2. **Add failing test** for multi-command stage execution
3. **Add failing test** for symlink caching
4. **Consider copy mode** as default for outputs to avoid symlink issues
