---
name: adversarial-fingerprint-testing
description: Use when testing Pivot fingerprint correctness against a real pipeline, hunting for undetected code changes that cause stale cached outputs, or validating Change Detection Matrix claims against production code
---

# Adversarial Fingerprint Testing

## Overview

Find cases where Pivot's fingerprinting fails to detect a code change that would change a stage's runtime behavior, causing stale cached outputs to be incorrectly reused.

**Core principle:** A real bug requires the stage output to actually differ. If the change doesn't affect runtime behavior, it's out of scope regardless of whether the fingerprint catches it.

## When to Use

- Testing fingerprint correctness against a real production pipeline
- Regression testing after changes to `fingerprint.py`
- Validating Change Detection Matrix claims against production-scale import graphs

## What Counts as a Real Bug

The change MUST satisfy ALL three:

1. **Executed at runtime** — the changed code actually runs when the stage executes
2. **Changes output** — same inputs produce different stage outputs after the change
3. **Claimed detected** — the Change Detection Matrix says this category should be detected

### Out of Scope

- **Type annotation-only changes** — Python doesn't enforce annotations at runtime. Exception: Pydantic models where annotations drive validation (tracked via `schema:` entries).
- **Adding methods never called** — `def display_label(self)` on a model no stage calls
- **Module-level type annotations as documentation** — `_CONFIG: MyTypedDict = {...}` where only dict values matter (tracked via params hash)
- **Docstring/comment/whitespace changes** — intentionally ignored
- **Decorator changes on stage functions** — known limitation, documented
- **Lazy imports inside function bodies** — known limitation, documented

## Pre-Attempt Checklist

### 1. Check known root causes

Read `known-root-causes.md` in this skill directory. If your planned test exercises a cataloged root cause through a similar code path, skip it unless testing regression of a fix.

### 2. Trace the execution path

Ask: "If I run this stage before and after my change with the same inputs, would the outputs actually differ?" If no, skip. This eliminates annotation-only and dead-code changes.

### 3. Check both defenses

The highest-value bugs defeat BOTH:

- **Code fingerprint** — change not detected via closure vars, type hints, module attrs, class source hashing, or class body dependency walking
- **Params hash** — change doesn't alter serialized params values (e.g., output-preserving refactor, method additions that don't affect `model_dump()`)

A change caught by either defense is a PASS, not a bug.

## Prioritization

### Tier 1 — High impact

| Pattern | Why |
|---------|-----|
| Primitive collections in closures | Tuples/lists/dicts of numbers/strings used in function bodies via direct reference. The closure path routes collections to `_process_collection_dependency` which only tracks callables, ignoring data. Asymmetric with the module attribute path. |
| `default_factory` transitive deps | Functions called inside `default_factory=lambda: helper(...)` in StageParams class bodies. Lambda AST is hashed but transitive deps not followed via `_add_callable_to_manifest`. |
| Module-load-time computed values | `CONSTANT = expensive_func()` where the function is only called at module level, never in a tracked function body. The derived value may not be fingerprinted. |
| Deep transitive deps via unusual patterns | 4+ levels deep, mixed import styles, indirect discovery paths through patterns not covered by unit tests. |

### Tier 2 — Moderate impact

| Pattern | Why |
|---------|-----|
| Functions accessed through class methods/`__init__` | Instance method calls on user objects are a documented gap (xfail test exists). |
| Enum value changes | Enum members have deterministic `repr()` — are they caught by the catch-all? Test it. |
| `functools.lru_cache` decorated helpers | Does the decorator wrapper interfere with source extraction or closure analysis? |
| Nested Pydantic structural changes 2+ levels deep | `model_fields` walking handles one level — does it recurse through nested model references? |

### Tier 3 — Low impact (skip unless Tier 1-2 exhausted)

- More variations of already-confirmed root causes through similar paths
- Annotation-only or dead-code changes
- Simple function body changes through well-tested paths (direct closure, module attr)

## Reference Materials

Study before planning tests:

- `packages/pivot/tests/fingerprint/README.md` — Change Detection Matrix (what's claimed)
- `packages/pivot/tests/fingerprint/test_change_detection.py` — Detection tests (what's verified)
- `packages/pivot/src/pivot/fingerprint.py` — Implementation (how it works)

Key functions to understand: `_process_closure_values`, `_process_collection_dependency`, `_process_module_dependency`, `_process_type_hint_dependencies`, `_hash_pydantic_schema`, `_add_callable_to_manifest`, `_collect_nested_code_globals`, `_process_class_body_dependencies`.

## Workflow

### 1. Read prior attempts

Read the pipeline's fingerprint test log (typically `fingerprint_test_log.yaml` in the pipeline repo). Don't repeat prior tests. If no log exists, create one using the format below.

### 2. Plan and verify

Before changing code, confirm:

- The code is **executed** when the stage runs (trace the call chain from the stage function)
- The change **produces different output** for the same inputs
- The category is marked **DETECTED** in the Change Detection Matrix

### 3. Make exactly ONE change

Small, targeted, semantically meaningful. Must change what the function computes, not just naming or annotations.

### 4. Test

```bash
pivot repro --all --dry-run 2>&1
```

Check affected stages: "would skip (unchanged)" for a stage that SHOULD be invalidated = potential bug. "would run (changed)" = detected.

If dry-run suggests a bug, confirm with `pivot repro --all` (without `--dry-run`).

### 5. Log the result

Append to the pipeline's test log following the format below.

### 6. REVERT the change

```bash
# Restore only changed pipeline files (NOT the log)
jj restore --from @- <path-to-changed-file>
# Or with git: git checkout -- <path-to-changed-file>

# Verify all stages cached again
pivot repro --all --dry-run 2>&1 | grep -c "would skip"
```

## Log Format

```yaml
- attempt: <N>
  category: "<from Change Detection Matrix>"
  description: >
    What was changed, how it's reachable from a stage function,
    why the fingerprint system should detect it.
  file_changed: <path>
  expected: "<detected | not_detected>"
  actual: "<detected | not_detected>"
  pivot_output: >
    <Relevant excerpt — which stages changed status and why>
  verdict: "<PASS | BUG>"
  root_cause: "<name from known-root-causes.md, or NEW: brief description>"
  notes: >
    <See guidelines below>
```

### Notes guidelines

- **PASS:** 5-10 lines. State the detection path (which mechanism caught it) and one key insight. Don't re-explain how `getclosurevars` or `_add_callable_to_manifest` work.
- **BUG (known root cause):** Reference the root cause by name. Note only what's different about this instance. 10-15 lines max.
- **BUG (new root cause):** Full description of the novel mechanism, how it differs from known causes, and proposed fix. Add to `known-root-causes.md`.

### Root cause deduplication

After finding a bug, name the root cause explicitly. Don't log the same root cause more than twice (original discovery + one confirmation) unless the new instance tests a fundamentally different code path — not just a different class in a different module exercising the same gap.

## Rules

1. ONE change per session
2. Don't repeat tests already in the log
3. ALWAYS revert pipeline code before finishing
4. ALWAYS update the log
5. Don't modify Pivot tool code — only pipeline code
6. Focus on changes that **actually change runtime behavior**
7. Deduplicate root causes — check known causes before each attempt

## Common Mistakes

| Mistake | Why it wastes time |
|---------|--------------------|
| Testing annotation-only changes | Python doesn't enforce annotations at runtime. Zero output impact. |
| Adding methods no stage calls | Method additions to models don't change `model_dump()` or any call site. |
| Re-finding the same root cause 8 times | First confirmation is valuable. Further variations through similar paths aren't. |
| Testing well-trodden paths "to confirm they work" | Direct function body changes, direct import closure tracking — these are extensively unit-tested. Target seams between mechanisms instead. |
| Burying high-value findings late | Prioritize Tier 1 patterns that defeat both defenses. Don't spend early attempts on safe confirmations. |
| Verbose notes re-explaining the fingerprint system | The README documents the system. Notes should describe what's novel about THIS attempt. |
