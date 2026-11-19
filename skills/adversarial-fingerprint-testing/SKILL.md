---
name: adversarial-fingerprint-testing
description: Use when testing Pivot fingerprint correctness against a real pipeline, hunting for undetected code changes that cause stale cached outputs, regression-testing a change to fingerprint.py, or validating Change Detection Matrix claims against production code
---

# Adversarial Fingerprint Testing

Find code changes that Pivot's fingerprint misses although they change a stage's output, so a stale cached output is reused. A change that does not change runtime behavior is out of scope, whether or not the fingerprint catches it.

## What counts as a bug

All three must hold:

1. **Executed**: the changed code runs when the stage executes.
2. **Changes output**: the same inputs give a different stage output after the change.
3. **Claimed detected**: the Change Detection Matrix marks the category detected (✅).

A change caught by either defense is a PASS:

- **Code fingerprint**: the stage's manifest (`code_manifest` in its lock): globals, closure cells, defaults, type hints, `module.attr` reads, class source and class bodies.
- **Params**: the stage's `StageParams` values (`params` in its lock).

The most valuable bugs defeat both: the manifest misses the change and the params values stay the same.

Out of scope:

- Annotation-only changes: Python does not enforce annotations. Pydantic models are the exception, because their annotations drive validation (`schema:` entries).
- Methods that no stage calls.
- Module-level variable annotations (`_BASE: MyTypedDict = {...}`): documentation only.
- Docstring, comment, whitespace and function-name changes: ignored by design.
- Decorators on the stage function: a stated gap.
- Helpers marked `@pivot.fingerprint_by_name`: recorded by name by design.

## Before planning

Read these first:

- `packages/pivot/tests/fingerprint/README.md`: the Change Detection Matrix, which says what is claimed and which test shows it.
- `docs/architecture/fingerprinting.md`, "Known Limitations": stated gaps, which are not bugs.
- `packages/pivot/src/pivot/fingerprint.py`: start at `get_stage_fingerprint`, `_expand_function` and `_expand_class`. Then read `_process_closure_values`, `_process_captured_value`, `_process_module_dependency`, `_process_type_hint_dependencies`, `_process_class_body_dependencies`, `_process_dataclass_defaults`, `_process_model_schema`, `_form` (values that partials, defaults and loader config hold) and `_try_manifest_cache_hit` (a warm run replays recorded reads instead of walking).

## Where to look

Tier 1, high impact:

- Seams between mechanisms: code reached through a path the unit tests do not take, four or more levels deep, or through mixed import styles.
- Code held inside values that `_form` records: partial arguments, dataclass and pydantic field defaults, loader config.
- A change that a full walk sees but a manifest cache hit does not read again.

Tier 2, moderate impact:

- Helpers reached only through methods of user classes (`__init__`, properties, classmethods).
- Enum members inside values (a partial argument, a dataclass default).
- `functools` wrappers on helpers: `lru_cache`, `wraps`, `partialmethod`.
- Pydantic models nested two or more levels deep (field classes are `fc:` edges).

Tier 3, skip until Tiers 1 and 2 are exhausted:

- More instances of a logged root cause through similar paths.
- Annotation-only or dead-code changes.
- Paths the unit tests cover well: a function body, a direct import, a module attribute.

## Procedure

1. Read the pipeline's test log (by convention `fingerprint_test_log.yaml` in the pipeline repository; create it if it does not exist). Do not repeat a logged attempt.
2. Read `references/known-root-causes.md` before you choose a change. If the change exercises a listed root cause through a similar path, skip it, unless you are checking that a fixed one stays fixed.
3. Before you edit, confirm the three bug conditions: trace the call chain from the stage function, say how the output will differ, and find the matrix row.
4. Make one small change that alters what the code computes, not only names or annotations. Make one change per session, in pipeline code only; do not edit Pivot.
5. Run `pivot repro --all --dry-run`. A stage that should rerun but prints `would skip (unchanged)` is a candidate bug; a `would run` line whose reason says `Code changed` or `Params changed` is a detection. Confirm a candidate with `pivot repro --all`.
6. Append the result to the log in the format below.
7. Revert the pipeline change and keep the log: `git checkout -- <file>` (or `jj restore --from @- <file>`). If step 5 ran `pivot repro --all`, run it again. Then `pivot repro --all --dry-run` must show `would skip` for every stage.

## Log format

```yaml
- attempt: <N>
  category: "<row from the Change Detection Matrix>"
  description: >
    What was changed, how it is reachable from a stage function,
    why the fingerprint should detect it.
  file_changed: <path>
  expected: "<detected | not_detected>"
  actual: "<detected | not_detected>"
  pivot_output: >
    <Relevant excerpt: which stages changed status and why>
  verdict: "<PASS | BUG>"
  root_cause: "<ID from references/known-root-causes.md, or NEW: brief description>"
  notes: >
    <See below>
```

Notes:

- **PASS**: 5 to 10 lines. Name the mechanism that caught the change and one insight. Do not explain the fingerprint system again; the matrix documents it.
- **BUG, known root cause**: name the root cause and note only what is different in this instance, in 15 lines or fewer.
- **BUG, new root cause**: describe the mechanism, how it differs from the listed causes, and a fix. Add it to `references/known-root-causes.md`.

Log a root cause at most twice (the discovery and one confirmation), unless the new instance takes a different code path. A different class in another module through the same gap is not a different path.
