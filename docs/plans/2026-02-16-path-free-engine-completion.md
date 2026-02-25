# Path-Free Engine Completion Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the external-input resolution bug that breaks `--all` mode, fix the console explain crash, add the presentation symlink tree, and verify end-to-end.

**Architecture:** The compose API's `_InputNode.path` must flow through `Pipeline.input_bindings` → `StoreSpec.input_bindings` → `WorkspaceStore._resolve_input_path`. The presentation layer materializes CAS refs into conventional workspace symlinks after a successful run.

**Tech Stack:** Python 3.13, Pydantic, TypedDicts, lmdb (StateDB), Textual/Rich, pytest, ruff, basedpyright.

---

### Task 1: Fix external input resolution (compose → store path propagation)

**Root cause:** `p.input("small_tasks_runs.jsonl", external=True)` creates `_InputNode(path="data/external/small_tasks_runs.jsonl")`, but `compose.build()` discards the path when creating `ArtifactRef`. The `WorkspaceStore` always resolves inputs to `data/raw/{name}` because `input_bindings` is always empty.

**Critical for `--all` mode:** When `_discover_all_pipelines` creates `Pipeline("all")` and calls `combined.include(child)`, the combined pipeline must inherit input bindings from each child. Without this, the fix only works for single-pipeline mode.

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py` (expose input bindings)
- Modify: `packages/pivot/src/pivot/pipeline/pipeline.py` (store, propagate, and merge input bindings via `include()`)
- Modify: `packages/pivot/src/pivot/engine/engine.py` (pass bindings to StoreSpec — 2 sites: lines 1367, 1497)
- Modify: `packages/pivot/src/pivot/cli/helpers.py` (pass bindings to workspace store)
- Modify: `packages/pivot/src/pivot/cli/repro.py` (pass bindings to workspace store)
- Modify: `packages/pivot/src/pivot/executor/commit.py` (pass bindings to StoreSpec — line 84)
- Test: `packages/pivot/tests/test_compose.py`
- Test: `packages/pivot/tests/pipeline/test_pipeline.py` (include merges bindings)

**Step 1: Write the failing tests**

Test 1 — compose external inputs produce bindings:

```python
def test_external_input_produces_bindings(tmp_path: Path) -> None:
    """compose.Pipeline.build() should propagate external input paths as input_bindings."""
    project_root = tmp_path
    (project_root / "data" / "external").mkdir(parents=True)
    (project_root / "data" / "external" / "ext_data.jsonl").write_text("[]")

    with compose.Pipeline("test", root=project_root) as p:
        ext = p.input("ext_data.jsonl", t=list, external=True)

        @pivot.stage
        def consume(data: list = ext) -> Annotated[list, Out("out.json", JSON())]:
            return data

    pipeline = p.build()
    bindings = pipeline.input_bindings

    assert "ext_data.jsonl" in bindings
    assert bindings["ext_data.jsonl"] == "data/external/ext_data.jsonl"
```

Test 2 — include merges bindings:

```python
def test_include_merges_input_bindings(tmp_path: Path) -> None:
    """Pipeline.include() should merge input_bindings from the included pipeline."""
    child = Pipeline("child", root=tmp_path)
    child.set_input_bindings({"ext.jsonl": "data/external/ext.jsonl"})

    parent = Pipeline("parent", root=tmp_path)
    parent.include(child)

    assert parent.input_bindings["ext.jsonl"] == "data/external/ext.jsonl"
```

Test 3 — WorkspaceStore resolves external input correctly:

```python
def test_workspace_store_resolves_external_input(tmp_path: Path) -> None:
    """WorkspaceStore with input_bindings should resolve to the bound path."""
    (tmp_path / "data" / "external" / "ext.jsonl").parent.mkdir(parents=True)
    (tmp_path / "data" / "external" / "ext.jsonl").write_text("[]")

    ws = WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="test",
        input_bindings={"ext.jsonl": "data/external/ext.jsonl"},
    )
    ref = ArtifactRef(
        identity=ArtifactIdentity("ext.jsonl", None),
        format=loaders.DataFrameJSONL(),
        python_type=list,
        tag=ArtifactTag.DATA,
    )
    assert ws.exists(ref)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/pivot/tests/test_compose.py::test_external_input_produces_bindings -v
uv run pytest packages/pivot/tests/pipeline/test_pipeline.py::test_include_merges_input_bindings -v
```

Expected: FAIL — `pipeline.input_bindings` doesn't exist.

**Step 3: Write minimal implementation**

1. In `compose.py`, expose input bindings from `build()`:

   ```python
   def build(self) -> pipeline_mod.Pipeline:
       legacy = pipeline_mod.Pipeline(self._name, root=self._root)
       legacy.set_input_bindings({name: node.path for name, node in self._inputs.items()})
       # ... rest of build() unchanged
   ```

2. In `pipeline/pipeline.py`, add storage, accessor, and merge in `include()`:

   ```python
   _input_bindings: dict[str, str]

   def __init__(self, name: str, ...):
       ...
       self._input_bindings = {}

   def set_input_bindings(self, bindings: dict[str, str]) -> None:
       self._input_bindings = bindings.copy()

   @property
   def input_bindings(self) -> dict[str, str]:
       return self._input_bindings.copy()

   def include(self, other: Pipeline) -> None:
       # ... existing stage copy logic ...
       # Merge input bindings (child bindings don't overwrite existing)
       for key, path in other._input_bindings.items():
           if key not in self._input_bindings:
               self._input_bindings[key] = path
   ```

3. In `engine/engine.py`, replace `input_bindings={}` with `input_bindings=pipeline.input_bindings` at both `StoreSpec` sites (lines 1367, 1497). Also update the `WorkspaceStore` at line 853.

4. In `cli/helpers.py:get_workspace_store()`, replace `input_bindings={}` with `input_bindings=pipeline.input_bindings`.

5. In `cli/repro.py:_get_explanations()`, pass bindings to the workspace store.

6. In `executor/commit.py`, replace `input_bindings={}` with pipeline bindings (requires passing pipeline or bindings to `commit_stages`).

**Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/pivot/tests/test_compose.py::test_external_input_produces_bindings -v
uv run pytest packages/pivot/tests/pipeline/test_pipeline.py::test_include_merges_input_bindings -v
uv run pytest packages/pivot/tests -n auto --tb=short -q
```

Expected: PASS — all tests green.

**Step 5: Commit**

```bash
git add packages/pivot/src/pivot/compose.py packages/pivot/src/pivot/pipeline/pipeline.py packages/pivot/src/pivot/engine/engine.py packages/pivot/src/pivot/cli/helpers.py packages/pivot/src/pivot/cli/repro.py packages/pivot/tests/
git commit -m "fix: propagate external input bindings from compose to store"
```

---

### Task 2: Fix console explain crash (`DepChange` key mismatch)

**Root cause:** `console.py:298` passes `"path"` as the `key_field` for dep changes, but `DepChange` TypedDict uses `"identity"` (an `ArtifactIdentity` NamedTuple). This crashes with `KeyError: 'path'` when console explain prints dependency changes.

**Files:**
- Modify: `packages/pivot/src/pivot/cli/console.py:298`
- Test: `packages/pivot/tests/cli/test_console.py` (or add if missing)

**Step 1: Write the failing test**

```python
def test_explain_stage_dep_changes(capsys: CaptureFixture[str]) -> None:
    """Console.explain_stage should render dep changes without crashing."""
    con = Console(color=False)
    explanation = StageExplanation(
        stage_name="train",
        will_run=True,
        is_forced=False,
        reason="Dependencies changed",
        code_changes=[],
        param_changes=[],
        dep_changes=[
            DepChange(
                identity=ArtifactIdentity("input_data", None),
                old_hash="aaa",
                new_hash="bbb",
                change_type=ChangeType.MODIFIED,
            )
        ],
        upstream_stale=[],
    )
    con.explain_stage(explanation)
    # Should not crash — and should display the identity
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest packages/pivot/tests/cli/test_console.py::test_explain_stage_dep_changes -v
```

Expected: FAIL with `KeyError: 'path'`.

**Step 3: Write minimal implementation**

Two changes in `console.py`:

1. Line 298: change `"path"` to `"identity"`:

   ```python
   self._print_changes(
       "Dependency Changes:", explanation["dep_changes"], "identity", "old_hash", "new_hash"
   )
   ```

2. In `_print_changes` (line 257), format `ArtifactIdentity` values for display. Add `ArtifactIdentity` and `identity_key` to the existing `from pivot.types import (...)` block at the top of the file, then insert after line 257:

   ```python
   key = change[key_field]
   if isinstance(key, ArtifactIdentity):
       key = identity_key(key)
   ```

   This keeps `_print_changes` generic (works for strings and identities) without polluting the call sites.

**Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/pivot/tests/cli/test_console.py -v
uv run pytest packages/pivot/tests -n auto --tb=short -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add packages/pivot/src/pivot/cli/console.py packages/pivot/tests/cli/test_console.py
git commit -m "fix: console explain uses identity field for dep changes"
```

---

### Task 3: Presentation layer + engine hook

**Context:** `CacheStore` writes outputs to `refs/{producer}/{key}` as symlinks to CAS blobs (using `SINGLE_OUTPUT_KEY="_single"` for `key=None`). But there's no user-facing view of these outputs. The presentation layer materializes a conventional directory tree (`data/`, `metrics/`, `plots/`) of symlinks pointing to the CAS ref paths, so users can browse outputs at familiar locations.

**Design decision:** The presentation tree reuses `WorkspaceStore._resolve_output_path` for path layout — same `data/{pipeline}/{stage}.{ext}` structure. Symlinks point to the CAS ref paths (which themselves symlink to CAS blobs), forming a two-hop chain: `data/pipeline/stage.csv → .pivot/cache/files/refs/stage/_single → .pivot/cache/files/ab/cd1234...`.

**Files:**
- Create: `packages/pivot/src/pivot/storage/presentation.py`
- Modify: `packages/pivot/src/pivot/engine/engine.py` (call presentation after run — after `_write_run_history`, before `return results`)
- Test: `packages/pivot/tests/storage/test_presentation.py`

**Step 1: Write the failing tests**

```python
def _make_stage(
    name: str, tag: ArtifactTag, fmt: object, *, key: str | None
) -> RegistryStageInfo:
    """Test helper: create a minimal RegistryStageInfo with one output."""
    return RegistryStageInfo(
        func=lambda: None,
        name=name,
        deps={},
        outs=[ArtifactRef(
            identity=ArtifactIdentity(name, key),
            format=fmt,
            python_type=object,
            tag=tag,
        )],
        params=None,
        mutex=list[str](),
        variant=None,
        signature=None,
        fingerprint=dict[str, str](),
        params_arg_name=None,
        state_dir=None,
    )


def test_presentation_creates_symlinks_for_single_output(tmp_path: Path) -> None:
    """present() should symlink workspace path → CAS ref for key=None outputs."""
    # cache_dir mirrors CacheStore's cache_dir (e.g., .pivot/cache/files)
    cache_dir = tmp_path / ".pivot" / "cache" / "files"
    refs_dir = cache_dir / "refs"

    # CAS ref for key=None uses SINGLE_OUTPUT_KEY = "_single"
    ref_path = refs_dir / "train" / "_single"
    ref_path.parent.mkdir(parents=True)
    ref_path.write_text("fake data")

    stages = {"train": _make_stage("train", ArtifactTag.DATA, CSV(), key=None)}

    present(
        project_root=tmp_path,
        pipeline_name="default",
        cache_dir=cache_dir,
        stages=stages,
    )

    display_path = tmp_path / "data" / "default" / "train.csv"
    assert display_path.is_symlink()
    assert display_path.resolve() == ref_path.resolve()


def test_presentation_creates_symlinks_for_keyed_output(tmp_path: Path) -> None:
    """present() should symlink keyed outputs (metrics, plots) correctly."""
    cache_dir = tmp_path / ".pivot" / "cache" / "files"
    refs_dir = cache_dir / "refs"

    ref_path = refs_dir / "eval" / "accuracy"
    ref_path.parent.mkdir(parents=True)
    ref_path.write_text("{}")

    stages = {"eval": _make_stage("eval", ArtifactTag.METRIC, JSON(), key="accuracy")}

    present(
        project_root=tmp_path,
        pipeline_name="default",
        cache_dir=cache_dir,
        stages=stages,
    )

    display_path = tmp_path / "metrics" / "default" / "eval" / "accuracy.json"
    assert display_path.is_symlink()


def test_presentation_skips_missing_refs(tmp_path: Path) -> None:
    """present() should silently skip outputs with no CAS ref on disk."""
    cache_dir = tmp_path / ".pivot" / "cache" / "files"
    (cache_dir / "refs").mkdir(parents=True)

    stages = {"missing": _make_stage("missing", ArtifactTag.DATA, CSV(), key=None)}

    # Should not raise
    present(
        project_root=tmp_path,
        pipeline_name="default",
        cache_dir=cache_dir,
        stages=stages,
    )

    display_path = tmp_path / "data" / "default" / "missing.csv"
    assert not display_path.exists()
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/pivot/tests/storage/test_presentation.py -v
```

Expected: FAIL — `presentation` module doesn't exist.

**Step 3: Write minimal implementation**

Create `packages/pivot/src/pivot/storage/presentation.py`:

```python
"""Presentation layer: materializes CAS refs as workspace symlinks.

After a successful pipeline run, creates a conventional directory tree
(data/, metrics/, plots/) with symlinks pointing to CAS ref paths.
This gives users browsable output at familiar locations while the
actual data lives in content-addressed storage.
"""
from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

from pivot import compose, types
from pivot.storage import store as store_mod

if TYPE_CHECKING:
    from pivot.types import RegistryStageInfo

logger = logging.getLogger(__name__)


def present(
    *,
    project_root: pathlib.Path,
    pipeline_name: str,
    cache_dir: pathlib.Path,
    stages: dict[str, RegistryStageInfo],
) -> None:
    """Materialize CAS ref symlinks into workspace display paths.

    For each output of each stage, creates a symlink at the conventional
    workspace location (e.g., data/pipeline/stage.csv) pointing to the
    CAS ref path (e.g., .pivot/cache/refs/stage/_single).

    Only creates symlinks for outputs that have CAS refs on disk.
    """
    refs_dir = cache_dir / "refs"
    if not refs_dir.exists():
        return

    ws = store_mod.WorkspaceStore(
        project_root=project_root,
        pipeline_name=pipeline_name,
        input_bindings={},
    )

    created = 0
    for _stage_name, info in stages.items():
        for out in info["outs"]:
            ref_path = _ref_path(refs_dir, out)
            if not ref_path.exists() and not ref_path.is_symlink():
                continue

            display_path = ws.resolve_display_path(out)
            _ensure_symlink(display_path, ref_path)
            created += 1

    if created:
        logger.debug("Presentation layer: created %d symlinks", created)


def _ref_path(refs_dir: pathlib.Path, ref: types.ArtifactRef) -> pathlib.Path:
    """Compute the CAS ref path for an artifact.

    Mirrors CacheStore._ref_path: uses SINGLE_OUTPUT_KEY for key=None.
    """
    key = ref.identity.key or compose.SINGLE_OUTPUT_KEY
    return refs_dir / ref.identity.producer / key


def _ensure_symlink(display_path: pathlib.Path, ref_path: pathlib.Path) -> None:
    """Create or update a symlink at display_path pointing to ref_path."""
    display_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing symlink/file if present
    if display_path.is_symlink() or display_path.exists():
        display_path.unlink()

    display_path.symlink_to(ref_path.resolve())
```

Then hook into the engine. In `engine.py`, after `_write_run_history` (line ~1197) and before `return results`:

```python
# Materialize presentation symlinks for successful stages
from pivot.storage import presentation
try:
    presentation.present(
        project_root=project_root,
        pipeline_name=pipeline.name,
        cache_dir=cache_dir / "files",
        stages=all_stages,
    )
except Exception:
    _logger.debug("Presentation layer failed", exc_info=True)
```

Note: `cache_dir` in the engine is `.pivot/cache` (from `config.get_cache_dir()`). The CAS files directory is `cache_dir / "files"`, and CAS refs live under `cache_dir / "files" / "refs"`. The `present()` function's `cache_dir` parameter should match what `CacheStore` receives — i.e., `.pivot/cache/files`.

**Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/pivot/tests/storage/test_presentation.py -v
uv run pytest packages/pivot/tests -n auto --tb=short -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add packages/pivot/src/pivot/storage/presentation.py packages/pivot/tests/storage/test_presentation.py packages/pivot/src/pivot/engine/engine.py
git commit -m "feat: add presentation symlink tree for CAS outputs"
```

---

### Task 4: Full verification + eval pipeline

**Files:**
- Modify (if needed): any files from previous tasks

**Step 1: Run quality checks**

```bash
uv run ruff format .
uv run ruff check .
uv run basedpyright
```

**Step 2: Run all tests**

```bash
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
```

(Note: run pivot and pivot-tui tests separately if conftest import conflict persists.)

**Step 3: End-to-end verification**

```bash
cd /home/sami/eval-pipeline/pivot
uv run pivot repro --all
uv run pivot repro --all
```

Expected:
- First run: succeeds — external inputs resolve correctly, all stages execute
- Second run: fully cached — all stages skip
- Presentation symlinks exist under `data/`, `metrics/`, `plots/` (if using CacheStore mode)

**Step 4: Commit**

```bash
git add .
git commit -m "test: verify path-free engine end-to-end with eval pipeline"
```

---

## Superseded Tasks (from original plan)

The following tasks from `2026-02-15-path-free-engine.md` are **already complete** — all tests pass:

- ~~Task 1: Canonical identity encoding + structured identity types~~ — DONE (types.py, agent_rpc.py, client.py)
- ~~Task 2: Lockfile + StateDB identity normalization~~ — DONE (lock.py, state.py, skip.py)
- ~~Task 3: Worker/explain hashing uses Store + identity keys~~ — DONE (worker.py, explain.py)
- ~~Task 4 (partial): TUI diff panels~~ — DONE (diff_panels.py uses `identity_key()`)
- ~~Task 4 (partial): RPC contract~~ — DONE (agent_rpc.py emits structured identities)

## Execution Mode

Proceed with **Subagent-Driven** execution in this session; dispatch one task at a time and review each diff.
