# Path-Free Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the path-free engine by making ArtifactIdentity structured everywhere, fixing the remaining integration test failures, and finishing presentation + CLI/TUI identity display.

**Architecture:** ArtifactIdentity is the canonical identity across engine/worker/lock/state/CLI/TUI. Paths are derived only via Store and presentation. Lockfiles and RPC serialize identities as `{producer, key}` objects; StateDB uses a single canonical identity key encoder for internal prefixes.

**Tech Stack:** Python 3.13, Pydantic, TypedDicts, lmdb (StateDB), Textual/Rich, pytest, ruff, basedpyright.

---

### Task 1: Canonical identity encoding + structured identity types

**Files:**
- Modify: `packages/pivot/src/pivot/types.py`
- Modify: `packages/pivot/src/pivot/engine/agent_rpc.py`
- Modify: `packages/pivot-tui/src/pivot_tui/client.py`
- Modify: `packages/pivot-tui/src/pivot_tui/rpc_client_impl.py`
- Test: `packages/pivot/tests/core/test_types.py`
- Test: `packages/pivot-tui/tests/test_rpc_client_impl.py`

**Step 1: Write the failing test**

Add a test in `packages/pivot/tests/core/test_types.py`:

```python
def test_artifact_identity_json_roundtrip() -> None:
    identity = ArtifactIdentity("train", "metrics")
    payload = identity_to_json(identity)
    assert payload == {"producer": "train", "key": "metrics"}
    assert identity_from_json(payload) == identity
```

Add a test in `packages/pivot-tui/tests/test_rpc_client_impl.py` expecting `stage_info()` to return structured identities:

```python
async def test_stage_info_structured_identity(tmp_path: Path) -> None:
    server.set_stage_info("train", deps=[{"producer": "input", "key": None}], outs=[{"producer": "train", "key": None}])
    result = await client.stage_info("train")
    assert result["deps"][0].producer == "input"
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest packages/pivot/tests/core/test_types.py::test_artifact_identity_json_roundtrip -v
```

Expected: FAIL (identity_to_json/identity_from_json missing)

Run:

```bash
uv run pytest packages/pivot-tui/tests/test_rpc_client_impl.py::test_stage_info_structured_identity -v
```

Expected: FAIL (stage_info still returns strings)

**Step 3: Write minimal implementation**

- In `pivot/types.py`, add `ArtifactIdentityJson` TypedDict and helpers:
  - `identity_to_json(identity) -> ArtifactIdentityJson`
  - `identity_from_json(payload) -> ArtifactIdentity`
  - `identity_key(identity) -> str` (canonical internal key, producer or producer:key)
- Update `DepChange.identity` and `OutputChange.path` to use `ArtifactIdentity`.
- Update RPC types to use `ArtifactIdentityJson` for wire format.
- Update `pivot_tui` client/rpc to decode identity JSON to `ArtifactIdentity` instances.

**Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/pivot/tests/core/test_types.py::test_artifact_identity_json_roundtrip -v
uv run pytest packages/pivot-tui/tests/test_rpc_client_impl.py::test_stage_info_structured_identity -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add packages/pivot/src/pivot/types.py packages/pivot/src/pivot/engine/agent_rpc.py packages/pivot-tui/src/pivot_tui/client.py packages/pivot-tui/src/pivot_tui/rpc_client_impl.py packages/pivot/tests/core/test_types.py packages/pivot-tui/tests/test_rpc_client_impl.py
git commit -m "feat: add structured artifact identity serialization"
```

---

### Task 2: Lockfile + StateDB identity normalization

**Files:**
- Modify: `packages/pivot/src/pivot/storage/lock.py`
- Modify: `packages/pivot/src/pivot/storage/state.py`
- Modify: `packages/pivot/src/pivot/skip.py`
- Modify: `packages/pivot/src/pivot/explain.py`
- Modify: `packages/pivot/src/pivot/types.py`
- Test: `packages/pivot/tests/storage/test_lock.py` (or nearest lock tests)
- Test: `packages/pivot/tests/core/test_skip.py`

**Step 1: Write the failing test**

Add a lockfile round‑trip test verifying identity objects are preserved:

```python
def test_lock_roundtrip_identity_entries(tmp_path: Path) -> None:
    data = LockData(
        code_manifest={},
        params={},
        dep_hashes={ArtifactIdentity("input", None): {"hash": "aaa"}},
        output_hashes={ArtifactIdentity("train", None): {"hash": "bbb"}},
    )
    lock = StageLock("train", tmp_path)
    lock.write(data)
    loaded = lock.read()
    assert loaded is not None
    assert ArtifactIdentity("input", None) in loaded["dep_hashes"]
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest packages/pivot/tests/storage/test_lock.py::test_lock_roundtrip_identity_entries -v
```

Expected: FAIL (lock uses string keys)

**Step 3: Write minimal implementation**

- Change `LockData.dep_hashes` and `output_hashes` to `dict[ArtifactIdentity, HashInfo]`.
- Update lock serialization to store entries as a list of `{identity, hash_info}` objects.
- Update `skip.diff_dep_hashes` to compare identity‑keyed dicts and return `DepChange(identity=ArtifactIdentity)`.
- Update StateDB key helpers to use `identity_key(identity)` for internal prefixes.
- Update explain to compute current dep hashes keyed by `ArtifactIdentity` via Store resolution, not raw path strings.

**Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/pivot/tests/storage/test_lock.py::test_lock_roundtrip_identity_entries -v
uv run pytest packages/pivot/tests/core/test_skip.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add packages/pivot/src/pivot/storage/lock.py packages/pivot/src/pivot/storage/state.py packages/pivot/src/pivot/skip.py packages/pivot/src/pivot/explain.py packages/pivot/src/pivot/types.py packages/pivot/tests/storage/test_lock.py packages/pivot/tests/core/test_skip.py
git commit -m "feat: store identities in locks and state keys"
```

---

### Task 3: Worker/explain hashing uses Store + identity keys

**Files:**
- Modify: `packages/pivot/src/pivot/executor/worker.py`
- Modify: `packages/pivot/src/pivot/explain.py`
- Test: `packages/pivot/tests/test_run_cache_lock_update.py`
- Test: `packages/pivot/tests/test_skip_detection_integration.py`

**Step 1: Write the failing test**

The four failing tests already act as the RED cases. No new test needed.

**Step 2: Run tests to verify they fail**

```bash
uv run pytest packages/pivot/tests/test_run_cache_lock_update.py::test_explain_shows_cached_after_run_cache_skip -v
uv run pytest packages/pivot/tests/test_skip_detection_integration.py::test_generation_skip_with_pivot_produced_deps -v
```

Expected: FAIL (identity/path mismatches)

**Step 3: Write minimal implementation**

- Ensure worker hash_dependencies returns identity‑keyed `dep_hashes` and `file_hash_entries`.
- Use Store‑resolved paths for hashing to avoid absolute/relative drift.
- Ensure `apply_deferred_writes` is called with identity keys for output generations.
- Update explain to compare identity‑keyed hashes only; remove any reliance on path string equality.

**Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/pivot/tests/test_run_cache_lock_update.py -v
uv run pytest packages/pivot/tests/test_skip_detection_integration.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add packages/pivot/src/pivot/executor/worker.py packages/pivot/src/pivot/explain.py packages/pivot/tests/test_run_cache_lock_update.py packages/pivot/tests/test_skip_detection_integration.py
git commit -m "fix: normalize identity hashing in worker/explain"
```

---

### Task 4: CLI/TUI identity display + RPC updates

**Files:**
- Modify: `packages/pivot/src/pivot/cli/console.py`
- Modify: `packages/pivot/src/pivot/cli/status.py`
- Modify: `packages/pivot/src/pivot/engine/agent_rpc.py`
- Modify: `packages/pivot-tui/src/pivot_tui/diff_panels.py`
- Modify: `packages/pivot-tui/src/pivot_tui/event_poller.py`
- Test: `packages/pivot-tui/tests/test_diff_panels.py`
- Test: `packages/pivot/tests/cli/test_console.py` (or closest existing CLI tests)

**Step 1: Write the failing test**

Add a TUI test that expects dep changes keyed by identity strings derived from `ArtifactIdentity`.

**Step 2: Run test to verify it fails**

```bash
uv run pytest packages/pivot-tui/tests/test_diff_panels.py -v
```

Expected: FAIL (still using `path` key)

**Step 3: Write minimal implementation**

- Update CLI explain output to render `ArtifactIdentity` via a single helper (producer or producer:key).
- Update TUI `InputDiffPanel` and `OutputDiffPanel` to index changes by identity display string, not `path`.
- Update RPC server responses to emit identity JSON objects; TUI client decodes to `ArtifactIdentity`.

**Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/pivot-tui/tests/test_diff_panels.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add packages/pivot/src/pivot/cli/console.py packages/pivot/src/pivot/cli/status.py packages/pivot/src/pivot/engine/agent_rpc.py packages/pivot-tui/src/pivot_tui/diff_panels.py packages/pivot-tui/src/pivot_tui/event_poller.py packages/pivot-tui/tests/test_diff_panels.py
git commit -m "feat: render artifact identities in CLI/TUI"
```

---

### Task 5: Presentation layer (CookieCutterFS) integration

**Files:**
- Create: `packages/pivot/src/pivot/storage/presentation.py`
- Modify: `packages/pivot/src/pivot/storage/cache.py`
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Test: `packages/pivot/tests/storage/test_presentation.py`

**Step 1: Write the failing test**

```python
def test_presentation_tree_symlinks(tmp_path: Path) -> None:
    # Build fake CAS entries, call present(), verify symlink layout
    ...
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest packages/pivot/tests/storage/test_presentation.py -v
```

Expected: FAIL (presentation module missing)

**Step 3: Write minimal implementation**

- Implement `present()` to create a symlink tree for outputs by tag (data/metrics/plots).
- Use CAS link helpers from `cache.py` for atomic symlink creation.
- Materialize group artifacts as directories with per‑key files.
- Call `present()` at engine run completion when not in watch mode.

**Step 4: Run tests to verify they pass**

```bash
uv run pytest packages/pivot/tests/storage/test_presentation.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add packages/pivot/src/pivot/storage/presentation.py packages/pivot/src/pivot/storage/cache.py packages/pivot/src/pivot/engine/engine.py packages/pivot/tests/storage/test_presentation.py
git commit -m "feat: add presentation symlink tree"
```

---

### Task 6: Full verification + eval pipeline

**Files:**
- Modify (if needed): `packages/pivot/tests/...`, `packages/pivot-tui/tests/...`

**Step 1: Run quality checks**

```bash
uv run ruff format .
uv run ruff check .
uv run basedpyright
```

**Step 2: Run tests**

```bash
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
```

**Step 3: End‑to‑end verification**

```bash
cd /home/sami/eval-pipeline/pivot
pivot repro
pivot repro
```

Expected: first run succeeds, second run fully cached; lockfiles have identity entries and Merkle IDs; presentation symlinks exist.

**Step 4: Commit**

```bash
git add .
git commit -m "test: verify path-free engine end-to-end"
```

---

## Execution Mode

Proceed with **Subagent‑Driven** execution in this session; dispatch one task at a time and review each diff.
