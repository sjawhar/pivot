# Watch Mode: Restart Worker Pool on Code/Config Reload


**Goal:** When watch mode detects code/config changes, restart the loky worker pool so workers execute updated code instead of stale modules.

**Architecture:** `_handle_code_or_config_changed` already reloads the pipeline registry and clears module caches. We add a `restart_workers()` call after successful reload but before re-execution. We also persist `parallel` and `max_workers` from the initial `RunRequested` event so re-runs use the same sizing.

**Tech Stack:** Python 3.13+, loky, anyio, pytest

---

### Task 1: Persist `parallel` and `max_workers` in stored orchestration params

The engine already stores `no_commit` and `on_error` from the initial `RunRequested` for watch-mode re-runs (see `engine.py:103-106`). But `parallel` and `max_workers` are NOT stored — `_execute_affected_stages` hardcodes `parallel=True, max_workers=None`. We need to persist these too.

**Files:**
- Modify: `src/pivot/engine/engine.py:98-106` (add stored fields)
- Modify: `src/pivot/engine/engine.py:147-150` (add init defaults)
- Modify: `src/pivot/engine/engine.py:341-346` (store on RunRequested)
- Modify: `src/pivot/engine/engine.py:1200-1218` (use stored values in _execute_affected_stages)
- Test: `tests/engine/test_engine.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_engine.py`:

```python
@pytest.mark.anyio
async def test_engine_stores_parallel_and_max_workers_on_run_requested() -> None:
    """Engine stores parallel/max_workers from RunRequested for watch re-runs."""
    from pivot.engine.types import RunRequested
    from pivot.types import OnError

    async with Engine() as engine:
        event = RunRequested(
            type="run_requested",
            stages=None,
            force=False,
            reason="test",
            single_stage=False,
            parallel=True,
            max_workers=4,
            no_commit=False,
            on_error=OnError.FAIL,
            cache_dir=None,
            allow_uncached_incremental=False,
            checkout_missing=False,
        )

        async def mock_orchestrate(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {}

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue]
        engine._require_pipeline = lambda: None  # pyright: ignore[reportAttributeAccessIssue]

        await engine._handle_run_requested(event)

        assert engine._stored_parallel is True
        assert engine._stored_max_workers == 4
```

**Step 2: Run test to verify it fails**

Run: `cd /home/sami/pivot/roadmap-376 && uv run pytest tests/engine/test_engine.py::test_engine_stores_parallel_and_max_workers_on_run_requested -v`
Expected: FAIL with `AttributeError: 'Engine' object has no attribute '_stored_parallel'`

**Step 3: Add stored fields and update _handle_run_requested**

In `src/pivot/engine/engine.py`, add instance variables alongside the existing stored params:

```python
# Near line 98 (class-level annotations section):
_stored_parallel: bool
_stored_max_workers: int | None

# Near line 147 (in __init__, alongside existing stored param defaults):
self._stored_parallel = True
self._stored_max_workers = None

# In _handle_run_requested (near line 344-346), add after existing storage:
self._stored_parallel = event["parallel"]
self._stored_max_workers = event["max_workers"]
```

**Step 4: Run test to verify it passes**

Run: `cd /home/sami/pivot/roadmap-376 && uv run pytest tests/engine/test_engine.py::test_engine_stores_parallel_and_max_workers_on_run_requested -v`
Expected: PASS

**Step 5: Write test for _execute_affected_stages using stored params**

Add to `tests/engine/test_engine.py`:

```python
@pytest.mark.anyio
async def test_engine_execute_affected_stages_uses_stored_parallel_and_max_workers() -> None:
    """_execute_affected_stages() uses stored parallel/max_workers instead of hardcoded defaults."""
    from pivot.types import OnError

    async with Engine() as engine:
        engine._stored_parallel = True
        engine._stored_max_workers = 4
        engine._stored_no_commit = False
        engine._stored_on_error = OnError.FAIL

        captured_kwargs: dict[str, object] = {}

        async def mock_orchestrate(**kwargs: object) -> dict[str, object]:
            captured_kwargs.update(kwargs)
            return {}

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue]

        await engine._execute_affected_stages(["stage_a"])

        assert captured_kwargs["parallel"] is True
        assert captured_kwargs["max_workers"] == 4
```

**Step 6: Run test to verify it fails**

Run: `cd /home/sami/pivot/roadmap-376 && uv run pytest tests/engine/test_engine.py::test_engine_execute_affected_stages_uses_stored_parallel_and_max_workers -v`
Expected: FAIL — `parallel` is hardcoded to `True` and `max_workers` to `None`

**Step 7: Update _execute_affected_stages to use stored params**

In `_execute_affected_stages` (around line 1208), change:

```python
# Before:
parallel=True,
max_workers=None,

# After:
parallel=self._stored_parallel,
max_workers=self._stored_max_workers,
```

**Step 8: Run tests to verify both pass**

Run: `cd /home/sami/pivot/roadmap-376 && uv run pytest tests/engine/test_engine.py::test_engine_stores_parallel_and_max_workers_on_run_requested tests/engine/test_engine.py::test_engine_execute_affected_stages_uses_stored_parallel_and_max_workers -v`
Expected: PASS

---

### Task 2: Restart worker pool after code/config reload

Wire `executor_core.restart_workers()` into `_handle_code_or_config_changed` after successful registry reload, before re-execution.

**Files:**
- Modify: `src/pivot/engine/engine.py:1162-1198` (`_handle_code_or_config_changed`)
- Test: `tests/engine/test_engine.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_engine.py`:

```python
@pytest.mark.anyio
async def test_engine_handle_code_or_config_changed_restarts_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_handle_code_or_config_changed restarts worker pool after successful reload."""
    from unittest.mock import MagicMock

    from pivot.engine.types import CodeOrConfigChanged

    restart_calls = list[tuple[int, int | None]]()

    def mock_restart_workers(stage_count: int, max_workers: int | None = None) -> int:
        restart_calls.append((stage_count, max_workers))
        return stage_count

    monkeypatch.setattr("pivot.executor.core.restart_workers", mock_restart_workers)

    async with Engine() as engine:
        engine._stored_parallel = True
        engine._stored_max_workers = 4

        # Mock reload to succeed with 3 stages
        mock_pipeline = MagicMock()
        mock_pipeline.list_stages.return_value = ["a", "b", "c"]
        mock_pipeline.snapshot.return_value = {}
        mock_pipeline._registry = None
        mock_pipeline.get.return_value = MagicMock()
        mock_pipeline.resolve_external_dependencies.return_value = None
        mock_pipeline.invalidate_dag_cache.return_value = None
        engine._pipeline = mock_pipeline

        # Mock _reload_registry to return success
        engine._reload_registry = lambda: ({}, None)  # pyright: ignore[reportAttributeAccessIssue]

        # Mock _execute_affected_stages to avoid actual execution
        engine._execute_affected_stages = lambda stages: None  # pyright: ignore[reportAttributeAccessIssue]

        # Mock graph building
        monkeypatch.setattr(
            "pivot.engine.engine.engine_graph.build_graph", lambda *a, **kw: MagicMock()
        )
        monkeypatch.setattr(
            "pivot.engine.engine.engine_graph.get_watch_paths", lambda *a, **kw: set()
        )

        event = CodeOrConfigChanged(type="code_or_config_changed", paths=["pipeline.py"])
        await engine._handle_code_or_config_changed(event)

        assert len(restart_calls) == 1
        assert restart_calls[0] == (3, 4)  # 3 stages, max_workers=4
```

**Step 2: Run test to verify it fails**

Run: `cd /home/sami/pivot/roadmap-376 && uv run pytest tests/engine/test_engine.py::test_engine_handle_code_or_config_changed_restarts_workers -v`
Expected: FAIL — `restart_workers` is never called

**Step 3: Add restart_workers call to _handle_code_or_config_changed**

In `_handle_code_or_config_changed` (around line 1196), after rebuilding the graph and before re-running stages, add:

```python
        # Re-run all stages
        stages = self._list_stages()

        if stages:
            # Restart worker pool so workers pick up reloaded code
            if self._stored_parallel:
                executor_core.restart_workers(len(stages), self._stored_max_workers)
                _logger.info("Worker pool restarted for code reload (%d stages)", len(stages))

            await self._execute_affected_stages(stages)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/sami/pivot/roadmap-376 && uv run pytest tests/engine/test_engine.py::test_engine_handle_code_or_config_changed_restarts_workers -v`
Expected: PASS

**Step 5: Write test that data-only changes do NOT restart workers**

Add to `tests/engine/test_engine.py`:

```python
@pytest.mark.anyio
async def test_engine_handle_data_artifact_changed_does_not_restart_workers(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_handle_data_artifact_changed does NOT restart worker pool (data-only change)."""
    restart_calls = list[object]()

    def mock_restart_workers(*args: object, **kwargs: object) -> int:
        restart_calls.append(args)
        return 1

    monkeypatch.setattr("pivot.executor.core.restart_workers", mock_restart_workers)

    async with Engine() as engine:
        # Build minimal graph: input.csv -> stage_a
        input_path = tmp_path / "input.csv"

        g: nx.DiGraph[str] = nx.DiGraph()
        input_node = engine_graph.artifact_node(input_path)
        stage_node = engine_graph.stage_node("stage_a")
        g.add_node(input_node, type=NodeType.ARTIFACT)
        g.add_node(stage_node, type=NodeType.STAGE)
        g.add_edge(input_node, stage_node)
        engine._graph = g

        # Mock _execute_affected_stages to avoid actual execution
        executed_stages = list[list[str]]()

        async def mock_execute(stages: list[str]) -> None:
            executed_stages.append(stages)

        engine._execute_affected_stages = mock_execute  # pyright: ignore[reportAttributeAccessIssue]

        event = DataArtifactChanged(type="data_artifact_changed", paths=[str(input_path)])
        await engine._handle_data_artifact_changed(event)

        assert len(restart_calls) == 0, "Data changes should NOT restart workers"
```

**Step 6: Run test to verify it passes (it should already pass)**

Run: `cd /home/sami/pivot/roadmap-376 && uv run pytest tests/engine/test_engine.py::test_engine_handle_data_artifact_changed_does_not_restart_workers -v`
Expected: PASS (no code change needed — data path never calls restart_workers)

---

### Task 3: Handle non-parallel mode (skip restart when parallel=False)

When the user starts with `--no-parallel`, we skip worker pool restart since execution is single-process.

**Files:**
- Test: `tests/engine/test_engine.py`

**Step 1: Write test for non-parallel mode**

Add to `tests/engine/test_engine.py`:

```python
@pytest.mark.anyio
async def test_engine_handle_code_or_config_changed_skips_restart_when_not_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_handle_code_or_config_changed skips worker restart when parallel=False."""
    from unittest.mock import MagicMock

    from pivot.engine.types import CodeOrConfigChanged

    restart_calls = list[object]()

    def mock_restart_workers(*args: object, **kwargs: object) -> int:
        restart_calls.append(args)
        return 1

    monkeypatch.setattr("pivot.executor.core.restart_workers", mock_restart_workers)

    async with Engine() as engine:
        engine._stored_parallel = False  # Non-parallel mode
        engine._stored_max_workers = None

        mock_pipeline = MagicMock()
        mock_pipeline.list_stages.return_value = ["a", "b"]
        mock_pipeline.snapshot.return_value = {}
        mock_pipeline._registry = None
        mock_pipeline.resolve_external_dependencies.return_value = None
        mock_pipeline.invalidate_dag_cache.return_value = None
        engine._pipeline = mock_pipeline

        engine._reload_registry = lambda: ({}, None)  # pyright: ignore[reportAttributeAccessIssue]
        engine._execute_affected_stages = lambda stages: None  # pyright: ignore[reportAttributeAccessIssue]

        monkeypatch.setattr(
            "pivot.engine.engine.engine_graph.build_graph", lambda *a, **kw: MagicMock()
        )
        monkeypatch.setattr(
            "pivot.engine.engine.engine_graph.get_watch_paths", lambda *a, **kw: set()
        )

        event = CodeOrConfigChanged(type="code_or_config_changed", paths=["pipeline.py"])
        await engine._handle_code_or_config_changed(event)

        assert len(restart_calls) == 0, "Should not restart workers in non-parallel mode"
```

**Step 2: Run test to verify it passes**

Run: `cd /home/sami/pivot/roadmap-376 && uv run pytest tests/engine/test_engine.py::test_engine_handle_code_or_config_changed_skips_restart_when_not_parallel -v`
Expected: PASS (the `if self._stored_parallel:` guard already handles this)

---

### Task 4: Handle failed reload (no restart when registry reload fails)

When `_reload_registry()` returns `None` (pipeline invalid), we must NOT restart workers.

**Files:**
- Test: `tests/engine/test_engine.py`

**Step 1: Write test for failed reload**

Add to `tests/engine/test_engine.py`:

```python
@pytest.mark.anyio
async def test_engine_handle_code_or_config_changed_no_restart_on_failed_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_handle_code_or_config_changed does NOT restart workers when reload fails."""
    from pivot.engine.types import CodeOrConfigChanged

    restart_calls = list[object]()

    def mock_restart_workers(*args: object, **kwargs: object) -> int:
        restart_calls.append(args)
        return 1

    monkeypatch.setattr("pivot.executor.core.restart_workers", mock_restart_workers)

    async with Engine() as engine:
        engine._stored_parallel = True

        # Mock reload to FAIL
        engine._reload_registry = lambda: None  # pyright: ignore[reportAttributeAccessIssue]
        engine._invalidate_caches = lambda: None  # pyright: ignore[reportAttributeAccessIssue]

        event = CodeOrConfigChanged(type="code_or_config_changed", paths=["pipeline.py"])
        await engine._handle_code_or_config_changed(event)

        assert len(restart_calls) == 0, "Should not restart workers on failed reload"
```

**Step 2: Run test to verify it passes**

Run: `cd /home/sami/pivot/roadmap-376 && uv run pytest tests/engine/test_engine.py::test_engine_handle_code_or_config_changed_no_restart_on_failed_reload -v`
Expected: PASS (the early return at line 1174 prevents reaching the restart code)

---

### Task 5: Run full test suite and quality checks

**Step 1: Run all engine tests**

Run: `cd /home/sami/pivot/roadmap-376 && uv run pytest tests/engine/test_engine.py -v`
Expected: All tests PASS

**Step 2: Run ruff format and check**

Run: `cd /home/sami/pivot/roadmap-376 && uv run ruff format . && uv run ruff check .`
Expected: No errors

**Step 3: Run basedpyright**

Run: `cd /home/sami/pivot/roadmap-376 && uv run basedpyright`
Expected: No new errors

**Step 4: Run full test suite**

Run: `cd /home/sami/pivot/roadmap-376 && uv run pytest tests/ -n auto`
Expected: All tests PASS

---

## Summary of Changes

**`src/pivot/engine/engine.py`:**
1. Add `_stored_parallel: bool` and `_stored_max_workers: int | None` fields (alongside existing stored params)
2. Initialize to `True` and `None` in `__init__`
3. Store from `RunRequested` event in `_handle_run_requested`
4. Use stored values in `_execute_affected_stages` instead of hardcoded `parallel=True, max_workers=None`
5. Call `executor_core.restart_workers()` in `_handle_code_or_config_changed` after successful reload, guarded by `self._stored_parallel`

**No changes to `src/pivot/executor/core.py`** — `restart_workers()` already exists and works correctly.

**Tests added to `tests/engine/test_engine.py`:**
1. `test_engine_stores_parallel_and_max_workers_on_run_requested` — verifies fields are stored
2. `test_engine_execute_affected_stages_uses_stored_parallel_and_max_workers` — verifies stored values are used
3. `test_engine_handle_code_or_config_changed_restarts_workers` — verifies restart_workers called
4. `test_engine_handle_data_artifact_changed_does_not_restart_workers` — verifies data-only doesn't restart
5. `test_engine_handle_code_or_config_changed_skips_restart_when_not_parallel` — non-parallel guard
6. `test_engine_handle_code_or_config_changed_no_restart_on_failed_reload` — failed reload guard
