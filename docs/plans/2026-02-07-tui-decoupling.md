# TUI Decoupling: Remove CLI Context Dependency


**Goal:** Remove all `pivot.cli.helpers` imports from TUI modules by passing registry lookups as explicit constructor arguments, so the TUI is no longer coupled to Click context.

**Architecture:** Define a `StageDataProvider` Protocol in `tui/types.py` with two methods: `get_stage()` and `ensure_fingerprint()`. Pass a provider instance to `PivotApp`, which forwards it to `InputDiffPanel` and `OutputDiffPanel`. The CLI creates a concrete provider from the Pipeline it already holds. The TUI becomes a pure display client that receives data through explicit dependencies — no more reaching into Click context via `cli_helpers`.

**Tech Stack:** Python 3.13+, Textual (existing), Protocol (typing)

---

## Current Coupling

8 call sites in 2 TUI files reach through `cli_helpers` to the Click-context-bound Pipeline:

| File | Line | Call | Purpose |
|------|------|------|---------|
| `run.py` | 573 | `cli_helpers.get_stage(name)` | Input snapshot for history |
| `run.py` | 574 | `cli_helpers.get_registry().ensure_fingerprint(name)` | Input snapshot fingerprint |
| `run.py` | 630 | `cli_helpers.get_stage(name)` | Output snapshot for history |
| `diff_panels.py` | 432 | `cli_helpers.get_stage(name)` | InputDiffPanel load |
| `diff_panels.py` | 439 | `cli_helpers.get_registry().ensure_fingerprint(name)` | InputDiffPanel fingerprint |
| `diff_panels.py` | 692 | `cli_helpers.get_stage(name)` | InputDiffPanel snapshot |
| `diff_panels.py` | 744 | `cli_helpers.get_stage(name)` | OutputDiffPanel load |
| `diff_panels.py` | 1119 | `cli_helpers.get_stage(name)` | OutputDiffPanel snapshot |

## Target State

- Zero `pivot.cli.helpers` imports in `src/pivot/tui/`
- `StageDataProvider` Protocol in `tui/types.py`
- `PivotApp.__init__` accepts `stage_data_provider: StageDataProvider`
- `InputDiffPanel` and `OutputDiffPanel` receive provider via constructor
- CLI creates a `PipelineStageDataProvider` wrapper and passes it in

---

### Task 1: Add `StageDataProvider` Protocol to `tui/types.py`

**Files:**
- Modify: `src/pivot/tui/types.py`
- Test: `tests/tui/test_run.py` (verify import works)

**Step 1: Write the test**

Add to `tests/tui/test_run.py` (new test at the bottom):

```python
def test_stage_data_provider_protocol_is_importable() -> None:
    """StageDataProvider protocol can be imported from tui.types."""
    from pivot.tui.types import StageDataProvider
    assert hasattr(StageDataProvider, "get_stage")
    assert hasattr(StageDataProvider, "ensure_fingerprint")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tui/test_run.py::test_stage_data_provider_protocol_is_importable -xvs`
Expected: FAIL with `ImportError`

**Step 3: Add the Protocol to `tui/types.py`**

Add these imports at the top of `src/pivot/tui/types.py` (after existing imports):

```python
from typing import Protocol
```

And under `TYPE_CHECKING`:

```python
if TYPE_CHECKING:
    from pivot.registry import RegistryStageInfo
```

Then add the Protocol class after `parse_stage_name` but before `LogEntry`:

```python
class StageDataProvider(Protocol):
    """Protocol for TUI to look up stage metadata without CLI context.

    Decouples the TUI from pivot.cli.helpers by defining the two
    operations the TUI actually needs from the registry.
    """

    def get_stage(self, name: str) -> RegistryStageInfo:
        """Look up stage metadata by name. Raises KeyError if not found."""
        ...

    def ensure_fingerprint(self, name: str) -> dict[str, str]:
        """Compute/return cached code fingerprint for a stage."""
        ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tui/test_run.py::test_stage_data_provider_protocol_is_importable -xvs`
Expected: PASS

**Step 5: Run basedpyright**

Run: `uv run basedpyright src/pivot/tui/types.py`
Expected: 0 errors

---

### Task 2: Add `stage_data_provider` to `PivotApp.__init__`

**Files:**
- Modify: `src/pivot/tui/run.py`
- Test: `tests/tui/test_run.py`

**Step 1: Write the test**

Add to `tests/tui/test_run.py`:

```python
def test_pivot_app_accepts_stage_data_provider() -> None:
    """PivotApp stores stage_data_provider when passed."""
    from pivot.tui.types import StageDataProvider

    class FakeProvider:
        def get_stage(self, name: str) -> dict:
            return {}
        def ensure_fingerprint(self, name: str) -> dict[str, str]:
            return {}

    provider: StageDataProvider = FakeProvider()
    app = run_tui.PivotApp(stage_names=["s1"], stage_data_provider=provider)
    assert app._stage_data_provider is provider
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tui/test_run.py::test_pivot_app_accepts_stage_data_provider -xvs`
Expected: FAIL with `TypeError` (unexpected keyword argument)

**Step 3: Modify `PivotApp.__init__`**

In `src/pivot/tui/run.py`:

1. Add import in the existing `TYPE_CHECKING` block (or alongside the existing types import from `tui.types`):

```python
from pivot.tui.types import StageDataProvider
```

(Add `StageDataProvider` to the existing import line: `from pivot.tui.types import ExecutionHistoryEntry, LogEntry, PendingHistoryState, StageDataProvider, StageInfo`)

2. Add `stage_data_provider` parameter to `__init__`:

In the `__init__` signature (after `serve: bool = False`), add:
```python
stage_data_provider: StageDataProvider | None = None,
```

3. Store it in the body (after `self._quit_lock`):
```python
self._stage_data_provider: StageDataProvider | None = stage_data_provider
```

4. Add the instance attribute annotation at the class level (near `_cancel_event`):
```python
_stage_data_provider: StageDataProvider | None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tui/test_run.py::test_pivot_app_accepts_stage_data_provider -xvs`
Expected: PASS

**Step 5: Run basedpyright**

Run: `uv run basedpyright src/pivot/tui/run.py`
Expected: 0 errors

---

### Task 3: Wire `stage_data_provider` into `run.py` history methods

**Files:**
- Modify: `src/pivot/tui/run.py` (lines 569-650)
- Test: `tests/tui/test_run.py`

This task replaces the 3 `cli_helpers` calls in `run.py` with `self._stage_data_provider`.

**Step 1: Write the test**

Add to `tests/tui/test_run.py`:

```python
def test_create_history_entry_uses_provider(mocker: MockerFixture) -> None:
    """_create_history_entry uses stage_data_provider instead of cli_helpers."""
    from pivot.tui.types import StageDataProvider

    mock_provider = mocker.MagicMock(spec=StageDataProvider)
    mock_provider.get_stage.return_value = {
        "deps_paths": [],
        "outs_paths": [],
        "params": None,
    }
    mock_provider.ensure_fingerprint.return_value = {"func": "abc123"}

    app = run_tui.PivotApp(
        stage_names=["stage_a"],
        watch_mode=True,
        stage_data_provider=mock_provider,
    )

    # Mock explain to avoid real IO
    mocker.patch("pivot.tui.run.explain.get_stage_explanation", return_value=None)
    mocker.patch("pivot.tui.run.parameters.load_params_yaml", return_value={})
    mocker.patch("pivot.tui.run.config.get_state_dir", return_value=pathlib.Path("/fake"))

    app._create_history_entry("stage_a", "run-1")

    mock_provider.get_stage.assert_called_with("stage_a")
    mock_provider.ensure_fingerprint.assert_called_with("stage_a")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tui/test_run.py::test_create_history_entry_uses_provider -xvs`
Expected: FAIL (still calls `cli_helpers`)

**Step 3: Replace `cli_helpers` calls in `_create_history_entry` and `_finalize_history_entry`**

In `src/pivot/tui/run.py`, modify `_create_history_entry` (around line 569-592):

Replace:
```python
        input_snapshot = None
        try:
            registry_info = cli_helpers.get_stage(stage_name)
            fingerprint = cli_helpers.get_registry().ensure_fingerprint(stage_name)
```

With:
```python
        input_snapshot = None
        if self._stage_data_provider is None:
            self._pending_history[stage_name] = PendingHistoryState(
                run_id=run_id,
                timestamp=time.time(),
            )
            return
        try:
            registry_info = self._stage_data_provider.get_stage(stage_name)
            fingerprint = self._stage_data_provider.ensure_fingerprint(stage_name)
```

Modify `_finalize_history_entry` (around line 628-634):

Replace:
```python
            try:
                registry_info = cli_helpers.get_stage(stage_name)
```

With:
```python
            if self._stage_data_provider is not None:
              try:
                registry_info = self._stage_data_provider.get_stage(stage_name)
```

(And adjust the indentation of the subsequent `state_dir`, `stages_dir`, `lock_data`, `output_snapshot` lines to be inside this new `if` block. If `_stage_data_provider` is None, `output_snapshot` stays None.)

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tui/test_run.py::test_create_history_entry_uses_provider -xvs`
Expected: PASS

**Step 5: Run full TUI tests**

Run: `uv run pytest tests/tui/ -x --tb=short`
Expected: All pass (existing tests don't pass a provider, so `_stage_data_provider` is None and history capture gracefully degrades)

---

### Task 4: Add `stage_data_provider` to diff panels

**Files:**
- Modify: `src/pivot/tui/diff_panels.py`
- Test: `tests/tui/test_diff_panels.py`

This task replaces the 5 `cli_helpers` calls in `diff_panels.py`.

**Step 1: Write the test**

Add to `tests/tui/test_diff_panels.py`:

```python
def test_input_panel_load_uses_provider(mocker: MockerFixture) -> None:
    """InputDiffPanel._load_stage_data uses provider instead of cli_helpers."""
    from pivot.tui.types import StageDataProvider

    mock_provider = mocker.MagicMock(spec=StageDataProvider)
    mock_provider.get_stage.return_value = {
        "deps_paths": [],
        "outs_paths": [],
        "params": None,
    }
    mock_provider.ensure_fingerprint.return_value = {"func": "abc"}

    panel = diff_panels.InputDiffPanel(stage_data_provider=mock_provider)

    # Mock explain to avoid real IO
    mocker.patch("pivot.tui.diff_panels.explain.get_stage_explanation", return_value=None)
    mocker.patch("pivot.tui.diff_panels.parameters.load_params_yaml", return_value={})
    mocker.patch("pivot.tui.diff_panels.config.get_state_dir", return_value=pathlib.Path("/fake"))

    panel._load_stage_data("my_stage")

    mock_provider.get_stage.assert_called_with("my_stage")
    mock_provider.ensure_fingerprint.assert_called_with("my_stage")


def test_output_panel_load_uses_provider(mocker: MockerFixture) -> None:
    """OutputDiffPanel._load_stage_data uses provider instead of cli_helpers."""
    from pivot.tui.types import StageDataProvider

    mock_provider = mocker.MagicMock(spec=StageDataProvider)
    mock_provider.get_stage.return_value = {
        "deps_paths": [],
        "outs_paths": [],
        "outs": [],
        "params": None,
    }

    panel = diff_panels.OutputDiffPanel(stage_data_provider=mock_provider)

    mocker.patch("pivot.tui.diff_panels.config.get_state_dir", return_value=pathlib.Path("/fake"))
    mocker.patch("pivot.tui.diff_panels.lock.StageLock")

    panel._load_stage_data("my_stage")

    mock_provider.get_stage.assert_called_with("my_stage")
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tui/test_diff_panels.py::test_input_panel_load_uses_provider tests/tui/test_diff_panels.py::test_output_panel_load_uses_provider -xvs`
Expected: FAIL (no `stage_data_provider` parameter)

**Step 3: Add `stage_data_provider` to diff panels**

In `src/pivot/tui/diff_panels.py`:

1. Add import (add `StageDataProvider` to existing import from `pivot.tui.types` — there's currently no import from `tui.types`, so add it under the existing TYPE_CHECKING block or as a runtime import):

Under `TYPE_CHECKING`:
```python
from pivot.tui.types import StageDataProvider
```

2. Modify `InputDiffPanel.__init__`:

Add parameter:
```python
def __init__(self, *, id: str | None = None, classes: str | None = None, stage_data_provider: StageDataProvider | None = None) -> None:
```

Store it:
```python
self._stage_data_provider: StageDataProvider | None = stage_data_provider
```

Add class-level annotation:
```python
_stage_data_provider: StageDataProvider | None
```

3. Replace `cli_helpers` calls in `InputDiffPanel._load_stage_data` (line 429-451):

Replace `cli_helpers.get_stage(stage_name)` with:
```python
if self._stage_data_provider is None:
    return
self._registry_info = self._stage_data_provider.get_stage(stage_name)
```

Replace `cli_helpers.get_registry().ensure_fingerprint(stage_name)` with:
```python
fingerprint = self._stage_data_provider.ensure_fingerprint(stage_name)
```

4. Replace `cli_helpers.get_stage` in `InputDiffPanel.set_from_snapshot` (line 692):

Replace:
```python
self._registry_info = cli_helpers.get_stage(snapshot["stage_name"])
```
With:
```python
if self._stage_data_provider is not None:
    try:
        self._registry_info = self._stage_data_provider.get_stage(snapshot["stage_name"])
    except KeyError:
        self._registry_info = None
```

5. Do the same for `OutputDiffPanel`:

Add `stage_data_provider` to `__init__`, store it, add class annotation.

Replace `cli_helpers.get_stage` in `_load_stage_data` (line 744) and `set_from_snapshot` (line 1119).

6. Remove the import: `from pivot.cli import helpers as cli_helpers` from `diff_panels.py`.

**Step 4: Run tests**

Run: `uv run pytest tests/tui/test_diff_panels.py -x --tb=short`
Expected: All pass

**Step 5: Run basedpyright**

Run: `uv run basedpyright src/pivot/tui/diff_panels.py`
Expected: 0 errors

---

### Task 5: Wire provider through `PivotApp` to diff panels

**Files:**
- Modify: `src/pivot/tui/run.py` (the `_update_history_view` and `compose` or panel creation)
- Modify: `src/pivot/tui/widgets/panels.py` (TabbedDetailPanel creates InputDiffPanel/OutputDiffPanel)

The diff panels are created inside `TabbedDetailPanel.compose()`. We need to pass `stage_data_provider` through.

**Step 1: Find where panels are created**

Read `src/pivot/tui/widgets/panels.py` to find `TabbedDetailPanel.compose()` and how it creates `InputDiffPanel` and `OutputDiffPanel`.

**Step 2: Add `stage_data_provider` to `TabbedDetailPanel.__init__`**

Pass it through to `InputDiffPanel(stage_data_provider=...)` and `OutputDiffPanel(stage_data_provider=...)` in `compose()`.

**Step 3: Pass it from `PivotApp.compose()`**

In `PivotApp.compose()`, pass `stage_data_provider=self._stage_data_provider` when creating `TabbedDetailPanel`.

**Step 4: Run full TUI tests**

Run: `uv run pytest tests/tui/ -x --tb=short`
Expected: All pass

---

### Task 6: Remove `cli_helpers` import from `run.py`

**Files:**
- Modify: `src/pivot/tui/run.py`

**Step 1: Remove the import line**

Remove: `from pivot.cli import helpers as cli_helpers`

**Step 2: Verify no remaining references**

Run: `uv run basedpyright src/pivot/tui/run.py`
Expected: 0 errors

Run: `grep -r "cli_helpers" src/pivot/tui/`
Expected: No matches

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -n auto --tb=short -q`
Expected: All pass

---

### Task 7: Update CLI callers to pass provider

**Files:**
- Modify: `src/pivot/cli/run.py` (line ~103)
- Modify: `src/pivot/cli/repro.py` (lines ~381, ~614)

The CLI already has the Pipeline. Create a simple wrapper and pass it.

**Step 1: Create `PipelineStageDataProvider` in CLI**

In each CLI file that creates `PivotApp`, create the provider from the pipeline. Since this is a simple 2-method wrapper, don't create a separate module — just define it inline or in `cli/helpers.py`.

Actually, simplest approach: add a factory function to `cli/helpers.py`:

```python
def make_stage_data_provider(pipeline: Pipeline) -> StageDataProvider:
    """Create a StageDataProvider from a Pipeline for TUI use."""
    class _Provider:
        def get_stage(self, name: str) -> RegistryStageInfo:
            return pipeline.get(name)
        def ensure_fingerprint(self, name: str) -> dict[str, str]:
            return pipeline._registry.ensure_fingerprint(name)
    return _Provider()
```

Wait — that keeps the import in `cli/helpers.py`, which is fine since `cli/helpers.py` IS the CLI layer. But actually, `Pipeline` itself already has `get()` and access to `_registry`. The simplest approach is to make `Pipeline` satisfy the `StageDataProvider` protocol directly by adding an `ensure_fingerprint` method.

**Actually, the cleanest approach:** Pipeline already has `get(name)` which returns `RegistryStageInfo`. It just needs `ensure_fingerprint(name)`. Let's add it.

In `src/pivot/pipeline/pipeline.py`, add a public method:

```python
def ensure_fingerprint(self, stage_name: str) -> dict[str, str]:
    """Compute/return cached code fingerprint for a stage."""
    return self._registry.ensure_fingerprint(stage_name)
```

Then `Pipeline` structurally satisfies `StageDataProvider` (it has `get()` but we need `get_stage()` — ah, Pipeline uses `get()` not `get_stage()`).

Two options:
1. Name the Protocol method `get()` instead of `get_stage()` — but `get` is generic.
2. Add a `get_stage()` alias to Pipeline.
3. Use a simple lambda adapter in the CLI.

**Best approach: Name the Protocol methods to match Pipeline's existing API.** Pipeline has:
- `get(name) -> RegistryStageInfo`
- (need to add) `ensure_fingerprint(name) -> dict[str, str]`

So define the Protocol as:
```python
class StageDataProvider(Protocol):
    def get(self, name: str) -> RegistryStageInfo: ...
    def ensure_fingerprint(self, name: str) -> dict[str, str]: ...
```

Wait — but then all 8 call sites in the TUI use `provider.get(name)` which looks like `dict.get()`. That's confusing. Let's keep `get_stage` in the Protocol and add a one-line `get_stage` to Pipeline (it just delegates to `get`).

**Revised approach in Task 1:** The Protocol uses `get_stage()`. In this task, add `get_stage()` and `ensure_fingerprint()` to Pipeline, so Pipeline structurally satisfies the protocol.

**Step 1: Add methods to Pipeline**

In `src/pivot/pipeline/pipeline.py`, add:

```python
def get_stage(self, name: str) -> RegistryStageInfo:
    """Get stage info by name. Alias for get() to satisfy StageDataProvider."""
    return self.get(name)

def ensure_fingerprint(self, stage_name: str) -> dict[str, str]:
    """Compute/return cached code fingerprint for a stage."""
    return self._registry.ensure_fingerprint(stage_name)
```

**Step 2: Update CLI callers to pass pipeline as provider**

In `src/pivot/cli/run.py` (around line 100-107):

```python
pipeline = cli_decorators.get_pipeline_from_context()

app = tui_run.PivotApp(
    stage_names=stages_list,
    tui_log=tui_log,
    cancel_event=cancel_event,
    stage_data_provider=pipeline,
)
```

In `src/pivot/cli/repro.py` (around line 381):

```python
app = tui_run.PivotApp(
    stage_names=display_order,
    tui_log=tui_log,
    watch_mode=True,
    no_commit=no_commit,
    serve=serve,
    stage_data_provider=pipeline,
)
```

(And the other `PivotApp()` call in repro.py around line 614.)

**Step 3: Run basedpyright on all modified files**

Run: `uv run basedpyright src/pivot/pipeline/pipeline.py src/pivot/cli/run.py src/pivot/cli/repro.py`
Expected: 0 errors

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -n auto --tb=short -q`
Expected: All pass

---

### Task 8: Verify zero `cli_helpers` imports in TUI and run quality checks

**Step 1: Verify no TUI-to-CLI coupling remains**

Run: `grep -r "from pivot.cli" src/pivot/tui/`
Expected: No output

Run: `grep -r "cli_helpers" src/pivot/tui/`
Expected: No output

**Step 2: Full quality checks**

Run: `uv run ruff format .`
Run: `uv run ruff check .`
Run: `uv run basedpyright`
Expected: 0 errors, 0 warnings

**Step 3: Full test suite**

Run: `uv run pytest tests/ -n auto --tb=short -q`
Expected: All pass
