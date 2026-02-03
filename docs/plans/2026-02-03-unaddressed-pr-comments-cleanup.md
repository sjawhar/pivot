# Unaddressed PR Comments Cleanup - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address all 23 unaddressed review comments from PRs merged Jan 30 - Feb 3, 2026 (tracked in issue #340)

**Architecture:** This is a cleanup PR addressing code quality, security, and consistency issues across multiple files. Changes are independent and can be committed in logical groups.

**Tech Stack:** Python 3.13, Click CLI, Textual TUI, semantic-release, GitHub Actions

---

## Task 1: Fix `contextlib` Import Location (Low - #11)

**Files:**
- Modify: `src/pivot/cli/data.py:122-127`

**Step 1: Read current state**

Run: `head -130 src/pivot/cli/data.py | tail -20`

**Step 2: Move import to module level**

The import `import contextlib` is inside the `finally` block. Move it to the top of the file with other imports.

```python
# At top of file, add to existing imports:
import contextlib
```

**Step 3: Remove inline import**

Remove line 123: `import contextlib`

**Step 4: Run tests**

Run: `uv run pytest tests/cli/test_data.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "fix: move contextlib import to module level in data.py"
```

---

## Task 2: Add Stack Trace to Debug Log (Low - #12)

**Files:**
- Modify: `src/pivot/storage/lock.py:421`

**Step 1: Update log call with exc_info**

Change:
```python
logger.debug(f"Lock takeover failed for {sentinel}: {e}")
```

To:
```python
logger.debug("Lock takeover failed for %s", sentinel, exc_info=True)
```

**Step 2: Run tests**

Run: `uv run pytest tests/storage/test_lock.py -v`
Expected: PASS

**Step 3: Commit**

```bash
jj describe -m "fix: preserve stack trace in lock takeover debug log"
```

---

## Task 3: Consistent Author Name in Plugin Files (Medium - #8)

**Files:**
- Modify: `.claude-plugin/marketplace.json:5`
- Modify: `.claude-plugin/plugin.json:6`

**Step 1: Standardize to "Sami Jawhar"**

In `plugin.json`, change:
```json
"author": {
  "name": "Sami Alabed"
}
```

To:
```json
"author": {
  "name": "Sami Jawhar"
}
```

**Step 2: Run quality checks**

Run: `jq . .claude-plugin/plugin.json && jq . .claude-plugin/marketplace.json`
Expected: Valid JSON output

**Step 3: Commit**

```bash
jj describe -m "fix: use consistent author name in plugin files"
```

---

## Task 4: Remove CSS Property Duplication (Low - #14)

**Files:**
- Modify: `src/pivot/tui/widgets/footer.py:49-56`

**Step 1: Review CSS duplication**

`pivot.tcss` already defines `#pivot-footer` with `dock`, `height`, `background`, `padding`.
`PivotFooter.DEFAULT_CSS` duplicates `dock`, `height`, `background`.

**Step 2: Simplify DEFAULT_CSS**

Change:
```python
DEFAULT_CSS: ClassVar[str] = """
PivotFooter {
    dock: bottom;
    height: 1;
    background: $surface;
    color: $text-muted;
}
"""
```

To:
```python
DEFAULT_CSS: ClassVar[str] = """
PivotFooter {
    color: $text-muted;
}
"""
```

**Step 3: Run TUI tests**

Run: `uv run pytest tests/tui/ -v -k footer`
Expected: PASS

**Step 4: Commit**

```bash
jj describe -m "fix: remove CSS property duplication in PivotFooter"
```

---

## Task 5: Pin semantic-release Version (Medium - #6)

**Files:**
- Modify: `.github/workflows/release.yaml:58-59`

**Step 1: Check latest stable version**

Run: `pip index versions python-semantic-release 2>/dev/null | head -5 || echo "Check pypi.org for latest version"`

**Step 2: Pin to specific version**

Change:
```yaml
uvx --from="python-semantic-release@10" semantic-release version || exit $?
uvx --from="python-semantic-release@10" semantic-release publish || exit $?
```

To (using a specific version like 10.0.2):
```yaml
uvx --from="python-semantic-release==10.0.2" semantic-release version || exit $?
uvx --from="python-semantic-release==10.0.2" semantic-release publish || exit $?
```

**Step 3: Commit**

```bash
jj describe -m "ci: pin python-semantic-release to exact version"
```

---

## Task 6: Add `commit_assets` Configuration (Medium - #5)

**Files:**
- Modify: `pyproject.toml:188-190`

**Step 1: Add commit_assets**

After line 190:
```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
build_command = "uv lock --upgrade-package pivot && git add uv.lock && uv build"
commit_assets = ["uv.lock"]
```

**Step 2: Run quality checks**

Run: `uv run ruff check pyproject.toml`
Expected: No errors

**Step 3: Commit**

```bash
jj describe -m "ci: add commit_assets for uv.lock in semantic-release"
```

---

## Task 7: Rename EventSink to BroadcastEventSink (Low - #22)

**Files:**
- Modify: `src/pivot/engine/agent_rpc.py:474`

**Step 1: Rename the class**

Change:
```python
class EventSink:
    """Async sink that broadcasts events to connected agents.
```

To:
```python
class BroadcastEventSink:
    """Async sink that broadcasts events to connected agents.
```

**Step 2: Update all usages in the same file**

Search for `EventSink` usages and update to `BroadcastEventSink`.

**Step 3: Run tests**

Run: `uv run pytest tests/engine/test_agent_rpc.py -v`
Expected: PASS

**Step 4: Commit**

```bash
jj describe -m "refactor: rename EventSink to BroadcastEventSink to avoid Protocol collision"
```

---

## Task 8: Add TabbedContent Tab Activated Handler (Medium - #4)

**Files:**
- Modify: `src/pivot/tui/run.py`

**Step 1: Add handler method**

Add after the `action_next_tab` method (around line 870):
```python
def on_tabbed_content_tab_activated(
    self, event: textual.widgets.TabbedContent.TabActivated
) -> None:
    """Update footer context when tab is clicked."""
    self._update_footer_context()
```

**Step 2: Run TUI tests**

Run: `uv run pytest tests/tui/ -v`
Expected: PASS

**Step 3: Commit**

```bash
jj describe -m "fix: update footer context on tab mouse click"
```

---

## Task 9: Use Lazy Imports in _run_common.py (High - #3)

**Files:**
- Modify: `src/pivot/cli/_run_common.py:15-16, 225-258`

**Step 1: Move imports to TYPE_CHECKING block**

Change line 15-16:
```python
from pivot.engine import engine, sinks
```

To add in TYPE_CHECKING block:
```python
if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator

    import networkx as nx

    from pivot.engine import engine as engine_mod
    from pivot.engine import sinks as sinks_mod
    from pivot.engine.types import OutputEvent, StageCompleted
    from pivot.executor import ExecutionSummary
    from pivot.pipeline.pipeline import Pipeline
    from pivot.tui.run import MessagePoster
```

**Step 2: Update configure_result_collector**

```python
def configure_result_collector(eng: engine_mod.Engine) -> sinks_mod.ResultCollectorSink:
    """Add ResultCollectorSink to collect execution results."""
    from pivot.engine import sinks

    result_sink = sinks.ResultCollectorSink()
    eng.add_sink(result_sink)
    return result_sink
```

**Step 3: Update configure_output_sink**

```python
def configure_output_sink(
    eng: engine_mod.Engine,
    ...
) -> None:
    """Configure output sinks based on display mode."""
    from pivot.engine import sinks
    import rich.console
    ...
```

**Step 4: Run CLI tests**

Run: `uv run pytest tests/cli/ -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "perf: lazy import engine modules in _run_common.py"
```

---

## Task 10: Extract Atomic Write Helper for Loaders (Medium - #10)

**Files:**
- Modify: `src/pivot/loaders.py:175-268`

**Step 1: Create internal helper**

Add near the top of the file after imports:
```python
def _atomic_write_text(
    path: pathlib.Path,
    content_writer: Callable[[typing.TextIO], None],
    suffix: str,
) -> None:
    """Atomically write text content using temp file + rename.

    Args:
        path: Target file path.
        content_writer: Function that writes content to file handle.
        suffix: Temp file suffix (e.g., ".txt.tmp").
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=suffix)
    try:
        with os.fdopen(fd, "w") as f:
            content_writer(f)
        os.rename(tmp_path_str, path)
    except Exception:
        tmp_path = pathlib.Path(tmp_path_str)
        if tmp_path.exists():
            tmp_path.unlink()
        raise
```

**Step 2: Update Text.save**

```python
@override
def save(self, data: str, path: pathlib.Path) -> None:
    if not isinstance(data, str):
        raise TypeError(f"Text save expects str, got {type(data).__name__}")
    _atomic_write_text(path, lambda f: f.write(data), ".txt.tmp")
```

**Step 3: Update JSONL.save**

```python
@override
def save(self, data: list[dict[str, Any]], path: pathlib.Path) -> None:
    if not isinstance(data, list):
        raise TypeError(f"JSONL save expects list, got {type(data).__name__}")

    def write_jsonl(f: typing.TextIO) -> None:
        for item in data:
            f.write(json.dumps(item) + "\n")

    _atomic_write_text(path, write_jsonl, ".jsonl.tmp")
```

**Step 4: Update DataFrameJSONL.save**

```python
@override
def save(self, data: pandas.DataFrame, path: pathlib.Path) -> None:
    if not isinstance(data, pandas.DataFrame):
        raise TypeError(f"DataFrameJSONL save expects DataFrame, got {type(data).__name__}")
    _atomic_write_text(path, lambda f: data.to_json(f, lines=True, orient="records"), ".jsonl.tmp")
```

**Step 5: Run loader tests**

Run: `uv run pytest tests/test_loaders.py -v`
Expected: PASS

**Step 6: Commit**

```bash
jj describe -m "refactor: extract atomic write helper for text loaders"
```

---

## Task 11: Remove Redundant Test Imports (Low - #13, #15, #16, #19, #23)

**Files:**
- Modify: `tests/test_dep_injection.py:808` (remove `import json`)
- Modify: `tests/core/test_discovery.py:425` (use `Path` instead of `pathlib.Path`)
- Modify: `tests/engine/test_agent_rpc.py` (remove redundant imports at lines 960, 991, 1031, 1108)
- Modify: `tests/engine/test_rpc_queries.py:201` (remove redundant `import json`)
- Modify: `tests/engine/test_engine.py` (move `import networkx as nx` to module level)
- Modify: `tests/fingerprint/test_fingerprint.py:535` (remove `import math`)

**Step 1: Search for all redundant imports**

Run: `uv run ruff check tests/ --select F811 2>/dev/null || echo "No issues or ruff handles differently"`

**Step 2: Fix each file**

For each file listed, remove the redundant inline import.

**Step 3: Run affected tests**

Run: `uv run pytest tests/test_dep_injection.py tests/core/test_discovery.py tests/engine/test_agent_rpc.py tests/engine/test_rpc_queries.py tests/engine/test_engine.py tests/fingerprint/test_fingerprint.py -v`
Expected: PASS

**Step 4: Commit**

```bash
jj describe -m "fix: remove redundant imports from test files"
```

---

## Task 12: Use contextlib.closing for Socket (Low - #18)

**Files:**
- Modify: `tests/engine/test_rpc_queries.py:228`

**Step 1: Wrap socket in contextlib.closing**

Find the socket usage and wrap it:
```python
import contextlib
...
with contextlib.closing(sock) as s:
    # socket operations
```

**Step 2: Run tests**

Run: `uv run pytest tests/engine/test_rpc_queries.py -v`
Expected: PASS

**Step 3: Commit**

```bash
jj describe -m "fix: use contextlib.closing for socket in test"
```

---

## Task 13: Final Quality Checks

**Step 1: Run full quality suite**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: No errors

**Step 2: Run full test suite**

Run: `uv run pytest tests/ -n auto`
Expected: All tests PASS

**Step 3: Push changes**

```bash
jj git push --named=pr-review-cleanup=@
```

---

## Deferred Items (Not In This PR)

The following items require more significant changes or discussion:

1. **#1 - Refactor `_orchestrate_execution`** (~280 lines): Major refactor, needs separate PR
2. **#2 - Tests accessing private attributes**: Would require adding public properties, architectural decision
3. **#7 - Security validation for `state_dir`**: Needs design discussion for path traversal protection
4. **#9 - Watch mode `--keep-going` flag**: Behavioral change, needs user-facing documentation
5. **#17 - Missing explanatory comment**: File doesn't exist (helpers.py), may have been refactored
6. **#20 - PEP8 naming (CSV→Csv)**: Breaking change, defer to future release
7. **#21 - Runtime validation for Dep/Reader**: Needs design discussion

These are tracked in issue #340 and can be addressed in follow-up PRs.

---

## Summary

| Task | Priority | Description |
|------|----------|-------------|
| 1 | Low | Move `contextlib` import to module level |
| 2 | Low | Add `exc_info=True` to debug log |
| 3 | Medium | Consistent author name in plugin files |
| 4 | Low | Remove CSS property duplication |
| 5 | Medium | Pin semantic-release version |
| 6 | Medium | Add `commit_assets` configuration |
| 7 | Low | Rename EventSink to BroadcastEventSink |
| 8 | Medium | Add tab click handler for footer context |
| 9 | High | Lazy imports in _run_common.py |
| 10 | Medium | Extract atomic write helper for loaders |
| 11 | Low | Remove redundant test imports |
| 12 | Low | Use contextlib.closing for socket |
| 13 | - | Final quality checks |
