# Purge Legacy SQLite / state.db / DVC Compat / Backcompat Residue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete every remaining reference to the pre-LMDB SQLite era, every backward-compatibility shim kept "to be gradually updated", every thin wrapper or deprecated-but-kept API, and every piece of legacy-framed code inside Pivot itself. Includes removing both directions of the DVC bridge (`pivot export` / `dvc_compat.py`) per user direction.

**Architecture:** Pure deletions, type-signature tightening, and mechanical test surgery. No new runtime features. Changes span source code (`packages/pivot/src/pivot`), tests (`packages/pivot/tests`, `packages/pivot-tui/tests`), AGENTS.md prose, and user-facing documentation for the removed `pivot export` command. Multiple logical jj revisions stacked on top of the `fix/remove-state-db-sentinel` branch (PR #455).

**Tech Stack:** Python 3.13+, pytest, ruff, basedpyright, jj (NOT git).

**Out of scope (user-confirmed):**
- `pivot import-dvc` command and `dvc_import.py` — this is a real onboarding feature, not legacy.
- `docs/solutions/*` — institutional knowledge, intentionally preserved per `/home/sami/pivot/review/AGENTS.md`.
- Adding new deprecation shims, warnings, or migration helpers. Pre-alpha project: breaking changes are acceptable and desired.

---

## Context for a Fresh Engineer

1. **Pivot uses jj, not git.** Every "commit" step below means `jj describe -m "…" && jj new`. Never use `git add`, `git commit`, `git push`, `git rebase`. See `/home/sami/pivot/review/AGENTS.md` (root) and `~/.claude/CLAUDE.md` for the full rule set. `jj git push` is the push command.
2. **Run tests with `uv`.** The canonical quality-check command block is:
   ```
   uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
   uv run ruff format . && uv run ruff check . && uv run basedpyright
   ```
   All paths in this plan are relative to repo root `/home/sami/pivot/review`.
3. **Why this plan exists:** A SQLite-backed state cache was replaced by LMDB in PR #45 (Jan 2026). That migration kept the old `state.db` filename as a dead sentinel. PR #245 then accidentally gated the O(1) skip-detection fast path on a phantom `state_db_path.exists()` check — which always returned False, silently disabling it. PR #455 (the branch this plan stacks onto, `fix/remove-state-db-sentinel`) removes the phantom. A follow-up audit found *the same anti-pattern* (deprecated-but-kept code, thin wrappers, optional-None defaults hiding real bugs, re-exports for tests) in seven other places, plus an entire vestigial DVC-export code path. This plan finishes every one of them.
4. **What makes this safe:** This is a pre-alpha project. `AGENTS.md` at the repo root explicitly says "breaking changes acceptable, no migration code or compatibility shims needed". Old lock files, stale state databases, and orphaned CLI flags should fail loudly (or disappear), not be silently tolerated.
5. **Useful facts:**
   - `StageLock.read()` already returns `None` when parsing fails, and every caller treats `None` as "no previous run → rerun". That is how Task 2 "rejects" old lock files — make `is_lock_data` reject them, and the downstream behavior takes care of itself.
   - `packages/pivot/tests/engine/test_graph.py` has 39 instances of `@pytest.mark.usefixtures("clean_registry")`. Removing them is mechanical.
   - `packages/pivot/tests/AGENTS.md:58` claims `clean_registry` is an autouse fixture. It isn't — only `reset_pivot_state` is `autouse=True`. Fix that misclaim as part of the prose scrub.
   - The CLI `pivot export` command is declared in `packages/pivot/src/pivot/cli/__init__.py:34` and in `COMMAND_CATEGORIES["Other"]` on line 16. Both entries go away along with the module.
6. **Stop-gate rule:** If a "Step 1: Confirm preconditions" step finds something that contradicts this plan (unexpected caller, unexpected fixture reference), STOP and ask the user rather than guessing. The PR #455 author is watching, don't introduce surprises.

---

## File Structure

Files modified or deleted by this plan:

| Path | Role | Action |
|---|---|---|
| `packages/pivot/src/pivot/discovery.py.orig` | Refactor backup (396 lines) | Delete |
| `packages/pivot/src/pivot/dvc_compat.py` | DVC import+export bridge (389 lines) | **Delete entirely** |
| `packages/pivot/src/pivot/cli/export.py` | `pivot export` CLI (42 lines) | **Delete** |
| `packages/pivot/src/pivot/cli/__init__.py` | CLI registry | Modify (remove `export` and `import-dvc`'s sibling exports) |
| `packages/pivot/src/pivot/exceptions.py` | Exception hierarchy | Modify (delete `DVCCompatError`, `ExportError`; migrate `DVCImportError` to `PivotError`) |
| `packages/pivot/src/pivot/dvc_import.py` | Real `pivot import-dvc` codepath | Modify (rewire `DVCImportError` parent class) |
| `packages/pivot/tests/compat/test_dvc_compat.py` | Tests for deleted module (465 lines) | **Delete entirely** |
| `packages/pivot/tests/cli/test_cli_export.py` | Tests for deleted CLI (248 lines) | **Delete entirely** |
| `packages/pivot/tests/compat/` directory | Should become empty | Delete |
| `docs/gen_ref_pages.py` | MkDocs autogen | Modify (drop `pivot.dvc_compat` entry) |
| `docs/cli/index.md` | User-facing CLI doc | Modify (remove `pivot export` section) |
| `docs/comparison.md` | Comparison doc | Modify (remove `pivot export` example) |
| `docs/migrating-from-dvc.md` | DVC-migration guide | Modify (remove `pivot export` sections and flag) |
| `docs/getting-started/installation.md` | Install guide | Modify (drop "For `pivot export`" section) |
| `packages/pivot-tui/src/pivot_tui/widgets/panels.py:40` | TUI panel constructor | Modify (remove `stage_data_provider` dead param) |
| `packages/pivot/src/pivot/storage/lock.py:56-88` | `is_lock_data` | Modify (reject `dep_generations`) |
| `packages/pivot/src/pivot/storage/lock.py:210-221` | `StageLock.is_changed` (3-line wrapper) | **Delete** — inline into test helper |
| `packages/pivot/src/pivot/storage/lock.py:223-245` | `StageLock.is_changed_with_lock_data` | Modify (require `out_paths`) |
| `packages/pivot/src/pivot/storage/AGENTS.md` | Storage guidelines | Modify (remove legacy paragraph) |
| `packages/pivot/src/pivot/explain.py:29-32` | Explain re-exports | **Delete** |
| `packages/pivot/tests/conftest.py:188-196` | Root test conftest | Modify (delete `clean_registry`) |
| `packages/pivot-tui/tests/conftest.py:118-126` | TUI test conftest | Modify (delete `clean_registry`) |
| `packages/pivot/tests/AGENTS.md:58` | Testing guidelines | Modify (fix autouse claim; drop `clean_registry`) |
| `packages/pivot/tests/storage/test_lock.py` | Lock tests | Modify (add `_is_changed` helper; swap `ignored`→`rejected`; delete `out_paths=None` test; rewire 10 call sites) |
| `packages/pivot/tests/core/test_explain.py` | Explain tests | Modify (9 call-site renames) |
| `packages/pivot/tests/engine/test_graph.py` | Graph tests | Modify (delete 39 `usefixtures` lines) |
| `packages/pivot/tests/test_run_cache_lock_update.py` | Run-cache tests | Modify (rename `state_db_path` → `state_dir`) |
| `packages/pivot/tests/execution/test_execution_modes.py` | Execution-mode tests | Modify (rename `state_db_path` → `state_dir`) |

No new source modules. No net-new runtime code.

---

## Task 1 — Delete `discovery.py.orig`

**Files:**
- Delete: `packages/pivot/src/pivot/discovery.py.orig`

- [ ] **Step 1: Confirm the file is unreferenced**

Run:
```
rg -n 'discovery\.py\.orig|import.*discovery.*orig' .
```
Expected output: empty (zero matches).

- [ ] **Step 2: Delete the file**

```
rm packages/pivot/src/pivot/discovery.py.orig
```

- [ ] **Step 3: Verify deletion**

```
ls packages/pivot/src/pivot/discovery.py.orig
```
Expected: `ls: cannot access '…/discovery.py.orig': No such file or directory`.

- [ ] **Step 4: Run the full test suite**

```
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
```
Expected: all green (same pass/fail baseline as before).

- [ ] **Step 5: Commit**

```
jj describe -m "chore: remove discovery.py.orig refactor leftover"
jj new
```

---

## Task 2 — Delete `pivot export` + `dvc_compat.py` entirely

This removes the entire DVC-export bridge (both "Pivot → DVC" export and the parallel vestigial "DVC → Pivot" path in `dvc_compat.py`). The *real* DVC import via `dvc_import.py` / `pivot import-dvc` stays intact.

**Files:**
- Delete: `packages/pivot/src/pivot/cli/export.py`
- Delete: `packages/pivot/src/pivot/dvc_compat.py`
- Delete: `packages/pivot/tests/cli/test_cli_export.py`
- Delete: `packages/pivot/tests/compat/test_dvc_compat.py` and the now-empty `packages/pivot/tests/compat/` directory
- Modify: `packages/pivot/src/pivot/cli/__init__.py` (remove 2 references)
- Modify: `packages/pivot/src/pivot/exceptions.py` (delete `DVCCompatError`, `ExportError`; reparent `DVCImportError`)
- Modify: `packages/pivot/src/pivot/dvc_import.py` if its `DVCImportError` references need updating (spoiler: no, it uses `exceptions.DVCImportError` by attribute access, so no source change)
- Modify: `docs/gen_ref_pages.py` (drop `pivot.dvc_compat` entry)
- Modify: `docs/cli/index.md`, `docs/comparison.md`, `docs/migrating-from-dvc.md`, `docs/getting-started/installation.md` (remove `pivot export` sections)

- [ ] **Step 1: Confirm no unexpected consumers**

```
rg -n 'from pivot import dvc_compat|from pivot.dvc_compat|import dvc_compat|dvc_compat\.' packages/
rg -n 'from pivot.cli.export|pivot\.cli\.export|pivot_export' packages/
rg -n '"export"|\'export\'' packages/pivot/src/pivot/cli/__init__.py
```
Expected consumers:
- `packages/pivot/src/pivot/cli/export.py:24,40` — `from pivot import dvc_compat` and `dvc_compat.export_dvc_yaml(...)`. That file is being deleted.
- `packages/pivot/tests/compat/test_dvc_compat.py` — entire file being deleted.
- `packages/pivot/src/pivot/cli/__init__.py:16,34` — the command registration; being edited.
- `docs/gen_ref_pages.py:18` — MkDocs autogen; being edited.

If any consumer appears outside this list, **STOP** and ask the user.

- [ ] **Step 2: Delete `cli/export.py`**

```
rm packages/pivot/src/pivot/cli/export.py
```

- [ ] **Step 3: Delete `dvc_compat.py`**

```
rm packages/pivot/src/pivot/dvc_compat.py
```

- [ ] **Step 4: Delete the tests and the empty `compat/` dir**

```
rm packages/pivot/tests/cli/test_cli_export.py
rm packages/pivot/tests/compat/test_dvc_compat.py
rmdir packages/pivot/tests/compat
```

If `rmdir` fails because the directory has other files, list them and stop:
```
ls packages/pivot/tests/compat/
```
Expected: the directory should only contain `test_dvc_compat.py` (verified at plan-writing time). If there's an `__init__.py` or other file, delete it too only after confirming it's not imported elsewhere.

- [ ] **Step 5: Remove the CLI registry entries**

Edit `packages/pivot/src/pivot/cli/__init__.py`:

**5a.** In `COMMAND_CATEGORIES["Other"]` (currently lines 14–24), delete the string `"export",` (line 16). After edit:

```python
    "Other": [
        "init",
        "import-dvc",
        "config",
        "completion",
        "schema",
        "check-ignore",
        "doctor",
        "fingerprint",
    ],
```

**5b.** In `_LAZY_COMMANDS` (the dict starting at line 28), delete the `"export":` entry (line 34):

```python
    "export": ("pivot.cli.export", "export", "Export pipeline to DVC YAML format."),
```

Delete that entire line including its trailing comma and newline.

- [ ] **Step 6: Prune the exception hierarchy**

Edit `packages/pivot/src/pivot/exceptions.py` lines 157–166. Current content:

```python
class DVCCompatError(PivotError):
    """Base class for DVC compatibility errors."""


class ExportError(DVCCompatError):
    """Raised when stage export to DVC format fails."""


class DVCImportError(DVCCompatError):
    """Raised when dvc.yaml import fails."""
```

Replace with:

```python
class DVCImportError(PivotError):
    """Raised when dvc.yaml import fails."""
```

Rationale: `DVCImportError` is used by the live `dvc_import.py` codepath. `DVCCompatError` and `ExportError` only existed for the deleted `dvc_compat.py` module.

- [ ] **Step 7: Drop the MkDocs autogen entry**

Edit `docs/gen_ref_pages.py` line 18. Delete:

```python
    ("pivot.dvc_compat", "DVC Compatibility"),
```

(That's the last entry in the `PUBLIC_MODULES` list. Preserve the closing `]` on the next line.)

- [ ] **Step 8: Scrub `pivot export` from user-facing docs**

**8a.** In `docs/cli/index.md`, find and delete the entire `### \`pivot export\`` section. Locate the heading at line 254; delete from that heading through (and including) the end of the code fence that follows its `pivot export [STAGES...] [OPTIONS]` example (and any subsequent paragraphs that are part of the same subsection, up to the next `### ` heading). When complete, running `rg -n 'pivot export' docs/cli/index.md` should return zero matches.

**8b.** In `docs/comparison.md`, find line 103 which contains:
```
pivot export > dvc.yaml
```
Delete that line. If it is part of a before/after comparison table or block, also delete the surrounding explanation lines that reference it. After edit, `rg -n 'pivot export' docs/comparison.md` must return zero matches.

**8c.** In `docs/migrating-from-dvc.md`, find the two `pivot export` usage sections (around lines 281 and 296/306). Delete every `pivot export` invocation and the surrounding paragraphs that describe how to use them. This may require deleting an entire subsection ("Exporting to DVC", "Running DVC against Pivot's output", or similar heading). After edit, `rg -n 'pivot export' docs/migrating-from-dvc.md` must return zero matches.

**8d.** In `docs/getting-started/installation.md`, find line 26 which starts "For `pivot export` to generate DVC-compatible YAML:". Delete that line and any subsequent lines describing optional DVC install for export. If the section becomes empty, delete its heading too.

- [ ] **Step 9: Verify tests still pass and no orphans remain**

```
rg -n 'dvc_compat|DVCCompatError|ExportError|pivot export' packages/ docs/
```
Expected: zero matches (ignore this plan file; if matches appear only in `docs/superpowers/plans/` or `docs/solutions/`, that's fine).

```
rg -n 'pivot\.cli\.export|cli\.export' packages/
```
Expected: empty.

```
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
```
Expected: all green. Test counts will drop significantly (deleted 465 + 248 lines of tests).

```
uv run basedpyright packages/pivot/src packages/pivot-tui/src
```
Expected: no new errors. Pre-existing errors unchanged.

- [ ] **Step 10: Commit**

```
jj describe -m "feat!: remove pivot export command and dvc_compat module

DVC-export bridge was a one-way migration helper. Deleting it removes
389+42 lines of production code, 465+248 lines of tests, 2 unused
exception classes, and 1 ~400-line vestigial DVC-import path in
dvc_compat.py that duplicated dvc_import.py.

BREAKING: pivot export command removed. Use dvc_import.py / pivot
import-dvc for the opposite direction if needed."
jj new
```

---

## Task 3 — Delete deprecated `stage_data_provider` TUI parameter

**Files:**
- Modify: `packages/pivot-tui/src/pivot_tui/widgets/panels.py:40`

- [ ] **Step 1: Confirm no caller passes `stage_data_provider`**

```
rg -n 'stage_data_provider' packages/ docs/
```
Expected: only the definition line in `panels.py:40`. The 3 caller sites (`run.py:391`, `tests/test_history.py:170,177`, `tests/test_run.py:449`) all instantiate `TabbedDetailPanel()` without this argument.

- [ ] **Step 2: Remove the parameter**

Edit `packages/pivot-tui/src/pivot_tui/widgets/panels.py`. The current `__init__` signature (lines 35–41):

```python
    def __init__(
        self,
        *,
        id: str | None = None,
        classes: str | None = None,
        stage_data_provider: object = None,  # Deprecated: accepted but ignored
    ) -> None:
```

Delete the `stage_data_provider` line so the signature becomes:

```python
    def __init__(
        self,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
```

- [ ] **Step 3: Verify**

```
rg -n 'stage_data_provider' packages/
```
Expected: empty.

```
uv run pytest packages/pivot-tui/tests -n auto
```
Expected: all green.

- [ ] **Step 4: Commit**

```
jj describe -m "refactor(tui): remove deprecated stage_data_provider parameter

The parameter was accepted but ignored. No caller passes it."
jj new
```

---

## Task 4 — Reject lock files with removed `dep_generations` field

**Files:**
- Test: `packages/pivot/tests/storage/test_lock.py:509-520` (replace existing test)
- Modify: `packages/pivot/src/pivot/storage/lock.py:56-88` (body of `is_lock_data`)

- [ ] **Step 1: Replace the obsolete "ignored" test with a "rejected" test (RED)**

Edit `packages/pivot/tests/storage/test_lock.py` lines 509–520. The function currently named `test_read_lock_with_dep_generations_ignored` should be entirely replaced with:

```python
def test_read_lock_with_dep_generations_rejected(tmp_path: Path) -> None:
    """Lock files containing the removed dep_generations field are rejected.

    After the SQLite→LMDB migration, dep_generations lives only in StateDB.
    Stale lock files that still have this field are rejected so callers
    treat them as 'no previous run' and regenerate cleanly.
    """
    stage_lock = lock.StageLock("stale", tmp_path)
    stage_lock.path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.path.write_text(
        "code_manifest: {}\nparams: {}\ndeps: []\nouts: []\ndep_generations: {}\n"
    )

    result = stage_lock.read()

    assert result is None, "stale lock files with dep_generations must be rejected"
```

Leave `test_is_lock_data_accepts_missing_dep_generations` (lines 497–506) untouched — that one describes correct behavior (the field can be absent, which is true both before and after this change).

- [ ] **Step 2: Run the new test and watch it fail**

```
uv run pytest packages/pivot/tests/storage/test_lock.py::test_read_lock_with_dep_generations_rejected -v
```
Expected: FAIL. Current `is_lock_data` accepts extra keys, so `read()` returns a non-None `LockData`, and the assertion `result is None` fails.

- [ ] **Step 3: Tighten `is_lock_data` (GREEN)**

Edit `packages/pivot/src/pivot/storage/lock.py`. Current body starts at line 64. After the existing `_REQUIRED_LOCK_KEYS.issubset(typed_data.keys())` check (around line 69), and before the existing `# Reject null values for required keys` comment, insert:

```python
    # Reject the removed dep_generations field. Pre-alpha: lock files from the
    # SQLite-era schema must regenerate rather than be silently partially-read.
    if "dep_generations" in typed_data:
        return False
```

Do NOT remove the existing `# Require all required keys (allow extra keys for forward compatibility)` comment on the `_REQUIRED_LOCK_KEYS.issubset` line. That forward-compat stance remains legitimate for fields we *haven't* removed.

- [ ] **Step 4: Run the new test and watch it pass**

```
uv run pytest packages/pivot/tests/storage/test_lock.py::test_read_lock_with_dep_generations_rejected -v
```
Expected: PASS.

- [ ] **Step 5: Run the full `test_lock.py` suite**

```
uv run pytest packages/pivot/tests/storage/test_lock.py -v
```
Expected: all tests pass, including `test_is_lock_data_accepts_missing_dep_generations`.

- [ ] **Step 6: Commit**

```
jj describe -m "refactor(storage): reject lock files with removed dep_generations field"
jj new
```

---

## Task 5 — Require `out_paths` on both `is_changed` and `is_changed_with_lock_data` + inline `is_changed`

`StageLock.is_changed` is a 3-line wrapper around `is_changed_with_lock_data` (reads lock file, then calls the comparison). Only tests call it. Per user direction: inline it into the test file as a module-level helper. Both functions currently accept `out_paths: list[str] | None = None` — same failure-mode as the `state.db` guard (silent skip of a real check when None). Since the production path (`is_changed_with_lock_data`) is becoming required, and `is_changed` is going away, this task handles both in one go.

**Files:**
- Modify: `packages/pivot/src/pivot/storage/lock.py:205-244` (delete `is_changed`, tighten `is_changed_with_lock_data`)
- Modify: `packages/pivot/tests/storage/test_lock.py` (add `_is_changed` module helper; rewire 10 `is_changed(...)` call sites; delete `test_stage_unchanged_when_out_paths_none`)

- [ ] **Step 1: Confirm all production callers of `is_changed_with_lock_data` already pass `out_paths`**

```
rg -n 'is_changed_with_lock_data' packages/pivot/src
```
Expected hits:
- `packages/pivot/src/pivot/storage/lock.py:219` (internal call from `is_changed`, which we are deleting)
- `packages/pivot/src/pivot/storage/lock.py:223` (the definition)
- `packages/pivot/src/pivot/executor/commit.py:137` — must pass `out_paths=...`
- `packages/pivot/src/pivot/executor/worker.py:512` — must pass `out_paths=...`

Eyeball the last two to confirm they pass `out_paths` explicitly. If either passes `None` or omits the argument, **STOP** and ask.

```
rg -n 'stage_lock\.is_changed\b|\.is_changed\(' packages/pivot/src
```
Expected: only `packages/pivot/src/pivot/storage/lock.py:210` (the definition we're about to delete). If any source file calls `is_changed(...)` outside the lock module, the scope of this task needs to widen — **STOP** and ask.

- [ ] **Step 2: Add the test helper**

Edit `packages/pivot/tests/storage/test_lock.py`. At the top of the file, after the imports (and after any existing module-level type aliases / helper functions), add:

```python
def _is_changed(
    stage_lock: lock.StageLock,
    current_fingerprint: dict[str, str],
    current_params: dict[str, Any],
    dep_hashes: dict[str, HashInfo],
    out_paths: list[str] | None = None,
) -> tuple[bool, str]:
    """Test helper: read lock file then compare (mirrors the old StageLock.is_changed).

    Production code calls is_changed_with_lock_data directly with lock_data it
    already has; only tests want the read-then-compare convenience.
    """
    lock_data = stage_lock.read()
    return stage_lock.is_changed_with_lock_data(
        lock_data,
        current_fingerprint,
        current_params,
        dep_hashes,
        out_paths if out_paths is not None else [],
    )
```

Notes:
- The `out_paths if out_paths is not None else []` expression preserves the old "out_paths=None is a legitimate test shortcut" behavior for the handful of tests that want to skip out-path comparison. Passing `[]` effectively tells the comparison "no output paths declared" — consistent with stages that have no outputs.
- This helper lives in `test_lock.py` only. Do not put it in a shared `conftest.py` — per project AGENTS.md, helpers live with their consumers, and only this file calls `is_changed`.

- [ ] **Step 3: Rewire the 10 `is_changed` call sites**

In `packages/pivot/tests/storage/test_lock.py`, every `stage_lock.is_changed(...)` call becomes `_is_changed(stage_lock, ...)`. Concrete replacements (with their line numbers in the current file):

- Line 132: `stage_lock.is_changed(` → `_is_changed(stage_lock,`
- Line 160: `stage_lock.is_changed(fingerprint, params, dep_hashes)` → `_is_changed(stage_lock, fingerprint, params, dep_hashes)`
- Line 178: `stage_lock.is_changed(` → `_is_changed(stage_lock,`
- Line 200: `stage_lock.is_changed(` → `_is_changed(stage_lock,`
- Line 222: `stage_lock.is_changed(` → `_is_changed(stage_lock,`
- Line 248: `stage_lock.is_changed(` → `_is_changed(stage_lock,`
- Line 275: `stage_lock.is_changed(` → `_is_changed(stage_lock,`
- Line 301: `stage_lock.is_changed(` → `_is_changed(stage_lock,`
- Line 540: `stage_lock.is_changed(` → `_is_changed(stage_lock,`
- Line 558: `stage_lock.is_changed(` → `_is_changed(stage_lock,`

Each call retains its existing arguments; only the call-form changes from method-on-object to function-plus-object.

- [ ] **Step 4: Delete the obsolete `out_paths=None` test**

Delete the entire function `test_stage_unchanged_when_out_paths_none` (currently lines 693–714) from `packages/pivot/tests/storage/test_lock.py`. Start marker:

```python
def test_stage_unchanged_when_out_paths_none(tmp_path: Path) -> None:
```

End marker: the next blank line before the next `def test_...` or `def _...`.

Rationale: that test documents "when `out_paths=None` we skip the output-path comparison". After this task, `is_changed_with_lock_data` requires `out_paths`, and the `_is_changed` test helper defaults to `[]` instead of `None`, so the behavior being tested no longer exists.

- [ ] **Step 5: Run the test file (should still pass before the production change)**

```
uv run pytest packages/pivot/tests/storage/test_lock.py -v
```
Expected: all tests pass. (Production code still has `is_changed` defined; the `_is_changed` helper just calls through.)

- [ ] **Step 6: Delete `StageLock.is_changed` and tighten `is_changed_with_lock_data`**

Edit `packages/pivot/src/pivot/storage/lock.py`.

**6a.** Delete lines 210–222 entirely (the `is_changed` method definition):

```python
    def is_changed(
        self,
        current_fingerprint: dict[str, str],
        current_params: dict[str, Any],
        dep_hashes: dict[str, HashInfo],
        out_paths: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Check if stage needs re-run (reads lock file)."""
        lock_data = self.read()
        return self.is_changed_with_lock_data(
            lock_data, current_fingerprint, current_params, dep_hashes, out_paths
        )
```

**6b.** Change the signature of `is_changed_with_lock_data` at what is currently line 229. Replace:

```python
        out_paths: list[str] | None = None,
```

with:

```python
        out_paths: list[str],
```

**6c.** Remove the `if out_paths is not None:` guard inside the function body. The current code around lines 241–244:

```python
        if out_paths is not None:
            locked_out_paths = sorted(lock_data["output_hashes"].keys())
            if sorted(out_paths) != locked_out_paths:
                return True, "Output paths changed"
```

becomes:

```python
        locked_out_paths = sorted(lock_data["output_hashes"].keys())
        if sorted(out_paths) != locked_out_paths:
            return True, "Output paths changed"
```

- [ ] **Step 7: Run typecheck**

```
uv run basedpyright packages/pivot/src/pivot/storage/lock.py packages/pivot/src/pivot/executor/commit.py packages/pivot/src/pivot/executor/worker.py
```
Expected: no new errors. If a caller somewhere was silently relying on the default, basedpyright will flag it.

- [ ] **Step 8: Run the full test suite**

```
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
```
Expected: all green.

- [ ] **Step 9: Commit**

```
jj describe -m "refactor(storage): require out_paths, inline is_changed into test helper

StageLock.is_changed was a 3-line wrapper around is_changed_with_lock_data
used only by tests. Inlined as a test-only helper in test_lock.py.

Both functions had out_paths: list[str] | None = None defaults that
silently skipped output-path validation when None — the same failure
mode as the state.db guard. out_paths is now required."
jj new
```

---

## Task 6 — Delete `explain` backcompat re-exports

**Files:**
- Modify: `packages/pivot/src/pivot/explain.py:28-33`
- Modify: `packages/pivot/tests/core/test_explain.py` (9 call sites)

- [ ] **Step 1: Update the test imports and call sites**

Edit `packages/pivot/tests/core/test_explain.py`. Ensure `skip` is imported at the top of the file. Locate the existing line:

```python
from pivot import explain
```

(If it's part of a longer import list, the exact form may differ.) Make it:

```python
from pivot import explain, skip
```

keeping alphabetical order.

Then apply these 9 mechanical renames within the file:

- Line 112: `changes = explain.diff_code_manifests(old, new)` → `changes = skip.diff_code_manifests(old, new)`
- Line 127: `changes = explain.diff_code_manifests(old, new)` → `changes = skip.diff_code_manifests(old, new)`
- Line 140: `changes = explain.diff_code_manifests(manifest, manifest)` → `changes = skip.diff_code_manifests(manifest, manifest)`
- Line 175: `changes = explain.diff_params(old, new)` → `changes = skip.diff_params(old, new)`
- Line 190: `changes = explain.diff_params(old, new)` → `changes = skip.diff_params(old, new)`
- Line 201: `changes = explain.diff_params(params, params)` → `changes = skip.diff_params(params, params)`
- Line 252: `changes = explain.diff_dep_hashes(old, new)` → `changes = skip.diff_dep_hashes(old, new)`
- Line 277: `changes = explain.diff_dep_hashes(old, new)` → `changes = skip.diff_dep_hashes(old, new)`
- Line 289: `changes = explain.diff_dep_hashes(dep_hashes, dep_hashes)` → `changes = skip.diff_dep_hashes(dep_hashes, dep_hashes)`

Do NOT delete the `explain` import — other tests in this file still use `explain.get_stage_explanation`.

- [ ] **Step 2: Run the affected tests (should still pass; re-exports still exist)**

```
uv run pytest packages/pivot/tests/core/test_explain.py -v
```
Expected: all tests pass.

- [ ] **Step 3: Delete the re-exports from `explain.py`**

Edit `packages/pivot/src/pivot/explain.py`. Current lines 28–33:

```
28:<blank>
29:# Re-exports for backward compatibility (tests reference these)
30:diff_code_manifests = skip.diff_code_manifests
31:diff_params = skip.diff_params
32:diff_dep_hashes = skip.diff_dep_hashes
33:<blank>
```

Replace with a single blank line so the imports block and the next `def _find_tracked_ancestor` are separated by one blank line.

- [ ] **Step 4: Run the affected tests (now with the re-exports gone)**

```
uv run pytest packages/pivot/tests/core/test_explain.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Grep for stragglers**

```
rg -n 'explain\.diff_' packages/
rg -n '# Re-exports for backward' packages/
```
Expected: empty output from both.

- [ ] **Step 6: Run the full test suite**

```
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
```
Expected: all green.

- [ ] **Step 7: Commit**

```
jj describe -m "refactor(explain): delete backward-compat re-exports of skip.diff_*"
jj new
```

---

## Task 7 — Remove all `@pytest.mark.usefixtures("clean_registry")` decorators

Done *before* fixture deletion so test collection never breaks between steps.

**Files:**
- Modify: `packages/pivot/tests/engine/test_graph.py` (39 decorator lines)

- [ ] **Step 1: Enumerate every decorator occurrence**

```
rg -n '@pytest\.mark\.usefixtures\("clean_registry"\)' packages/
```
Expected: exactly 39 matches, all in `packages/pivot/tests/engine/test_graph.py`. If any other file appears, add it to the scope of Step 2.

- [ ] **Step 2: Delete each decorator line**

For each matching line in `packages/pivot/tests/engine/test_graph.py`, delete the entire line. Each occurrence is a self-contained line immediately above a test function; deleting it does not affect the function body or any other decorators. The line is always exactly:

```python
@pytest.mark.usefixtures("clean_registry")
```

Hint: use `Edit` with one `{op: "replace", pos: "<L>#<ID>", lines: null}` per line, applied bottom-up. The edit tool handles line-number drift automatically when edits are bundled.

- [ ] **Step 3: Grep to confirm every decorator is gone**

```
rg -n '@pytest\.mark\.usefixtures\("clean_registry"\)' packages/
```
Expected: empty output.

- [ ] **Step 4: Run the affected test file**

```
uv run pytest packages/pivot/tests/engine/test_graph.py -v
```
Expected: all tests pass. The fixture definition still exists (being deleted in Task 8), so this intermediate state is safe.

- [ ] **Step 5: Commit**

```
jj describe -m "test: remove obsolete clean_registry usefixtures decorators"
jj new
```

---

## Task 8 — Delete `clean_registry` fixture definitions

**Files:**
- Modify: `packages/pivot/tests/conftest.py:188-196`
- Modify: `packages/pivot-tui/tests/conftest.py:118-126`

- [ ] **Step 1: Confirm no decorators or params reference the fixture**

```
rg -n 'clean_registry' packages/
```
Expected: only 4 lines:
- `packages/pivot/tests/conftest.py:189` (fixture def)
- `packages/pivot/tests/conftest.py:194` (docstring text)
- `packages/pivot-tui/tests/conftest.py:119` (fixture def)
- `packages/pivot-tui/tests/conftest.py:124` (docstring text)
- `packages/pivot/tests/AGENTS.md:58` (prose — Task 9 fixes)

If any `usefixtures(...)` or test-function-parameter reference remains, **STOP** — Task 7 missed something; loop back.

- [ ] **Step 2: Delete from `packages/pivot/tests/conftest.py`**

Delete lines 188–196 inclusive:

```python
@pytest.fixture
def clean_registry() -> None:
    """No-op fixture for backwards compatibility.

    Previously cleared the global REGISTRY between tests.
    Now that REGISTRY is removed, this is kept for tests that still use
    @pytest.mark.usefixtures("clean_registry") - they can be gradually updated.
    """
    pass
```

Leave the surrounding fixtures (`test_registry` above, the `_PIVOT_LOGGERS` constant below) untouched.

- [ ] **Step 3: Delete from `packages/pivot-tui/tests/conftest.py`**

Delete lines 118–126 inclusive (identical content as in Step 2).

- [ ] **Step 4: Verify**

```
rg -n 'clean_registry' packages/
```
Expected: exactly one remaining hit, `packages/pivot/tests/AGENTS.md:58`. Task 9 fixes that.

- [ ] **Step 5: Run the full test suite**

```
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
```
Expected: all green. If any test errors with `ERROR ... fixture 'clean_registry' not found`, it means Task 7 missed a decorator — STOP and fix that first.

- [ ] **Step 6: Commit**

```
jj describe -m "test: delete no-op clean_registry fixture definitions"
jj new
```

---

## Task 9 — Scrub AGENTS.md prose

**Files:**
- Modify: `packages/pivot/src/pivot/storage/AGENTS.md:13-15`
- Modify: `packages/pivot/tests/AGENTS.md:56-58`

- [ ] **Step 1: Fix `storage/AGENTS.md`**

Edit `packages/pivot/src/pivot/storage/AGENTS.md` line 15. Current:

> Dependency generations (`{dep_path: generation}`) live **only in StateDB**, not in lock files. Workers compute them via `compute_dep_generation_map()` and return them in `DeferredWrites`. The coordinator applies them via `apply_deferred_writes()`. Old lock files that may have contained `dep_generations` are handled gracefully (field ignored on read).

Replace with:

> Dependency generations (`{dep_path: generation}`) live **only in StateDB**, not in lock files. Workers compute them via `compute_dep_generation_map()` and return them in `DeferredWrites`. The coordinator applies them via `apply_deferred_writes()`. Lock files that contain a `dep_generations` field are rejected at parse time (treated as "no previous run") so stale state regenerates cleanly.

- [ ] **Step 2: Fix `tests/AGENTS.md`**

Edit `packages/pivot/tests/AGENTS.md` line 58. Current:

```
`conftest.py` has autouse fixtures that reset state between tests: `clean_registry`, `reset_pivot_state`.
```

Replace with:

```
`conftest.py` has an autouse fixture that resets state between tests: `reset_pivot_state`.
```

(This fixes a doubly-wrong claim: `clean_registry` was never `autouse`, and now it's deleted.)

- [ ] **Step 3: Final prose sweep**

```
rg -ni 'backwards? compat|# Re-exports for backward|ignored on read|handled gracefully' packages/ docs/ 2>/dev/null | rg -v 'docs/solutions|docs/superpowers/plans/2026-04-18-burn-legacy-state-db.md'
```
Expected: empty output. If anything shows up, evaluate:
- Is it accurate documentation of current behavior? Leave.
- Is it a stale reference to something deleted? Delete.
- Is it "reserved for future use" hedging? Delete (YAGNI).

- [ ] **Step 4: Sanity check — run suite and linters**

```
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
uv run ruff format . && uv run ruff check .
```
Expected: all green. No code changed; this is just verifying no test failed due to a doc path being watched.

- [ ] **Step 5: Commit**

```
jj describe -m "docs: purge legacy/backcompat language from AGENTS.md"
jj new
```

---

## Task 10 — Rename `state_db_path` variables to `state_dir` in tests

Final cosmetic pass — the variable names still lie about what they hold.

**Files:**
- Modify: `packages/pivot/tests/test_run_cache_lock_update.py` (lines 19, 25, 100, 106, 129, 190, 195, 211, 315, 320, 322, 340, 342, 363, 366)
- Modify: `packages/pivot/tests/execution/test_execution_modes.py` (lines 212, 214, 285, 287)

- [ ] **Step 1: Rename in `test_run_cache_lock_update.py`**

Rename every occurrence of the local variable/parameter `state_db_path` to `state_dir`:

- Line 19: function parameter `state_db_path: pathlib.Path` → `state_dir: pathlib.Path` (inside `_apply_deferred_writes`)
- Line 25: `with state.StateDB(state_db_path) as db:` → `with state.StateDB(state_dir) as db:`
- Lines 100, 190, 315: `state_db_path = tmp_path / ".pivot"` → `state_dir = tmp_path / ".pivot"`
- Lines 106, 129, 195, 211, 320, 340, 363: `_apply_deferred_writes(..., state_db_path)` → `_apply_deferred_writes(..., state_dir)`
- Lines 322, 342, 366: `with state.StateDB(state_db_path) as db:` → `with state.StateDB(state_dir) as db:`

- [ ] **Step 2: Rename in `test_execution_modes.py`**

- Lines 212, 285: `state_db_path = worker_env.parent` → `state_dir = worker_env.parent`
- Lines 214, 287: `with state.StateDB(state_db_path) as db:` → `with state.StateDB(state_dir) as db:`

- [ ] **Step 3: Grep for stragglers**

```
rg -n 'state_db_path' packages/
```
Expected: empty output.

- [ ] **Step 4: Run the two affected test files**

```
uv run pytest packages/pivot/tests/test_run_cache_lock_update.py packages/pivot/tests/execution/test_execution_modes.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```
jj describe -m "test: rename state_db_path to state_dir for consistency with StateDB API"
jj new
```

---

## Task 11 — Final acceptance sweep

- [ ] **Step 1: Final greps (all must return empty)**

Each of the below, after excluding this plan file and `docs/solutions/`, must return empty:

```
rg -ni 'sqlite|state\.db\b|get_state_db_path|state_db_path' packages/ docs/ \
  | rg -v 'docs/solutions|docs/superpowers/plans/2026-04-18-burn-legacy-state-db.md'
```

```
rg -ni 'backwards? compat|# Re-exports for backward|ignored on read|handled gracefully' packages/ docs/ \
  | rg -v 'docs/solutions|docs/superpowers/plans/2026-04-18-burn-legacy-state-db.md'
```

```
rg -n 'clean_registry|explain\.diff_|dvc_compat|DVCCompatError|ExportError|stage_data_provider' packages/ \
  | rg -v 'docs/solutions|docs/superpowers/plans/2026-04-18-burn-legacy-state-db.md'
```

```
rg -n 'pivot export\b' packages/ docs/ \
  | rg -v 'docs/solutions|docs/superpowers/plans/2026-04-18-burn-legacy-state-db.md'
```

```
ls packages/pivot/src/pivot/discovery.py.orig 2>&1
```
Expected for last command: `ls: cannot access '…': No such file or directory`.

If any grep returns output, fix before proceeding.

- [ ] **Step 2: Full quality gate**

```
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
uv run ruff format . && uv run ruff check . && uv run basedpyright
```

Expected:
- pytest: all green. Test counts will be lower than baseline (Task 2 deletes 713 lines of tests; Task 4 swaps 1 test, Task 5 deletes 1 test; net: large decrease).
- ruff format: no changes needed.
- ruff check: 0 issues.
- basedpyright: same or lower error count than baseline (Task 2 removes several modules, so error counts may drop; definitely no new errors).

- [ ] **Step 3: Push to remote**

Only after all tasks have committed successfully and all acceptance criteria pass:

```
jj bookmark set fix/remove-state-db-sentinel
jj git push --bookmark fix/remove-state-db-sentinel
```

- [ ] **Step 4: Comment on PR #455**

Post a comment to the PR summarizing the stacked cleanup commits:

```
gh pr comment 455 --body "$(cat <<'EOF'
Stacked 10 cleanup commits on top of this PR that finish purging legacy / backcompat residue from the SQLite→LMDB migration:

1. Remove discovery.py.orig refactor leftover
2. Remove pivot export command and dvc_compat module (breaking: export command gone)
3. Remove deprecated stage_data_provider parameter from TUI
4. Reject lock files with removed dep_generations field
5. Require out_paths on StageLock, inline is_changed into test helper
6. Delete explain backward-compat re-exports and update 9 test call sites
7. Remove 39 @pytest.mark.usefixtures("clean_registry") decorators
8. Delete no-op clean_registry fixture definitions
9. Scrub legacy/backcompat language from AGENTS.md
10. Rename state_db_path variables to state_dir in tests

If this volume of follow-up feels out of scope for #455 vs its own PR, happy to split — let me know.
EOF
)"
```

Do NOT merge the PR yourself — wait for the original reviewer.

---

## Acceptance Criteria (for the whole plan)

Before declaring done:

1. All 11 tasks' per-task verification steps passed.
2. The Task 11 grep block returns entirely empty (modulo the plan file and `docs/solutions/`).
3. The Task 11 quality gate passes (`pytest`, `ruff`, `basedpyright`).
4. 10 commits stacked on top of the `fix/remove-state-db-sentinel` branch, each with a descriptive `jj` message.
5. No `--amend` was used. Each `jj describe` was followed by `jj new`.
6. The PR #455 author (`tbroadley`) and reviewer (`sjawhar`) are aware (via the PR comment) that the branch grew.

---

## Self-Review (applied by plan author)

### Spec coverage against user directives from brainstorming

Walked through the user's answers to confirm every decision has a task:

| User directive | Task that implements it |
|---|---|
| DVC import feature OUT of scope | Stated in Out of Scope header; Task 2 deletes `dvc_compat.py` but explicitly keeps `dvc_import.py` / `pivot import-dvc` |
| Burn `dep_generations` backcompat | Task 4 |
| Delete `explain` re-exports + fix tests | Task 6 |
| Delete no-op fixtures + all references | Tasks 7 + 8 (split to keep test collection green in between) |
| Delete `discovery.py.orig` | Task 1 |
| Scrub docs/AGENTS.md legacy prose everywhere except `docs/solutions/` | Task 9 + Task 11's final sweep |
| Plan first, execute second | This document; execution gated on user approval |
| Delete `pivot export` and `dvc_compat.py` (import + export both gone) | Task 2 |
| Delete `stage_data_provider` | Task 3 |
| Require `out_paths` on both `is_changed` AND `is_changed_with_lock_data` | Task 5 |
| Inline `StageLock.is_changed` as a test helper since tests are its only callers | Task 5 (via `_is_changed` helper in `test_lock.py`) |

No gaps.

### Placeholder scan

Searched the plan for the "No Placeholders" red flags:
- No "TBD", "TODO", "implement later", "fill in details" — checked.
- No "add appropriate error handling" / "handle edge cases" — no new code paths introduce branching. Checked.
- No "write tests for the above" without code — Task 4 shows the complete new test body; Task 5 shows the complete `_is_changed` helper. Checked.
- No "similar to Task N" — each task's code is inlined in full. Checked.
- No "what-not-how" steps — every code step has before/after blocks or explicit line numbers. Checked.
- References to unknown symbols — no. The plan only references `is_lock_data`, `is_changed`, `is_changed_with_lock_data`, `clean_registry`, `skip.diff_*`, `explain.get_stage_explanation`, `DVCImportError`, `PivotError`, `stage_data_provider`, `TabbedDetailPanel`, `dvc_compat.export_dvc_yaml` — all of which exist in the pre-plan codebase and are cross-referenced to concrete line numbers. Checked.

### Type / signature consistency

- Task 5 changes `out_paths: list[str] | None = None` → `out_paths: list[str]` on `is_changed_with_lock_data`. The `_is_changed` helper's signature keeps `out_paths: list[str] | None = None` as a test-only convenience; when `None` it passes `[]` through, so the signatures line up without leaking the None through to production.
- Task 2 deletes `DVCCompatError` and `ExportError`; `DVCImportError` is re-parented to `PivotError`. Verified that `dvc_import.py` uses `exceptions.DVCImportError` by attribute access — no import needs updating.
- Task 6 swaps `explain.diff_code_manifests` → `skip.diff_code_manifests`. `skip` module already exports these three names (they're what `explain` currently re-exports), so no new surface is added.
- Task 8 deletes the `clean_registry` fixture. Task 7 already deleted all references in Step 3's grep check. The intermediate state after Task 7 is safe because pytest is fine with an unused fixture; the intermediate state after Task 8 would be broken if Task 7 missed anything, but Task 8 Step 1 enforces that check.
- Task 3 deletes `stage_data_provider`. Only 3 caller sites (`run.py:391`, 2 tests) and none pass the argument (verified by grep). Safe.

### Behavioral consistency

- Task 4 makes `StageLock.read()` return `None` for lock files with `dep_generations`. Both `executor/worker.py` and `executor/commit.py` handle `None` → "no previous run → rerun". Verified.
- Task 5 forces all callers of `is_changed_with_lock_data` to pass `out_paths`. Both production callers already do; Task 5 Step 1 is a stop-gate to confirm before editing.
- Task 2 removes `DVCCompatError` and `ExportError` from the exception hierarchy. `DVCImportError` stays (reparented) because `dvc_import.py` still raises it. Test files (`test_cli_import_dvc.py`, `test_dvc_import.py`) catch `exceptions.DVCImportError` — that attribute access resolves to the new `DVCImportError(PivotError)` class, which is a strict superclass relationship so `isinstance` / `pytest.raises` checks still pass.

### Ordering hazards

- Task 7 must precede Task 8 (delete decorators before fixture they reference). Enforced by task ordering AND Task 8's Step 1 stop-gate.
- Task 2 must precede Task 9's doc scrub (otherwise the doc scrub needs a larger footprint; keeping them separate means the doc scrub in Task 9 only touches AGENTS.md while Task 2 handles its own doc updates for `pivot export`).
- Task 5 can run after Task 4 (independent — different methods, different tests). Order chosen is just file-order-readable.
- Tasks 1, 3, 6, 10 are independent; any order works.

### Risks

| Risk | Mitigation |
|---|---|
| Task 2's doc scrub (Step 8) is the fuzziest step — real prose editing, not line-number-exact | Stopping gate: run `rg -n 'pivot export' docs/` after each sub-step and require 0 matches before moving on. If a doc has `pivot export` embedded in a compound example, replace the example rather than deleting blindly. |
| `dvc_import.py` might have a transitive import on `dvc_compat` | Step 1 grep covers this. If it appears, STOP. |
| A fixture or test I didn't grep for uses `DVCCompatError` as an isinstance check (not via `exceptions.DVC…`) | `rg -n 'DVCCompatError'` in Step 1 covers this. |
| `test_run_cache_lock_update.py` ordering interacts with deleted `clean_registry` | Not a real risk — `clean_registry` is a no-op; file ordering doesn't affect it. But `uv run pytest -n auto` in Task 11 final sweep catches any regression. |
| Users with stale `.pivot/stages/*.lock` files that contain `dep_generations` will see all stages rerun | Exactly the intended behavior. Pre-alpha; breaking changes fine. |

No unresolved concerns.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-18-burn-legacy-state-db.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task with review between tasks. Strongly recommended for Task 2 (biggest blast radius) and Task 5 (test-surgery heavy).

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, with checkpoints after Tasks 2, 5, and 8.

**Which approach?**
