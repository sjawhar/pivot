# .pvt Hash Verification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable `pivot verify --allow-missing` and `pivot run --dry-run` to use hashes from `.pvt` files when actual dependency files are missing.

**Architecture:** Thread `allow_missing` flag and tracked file data (`tracked_files` dict + `tracked_trie`) from DAG building through to `get_stage_explanation()`. When `allow_missing=True` and a dep file is missing, look up its hash from `.pvt` data instead of failing.

**Tech Stack:** Python, pygtrie (existing dependency), pytest

---

### Task 1: Add `_find_tracked_hash` helper in `explain.py`

**Files:**
- Modify: `src/pivot/explain.py`
- Test: `tests/core/test_explain.py`

**Step 1: Write the failing test**

Add to `tests/core/test_explain.py`:

```python
# =============================================================================
# _find_tracked_hash tests
# =============================================================================


def test_find_tracked_hash_exact_match() -> None:
    """Finds hash for exact tracked file match."""
    import pygtrie

    from pivot import explain
    from pivot.storage.track import PvtData

    tracked_files: dict[str, PvtData] = {
        "/project/data.csv": PvtData(path="data.csv", hash="abc123", size=100)
    }
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[("/", "project", "data.csv")] = "/project/data.csv"

    result = explain._find_tracked_hash(
        Path("/project/data.csv"), tracked_files, tracked_trie
    )

    assert result is not None
    assert result["hash"] == "abc123"


def test_find_tracked_hash_inside_directory() -> None:
    """Finds hash for file inside tracked directory via manifest."""
    import pygtrie

    from pivot import explain
    from pivot.storage.track import PvtData

    tracked_files: dict[str, PvtData] = {
        "/project/data": PvtData(
            path="data",
            hash="tree_hash",
            size=200,
            num_files=2,
            manifest=[
                {"relpath": "file1.csv", "hash": "hash1", "size": 100, "isexec": False},
                {"relpath": "subdir/file2.csv", "hash": "hash2", "size": 100, "isexec": False},
            ],
        )
    }
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[("/", "project", "data")] = "/project/data"

    result = explain._find_tracked_hash(
        Path("/project/data/subdir/file2.csv"), tracked_files, tracked_trie
    )

    assert result is not None
    assert result["hash"] == "hash2"


def test_find_tracked_hash_not_tracked() -> None:
    """Returns None for untracked file."""
    import pygtrie

    from pivot import explain

    tracked_files: dict[str, PvtData] = {}
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()

    result = explain._find_tracked_hash(
        Path("/project/untracked.csv"), tracked_files, tracked_trie
    )

    assert result is None


def test_find_tracked_hash_not_in_manifest() -> None:
    """Returns None for file inside tracked dir but not in manifest."""
    import pygtrie

    from pivot import explain
    from pivot.storage.track import PvtData

    tracked_files: dict[str, PvtData] = {
        "/project/data": PvtData(
            path="data",
            hash="tree_hash",
            size=100,
            num_files=1,
            manifest=[
                {"relpath": "file1.csv", "hash": "hash1", "size": 100, "isexec": False},
            ],
        )
    }
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[("/", "project", "data")] = "/project/data"

    result = explain._find_tracked_hash(
        Path("/project/data/not_in_manifest.csv"), tracked_files, tracked_trie
    )

    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_explain.py::test_find_tracked_hash_exact_match -v`
Expected: FAIL with `AttributeError: module 'pivot.explain' has no attribute '_find_tracked_hash'`

**Step 3: Write minimal implementation**

Add to `src/pivot/explain.py` after the imports:

```python
import pygtrie

if TYPE_CHECKING:
    from pivot.storage.track import PvtData
```

Add the helper function before `get_stage_explanation`:

```python
def _find_tracked_ancestor(
    dep: Path, tracked_trie: pygtrie.Trie[str]
) -> Path | None:
    """Find the tracked path that contains dep (exact match or ancestor)."""
    dep_key = dep.parts

    # Exact match
    if dep_key in tracked_trie:
        return Path(tracked_trie[dep_key])

    # Dependency is inside a tracked directory
    prefix_item = tracked_trie.shortest_prefix(dep_key)
    if prefix_item is not None and prefix_item.value is not None:
        return Path(prefix_item.value)

    return None


def _find_tracked_hash(
    dep: Path,
    tracked_files: dict[str, PvtData],
    tracked_trie: pygtrie.Trie[str],
) -> HashInfo | None:
    """Find hash for dep from tracked files data.

    Returns HashInfo if dep is tracked (exact match or inside tracked directory),
    None otherwise.
    """
    tracked_path = _find_tracked_ancestor(dep, tracked_trie)
    if not tracked_path:
        return None

    pvt_data = tracked_files[str(tracked_path)]

    # Exact match - use top-level hash
    if dep == tracked_path:
        if "manifest" in pvt_data:
            return {"hash": pvt_data["hash"], "manifest": pvt_data["manifest"]}
        return {"hash": pvt_data["hash"]}

    # Nested path - find in manifest
    if "manifest" not in pvt_data:
        return None  # Single file .pvt can't contain nested paths

    relpath = str(dep.relative_to(tracked_path))
    for entry in pvt_data["manifest"]:
        if entry["relpath"] == relpath:
            return {"hash": entry["hash"]}

    return None  # Path not found in manifest
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_explain.py::test_find_tracked_hash_exact_match tests/core/test_explain.py::test_find_tracked_hash_inside_directory tests/core/test_explain.py::test_find_tracked_hash_not_tracked tests/core/test_explain.py::test_find_tracked_hash_not_in_manifest -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(explain): add _find_tracked_hash helper for .pvt lookup"
```

---

### Task 2: Update `get_stage_explanation` signature

**Files:**
- Modify: `src/pivot/explain.py`
- Test: `tests/core/test_explain.py`

**Step 1: Write the failing test**

Add to `tests/core/test_explain.py`:

```python
def test_get_stage_explanation_with_allow_missing_uses_pvt_hash(tmp_path: Path) -> None:
    """Uses .pvt hash when allow_missing=True and file is missing."""
    import pygtrie

    from pivot import project
    from pivot.storage.track import PvtData

    # Create tracked file data (simulating .pvt file)
    data_path = tmp_path / "data.csv"
    normalized_path = str(project.normalize_path(str(data_path)))

    tracked_files: dict[str, PvtData] = {
        normalized_path: PvtData(path="data.csv", hash="pvt_hash_123", size=100)
    }
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[pathlib.Path(normalized_path).parts] = normalized_path

    # Create lock file with matching hash
    stage_lock = lock.StageLock("pvt_stage", tmp_path / "stages")
    stage_lock.write(
        LockData(
            code_manifest={"self:pvt_stage": "abc"},
            params={},
            dep_hashes={normalized_path: {"hash": "pvt_hash_123"}},
            output_hashes={},
            dep_generations={},
        )
    )

    # File does NOT exist on disk
    assert not data_path.exists()

    result = explain.get_stage_explanation(
        stage_name="pvt_stage",
        fingerprint={"self:pvt_stage": "abc"},
        deps=[str(data_path)],
        outs_paths=[],
        params_instance=None,
        overrides=None,
        state_dir=tmp_path,
        allow_missing=True,
        tracked_files=tracked_files,
        tracked_trie=tracked_trie,
    )

    # Should NOT report as missing deps - should use .pvt hash
    assert "missing" not in result["reason"].lower(), f"Got: {result['reason']}"
    assert result["will_run"] is False, "Stage should be cached (hashes match)"


def test_get_stage_explanation_with_allow_missing_stale_pvt(tmp_path: Path) -> None:
    """Detects staleness when .pvt hash differs from lock file."""
    import pygtrie

    from pivot import project
    from pivot.storage.track import PvtData

    data_path = tmp_path / "data.csv"
    normalized_path = str(project.normalize_path(str(data_path)))

    # .pvt has different hash than lock file
    tracked_files: dict[str, PvtData] = {
        normalized_path: PvtData(path="data.csv", hash="new_pvt_hash", size=100)
    }
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    tracked_trie[pathlib.Path(normalized_path).parts] = normalized_path

    stage_lock = lock.StageLock("pvt_stage", tmp_path / "stages")
    stage_lock.write(
        LockData(
            code_manifest={"self:pvt_stage": "abc"},
            params={},
            dep_hashes={normalized_path: {"hash": "old_lock_hash"}},
            output_hashes={},
            dep_generations={},
        )
    )

    result = explain.get_stage_explanation(
        stage_name="pvt_stage",
        fingerprint={"self:pvt_stage": "abc"},
        deps=[str(data_path)],
        outs_paths=[],
        params_instance=None,
        overrides=None,
        state_dir=tmp_path,
        allow_missing=True,
        tracked_files=tracked_files,
        tracked_trie=tracked_trie,
    )

    assert result["will_run"] is True
    assert "dep" in result["reason"].lower() or "input" in result["reason"].lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_explain.py::test_get_stage_explanation_with_allow_missing_uses_pvt_hash -v`
Expected: FAIL with `TypeError: get_stage_explanation() got an unexpected keyword argument 'allow_missing'`

**Step 3: Update function signature and implementation**

Modify `get_stage_explanation` in `src/pivot/explain.py`:

```python
def get_stage_explanation(
    stage_name: str,
    fingerprint: dict[str, str],
    deps: list[str],
    outs_paths: list[str],
    params_instance: pydantic.BaseModel | None,
    overrides: parameters.ParamsOverrides | None,
    state_dir: Path,
    force: bool = False,
    allow_missing: bool = False,
    tracked_files: dict[str, PvtData] | None = None,
    tracked_trie: pygtrie.Trie[str] | None = None,
) -> StageExplanation:
    """Compute detailed explanation of why a stage would run.

    Args:
        allow_missing: If True and a dep file is missing, try to use hash from
            tracked_files (.pvt data) instead of reporting as missing.
        tracked_files: Dict of absolute path -> PvtData from .pvt files.
        tracked_trie: Trie of tracked paths for efficient lookup.
    """
    # ... existing code until dep hashing ...

    # Replace the simple hash_dependencies call with pvt-aware logic
    if allow_missing and tracked_files and tracked_trie:
        deps_to_hash = list[str]()
        pvt_hashes = dict[str, HashInfo]()
        missing_deps = list[str]()

        for dep in deps:
            dep_path = pathlib.Path(dep)
            if dep_path.exists():
                deps_to_hash.append(dep)
            else:
                hash_info = _find_tracked_hash(dep_path, tracked_files, tracked_trie)
                if hash_info:
                    normalized = str(project.normalize_path(dep))
                    pvt_hashes[normalized] = hash_info
                else:
                    missing_deps.append(dep)

        file_hashes, more_missing, unreadable_deps = worker.hash_dependencies(deps_to_hash)
        dep_hashes = {**file_hashes, **pvt_hashes}
        missing_deps.extend(more_missing)
    else:
        dep_hashes, missing_deps, unreadable_deps = worker.hash_dependencies(deps)

    # ... rest of existing code ...
```

Add the import at the top:

```python
from pivot import project
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_explain.py::test_get_stage_explanation_with_allow_missing_uses_pvt_hash tests/core/test_explain.py::test_get_stage_explanation_with_allow_missing_stale_pvt -v`
Expected: PASS

**Step 5: Run all explain tests**

Run: `uv run pytest tests/core/test_explain.py -v`
Expected: All tests pass (existing tests should still work with default args)

**Step 6: Commit**

```bash
jj describe -m "feat(explain): add allow_missing support with .pvt hash fallback"
```

---

### Task 3: Update `status.py` to pass tracked data to explain

**Files:**
- Modify: `src/pivot/status.py`
- Test: `tests/cli/test_verify.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_verify.py`:

```python
def test_verify_allow_missing_uses_pvt_hash_for_deps(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """verify --allow-missing uses .pvt hash when dep file is missing."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # Create input.txt and run stage to cache
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")
        executor.run(show_output=False)

        # Track the input file (create .pvt)
        from pivot.storage import track, cache

        input_hash = cache.hash_file(pathlib.Path("input.txt"))
        pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
        track.write_pvt_file(pathlib.Path("input.txt.pvt"), pvt_data)

        # Delete the actual input file (simulating CI without data)
        pathlib.Path("input.txt").unlink()

        _setup_mock_remote(mocker, files_exist_on_remote=True)

        result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

        # Should NOT fail with "Missing deps" - should use .pvt hash
        assert "Missing deps" not in result.output, f"Got: {result.output}"
        assert result.exit_code == 0, f"Expected pass, got: {result.output}"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_verify.py::test_verify_allow_missing_uses_pvt_hash_for_deps -v`
Expected: FAIL with "Missing deps" in output (current behavior)

**Step 3: Update `_get_explanations_in_parallel` in `status.py`**

Modify `src/pivot/status.py`:

Add imports:

```python
import pygtrie

if TYPE_CHECKING:
    from pivot.storage.track import PvtData
```

Update `_get_explanations_in_parallel`:

```python
def _get_explanations_in_parallel(
    execution_order: list[str],
    state_dir: pathlib.Path,
    overrides: parameters.ParamsOverrides | None,
    force: bool = False,
    allow_missing: bool = False,
    tracked_files: dict[str, PvtData] | None = None,
    tracked_trie: pygtrie.Trie[str] | None = None,
) -> dict[str, StageExplanation]:
    """Compute stage explanations in parallel (I/O-bound: lock file reads, hashing)."""
    max_workers = min(8, len(execution_order))
    explanations_by_name = dict[str, StageExplanation]()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = dict[Future[StageExplanation], str]()
        for stage_name in execution_order:
            stage_info = registry.REGISTRY.get(stage_name)
            future = pool.submit(
                explain.get_stage_explanation,
                stage_name,
                stage_info["fingerprint"],
                stage_info["deps_paths"],
                stage_info["outs_paths"],
                stage_info["params"],
                overrides,
                state_dir,
                force=force,
                allow_missing=allow_missing,
                tracked_files=tracked_files,
                tracked_trie=tracked_trie,
            )
            futures[future] = stage_name
        # ... rest unchanged ...
```

**Step 4: Update `get_pipeline_status` to discover and pass tracked files**

```python
def get_pipeline_status(
    stages: list[str] | None,
    single_stage: bool,
    validate: bool = True,
    allow_missing: bool = False,
) -> tuple[list[PipelineStatusInfo], DiGraph[str]]:
    """Get status for all stages, tracking upstream staleness.

    Args:
        stages: Stage names to check, or None for all stages.
        single_stage: If True, check only specified stages without dependencies.
        validate: If True, validate dependency files exist during DAG building.
            Set to False with --allow-missing to skip validation.
        allow_missing: If True, use .pvt hashes for missing dependency files.
    """
    with metrics.timed("status.get_pipeline_status"):
        # Discover tracked files for both DAG validation and hash lookup
        tracked_files = None
        tracked_trie = None
        if allow_missing:
            from pivot import project

            tracked_files = track.discover_pvt_files(project.get_project_root())
            tracked_trie = dag.build_tracked_trie(tracked_files)

        graph = registry.REGISTRY.build_dag(validate=validate)
        execution_order = dag.get_execution_order(graph, stages, single_stage=single_stage)

        if not execution_order:
            return [], graph

        state_dir = config.get_state_dir()
        overrides = parameters.load_params_yaml()

        explanations_by_name = _get_explanations_in_parallel(
            execution_order,
            state_dir,
            overrides,
            allow_missing=allow_missing,
            tracked_files=tracked_files,
            tracked_trie=tracked_trie,
        )

        # ... rest unchanged ...
```

**Step 5: Export `build_tracked_trie` from `dag.py`**

The function `_build_tracked_trie` in `dag.py` is private. Rename it to `build_tracked_trie` (remove underscore) to make it public.

In `src/pivot/dag.py`, rename:

```python
def build_tracked_trie(tracked_files: dict[str, PvtData]) -> pygtrie.Trie[str]:
    """Build trie of tracked file paths for dependency checking.

    Keys are path tuples (from Path.parts), values are the absolute path string.
    """
    trie: pygtrie.Trie[str] = pygtrie.Trie()
    for abs_path in tracked_files:
        path_key = pathlib.Path(abs_path).parts
        trie[path_key] = abs_path
    return trie
```

Update the call in `build_dag`:

```python
tracked_trie = build_tracked_trie(tracked_files) if tracked_files else None
```

**Step 6: Update `verify.py` to pass `allow_missing` to status**

In `src/pivot/cli/verify.py`, update the call:

```python
pipeline_status, _ = status_mod.get_pipeline_status(
    stages_list, single_stage=False, validate=not allow_missing, allow_missing=allow_missing
)
```

**Step 7: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_verify.py::test_verify_allow_missing_uses_pvt_hash_for_deps -v`
Expected: PASS

**Step 8: Run all verify tests**

Run: `uv run pytest tests/cli/test_verify.py -v`
Expected: All tests pass

**Step 9: Commit**

```bash
jj describe -m "feat(verify): thread tracked files through status for .pvt hash lookup"
```

---

### Task 4: Add test for nested path inside tracked directory

**Files:**
- Test: `tests/cli/test_verify.py`

**Step 1: Write the test**

Add to `tests/cli/test_verify.py`:

```python
class _DirDepOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _helper_dir_dep_stage(
    data_file: Annotated[pathlib.Path, outputs.Dep("data/file.csv", loaders.PathOnly())],
) -> _DirDepOutputs:
    _ = data_file
    pathlib.Path("output.txt").write_text("done")
    return _DirDepOutputs(output=pathlib.Path("output.txt"))


def test_verify_allow_missing_uses_pvt_hash_for_nested_dep(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """verify --allow-missing uses directory .pvt manifest for nested file dep."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # Create data directory with file
        data_dir = pathlib.Path("data")
        data_dir.mkdir()
        (data_dir / "file.csv").write_text("content")

        register_test_stage(_helper_dir_dep_stage, name="process")
        executor.run(show_output=False)

        # Track the directory (create .pvt with manifest)
        from pivot.storage import track, cache

        dir_hash, manifest = cache.hash_directory(data_dir)
        pvt_data = track.PvtData(
            path="data",
            hash=dir_hash,
            size=7,
            num_files=1,
            manifest=manifest,
        )
        track.write_pvt_file(pathlib.Path("data.pvt"), pvt_data)

        # Delete the actual data directory (simulating CI without data)
        import shutil
        shutil.rmtree(data_dir)

        _setup_mock_remote(mocker, files_exist_on_remote=True)

        result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

        # Should use manifest entry hash for data/file.csv
        assert "Missing deps" not in result.output, f"Got: {result.output}"
        assert result.exit_code == 0, f"Expected pass, got: {result.output}"
```

**Step 2: Run test**

Run: `uv run pytest tests/cli/test_verify.py::test_verify_allow_missing_uses_pvt_hash_for_nested_dep -v`
Expected: PASS (implementation already handles this)

**Step 3: Commit**

```bash
jj describe -m "test(verify): add test for nested dep inside tracked directory"
```

---

### Task 5: Update `run --dry-run` to support `--allow-missing`

**Files:**
- Modify: `src/pivot/cli/run.py`
- Modify: `src/pivot/status.py`
- Test: `tests/cli/test_run.py` (or create new test file)

**Step 1: Write the failing test**

Create or add to appropriate test file:

```python
def test_run_dry_run_allow_missing_uses_pvt_hash(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """run --dry-run with missing dep uses .pvt hash when tracked."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # Create and run
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")
        executor.run(show_output=False)

        # Track input
        from pivot.storage import track, cache
        input_hash = cache.hash_file(pathlib.Path("input.txt"))
        pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
        track.write_pvt_file(pathlib.Path("input.txt.pvt"), pvt_data)

        # Delete input (simulating CI)
        pathlib.Path("input.txt").unlink()

        result = runner.invoke(cli.cli, ["run", "--dry-run", "--allow-missing"])

        # Should show "would skip" not "Missing deps"
        assert "Missing deps" not in result.output, f"Got: {result.output}"
        assert "would skip" in result.output.lower(), f"Got: {result.output}"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_run.py::test_run_dry_run_allow_missing_uses_pvt_hash -v`
Expected: FAIL (--allow-missing not supported on run --dry-run yet)

**Step 3: Add --allow-missing to run command**

In `src/pivot/cli/run.py`, add the option to the `run` command:

```python
@click.option("--allow-missing", is_flag=True, help="Allow missing dep files if tracked (.pvt exists)")
```

Add to function signature:

```python
def run(
    ...
    allow_missing: bool,
) -> None:
```

Update the dry-run handling:

```python
if dry_run:
    if explain:
        _output_explain(stages_list, single_stage, force, allow_missing=allow_missing)
    else:
        ctx.invoke(
            dry_run_cmd,
            stages=stages,
            single_stage=single_stage,
            force=force,
            as_json=as_json,
            allow_missing=allow_missing,
        )
    return
```

**Step 4: Update `_output_explain` to accept `allow_missing`**

```python
def _output_explain(
    stages_list: list[str] | None,
    single_stage: bool,
    force: bool = False,
    allow_missing: bool = False,
) -> None:
    """Output detailed stage explanations using status logic."""
    from pivot import status as status_mod
    from pivot.cli import status as status_cli

    explanations = status_mod.get_pipeline_explanations(
        stages_list, single_stage, force, allow_missing=allow_missing
    )
    status_cli.output_explain_text(explanations)
```

**Step 5: Update `dry_run_cmd` to accept `allow_missing`**

```python
@click.option("--allow-missing", is_flag=True, help="Allow missing dep files if tracked")
def dry_run_cmd(
    stages: tuple[str, ...],
    single_stage: bool,
    force: bool,
    as_json: bool,
    allow_missing: bool,
) -> None:
    """Show what would run without executing."""
    from pivot import status as status_mod

    stages_list = cli_helpers.stages_to_list(stages)
    _validate_stages(stages_list, single_stage)

    explanations = status_mod.get_pipeline_explanations(
        stages_list, single_stage, force=force, allow_missing=allow_missing
    )
    # ... rest unchanged ...
```

**Step 6: Update `get_pipeline_explanations` in `status.py`**

```python
def get_pipeline_explanations(
    stages: list[str] | None,
    single_stage: bool,
    force: bool = False,
    allow_missing: bool = False,
) -> list[StageExplanation]:
    """Get detailed explanations for all stages with upstream staleness populated."""
    with metrics.timed("status.get_pipeline_explanations"):
        # Discover tracked files if allow_missing
        tracked_files = None
        tracked_trie = None
        if allow_missing:
            from pivot import project

            tracked_files = track.discover_pvt_files(project.get_project_root())
            tracked_trie = dag.build_tracked_trie(tracked_files)

        graph = registry.REGISTRY.build_dag(validate=not allow_missing)
        execution_order = dag.get_execution_order(graph, stages, single_stage=single_stage)

        if not execution_order:
            return []

        state_dir = config.get_state_dir()
        overrides = parameters.load_params_yaml()

        explanations_by_name = _get_explanations_in_parallel(
            execution_order,
            state_dir,
            overrides,
            force=force,
            allow_missing=allow_missing,
            tracked_files=tracked_files,
            tracked_trie=tracked_trie,
        )

        # ... rest unchanged ...
```

**Step 7: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_run.py::test_run_dry_run_allow_missing_uses_pvt_hash -v`
Expected: PASS

**Step 8: Commit**

```bash
jj describe -m "feat(cli): add --allow-missing to run --dry-run"
```

---

### Task 6: Run full test suite and type checking

**Step 1: Run type checker**

Run: `uv run basedpyright .`
Expected: No errors

**Step 2: Run linter**

Run: `uv run ruff check .`
Expected: No errors

**Step 3: Run formatter**

Run: `uv run ruff format .`

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -n auto`
Expected: All tests pass

**Step 5: Final commit**

```bash
jj describe -m "feat: use .pvt hashes for verification when files missing (#265)

When --allow-missing is set, pivot verify and pivot run --dry-run now
use hashes from .pvt files for dependencies that are missing on disk.
This enables CI verification without pulling actual data files.

- Add _find_tracked_hash helper in explain.py
- Thread allow_missing and tracked files through status module
- Export build_tracked_trie from dag.py
- Add --allow-missing to run command for dry-run mode
- Add tests for exact match and nested directory dependencies"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add `_find_tracked_hash` helper | `explain.py`, `test_explain.py` |
| 2 | Update `get_stage_explanation` signature | `explain.py`, `test_explain.py` |
| 3 | Thread tracked files through status | `status.py`, `dag.py`, `verify.py`, `test_verify.py` |
| 4 | Test nested directory deps | `test_verify.py` |
| 5 | Add `--allow-missing` to `run --dry-run` | `run.py`, `status.py` |
| 6 | Final verification | All files |
