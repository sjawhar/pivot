# Auto-Resolve Sibling Pipeline Dependencies

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically resolve cross-pipeline dependencies by searching from the dependency's directory, not just parent directories.

**Architecture:** Rename `resolve_from_parents()` to `resolve_external_dependencies()`. Change resolution to start searching from each dependency's directory (where the file would be located), traversing up to project root. Call this automatically in `build_dag()`. This enables sibling pipelines to depend on each other without explicit `include()` calls.

**Tech Stack:** Python, NetworkX (DAG), pathlib

---

## Background

Currently:
- `resolve_from_parents()` exists but is never called automatically
- It only searches parent directories of the *consuming pipeline*, not the dependency's location
- Sibling pipelines (e.g., `model_reports/time_horizon_1_0` depending on `model_reports/time_horizon_1_1`) cannot resolve each other's outputs

The fix:
1. Wire up automatic resolution in `build_dag()`
2. Change search strategy: for each unresolved dependency, search from the dependency's directory up to project root

---

## Task 1: Add `find_pipeline_paths_for_dependency()` to discovery.py

**Files:**
- Modify: `src/pivot/discovery.py:150-185`
- Test: `tests/unit/test_discovery.py`

**Step 1: Write failing test for the new function**

Add to `tests/unit/test_discovery.py`:

```python
def test_find_pipeline_paths_for_dependency_finds_sibling(
    tmp_path: pathlib.Path,
) -> None:
    """Should find pipeline in dependency's directory, not just parents."""
    # Create sibling pipeline structure:
    # tmp_path/
    #   sibling_a/pipeline.py  <- consuming pipeline
    #   sibling_b/pipeline.py  <- produces the dependency
    sibling_a = tmp_path / "sibling_a"
    sibling_b = tmp_path / "sibling_b"
    sibling_a.mkdir()
    sibling_b.mkdir()

    (sibling_a / "pipeline.py").write_text("# consumer")
    (sibling_b / "pipeline.py").write_text("# producer")

    # Dependency path is in sibling_b
    dep_path = sibling_b / "data" / "output.csv"

    from pivot import discovery

    paths = list(discovery.find_pipeline_paths_for_dependency(dep_path, tmp_path))

    # Should find sibling_b's pipeline (closest to dependency)
    assert sibling_b / "pipeline.py" in paths
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_discovery.py::test_find_pipeline_paths_for_dependency_finds_sibling -v`
Expected: FAIL with "AttributeError: module 'pivot.discovery' has no attribute 'find_pipeline_paths_for_dependency'"

**Step 3: Implement `find_pipeline_paths_for_dependency()`**

In `src/pivot/discovery.py`, add after line 185:

```python
def find_pipeline_paths_for_dependency(
    dep_path: pathlib.Path,
    stop_at: pathlib.Path,
) -> Iterator[pathlib.Path]:
    """Find pipeline config files starting from a dependency's directory.

    Starts from the dependency's parent directory and traverses up to stop_at,
    yielding each pivot.yaml or pipeline.py found. Closest directories first.

    This enables resolution of sibling pipeline dependencies - if a dependency
    is in ../sibling_b/data/file.csv, we search sibling_b/ for a pipeline that
    produces it.

    Args:
        dep_path: Path to the dependency (file or directory).
        stop_at: Stop traversal at this directory (inclusive).

    Yields:
        Paths to pivot.yaml or pipeline.py files.

    Raises:
        DiscoveryError: If a directory has both pivot.yaml and pipeline.py,
            or if path resolution fails.
    """
    try:
        # Start from dependency's parent directory (the directory containing the dep)
        current = dep_path.resolve().parent
        stop_at_resolved = stop_at.resolve()
    except OSError as e:
        raise DiscoveryError(f"Failed to resolve paths: {e}") from e

    # Traverse up to project root
    while current.is_relative_to(stop_at_resolved):
        config_path = _find_config_path_in_dir(current)
        if config_path:
            yield config_path

        if current == stop_at_resolved or current.parent == current:
            break
        current = current.parent
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_discovery.py::test_find_pipeline_paths_for_dependency_finds_sibling -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "feat(discovery): add find_pipeline_paths_for_dependency for sibling resolution"
```

---

## Task 2: Add more test cases for find_pipeline_paths_for_dependency

**Files:**
- Test: `tests/unit/test_discovery.py`

**Step 1: Write additional test cases**

Add to `tests/unit/test_discovery.py`:

```python
def test_find_pipeline_paths_for_dependency_nested(
    tmp_path: pathlib.Path,
) -> None:
    """Should find pipelines at multiple levels when dependency is deeply nested."""
    # Structure:
    # tmp_path/
    #   pipeline.py           <- root pipeline
    #   subdir/
    #     pipeline.py         <- intermediate pipeline
    #     deep/
    #       data/output.csv   <- dependency location

    subdir = tmp_path / "subdir"
    deep = subdir / "deep"
    deep.mkdir(parents=True)

    (tmp_path / "pipeline.py").write_text("# root")
    (subdir / "pipeline.py").write_text("# intermediate")

    dep_path = deep / "data" / "output.csv"

    from pivot import discovery

    paths = list(discovery.find_pipeline_paths_for_dependency(dep_path, tmp_path))

    # Should find both, closest first
    assert paths == [subdir / "pipeline.py", tmp_path / "pipeline.py"]


def test_find_pipeline_paths_for_dependency_stops_at_project_root(
    tmp_path: pathlib.Path,
) -> None:
    """Should not traverse above project root."""
    # Dependency outside project root should still be bounded
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pipeline.py").write_text("# project root")

    # Dep path within project
    dep_path = project_root / "data" / "file.csv"

    from pivot import discovery

    paths = list(discovery.find_pipeline_paths_for_dependency(dep_path, project_root))

    assert paths == [project_root / "pipeline.py"]


def test_find_pipeline_paths_for_dependency_no_pipelines(
    tmp_path: pathlib.Path,
) -> None:
    """Should return empty when no pipelines exist."""
    subdir = tmp_path / "empty"
    subdir.mkdir()
    dep_path = subdir / "data.csv"

    from pivot import discovery

    paths = list(discovery.find_pipeline_paths_for_dependency(dep_path, tmp_path))

    assert paths == []
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_discovery.py -k find_pipeline_paths_for_dependency -v`
Expected: All PASS

**Step 3: Commit**

```bash
jj describe -m "test(discovery): add coverage for find_pipeline_paths_for_dependency"
```

---

## Task 3: Rename and refactor `resolve_from_parents()` to `resolve_external_dependencies()`

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py:319-400`
- Test: `tests/integration/test_lazy_resolution.py`

**Step 1: Write failing test for sibling resolution**

Add to `tests/integration/test_lazy_resolution.py`:

```python
def test_resolve_external_dependencies_sibling_pipelines(
    set_project_root: pathlib.Path,
) -> None:
    """Should resolve dependencies from sibling pipeline directories.

    This is the core use case: time_horizon_1_0 depends on output from
    time_horizon_1_1, where both are siblings under model_reports/.
    """
    # Create sibling structure:
    # project_root/
    #   model_reports/
    #     sibling_a/pipeline.py  <- consumer, depends on ../sibling_b/data/output.txt
    #     sibling_b/pipeline.py  <- producer of data/output.txt

    model_reports = set_project_root / "model_reports"
    sibling_a = model_reports / "sibling_a"
    sibling_b = model_reports / "sibling_b"
    sibling_a.mkdir(parents=True)
    sibling_b.mkdir(parents=True)

    # Producer in sibling_b
    (sibling_b / "pipeline.py").write_text(
        _make_producer_pipeline_code("sibling_b", "producer", "data/output.txt")
    )

    # Consumer in sibling_a depends on sibling_b's output
    (sibling_a / "pipeline.py").write_text(
        _make_consumer_pipeline_code(
            "sibling_a", "consumer", "../sibling_b/data/output.txt", "result.txt"
        )
    )

    # Load consumer pipeline and resolve
    consumer = discovery.load_pipeline_from_path(sibling_a / "pipeline.py")
    assert consumer is not None

    consumer.resolve_external_dependencies()

    # Should include producer from sibling
    assert "producer" in consumer.list_stages()
    assert "consumer" in consumer.list_stages()

    # Build DAG should succeed
    dag = consumer.build_dag(validate=True)
    assert dag.has_edge("consumer", "producer")

    # Producer's state_dir should be sibling_b's .pivot
    producer_info = consumer.get("producer")
    assert producer_info["state_dir"] == sibling_b / ".pivot"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_lazy_resolution.py::test_resolve_external_dependencies_sibling_pipelines -v`
Expected: FAIL with "AttributeError: 'Pipeline' object has no attribute 'resolve_external_dependencies'"

**Step 3: Rename and refactor the method**

In `src/pivot/pipeline/pipeline.py`, replace `resolve_from_parents()` (lines 319-400) with:

```python
    def resolve_external_dependencies(self) -> None:
        """Resolve unresolved dependencies by searching for external pipelines.

        For each dependency that has no local producer:
        1. Start from the dependency's directory and traverse up to project root
        2. Load each pipeline found and search for a stage producing the artifact
        3. Include that stage and add its dependencies to the work queue

        This enables both parent and sibling pipeline dependencies to be resolved
        automatically. Dependencies that exist on disk are treated as external inputs.

        Uses per-call caching (pipelines loaded once per resolve, discarded after).
        """
        project_root = project.get_project_root()

        # Build set of locally produced outputs and unresolved dependencies in single pass
        local_outputs = set[str]()
        all_deps = set[str]()
        for stage_name in self.list_stages():
            info = self.get(stage_name)
            local_outputs.update(info["outs_paths"])
            all_deps.update(info["deps_paths"])

        # Work queue is deps not satisfied locally
        work = all_deps - local_outputs

        if not work:
            return

        # Per-call cache: avoid reloading same pipeline for each unresolved dep
        loaded_pipelines: dict[pathlib.Path, Pipeline | None] = {}

        def get_pipeline(path: pathlib.Path) -> Pipeline | None:
            if path not in loaded_pipelines:
                loaded_pipelines[path] = discovery.load_pipeline_from_path(path)
            return loaded_pipelines[path]

        # Process work queue iteratively
        while work:
            dep_path = work.pop()

            # Skip if already resolved (by a stage we just added) or exists on disk
            if dep_path in local_outputs or pathlib.Path(dep_path).exists():
                continue

            # Search for pipelines starting from the dependency's directory
            pipeline_files = discovery.find_pipeline_paths_for_dependency(
                pathlib.Path(dep_path), project_root
            )

            for pipeline_file in pipeline_files:
                pipeline = get_pipeline(pipeline_file)
                if pipeline is None:
                    continue

                # Find stage that produces this artifact
                producer_name = next(
                    (
                        name
                        for name in pipeline.list_stages()
                        if dep_path in pipeline.get(name)["outs_paths"]
                    ),
                    None,
                )
                if producer_name is None:
                    continue

                # Skip if already included (idempotency)
                if producer_name in self._registry.list_stages():
                    break

                # Include the producer stage
                stage_info = copy.deepcopy(pipeline.get(producer_name))
                self._registry.add_existing(stage_info)
                local_outputs.update(stage_info["outs_paths"])

                # Add producer's unresolved dependencies to work queue
                work.update(dep for dep in stage_info["deps_paths"] if dep not in local_outputs)

                logger.debug(
                    f"Included stage '{producer_name}' from pipeline '{pipeline.name}'"
                )
                break

    # Keep old name as alias for backwards compatibility
    resolve_from_parents = resolve_external_dependencies
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_lazy_resolution.py::test_resolve_external_dependencies_sibling_pipelines -v`
Expected: PASS

**Step 5: Verify existing tests still pass**

Run: `uv run pytest tests/integration/test_lazy_resolution.py -v`
Expected: All PASS (old tests use `resolve_from_parents()` which is now an alias)

**Step 6: Commit**

```bash
jj describe -m "feat(pipeline): rename resolve_from_parents to resolve_external_dependencies

Search from dependency's directory instead of consuming pipeline's directory.
Enables sibling pipeline dependencies to be resolved automatically."
```

---

## Task 4: Wire up automatic resolution in `build_dag()`

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py:249-262`
- Test: `tests/integration/test_lazy_resolution.py`

**Step 1: Write failing test for automatic resolution**

Add to `tests/integration/test_lazy_resolution.py`:

```python
def test_build_dag_auto_resolves_external_dependencies(
    set_project_root: pathlib.Path,
) -> None:
    """build_dag() should automatically resolve external dependencies.

    Users shouldn't need to call resolve_external_dependencies() explicitly.
    """
    # Same sibling structure as previous test
    model_reports = set_project_root / "model_reports"
    sibling_a = model_reports / "sibling_a"
    sibling_b = model_reports / "sibling_b"
    sibling_a.mkdir(parents=True)
    sibling_b.mkdir(parents=True)

    (sibling_b / "pipeline.py").write_text(
        _make_producer_pipeline_code("sibling_b", "producer", "data/output.txt")
    )
    (sibling_a / "pipeline.py").write_text(
        _make_consumer_pipeline_code(
            "sibling_a", "consumer", "../sibling_b/data/output.txt", "result.txt"
        )
    )

    # Load consumer pipeline - do NOT call resolve_external_dependencies()
    consumer = discovery.load_pipeline_from_path(sibling_a / "pipeline.py")
    assert consumer is not None

    # build_dag should auto-resolve and succeed
    dag = consumer.build_dag(validate=True)

    # Should have included producer automatically
    assert "producer" in consumer.list_stages()
    assert dag.has_edge("consumer", "producer")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_lazy_resolution.py::test_build_dag_auto_resolves_external_dependencies -v`
Expected: FAIL with "DependencyNotFoundError"

**Step 3: Modify `build_dag()` to auto-resolve**

In `src/pivot/pipeline/pipeline.py`, replace the `build_dag()` method (lines 249-262):

```python
    def build_dag(self, validate: bool = True) -> DiGraph[str]:
        """Build DAG from registered stages.

        Automatically resolves external dependencies before building. For each
        dependency without a local producer, searches for pipelines starting from
        the dependency's directory and traversing up to project root.

        Args:
            validate: If True, validate that all dependencies exist

        Returns:
            NetworkX DiGraph with stages as nodes and dependencies as edges

        Raises:
            CyclicGraphError: If graph contains cycles
            DependencyNotFoundError: If dependency doesn't exist (when validate=True)
        """
        # Auto-resolve external dependencies before building
        self.resolve_external_dependencies()
        return self._registry.build_dag(validate=validate)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_lazy_resolution.py::test_build_dag_auto_resolves_external_dependencies -v`
Expected: PASS

**Step 5: Run all lazy resolution tests**

Run: `uv run pytest tests/integration/test_lazy_resolution.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
jj describe -m "feat(pipeline): auto-resolve external dependencies in build_dag()

build_dag() now calls resolve_external_dependencies() automatically,
so users no longer need to call it explicitly."
```

---

## Task 5: Run full test suite and quality checks

**Files:**
- All modified files

**Step 1: Run linting**

Run: `uv run ruff format . && uv run ruff check .`
Expected: No errors

**Step 2: Run type checking**

Run: `uv run basedpyright`
Expected: No errors

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -n auto`
Expected: All tests pass

**Step 4: Final commit**

```bash
jj describe -m "feat(pipeline): auto-resolve sibling and parent dependencies (#XXX)

- Add find_pipeline_paths_for_dependency() to discovery.py
- Rename resolve_from_parents() to resolve_external_dependencies()
- Search from dependency's directory instead of consuming pipeline's directory
- Auto-call resolution in build_dag()

Fixes: dependency resolution for sibling pipelines like time_horizon_1_0
depending on time_horizon_1_1."
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add `find_pipeline_paths_for_dependency()` | `discovery.py`, `test_discovery.py` |
| 2 | Add test coverage for new function | `test_discovery.py` |
| 3 | Rename/refactor to `resolve_external_dependencies()` | `pipeline.py`, `test_lazy_resolution.py` |
| 4 | Wire up auto-resolution in `build_dag()` | `pipeline.py`, `test_lazy_resolution.py` |
| 5 | Quality checks and final verification | All files |
