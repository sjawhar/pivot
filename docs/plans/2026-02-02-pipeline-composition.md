# Pipeline Composition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `Pipeline.include()` method allowing Pipeline A to include Pipeline B's stages while preserving B's state directory.

**Architecture:** When Pipeline A includes Pipeline B, all of B's stages are copied into A's registry with their original `state_dir` preserved. This enables seamless composition where sub-pipeline stages maintain independent state tracking (lock files, state.db) while participating in the parent pipeline's DAG.

**Tech Stack:** Python 3.13+, pytest, pydantic

---

## Background

### Current State
- `Pipeline` class exists with internal `StageRegistry`
- `RegistryStageInfo` has `state_dir` field (added in #314)
- Each stage carries its own `state_dir` set at registration time
- No way to compose pipelines together

### Target State
- `Pipeline.include(other)` copies all stages from `other` into `self`
- Included stages keep their original `state_dir` (from their source pipeline)
- Name collisions raise `PipelineConfigError`
- DAG builds correctly across included stages

### Key Design Decisions
1. **Copy, don't reference** - Stages are copied into parent registry (simpler, no circular references)
2. **Preserve state_dir** - Included stages keep their original state_dir
3. **Fail on collision** - Duplicate stage names raise error (no silent overwrite)
4. **No prefix/namespace** - Stage names stay as-is (user can rename via `name=` at registration if needed)

---

## Task 1: Add `include()` Method to Pipeline

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py:148-151`
- Test: `tests/pipeline/test_pipeline.py`

**Step 1: Add module-level import for PipelineConfigError in test file**

At the top of `tests/pipeline/test_pipeline.py`, the import already exists (line 9):
```python
from pivot.pipeline.pipeline import Pipeline
```

Add `PipelineConfigError` to the imports from `pivot.pipeline.yaml` (check if already imported, otherwise add):
```python
from pivot.pipeline.yaml import PipelineConfigError
```

**Step 2: Write the failing tests**

Add to `tests/pipeline/test_pipeline.py`:

```python
# =============================================================================
# include() Tests
# =============================================================================


def test_pipeline_include_copies_stages(tmp_path: pathlib.Path) -> None:
    """include() should copy all stages from the included pipeline."""
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_simple_stage, name="sub_stage")

    main.include(sub)

    assert "sub_stage" in main.list_stages()


def test_pipeline_include_preserves_state_dir(tmp_path: pathlib.Path) -> None:
    """Included stages should keep their original state_dir."""
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_simple_stage, name="sub_stage")
    main.include(sub)

    # Stage should have sub's state_dir, not main's
    info = main.get("sub_stage")
    assert info["state_dir"] == tmp_path / "sub" / ".pivot"


def test_pipeline_include_name_collision_raises(tmp_path: pathlib.Path) -> None:
    """include() should raise PipelineConfigError on stage name collision."""
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    main.register(_simple_stage, name="train")
    sub.register(_simple_stage, name="train")

    with pytest.raises(PipelineConfigError, match="train.*already exists"):
        main.include(sub)


def test_pipeline_include_empty_pipeline(tmp_path: pathlib.Path) -> None:
    """include() with empty pipeline should be a no-op."""
    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="existing")

    empty = Pipeline("empty", root=tmp_path / "empty")

    main.include(empty)  # Should not raise

    assert main.list_stages() == ["existing"]


def test_pipeline_include_self_raises(tmp_path: pathlib.Path) -> None:
    """Including a pipeline into itself should raise."""
    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="stage")

    with pytest.raises(PipelineConfigError, match="cannot include itself"):
        main.include(main)


def test_pipeline_include_same_pipeline_twice_raises(tmp_path: pathlib.Path) -> None:
    """Including the same pipeline twice should raise on collision."""
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")
    sub.register(_simple_stage, name="sub_stage")

    main.include(sub)

    with pytest.raises(PipelineConfigError, match="sub_stage.*already exists"):
        main.include(sub)


def test_pipeline_include_multiple_different_pipelines(tmp_path: pathlib.Path) -> None:
    """Including multiple different pipelines should work."""
    main = Pipeline("main", root=tmp_path / "main")
    sub1 = Pipeline("sub1", root=tmp_path / "sub1")
    sub2 = Pipeline("sub2", root=tmp_path / "sub2")

    sub1.register(_simple_stage, name="stage1")
    sub2.register(_simple_stage, name="stage2")

    main.include(sub1)
    main.include(sub2)

    assert set(main.list_stages()) == {"stage1", "stage2"}


def test_pipeline_include_preserves_params(tmp_path: pathlib.Path) -> None:
    """Included stages should keep their params."""
    from pivot.stage_def import StageParams

    class _TestParams(StageParams):
        value: int = 42

    def _parameterized_stage(params: _TestParams) -> _SimpleOutput:
        pathlib.Path("result.txt").write_text(str(params.value))
        return _SimpleOutput(result=pathlib.Path("result.txt"))

    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_parameterized_stage, name="param_stage", params=_TestParams(value=100))
    main.include(sub)

    info = main.get("param_stage")
    assert info["params"] is not None
    assert info["params"].value == 100


def test_pipeline_include_dag_connects_across_pipelines(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """DAG should connect main stage depending on included sub-pipeline stage."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path)
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_stage_a, name="producer")
    main.include(sub)
    main.register(_stage_b, name="consumer")

    dag = main.build_dag(validate=True)

    assert "producer" in dag.nodes
    assert "consumer" in dag.nodes
    # consumer depends on producer (edge from consumer -> producer)
    assert dag.has_edge("consumer", "producer")
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_pipeline.py::test_pipeline_include_copies_stages -v`
Expected: FAIL with "AttributeError: 'Pipeline' object has no attribute 'include'"

**Step 4: Write the implementation**

Add to `src/pivot/pipeline/pipeline.py` after `clear()` method (line 150):

```python
    def include(self, other: Pipeline) -> None:
        """Include all stages from another pipeline.

        Stages are copied with their original state_dir preserved, enabling
        composition where sub-pipeline stages maintain independent state tracking.

        Args:
            other: Pipeline whose stages to include.

        Raises:
            PipelineConfigError: If including self or any stage name already exists.
        """
        if other is self:
            raise PipelineConfigError(
                f"Pipeline '{self.name}' cannot include itself"
            )

        for stage_name in other.list_stages():
            if stage_name in self._registry.list_stages():
                raise PipelineConfigError(
                    f"Cannot include pipeline '{other.name}': "
                    f"stage '{stage_name}' already exists in '{self.name}'. "
                    f"Rename the stage using name= at registration time."
                )
            self._registry._stages[stage_name] = other.get(stage_name)

        self._registry.invalidate_dag_cache()
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_pipeline.py -k include -v`
Expected: All 9 tests PASS

**Step 6: Run quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: PASS

**Step 7: Commit**

```bash
jj describe -m "feat(pipeline): add include() method for pipeline composition

- include() copies stages preserving original state_dir
- Self-include raises PipelineConfigError
- Name collisions raise PipelineConfigError with fix hint
- DAG builds correctly across included stages"
```

---

## Task 2: Update writing-pivot-stages Skill

**Files:**
- Modify: `skills/writing-pivot-stages/SKILL.md`

**Step 1: Add Pipeline Composition section**

Add after the "Running Stages" section (search for `## Running Stages` and add after its content):

```markdown
## Pipeline Composition

Include stages from sub-pipelines while preserving their state directories:

```python
# Define sub-pipeline
preprocessing = Pipeline("preprocessing")
preprocessing.register(clean_data, name="clean")
preprocessing.register(normalize, name="normalize")

# Include in main pipeline
main = Pipeline("main")
main.include(preprocessing)  # Copies stages, preserves state_dir
main.register(train, name="train")  # Can depend on preprocessing outputs
```

**Behavior:**
- Included stages keep their original `state_dir` (for lock files, state.db)
- Including empty pipeline is a no-op
- Including same pipeline twice raises (name collision)
- Stages included transitively (if B includes C, then A includes B, A gets C's stages too)

**Rules:**
- Stage name collisions raise `PipelineConfigError`
- Cannot include a pipeline into itself
```

**Step 2: Update Common Errors table**

Add to the Common Errors table:

```markdown
| `stage 'X' already exists` | Name collision in `include()` | Rename stage with `name=` at registration |
| `cannot include itself` | Self-include attempted | Use a separate Pipeline instance |
```

**Step 3: Commit**

```bash
jj describe -m "docs(skill): add pipeline composition to writing-pivot-stages"
```

---

## Task 3: Update Reference Documentation

**Files:**
- Modify: `docs/reference/pipelines.md` (if exists, otherwise add to appropriate docs)

**Step 1: Check docs structure**

Run: `ls docs/reference/` to see existing files.

**Step 2: Add Pipeline Composition section**

Add to `docs/reference/pipelines.md` (or create if needed):

```markdown
## Pipeline Composition

Pipelines can include other pipelines to compose larger workflows:

```python
from pivot.pipeline import Pipeline

# Create sub-pipeline for data preprocessing
preprocessing = Pipeline("preprocessing")
preprocessing.register(clean_data)
preprocessing.register(normalize)

# Create main pipeline that includes preprocessing
main = Pipeline("main")
main.include(preprocessing)
main.register(train)
main.register(evaluate)
```

### State Isolation

When Pipeline A includes Pipeline B:
- B's stages are copied into A's registry
- B's stages keep their original `state_dir` (`.pivot/` in B's root)
- Lock files and state.db remain in B's directory
- The project-wide cache is shared

This enables modular pipeline organization where each sub-pipeline can be developed, tested, and run independently.

### Name Collisions

If an included pipeline has a stage with the same name as an existing stage, `include()` raises `PipelineConfigError`. Rename stages at registration time to avoid collisions:

```python
sub.register(my_stage, name="sub_preprocess")  # Use unique name
main.include(sub)
```

### Edge Cases

- **Empty pipeline:** Including an empty pipeline is a no-op
- **Self-include:** Raises `PipelineConfigError`
- **Multiple includes:** Can include multiple different pipelines; including same pipeline twice raises collision error
- **Transitivity:** If B includes C, then A includes B, A gets all of C's stages (they're already in B's registry)
```

**Step 3: Commit**

```bash
jj describe -m "docs: add pipeline composition documentation"
```

---

## Task 4: Clean Up Outdated REGISTRY Comment

**Files:**
- Modify: `tests/engine/test_graph.py:80`

**Step 1: Update the outdated comment**

Change line 80 from:
```python
    # Register stages directly (bypass REGISTRY for isolated test)
```

To:
```python
    # Create stages dict directly for isolated graph test
```

**Step 2: Commit**

```bash
jj describe -m "chore: remove outdated REGISTRY comment from test"
```

---

## Task 5: Final Verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/pipeline/ -v`
Expected: All tests PASS

**Step 2: Run quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: PASS

**Step 3: Verify no REGISTRY in active code**

Run: `rg "REGISTRY" src/ tests/ --type py -l`
Expected: Only `src/pivot/dvc_compat.py` (deprecation note) and `tests/conftest.py` (historical comment)

**Step 4: Push changes**

```bash
jj git push
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add `include()` method + comprehensive tests | pipeline.py, test_pipeline.py |
| 2 | Update skill | writing-pivot-stages/SKILL.md |
| 3 | Update docs | docs/reference/pipelines.md |
| 4 | Clean up REGISTRY comment | test_graph.py |
| 5 | Final verification | - |
