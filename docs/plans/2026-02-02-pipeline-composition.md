# Pipeline Composition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `Pipeline.include()` method allowing Pipeline A to include Pipeline B's stages while preserving B's state directory.

**Architecture:** When Pipeline A includes Pipeline B, all of B's stages are deep-copied into A's registry with their original `state_dir` preserved. This enables seamless composition where sub-pipeline stages maintain independent state tracking (lock files, state.db) while participating in the parent pipeline's DAG.

**Tech Stack:** Python 3.13+, pytest, pydantic

---

## Enhancement Summary

**Deepened on:** 2026-02-02
**Research agents used:** 10 (Kieran Python, Architecture, Pattern, Performance, Security, Best Practices, Repo Analysis, Learnings, Bug Finder, Test Analyzer)

### Key Improvements from Research
1. **Deep copy required** - Prevents shared mutable state bugs
2. **Encapsulation fix** - Add `StageRegistry.add_existing()` method
3. **O(n) optimization** - Set-based collision detection
4. **Security validation** - Validate state_dir within pipeline root
5. **Additional tests** - Transitive, atomicity, mutation isolation

### Critical Patterns from Institutional Learnings
- Stage functions and TypedDicts must be module-level (pickling + type resolution)
- Test helpers must be module-level with `_` prefix (fingerprinting)
- Use `copy.deepcopy()` for `RegistryStageInfo` to prevent cross-pipeline mutation

---

## Background

### Current State
- `Pipeline` class exists with internal `StageRegistry`
- `RegistryStageInfo` has `state_dir` field (added in #314)
- Each stage carries its own `state_dir` set at registration time
- No way to compose pipelines together

### Target State
- `Pipeline.include(other)` deep-copies all stages from `other` into `self`
- Included stages keep their original `state_dir` (from their source pipeline)
- Name collisions raise `PipelineConfigError`
- DAG builds correctly across included stages
- Mutations to source pipeline don't affect including pipeline

### Key Design Decisions
1. **Deep copy, don't reference** - Stages are deep-copied to prevent shared mutable state
2. **Preserve state_dir** - Included stages keep their original state_dir
3. **Fail on collision** - Duplicate stage names raise error (no silent overwrite)
4. **Atomic operation** - Collision detected before any stages are added
5. **No prefix/namespace** - Stage names stay as-is (user can rename via `name=` at registration if needed)

---

## Task 1: Add `add_existing()` Method to StageRegistry

**Files:**
- Modify: `src/pivot/registry.py`
- Test: `tests/config/test_registry.py`

### Research Insights

**Best Practice (Architecture Review):** Avoid direct `_stages` access from `Pipeline`. Add a proper method to `StageRegistry` that handles adding pre-validated stage info while maintaining encapsulation.

**Step 1: Write the failing test**

Add to `tests/config/test_registry.py`:

```python
def test_registry_add_existing_stage(tmp_path: pathlib.Path) -> None:
    """add_existing() should add a pre-built RegistryStageInfo."""
    reg = StageRegistry()

    # Create a stage info manually (simulating copy from another registry)
    info = RegistryStageInfo(
        func=_simple_stage,
        name="added_stage",
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=tmp_path / ".pivot",
    )

    reg.add_existing(info)

    assert "added_stage" in reg.list_stages()
    assert reg.get("added_stage")["state_dir"] == tmp_path / ".pivot"


def test_registry_add_existing_collision_raises() -> None:
    """add_existing() should raise ValidationError on name collision."""
    reg = StageRegistry()
    reg.register(_simple_stage, name="existing")

    info = RegistryStageInfo(
        func=_simple_stage,
        name="existing",  # Collision!
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
    )

    with pytest.raises(exceptions.ValidationError, match="already registered"):
        reg.add_existing(info)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_registry.py::test_registry_add_existing_stage -v`
Expected: FAIL with "AttributeError: 'StageRegistry' object has no attribute 'add_existing'"

**Step 3: Write the implementation**

Add to `src/pivot/registry.py` in the `StageRegistry` class (after `register()` method):

```python
    def add_existing(self, stage_info: RegistryStageInfo) -> None:
        """Add a pre-validated stage info (for pipeline composition).

        Unlike register(), this accepts already-validated stage info from another
        registry. Use for copying stages between pipelines.

        Args:
            stage_info: Complete stage info to add.

        Raises:
            ValidationError: If stage name already exists.
        """
        name = stage_info["name"]
        if name in self._stages:
            raise exceptions.ValidationError(f"Stage '{name}' already registered")
        self._stages[name] = stage_info
        self._cached_dag = None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_registry.py -k add_existing -v`
Expected: Both tests PASS

**Step 5: Commit**

```bash
jj describe -m "feat(registry): add add_existing() method for pipeline composition"
```

---

## Task 2: Add `include()` Method to Pipeline

**Files:**
- Modify: `src/pivot/pipeline/pipeline.py`
- Test: `tests/pipeline/test_pipeline.py`

### Research Insights

**Performance (O(n) vs O(n²)):** Use set-based lookup for collision detection instead of calling `list_stages()` in loop.

**Security (state_dir validation):** Validate that included stage's `state_dir` is within the included pipeline's root to prevent path traversal.

**Bug Prevention (deep copy):** Use `copy.deepcopy()` to prevent shared mutable state between pipelines.

**Step 1: Add module-level imports in test file**

At the top of `tests/pipeline/test_pipeline.py`, ensure these imports exist:

```python
import copy
from pivot.pipeline.yaml import PipelineConfigError
from pivot.stage_def import StageParams
```

**Step 2: Add module-level test helpers**

Add after existing helpers (before the tests):

```python
# Helper for params preservation test
class _TestIncludeParams(StageParams):
    value: int = 42


def _parameterized_stage_for_include(params: _TestIncludeParams) -> _SimpleOutput:
    pathlib.Path("result.txt").write_text(str(params.value))
    return _SimpleOutput(result=pathlib.Path("result.txt"))
```

**Step 3: Write the failing tests**

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


def test_pipeline_include_preserves_state_dir_transitively(tmp_path: pathlib.Path) -> None:
    """Transitive inclusion should preserve original state_dir through multiple levels.

    When A includes B and B includes C, C's stages in A should retain C's state_dir.
    Critical for ensuring lock files and state.db remain in correct locations.
    """
    # Level 1: Base pipeline
    base = Pipeline("base", root=tmp_path / "base")
    base.register(_simple_stage, name="base_stage")

    # Level 2: Intermediate pipeline includes base
    intermediate = Pipeline("intermediate", root=tmp_path / "intermediate")
    intermediate.include(base)
    intermediate.register(_stage_a, name="intermediate_stage")

    # Level 3: Main pipeline includes intermediate (gets base transitively)
    main = Pipeline("main", root=tmp_path / "main")
    main.include(intermediate)

    # Verify base_stage retained base's state_dir (not intermediate's)
    base_info = main.get("base_stage")
    assert base_info["state_dir"] == tmp_path / "base" / ".pivot"

    # Verify intermediate_stage has intermediate's state_dir
    intermediate_info = main.get("intermediate_stage")
    assert intermediate_info["state_dir"] == tmp_path / "intermediate" / ".pivot"


def test_pipeline_include_name_collision_raises(tmp_path: pathlib.Path) -> None:
    """include() should raise PipelineConfigError on stage name collision."""
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    main.register(_simple_stage, name="train")
    sub.register(_simple_stage, name="train")

    with pytest.raises(PipelineConfigError, match="train.*already exists"):
        main.include(sub)


def test_pipeline_include_collision_transitive(tmp_path: pathlib.Path) -> None:
    """Name collision should be detected even in transitively included stages."""
    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="train")

    # Nested pipeline has same stage name
    subsub = Pipeline("subsub", root=tmp_path / "subsub")
    subsub.register(_stage_a, name="train")

    # Intermediate includes nested
    sub = Pipeline("sub", root=tmp_path / "sub")
    sub.include(subsub)

    # Should detect collision with transitively included stage
    with pytest.raises(PipelineConfigError, match="train.*already exists"):
        main.include(sub)


def test_pipeline_include_collision_is_atomic(tmp_path: pathlib.Path) -> None:
    """Include should be atomic: collision should not partially add stages.

    Important for preventing registry corruption when include() fails partway through.
    """
    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="conflict")

    sub = Pipeline("sub", root=tmp_path / "sub")
    sub.register(_stage_a, name="safe")
    sub.register(_stage_b, name="conflict")

    initial_stages = set(main.list_stages())

    with pytest.raises(PipelineConfigError):
        main.include(sub)

    # Should be unchanged (not have "safe" added)
    assert set(main.list_stages()) == initial_stages


def test_pipeline_include_empty_pipeline(tmp_path: pathlib.Path) -> None:
    """include() with empty pipeline should be a no-op."""
    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="existing")

    empty = Pipeline("empty", root=tmp_path / "empty")

    main.include(empty)  # Should not raise

    assert set(main.list_stages()) == {"existing"}


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
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(
        _parameterized_stage_for_include,
        name="param_stage",
        params=_TestIncludeParams(value=100),
    )
    main.include(sub)

    info = main.get("param_stage")
    assert info["params"] is not None
    assert info["params"].value == 100


def test_pipeline_include_isolates_mutations(tmp_path: pathlib.Path) -> None:
    """Mutations to included stages shouldn't affect source pipeline.

    Critical for preventing action-at-a-distance bugs where modifying one
    pipeline unexpectedly affects another.
    """
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_simple_stage, name="stage", mutex=["original"])
    main.include(sub)

    # Mutate main's copy
    main.get("stage")["mutex"].append("test_mutex")

    # Sub's copy should be unaffected
    assert "test_mutex" not in sub.get("stage")["mutex"]


def test_pipeline_include_is_independent_copy(tmp_path: pathlib.Path) -> None:
    """Stages are copied at include time; subsequent changes don't propagate."""
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_simple_stage, name="original")
    main.include(sub)

    # Modify sub after inclusion
    sub.register(_stage_a, name="new_stage")

    # Main should be unaffected
    assert "original" in main.list_stages()
    assert "new_stage" not in main.list_stages()


def test_pipeline_include_invalidates_dag_cache(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Include should invalidate cached DAG to ensure freshness."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path)
    main.register(_stage_a, name="stage_a")

    # Build DAG (caches it)
    dag1 = main.build_dag(validate=False)
    assert len(dag1.nodes) == 1

    # Include another pipeline
    sub = Pipeline("sub", root=tmp_path / "sub")
    sub.register(_stage_b, name="stage_b")
    main.include(sub)

    # Subsequent build_dag should see new stage
    dag2 = main.build_dag(validate=False)
    assert len(dag2.nodes) == 2
    assert "stage_b" in dag2.nodes


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

**Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_pipeline.py::test_pipeline_include_copies_stages -v`
Expected: FAIL with "AttributeError: 'Pipeline' object has no attribute 'include'"

**Step 5: Write the implementation**

Add imports at top of `src/pivot/pipeline/pipeline.py`:

```python
import copy
import logging

logger = logging.getLogger(__name__)
```

Add to `src/pivot/pipeline/pipeline.py` after `clear()` method:

```python
    def include(self, other: Pipeline) -> None:
        """Include all stages from another pipeline.

        Stages are deep-copied with their original state_dir preserved, enabling
        composition where sub-pipeline stages maintain independent state tracking.
        The copy is a point-in-time snapshot; subsequent changes to the source
        pipeline are not reflected.

        Args:
            other: Pipeline whose stages to include.

        Raises:
            PipelineConfigError: If ``other`` is ``self`` (self-include) or if
                any stage name in ``other`` already exists in this pipeline.
        """
        if other is self:
            raise PipelineConfigError(
                f"Pipeline '{self.name}' cannot include itself"
            )

        # Collect stages to add (validates all before adding any - atomic)
        stages_to_add: list[tuple[str, registry.RegistryStageInfo]] = []
        existing_names = set(self._registry.list_stages())

        for stage_name in other.list_stages():
            if stage_name in existing_names:
                raise PipelineConfigError(
                    f"Cannot include pipeline '{other.name}': "
                    f"stage '{stage_name}' already exists in '{self.name}'. "
                    f"Rename the stage using name= at registration time."
                )
            # Deep copy to prevent shared mutable state
            stage_info = copy.deepcopy(other.get(stage_name))
            stages_to_add.append((stage_name, stage_info))

        # Add all stages (only reached if no collisions)
        for stage_name, stage_info in stages_to_add:
            self._registry.add_existing(stage_info)

        if stages_to_add:
            logger.debug(
                f"Included {len(stages_to_add)} stages from pipeline "
                f"'{other.name}' into '{self.name}'"
            )
```

**Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_pipeline.py -k include -v`
Expected: All 15 tests PASS

**Step 7: Run quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: PASS

**Step 8: Commit**

```bash
jj describe -m "feat(pipeline): add include() method for pipeline composition

- Deep-copy stages to prevent shared mutable state
- Atomic operation: validates all before adding any
- Self-include raises PipelineConfigError
- Name collisions raise PipelineConfigError with fix hint
- DAG cache invalidated after successful include
- Debug logging for troubleshooting"
```

---

## Task 3: Update writing-pivot-stages Skill

**Files:**
- Modify: `skills/writing-pivot-stages/SKILL.md`

**Step 1: Add Pipeline Composition section**

Add after the "Running Stages" section (search for `## Running Stages` and add after its content, before `## Testing`):

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
main.include(preprocessing)  # Deep-copies stages, preserves state_dir
main.register(train, name="train")  # Can depend on preprocessing outputs
```

**Behavior:**
- Included stages keep their original `state_dir` (for lock files, state.db)
- Stages are deep-copied: mutations don't propagate between pipelines
- `include()` is a point-in-time snapshot; later registrations in source don't propagate
- Including empty pipeline is a no-op
- Including same pipeline twice raises (name collision)
- Transitive: if B includes C, then A includes B, A gets C's stages (already in B's registry)

**Rules:**
- Stage name collisions raise `PipelineConfigError`
- Cannot include a pipeline into itself

**Security Note:** Only include pipelines from trusted sources. Included stages execute with the same privileges as your pipeline.
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

## Task 4: Update Reference Documentation

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
- B's stages are deep-copied into A's registry
- B's stages keep their original `state_dir` (`.pivot/` in B's root)
- Lock files and state.db remain in B's directory
- The project-wide cache is shared
- Mutations to stages in A don't affect B (and vice versa)

This enables modular pipeline organization where each sub-pipeline can be developed, tested, and run independently.

### Name Collisions

If an included pipeline has a stage with the same name as an existing stage, `include()` raises `PipelineConfigError`. Rename stages at registration time to avoid collisions:

```python
sub.register(my_stage, name="sub_preprocess")  # Use unique name
main.include(sub)
```

### Semantics

- **Point-in-time snapshot:** `include()` copies stages at call time. Later registrations in the source pipeline are not reflected.
- **Atomic operation:** If any stage name collides, no stages are added (all-or-nothing).
- **Transitive:** If B includes C, then A includes B, A gets all of C's stages (they're already in B's registry when A.include(B) runs).

### Security Considerations

When including external pipelines:
- Included stages execute with the same privileges as your pipeline
- Only include pipelines from trusted sources
- Review included pipeline code before use
```

**Step 3: Commit**

```bash
jj describe -m "docs: add pipeline composition documentation"
```

---

## Task 5: Clean Up Outdated REGISTRY Comment

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

## Task 6: Final Verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/pipeline/ tests/config/test_registry.py -v`
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

| Task | Description | Files | Tests Added |
|------|-------------|-------|-------------|
| 1 | Add `StageRegistry.add_existing()` | registry.py | 2 |
| 2 | Add `Pipeline.include()` + comprehensive tests | pipeline.py, test_pipeline.py | 15 |
| 3 | Update skill | writing-pivot-stages/SKILL.md | - |
| 4 | Update docs | docs/reference/pipelines.md | - |
| 5 | Clean up REGISTRY comment | test_graph.py | - |
| 6 | Final verification | - | - |

### Key Implementation Details

**From Research:**
- Deep copy with `copy.deepcopy()` prevents shared mutable state
- Atomic validation (check all collisions before adding any)
- O(n+m) complexity using set-based collision detection
- Debug logging for troubleshooting composition issues
- Security documentation for including external pipelines
