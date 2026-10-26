# Cross-Pipeline Handle Support

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow `ArtifactHandle` objects from one `compose.Pipeline` to be passed as dependencies to stages in another `compose.Pipeline`, including only the producing stage(s) and their transitive upstream closure in the consumer's built pipeline.

**Architecture:** Subtractive fix — stop mutating `_StageNode.name` in `build()`, compute qualified names locally when emitting `RegistryStageInfo`. This makes producer qualification unconditional and deterministic. Collect upstream closure of foreign stages via DFS, emit `RegistryStageInfo` for each, and error on input-binding conflicts.

**Tech Stack:** Python 3.13+, pivot compose API, pytest

---

## Context

The design spec (`docs/plans/2026-02-14-compositional-pipeline-api.md`, line 259) states:

> Cross-pipeline references are Python imports. Artifact handles are module-level variables.

```python
# base/pipeline.py
with Pipeline("base") as p_base:
    model = train(p_base.input("data.csv", ...))

# horizon/pipeline.py
from base.pipeline import model
with Pipeline("horizon") as p_horizon:
    result = evaluate(model)  # Cross-pipeline dep
```

The validation check at `compose.py:260` rejects this with "different Pipeline instance".

Additionally, `build()` mutates `_StageNode.name` in-place (line 337), which makes cross-pipeline identity dependent on build order. This plan removes that source of complexity.

## Key Invariants

1. Pipeline names contain no `/` (enforced by `_PIPELINE_NAME_PATTERN` in `pipeline/pipeline.py:20`).
2. Bare stage names contain no `/` (derived from `func.__name__` + `@` suffixes).
3. `_StageNode.name` is **always bare** after this change. Never mutated after recording.
4. Qualified names are computed at emission: `f"{pipeline._name}/{node.name}"`.
5. Input node names are **not** qualified (confirmed by `test_build_does_not_prefix_input_identity`).
6. Legacy registry invariant: `out.identity.producer == stage_info["name"]` (enforced by `StageRegistry.add_existing`).
7. **New rule:** input binding conflicts across pipelines error unless the path is identical.

## Why "Pipeline-Level Cycles" Are Not Real Cycles

Python scoping prevents true stage-level cycles: you cannot reference `b_out` before it's defined. So the pattern:
```python
with p_a: a_out = produce()           # stage A1
with p_b: b_out = consume(data=a_out) # stage B1 depends on A1
with p_a: consume_dict(b_out)         # stage A2 depends on B1
```
produces a valid linear chain: A1 → B1 → A2. The DFS visited set prevents infinite loops; DAG-level cycle detection in `build_dag()` catches any real cycles.

## Key Functions to Know

| Function | File:Line | Role |
|----------|-----------|------|
| `_record_stage` | `compose.py:221` | Records stage call. Has the `_pipeline is not self` check to remove. |
| `_handle_to_artifact_ref` | `compose.py:69` | Converts `ArtifactHandle` → `ArtifactRef`. Currently uses bare `source.name`. |
| `build()` | `compose.py:329` | Converts compose.Pipeline → legacy Pipeline. Currently mutates `_StageNode.name` on line 337. |

## Test Helpers Already Defined (in `test_compose.py`)

```python
@stage
def _helper_produce(params: StageParams) -> pd.DataFrame: ...
@stage
def _helper_consume(data: pd.DataFrame) -> dict: ...
@stage
def _helper_consume_dict(data: dict) -> dict: ...
@stage
def _helper_consume_list(main_data: pd.DataFrame, extra_data: list[pd.DataFrame]) -> pd.DataFrame: ...
```

---

### Task 1: Remove cross-pipeline validation check

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py:258-275`
- Modify: `packages/pivot/tests/test_compose.py:1173-1184`

**Step 1: Write the failing test**

Replace `test_cross_pipeline_handle_rejected` (line 1173) with:

```python
def test_cross_pipeline_handle_accepted(tmp_path: pathlib.Path) -> None:
    """Foreign-pipeline handles are accepted at record time."""
    p1 = Pipeline("pipe1", root=tmp_path)
    with p1:
        handle = p1.input("data", path="data/raw/data.csv", t=pd.DataFrame)

    p2 = Pipeline("pipe2", root=tmp_path)
    with p2:
        _helper_consume(data=handle)

    assert len(p2._stages) == 1
    assert "data" in p2._stages[0].input_handles
    assert p2._stages[0].input_handles["data"]._pipeline is p1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_cross_pipeline_handle_accepted -xvs`
Expected: FAIL — ValueError "different Pipeline instance"

**Step 3: Remove both validation blocks**

In `_record_stage`, delete the single-handle check (lines 260-264):

```python
# DELETE these 4 lines:
                if value._pipeline is not self:
                    self._validation_errors.append(
                        f"{stage_name}: parameter '{param_name}' is a handle from a "
                        f"different Pipeline instance. All handles must come from the same pipeline."
                    )
```

And the list/tuple-handle check (lines 269-275):

```python
# DELETE these 6 lines:
                for v in value:
                    if isinstance(v, ArtifactHandle) and v._pipeline is not self:
                        self._validation_errors.append(
                            f"{stage_name}: parameter '{param_name}' is a handle from a "
                            f"different Pipeline instance. All handles must come from the same pipeline."
                        )
                        break
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_cross_pipeline_handle_accepted -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/pivot/src/pivot/compose.py packages/pivot/tests/test_compose.py
git commit -m "feat(compose): allow cross-pipeline artifact handles"
```

---

### Task 2: Make `build()` non-mutating

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py:329-418` (`build()`)
- Test: `packages/pivot/tests/test_compose.py`

**Step 1: Write the failing test**

```python
def test_build_does_not_mutate_stage_node_names(tmp_path: pathlib.Path) -> None:
    """build() must not mutate _StageNode.name — names stay bare."""
    with Pipeline("my_pipeline", root=tmp_path) as pipeline:
        _helper_produce(params=stage_def.StageParams())

    assert pipeline._stages[0].name == "_helper_produce"
    pipeline.build()
    assert pipeline._stages[0].name == "_helper_produce"  # Still bare after build
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_build_does_not_mutate_stage_node_names -xvs`
Expected: FAIL — after build, name is "my_pipeline/_helper_produce" (mutated)

**Step 3: Make build() non-mutating**

Delete lines 336-337 (the mutation loop):

```python
# DELETE these 2 lines:
        for node in self._stages:
            node.name = f"{self._name}/{node.name}"
```

In the stage-conversion loop (starts at line 339), compute the qualified name locally at the top of each iteration. Replace every `node.name` in the loop body with `qualified_name`:

```python
        for node in self._stages:
            qualified_name = f"{self._name}/{node.name}"

            func = node.original_func
            # ... (rest of loop body unchanged, but replace node.name → qualified_name)
```

There are 5 occurrences of `node.name` in the loop body to replace with `qualified_name`:
- Line 348: `_handle_to_artifact_ref(handle, node.name)` → `_handle_to_artifact_ref(handle, qualified_name)`
- Line 352: `_handle_to_artifact_ref(handle, node.name)` → `_handle_to_artifact_ref(handle, qualified_name)`
- Line 365: `ArtifactIdentity(producer=node.name, ...)` → `ArtifactIdentity(producer=qualified_name, ...)`
- Line 391: `f"Stage '{node.name}' parameter..."` → `f"Stage '{qualified_name}' parameter..."`
- Line 403: `name=node.name` → `name=qualified_name`

**Step 4: Run tests**

Run: `uv run pytest packages/pivot/tests/test_compose.py -k "prefix or does_not_mutate" -xvs`
Expected: All pass — prefixing still happens in emitted `RegistryStageInfo`, just not by mutation

**Step 5: Commit**

```bash
git add packages/pivot/src/pivot/compose.py packages/pivot/tests/test_compose.py
git commit -m "refactor(compose): make build() non-mutating — compute qualified names locally"
```

---

### Task 3: Fix `_handle_to_artifact_ref` for unconditional qualification

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py:69-117`
- Test: `packages/pivot/tests/test_compose.py`

Now that `_StageNode.name` is always bare, qualification is unconditional.

**Step 1: Write the failing test**

```python
def test_handle_to_artifact_ref_qualifies_producer_name(tmp_path: pathlib.Path) -> None:
    """_handle_to_artifact_ref qualifies stage producer with pipeline name."""
    from pivot.compose import _handle_to_artifact_ref

    p1 = Pipeline("base", root=tmp_path)
    with p1:
        result = _helper_produce(params=stage_def.StageParams())

    ref = _handle_to_artifact_ref(result, "horizon/consumer")
    assert ref.identity.producer == "base/_helper_produce"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_handle_to_artifact_ref_qualifies_producer_name -xvs`
Expected: FAIL — producer is `"_helper_produce"` (bare)

**Step 3: Fix `_handle_to_artifact_ref`**

After the `_InputNode` early return (line 78), add qualification. Replace lines 80-110:

```python
    source = handle._source
    # Qualify producer name. _StageNode.name is always bare;
    # use handle's pipeline name for both same-pipeline and cross-pipeline handles.
    qualified_name = f"{handle._pipeline._name}/{source.name}"

    if not source.output_specs:
        raise TypeError(
            f"Stage '{consumer_name}' depends on '{qualified_name}' which has no outputs "
            f"(returns None). A stage must produce outputs to be used as a dependency."
        )
    if len(source.output_specs) == 1:
        output_spec = source.output_specs[0]
        output_key = None if output_spec.key == SINGLE_OUTPUT_KEY else output_spec.key
    else:
        matched = next(
            (spec for spec in source.output_specs if spec.key == handle._output_key), None
        )
        if matched is None:
            available = [spec.key for spec in source.output_specs]
            raise TypeError(
                f"Stage '{consumer_name}' received handle from multi-output stage "
                f"'{qualified_name}' without selecting an output. "
                f"Use handle.key or handle['key']. Available: {available}"
            )
        output_spec = matched
        output_key = output_spec.key

    if isinstance(output_spec.tag, _MetricTag):
        dep_tag = types.ArtifactTag.METRIC
    elif isinstance(output_spec.tag, _PlotTag):
        dep_tag = types.ArtifactTag.PLOT
    else:
        dep_tag = types.ArtifactTag.DATA

    identity = types.ArtifactIdentity(producer=qualified_name, key=output_key)
    types.validate_artifact_identity(identity)
    return types.ArtifactRef(
        identity=identity,
        format=output_spec.format,
        python_type=handle._python_type,
        tag=dep_tag,
    )
```

Changes from original: `source.name` → `qualified_name` in exactly 3 places (identity on old line 110, error message on old line 83, error message on old line 97).

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_handle_to_artifact_ref_qualifies_producer_name -xvs`
Expected: PASS

**Step 5: Write cross-pipeline dep identity test (end-to-end)**

```python
def test_cross_pipeline_dep_identity_qualified(tmp_path: pathlib.Path) -> None:
    """Cross-pipeline dep identity uses the source pipeline's name as prefix."""
    p1 = Pipeline("base", root=tmp_path)
    with p1:
        result = _helper_produce(params=stage_def.StageParams())

    p2 = Pipeline("horizon", root=tmp_path)
    with p2:
        _helper_consume(data=result)

    legacy = p2.build()
    consumer = legacy.get("horizon/_helper_consume")
    dep = consumer["deps"]["data"]
    assert dep.identity.producer == "base/_helper_produce"
```

**Step 6: Write test that input identity stays bare**

```python
def test_cross_pipeline_input_identity_not_qualified(tmp_path: pathlib.Path) -> None:
    """Input handles from foreign pipelines keep their bare identity."""
    p1 = Pipeline("base", root=tmp_path)
    with p1:
        data = p1.input("scores", path="data/raw/scores.csv", t=pd.DataFrame)

    p2 = Pipeline("horizon", root=tmp_path)
    with p2:
        _helper_consume(data=data)

    legacy = p2.build()
    consumer = legacy.get("horizon/_helper_consume")
    dep = consumer["deps"]["data"]
    assert dep.identity.producer == "scores"  # NOT "base/scores"
```

**Step 7: Run all prefix and identity tests**

Run: `uv run pytest packages/pivot/tests/test_compose.py -k "prefix or cross_pipeline_dep or cross_pipeline_input_identity" -xvs`
Expected: All pass

**Step 8: Commit**

```bash
git add packages/pivot/src/pivot/compose.py packages/pivot/tests/test_compose.py
git commit -m "fix(compose): qualify stage producer names unconditionally in _handle_to_artifact_ref"
```

---

### Task 4: Closure collection — include only required foreign stages

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py` (add `_upstream_closure`, `_emit_stage_info`, update `build()`)
- Modify: `packages/pivot/tests/test_compose.py`

**Step 1: Add test helper stages for closure exclusion test**

Add at module level in `test_compose.py`, near other `_helper_` functions:

```python
@stage
def _helper_base_a(params: stage_def.StageParams) -> pd.DataFrame:
    return pd.DataFrame()


@stage
def _helper_base_b(data: pd.DataFrame) -> dict:
    return {"rows": len(data)}


@stage
def _helper_base_c_unrelated(params: stage_def.StageParams) -> dict:
    return {"ok": True}
```

**Step 2: Write the failing test**

```python
def test_build_includes_only_foreign_closure(tmp_path: pathlib.Path) -> None:
    """build() includes transitive foreign deps but excludes unrelated foreign stages."""
    base = Pipeline("base", root=tmp_path)
    with base:
        a = _helper_base_a(params=stage_def.StageParams())
        b = _helper_base_b(a)
        _helper_base_c_unrelated(params=stage_def.StageParams())

    consumer = Pipeline("consumer", root=tmp_path)
    with consumer:
        _helper_consume_dict(b)

    legacy = consumer.build()
    names = set(legacy.list_stages())
    assert "consumer/_helper_consume_dict" in names
    assert "base/_helper_base_b" in names       # direct dep
    assert "base/_helper_base_a" in names       # transitive dep
    assert "base/_helper_base_c_unrelated" not in names  # excluded
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_build_includes_only_foreign_closure -xvs`
Expected: FAIL — foreign stages not included

**Step 4: Add `_upstream_closure` method to Pipeline**

Add after `_validate` (around line 327), before `build`:

```python
    def _upstream_closure(self) -> list[tuple[Pipeline, _StageNode]]:
        """Collect foreign stages reachable from this pipeline's handles via DFS.

        Returns (pipeline, stage_node) pairs for foreign stages that need
        RegistryStageInfo in the built pipeline. Skips handles pointing back
        to self (those are local stages). Uses id(stage_node) for dedup.
        """
        visited = set[int]()
        result = list[tuple[Pipeline, _StageNode]]()
        stack = list[tuple[Pipeline, _StageNode]]()

        # Seed: foreign stage handles from this pipeline's stages
        for node in self._stages:
            for handle in node.input_handles.values():
                if isinstance(handle._source, _StageNode) and handle._pipeline is not self:
                    stack.append((handle._pipeline, handle._source))
            for handles in node.list_input_handles.values():
                for handle in handles:
                    if isinstance(handle._source, _StageNode) and handle._pipeline is not self:
                        stack.append((handle._pipeline, handle._source))

        while stack:
            pipeline, stage_node = stack.pop()
            node_id = id(stage_node)
            if node_id in visited:
                continue
            visited.add(node_id)
            result.append((pipeline, stage_node))

            # Follow this foreign stage's own handles — skip handles back to self
            for handle in stage_node.input_handles.values():
                if isinstance(handle._source, _StageNode) and handle._pipeline is not self:
                    stack.append((handle._pipeline, handle._source))
            for handles in stage_node.list_input_handles.values():
                for handle in handles:
                    if isinstance(handle._source, _StageNode) and handle._pipeline is not self:
                        stack.append((handle._pipeline, handle._source))

        return result
```

**Step 5: Extract `_emit_stage_info` static method**

Extract the stage-conversion logic from `build()`'s loop into a reusable method. Add as a `@staticmethod` on Pipeline:

```python
    @staticmethod
    def _emit_stage_info(
        node: _StageNode,
        qualified_name: str,
        state_dir: pathlib.Path,
    ) -> registry.RegistryStageInfo:
        """Convert a _StageNode into a RegistryStageInfo."""
        func = node.original_func
        assert not hasattr(func, "_is_stage")
        if getattr(node.func, "__pivot_no_fingerprint__", False):
            func.__pivot_no_fingerprint__ = True  # pyright: ignore[reportFunctionMemberAccess]

        deps = dict[str, types.ArtifactRef]()
        for param_name, handle in node.input_handles.items():
            deps[param_name] = _handle_to_artifact_ref(handle, qualified_name)
        for param_name, handles in node.list_input_handles.items():
            for i, handle in enumerate(handles):
                deps[f"{param_name}[{i}]"] = _handle_to_artifact_ref(handle, qualified_name)

        outs = list[types.ArtifactRef]()
        for output_spec in node.output_specs:
            output_key = None if output_spec.key == SINGLE_OUTPUT_KEY else output_spec.key
            if isinstance(output_spec.tag, _MetricTag):
                out_tag = types.ArtifactTag.METRIC
            elif isinstance(output_spec.tag, _PlotTag):
                out_tag = types.ArtifactTag.PLOT
            else:
                out_tag = types.ArtifactTag.DATA
            identity = types.ArtifactIdentity(producer=qualified_name, key=output_key)
            types.validate_artifact_identity(identity)
            outs.append(
                types.ArtifactRef(
                    identity=identity,
                    format=output_spec.format,
                    python_type=output_spec.python_type,
                    tag=out_tag,
                )
            )

        params_arg_name: str | None = None
        hints = get_type_hints(func, include_extras=True)
        for name, hint in hints.items():
            if name == "return":
                continue
            base_hint = hint
            if get_origin(base_hint) is Annotated:
                base_hint = get_args(base_hint)[0]
            origin = get_origin(base_hint)
            if origin is _stdlib_types.UnionType or origin is typing.Union:  # pyright: ignore[reportDeprecated] - runtime identity check
                for union_arg in get_args(base_hint):
                    if isinstance(union_arg, type) and issubclass(
                        union_arg, stage_def.StageParams
                    ):
                        raise TypeError(
                            f"Stage '{qualified_name}' parameter '{name}' has type "
                            f"'{base_hint}' — StageParams must not be in a union. "
                            f"Use 'params: {union_arg.__name__}' directly, with a "
                            f"default if needed (params: {union_arg.__name__} = "
                            f"{union_arg.__name__}())."
                        )
            if isinstance(base_hint, type) and issubclass(base_hint, stage_def.StageParams):
                params_arg_name = name
                break

        return registry.RegistryStageInfo(
            func=func,
            name=qualified_name,
            deps=deps,
            outs=outs,
            params=node.params,
            mutex=[],
            variant=node.variant,
            signature=inspect.signature(func),
            fingerprint=None,
            params_arg_name=params_arg_name,
            state_dir=state_dir,
            collection_params={k: str(v) for k, v in node.collection_params.items()},
        )
```

**Step 6: Rewrite `build()` to use the helpers**

Replace the body of `build()`:

```python
    def build(self) -> pipeline_mod.Pipeline:
        if self._built:
            raise RuntimeError("Pipeline already built")
        self._validate()
        legacy = pipeline_mod.Pipeline(self._name, root=self._root)
        legacy.set_input_bindings({name: node.path for name, node in self._inputs.items()})

        # Emit foreign upstream stages (closure-only, not entire foreign pipelines).
        for foreign_pipeline, foreign_node in self._upstream_closure():
            qualified_name = f"{foreign_pipeline._name}/{foreign_node.name}"
            stage_info = self._emit_stage_info(
                foreign_node, qualified_name, foreign_pipeline._root / ".pivot"
            )
            legacy._registry.add_existing(stage_info)

        # Collect and merge foreign input bindings (Task 5).
        self._merge_foreign_input_bindings(legacy)

        # Emit this pipeline's own stages.
        for node in self._stages:
            qualified_name = f"{self._name}/{node.name}"
            stage_info = self._emit_stage_info(node, qualified_name, self._root / ".pivot")
            legacy._registry.add_existing(stage_info)

        self._built = True
        return legacy
```

Add the placeholder for Task 5:

```python
    def _merge_foreign_input_bindings(self, legacy: pipeline_mod.Pipeline) -> None:
        """Merge foreign input bindings into legacy pipeline. Implemented in Task 5."""
        pass
```

**Step 7: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_build_includes_only_foreign_closure -xvs`
Expected: PASS

**Step 8: Run all existing tests**

Run: `uv run pytest packages/pivot/tests/test_compose.py -xvs`
Expected: All pass

**Step 9: Commit**

```bash
git add packages/pivot/src/pivot/compose.py packages/pivot/tests/test_compose.py
git commit -m "feat(compose): collect upstream closure and emit foreign stages in build()"
```

---

### Task 5: Input-binding merge with conflict detection

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py` (`_merge_foreign_input_bindings`)
- Test: `packages/pivot/tests/test_compose.py`

**Step 1: Write test — foreign input bindings propagated**

```python
def test_cross_pipeline_input_bindings_propagated(tmp_path: pathlib.Path) -> None:
    """Input bindings from foreign pipelines are propagated to the consumer."""
    p1 = Pipeline("base", root=tmp_path)
    with p1:
        data = p1.input("external_data", path="data/external/scores.csv", t=pd.DataFrame)
        result = _helper_consume(data=data)

    p2 = Pipeline("horizon", root=tmp_path)
    with p2:
        _helper_consume_dict(result)

    legacy = p2.build()
    assert "external_data" in legacy.input_bindings
    assert legacy.input_bindings["external_data"] == "data/external/scores.csv"
```

**Step 2: Write test — same name, same path is OK**

```python
def test_cross_pipeline_input_binding_same_path_ok(tmp_path: pathlib.Path) -> None:
    """Same input name with identical path across pipelines is not a conflict."""
    p1 = Pipeline("base", root=tmp_path)
    with p1:
        data1 = p1.input("raw", path="data/raw/shared.csv", t=pd.DataFrame)
        result = _helper_consume(data=data1)

    p2 = Pipeline("consumer", root=tmp_path)
    with p2:
        _ = p2.input("raw", path="data/raw/shared.csv", t=pd.DataFrame)
        _helper_consume_dict(result)

    legacy = p2.build()  # Should not raise
    assert legacy.input_bindings["raw"] == "data/raw/shared.csv"
```

**Step 3: Write test — conflicting bindings error**

```python
def test_cross_pipeline_input_binding_conflict_raises(tmp_path: pathlib.Path) -> None:
    """Different paths for same input name across pipelines raise a clear error."""
    p1 = Pipeline("base", root=tmp_path)
    with p1:
        data1 = p1.input("raw", path="data/raw/base.csv", t=pd.DataFrame)
        result = _helper_consume(data=data1)

    p2 = Pipeline("consumer", root=tmp_path)
    with p2:
        _ = p2.input("raw", path="data/raw/consumer.csv", t=pd.DataFrame)
        _helper_consume_dict(result)

    with pytest.raises(ValueError, match="Input binding conflict.*raw"):
        p2.build()
```

**Step 4: Run tests to verify they fail**

Run: `uv run pytest packages/pivot/tests/test_compose.py -k "input_binding" -xvs`
Expected: Propagation test FAILS (bindings not merged yet)

**Step 5: Implement `_merge_foreign_input_bindings`**

Replace the placeholder:

```python
    def _merge_foreign_input_bindings(self, legacy: pipeline_mod.Pipeline) -> None:
        """Collect input bindings from foreign stages in the closure and merge.

        Raises ValueError if a foreign input name conflicts with a local binding
        (same name, different path).
        """
        foreign_bindings = dict[str, str]()

        for _foreign_pipeline, foreign_node in self._upstream_closure():
            for handle in foreign_node.input_handles.values():
                if isinstance(handle._source, _InputNode):
                    inp = handle._source
                    if inp.name in foreign_bindings and foreign_bindings[inp.name] != inp.path:
                        raise ValueError(
                            f"Input binding conflict: input '{inp.name}' is bound to "
                            f"'{foreign_bindings[inp.name]}' and '{inp.path}' "
                            f"in different pipelines."
                        )
                    foreign_bindings[inp.name] = inp.path
            for handles in foreign_node.list_input_handles.values():
                for handle in handles:
                    if isinstance(handle._source, _InputNode):
                        inp = handle._source
                        if inp.name in foreign_bindings and foreign_bindings[inp.name] != inp.path:
                            raise ValueError(
                                f"Input binding conflict: input '{inp.name}' is bound to "
                                f"'{foreign_bindings[inp.name]}' and '{inp.path}' "
                                f"in different pipelines."
                            )
                        foreign_bindings[inp.name] = inp.path

        # Merge into legacy, checking for conflicts with local bindings
        local_bindings = legacy.input_bindings
        for name, path in foreign_bindings.items():
            if name in local_bindings and local_bindings[name] != path:
                raise ValueError(
                    f"Input binding conflict: input '{name}' is bound to "
                    f"'{local_bindings[name]}' locally and '{path}' in a foreign pipeline."
                )
            local_bindings[name] = path
        legacy.set_input_bindings(local_bindings)
```

**Step 6: Run tests**

Run: `uv run pytest packages/pivot/tests/test_compose.py -k "input_binding" -xvs`
Expected: All pass

**Step 7: Commit**

```bash
git add packages/pivot/src/pivot/compose.py packages/pivot/tests/test_compose.py
git commit -m "feat(compose): merge foreign input bindings with conflict detection"
```

---

### Task 6: Edge-case tests

**Files:**
- Test: `packages/pivot/tests/test_compose.py`

**Step 1: Write test — transitive A→B→C across three pipelines**

```python
def test_cross_pipeline_transitive_three_pipelines(tmp_path: pathlib.Path) -> None:
    """A→B→C transitive cross-pipeline dependencies resolve correctly."""
    p_a = Pipeline("a", root=tmp_path)
    with p_a:
        a_out = _helper_produce(params=stage_def.StageParams())

    p_b = Pipeline("b", root=tmp_path)
    with p_b:
        b_out = _helper_consume(data=a_out)

    p_c = Pipeline("c", root=tmp_path)
    with p_c:
        _helper_consume_dict(b_out)

    legacy = p_c.build()
    stage_names = legacy.list_stages()
    assert "a/_helper_produce" in stage_names
    assert "b/_helper_consume" in stage_names
    assert "c/_helper_consume_dict" in stage_names
```

**Step 2: Run test**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_cross_pipeline_transitive_three_pipelines -xvs`
Expected: PASS

**Step 3: Write test — list handles across pipelines**

```python
def test_cross_pipeline_list_handles(tmp_path: pathlib.Path) -> None:
    """list[ArtifactHandle] from foreign pipelines are included and resolved."""
    p1 = Pipeline("base", root=tmp_path)
    with p1:
        a = _helper_produce(params=stage_def.StageParams())

    p2 = Pipeline("consumer", root=tmp_path)
    with p2:
        local_input = p2.input("local", path="data/raw/local.csv", t=pd.DataFrame)
        _helper_consume_list(main_data=local_input, extra_data=[a, a])

    legacy = p2.build()
    assert "base/_helper_produce" in legacy.list_stages()
    assert "consumer/_helper_consume_list" in legacy.list_stages()

    consumer_info = legacy.get("consumer/_helper_consume_list")
    assert consumer_info["deps"]["extra_data[0]"].identity.producer == "base/_helper_produce"
```

**Step 4: Run test**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_cross_pipeline_list_handles -xvs`
Expected: PASS

**Step 5: Write test — foreign stages use their pipeline's state_dir**

```python
def test_cross_pipeline_foreign_stage_uses_own_state_dir(tmp_path: pathlib.Path) -> None:
    """Foreign stages keep their own pipeline's state_dir, not the consumer's."""
    base_root = tmp_path / "base"
    base_root.mkdir()
    horizon_root = tmp_path / "horizon"
    horizon_root.mkdir()

    p1 = Pipeline("base", root=base_root)
    with p1:
        result = _helper_produce(params=stage_def.StageParams())

    p2 = Pipeline("horizon", root=horizon_root)
    with p2:
        _helper_consume(data=result)

    legacy = p2.build()
    base_stage = legacy.get("base/_helper_produce")
    horizon_stage = legacy.get("horizon/_helper_consume")
    assert base_stage["state_dir"] == base_root / ".pivot"
    assert horizon_stage["state_dir"] == horizon_root / ".pivot"
```

**Step 6: Run test**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_cross_pipeline_foreign_stage_uses_own_state_dir -xvs`
Expected: PASS

**Step 7: Write test — "pipeline-level cycle" is a valid chain**

```python
def test_pipeline_level_cycle_is_valid_chain(tmp_path: pathlib.Path) -> None:
    """A→B→A at pipeline level is a linear stage chain, not a cycle."""
    p_a = Pipeline("alpha", root=tmp_path)
    p_b = Pipeline("beta", root=tmp_path)

    with p_a:
        a_out = _helper_produce(params=stage_def.StageParams())

    with p_b:
        b_out = _helper_consume(data=a_out)

    with p_a:
        _helper_consume_dict(b_out)

    # The stage graph is: A1 → B1 → A2 (linear chain, not a cycle)
    legacy = p_a.build()
    assert "alpha/_helper_produce" in legacy.list_stages()
    assert "beta/_helper_consume" in legacy.list_stages()
    assert "alpha/_helper_consume_dict" in legacy.list_stages()
```

**Step 8: Run test**

Run: `uv run pytest packages/pivot/tests/test_compose.py::test_pipeline_level_cycle_is_valid_chain -xvs`
Expected: PASS

**Step 9: Commit**

```bash
git add packages/pivot/tests/test_compose.py
git commit -m "test(compose): add transitive, list, state_dir, and pipeline-cycle edge case tests"
```

---

### Task 7: Full test suite and quality checks

**Step 1: Run the full compose test suite**

Run: `uv run pytest packages/pivot/tests/test_compose.py -xvs`
Expected: All tests pass

**Step 2: Run the pipeline test suite**

Run: `uv run pytest packages/pivot/tests/pipeline/test_pipeline.py -xvs`
Expected: All tests pass

**Step 3: Run the full test suite**

Run: `uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto`
Expected: All tests pass

**Step 4: Run quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: No errors or warnings

**Step 5: Fix any issues found**

Address type errors, lint issues, or test failures.

**Step 6: Commit if fixes were needed**

```bash
git add -A
git commit -m "fix: address quality check issues from cross-pipeline handle support"
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Non-mutating build | Eliminates build-order sensitivity. `_StageNode.name` stays bare and stable. No `startswith()` guards, no idempotent caching, no `_BuildState` enum. |
| Closure-only inclusion | Only reachable upstream stages emitted, not entire foreign pipelines. Unrelated stages excluded. |
| DFS with `id(stage_node)` | Standard dedup. True stage-level cycles impossible (Python scoping prevents forward refs). Pipeline-level cycles produce valid linear chains. |
| `_emit_stage_info` static method | Extracted from `build()` loop to DRY the conversion for both local and foreign stages. |
| Input-binding conflict = error | Silent merge (first-wins) hides bugs. Same-path is OK; different-path errors immediately. |
| Foreign `state_dir` = foreign `_root / ".pivot"` | Lock files must live in the producing pipeline's state directory. |
| No `_BuildState` / build caching | Not needed without recursive `foreign.build()`. Simple `_built` flag suffices. |

## Non-Goals

- Do **not** modify the legacy `pipeline.pipeline.Pipeline` class.
- Do **not** include entire foreign pipelines; only the required closure.
- Do **not** add build caching or `_BuildState` tracking — unnecessary without recursive build.
