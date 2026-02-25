# Always-Prefix Stage Names Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stage names are always `{pipeline_name}/{bare_name}` from creation. This makes cross-pipeline dep resolution trivially correct, eliminates collision-based renaming in `include()`, and removes an entire class of identity-mismatch bugs.

**Architecture:** Prefix `_StageNode.name` during `Pipeline.build()` -- before identities or `RegistryStageInfo` are created, so all downstream references inherit the prefix automatically. Input nodes (external resources) keep bare names. `include()` becomes a simple deep-copy -- no collision detection, no rename_map, no identity rewriting. Display layer strips the prefix in single-pipeline mode.

**Tech Stack:** Python, pytest

**Breaking changes (pre-alpha, acceptable):**
- Lock file names change: `train.lock` -> `horizon/train.lock`
- StateDB keys change: `dep:train:...` -> `dep:horizon/train:...`
- Fingerprint manifest cache keys change
- One-time full re-run on upgrade

---

### Task 1: Add `add_existing()` invariant for output identity consistency

The registry should reject stages where `out.identity.producer != stage_info["name"]`. This structural invariant prevents identity drift regardless of how stages are created.

**Files:**
- Modify: `packages/pivot/src/pivot/registry.py` -- `add_existing()` method
- Test: `packages/pivot/tests/pipeline/test_pipeline.py`

**Step 1: Write the failing test**

Add helpers and the test to `test_pipeline.py`:

```python
import inspect

from pivot import loaders, registry, types


def _make_stage_info(
    name: str,
    deps: dict[str, types.ArtifactRef] | None = None,
    outs: list[types.ArtifactRef] | None = None,
) -> registry.RegistryStageInfo:
    def _fn() -> None: ...
    if outs is None:
        outs = [
            types.ArtifactRef(
                identity=types.ArtifactIdentity(producer=name, key="out"),
                format=loaders.YAML(),
                python_type=dict,
                tag=types.ArtifactTag.DATA,
            )
        ]
    return registry.RegistryStageInfo(
        func=_fn, name=name, deps=deps or {}, outs=outs,
        params=None, mutex=[], variant=None, signature=inspect.signature(_fn),
        fingerprint=None, params_arg_name=None, state_dir=None, collection_params={},
    )


def test_add_existing_rejects_mismatched_out_producer(tmp_path: pathlib.Path) -> None:
    pipeline = pipeline_mod.Pipeline("test", root=tmp_path)
    bad_out = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="wrong", key="out"),
        format=loaders.YAML(), python_type=dict, tag=types.ArtifactTag.DATA,
    )
    info = _make_stage_info("my_stage", outs=[bad_out])
    with pytest.raises(ValueError, match="producer.*wrong.*my_stage"):
        pipeline._registry.add_existing(info)
```

**Step 2:** Run test -> FAIL (no validation exists)

**Step 3: Implement the invariant in `add_existing()`**

```python
for out in stage_info["outs"]:
    if out.identity.producer != stage_info["name"]:
        raise ValueError(
            f"Stage '{stage_info['name']}' has output with producer "
            f"'{out.identity.producer}' -- must match stage name"
        )
```

**Step 4:** Run test -> PASS

**Step 5:** Commit: `feat(registry): add invariant -- out.identity.producer must equal stage name`

---

### Task 2: Prefix stage names in `compose.py:build()` and update tests

The core change. Before the build loop iterates stages, prefix each `_StageNode.name` with `{pipeline_name}/`. All downstream code -- `_handle_to_artifact_ref`, identity creation, `RegistryStageInfo` -- automatically inherits the prefix.

Input nodes (`_InputNode`) are NOT prefixed -- they represent external resources.

Guard against double-prefixing: `build()` mutates `node.name` in-place, so calling `build()` twice on the same `Pipeline` object would double the prefix. Add a `self._built` flag to prevent this.

**Files:**
- Modify: `packages/pivot/src/pivot/compose.py` -- `build()` method
- Modify: `packages/pivot/tests/test_compose.py` -- new tests + update existing assertions
- Modify: any other test files that break (conftest.py, etc.)

**Step 1: Write the failing tests**

```python
def test_build_prefixes_stage_names_with_pipeline_name(tmp_path: pathlib.Path) -> None:
    with Pipeline("my_pipeline", root=tmp_path) as pipeline:
        _helper_produce(params=stage_def.StageParams())

    legacy = pipeline.build()
    assert "my_pipeline/_helper_produce" in legacy.list_stages()
    assert "_helper_produce" not in legacy.list_stages()


def test_build_prefixes_output_identity_producer(tmp_path: pathlib.Path) -> None:
    with Pipeline("my_pipeline", root=tmp_path) as pipeline:
        _helper_produce(params=stage_def.StageParams())

    legacy = pipeline.build()
    info = legacy.get("my_pipeline/_helper_produce")
    for out in info["outs"]:
        assert out.identity.producer == "my_pipeline/_helper_produce"


def test_build_prefixes_dep_identity_producer(tmp_path: pathlib.Path) -> None:
    with Pipeline("my_pipeline", root=tmp_path) as pipeline:
        result = _helper_produce(params=stage_def.StageParams())
        _helper_consume(data=result)

    legacy = pipeline.build()
    consumer = legacy.get("my_pipeline/_helper_consume")
    dep = consumer["deps"]["data"]
    assert dep.identity.producer == "my_pipeline/_helper_produce"


def test_build_does_not_prefix_input_identity(tmp_path: pathlib.Path) -> None:
    with Pipeline("my_pipeline", root=tmp_path) as pipeline:
        inp = pipeline.input("data.csv", path="data/raw/data.csv", t=pd.DataFrame)
        _helper_consume(data=inp)

    legacy = pipeline.build()
    consumer = legacy.get("my_pipeline/_helper_consume")
    dep = consumer["deps"]["data"]
    assert dep.identity.producer == "data.csv"  # NOT "my_pipeline/data.csv"
```

**Step 2:** Run tests -> FAIL (bare names still used)

**Step 3: Add prefix in `build()`**

At the top of the `build()` method, before the `for node in self._stages:` loop:

```python
for node in self._stages:
    node.name = f"{self._name}/{node.name}"
```

This single line is the core change. Everything else (`_handle_to_artifact_ref`, identity creation, `RegistryStageInfo`) reads `node.name` and gets the prefixed value.

**Step 4: Update existing tests for prefixed names**

Run `uv run pytest packages/pivot/tests -n auto` and fix failures. Many existing tests in `test_compose.py` assert bare stage names. Update them to expect `{pipeline_name}/{stage_name}`.

Common patterns to update:
- `legacy.get("_helper_produce")` -> `legacy.get("test/_helper_produce")`
- `assert node.name == "_helper_produce"` -> kept as-is (node names are still bare before `build()`)
- `assert stage_info["name"] == "..."` -> prefix added

**Step 5:** Run full suite -> ALL PASS

**Step 6:** Commit: `refactor(compose): always prefix stage names with pipeline_name/ at build time`

---

### Task 3: Simplify `include()` and `resolve_external_dependencies()` -- remove collision logic

With always-prefixed names, neither `include()` nor `resolve_external_dependencies()` needs collision detection or renaming. Both have the same bug (rename stage name without updating identity references), and both become trivial with prefixed names.

**Files:**
- Modify: `packages/pivot/src/pivot/pipeline/pipeline.py` -- `include()` and `resolve_external_dependencies()`
- Test: `packages/pivot/tests/pipeline/test_pipeline.py`

**Step 1: Write tests for the simplified include**

```python
def test_include_preserves_prefixed_names(tmp_path: pathlib.Path) -> None:
    child = pipeline_mod.Pipeline("child", root=tmp_path)
    child._registry.add_existing(_make_stage_info("child/alpha"))

    parent = pipeline_mod.Pipeline("parent", root=tmp_path)
    parent._registry.add_existing(_make_stage_info("parent/beta"))
    parent.include(child)

    assert "child/alpha" in parent.list_stages()
    assert "parent/beta" in parent.list_stages()


def test_include_no_collision_with_same_bare_name(tmp_path: pathlib.Path) -> None:
    a = pipeline_mod.Pipeline("a", root=tmp_path)
    a._registry.add_existing(_make_stage_info("a/train"))

    b = pipeline_mod.Pipeline("b", root=tmp_path)
    b._registry.add_existing(_make_stage_info("b/train"))

    combined = pipeline_mod.Pipeline("all", root=tmp_path)
    combined.include(a)
    combined.include(b)

    assert "a/train" in combined.list_stages()
    assert "b/train" in combined.list_stages()


def test_include_dep_identities_need_no_rewriting(tmp_path: pathlib.Path) -> None:
    child = pipeline_mod.Pipeline("child", root=tmp_path)
    dep_ref = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="child/upstream", key="out"),
        format=loaders.YAML(), python_type=dict, tag=types.ArtifactTag.DATA,
    )
    child._registry.add_existing(_make_stage_info("child/upstream"))
    child._registry.add_existing(_make_stage_info("child/downstream", deps={"data": dep_ref}))

    combined = pipeline_mod.Pipeline("all", root=tmp_path)
    combined.include(child)

    downstream = combined.get("child/downstream")
    assert downstream["deps"]["data"].identity.producer == "child/upstream"
```

**Step 2:** Run tests -> PASS

**Step 3: Simplify `include()`**

Replace the entire collision-detection + rename_map logic with:

```python
def include(self, other: Pipeline) -> None:
    if other is self:
        raise PipelineConfigError(f"Pipeline '{self.name}' cannot include itself")

    for stage_name in other.list_stages():
        stage_info = copy.deepcopy(other.get(stage_name))
        self._registry.add_existing(stage_info)

    for name, path in other.input_bindings.items():
        if name not in self._input_bindings:
            self._input_bindings[name] = path

    if other.list_stages():
        self._dag_built = False
```

~40 lines of collision logic replaced by ~10 lines.

Also simplify `resolve_external_dependencies()` (line 434-438) which has the same collision-based rename pattern:

```python
# Before (buggy — renames stage but not identity references):
if stage_info["name"] in self._registry.list_stages():
    stage_info["name"] = f"{source_pipeline_name}/{stage_info['name']}"
self._registry.add_existing(stage_info)

# After (names are already prefixed by build(), collision impossible):
self._registry.add_existing(stage_info)
```

Remove the `if` check and the `source_pipeline_name` variable entirely.

**Step 4:** Run full suite -> ALL PASS

**Step 5:** Commit: `refactor(pipeline): simplify include() and resolve_external_dependencies() -- prefixed names eliminate collision logic`

---

### Task 4: Drop `_pipeline_name` from `WorkspaceStore` output paths

Now that stage names contain the pipeline prefix (e.g., `horizon/train`), the `WorkspaceStore._pipeline_name` is redundant in output path resolution. Without this change, single-pipeline mode produces doubled names: `data/horizon/horizon/train/output.csv`.

**Files:**
- Modify: `packages/pivot/src/pivot/storage/store.py` -- `_resolve_output_path()`
- Modify: `packages/pivot/src/pivot/storage/store.py` -- constructor (remove or deprecate `_pipeline_name`)
- Test: existing store tests + new test for prefixed producer paths

**Step 1: Write the failing test**

```python
def test_resolve_output_path_uses_prefixed_producer(tmp_path: pathlib.Path) -> None:
    store = WorkspaceStore(tmp_path, "ignored", {})
    ref = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="horizon/train", key="output"),
        format=loaders.CSV(), python_type=pd.DataFrame, tag=types.ArtifactTag.DATA,
    )
    path = store._resolve_output_path(ref)
    assert path == tmp_path / "data" / "horizon" / "train" / "output.csv"
    assert "ignored" not in str(path)  # pipeline_name should not appear
```

**Step 2:** Run test -> FAIL (currently produces `data/ignored/horizon/train/output.csv`)

**Step 3: Remove `_pipeline_name` from path construction**

In `_resolve_output_path`, change:
```python
rel = pathlib.Path(prefix) / self._pipeline_name / stage_name / f"{key}.{ext}"
```
to:
```python
rel = pathlib.Path(prefix) / stage_name / f"{key}.{ext}"
```

Apply the same change to all path patterns in `_resolve_output_path` (key=None, DIRECTORY, etc.).

Update `_resolve_input_path`, `_is_input_ref`, and `presentation.py` if they reference `_pipeline_name`.

**Step 4:** Fix any tests that assert old path format. Run full suite -> ALL PASS

**Step 5:** Commit: `refactor(store): drop _pipeline_name from output paths -- stage prefix is sufficient`

---

### Task 5: Display and CLI input -- strip prefix in single-pipeline mode

When running a single pipeline, the `{pipeline_name}/` prefix is noise in both output AND input. Users should see `train` and type `train`, not `horizon/train`.

This covers two directions:
- **Output**: stage names in status, TUI, explain, metrics
- **Input**: CLI target resolution (`pivot repro train`), tab completion, `--force` stage selection

**Files:**
- Modify: `packages/pivot/src/pivot/engine/sinks.py` -- stage name formatting functions
- Modify: `packages/pivot/src/pivot/cli/targets.py:88` -- resolve bare names to prefixed names
- Modify: `packages/pivot/src/pivot/cli/completion.py` or wherever tab completion is generated
- Modify: `packages/pivot-tui/src/pivot_tui/run.py` and widgets
- Modify: `packages/pivot/src/pivot/show/metrics.py`
- Modify: `packages/pivot/src/pivot/cli/repro.py` -- explain output, `--force` handling
- Test: relevant test files

**Step 1:** Add display and resolution helpers:

```python
def display_stage_name(stage_name: str, strip_prefix: str | None) -> str:
    if strip_prefix and stage_name.startswith(f"{strip_prefix}/"):
        return stage_name[len(strip_prefix) + 1:]
    return stage_name

def resolve_stage_name(name: str, all_stages: Mapping[str, Any]) -> str:
    """Resolve a possibly bare stage name to its prefixed form."""
    if name in all_stages:
        return name
    matches = [s for s in all_stages if s.endswith(f"/{name}")]
    if len(matches) == 1:
        return matches[0]
    return name  # Fall through to normal error handling
```

When only one pipeline is active, pass its name as `strip_prefix`. When multiple (`--all`), pass `None`.

**Step 2:** Apply the display helper at each output site. Apply the resolution helper in `targets.py:resolve_cli_target` and `repro.py --force` handling.

**Step 3:** Update tab completion to offer bare names in single-pipeline mode.

**Step 4:** Run quality checks and full suite.

**Step 5:** Commit: `feat(cli): strip pipeline prefix in single-pipeline display and accept bare names in input`

---

### Task 6: Verify with eval-pipeline

**Step 1:** Check dep resolution in `--all` mode:

```bash
cd /home/sami/eval-pipeline/pivot && uv run --active python3 -c "
from pivot.discovery import _discover_all_pipelines
from pivot import types
import pathlib

combined = _discover_all_pipelines(pathlib.Path('.'))
for name in sorted(combined.list_stages()):
    if 'benchmark' in name.lower():
        info = combined.get(name)
        for dep_name, ref in info['deps'].items():
            if 'filtered' in dep_name.lower():
                print(f'{name} -> {dep_name}: producer={ref.identity.producer}')
"
```

Expected: each `generate_benchmark_results` depends on its own pipeline's `compute_task_weights`.

**Step 2:** Dry-run: `pivot repro --all --dry-run | head -30`
