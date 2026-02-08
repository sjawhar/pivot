# Graph Model Adapter: Decouple Renderer from Engine Node Encoding


**Goal:** Decouple `dag/render.py` from engine-specific string-encoded node IDs (`parse_node`, `NodeType`) by introducing a `GraphView` adapter that exposes nodes and edges as plain data.

**Architecture:** Add an `extract_graph_view()` function to `engine/graph.py` that converts the bipartite `nx.DiGraph[str]` into a `GraphView` TypedDict containing pre-extracted stage names, artifact paths, and edges. The renderer consumes `GraphView` instead of walking the raw NetworkX graph with `parse_node`. The bipartite graph and `parse_node` remain internal to `engine/graph.py`—only `GraphView` crosses the boundary.

**Tech Stack:** NetworkX (existing), TypedDict for the adapter contract

---

## Current State

`dag/render.py` directly imports:
- `engine_graph.parse_node` — to decode `"stage:train"` → `(NodeType.STAGE, "train")`
- `engine_types.NodeType` — to filter nodes by type

The coupling is in `_extract_nodes_and_edges()` (render.py:56-110), which walks every node and edge in the bipartite graph, calling `parse_node` on each to determine type and extract labels.

## Target State

`dag/render.py` receives a `GraphView` (TypedDict with `stages`, `artifacts`, `stage_edges`, `artifact_edges` lists) and renders from that — no imports from `engine/graph.py` or `engine/types.py`.

The `extract_graph_view()` function lives in `engine/graph.py` and encapsulates the `parse_node`/`NodeType` logic. `dag/render.py` only imports and consumes `GraphView`.

**Edge direction semantics (documented in docstrings):**
- `stage_edges`: `(producer, consumer)` — data-flow direction (matches the bipartite graph's stage→artifact→stage flow)
- `artifact_edges`: `(input, output)` — data-flow direction (artifact consumed by stage that produces another artifact)

---

## Task 1: Add `GraphView` TypedDict and `extract_graph_view()` to `engine/graph.py`

**Files:**
- Modify: `src/pivot/engine/graph.py`
- Test: `tests/engine/test_graph.py`

### Step 1: Write failing tests for `extract_graph_view`

Add to `tests/engine/test_graph.py`:

```python
# --- extract_graph_view tests ---


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_empty() -> None:
    """extract_graph_view on empty graph returns empty lists."""
    g = graph.build_graph({})
    view = graph.extract_graph_view(g)

    assert view["stages"] == []
    assert view["artifacts"] == []
    assert view["stage_edges"] == []
    assert view["artifact_edges"] == []


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_single_stage(tmp_path: Path) -> None:
    """extract_graph_view extracts stage and artifact from single-stage graph."""
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert view["stages"] == ["stage_a"]
    assert set(view["artifacts"]) == {str(input_file), str(output_file)}
    # Single stage with no downstream — no stage edges
    assert view["stage_edges"] == []
    # Artifact edges: input -> output (through stage_a)
    assert (str(input_file), str(output_file)) in view["artifact_edges"]


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_linear_chain(tmp_path: Path) -> None:
    """extract_graph_view extracts correct edges for a linear chain."""
    input_file = tmp_path / "input.csv"
    intermediate = tmp_path / "intermediate.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(intermediate)]),
        "stage_b": _create_stage("stage_b", [str(intermediate)], [str(output_file)]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert set(view["stages"]) == {"stage_a", "stage_b"}
    assert set(view["artifacts"]) == {str(input_file), str(intermediate), str(output_file)}
    # stage_a -> stage_b (producer -> consumer, data-flow direction)
    assert ("stage_a", "stage_b") in view["stage_edges"]
    # artifact edges: input -> intermediate, intermediate -> output
    assert (str(input_file), str(intermediate)) in view["artifact_edges"]
    assert (str(intermediate), str(output_file)) in view["artifact_edges"]


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_diamond(tmp_path: Path) -> None:
    """extract_graph_view handles diamond DAG correctly."""
    input_file = tmp_path / "input.csv"
    clean = tmp_path / "clean.csv"
    feats = tmp_path / "feats.csv"
    model = tmp_path / "model.pkl"
    input_file.touch()

    stages = {
        "preprocess": _create_stage("preprocess", [str(input_file)], [str(clean)]),
        "features": _create_stage("features", [str(input_file)], [str(feats)]),
        "train": _create_stage("train", [str(clean), str(feats)], [str(model)]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert set(view["stages"]) == {"preprocess", "features", "train"}
    # Stage edges (producer -> consumer)
    assert ("preprocess", "train") in view["stage_edges"]
    assert ("features", "train") in view["stage_edges"]
```

### Step 2: Run tests to verify they fail

```bash
cd /home/sami/pivot/roadmap-380 && uv run pytest tests/engine/test_graph.py -k "extract_graph_view" -v
```

Expected: FAIL — `AttributeError: module 'pivot.engine.graph' has no attribute 'extract_graph_view'`

### Step 3: Add `GraphView` TypedDict and `extract_graph_view()` implementation

In `src/pivot/engine/graph.py`, add the TypedDict near the top (after imports, before existing functions) and the function after `get_stage_dag`:

```python
class GraphView(TypedDict):
    """Pre-extracted graph data for rendering.

    Decouples renderers from the internal bipartite graph representation.
    All node identifiers are plain strings (stage names, artifact paths)
    with no encoding prefixes.

    Edge direction is data-flow: producer → consumer / input → output.
    """

    stages: list[str]
    artifacts: list[str]
    stage_edges: list[tuple[str, str]]
    artifact_edges: list[tuple[str, str]]
```

```python
def extract_graph_view(g: nx.DiGraph[str]) -> GraphView:
    """Extract a renderer-friendly view from the bipartite graph.

    Walks the bipartite graph once, collecting stage names, artifact paths,
    and derived edges without exposing the internal node encoding.

    Edge semantics (data-flow direction):
    - stage_edges: (producer_stage, consumer_stage)
    - artifact_edges: (input_artifact, output_artifact)

    Args:
        g: Bipartite artifact-stage graph from build_graph().

    Returns:
        GraphView with plain-string nodes and edges.
    """
    stages = list[str]()
    artifacts = list[str]()
    stage_edges = list[tuple[str, str]]()
    artifact_edges = list[tuple[str, str]]()

    # Collect nodes by type
    for node in g.nodes():
        node_type, value = parse_node(node)
        if node_type == NodeType.STAGE:
            stages.append(value)
        else:
            artifacts.append(value)

    # Derive stage-to-stage edges (producer -> consumer)
    # Walk: stage -> artifact (produces) -> stage (consumes)
    for node in g.nodes():
        node_type, source_name = parse_node(node)
        if node_type != NodeType.STAGE:
            continue
        for artifact_node_id in g.successors(node):
            if g.nodes[artifact_node_id]["type"] != NodeType.ARTIFACT:
                continue
            for consumer_node in g.successors(artifact_node_id):
                if g.nodes[consumer_node]["type"] == NodeType.STAGE:
                    consumer_name = parse_node(consumer_node)[1]
                    stage_edges.append((source_name, consumer_name))

    # Derive artifact-to-artifact edges (input -> output)
    # Walk: artifact -> stage (consumes) -> artifact (produces)
    for node in g.nodes():
        node_type, source_path = parse_node(node)
        if node_type != NodeType.ARTIFACT:
            continue
        for stage_node_id in g.successors(node):
            if g.nodes[stage_node_id]["type"] != NodeType.STAGE:
                continue
            for output_node in g.successors(stage_node_id):
                if g.nodes[output_node]["type"] == NodeType.ARTIFACT:
                    output_path = parse_node(output_node)[1]
                    artifact_edges.append((source_path, output_path))

    return GraphView(
        stages=stages,
        artifacts=artifacts,
        stage_edges=stage_edges,
        artifact_edges=artifact_edges,
    )
```

Add `"GraphView"` and `"extract_graph_view"` to the `__all__` list.

Add `TypedDict` to the imports from `typing`:
```python
from typing import TYPE_CHECKING, TypedDict
```

### Step 4: Run tests to verify they pass

```bash
cd /home/sami/pivot/roadmap-380 && uv run pytest tests/engine/test_graph.py -k "extract_graph_view" -v
```

Expected: All 4 new tests PASS.

### Step 5: Run full graph test suite (no regressions)

```bash
cd /home/sami/pivot/roadmap-380 && uv run pytest tests/engine/test_graph.py -v
```

Expected: All tests PASS.

---

## Task 2: Update `dag/render.py` to consume `GraphView` instead of raw bipartite graph

**Files:**
- Modify: `src/pivot/dag/render.py`
- Modify: `src/pivot/dag/__init__.py`
- Test: `tests/core/test_dag_render.py`

### Step 1: Replace `_extract_nodes_and_edges` with `GraphView`-based extraction

Rewrite `dag/render.py` to:
1. Remove imports of `engine_graph` and `engine_types`
2. Import `GraphView` from `pivot.engine.graph` under `TYPE_CHECKING`
3. Replace `_extract_nodes_and_edges(g, stages)` with a simple function that selects the right fields from `GraphView`
4. Change all three render functions (`render_ascii`, `render_mermaid`, `render_dot`) to accept `GraphView` instead of `nx.DiGraph[str]`

The full updated `render.py` top section and changed functions:

**Remove these imports:**
```python
from pivot.engine import graph as engine_graph
from pivot.engine import types as engine_types
```
And the conditional `import networkx as nx` in `TYPE_CHECKING`.

**Add this import:**
```python
if TYPE_CHECKING:
    from pivot.engine.graph import GraphView
```

**Replace `_extract_nodes_and_edges` with:**
```python
def _select_view(
    view: GraphView,
    stages: bool,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Select nodes and edges for the requested view (stages or artifacts).

    Args:
        view: Pre-extracted graph data.
        stages: If True, return stage nodes/edges; if False, artifact nodes/edges.

    Returns:
        Tuple of (node labels, edges between nodes).
    """
    if stages:
        return view["stages"], view["stage_edges"]
    return view["artifacts"], view["artifact_edges"]
```

**Update render function signatures** — change first parameter from `g: nx.DiGraph[str]` to `view: GraphView`:

```python
def render_ascii(view: GraphView, stages: bool = False) -> str:
    """Render graph as ASCII art using grandalf Sugiyama layout.

    Args:
        view: GraphView from extract_graph_view().
        stages: If True, render stage nodes; if False (default), render artifact nodes.

    Returns:
        ASCII art representation of the graph.
    """
    nodes, edges = _select_view(view, stages)
    # ... rest unchanged ...
```

```python
def render_mermaid(view: GraphView, stages: bool = False) -> str:
    """Render graph as Mermaid flowchart TD format.

    Args:
        view: GraphView from extract_graph_view().
        stages: If True, render stage nodes; if False (default), render artifact nodes.

    Returns:
        Mermaid flowchart string.
    """
    nodes, edges = _select_view(view, stages)
    # ... rest unchanged ...
```

```python
def render_dot(view: GraphView, stages: bool = False) -> str:
    """Render graph as Graphviz DOT format.

    Args:
        view: GraphView from extract_graph_view().
        stages: If True, render stage nodes; if False (default), render artifact nodes.

    Returns:
        DOT format string.
    """
    nodes, edges = _select_view(view, stages)
    # ... rest unchanged ...
```

### Step 2: Update `dag/__init__.py` re-exports

No changes needed — `render_ascii`, `render_mermaid`, `render_dot` names stay the same.

### Step 3: Run render tests (expect failures from callers still passing raw graph)

```bash
cd /home/sami/pivot/roadmap-380 && uv run pytest tests/core/test_dag_render.py -v
```

Expected: FAIL — tests still pass raw bipartite graphs to render functions.

---

## Task 3: Update render test helpers to use `GraphView`

**Files:**
- Modify: `tests/core/test_dag_render.py`

### Step 1: Update test helper `_build_graph` to return `GraphView`

Replace the existing `_build_graph` helper and imports:

**Remove:**
```python
from pivot.engine import graph as engine_graph
```

**Add:**
```python
from pivot.engine import graph as engine_graph
```

**Replace `_build_graph`:**
```python
def _build_view(stages_dict: dict[str, RegistryStageInfo]) -> engine_graph.GraphView:
    """Build GraphView from stages dict."""
    bipartite = engine_graph.build_graph(stages_dict)
    return engine_graph.extract_graph_view(bipartite)
```

**Update all test functions** that call `_build_graph` to call `_build_view` instead, and rename the local variable from `g` to `view`. For example:

```python
def test_render_ascii_empty_graph() -> None:
    """Empty graph returns placeholder text."""
    view = _build_view({})
    result = dag.render_ascii(view)
    assert result == "(empty graph)"
```

Apply this rename across all tests. The two subgraph tests (`test_render_ascii_subgraph`, `test_render_mermaid_subgraph`) need special handling — they manually build a subgraph from the bipartite graph. Update them to build the subgraph first, then extract a `GraphView` from it:

```python
def test_render_ascii_subgraph() -> None:
    """Render a subgraph containing only part of the pipeline."""
    stages = {
        "extract": _create_stage("extract", [], ["raw.csv"]),
        "transform": _create_stage("transform", ["raw.csv"], ["clean.csv"]),
        "load": _create_stage("load", ["clean.csv"], ["output.csv"]),
    }
    g = engine_graph.build_graph(stages)

    # Get subgraph of just extract and transform
    subgraph = g.subgraph(
        [
            engine_graph.stage_node("extract"),
            engine_graph.stage_node("transform"),
            engine_graph.artifact_node(pathlib.Path("raw.csv")),
            engine_graph.artifact_node(pathlib.Path("clean.csv")),
        ]
    )
    view = engine_graph.extract_graph_view(subgraph)

    result = dag.render_ascii(view, stages=True)

    assert "extract" in result
    assert "transform" in result
    assert "load" not in result
```

(Same pattern for `test_render_mermaid_subgraph`.)

### Step 2: Run render tests

```bash
cd /home/sami/pivot/roadmap-380 && uv run pytest tests/core/test_dag_render.py -v
```

Expected: All tests PASS.

---

## Task 4: Update `cli/dag.py` to pass `GraphView` to renderers

**Files:**
- Modify: `src/pivot/cli/dag.py`

### Step 1: Update the `dag_cmd` function

The `dag_cmd` function builds a bipartite graph, optionally filters it to a subgraph, then passes it to renderers. Update it to call `extract_graph_view()` before rendering.

**Changes in `dag_cmd`:**

After the subgraph filtering (line ~124), add:
```python
    # Extract view for rendering
    view = engine_graph.extract_graph_view(bipartite_graph)
```

Then update the match block to pass `view` instead of `bipartite_graph`:
```python
    match output_format:
        case "dot":
            output = dag.render_dot(view, stages=show_stages)
        case "mermaid":
            output = dag.render_mermaid(view, stages=show_stages)
        case "md":
            mermaid = dag.render_mermaid(view, stages=show_stages)
            output = f"```mermaid\n{mermaid}\n```"
        case _:
            output = dag.render_ascii(view, stages=show_stages)
```

Note: The rest of `dag_cmd` (target resolution, subgraph filtering) still operates on the raw bipartite graph — only the final rendering step uses the adapter.

### Step 2: Run dag CLI tests

```bash
cd /home/sami/pivot/roadmap-380 && uv run pytest tests/cli/test_dag_cmd.py -v
```

Expected: All tests PASS.

---

## Task 5: Remove `parse_node` from `__all__` and clean up

**Files:**
- Modify: `src/pivot/engine/graph.py`

### Step 1: Remove `parse_node` from `__all__`

`parse_node` is now only used internally within `engine/graph.py`. Remove it from `__all__` to signal it's not part of the public API. Keep the function itself (it's still used by `_check_acyclic`, `get_consumers`, `get_producer`, `get_watch_paths`, `get_downstream_stages`, `update_stage`, `get_upstream_stages`, `get_stage_dag`, and `extract_graph_view`).

**Note:** `parse_node` is still imported by `cli/dag.py`'s `_get_upstream_subgraph` and `_resolve_targets_to_stages` — but those use `engine_graph.get_producer()` and `engine_graph.stage_node()`, not `parse_node` directly. Verify this.

Actually, checking the code again: `cli/dag.py` does NOT import `parse_node` directly — it uses `engine_graph.get_producer()` and `engine_graph.stage_node()`. The imports of `engine_types.NodeType` in `cli/dag.py` are used for the `_get_upstream_subgraph` function (checking `bipartite_graph.nodes[succ]["type"] == engine_types.NodeType.ARTIFACT`). That's a separate concern and acceptable for now — it's the subgraph construction, not rendering.

Remove `"parse_node"` from `__all__` in `engine/graph.py`.

### Step 2: Verify no external callers of `parse_node` remain

```bash
cd /home/sami/pivot/roadmap-380 && grep -rn "parse_node" src/pivot/ --include="*.py" | grep -v "engine/graph.py"
```

Expected: No results (render.py no longer imports it, tests may reference it but that's fine).

**If `parse_node` is still used in tests** (`tests/engine/test_graph.py` has 3 tests for it), those tests can remain — they test the internal function directly. No external production code should reference it.

### Step 3: Run all tests

```bash
cd /home/sami/pivot/roadmap-380 && uv run pytest tests/engine/test_graph.py tests/core/test_dag_render.py tests/cli/test_dag_cmd.py -v
```

Expected: All tests PASS.

---

## Task 6: Quality checks and final verification

**Files:** None (verification only)

### Step 1: Run full quality checks

```bash
cd /home/sami/pivot/roadmap-380 && uv run ruff format . && uv run ruff check . && uv run basedpyright
```

Expected: No errors.

### Step 2: Run full test suite

```bash
cd /home/sami/pivot/roadmap-380 && uv run pytest tests/ -n auto
```

Expected: All tests PASS with no regressions.

---

## Summary of Changes

| File | Change |
|------|--------|
| `src/pivot/engine/graph.py` | Add `GraphView` TypedDict, `extract_graph_view()` function; remove `parse_node` from `__all__` |
| `src/pivot/dag/render.py` | Accept `GraphView` instead of `nx.DiGraph[str]`; remove `engine_graph`/`engine_types` imports; replace `_extract_nodes_and_edges` with `_select_view` |
| `src/pivot/cli/dag.py` | Call `extract_graph_view()` before passing to renderers |
| `tests/engine/test_graph.py` | Add `extract_graph_view` tests |
| `tests/core/test_dag_render.py` | Update to build `GraphView` instead of passing raw bipartite graph |

**What doesn't change:**
- `engine/engine.py` — still uses bipartite graph internally for orchestration
- `status.py`, `cli/status.py`, `cli/repro.py` — use `build_graph`/`get_stage_dag`/`get_execution_order` which are unchanged
- `pipeline/pipeline.py`, `registry.py` — unchanged
- Execution order, skip detection, watch mode — all unchanged
