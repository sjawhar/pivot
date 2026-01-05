# DAG Implementation Plan - Milestone 2.1

## Research Summary: DVC's DAG Implementation

### Key Files Analyzed
1. `/workspaces/treeverse/dvc/dvc/repo/graph.py` - Core DAG building logic
2. `/workspaces/treeverse/dvc/dvc/repo/reproduce.py` - DAG traversal for execution
3. `/workspaces/treeverse/dvc/tests/unit/repo/test_graph.py` - Graph tests
4. `/workspaces/treeverse/dvc/tests/unit/repo/test_reproduce.py` - Execution order tests
5. `/workspaces/treeverse/dvc/tests/func/test_run.py` - Cycle detection tests

### DVC's Graph Structure

**Edge Direction:** Stage → Dependency (consumer → producer)
```
Example: C depends on B, B depends on A
Graph edges: C → B, B → A
Execution order (postorder DFS): A, B, C
```

**Visualization:**
```
    C.dvc → B.dvc → A.dvc
    |          |
    |          --> descendants (execution order)
    |
    <-- ancestors (dependencies)
```

### DVC's build_graph() Algorithm

1. **Initialize:** Create empty DiGraph, add all stages as nodes
2. **Build edges:** For each stage:
   - Skip special stages (repo imports, db imports)
   - For each dependency:
     - Use Trie to find stages that produce this dependency (output matching)
     - Add edge: current_stage → producing_stage
3. **Validate:** Call check_acyclic() using nx.find_cycle()

**Why DVC uses a Trie:**
- Efficiently handles overlapping outputs (e.g., `data/` vs `data/foo.txt`)
- Finds all outputs that match a dependency path
- Handles nested directories

### Execution Order (plan_repro)

DVC uses **DFS postorder traversal** (nx.dfs_postorder_nodes):
```python
# Example graph:
#      1
#    /   \
#   2     3
#  / \   / \
# 4   5 6   7

# Postorder: [4, 5, 2, 6, 7, 3, 1]
# (leaves first, root last)
```

This ensures dependencies run before dependents.

---

## Pivot's Simplified Approach

### Key Simplifications

1. **No Trie needed:** We already validate output conflicts at registration (Milestone 2.0.5)
   - Each output is produced by exactly one stage
   - Simple dict lookup: `outputs_map[path] = stage_name`

2. **Two dependency types:**
   - **File dependency:** `deps=['data.csv']` - find stage that produces this file
   - **Stage dependency:** `deps=['stage:preprocess']` - direct stage reference

3. **Validation already done:** Registration time catches:
   - Duplicate outputs
   - Empty stage names
   - Invalid paths
   - DAG validation is separate (done after all stages registered)

---

## Implementation Plan

### Module: `src/pivot/dag.py`

```python
import networkx as nx
from typing import Any

class DAGError(Exception):
    """Base class for DAG-related errors."""
    pass

class CyclicGraphError(DAGError):
    """Raised when DAG contains cycles."""
    pass

def build_dag(stages: dict[str, dict[str, Any]]) -> nx.DiGraph:
    """Build DAG from registered stages.

    Args:
        stages: Dict of stage_name -> stage_info

    Returns:
        DiGraph with edges from consumer to producer

    Raises:
        CyclicGraphError: If graph contains cycles

    Example:
        >>> stages = {
        ...     'preprocess': {'deps': ['data.csv'], 'outs': ['clean.csv']},
        ...     'train': {'deps': ['stage:preprocess'], 'outs': ['model.pkl']}
        ... }
        >>> graph = build_dag(stages)
        >>> list(nx.dfs_postorder_nodes(graph))
        ['preprocess', 'train']
    """
    graph = nx.DiGraph()

    # Step 1: Add all stages as nodes
    for stage_name, stage_info in stages.items():
        graph.add_node(stage_name, **stage_info)

    # Step 2: Build output map for file dependency resolution
    outputs_map = _build_outputs_map(stages)

    # Step 3: Add edges (stage -> its dependencies)
    for stage_name, stage_info in stages.items():
        for dep in stage_info.get('deps', []):
            if dep.startswith('stage:'):
                # Direct stage reference
                upstream_stage = dep.replace('stage:', '')
                if upstream_stage in stages:
                    graph.add_edge(stage_name, upstream_stage)
            else:
                # File dependency - find producing stage
                producer = outputs_map.get(dep)
                if producer:
                    graph.add_edge(stage_name, producer)

    # Step 4: Check for cycles
    _check_acyclic(graph)

    return graph

def _build_outputs_map(stages: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Build mapping from output path to stage name.

    Returns:
        Dict of output_path -> stage_name
    """
    outputs_map = {}
    for stage_name, stage_info in stages.items():
        for out in stage_info.get('outs', []):
            outputs_map[out] = stage_name
    return outputs_map

def _check_acyclic(graph: nx.DiGraph) -> None:
    """Check graph for cycles, raise if found."""
    try:
        cycle = nx.find_cycle(graph, orientation='original')
    except nx.NetworkXNoCycle:
        return

    # Extract stage names from cycle
    stages_in_cycle = set()
    for from_node, to_node, _ in cycle:
        stages_in_cycle.add(from_node)
        stages_in_cycle.add(to_node)

    raise CyclicGraphError(
        f"Circular dependency detected: {' -> '.join(stages_in_cycle)}"
    )

def get_execution_order(
    graph: nx.DiGraph,
    stages: list[str] | None = None
) -> list[str]:
    """Get execution order using DFS postorder traversal.

    Args:
        graph: DAG of stages
        stages: Optional list of stages to execute (default: all)

    Returns:
        List of stage names in execution order (dependencies first)
    """
    if stages:
        # Get subgraph containing only requested stages and their deps
        subgraph = _get_subgraph(graph, stages)
        return list(nx.dfs_postorder_nodes(subgraph))

    return list(nx.dfs_postorder_nodes(graph))

def _get_subgraph(
    graph: nx.DiGraph,
    source_stages: list[str]
) -> nx.DiGraph:
    """Get subgraph containing sources and all their dependencies."""
    nodes = []
    for stage in source_stages:
        # DFS from this stage (follows edges backward to dependencies)
        nodes.extend(nx.dfs_postorder_nodes(graph, stage))
    return graph.subgraph(nodes)

def get_downstream_stages(
    graph: nx.DiGraph,
    stage: str
) -> list[str]:
    """Get all stages that depend on given stage (directly or transitively).

    Uses reverse graph to traverse from stage to dependents.
    """
    reversed_graph = graph.reverse(copy=False)
    return list(nx.dfs_postorder_nodes(reversed_graph, stage))
```

---

## Test Plan

### Test File: `tests/test_dag.py`

**Tests to implement** (based on DVC patterns):

1. **Basic DAG construction:**
   - `test_build_dag_simple_chain()` - A → B → C
   - `test_build_dag_diamond()` - Diamond dependency pattern
   - `test_build_dag_independent_stages()` - No dependencies
   - `test_build_dag_empty()` - No stages

2. **Dependency resolution:**
   - `test_file_dependency_resolution()` - Find stage by output file
   - `test_stage_reference_resolution()` - stage:name dependencies
   - `test_mixed_dependencies()` - Both file and stage references
   - `test_missing_file_dependency()` - Dep file not produced by any stage

3. **Cycle detection:**
   - `test_circular_dependency_raises_error()` - A → B → A
   - `test_self_dependency_raises_error()` - A → A
   - `test_transitive_cycle_raises_error()` - A → B → C → A

4. **Execution order:**
   - `test_execution_order_simple_chain()` - Linear order
   - `test_execution_order_diamond()` - Verify deps before dependents
   - `test_execution_order_parallel_branches()` - Independent branches
   - `test_execution_order_subset()` - Execute only specific stages

5. **Subgraph extraction:**
   - `test_get_subgraph_single_stage()` - Stage + its deps
   - `test_get_subgraph_multiple_stages()` - Multiple stages + deps
   - `test_get_downstream_stages()` - Find dependents

6. **Edge cases:**
   - `test_stage_with_no_deps()` - Leaf nodes
   - `test_stage_with_no_outs()` - Terminal nodes
   - `test_multiple_stages_same_dependency()` - Fan-in pattern

### Example Test (DFS postorder verification):

```python
def test_execution_order_diamond():
    """Verify diamond dependency execution order.

         train
        /     \\
    preproc  features
        \\     /
          data
    """
    stages = {
        'data': {'deps': [], 'outs': ['data.csv']},
        'preproc': {'deps': ['data.csv'], 'outs': ['clean.csv']},
        'features': {'deps': ['data.csv'], 'outs': ['features.csv']},
        'train': {'deps': ['clean.csv', 'features.csv'], 'outs': ['model.pkl']}
    }

    graph = build_dag(stages)
    order = get_execution_order(graph)

    # data must run first
    assert order[0] == 'data'

    # preproc and features can run in any order (both after data)
    assert set(order[1:3]) == {'preproc', 'features'}

    # train must run last
    assert order[3] == 'train'
```

---

## Integration with Existing Code

### `registry.py` integration:

Add method to StageRegistry:
```python
def build_dag(self) -> nx.DiGraph:
    """Build DAG from registered stages."""
    from pivot import dag
    return dag.build_dag(self._stages)
```

### `project.py` integration:

Resolve paths before DAG building:
```python
# In dag.py, before building outputs_map
from pivot import project

def _normalize_paths(stages):
    """Resolve all relative paths."""
    for stage_info in stages.values():
        stage_info['deps'] = [project.resolve_path(d) for d in stage_info['deps']]
        stage_info['outs'] = [project.resolve_path(o) for o in stage_info['outs']]
```

---

## Design Decisions

### 1. Why not use a Trie?

**DVC needs it because:**
- Validates overlapping outputs at graph build time
- Handles directory vs file conflicts (`data/` vs `data/foo.txt`)

**Pivot doesn't need it because:**
- We validate output conflicts at registration time (Milestone 2.0.5)
- Each output is produced by exactly one stage (guaranteed)
- Simple dict lookup is sufficient and faster

### 2. Edge direction: consumer → producer

**Why this direction?**
- Matches DVC's convention
- DFS postorder naturally gives execution order
- Easier to find "what depends on X" (predecessors)

### 3. When to build DAG?

**Options:**
1. **Lazy:** Build when needed (first execution)
2. **Eager:** Build at registration time

**Choice: Lazy**
- Allow stages to be registered in any order
- Only validate DAG when user calls `validate_dag()` or `run()`
- More flexible for incremental pipeline building

---

## Files to Create

1. `src/pivot/dag.py` (~150 lines)
2. `tests/test_dag.py` (~400 lines, ~20 tests)

---

## Success Criteria

- ✅ All 20+ tests pass
- ✅ Cycle detection works correctly
- ✅ Execution order matches DFS postorder
- ✅ Handles both file and stage dependencies
- ✅ Coverage > 90%
- ✅ Integration with registry works
- ✅ All code quality checks pass

---

## Next Steps After DAG

With DAG complete, we'll have:
1. ✅ Fingerprinting (Week 1)
2. ✅ Registry (Week 1)
3. ✅ Project root detection (Week 2)
4. ✅ Validation (Week 2)
5. ✅ **DAG construction (Week 2)** ← We are here

Next: Per-stage lock files (Milestone 2.2)
