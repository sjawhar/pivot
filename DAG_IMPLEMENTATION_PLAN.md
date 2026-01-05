# DAG Implementation Plan - Milestone 2.1

## ✅ Status Update (2026-01-05) - MILESTONE 2.1 COMPLETE

**Completed:**
- ✅ Week 1: Fingerprinting + Registry (126 tests, 94.17% coverage)
- ✅ Milestone 2.0: Project root detection
- ✅ Milestone 2.0.5: Input validation and path normalization
- ✅ Trie-based output overlap detection (adapted from DVC)
- ✅ Removed `stage:` dependency syntax
- ✅ Code simplifications applied (extracted `_normalize_paths` helper)
- ✅ Bug fix: `ast_utils.normalize_ast()` now handles empty body edge case
- ✅ **Milestone 2.1: DAG Construction (COMPLETE)**
  - ✅ `src/pivot/dag.py` implemented (49 lines, 98.59% coverage)
  - ✅ 21 comprehensive tests in `tests/test_dag.py`
  - ✅ Registry integration with `build_dag()` method
  - ✅ All cycle detection, dependency validation, execution order tests passing

**Ready for:**
- 🔲 Milestone 2.2: Per-stage lock files (next task)

**Files:**
- `src/pivot/fingerprint.py` (119 lines, 92.18% coverage)
- `src/pivot/registry.py` (74 lines, 97.78% coverage) ⬆️
- `src/pivot/dag.py` (49 lines, 98.59% coverage) 🆕
- `src/pivot/project.py` (22 lines, 100% coverage)
- `src/pivot/trie.py` (22 lines, 90.62% coverage)
- `src/pivot/exceptions.py` (19 lines, 100% coverage)
- `src/pivot/ast_utils.py` (44 lines, 91.94% coverage)

---

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

## Pivot's Approach

### Key Design Decisions

1. **✅ Trie used at registration time (IMPLEMENTED):**
   - We validate output conflicts at registration using `src/pivot/trie.py`
   - Detects both exact duplicates AND overlapping paths (parent/child relationships)
   - Raises `OutputDuplicationError` or `OverlappingOutputPathsError`
   - At DAG build time, simple dict lookup suffices: `outputs_map[path] = stage_name`
   - Each output is guaranteed to be produced by exactly one stage

2. **✅ No `stage:` references (IMPLEMENTED):** Just use file paths
   - DAG builder will resolve which stage produces each dependency
   - Cleaner API, less for users to remember
   - Removed all references from docstrings and tests

3. **Dependency validation (at DAG build time):**
   - Each dependency must be EITHER:
     - An output of another registered stage, OR
     - A file that exists on disk
   - This ensures pipeline integrity

4. **✅ Path normalization (IMPLEMENTED):** All paths normalized at registration
   - Uses `project.resolve_path()` to convert relative → absolute paths
   - Eliminates path ambiguity (`./data.csv` vs `data.csv`)
   - Output conflict detection works correctly
   - Extracted `_normalize_paths()` helper to reduce code duplication

5. **Future:** `.pvt` file tracking (deferred to later milestone)
   - Similar to DVC's `.dvc` files
   - Track individual files and directories
   - Git integration using python library (not shell)

---

## ⚠️ Open Questions (To Resolve Before Implementation)

### 1. Dependency Validation Strategy

**Question:** When should we validate that dependencies exist?

**Options:**
- **A) Validate at DAG build time (RECOMMENDED)**
  - Pro: Fail fast, clear error messages
  - Pro: User knows immediately if pipeline has missing inputs
  - Con: May reject valid pipelines where files are generated externally

- **B) Validate at execution time**
  - Pro: Allows external file generation before execution
  - Con: Later error discovery, harder to debug

**Decision:** Validate at DAG build time by default, with `validate=False` escape hatch for advanced use cases.

### 2. Missing Dependency Behavior

**Question:** What if a dependency is not produced by any stage AND doesn't exist on disk?

**Options:**
- **A) Strict: Raise DependencyNotFoundError (RECOMMENDED)**
  - Clear error: "Stage 'train' depends on 'data.csv' which is not produced by any stage and does not exist on disk"
  - User must either: add upstream stage, create file, or use validate=False

- **B) Lenient: Log warning and continue**
  - Pro: More flexible for iterative development
  - Con: Silent failures, confusing behavior

**Decision:** Strict by default. Users can use `validate=False` if they know what they're doing.

### 3. Test Data Creation Strategy

**Question:** Should DAG tests use real files on disk?

**Options:**
- **A) Use tmp_path and create real files (RECOMMENDED)**
  - Pro: Integration tests are realistic
  - Pro: Tests actual file existence checks
  - Example: `(tmp_path / 'data.csv').touch()`

- **B) Mock file existence**
  - Pro: Faster tests
  - Con: May miss real-world edge cases

**Decision:** Use tmp_path for integration tests, mock for unit tests of individual functions.

### 4. Execution Order for Disconnected Components

**Question:** If pipeline has independent branches (no shared dependencies), what order?

```python
# Example: Two independent pipelines
Stage A → Stage B
Stage X → Stage Y
```

**Answer:** DFS postorder is deterministic but order between branches is undefined. This is acceptable—independent stages can run in any order. Document this behavior.

---

## Implementation Plan

### Module: `src/pivot/dag.py`

```python
import networkx as nx
from pathlib import Path
from typing import Any

from pivot.exceptions import CyclicGraphError, DependencyNotFoundError

def build_dag(stages: dict[str, dict[str, Any]], validate: bool = True) -> nx.DiGraph:
    """Build DAG from registered stages.

    Args:
        stages: Dict of stage_name -> stage_info
        validate: If True, validate that all dependencies exist

    Returns:
        DiGraph with edges from consumer to producer

    Raises:
        CyclicGraphError: If graph contains cycles
        DependencyNotFoundError: If dependency doesn't exist (when validate=True)

    Example:
        >>> stages = {
        ...     'preprocess': {'deps': ['data.csv'], 'outs': ['clean.csv']},
        ...     'train': {'deps': ['clean.csv'], 'outs': ['model.pkl']}
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

    # Step 3: Add edges (stage -> its dependencies) and validate
    for stage_name, stage_info in stages.items():
        for dep in stage_info.get('deps', []):
            # File dependency - find producing stage
            producer = outputs_map.get(dep)
            if producer:
                graph.add_edge(stage_name, producer)
            elif validate:
                # Dependency not produced by any stage - check if file exists
                if not Path(dep).exists():
                    raise DependencyNotFoundError(
                        f"Stage '{stage_name}' depends on '{dep}' which is not "
                        f"produced by any stage and does not exist on disk"
                    )

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
   - `test_dependency_on_existing_file()` - Dep exists on disk (no edge created)
   - `test_missing_dependency_raises_error()` - Dep not produced AND doesn't exist
   - `test_missing_dependency_with_validate_false()` - No error when validate=False

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
def build_dag(self, validate: bool = True) -> nx.DiGraph:
    """Build DAG from registered stages."""
    from pivot import dag
    return dag.build_dag(self._stages, validate=validate)
```

**Note:** Path normalization already happens at registration time (Milestone 2.0.5),
so DAG builder works with absolute paths.

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

---

## Future Milestones (Post-MVP)

### Phase 4: File Tracking System (`.pvt` files)

**Goal:** Track individual files and directories like DVC's `.dvc` files

**Features:**
- Track files/directories with `.pvt` metadata files
- Enable `pvt add <file>` to track files
- Support granular pulls and updates for directories
- Integrate with git (using python library like `gitpython`, not shell)

**Dependency validation will then check:**
1. Output of registered stage, OR
2. Tracked by `.pvt` file, OR
3. Tracked by git, OR
4. Exists on disk (with warning)

**Investigation needed:**
- Study DVC's `.dvc` file format and implementation
- Study `dvc-data` repo for directory optimization strategies
- Design `.pvt` file format (likely similar to DVC's approach)
