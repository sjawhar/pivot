---
module: Engine
date: 2026-02-03
problem_type: logic_error
component: failure_cascade
symptoms:
  - "Redundant O(n^2) traversal in _cascade_failure method"
  - "Stack-based iteration treating pre-computed transitive list as direct-children-only"
  - "Unnecessary visited set and cycle detection in DAG"
root_cause: logic_error
resolution_type: code_simplification
severity: low
tags: [engine, dag-traversal, cascade-failure, nx-descendants, pivot]
---

# Troubleshooting: Redundant Stack-Based Traversal in _cascade_failure

## Problem

The `_cascade_failure` method used a stack-based iterative traversal pattern, treating `_stage_downstream` as containing only direct children. However, `_stage_downstream` already contains ALL transitive descendants (computed via `nx.descendants`), making the manual traversal redundant and O(n^2).

## Environment

- Module: Engine (`src/pivot/engine/engine.py`)
- Function: `_cascade_failure()` (line ~992)
- Python Version: 3.13+
- Date: 2026-02-03

## Symptoms

- Stack-based traversal with `visited` set and cycle detection warnings
- Each downstream stage was added to the stack and re-processed
- Code was correct but unnecessarily complex (15+ lines vs 4 lines)
- Dead code: cycle detection warning could never trigger (valid DAGs have no cycles)

## Root Cause: API Asymmetry

The graph API has an intentional but confusing asymmetry:

```python
# src/pivot/engine/graph.py

def get_upstream_stages(g, stage_name) -> list[str]:
    """Get stages whose outputs are consumed by this stage."""
    # Uses g.predecessors() -> returns DIRECT parents only
    for artifact in g.predecessors(node):
        for producer in g.predecessors(artifact):
            upstream.append(...)

def get_downstream_stages(g, stage_name) -> list[str]:
    """Get all stages transitively downstream of this one."""
    # Uses nx.descendants() -> returns ALL transitive descendants
    for descendant in nx.descendants(g, node):
        downstream.append(...)
```

**Key insight**: `get_upstream_stages()` returns direct dependencies only, but `get_downstream_stages()` returns all transitive descendants. The naming doesn't make this obvious.

## Solution

Simplified `_cascade_failure` to iterate once through the pre-computed transitive list:

**Before (redundant):**
```python
async def _cascade_failure(self, failed_stage: str) -> None:
    visited = set[str]()
    stack = [failed_stage]

    while stack:
        current_stage = stack.pop()
        for downstream_name in self._stage_downstream.get(current_stage, []):
            if downstream_name in visited:
                _logger.warning("Cycle detected...")  # Dead code
                continue
            visited.add(downstream_name)
            state = self._get_stage_state(downstream_name)
            if state in (StageExecutionState.PENDING, StageExecutionState.READY):
                await self._set_stage_state(downstream_name, StageExecutionState.BLOCKED)
                stack.append(downstream_name)  # Redundant - already have all descendants
```

**After (correct):**
```python
async def _cascade_failure(self, failed_stage: str) -> None:
    """Mark downstream stages as blocked due to upstream failure.

    Since _stage_downstream already contains all transitive descendants (computed
    via get_downstream_stages which uses nx.descendants), we simply iterate through
    them once without recursion.
    """
    for downstream_name in self._stage_downstream.get(failed_stage, []):
        state = self._get_stage_state(downstream_name)
        if state in (StageExecutionState.PENDING, StageExecutionState.READY):
            await self._set_stage_state(downstream_name, StageExecutionState.BLOCKED)
```

## Why This Works

1. **`_stage_downstream[stage]`** is populated at initialization via `get_downstream_stages()`, which uses `nx.descendants()` to compute ALL transitive descendants upfront.

2. **Single iteration is sufficient** because the transitive closure is already computed - no need to recursively walk the graph.

3. **No cycle detection needed** because NetworkX validates the graph is acyclic during construction.

## Prevention

- Document the asymmetry between `get_upstream_stages` (direct) and `get_downstream_stages` (transitive) in docstrings
- When working with pre-computed transitive closures, check if manual traversal is necessary
- Consider renaming to `get_transitive_downstream_stages()` to make the behavior explicit

## Related Documentation

- `docs/architecture/engine.md` - Graph query API overview
- `docs/architecture/execution.md` - Stage execution state machine
- `docs/plans/2026-02-02-consolidate-dag-graphs.md` - DAG consolidation details
