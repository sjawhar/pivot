---
date: 2026-02-04
topic: auto-resolve-sibling-deps-root
---

# Auto-resolve sibling dependencies root behavior

## What We Are Building
When a user runs `pivot repro` inside a pipeline subdirectory, Pivot should treat the repo root
as the project root, not the subdirectory. This prevents dependency resolution failures for
sibling pipelines and ensures artifacts can resolve across the repo.

Project root discovery should walk upward until it finds the top-most `.pivot/`. If no `.pivot/`
exists above the current working directory, Pivot should error and require `pivot init`.

## Why This Approach
We want a single, stable project root and state directory regardless of where a user runs Pivot.
Discovering the root by `.pivot/` aligns with the existing state directory and avoids new
configuration requirements.

## Key Decisions
- Project root resolves to the top-most directory containing `.pivot/`.
- State directory is always the repo-level `.pivot/`.
- If no `.pivot/` exists above the current directory, error and require `pivot init`.

## Open Questions
- Should the root discovery also consider VCS root or config markers, or keep it `.pivot/` only?
- How to surface a helpful error when `.pivot/` is missing (suggest `pivot init` and path hints).

## Next Steps
-> `/workflows:plan` for implementation details.
