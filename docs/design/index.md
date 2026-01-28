# Design Decisions

This section documents significant design decisions and explorations for Pivot.

These documents capture the reasoning behind architectural choices, alternatives considered, and trade-offs made. They're useful for understanding why Pivot works the way it does and for informing future development.

## Documents

| Document | Status | Summary |
|----------|--------|---------|
| [Watch Engine Design](watch-engine.md) | Implemented | Architecture of the file watching and auto-rerun system |
| [Hot Reload Exploration](hot-reload-exploration.md) | Exploratory | Investigation into faster code change handling via `importlib.reload()` |

## When to Write a Design Document

Consider writing a design document when:

- Making a significant architectural decision with multiple viable approaches
- Exploring a complex feature that requires research
- A decision has non-obvious trade-offs worth documenting
- Future developers might wonder "why was it done this way?"

Design documents don't need to be long. A few paragraphs explaining the problem, alternatives, and rationale is often sufficient.
