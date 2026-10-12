# Pivot: High-Performance Python Pipeline Tool

**Change your code. Pivot knows what to run.**

Pivot is a Python pipeline tool with automatic code change detection. Define stages
with typed Python functions and annotations, and Pivot figures out what needs to
re-run — no manual dependency declarations, no stale caches.

- **Automatic code change detection** using Python introspection
- **Per-stage lock files** for fast parallel writes
- **Warm worker pools** with preloaded imports
- **Content-addressable caching** with S3 remote storage
- **DVC compatibility** via YAML export

**Python:** 3.13+ | **Platform:** Unix (Linux/macOS)

## Quick Start

```bash
pip install pivot
```

Define stages as pure, typed functions:

```python
# pipeline.py
from typing import Annotated, TypedDict

import pandas
import pivot


class PreprocessOutputs(TypedDict):
    clean: Annotated[pandas.DataFrame, pivot.Out("processed.csv", pivot.loaders.CSV())]


def preprocess(
    raw: Annotated[pandas.DataFrame, pivot.Dep("data.csv", pivot.loaders.CSV())],
) -> PreprocessOutputs:
    return PreprocessOutputs(clean=raw.dropna())


pipeline = pivot.Pipeline("my_pipeline")
pipeline.register(preprocess)
```

Then run the pipeline — Pivot resolves the DAG from artifact dependencies:

```bash
pivot repro           # Run the entire pipeline
pivot repro           # Instant - nothing changed
pivot repro --watch   # Re-run automatically on file changes
```

## Learn More

- [Documentation](https://sjawhar.github.io/pivot/)
- [Source code](https://github.com/sjawhar/pivot)
- Interactive TUI: `pip install pivot[tui]`, then `pivot repro --tui`
