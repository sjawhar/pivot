# Pivot

**Change your code. Pivot knows what to run.**

Pivot is a Python pipeline tool with automatic code change detection. Define stages with typed Python functions and annotations, and Pivot figures out what needs to re-run—no manual dependency declarations, no stale caches.

```bash
pivot repro      # Run your pipeline
# edit a helper function...
pivot repro      # Pivot detects the change and re-runs affected stages
```

## Quick Example

```python
# pipeline.py
import json
import pathlib
from typing import Annotated, TypedDict

import pivot


class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, pivot.Out("clean.json", pivot.loaders.PathOnly())]


def preprocess(
    raw: Annotated[dict, pivot.Dep("data.json", pivot.loaders.JSON())],
) -> PreprocessOutputs:
    clean = [row for row in raw["records"] if all(row.values())]
    out_path = pathlib.Path("clean.json")
    out_path.write_text(json.dumps({"records": clean}, indent=2))
    return PreprocessOutputs(clean=out_path)


class SummarizeOutputs(TypedDict):
    summary: Annotated[pathlib.Path, pivot.Out("summary.json", pivot.loaders.PathOnly())]


def summarize(
    data: Annotated[dict, pivot.Dep("clean.json", pivot.loaders.JSON())],
) -> SummarizeOutputs:
    records = data["records"]
    values = [r["value"] for r in records]
    summary = {"count": len(values), "mean": sum(values) / len(values)}
    out_path = pathlib.Path("summary.json")
    out_path.write_text(json.dumps(summary, indent=2))
    return SummarizeOutputs(summary=out_path)


pipeline = pivot.Pipeline("my_pipeline")
pipeline.register(preprocess)
pipeline.register(summarize)
```

```bash
pivot repro  # Runs both stages
pivot repro  # Instant - nothing changed
```

Modify `preprocess`, and Pivot automatically re-runs both stages. Modify `train`, and only `train` re-runs.

## What Makes Pivot Different

### Automatic Code Change Detection

Change a helper function, and Pivot knows to re-run stages that call it:

```python
def normalize(records):
    max_val = max(r["value"] for r in records)
    return [{"name": r["name"], "value": r["value"] / max_val} for r in records]

def process(
    raw: Annotated[dict, pivot.Dep("data.json", pivot.loaders.JSON())],
) -> ProcessOutputs:
    return ProcessOutputs(result=normalize(raw["records"]))  # ...and Pivot re-runs process
```

No YAML to update. No manual declarations. Pivot parses your Python and tracks what each stage actually calls.

### See Why Stages Run

```bash
$ pivot status --explain train

Stage: train
  Status: WILL RUN
  Reason: Code dependency changed

  Changes:
    func:normalize
      Old: 5995c853
      New: a1b2c3d4
      File: src/utils.py:15
```

### Watch Mode

Edit code, save, see results:

```bash
pivot repro --watch  # Re-runs automatically on file changes
```

## Getting Started

```bash
uv add pivot
```

See the [Quick Start](getting-started/quickstart.md) to build your first pipeline.

## Requirements

- Python 3.13+
- Unix only (Linux/macOS)

## Learn More

**Start here:** Follow the [Concepts](concepts/index.md) guide — a linear learning path from
first principles to advanced caching.

Then explore task-oriented [Guides](guides/watch-mode.md) for specific workflows:
- [Watch Mode](guides/watch-mode.md) — Rapid iteration
- [Multi-Pipeline Projects](guides/multi-pipeline.md) — Large project organization
- [Remote Storage](guides/remote-storage.md) — Share cache across machines
- [CI Integration](guides/ci-integration.md) — Pipeline verification in CI

**Reference:**
- [CLI Reference](cli/index.md) — All commands and options
- [Architecture](architecture/overview.md) — For contributors
- [Comparison with DVC](comparison.md) — Feature comparison
