# Multi-Pipeline Projects

This tutorial shows how to organize larger projects with multiple pipelines that automatically discover each other's outputs.

## When to Use Multiple Pipelines

Split into multiple pipelines when you have:

- **Team boundaries** - Different teams own different parts of the workflow
- **Reusable components** - A data preparation step used by multiple analyses
- **Large monorepos** - Subdirectories that can run independently

## Example 1: Nested Pipelines (Parent/Child)

A common pattern: shared data preparation at the project root, with analysis pipelines in subdirectories.

### Project Structure

```
my_project/
├── .pivot/              # Project root (top-most .pivot/)
├── pipeline.py          # Produces shared/data.csv
├── shared/
│   └── data.csv
└── analysis/
    └── pipeline.py      # Consumes ../shared/data.csv
```

### Step 1: Initialize the Project

```bash
mkdir -p my_project/analysis my_project/shared
cd my_project
pivot init
```

### Step 2: Create the Parent Pipeline

Create `pipeline.py` at the project root:

```python
# my_project/pipeline.py
from pathlib import Path
from typing import Annotated, TypedDict

from pivot import loaders, outputs
from pivot.pipeline import Pipeline

pipeline = Pipeline("data_prep")


class PrepareOutputs(TypedDict):
    data: Annotated[Path, outputs.Out("shared/data.csv", loaders.PathOnly())]


def prepare() -> PrepareOutputs:
    """Generate shared dataset."""
    out = Path("shared/data.csv")
    out.parent.mkdir(exist_ok=True)
    out.write_text("id,value\n1,100\n2,200\n3,300\n")
    return PrepareOutputs(data=out)


pipeline.register(prepare)
```

### Step 3: Create the Child Pipeline

Create `analysis/pipeline.py`:

```python
# my_project/analysis/pipeline.py
from pathlib import Path
from typing import Annotated, TypedDict

from pivot import loaders, outputs
from pivot.pipeline import Pipeline

pipeline = Pipeline("analysis")


class AnalyzeOutputs(TypedDict):
    report: Annotated[Path, outputs.Out("report.txt", loaders.PathOnly())]


def analyze(
    data: Annotated[Path, outputs.Dep("../shared/data.csv", loaders.PathOnly())],
) -> AnalyzeOutputs:
    """Analyze the shared dataset."""
    content = data.read_text()
    lines = len(content.strip().split("\n")) - 1  # Exclude header

    out = Path("report.txt")
    out.write_text(f"Processed {lines} records\n")
    return AnalyzeOutputs(report=out)


pipeline.register(analyze)
```

### Step 4: Run from the Subdirectory

```bash
cd analysis
pivot repro
```

**What happens:**

1. Pivot finds project root by walking up to `my_project/` (top-most `.pivot/`)
2. The `analyze` stage needs `../shared/data.csv`
3. Pivot searches from that path upward and finds `my_project/pipeline.py`
4. The `prepare` stage is automatically included
5. Both stages run in correct order

You didn't need to call `include()` or configure anything - Pivot discovered the dependency automatically.

## Example 2: Sibling Pipelines

For larger projects, you might have pipelines at the same directory level that depend on each other.

### Project Structure

```
my_project/
├── .pivot/
└── pipelines/
    ├── feature_a/
    │   └── pipeline.py      # Produces output.csv
    └── feature_b/
        └── pipeline.py      # Consumes ../feature_a/output.csv
```

### Step 1: Create Sibling Pipelines

Create `pipelines/feature_a/pipeline.py`:

```python
# pipelines/feature_a/pipeline.py
from pathlib import Path
from typing import Annotated, TypedDict

from pivot import loaders, outputs
from pivot.pipeline import Pipeline

pipeline = Pipeline("feature_a")


class FeatureAOutputs(TypedDict):
    output: Annotated[Path, outputs.Out("output.csv", loaders.PathOnly())]


def compute_a() -> FeatureAOutputs:
    """Compute feature A."""
    out = Path("output.csv")
    out.write_text("feature,value\na1,10\na2,20\n")
    return FeatureAOutputs(output=out)


pipeline.register(compute_a)
```

Create `pipelines/feature_b/pipeline.py`:

```python
# pipelines/feature_b/pipeline.py
from pathlib import Path
from typing import Annotated, TypedDict

from pivot import loaders, outputs
from pivot.pipeline import Pipeline

pipeline = Pipeline("feature_b")


class FeatureBOutputs(TypedDict):
    combined: Annotated[Path, outputs.Out("combined.csv", loaders.PathOnly())]


def compute_b(
    a_data: Annotated[Path, outputs.Dep("../feature_a/output.csv", loaders.PathOnly())],
) -> FeatureBOutputs:
    """Combine with feature A."""
    a_content = a_data.read_text()

    out = Path("combined.csv")
    out.write_text(f"# Combined with feature_a\n{a_content}")
    return FeatureBOutputs(combined=out)


pipeline.register(compute_b)
```

### Step 2: Run Feature B

```bash
cd pipelines/feature_b
pivot repro
```

Pivot automatically discovers `feature_a/pipeline.py` because:

1. The dependency `../feature_a/output.csv` is in the `feature_a/` directory
2. Pivot searches from there and finds `feature_a/pipeline.py`
3. The `compute_a` stage produces that file, so it's included

## Project Structure Recommendations

**When to split pipelines:**

- Each pipeline should be runnable independently (for testing, CI)
- Split at natural boundaries (data prep vs. modeling vs. reporting)
- Keep pipelines that change together in the same directory

**Naming conventions:**

- Use descriptive directory names (`data_prep/`, `model_training/`, `reports/`)
- Pipeline names should match their purpose, not location

**See also:**

- [Pipeline Discovery & Resolution](../reference/discovery.md) - How the discovery algorithm works
- [Defining Pipelines](../reference/pipelines.md) - Pipeline basics and `include()`
