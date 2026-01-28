# Migrating from DVC

This guide helps you migrate an existing DVC pipeline to Pivot.

## Key Differences

| Feature | DVC | Pivot |
|---------|-----|-------|
| Pipeline definition | `dvc.yaml` | `pivot.yaml` |
| Parameters | `params.yaml` | Python `StageParams` classes |
| Code tracking | Manual `deps:` on `.py` files | Automatic code fingerprinting |
| Stage execution | Shell commands | Python functions |
| Lock file | Single `dvc.lock` | Per-stage `.pivot/stages/*.lock` |
| Cache | `.dvc/cache/` | `.pivot/cache/` |

## Concept Mapping

### Stages

**DVC:**
```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python scripts/preprocess.py
    deps:
      - data/raw.csv
      - scripts/preprocess.py
    outs:
      - data/processed.csv
```

**Pivot:**
```yaml
# pivot.yaml
stages:
  preprocess:
    python: scripts.preprocess.run
    deps:
      raw: data/raw.csv
    outs:
      processed: data/processed.csv
```

```python
# scripts/preprocess.py
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs


class PreprocessOutputs(TypedDict):
    processed: Annotated[pandas.DataFrame, outputs.Out("data/processed.csv", loaders.CSV())]


def run(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data/raw.csv", loaders.CSV())],
) -> PreprocessOutputs:
    df = raw.dropna()
    return {"processed": df}
```

Note: Pivot automatically tracks code changes. You don't need to list `.py` files in deps.

### Parameters

**DVC:**
```yaml
# params.yaml
train:
  learning_rate: 0.01
  epochs: 100
```

```yaml
# dvc.yaml
stages:
  train:
    cmd: python train.py
    params:
      - train.learning_rate
      - train.epochs
```

**Pivot:**
```python
# train.py
from pivot.stage_def import StageParams


class TrainParams(StageParams):
    learning_rate: float = 0.01
    epochs: int = 100


def train(params: TrainParams, ...):
    print(f"LR: {params.learning_rate}")
    ...
```

```yaml
# pivot.yaml
stages:
  train:
    python: train.train
    params:
      learning_rate: 0.05  # Override defaults
```

Benefits:
- Type checking and IDE support
- Validation at parse time
- Parameter changes detected via fingerprinting

### Metrics and Plots

**DVC:**
```yaml
# dvc.yaml
stages:
  train:
    cmd: python train.py
    metrics:
      - metrics.json:
          cache: false
    plots:
      - plots/loss.png
```

**Pivot:**
```python
class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]
    metrics: Annotated[dict, outputs.Metric("metrics.json")]
    plot: Annotated[pathlib.Path, outputs.Plot("plots/loss.png", loaders.PathOnly())]
```

```yaml
# pivot.yaml
stages:
  train:
    python: train.train
    outs:
      model: model.pkl
    metrics:
      metrics: metrics.json
    plots:
      plot: plots/loss.png
```

### Remote Storage

**DVC:**
```bash
dvc remote add -d myremote s3://mybucket/cache
dvc push
dvc pull
```

**Pivot:**
```bash
pivot config set remotes.origin s3://mybucket/cache
pivot config set default_remote origin
pivot push
pivot pull
```

## Migration Steps

### Step 1: Create pivot.yaml

Convert your `dvc.yaml` to `pivot.yaml`:

```yaml
# pivot.yaml
stages:
  preprocess:
    python: scripts.preprocess.run
    deps:
      raw: data/raw.csv
    outs:
      processed: data/processed.csv

  train:
    python: scripts.train.run
    deps:
      data: data/processed.csv
    outs:
      model: models/model.pkl
    metrics:
      metrics: metrics.json
```

### Step 2: Convert Scripts to Functions

Transform shell-command scripts into Python functions:

**Before (scripts/preprocess.py):**
```python
import pandas as pd

df = pd.read_csv("data/raw.csv")
df = df.dropna()
df.to_csv("data/processed.csv", index=False)
```

**After (scripts/preprocess.py):**
```python
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs


class PreprocessOutputs(TypedDict):
    processed: Annotated[pandas.DataFrame, outputs.Out("data/processed.csv", loaders.CSV())]


def run(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data/raw.csv", loaders.CSV())],
) -> PreprocessOutputs:
    df = raw.dropna()
    return {"processed": df}
```

### Step 3: Convert Parameters to StageParams

**Before (params.yaml + train.py):**
```python
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

lr = params["learning_rate"]
```

**After (train.py):**
```python
from pivot.stage_def import StageParams


class TrainParams(StageParams):
    learning_rate: float = 0.01
    epochs: int = 100


def run(params: TrainParams, ...):
    lr = params.learning_rate
    ...
```

### Step 4: Configure Remote

```bash
# Copy remote URL from DVC
dvc remote list
# myremote  s3://mybucket/cache

# Configure in Pivot
pivot config set remotes.origin s3://mybucket/cache
pivot config set default_remote origin
```

### Step 5: Run and Verify

```bash
# Run pipeline
pivot run

# Compare outputs with DVC
diff data/processed.csv data/processed.csv.dvc_backup
```

### Step 6: Export for Validation (Optional)

Pivot can export back to DVC format for validation:

```bash
pivot export

# Should show nothing needs to run
dvc repro --dry
```

## Running Side-by-Side

During migration, you can run both tools:

```bash
# Run with Pivot
pivot run

# Validate outputs match DVC
pivot export
dvc repro --dry  # Should show nothing to run
```

## Export Command

Export Pivot pipeline to DVC format:

```bash
# Generate dvc.yaml
pivot export

# Custom output path
pivot export --output my-pipeline.yaml

# Export specific stages
pivot export preprocess train
```

## Limitations of Export

The export captures:

- Stage commands (as Python function calls)
- Dependencies
- Outputs (with cache/persist settings)
- Metrics and plots

Not exported:

- Automatic code fingerprinting (DVC doesn't support this)
- Mutex groups
- Pydantic parameter types (exported as plain values)

## FAQs

### Do I need to migrate all at once?

No. You can migrate stage by stage. As long as output paths match, downstream DVC stages can consume Pivot outputs.

### What about my existing cache?

Pivot uses a different cache format. You'll need to re-run stages to populate the Pivot cache. Your DVC cache remains intact.

### Can I use params.yaml with Pivot?

Pivot supports `params.yaml` for overrides, but the primary source should be Python `StageParams` classes. This gives you type checking and IDE support.

### What about dvc plots and metrics?

Pivot has equivalent `Metric` and `Plot` output types. The workflow is similar:

```bash
# DVC
dvc metrics show
dvc plots show

# Pivot
pivot metrics show
pivot plots show
```
