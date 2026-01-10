# Pipeline Configuration

Pivot supports three ways to define pipelines, discovered in this order:

1. `pivot.yaml` / `pivot.yml` - YAML configuration
2. `pipeline.py` - Python file with decorators or Pipeline class

## Decorator-Based (Recommended)

The simplest approach using `@stage` decorators:

```python
# pipeline.py
from pivot import stage

@stage(deps=['data.csv'], outs=['processed.parquet'])
def preprocess():
    import pandas as pd
    df = pd.read_csv('data.csv')
    df.to_parquet('processed.parquet')

@stage(deps=['processed.parquet'], outs=['model.pkl'])
def train():
    import pandas as pd
    import pickle
    df = pd.read_parquet('processed.parquet')
    with open('model.pkl', 'wb') as f:
        pickle.dump({'rows': len(df)}, f)
```

```bash
pivot run  # Auto-discovers pipeline.py
```

## Pipeline Class

For dynamic pipeline construction:

```python
# pipeline.py
from pivot import Pipeline

def preprocess():
    import pandas as pd
    df = pd.read_csv('data.csv')
    df.to_parquet('processed.parquet')

def train():
    import pandas as pd
    import pickle
    df = pd.read_parquet('processed.parquet')
    with open('model.pkl', 'wb') as f:
        pickle.dump({'rows': len(df)}, f)

pipeline = Pipeline()
pipeline.add_stage(preprocess, deps=['data.csv'], outs=['processed.parquet'])
pipeline.add_stage(train, deps=['processed.parquet'], outs=['model.pkl'])
```

### Pipeline.add_stage()

```python
pipeline.add_stage(
    func,                      # The function to run
    name='custom_name',        # Optional custom name
    deps=['input.csv'],        # Dependencies
    outs=['output.csv'],       # Outputs (strings or Out/Metric/Plot)
    metrics=['metrics.json'],  # Shorthand for Metric()
    plots=['plot.png'],        # Shorthand for Plot()
    params=MyParams,           # Pydantic model
    mutex=['gpu'],             # Mutex groups
    cwd='subdir/',             # Working directory
)
```

## YAML Configuration

For teams that prefer configuration files:

```yaml
# pivot.yaml
stages:
  preprocess:
    python: stages.preprocess    # Module path to function
    deps:
      - data.csv
    outs:
      - processed.parquet

  train:
    python: stages.train
    deps:
      - processed.parquet
    outs:
      - model.pkl
    params:
      learning_rate: 0.01
```

### YAML Schema

```yaml
stages:
  stage_name:
    python: module.function      # Required: function to call
    deps:                        # Optional: input dependencies
      - path/to/file
    outs:                        # Optional: output files
      - path/to/output
    metrics:                     # Optional: metric files (git-tracked)
      - metrics.json
    plots:                       # Optional: plot files
      - loss.png
    params:                      # Optional: parameter overrides
      key: value
    mutex:                       # Optional: mutex groups
      - gpu
    cwd: subdir/                 # Optional: working directory
    matrix:                      # Optional: matrix expansion
      dimension:
        variant1: {}
        variant2: {}
```

### Matrix in YAML

```yaml
stages:
  train:
    python: stages.train
    deps:
      - "data/${dataset}.csv"
    outs:
      - "models/${model}_${dataset}.pkl"
    matrix:
      model: [bert, gpt]
      dataset: [train, test]
```

Generates: `train@bert_train`, `train@bert_test`, `train@gpt_train`, `train@gpt_test`

## Discovery Order

Pivot searches for pipeline definitions in this order:

1. `pivot.yaml`
2. `pivot.yml`
3. `pipeline.py`

The first file found is used. You can have only one pipeline definition.

## Mixing Approaches

You can combine YAML with decorators, but be careful about conflicts:

```yaml
# pivot.yaml - defines some stages
stages:
  preprocess:
    python: stages.preprocess
    deps: [data.csv]
    outs: [processed.parquet]
```

```python
# stages.py - function referenced by YAML
def preprocess():
    pass

# Additional stages via decorators
from pivot import stage

@stage(deps=['processed.parquet'], outs=['model.pkl'])
def train():
    pass
```

## See Also

- [API Reference: Pipeline](../reference/pivot/pipeline.md) - Full API documentation
