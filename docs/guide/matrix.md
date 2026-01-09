# Matrix Stages

Matrix stages let you create multiple variants of the same stage with different configurations.

## Basic Usage

Use `@stage.matrix()` with a list of `Variant` objects:

```python
from pivot import stage, Variant

@stage.matrix([
    Variant(name='train', deps=['data/train.csv'], outs=['models/train.pkl']),
    Variant(name='test', deps=['data/test.csv'], outs=['models/test.pkl']),
])
def process(variant: str):
    print(f"Processing variant: {variant}")
```

This creates two stages:

- `process@train`
- `process@test`

## Variant Configuration

Each `Variant` can specify:

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Variant identifier (required) |
| `deps` | `Sequence[str]` | Input dependencies |
| `outs` | `Sequence[OutSpec]` | Output files |
| `params` | `BaseModel \| None` | Pydantic parameters |
| `mutex` | `Sequence[str]` | Mutex groups |
| `cwd` | `str \| Path \| None` | Working directory |

## Different Parameters per Variant

```python
from pydantic import BaseModel

class TrainParams(BaseModel):
    epochs: int = 100
    learning_rate: float = 0.01

@stage.matrix([
    Variant(
        name='quick',
        deps=['data.csv'],
        outs=['quick_model.pkl'],
        params=TrainParams(epochs=10, learning_rate=0.1)
    ),
    Variant(
        name='full',
        deps=['data.csv'],
        outs=['full_model.pkl'],
        params=TrainParams(epochs=1000, learning_rate=0.001)
    ),
])
def train(params: TrainParams, variant: str):
    print(f"{variant}: {params.epochs} epochs at lr={params.learning_rate}")
```

## Accessing the Variant Name

The `variant` parameter is automatically injected:

```python
@stage.matrix([
    Variant(name='v1', deps=['data_v1.csv'], outs=['out_v1.csv']),
    Variant(name='v2', deps=['data_v2.csv'], outs=['out_v2.csv']),
])
def process(variant: str):
    print(f"Current variant: {variant}")  # "v1" or "v2"
```

## YAML Configuration

Matrix stages can also be defined in `pivot.yaml`:

```yaml
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
      - "configs/${model}.yaml"
    outs:
      - "models/${model}_${dataset}.pkl"
    matrix:
      model:
        bert:
          params:
            hidden_size: 768
        gpt:
          params:
            hidden_size: 1024
          deps+:
            - data/gpt_tokenizer.json
      dataset: [swe, human]
```

This generates 4 stages:

- `train@bert_swe`
- `train@bert_human`
- `train@gpt_swe`
- `train@gpt_human`

### Matrix Expansion Rules

- Each dimension creates variants for all values
- Variants are combined as cross-product
- `deps+:` appends to base deps (instead of replacing)
- Variable substitution uses `${variable}` syntax

## Running Matrix Stages

```bash
# Run all variants
pivot run

# Run specific variant
pivot run train@bert_swe

# Run all variants of a base stage
pivot run "train@*"
```

## Listing Matrix Stages

```bash
pivot list
# Output:
# train@bert_swe
# train@bert_human
# train@gpt_swe
# train@gpt_human
```

## Variant Name Rules

Variant names must:

- Contain only alphanumeric characters, underscores, and hyphens
- Be at most 64 characters long
- Be unique within a matrix

```python
# Valid
Variant(name='train_v1')
Variant(name='bert-large')
Variant(name='2024')

# Invalid
Variant(name='')           # Empty
Variant(name='has spaces') # Spaces
Variant(name='has@at')     # @ is reserved
```

## See Also

- [API Reference: Variant](../reference/pivot/registry.md) - Full API documentation
