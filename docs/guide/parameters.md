# Parameters

Pivot uses Pydantic models for type-safe, validated parameters.

## Basic Usage

Define a Pydantic model and pass it to `@stage`:

```python
from pydantic import BaseModel
from pivot import stage

class TrainParams(BaseModel):
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32

@stage(deps=['data.csv'], outs=['model.pkl'], params=TrainParams)
def train(params: TrainParams):
    print(f"Training with lr={params.learning_rate}")
    print(f"Epochs: {params.epochs}")
```

## Parameter Injection

When a stage has `params`, Pivot automatically injects the parameter instance:

```python
@stage(params=TrainParams)
def train(params: TrainParams):
    # params is automatically provided by Pivot
    pass
```

The function **must** have a `params` parameter when using `params=` in the decorator.

## Overriding Defaults

Override parameter defaults via `params.yaml`:

```yaml
# params.yaml
train:
  learning_rate: 0.001
  epochs: 200
```

Priority (highest to lowest):

1. `params.yaml` values
2. Model defaults

## Parameter Change Detection

Pivot tracks parameter changes and re-runs stages when parameters change:

```bash
# Change params.yaml
$ echo "train:\n  learning_rate: 0.005" > params.yaml

# Pivot detects the change
$ pivot explain train
Stage: train
  Status: WILL RUN
  Reason: Parameters changed

  Param changes:
    learning_rate: 0.01 -> 0.005
```

## Pre-configured Instances

Pass a pre-configured instance instead of a class:

```python
@stage(
    deps=['data.csv'],
    params=TrainParams(learning_rate=0.001, epochs=50)
)
def train(params: TrainParams):
    # Uses the pre-configured values
    # params.yaml can still override
    pass
```

## Validation

Pydantic validates parameters automatically:

```python
from pydantic import BaseModel, Field

class TrainParams(BaseModel):
    learning_rate: float = Field(gt=0, le=1)  # Must be 0 < lr <= 1
    epochs: int = Field(ge=1)                  # Must be >= 1

# Invalid params.yaml will raise ValidationError
```

## Viewing Parameters

```bash
# Show current parameter values
pivot params show

# JSON output
pivot params show --json

# Compare with git HEAD
pivot params diff
```

## Matrix Stage Parameters

Each matrix variant can have different parameters:

```python
from pivot import stage, Variant

@stage.matrix([
    Variant(
        name='small',
        deps=['data.csv'],
        params=TrainParams(epochs=10)
    ),
    Variant(
        name='large',
        deps=['data.csv'],
        params=TrainParams(epochs=1000)
    ),
])
def train(params: TrainParams, variant: str):
    print(f"Variant {variant}: {params.epochs} epochs")
```

## See Also

- [API Reference: parameters](../reference/pivot/parameters.md) - Full API documentation
