# Defining Stages

The `@stage` decorator is the primary way to define pipeline stages in Pivot.

## Basic Usage

```python
from pivot import stage

@stage(deps=['input.csv'], outs=['output.parquet'])
def process():
    import pandas as pd
    df = pd.read_csv('input.csv')
    df.to_parquet('output.parquet')
```

## Decorator Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `deps` | `Sequence[str]` | Input file dependencies |
| `outs` | `Sequence[str \| Out \| Metric \| Plot]` | Output files |
| `params` | `type[BaseModel] \| BaseModel` | Pydantic parameters |
| `mutex` | `Sequence[str]` | Mutex groups for exclusive execution |
| `name` | `str \| None` | Custom stage name (default: function name) |
| `cwd` | `str \| Path \| None` | Working directory for paths |

## Dependencies

Dependencies are files that your stage reads:

```python
@stage(deps=['data/raw.csv', 'config/settings.yaml'])
def preprocess():
    # Both files are tracked for changes
    pass
```

!!! tip "Automatic Code Dependencies"
    Pivot automatically tracks Python code dependencies. You don't need to list `.py` files in `deps`.

## Outputs

Outputs are files that your stage writes:

```python
from pivot import stage, Out, Metric, Plot

@stage(
    deps=['data.csv'],
    outs=[
        Out('model.pkl'),           # Cached output
        Metric('metrics.json'),     # Git-tracked metrics
        Plot('loss.png'),           # Cached plot
    ]
)
def train():
    pass
```

See [Output Types](outputs.md) for details on each type.

## Parameters

Use Pydantic models for type-safe parameters:

```python
from pydantic import BaseModel

class TrainParams(BaseModel):
    learning_rate: float = 0.01
    batch_size: int = 32

@stage(deps=['data.csv'], params=TrainParams)
def train(params: TrainParams):
    print(f"Learning rate: {params.learning_rate}")
```

See [Parameters](parameters.md) for details.

## Mutex Groups

Prevent stages from running concurrently:

```python
@stage(deps=['data.csv'], outs=['gpu_model.pkl'], mutex=['gpu'])
def train_gpu():
    # Uses GPU
    pass

@stage(deps=['data.csv'], outs=['gpu_model_2.pkl'], mutex=['gpu'])
def train_gpu_2():
    # Also uses GPU - won't run at same time as train_gpu
    pass
```

## Custom Stage Names

```python
@stage(deps=['data.csv'], name='my_custom_name')
def some_function():
    pass
```

The stage will be registered as `my_custom_name` instead of `some_function`.

## Working Directory

Set a working directory for path resolution:

```python
@stage(deps=['data.csv'], outs=['output.csv'], cwd='subproject/')
def process():
    # Paths are relative to subproject/
    # Stage executes from subproject/
    pass
```

## Function Requirements

Stage functions must be **pure and serializable** for multiprocessing.

### Why Serialization Matters

Pivot uses `ProcessPoolExecutor` with `forkserver` context for true parallelism. Worker processes are separate Python interpreters that receive serialized (pickled) functions. This means:

1. Functions must be defined at module level (not inside other functions)
2. Functions cannot capture local variables (closures)
3. The module containing the function must be importable

### Do

```python
# Module-level function in importable module
@stage(deps=['data.csv'], outs=['output.csv'])
def process_data():
    import pandas as pd
    df = pd.read_csv('data.csv')
    df.to_csv('output.csv')
```

### Don't

```python
# Lambda (not picklable)
# Error: Can't pickle <lambda>
process = stage(deps=['x'], outs=['y'])(lambda: ...)

# Closure capturing local variable
# Error: Could not pickle the task to send it to the workers
def make_stage(threshold):
    @stage(deps=['x'], outs=['y'])
    def process():
        if value > threshold:  # Captures threshold!
            pass
    return process

# Defined in __main__
# Error: Can't get attribute 'my_stage' on <module '__main__'>
if __name__ == '__main__':
    @stage(deps=['x'], outs=['y'])
    def my_stage():  # Can't be pickled!
        pass
```

!!! tip "Use Parameters Instead of Closures"
    If you need to configure stage behavior, use Pydantic parameters:

    ```python
    class ProcessParams(pydantic.BaseModel):
        threshold: float = 0.5

    @stage(deps=['data.csv'], params=ProcessParams)
    def process(params: ProcessParams):
        if value > params.threshold:
            pass
    ```

See [Troubleshooting](troubleshooting.md) for more common errors and solutions.

## See Also

- [API Reference: stage](../reference/pivot/registry.md) - Full API documentation
