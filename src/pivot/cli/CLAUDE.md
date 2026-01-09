# Pivot CLI - Development Guidelines

## Input Validation (Critical)

**Validate inputs as early as possible** - use Click's built-in validation in option/argument decorators.

### Numeric Options

Always use `click.IntRange` or `click.FloatRange` for numeric inputs:

```python
# Good - validates at argument parsing time
@click.option("--precision", type=click.IntRange(min=0), default=5)
@click.option("--jobs", type=click.IntRange(min=1), default=20)
@click.option("--debounce", type=click.IntRange(min=0), default=300)

# Bad - allows invalid values through
@click.option("--precision", type=int, default=5)  # Allows negative!
```

### Path Options

Use `click.Path` with appropriate parameters:

```python
@click.option("--output", type=click.Path(path_type=pathlib.Path))
@click.option("--config", type=click.Path(exists=True, dir_okay=False))
```

### Choice Options

Use `click.Choice` for limited valid values:

```python
@click.option("--format", type=click.Choice(["json", "yaml", "csv"]))
```

### Why Early Validation Matters

1. **Better error messages** - Click provides user-friendly error messages automatically
2. **Fail fast** - Don't waste time processing before discovering invalid input
3. **Type safety** - Validated inputs have correct types in the function body
4. **Consistency** - Users get the same error format for all validation failures
