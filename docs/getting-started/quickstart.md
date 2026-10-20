# Quick Start

This guide builds and runs a two-stage pipeline.

## Mental Model

Think **artifact-first**, not **stage-first**. A stage is a function from data to data; calling one stage with the result of another is what creates the dependency:

- **Wrong:** "Stage A triggers Stage B"
- **Right:** "This file changed. What needs to happen because of that?"

Invalidation is content-addressed: same inputs + same code + same parameters = same outputs.

## 1. Initialize the Project

Pivot projects live in a git repository:

```bash
mkdir quickstart && cd quickstart
git init
pivot init
```

`pivot init` creates:

- `.pivot/`: the project marker and its `config.yaml`
- `.pivot/.gitignore`: by default, keeps the state directory's cache, state database, artifact
  locks and agent socket out of git
- `.pivotignore`: patterns for files watch mode ignores

`core.state_dir` selects where Pivot writes state and defaults to `.pivot`. Commit the lock files
under `<state dir>/stages/`; they record what ran and what it produced.

## 2. Add Input Data

Raw inputs live under `data/raw/`:

```bash
mkdir -p data/raw
cat > data/raw/records.csv << 'EOF'
name,value
Alice,100
Bob,200
Charlie,
EOF
```

## 3. Create the Pipeline

Create `pipeline.py`:

```python
# pipeline.py
import pandas as pd

from pivot import Pipeline, StageParams, stage


class CleanParams(StageParams):
    min_value: float = 0.0


@stage
def clean(records: pd.DataFrame, params: CleanParams) -> pd.DataFrame:
    """Remove rows with missing values and values below the threshold."""
    cleaned = records.dropna()
    return cleaned[cleaned["value"] >= params.min_value]


@stage
def summarize(rows: pd.DataFrame) -> dict:
    """Compute summary statistics."""
    values = rows["value"]
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


with Pipeline("quickstart") as pipeline:
    records = pipeline.input("records.csv")
    cleaned = clean(records, params=CleanParams(min_value=0.0))
    summarize(cleaned)
```

What each part does:

- `@stage` marks a function as a stage. Outside a pipeline block it is an ordinary function you can call in tests.
- `CleanParams` is the stage's configuration, a Pydantic model. Each stage that needs configuration gets its own `StageParams` subclass.
- `pipeline.input("records.csv")` declares the raw input `data/raw/records.csv`. Its `.csv` extension tells Pivot to load it as a DataFrame.
- `clean(records, ...)` inside the `with` block does not run `clean`. It records a stage and returns a handle to its output; passing that handle to `summarize` makes `summarize` depend on `clean`.
- The return type chooses the file format: a `DataFrame` is saved as JSON Lines, a `dict` as YAML.

## 4. Run the Pipeline

```bash
pivot repro
```

Pivot will:

1. Find `pipeline.py` and run it to build the DAG
2. Run `quickstart/clean`, then `quickstart/summarize`
3. Write `data/quickstart/clean.jsonl` and `data/quickstart/summarize.yaml`
4. Record each stage's code, parameters and input hashes in
   `<state dir>/stages/quickstart/`

```bash
cat data/quickstart/summarize.yaml
```

## 5. Re-run (Cached)

```bash
pivot repro
```

Nothing changed, so both stages are reported as cached and nothing executes.

## 6. Change Code and Re-run

Edit `summarize` in `pipeline.py` to also report the total:

```python
@stage
def summarize(rows: pd.DataFrame) -> dict:
    """Compute summary statistics."""
    values = rows["value"]
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "total": float(values.sum()),
    }
```

```bash
pivot status
pivot repro
```

`pivot status` reports `quickstart/summarize` as stale ("Code changed"); `clean` stays cached. `pivot repro` runs only `summarize`.

## 7. Change a Parameter

Parameters can be overridden per stage in `params.yaml` without touching code:

```bash
cat > params.yaml << 'EOF'
quickstart/clean:
  min_value: 150
EOF
pivot status --explain
```

The explanation shows the parameter change for `quickstart/clean` and marks `summarize` as stale because `clean` will produce different output bytes. Run `pivot repro` to apply it, or delete `params.yaml` to return to the values in code.

## Next Steps

- [Artifacts & the DAG](../concepts/artifacts-and-dag.md): how handles form the graph
- [Pipelines](../concepts/pipelines.md): inputs, variants, and multiple pipelines
- [Outputs](../concepts/outputs.md): multi-output stages, metrics, and plots
- [Watch Mode](../guides/watch-mode.md): re-run automatically as you edit
