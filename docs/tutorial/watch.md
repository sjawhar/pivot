# Watch Mode & Rapid Iteration

This tutorial extends the [Quick Start](../getting-started/quickstart.md) pipeline to show how watch mode enables rapid iteration during development.

## Prerequisites

Complete the Quick Start tutorial. You should have:

- `pivot.yaml` with preprocess and train stages
- `stages.py` with the stage functions
- `data.csv` with sample data

## Start Watch Mode

Instead of running once, keep Pivot watching for changes:

```bash
pivot run --watch
```

Pivot will:

1. Run the pipeline once
2. Start watching for file changes
3. Automatically re-run affected stages when you save files

## The TUI

When running in a terminal, watch mode shows an interactive TUI with a two-panel layout:

```
┌─────────────────────────────────────────────────────────────────────┐
│  pivot run --watch                                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Stages (2) $2                         │  train $ LIVE              │
│  ─────────────────────────────────────┼──────────────────────────────│
│  → $ train              0.3s           │  ┌─────┬───────┬────────┐   │
│    $ preprocess         0.1s           │  │ Logs│ Input │ Output │   │
│                                        │  ├─────┴───────┴────────┘   │
│                                        │  │ (cached, no logs)        │
│                                        │                              │
│  Watching for changes...               │                              │
└─────────────────────────────────────────────────────────────────────┘
```

**Key features:**

- **Left panel:** Stage list with status symbols (`$` cached, `●` completed, `▶` running, `✗` failed)
- **Right panel:** Tabbed view with Logs, Input (what changed), and Output (files produced)
- **Selection indicator:** `→` shows the currently selected stage

**Essential shortcuts:**

- `j`/`k` - Navigate stages
- `Tab` - Switch tabs (Logs → Input → Output)
- `[`/`]` - View execution history
- `c` - Commit pending changes
- `?` - Show all keyboard shortcuts
- `q` - Quit watch mode

## Edit Code, See Auto-Rerun

Open `stages.py` in your editor. Change the preprocess function:

```python
def preprocess(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> PreprocessOutputs:
    df = raw.dropna()
    df['tripled'] = df['value'] * 3  # Changed from 2 to 3
    out_path = pathlib.Path("processed.parquet")
    df.to_parquet(out_path)
    return {"clean": out_path}
```

Save the file. Watch mode detects the change and re-runs:

```
│  ▶ preprocess           ...            │
│  ○ train                               │
```

Then:

```
│  ● preprocess         0.5s             │
│  ▶ train              ...              │
```

Both stages re-run because `train` depends on `preprocess`'s output.

## Edit Data, Re-run Downstream

Now edit `data.csv`:

```csv
name,value
Alice,100
Bob,200
Charlie,300
David,400
```

Save. Only `preprocess` and `train` re-run (both depend on `data.csv`).

## Output Filtering

Watch mode automatically filters output files. If `preprocess` writes `processed.parquet`, that change won't trigger another run. Only **external** changes (your edits) trigger re-execution.

## Debounce Control

Some editors save multiple times in quick succession. The `--debounce` flag controls how long to wait after a file change before triggering a re-run:

```bash
# Longer debounce for slow editors/network drives
pivot run --watch --debounce 1000

# Shorter debounce for fast iteration
pivot run --watch --debounce 100
```

Default is 300ms.

> **Note:** Debounce is currently handled by the underlying file watcher (watchfiles). Custom debounce values may not take effect in all situations.

## Plain Text Mode

For CI logs or non-TTY environments:

```bash
pivot run --watch --display plain
```

## Best Practices

1. **Keep stages small** - Smaller stages mean faster iteration cycles
2. **Use PathOnly for large files** - Let your code handle I/O for better performance
3. **Split data prep from training** - Cache expensive preprocessing

## Next Steps

- [Parameters & Experiments](parameters.md) - Run experiments with different settings
- [Watch Mode Reference](../reference/watch.md) - Full configuration options
