# Watch Mode

Watch mode automatically re-runs your pipeline when files change.

## Basic Usage

```bash
pivot run --watch
```

Pivot will:

1. Run the pipeline once
2. Watch for file changes
3. Re-run affected stages when changes are detected

Press `Ctrl+C` to stop watching.

## What Gets Watched

Watch mode automatically monitors:

- **All declared dependencies** - Files and directories in your stage definitions
- **Stage source code** - The Python functions that define your stages
- **Configuration files** - `pivot.yaml` or `pipeline.py`

No manual pattern configuration is needed. Pivot derives what to watch from your pipeline definition.

## Debounce

Control how long to wait after changes before re-running:

```bash
# Default: 300ms
pivot run --watch

# Longer debounce for slow file systems
pivot run --watch --debounce 1000

# Immediate (no debounce)
pivot run --watch --debounce 0
```

## How It Works

1. **File system monitoring** - Uses `watchfiles` library for efficient OS-native watching
2. **Change detection** - Filters changes to relevant files (deps, source code)
3. **Output filtering** - Ignores changes to output files (prevents infinite loops)
4. **Debouncing** - Waits for changes to settle before re-running

## Example Workflow

```bash
# Terminal 1: Watch mode
pivot run --watch

# Terminal 2: Edit files
vim pipeline.py  # Save -> pipeline re-runs
vim data.csv     # Save -> affected stages re-run
```

## Combining with Other Options

```bash
# Watch with single-stage mode
pivot run --watch --single-stage train

# Watch with custom debounce
pivot run --watch --debounce 500
```

## Limitations

- **Not for production** - Watch mode is for development only
- **File system support** - Requires OS support for file watching (Linux inotify, macOS FSEvents)
- **Network drives** - May not work reliably on network file systems

## Troubleshooting

### Watch Mode Not Detecting Changes

**Symptom:** `pivot run --watch` doesn't re-run when files change.

**Causes and solutions:**

1. **File not in dependencies** - Verify the file is declared as a dependency:
   ```yaml
   stages:
     process:
       python: stages.process
       deps:
         config: config.yaml
         data: data/
   ```

2. **File outside project directory** - Watch mode only monitors files within the project

3. **Atomic saves** - Some editors use atomic saves (write to temp, then rename) which may need a brief delay. Try increasing debounce:
   ```bash
   pivot run --watch --debounce 500
   ```

### Lambda Causes Unnecessary Re-runs

**Symptom:** A stage re-runs every time even though nothing changed.

**Cause:** Lambda functions used in stage definitions have non-deterministic fingerprints. When Pivot fingerprints a lambda that doesn't have accessible source code, it falls back to using `id(func)`, which changes every time Python starts.

**Solution:** Use named module-level functions instead of lambdas:

```python
# Bad - lambda fingerprint is non-deterministic
filter_func = lambda x: x > 0.5

# Good - named function has stable AST-based fingerprint
def filter_positive(x: float) -> bool:
    return x > 0.5
```

## See Also

- [Watch Mode Tutorial](../tutorial/watch.md) - Getting started with watch mode
- [Architecture: Watch Engine](../architecture/watch.md) - How watch mode works internally
