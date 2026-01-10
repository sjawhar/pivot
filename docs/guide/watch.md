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

## Custom Patterns

Watch specific file patterns:

```bash
# Watch only Python and CSV files
pivot run --watch "*.py,*.csv"

# Watch specific directories
pivot run --watch "src/**/*.py,data/**/*.csv"
```

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

# Watch with verbose output
pivot run --watch --verbose

# Watch specific patterns with custom debounce
pivot run --watch "src/**/*.py" --debounce 500
```

## Limitations

- **Not for production** - Watch mode is for development only
- **File system support** - Requires OS support for file watching (Linux inotify, macOS FSEvents)
- **Network drives** - May not work reliably on network file systems
