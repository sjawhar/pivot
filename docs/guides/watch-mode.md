# Watch Mode

Watch mode monitors your project for file changes and automatically re-runs affected stages. Combined with the interactive TUI, it gives you a live dashboard of your pipeline during development.

## Start Watching

```bash
pivot repro --watch
```

Pivot first runs the pipeline once with normal [skip detection](../concepts/caching.md), so stale stages run and cached ones are skipped (`--force` re-runs every stage on this first run). It then watches for changes. When you save a file, it re-runs only the stages affected by that change — the stage whose [dependency](../concepts/dependencies.md) changed, plus all downstream stages.

### With the Interactive TUI

```bash
pivot repro --watch --tui
```

This opens a two-panel interface:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stages (3)                          │  train  LIVE                 │
│  ────────────────────────────────────┼──────────────────────────────│
│  → $ train              0.3s         │  ┌─────┬───────┬────────┐   │
│    $ preprocess         0.1s         │  │ Logs│ Input │ Output │   │
│    ● load_data          0.2s         │  ├─────┴───────┴────────┘   │
│                                      │  │ (cached, no logs)        │
│  Watching for changes...             │                              │
└─────────────────────────────────────────────────────────────────────┘
```

**Left panel** — stage list with status icons:

| Icon | Meaning |
|------|---------|
| `$` | Cached (skipped) |
| `●` | Completed (ran this cycle) |
| `▶` | Running |
| `✗` | Failed |
| `○` | Pending |

**Right panel** — tabbed detail view for the selected stage: logs, input diff (what changed), and output diff (what was produced).

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `j` / `k` | Navigate stages |
| `Tab` | Switch tabs (Logs → Input → Output) |
| `[` / `]` | Browse execution history |
| `c` | Commit pending lock file changes |
| `?` | Show all shortcuts |
| `q` | Quit |

## What Triggers a Re-Run

Watch mode reacts to **external** file changes — files you edit in your editor. It automatically ignores changes to output files produced by stages, so a stage writing `output.csv` doesn't trigger an infinite loop.

Changes that trigger re-runs:

- Editing an input file declared with `pipeline.input()`
- Editing a file inside a directory declared with `pipeline.input()` reruns that directory's
  consumer, alongside any stage that declares the file directly
- Editing through a symlinked path reruns every stage that consumes the same underlying input file
- Editing a stage's Python source code (detected by [fingerprinting](../concepts/fingerprinting.md))
- Editing parameter files

Changes that do **not** trigger re-runs:

- Files written by stages (output filtering)
- Files outside the watch scope
- Changes to `.pivot/` internal state

## Scope the Watch

By default, watch mode runs the entire pipeline. Target specific stages to narrow the scope:

```bash
# Watch and re-run only demo/train and its dependencies
pivot repro demo/train --watch

# Watch multiple stages
pivot repro demo/train demo/evaluate --watch
```

Only the targeted stages (and their upstream [dependencies](../concepts/dependencies.md)) are monitored and re-run.

## Debounce

Some editors trigger multiple file-save events in quick succession. The debounce delay controls how long Pivot waits after the last change before triggering a re-run:

```bash
# Longer debounce for network drives or slow editors (ms)
pivot repro --watch --debounce 1000

# Shorter debounce for fast iteration
pivot repro --watch --debounce 100
```

The default is 300ms, configurable via `pivot config set watch.debounce <ms>`.

## Display Modes

| Command | Output |
|---------|--------|
| `pivot repro --watch` | Plain text — stage status printed to terminal |
| `pivot repro --watch --tui` | Interactive TUI with panels and keyboard navigation |
| `pivot repro --watch --jsonl` | JSONL stream — one JSON object per event (for tooling) |
| `pivot repro --watch --show-output` | Plain text with stage stdout/stderr streamed live |

## Serve Mode

For headless environments (CI watchers, remote dev servers), serve mode runs the watch loop with a Unix socket for programmatic control:

```bash
pivot repro --watch --serve
```

This creates `.pivot/agent.sock` — a JSON-RPC 2.0 endpoint that the TUI or external tools can connect to for status queries and triggering runs.

## Error Handling

By default, watch mode stops on the first failure (`--fail-fast`). Use `--keep-going` to continue running independent stages after a failure — only downstream dependents of the failed stage are skipped:

```bash
pivot repro --watch --keep-going
```

If a stage reads an input file that does not exist yet, for example because you saved `pipeline.py` with a new stage before creating its file, watch mode logs `Pipeline invalid - waiting for fix` with the missing path, runs nothing and keeps watching. Creating the file runs the stages that were waiting for it, also when the file arrives inside a new directory (`mkdir -p`, `mv`, `cp -r` or `tar -x`). With `--all --keep-going`, a stage whose input is missing is instead skipped with a `Skipping:` warning while the other stages run; watch mode waits as above only when no stage is left to run.

## Tips

- **Keep stages small.** Smaller stages mean faster feedback loops — only the changed part re-runs.
- **Split data loading from processing.** Expensive data reads cached separately from the logic you're iterating on.
- **Use `--no-commit` during exploration.** Skip writing lock files while you're experimenting, then `pivot commit` when you're satisfied.

## Related

- [Fingerprinting](../concepts/fingerprinting.md) — how code change detection works
- [Caching](../concepts/caching.md) — skip detection and when stages re-run
- [Artifacts & DAG](../concepts/artifacts-and-dag.md) — how dependencies determine execution order
- [TUI Guide](./tui.md) — detailed TUI usage
