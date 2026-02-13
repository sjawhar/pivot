# CI Integration

Pivot pipelines are reproducible by design — the same inputs and code always produce the same outputs. This makes CI verification straightforward: check that stages are cached and outputs exist, without re-running anything.

## Core Workflow

A typical CI pipeline has three steps:

1. **Restore cache** — pull cached outputs from remote storage
2. **Verify** — confirm all stages are cached and outputs are available
3. **Push cache** — after merging, push any new outputs to remote

```text
PR opened → pivot pull → pivot verify → ✓ merge
                                        ↓
                              pivot push (post-merge)
```

The key insight: developers run `pivot repro` locally and push outputs with `pivot push`. CI never re-executes stages — it only verifies that execution already happened.

## `pivot verify`

The `verify` command checks that all stages are cached (code, params, and deps match lock files) and that output files exist locally or on remote.

```bash
# Verify all stages
pivot verify

# Verify specific stages
pivot verify train evaluate

# Allow outputs to exist only on remote (not locally)
pivot verify --allow-missing

# Machine-readable output
pivot verify --json
```

**Exit codes:**

| Code | Meaning |
|------|---------|
| `0` | All stages pass verification |
| `1` | One or more stages are stale or have missing outputs |

### `--allow-missing` Mode

Without `--allow-missing`, verify requires output files to exist in the local cache. With `--allow-missing`, it checks both local cache **and** remote storage — dependencies and outputs that are missing locally pass if they exist on the configured remote.

This is ideal for CI environments where you don't want to download all data:

```bash
# CI: verify outputs exist somewhere (local or remote)
pivot verify --allow-missing
```

!!! note
    `--allow-missing` requires at least one remote to be configured. If no remotes exist, the command fails with an error.

### JSON Output

The `--json` flag produces structured output for programmatic consumption:

```json
{
  "passed": false,
  "stages": [
    {
      "name": "preprocess",
      "status": "passed",
      "reason": "",
      "missing_files": []
    },
    {
      "name": "train",
      "status": "failed",
      "reason": "Missing files: model.pkl",
      "missing_files": ["model.pkl"]
    }
  ]
}
```

## `pivot status --json`

For more detailed pipeline state inspection, `pivot status` with `--json` returns stage status, tracked file state, and optionally remote sync info:

```bash
# Full status as JSON
pivot status --json

# Only pipeline stage status
pivot status --json --stages-only

# Include remote sync counts
pivot status --json --remote

# Detailed explanation of why stages would run
pivot status --json --explain
```

The JSON output includes a `stages` array with status (`cached` or `stale`), a `tracked_files` array, and a `suggestions` list. With `--explain`, each stage entry includes `will_run`, `code_changes`, `param_changes`, `dep_changes`, and `upstream_stale` fields.

### Quiet Mode

For scripts that only need an exit code:

```bash
# Exit 0 if clean, exit 1 if stale stages or modified tracked files
pivot status --quiet
```

## GitHub Actions Example

```yaml
name: Pipeline Verification

on:
  pull_request:
    branches: [main]

jobs:
  verify-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Install dependencies
        run: |
          uv sync --active

      - name: Configure remote
        run: |
          pivot config set remotes.s3 s3://my-bucket/pivot-cache
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Pull cached outputs
        run: pivot pull

      - name: Verify pipeline
        run: pivot verify --allow-missing

      - name: Check tracked files
        run: pivot status --quiet
```

### Post-Merge Cache Push

Add a separate workflow triggered on `push` to `main` that runs `pivot push` after merging. This ensures future CI runs and teammates can pull the latest outputs. The setup steps (checkout, Python, install, configure remote) are the same as above.

## Debugging CI Failures

When `pivot verify` fails in CI:

```bash
# See which stages are stale and why
pivot status --explain

# JSON for scripting
pivot verify --json | jq '.stages[] | select(.status == "failed")'
```

Common causes:

| Symptom | Cause | Fix |
|---------|-------|-----|
| Stage is "stale" | Code or params changed without re-running | Run `pivot repro` locally, commit lock files |
| Missing output files | Outputs weren't pushed to remote | Run `pivot push` after `pivot repro` |
| Dep hash mismatch | Input data changed without re-running | Run `pivot repro` to regenerate outputs |

## Tips

- **Always commit lock files to version control.** Lock files (in `.pivot/stages/`) record what was executed. Without them, `verify` has nothing to check against.
- **Use `--allow-missing` in CI.** It avoids downloading large data files while still confirming reproducibility.
- **Run `pivot push` as part of your local workflow.** The CI verify step depends on outputs being available on remote.
- **Separate verify from execution.** CI should never run `pivot repro` — that belongs in the development workflow. CI only checks that developers did their job.

## Related

- [Remote Storage](remote-storage.md) — configuring S3 remotes for `push`/`pull`
- [Caching & Skip Detection](../concepts/caching.md) — how Pivot decides what to re-run
- [Fingerprinting](../concepts/fingerprinting.md) — how code changes are detected
