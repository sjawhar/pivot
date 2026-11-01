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
pivot verify demo/train demo/evaluate

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
      "name": "demo/preprocess",
      "status": "passed",
      "reason": "",
      "missing_files": []
    },
    {
      "name": "demo/train",
      "status": "failed",
      "reason": "Missing files: data/demo/train.jsonl",
      "missing_files": ["data/demo/train.jsonl"]
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

# Include the update check of each imported artifact
pivot status --json --check-imports
```

The JSON output includes a `stages` array with status (`cached`, `stale`, `not_expanded` for a mapped stage whose producer is not current, or `orphaned` for an instance of a mapped stage whose key left its producer's listing, which the next run that expands that mapped stage removes), a `tracked_files` array, and a `suggestions` list. With `--explain`, each stage entry includes `will_run`, `code_changes`, `param_changes`, `dep_changes`, and `upstream_stale` fields. Each `dep_changes` entry names its artifact as `"identity": {"producer": ..., "key": ...}`, with `key` null for an input or a single-output stage. A stale stage that need not run has a `recovery` field, with or without `--explain`: `pull_outputs` (its lock is current, and `pivot pull` restores its missing outputs), `pull_reads` (`pivot pull` fetches objects it read by hash that the cache lacks), or `upstream` (it is current on its own, and stale only because a stage upstream of it is). The field is present only when that action recovers every reason the stage is stale, its upstream stages included: a stage with a missing output beside a hand-edited metric, or downstream of a stage that must run, has none. `pull_outputs` does not check that a remote is configured; `pull_reads` does. A stale stage with no `recovery` must run. When a remote is configured, the `pivot pull` suggestion counts the `pull_outputs` and `pull_reads` stages.

With `--check-imports`, an `imports` array holds one entry per imported artifact: `path`, `status` (`up_to_date`, `update_available` or `error`), `current_rev` and `latest_rev` (the first 8 characters of each commit, set for `update_available`), and `error` (the message, or the exception's type name when it has no message, set for `error`); a field that does not apply is null.

Pivot writes exactly one JSON document to standard output for every option combination. Notes such as "No remotes configured" and warnings go to standard error. A failed run (for example `--remote-only` with no remote configured) exits non-zero, with its error on standard error and nothing on standard output.

Every document is valid RFC 8259 JSON, so a strict parser (JavaScript's `JSON.parse`, Go's `encoding/json`, Rust's `serde_json`) reads it as Python's `json` module and jq do. A value that JSON has no number for is written as a string: a non-finite float (for example a stage parameter set to `float("inf")`) reads as `"Infinity"`, `"-Infinity"` or `"NaN"` — the same names Python used to write bare — in every `--json` document and every `--jsonl` line. A reader that expects a number for such a field must accept these three strings.

A URL that Pivot prints has its credentials hidden: `https://user:token@host/repo.git` reads as `https://***@host/repo.git`, in JSON, in text and in warnings. The configured URL itself is unchanged, and a URL without credentials is printed byte for byte.

### Quiet Mode

For scripts that only need an exit code:

```bash
# Exit 0 if clean, exit 1 if stale stages or modified tracked files
pivot --quiet status
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
        run: pivot --quiet status
```

If a stage's params hold a secret (`pydantic.SecretStr` and friends), the lock file
records a digest keyed by the project's secret key. Set `PIVOT_SECRET_KEY` in CI to the
key the lock files were written with (the contents of a developer's `<state dir>/secret.key`,
stored as a CI secret); without it, CI makes its own key and those stages show as stale.
`core.state_dir` defaults to `.pivot`. See
[secret parameters](../concepts/parameters.md#secret-parameters).

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

- **Always commit lock files to version control.** Lock files (in
  `<state dir>/stages/`, where `core.state_dir` defaults to `.pivot`) record what was executed.
  Without them, `verify` has nothing to check against.
- **Use `--allow-missing` in CI.** It avoids downloading large data files while still confirming reproducibility.
- **Run `pivot push` as part of your local workflow.** The CI verify step depends on outputs being available on remote.
- **Separate verify from execution.** CI should never run `pivot repro` — that belongs in the development workflow. CI only checks that developers did their job.

## Related

- [Remote Storage](remote-storage.md) — configuring S3 and GCS remotes for `push`/`pull`
- [Caching & Skip Detection](../concepts/caching.md) — how Pivot decides what to re-run
- [Fingerprinting](../concepts/fingerprinting.md) — how code changes are detected
