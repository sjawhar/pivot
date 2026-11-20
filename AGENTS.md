# Pivot - Project Rules

Python 3.13+, Unix only, 90%+ test coverage. Pre-alpha: breaking changes are fine, and no migration code or
compatibility shims are needed.

## Where things are

- `packages/pivot/src/pivot/`: the library and the `pivot` CLI. Package guides: `cli/AGENTS.md`,
  `executor/AGENTS.md`, `remote/AGENTS.md`, `storage/AGENTS.md`; tests: `packages/pivot/tests/AGENTS.md`.
- `packages/pivot-tui/src/pivot_tui/`: the Textual TUI, a client of the engine's JSON-RPC server
  (`pivot_tui/AGENTS.md`).
- `docs/`: the published documentation; `docs/architecture/` explains the engine, execution, storage, fingerprinting and
  watch mode. `docs/solutions/` holds past learnings: read the ones on your topic before you change
  multiprocessing and worker lifecycle, typing, fingerprinting, LMDB/StateDB, or path resolution.
- The user-facing guide to writing pipelines is `skills/writing-pivot-stages/SKILL.md`.

## Development

```bash
uv sync --active                                                                                # install
PIVOT_TEST_UV="$(command -v uv)" uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto  # test
uv run ruff format . && uv run ruff check . && uv run basedpyright                              # quality
```

Run all three before you push or return work to the user. With `CI` set, the tests that install pivot into a
fresh venv also run and need `PIVOT_TEST_UV` (`tests.helpers.venv_pivot`), and
`test_helper_edited_in_the_second_its_process_started_on_a_whole_second_file_system` needs `mkfs.ext4` and
passwordless `sudo`. Test conventions: `packages/pivot/tests/AGENTS.md`.

## Core design

- Artifact-first: the DAG comes from artifact dependencies, not explicit wiring. Ask "this file changed; what has to
  happen because of it?"
- The registry, built from `pipeline.py`, is the one source of stage metadata (deps, outs, tags, params). Lock files
  record execution state (hashes, generations), never stage definitions: do not infer a stage's properties from a
  lock file.
- Invalidation is content-addressed: same inputs and same code give the same outputs. Skip detection is generation
  tracking, then lock-file comparison, then the run cache (`docs/architecture/execution.md`; StateDB prefixes in
  `docs/architecture/storage.md`).
- The `explain` and `run` paths use the same skip-decision code; never give one of them its own copy.

## Fingerprinting

- Fingerprint checks are cheap: O(1) shallow or static checks, no deep recursive traversal.
- A soundness risk (a change could go undetected) is an error; a noise risk (only spurious reruns) is a warning.
  Unsafe fingerprinting is opt-in through one config flag; the default is strict.
- Collect every fingerprinting error before reporting. A message names the captured variable or type and suggests
  `StageParams`.
- The manifest cache (`sm:`) is best-effort: correctness never depends on it.

## Architecture

- Avoid module-level mutable state: use `contextvars.ContextVar` or explicit parameters for context such as the
  current stage.
- Keep LMDB read transactions short: materialise a generator's output inside the context manager.
- All paths in lock files are relative to the stage's working directory.
- Before you add a lock, guard or extra validation, check whether DAG ordering, lock files or transaction isolation
  already makes the case safe.

## Code style

Follow `docs/contributing/style.md` (imports, Python 3.13 types, TypedDict, simplicity). Beyond it:

- Zero basedpyright warnings: a targeted ignore reads `# type: ignore[code] - reason`. Prefer stub packages over
  ignores; check `typings/` before you add one (`scripts/generate_stubs.py`).
- A module is imported either at run time or under `TYPE_CHECKING`, never both (`pathlib.Path` in annotations, not
  a second `from pathlib import Path`). Lazy imports are allowed in the CLI.
- `_prefix` for private functions.

## Error Handling

- Validate at boundaries (CLI, file I/O, config) and trust internals. Let errors propagate; catch only what you
  can handle, and only the specific exceptions you expect. Avoid a blanket `except Exception` unless it re-raises
  or logs at a boundary.
- A failed operation returns to the last known good state.
- Put an expensive diagnostic check (`exists()`, a stat) at debug level, behind
  `logger.isEnabledFor(logging.DEBUG)`.

## Pull requests and CI

- A draft pull request runs no suite, and a skipped job reads as passed: `draft-marker.yml` sets
  `ci/untested-draft` to failure on a draft's head. Mark it ready (`gh pr ready <n>`) to run Tests. If a ready
  head still reads untested and no Tests run of it is going, push again or run
  `gh pr ready --undo <n> && gh pr ready <n>`.
- A regression found on main gets a line in `.github/known-regressions`, by pull request. Main then stays red and
  nothing publishes until the fix's pull request deletes the line.
- A pull request that changes `packages/*/src`, `benchmarks/`, `.github/workflows/benchmark.yaml` or `uv.lock` gets
  the `benchmark` label before its final review (`gh pr edit <n> --add-label benchmark`), so the gate runs on the
  final head. Without the label Benchmark skips every job. The run costs about 140 job-minutes of a cap all of the
  account's repositories share, so add it when the change is ready for that review, not earlier.
- The gate's rule and the threshold commands are in the `benchmarks/compare.py` docstring. A change under
  `benchmarks/` changes the gate itself, because the job runs `compare.py` and `threshold.json` from the merge
  commit: review it as such.
- To recalibrate, take four or more A/A samples, each from its own `workflow_dispatch` run of Benchmark with `base`
  and `head` set to the same ref (not a re-run: GitHub keeps artifacts only for a run's last attempt), write a new
  `benchmarks/calibration.json` with `compare.py --write-calibration`, then derive `benchmarks/threshold.json`
  from it.
- Never pin `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` or `MKL_NUM_THREADS` for a measurement: the benchmark runs
  pivot with a user's environment.
