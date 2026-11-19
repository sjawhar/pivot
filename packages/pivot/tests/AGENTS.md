# Pivot tests

pytest with pytest-xdist, pytest-mock and pytest-asyncio (`asyncio_mode = "auto"`). Conventions
with examples: `docs/contributing/testing.md`.

Run one file with `uv run pytest packages/pivot/tests/<dir>/test_<module>.py`, one test by adding
`::test_name`; the suite command is in the root `AGENTS.md`.

## Writing tests

- Flat `def test_<behavior>()` functions in `test_<module>.py`: no `class Test*`, no
  `@pytest.mark.skip`, imports at module top.
- A helper that a stage or fingerprint test uses is a module-level `_helper_*` function:
  `getclosurevars()` does not see a nested function's module imports.
- Parametrize cases that share logic, with `pytest.param(..., id="...")`; no branching on the
  parameter in the body.
- Mock only boundaries (network, time, randomness), with `mocker` or `monkeypatch` and
  `autospec=True` (not when patching to a literal value). A test that asserts what its mock
  returns proves nothing.
- Assert on public behaviour, not private attributes. Check CLI output by containment or
  `--json`, not by line position. Put the reason in the assertion message, not in a comment.
- Every CLI command has an integration test: real files (stages, data, `.git`), the command run
  through `CliRunner().invoke()`, output and filesystem both checked, for success, error paths
  and each output format.
- A feature that wires components together (a CLI mode, a protocol) gets an end-to-end test
  through its public interface: a subprocess, a socket, the filesystem.
- State shared across processes goes through files; each worker has its own copy of a
  module-level list.
- Fingerprinting tests live in `tests/fingerprint/`; read and update the change-detection
  matrix in `tests/fingerprint/README.md` when you change them.

## Fixtures and helpers (`conftest.py`, `helpers.py`)

- Autouse `reset_pivot_state` points `project._project_root_cache` at a shared default project
  and clears the config cache; autouse `restore_import_machinery` undoes import-hook changes.
  Do not reset these yourself. `clean_registry` does nothing; do not add it.
- For a project of your own use `set_project_root` (on `tmp_path`), `pipeline_dir` (also changes
  into it), `mock_discovery` (discovery returns `test_pipeline`; add stages with
  `helpers.register_test_stage`), or `isolated_pivot_dir(runner, tmp_path)` in place of
  `runner.isolated_filesystem()`.
- `helpers.create_pipeline_py` writes a `pipeline.py` from stage functions for real discovery.
- `tmp_path` gives owner access back to every directory at teardown, so a read-only tree left
  there needs no cleanup.
- `git_repo` returns `(path, commit)`. `worker_env` and `output_queue` set up a worker.
  `moto_s3_bucket` (moto) and `gcs_bucket` (fake-gcs-server, needs Docker) give remote buckets.
- `helpers.run_pivot(cwd, home, *args)` runs the installed `pivot` beside `sys.executable`
  (CI has no `uv` on `PATH`), without `PIVOT*` variables and with `HOME` set to `home`.
