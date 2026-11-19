# Pivot tests

pytest with pytest-xdist, pytest-mock and pytest-asyncio (`asyncio_mode = "auto"`). Read
`docs/contributing/testing.md` before you write a test: it holds the test style, mocking rules and the
project fixtures (`set_project_root`, `pipeline_dir`, `mock_discovery`, `isolated_pivot_dir`, the autouse ones).

Run one file with `uv run pytest packages/pivot/tests/<dir>/test_<module>.py`, one test by adding
`::test_name`; the suite command is in the root `AGENTS.md`.

## Beyond the testing guide

- Every CLI command has an integration test: real files (stages, data, `.git`), the command run
  through `CliRunner().invoke()`, output and filesystem both checked, for success, error paths
  and each output format.
- A feature that wires components together (a CLI mode, a protocol) gets an end-to-end test
  through its public interface: a subprocess, a socket, the filesystem.
- State shared across processes goes through files; each worker process has its own copy of a
  module-level list.
- `helpers.register_test_stage` adds a stage under `mock_discovery`; `helpers.create_pipeline_py`
  writes a `pipeline.py` from stage functions for real discovery.
- `helpers.run_pivot(cwd, home, *args)` runs the installed `pivot` beside `sys.executable`
  (CI has no `uv` on `PATH`), without `PIVOT*` variables and with `HOME` set to `home`.
- `worker_env` and `output_queue` set up a worker. `moto_s3_bucket` (moto) and `gcs_bucket`
  (fake-gcs-server, needs Docker) give remote buckets.
- `tmp_path` gives owner access back to every directory at teardown, so a read-only tree left
  there needs no cleanup.
