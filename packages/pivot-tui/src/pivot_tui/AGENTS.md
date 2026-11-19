# Pivot TUI

Textual app that is a client of the engine's JSON-RPC 2.0 Unix socket and nothing else
(`docs/architecture/tui.md`).

- All state comes through engine RPC; do not read lock, cache or state files.
- Import from `pivot` only modules without runtime state (`pivot.types`, `pivot.display`), never
  `engine`, `storage`, `executor` or `config`.
- Do not raise the `pivot>=` bound in `packages/pivot-tui/pyproject.toml` by hand; the release
  writes it (`scripts/set_tui_pivot_bound.py`).
- Test against `FakeRpcServer` (`testing/fake_server.py`) and the protocols in `client.py`, not
  engine internals: `uv run pytest packages/pivot-tui/tests`.
