# Pivot TUI - Development Guidelines

Pure RPC client over the engine's JSON-RPC 2.0 Unix socket.

## Boundary Rules

- **No direct state access.** Must not read lock files, cache files, or state files. All state goes through engine RPC.
- **Import boundary:** May import `pivot.types` (pure type definitions). Must not import runtime modules (`engine`, `storage`, `config`).

## Lifecycle

One-shot and watch mode are source-configuration differences, not different lifecycles. The TUI should outlive run completion.

## Event Stream

Engine emits explanation events in the event stream. The TUI reconstructs history client-side from this stream.

## Testing

Use `FakeRpcServer` (`testing/fake_server.py`) for TUI tests — test against client interface contracts, not engine internals.
