# Critical Patterns - Required Reading

These patterns represent lessons learned from bugs that were non-obvious or repeated across modules. All code generation should follow these patterns.

---

## 1. E2E Tests Required for Major Features (ALWAYS REQUIRED)

### ❌ WRONG (Components work but feature is broken)
```python
# Only unit tests for individual components
def test_rpc_handler_returns_status():
    handler = AgentRpcHandler(engine=engine)
    result = handler.handle_query("status")
    assert result["state"] == "idle"  # ✓ Passes

def test_rpc_source_accepts_connections():
    source = AgentRpcSource(socket_path=path)
    # ... test socket creation  # ✓ Passes

# CLI integration code (UNTESTED):
def _run_serve_mode():
    eng.add_source(AgentRpcSource(socket_path=socket_path))  # Missing handler!
    # Missing AgentEventSink!
```

### ✅ CORRECT
```python
# Unit tests for components (keep these)
def test_rpc_handler_returns_status(): ...
def test_rpc_source_accepts_connections(): ...

# ALSO: E2E test for the complete CLI path
def test_serve_mode_cli_responds_to_status_query(tmp_path):
    """E2E test: exercises actual CLI, not just components."""
    # 1. Start actual CLI command
    proc = subprocess.Popen(["uv", "run", "pivot", "run", "--watch", "--serve"], ...)

    # 2. Exercise through public interface
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(socket_path))
        sock.sendall('{"jsonrpc":"2.0","method":"status","id":1}\n'.encode())
        response = json.loads(sock.recv(1024).decode())

    # 3. Verify end-to-end behavior
    assert "error" not in response  # Would catch missing handler
    assert response["result"]["state"] in ("idle", "active")
```

**Why:** Unit tests verify components work in isolation. E2E tests verify they're wired together correctly. Components can pass all unit tests but fail when assembled - missing parameters, missing sinks, wrong initialization order.

**Placement/Context:** Required for any major feature: new CLI modes, protocols, architectural components. Add E2E test that starts the actual CLI (subprocess if async) and exercises through public interface.

**Documented in:** `docs/solutions/integration-issues/missing-e2e-test-cli-serve-mode-20260201.md`
