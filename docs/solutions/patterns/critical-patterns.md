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

---

## 2. Don't Use Filesystem State to Short-Circuit DAG Construction (ALWAYS REQUIRED)

### ❌ WRONG (Producers silently disappear from DAG after first successful run)
```python
def resolve_external_dependencies(self) -> None:
    while work:
        dep_path = work.pop()
        # Skip if file already exists on disk
        if dep_path in local_outputs or pathlib.Path(dep_path).exists():
            continue  # BUG: skips finding the producer stage
        # ... resolution logic ...
```

### ✅ CORRECT
```python
def resolve_external_dependencies(self) -> None:
    while work:
        dep_path = work.pop()
        # Skip only if a LOCAL stage already produces this
        if dep_path in local_outputs:
            continue
        # Always search for the producer — file existence is irrelevant
        # ... resolution logic (three-tier discovery) ...
```

**Why:** File existence on disk tells you nothing about whether a producer stage exists in another pipeline. After a successful first run, ALL output files exist, so an `exists()` check makes the resolver skip every cross-pipeline dependency. The DAG loses its producer stages, changes to producers stop triggering consumer re-execution, and skip detection breaks silently. Filesystem state is for skip detection at execution time, not for graph construction.

**Placement/Context:** Any code that builds or modifies the dependency graph. Never use `os.path.exists()`, `pathlib.Path.exists()`, or similar checks to decide whether to include a stage in the DAG. The DAG should reflect the logical dependency graph, not the current state of files on disk.

**Documented in:** `docs/solutions/logic-errors/exists-check-prevents-reresolution-20260206.md`

---

## 3. Don't Use `state_dir` as a Proxy for Pipeline Source Directory (ALWAYS REQUIRED)

### ❌ WRONG (All sub-pipelines write "." to output index)
```python
def _write_output_index(self) -> None:
    for stage_name in self.list_stages():
        info = self.get(stage_name)
        # Derive pipeline dir from state_dir
        pipeline_dir = str(info["state_dir"].parent.relative_to(project_root))
        # BUG: state_dir is <root>/.pivot for ALL sub-pipelines using
        # root=project.get_project_root(), so pipeline_dir is always "."
```

### ✅ CORRECT
```python
def _write_output_index(self) -> None:
    for stage_name in self.list_stages():
        info = self.get(stage_name)
        # Derive pipeline dir from the stage function's source file
        pipeline_dir = _find_pipeline_dir_for_stage(info, project_root)
        # Uses inspect.getfile() + walk-up to find nearest pipeline.py

def _find_pipeline_dir_for_stage(info, project_root) -> str | None:
    source_file = pathlib.Path(inspect.getfile(info["func"])).resolve()
    current = source_file.parent
    while current.is_relative_to(project_root):
        if discovery.find_config_in_dir(current) is not None:
            return str(current.relative_to(project_root))
        current = current.parent
```

**Why:** `state_dir` is about state storage location (`.pivot/`), not source code location. These coincide for single-pipeline projects but diverge when sub-pipelines share a project root via `root=project.get_project_root()`. The stage function's source file is the reliable indicator of which pipeline directory a stage belongs to.

**Placement/Context:** Any code that needs to determine which pipeline a stage came from — output index writing, logging, error messages. Use `inspect.getfile(func)` and walk up to find the nearest `pipeline.py`/`pivot.yaml`, never `state_dir`.

**Documented in:** `docs/solutions/logic-errors/output-index-state-dir-shared-root-20260206.md`
