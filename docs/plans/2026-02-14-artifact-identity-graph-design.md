## Artifact-Identity Graph Design

### Summary
Rewrite the bipartite artifact-stage graph to use `ArtifactIdentity` (producer, key)
as artifact node IDs instead of filesystem paths. Remove path-based trie lookups and
directory dependency resolution. Provide helper parsing for artifact identity strings.

### Goals
- Artifact nodes are identity-based: `artifact:{producer}` or `artifact:{producer}:{key}`.
- Graph construction uses `ArtifactRef.identity` from `RegistryStageInfo`.
- Remove path-based outputs map/trie and tracked-file checks.
- Update graph query helpers to accept `ArtifactIdentity`.
- Update graph tests for identity-based artifacts.

### Non-Goals
- Update engine/registry callers outside `graph.py`.
- Resolve identities to filesystem paths (watch mode will be updated later).

### Architecture
- `artifact_node(identity: ArtifactIdentity) -> str` creates artifact nodes.
- `parse_node(node: str)` returns `(NodeType, value)` where artifact values are
  `producer[:key]` (no `artifact:` prefix).
- New helper `parse_artifact_identity(value: str) -> ArtifactIdentity` for parsing
  `producer[:key]` strings.

### Data Flow
- Build an outputs map: `ArtifactIdentity -> stage_name` from `info["outs"]`.
- For each dep `ArtifactRef`:
  - Add edge `artifact(identity) -> stage`.
  - If `validate=True`, ensure the dep identity producer exists in outputs map or
    is an external input (producer not in outputs map).
- For each out `ArtifactRef`:
  - Add edge `stage -> artifact(identity)`.

### Validation
- Remove filesystem existence checks and tracked-file lookups.
- Validation is identity-based only (producer presence in outputs map).
- Keep cycle detection intact.

### Graph Queries
- `get_consumers`, `get_producer`, `get_artifact_consumers` accept
  `ArtifactIdentity`.
- `get_watch_paths` returns `[]` with a TODO (identity->path resolution deferred).
- `extract_graph_view` emits artifact strings as `producer[:key]`.

### Tests
- Update `_create_stage()` to build `ArtifactRef`-based `RegistryStageInfo`.
- Update tests to use `ArtifactIdentity` and new artifact node strings.
- Add tests for `parse_artifact_identity()` and `parse_node()` artifact format.

### Risks
- Callers expecting path-based nodes will break until follow-up tasks update them.
- Watch-mode path resolution is deferred.
