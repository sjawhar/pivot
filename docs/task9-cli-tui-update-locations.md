# Task 9: CLI/TUI Update Locations for ArtifactIdentity/ArtifactRef

**Context:** CLI/TUI pending updates to use ArtifactIdentity/ArtifactRef instead of raw paths.

**Current State:** DepChange uses `identity: str` field, but TUI code references it as `path`. OutputChange uses `path: str` field.

---

## Type Definitions (pivot/types.py)

### Current Structure

```python
class DepChange(TypedDict):
    """Change info for an input dependency file."""
    identity: str  # Currently stores path-like strings
    old_hash: str | None
    new_hash: str | None
    change_type: ChangeType

class OutputChange(TypedDict):
    """Change info for an output file."""
    path: str  # Raw path string
    old_hash: str | None
    new_hash: str | None
    change_type: ChangeType | None  # None means unchanged
    output_type: Literal["out", "metric", "plot"]

class ArtifactIdentity(NamedTuple):
    producer: str
    key: str | None

@dataclasses.dataclass(eq=False)
class ArtifactRef:
    identity: ArtifactIdentity
    format: Reader[Any] | Writer[Any] | Loader[Any, Any]
    python_type: type
    tag: ArtifactTag
```

### Expected Updates

- **DepChange.identity**: Change from `str` to `ArtifactIdentity` (or keep as string representation)
- **OutputChange.path**: Change from `str` to `ArtifactIdentity` (or keep as string representation)
- Decision needed: Store full `ArtifactIdentity` objects or string representations like `"producer:key"`?

---

## CLI Rendering Locations

### 1. `cli/list.py` - Stage Listing

**Function:** `_identity_str(ref: ArtifactRef) -> str` (lines 29-33)
- Converts ArtifactRef to display string
- Format: `"producer:key"` or just `"producer"` if key is None
- Used for both deps and outs display

**Usage:**
- Line 41: `[_identity_str(out) for out in cli_helpers.get_stage(name)["outs"]]`
- Lines 71-72: JSON output for deps/outs
- Lines 90-91: Text output for deps/outs
- Lines 97-106: Verbose output showing dep sources

**Update Needed:**
- ✅ Already uses ArtifactRef via `_identity_str()` helper
- No changes needed if DepChange/OutputChange store string representations

---

### 2. `cli/console.py` - Explain Output

**Function:** `explain_stage(explanation: StageExplanation)` (lines 272-299)
- Renders detailed stage explanations
- Line 298: `"Dependency Changes:", explanation["dep_changes"], "path", "old_hash", "new_hash"`

**Issue:** References `"path"` field but DepChange uses `"identity"` field

**Update Needed:**
- Change `"path"` to `"identity"` in `_print_changes()` call
- Or update DepChange to use `"path"` field name consistently

**Function:** `_print_changes()` (lines 242-271)
- Generic change printer using field names
- Line 257: `key = change[key_field]` - expects string key
- Lines 262-270: Formats old/new values

**Update Needed:**
- If DepChange.identity becomes ArtifactIdentity object, need to convert to string for display
- May need helper like `_identity_str()` from list.py

---

### 3. `cli/status.py` - Status/Explain Commands

**Function:** `_output_explain_json()` (lines 228-267)
- Line 252: `dep_changes=exp["dep_changes"]` - passes through unchanged
- JSON serialization expects string fields

**Function:** `output_explain_text()` (lines 304-323)
- Line 319: `for exp in explanations:`
- Line 320: `con.explain_stage(exp)` - delegates to console.py

**Update Needed:**
- If DepChange/OutputChange use ArtifactIdentity objects, JSON serialization will fail
- Need to convert to string representations before JSON output
- Text output handled by console.py (see above)

---

### 4. `cli/targets.py` - Target Resolution

**Lines 168, 199-200:**
```python
rel_path = project.to_relative_path(project.normalize_path(out.path), proj_root)
```

**Update Needed:**
- Currently accesses `out.path` directly from ArtifactRef
- Need to convert ArtifactIdentity to path for filesystem operations
- May need new helper: `identity_to_path(identity: ArtifactIdentity) -> Path`

---

### 5. `cli/verify.py` - Cache Verification

**Lines 84-91:**
```python
cached_paths = {
    path_utils.canonicalize_artifact_path(str(out.path), project_root)
    for out in stage_info["outs"]
    if out.tag != ArtifactTag.METRIC or out.format.cache
}

return {
    path: h
    for path, h in lock_data["output_hashes"].items()
    if path_utils.canonicalize_artifact_path(path, project_root) in cached_paths
}
```

**Update Needed:**
- Accesses `out.path` from ArtifactRef
- Compares with lock file paths (currently strings)
- Need consistent path/identity representation

---

### 6. `cli/checkout.py` - Output Restoration

**Lines 46-55:**
```python
for out_path, out_hash in lock_data["output_hashes"].items():
    path_utils.canonicalize_artifact_path(str(out.path), project_root)
```

**Update Needed:**
- Lock file stores paths as strings
- Need to map between ArtifactIdentity and filesystem paths

---

## TUI Rendering Locations

### 1. `pivot_tui/client.py` - RPC Protocol

**Type:** `StageInfoResult` (lines 25-28)
```python
class StageInfoResult(TypedDict):
    name: str
    deps: list[str]  # Currently string list
    outs: list[str]  # Currently string list
```

**Update Needed:**
- If engine returns ArtifactIdentity objects, need to serialize to strings for RPC
- Or update protocol to accept structured identity objects

---

### 2. `pivot_tui/rpc_client_impl.py` - RPC Client

**Function:** `stage_info()` (lines 126-135)
```python
return StageInfoResult(
    name=name,
    deps=_as_str_list(r["deps"]),
    outs=_as_str_list(r["outs"]),
)
```

**Update Needed:**
- Currently expects string lists from RPC
- If protocol changes to structured objects, need to handle conversion

---

### 3. `pivot_tui/diff_panels.py` - Input/Output Panels

**Class:** `InputDiffPanel` (lines 294-561)

**Storage:**
- Line 305: `_dep_by_path: dict[str, DepChange]` - keyed by path string
- Line 343: `for path in self._dep_by_path:` - iterates path strings
- Line 558: `self._dep_by_path = {d["path"]: d for d in snapshot["dep_changes"]}`

**Issue:** References `d["path"]` but DepChange uses `"identity"` field

**Rendering:**
- Line 374: `change = self._find_dep_change(item_key)` - looks up by path string
- Line 384: Displays `item_key` (path string) in UI
- Line 449: `_render_dep_detail(path: str)` - takes path string parameter
- Line 457: `f"[bold]{rich.markup.escape(path)}[/]"` - displays path

**Update Needed:**
- Change all `"path"` references to `"identity"` to match DepChange field name
- If identity becomes ArtifactIdentity object, need to:
  - Convert to string for dict keys: `{_identity_str(d["identity"]): d for d in ...}`
  - Convert to string for display: `rich.markup.escape(_identity_str(identity))`

---

**Class:** `OutputDiffPanel` (lines 563-813)

**Storage:**
- Line 573: `_output_by_path: dict[str, OutputChange]` - keyed by path string
- Line 603: `for path, change in self._output_by_path.items()` - iterates paths
- Line 811: `self._output_by_path = {c["path"]: c for c in changes}` - uses path field

**Rendering:**
- Line 629: `f"{prefix}{indicator} {_escape_padded(change['path'], 25)} {hash_display}{suffix}"`
- Line 645: `f"[bold]{rich.markup.escape(change['path'])}[/]"`

**Update Needed:**
- If OutputChange.path becomes ArtifactIdentity, need string conversion
- Dict keys and display both need string representations

---

## Core Data Flow

### Skip Detection (`skip.py`)

**Function:** `diff_dep_hashes()` (lines 159-193)
```python
def diff_dep_hashes(old: dict[str, HashInfo], new: dict[str, HashInfo]) -> list[DepChange]:
    changes = list[DepChange]()
    all_keys = sorted(set(old.keys()) | set(new.keys()))
    for key in all_keys:
        # ...
        changes.append(
            DepChange(
                identity=key,  # key is a string from dict
                old_hash=old[key]["hash"],
                new_hash=new[key]["hash"],
                change_type=ChangeType.MODIFIED,
            )
        )
```

**Current:** Creates DepChange with string identity from lock file dict keys

**Update Needed:**
- If lock files store ArtifactIdentity objects, key will be structured
- If lock files store string representations, no change needed
- Need consistency between lock file format and DepChange.identity type

---

## Summary of Required Updates

### High Priority (Breaks Functionality)

1. **TUI diff_panels.py line 558**: Change `d["path"]` to `d["identity"]` to match DepChange field
2. **CLI console.py line 298**: Change `"path"` to `"identity"` in _print_changes call
3. **JSON serialization**: If identity becomes ArtifactIdentity object, add conversion to string before JSON output

### Medium Priority (Type Consistency)

4. **TUI diff_panels.py**: Rename `_dep_by_path` to `_dep_by_identity` for clarity
5. **CLI targets.py, verify.py, checkout.py**: Add helper to convert ArtifactIdentity to filesystem path
6. **RPC protocol**: Decide if deps/outs should be string list or structured objects

### Low Priority (Naming Consistency)

7. **DepChange field name**: Consider renaming `identity` to `path` for consistency with OutputChange
8. **OutputChange field name**: Consider renaming `path` to `identity` for consistency with DepChange

---

## Design Decisions Needed

### 1. Field Naming Convention

**Option A:** Both use `identity` field
- DepChange.identity: ArtifactIdentity | str
- OutputChange.identity: ArtifactIdentity | str

**Option B:** Both use `path` field (current OutputChange)
- DepChange.path: ArtifactIdentity | str
- OutputChange.path: ArtifactIdentity | str

**Option C:** Keep different names, document distinction
- DepChange.identity: for inputs (may reference other stages)
- OutputChange.path: for outputs (always filesystem paths)

### 2. Storage Format

**Option A:** Store full ArtifactIdentity objects
- Pros: Type-safe, structured
- Cons: Requires conversion for display, JSON serialization, dict keys

**Option B:** Store string representations (`"producer:key"`)
- Pros: Simple, works with existing code
- Cons: Loses structure, requires parsing if needed

**Option C:** Store both (identity object + cached string)
- Pros: Best of both worlds
- Cons: Redundant data, sync issues

### 3. Lock File Format

**Current:** Lock files use string paths as dict keys
```python
lock_data["dep_hashes"]: dict[str, HashInfo]
lock_data["output_hashes"]: dict[str, HashInfo]
```

**Question:** Should lock files store ArtifactIdentity objects or string representations?
- Affects skip.py diff_dep_hashes() function
- Affects all lock file read/write code

---

## Recommended Approach

1. **Keep string representations in DepChange/OutputChange** (Option B for storage)
   - Use `"producer:key"` format consistently
   - Add helper function to convert ArtifactRef → string
   - Minimal changes to existing code

2. **Standardize field name to `identity`** (Option A for naming)
   - More accurate for deps (may reference stages, not just files)
   - Update OutputChange.path → OutputChange.identity
   - Update all references in TUI/CLI

3. **Lock files continue using string keys**
   - No changes to lock file format
   - String keys work naturally with dicts
   - Easy to read/debug

4. **Add conversion helpers**
   ```python
   def artifact_identity_str(ref: ArtifactRef) -> str:
       """Convert ArtifactRef to identity string."""
       if ref.identity.key is None:
           return ref.identity.producer
       return f"{ref.identity.producer}:{ref.identity.key}"
   
   def identity_to_path(identity: ArtifactIdentity, stage_cwd: Path) -> Path:
       """Convert ArtifactIdentity to filesystem path for stage outputs."""
       # For stage outputs, producer is the path
       return stage_cwd / identity.producer
   ```

---

## Files Requiring Updates

### Type Definitions
- `pivot/types.py`: Rename OutputChange.path → identity (if standardizing)

### CLI
- `cli/console.py`: Change "path" → "identity" in line 298
- `cli/targets.py`: Add identity_to_path() helper usage
- `cli/verify.py`: Add identity_to_path() helper usage  
- `cli/checkout.py`: Add identity_to_path() helper usage
- `cli/list.py`: ✅ Already correct (uses _identity_str helper)

### TUI
- `pivot_tui/diff_panels.py`: 
  - Line 558: Change `d["path"]` → `d["identity"]`
  - Rename `_dep_by_path` → `_dep_by_identity` (optional)
  - Update all path references to identity

### Core
- `skip.py`: ✅ Already uses "identity" field correctly

---

## Testing Checklist

After updates, verify:

- [ ] `pivot list` shows deps/outs correctly
- [ ] `pivot list --deps` shows dependency sources
- [ ] `pivot status --explain` displays dep changes
- [ ] `pivot status --explain --json` serializes correctly
- [ ] TUI Input tab displays dep changes
- [ ] TUI Output tab displays output changes
- [ ] TUI navigation (j/k/n/N) works with identity strings
- [ ] Lock file read/write preserves identity format
- [ ] Cache operations resolve identities to paths correctly
