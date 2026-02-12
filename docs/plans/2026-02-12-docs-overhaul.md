# Documentation Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite Pivot's documentation for alpha release. Reorient around the Python API (YAML is second-class), create a linear learning path from first concepts to advanced workflows, and fill all content gaps.

**Architecture:** The docs move from a flat reference structure to a layered learning path: Getting Started → Concepts (8 pages, building on each other) → Guides (7 task-oriented pages) → Reference (auto-generated API + CLI) → Architecture (contributors). Every concept page leads with Python examples. YAML is mentioned only as an alternative, never first.

**Tech Stack:** MkDocs Material, mkdocstrings (auto-generated API), Markdown

---

## New Information Architecture

```
Getting Started/
  installation.md          # Install + verify
  quickstart.md            # First pipeline in 5 minutes

Concepts/                  # Linear learning path, each builds on previous
  artifacts-and-dag.md     # Artifact-first mental model, how deps create the graph
  dependencies.md          # Dep, PlaceholderDep, directory deps, pivot track
  outputs.md               # Out, Metric, Plot, IncrementalOut, DirectoryOut, cache=False
  loaders.md               # Reader/Writer/Loader, all built-ins, custom loaders
  parameters.md            # StageParams, Pydantic, overrides
  pipelines.md             # Pipeline class, register(), include(), transitive discovery
  fingerprinting.md        # Code change detection, @no_fingerprint
  caching.md               # Content-addressable cache, skip algorithm, checkout modes

Guides/                    # Task-oriented, separate pages
  cross-repo-import.md     # pivot import, .pvt files, pivot update
  watch-mode.md            # Watch mode + rapid iteration
  multi-pipeline.md        # include(), composition, --all
  remote-storage.md        # push/pull/config, S3 setup
  ci-integration.md        # verify --all --allow-missing, push in CI
  tui.md                   # Interactive TUI, keybindings, panels
  migrating-from-dvc.md    # Step-by-step DVC migration

CLI Reference/             # Complete command reference
  index.md                 # All commands (existing, updated)

API Reference/             # Auto-generated from docstrings
  (mkdocstrings)

Architecture/              # For contributors
  overview.md
  code-tour.md
  engine.md
  execution.md
  fingerprinting.md
  watch.md
  tui.md
  agent-server.md

Contributing/              # Dev setup, style, testing
  (existing, updated)
```

### Key Principles
- **Python-first**: Every example leads with Python. YAML mentioned only as "Alternative: YAML configuration" after the Python example.
- **Artifact-first**: The DAG page teaches the mental model before anything else.
- **All types listed**: Every Dep/Out/Loader variant documented with examples.
- **No duplication**: Concepts pages are the source of truth. Guides link back to concepts, don't re-explain.

---

## Task 1: Fix Broken Build (Prerequisite)

The mkdocs build is broken — nav references deleted files. Fix before anything else.

**Files:**
- Modify: `mkdocs.yml:96-98`
- Modify: `docs/design/index.md`
- Delete: `docs/test-coverage-analysis-2026-02-03.md`
- Delete: `signal-handler-thread-textual-tui-20260203.md`

**Step 1: Remove broken nav entries from mkdocs.yml**

Remove these lines from the Design Decisions nav:
```yaml
    - Watch Engine Design: design/watch-engine.md
    - Hot Reload Exploration: design/hot-reload-exploration.md
```

**Step 2: Update `docs/design/index.md`**

Replace the documents table with:
```markdown
# Design Decisions

No active design documents. Past designs have been implemented and archived. See [GitHub Issues](https://github.com/sjawhar/pivot/issues) for planned work.
```

**Step 3: Delete orphans**

```bash
rm docs/test-coverage-analysis-2026-02-03.md
rm signal-handler-thread-textual-tui-20260203.md
```

**Step 4: Verify build**

```bash
uv run mkdocs build --strict 2>&1 | tail -5
```

**Step 5: Commit**

```bash
jj describe -m "docs: fix broken nav and delete orphan files"
```

---

## Task 2: Restructure mkdocs.yml Nav

Set up the new nav structure. Create directories and placeholder files so the nav doesn't break.

**Files:**
- Modify: `mkdocs.yml` (rewrite nav section)
- Create: `docs/concepts/` directory with placeholder files
- Create: `docs/guides/` directory with placeholder files

**Step 1: Create directories and minimal placeholders**

```bash
mkdir -p docs/concepts docs/guides
```

Create `# Title` + TODO placeholder for each new page.

**Step 2: Rewrite mkdocs.yml nav**

```yaml
nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quickstart.md
  - Concepts:
    - Artifacts & the DAG: concepts/artifacts-and-dag.md
    - Dependencies: concepts/dependencies.md
    - Outputs: concepts/outputs.md
    - Loaders: concepts/loaders.md
    - Parameters: concepts/parameters.md
    - Pipelines: concepts/pipelines.md
    - Fingerprinting: concepts/fingerprinting.md
    - Caching & Skip Detection: concepts/caching.md
  - Guides:
    - Cross-Repo Import: guides/cross-repo-import.md
    - Watch Mode: guides/watch-mode.md
    - Multi-Pipeline Projects: guides/multi-pipeline.md
    - Remote Storage: guides/remote-storage.md
    - CI Integration: guides/ci-integration.md
    - TUI: guides/tui.md
    - Migrating from DVC: migrating-from-dvc.md
  - CLI Reference: cli/index.md
  - API Reference: reference/
  - Comparison: comparison.md
  - Architecture:
    - Overview: architecture/overview.md
    - Code Tour: architecture/code-tour.md
    - Fingerprinting: architecture/fingerprinting.md
    - Execution Model: architecture/execution.md
    - Engine: architecture/engine.md
    - Watch Engine: architecture/watch.md
    - TUI: architecture/tui.md
    - Agent Server: architecture/agent-server.md
  - Design Decisions: design/index.md
  - Solutions: solutions/index.md
  - Contributing:
    - Getting Started: contributing/setup.md
    - Code Style: contributing/style.md
    - Testing Guide: contributing/testing.md
    - CLI Development: contributing/cli.md
    - Adding Loaders: contributing/loaders.md
    - Common Gotchas: contributing/gotchas.md
  - Prevention Strategies:
    - TypedDict Field Synchronization: prevention/typeddictfield-synchronization.md
```

**Step 3: Verify build (without --strict), commit**

```bash
jj describe -m "docs: restructure nav for concepts/guides/reference layout"
```

---

## Task 3: Write Concepts — Artifacts & the DAG

The foundational page. Teaches the mental model before any API.

**Files:**
- Write: `docs/concepts/artifacts-and-dag.md`

**Content outline:**

1. **The artifact-first mental model** — "This file changed. What needs to happen because of that?"
2. **How the DAG forms** — stages produce artifacts (Out), other stages consume them (Dep). When an output path matches a dependency path, Pivot creates an edge. No explicit wiring needed.
3. **Simple example** — Two functions, one produces `processed.csv`, the other consumes it. Show the DAG that emerges. Python only.
4. **Visualizing the DAG** — `pivot dag`, `pivot dag --stages`, `pivot dag --mermaid`
5. **Key insight** — "You never declare 'stage A depends on stage B'. You declare what files each stage reads and writes, and Pivot figures out the rest."

All examples Python-first. No YAML.

**Step 1: Write the page** (~150-200 lines)

**Step 2: Verify build, commit**

```bash
jj describe -m "docs(concepts): add artifacts & DAG page"
```

---

## Task 4: Write Concepts — Dependencies

Everything about declaring inputs.

**Files:**
- Write: `docs/concepts/dependencies.md`

**Content outline:**

1. **Basic dependencies** — `Annotated[T, Dep(path, reader)]` pattern, with inline explanation of each part
2. **What happens at runtime** — Pivot calls `reader.load(path)` and passes the result to your function parameter
3. **Dependency types table** — All types at a glance:

| Type | Import | Use case |
|------|--------|----------|
| `Dep[R]` | `from pivot.outputs import Dep` | Standard file dependency |
| `PlaceholderDep[R]` | `from pivot.outputs import PlaceholderDep` | Dependency whose path is set at registration time |
| Directory dep | (string ending in `/`) | Depend on all files in a directory |

4. **Dep in detail** — Full example with CSV, JSON, PathOnly
5. **PlaceholderDep** — When you don't know the path at function definition time, only at `pipeline.register()`. Example with path overrides.
6. **Directory dependencies** — Depending on an entire directory. Changes to any file trigger re-run.
7. **Tracking files outside pipelines** — `pivot track` creates `.pvt` manifest files. `pivot status --tracked-only` to check. `pivot checkout` to restore from cache.
8. **Upstream stage outputs as dependencies** — When one stage's `Out("data.csv")` matches another's `Dep("data.csv")`, the DAG edge is automatic. Reference back to Artifacts & DAG page.

All examples Python-first. YAML path overrides mentioned briefly as an alternative in PlaceholderDep section only.

**Step 1: Write the page** (~200-250 lines)

**Step 2: Verify build, commit**

```bash
jj describe -m "docs(concepts): add dependencies page"
```

---

## Task 5: Write Concepts — Outputs

Everything about declaring what a stage produces.

**Files:**
- Write: `docs/concepts/outputs.md`

**Content outline:**

1. **Basic pattern** — TypedDict return type with `Annotated[T, Out(path, writer)]` fields
2. **Single output shorthand** — Annotate return type directly (no TypedDict needed)
3. **Output types table** — All at a glance:

| Type | Import | Cached | Git-tracked | Use case |
|------|--------|--------|-------------|----------|
| `Out[W]` | `outputs.Out` | Yes | No | Large data files, models |
| `Metric` | `outputs.Metric` | No | Yes | Small JSON metrics |
| `Plot[W]` | `outputs.Plot` | Yes | No | Visualizations |
| `IncrementalOut[W, R]` | `outputs.IncrementalOut` | Yes | No | Append-only / stateful |
| `DirectoryOut[T]` | `outputs.DirectoryOut` | Yes | No | Stage produces a directory of files |

4. **Out** — Standard cached output. `cache=False` option for outputs you don't want cached.
5. **Metric** — Git-tracked JSON. `pivot metrics show`, `pivot metrics diff`.
6. **Plot** — With `PathOnly()` (you save the file) or `MatplotlibFigure()` (Pivot saves it). `pivot plots show --open`.
7. **IncrementalOut** — Requires bidirectional `Loader[W, R]` (not just Writer). Explain the restore-before-run cycle. Copy mode.
8. **DirectoryOut** — For stages that produce a directory of files. How it differs from single-file Out.
9. **Decision tree** — "Which output type should I use?" flowchart.

All Python-first. No YAML.

**Step 1: Write the page** (~250-300 lines)

**Step 2: Verify build, commit**

```bash
jj describe -m "docs(concepts): add outputs page"
```

---

## Task 6: Write Concepts — Loaders

Everything about reading and writing data.

**Files:**
- Write: `docs/concepts/loaders.md`

**Content outline:**

1. **The type hierarchy** — `Reader[R]` (read), `Writer[W]` (write), `Loader[W, R]` (both). Why this matters: `Dep` needs a Reader, `Out` needs a Writer, `IncrementalOut` needs a Loader.
2. **Built-in loaders table** — ALL of them:

| Loader | Read type | Write type | Options | Notes |
|--------|-----------|------------|---------|-------|
| `CSV()` | DataFrame | DataFrame | `sep`, `index_col`, `dtype` | |
| `JSON()` | dict/list | dict/list | `indent` | |
| `YAML()` | dict | dict | | |
| `Text()` | str | str | `encoding` | |
| `JSONL()` | list[dict] | list[dict] | | |
| `DataFrameJSONL()` | DataFrame | DataFrame | | Orient=records |
| `Pickle()` | Any | Any | `protocol` | Security warning |
| `PathOnly()` | Path | (validates exists) | | You handle I/O |
| `MatplotlibFigure()` | — | Figure | `dpi`, `format` | Writer only, for Plot |

3. **Using built-in loaders** — Examples with Dep and Out
4. **Creating custom loaders** — `@dataclasses.dataclass(frozen=True)`, extend `Loader[T]`, implement `load()` and `save()`. Requirements: immutable, module-level, fingerprinted.
5. **Read-only and write-only loaders** — When to use `Reader[R]` or `Writer[W]` instead of full `Loader`.

**Step 1: Write the page** (~200 lines)

**Step 2: Verify build, commit**

```bash
jj describe -m "docs(concepts): add loaders page"
```

---

## Task 7: Write Concepts — Parameters

**Files:**
- Write: `docs/concepts/parameters.md`

**Content outline:**

1. **StageParams** — Pydantic BaseModel subclass. Define typed parameters with defaults.
2. **Using in stages** — `params: TrainParams` as first argument
3. **Overriding via params.yaml** — Stage-scoped overrides
4. **Parameter change detection** — Changes trigger re-runs
5. **CLI** — `pivot params show`, `pivot params diff`
6. **Matrix stages** — Brief mention, link to reference

**Step 1: Write the page** (~150 lines)

**Step 2: Verify build, commit**

```bash
jj describe -m "docs(concepts): add parameters page"
```

---

## Task 8: Write Concepts — Pipelines

**Files:**
- Write: `docs/concepts/pipelines.md`

**Content outline:**

1. **The Pipeline class** — `Pipeline(name)`, `pipeline.register(func)`. Python-first, always.
2. **Stage registration** — What happens when you call `register()`. Annotations are extracted, fingerprint computed, stage added to registry.
3. **Path overrides** — `out_path_overrides` and `dep_path_overrides` kwargs to `register()`.
4. **Pipeline composition** — `pipeline.include(other_pipeline)`. Deep copy semantics.
5. **Transitive pipeline discovery** — How Pivot finds pipelines in parent/sibling directories. The traverse-up algorithm. When you depend on `../sibling/data/output.csv`, Pivot discovers the sibling pipeline automatically.
6. **Discovery order** — `pivot.yaml` → `pivot.yml` → `pipeline.py`. First match wins.
7. **YAML configuration (alternative)** — Brief section: for projects needing path overrides without modifying Python. Show the YAML format. Emphasize this is the alternative, not the primary method.

**Step 1: Write the page** (~250 lines)

**Step 2: Verify build, commit**

```bash
jj describe -m "docs(concepts): add pipelines page"
```

---

## Task 9: Write Concepts — Fingerprinting

**Files:**
- Write: `docs/concepts/fingerprinting.md`

**Content outline:**

1. **What Pivot tracks** — AST of your stage function + all functions it calls (transitive)
2. **How it works** — getclosurevars + AST extraction for module.attr patterns
3. **What triggers re-runs** — Change a helper function, change a default value, change a global constant
4. **What doesn't trigger re-runs** — Comments, formatting, unreachable code
5. **@no_fingerprint()** — Opt out of AST fingerprinting, fall back to file-level hashing. When to use, trade-offs.
6. **Common surprises** — Lambda re-runs, dynamic attribute access

**Step 1: Write the page** (~150-200 lines)

**Step 2: Verify build, commit**

```bash
jj describe -m "docs(concepts): add fingerprinting page"
```

---

## Task 10: Write Concepts — Caching & Skip Detection

**Files:**
- Write: `docs/concepts/caching.md`

**Content outline:**

1. **Content-addressable cache** — .pivot/cache/files/ structure, xxhash64
2. **Per-stage lock files** — What's in a lock file (fingerprint, params, dep hashes, output hashes)
3. **Skip detection algorithm** — When does Pivot skip? Generation tracking → lock file comparison → run cache lookup
4. **Checkout modes** — hardlink (default), symlink, copy. When each is used.
5. **pivot checkout** — Restoring outputs from cache
6. **pivot commit** — Writing lock files without running stages
7. **pivot status** — Checking what would run. `--explain` for detailed breakdown.
8. **File change detection** — mtime + size + inode heuristic before hashing

**Step 1: Write the page** (~200 lines)

**Step 2: Verify build, commit**

```bash
jj describe -m "docs(concepts): add caching & skip detection page"
```

---

## Task 11: Write Guides — Cross-Repo Import

**Files:**
- Write: `docs/guides/cross-repo-import.md`

**Content outline:**

1. **Use case** — Sharing artifacts across repositories (models, datasets, reference data)
2. **Importing an artifact** — `pivot import <repo_url> <path>` walkthrough
3. **.pvt manifest files** — What they contain, how they work
4. **Checking for updates** — `pivot update`
5. **Downloading imported artifacts** — `pivot pull`
6. **Options** — `--rev`, `--out`, `--no-download`, `--force`

**Step 1: Write the page** (~150 lines)

**Step 2: Verify build, commit**

```bash
jj describe -m "docs(guides): add cross-repo import guide"
```

---

## Task 12: Write Guides — Watch Mode, Multi-Pipeline, Remote Storage

Move and rewrite existing tutorial pages, ensuring Python-first examples.

**Files:**
- Write: `docs/guides/watch-mode.md` (from `docs/tutorial/watch.md`)
- Write: `docs/guides/multi-pipeline.md` (from `docs/tutorial/multi-pipeline.md`)
- Write: `docs/guides/remote-storage.md` (from `docs/tutorial/remote.md`)

For each:
1. Read the existing tutorial page
2. Rewrite with Python-first examples (replace any YAML-first patterns)
3. Link to concept pages instead of re-explaining
4. Remove old tutorial/ files after new ones are confirmed working

**Step 1-3: Rewrite each guide**

**Step 4: Delete old tutorial/ files**

**Step 5: Verify build, commit**

```bash
jj describe -m "docs(guides): rewrite watch, multi-pipeline, and remote storage guides"
```

---

## Task 13: Write Guides — CI Integration and TUI

**Files:**
- Write: `docs/guides/ci-integration.md`
- Write: `docs/guides/tui.md`

**CI Integration content:**
1. **The CI workflow** — Run pipeline → push → (PR) → verify in CI
2. **pivot verify** — `pivot verify --all --allow-missing` for pre-merge gates
3. **Pushing in CI** — `pivot push` after successful runs
4. **Example GitHub Actions workflow** — Complete YAML
5. **pivot status --json** — For scripting

**TUI content:**
1. **Launching** — `pivot repro --tui`, `pivot repro --watch --tui`
2. **Layout** — Stage list, logs panel, input diff, output diff
3. **Keyboard shortcuts table** (from run.py source)
4. **Log search** — Ctrl+F, regex
5. **Force re-run** — `r` for selected, `R` for all
6. **Commit** — `c` to commit from TUI

**Step 1: Write CI integration guide** (~150 lines)

**Step 2: Write TUI guide** (~150 lines)

**Step 3: Verify build, commit**

```bash
jj describe -m "docs(guides): add CI integration and TUI guides"
```

---

## Task 14: Update CLI Reference

Add missing commands, update existing entries.

**Files:**
- Modify: `docs/cli/index.md`

**Changes:**
1. Add `pivot import` command documentation
2. Add `pivot update` command documentation
3. Add `pivot fingerprint reset` command
4. Add `--show-output` flag to `pivot repro` and `pivot run`
5. Verify all existing commands are accurate against current code

**Step 1: Read import_cmd.py, update.py, fingerprint.py for accurate flags**

**Step 2: Add new sections to cli/index.md**

**Step 3: Add --show-output to repro/run options tables**

**Step 4: Verify build, commit**

```bash
jj describe -m "docs(cli): add import, update, fingerprint reset, --show-output"
```

---

## Task 15: Rewrite README

Slim from 577 lines to ~200. Fix stale references.

**Files:**
- Modify: `README.md`

**Structure:**
1. Header + tagline (5 lines)
2. What is Pivot (3-4 sentences)
3. Quick Example (keep the existing pipeline.py example)
4. Key Features (1-line bullets with links to docs)
5. Installation (3 lines)
6. Documentation link
7. Requirements (Python 3.13+, Unix)
8. Development (brief)

**Fix:**
- `anthropics.github.io` → `sjawhar.github.io`
- Remove "Internal development team", "Last Updated: 2026-01-10", "Version: 0.1.0-dev"
- Remove duplicated comparison table, FAQ, architecture diagram, technical details, development roadmap

**Step 1: Rewrite, step 2: Commit**

```bash
jj describe -m "docs: rewrite README as concise landing page"
```

---

## Task 16: Update docs/index.md and Clean Up Old Pages

**Files:**
- Modify: `docs/index.md`
- Create: `docs/solutions/index.md`
- Delete: old `docs/reference/` pages replaced by concepts/ (dependencies.md, outputs.md, parameters.md, pipelines.md, discovery.md, watch.md, configuration.md, matrix.md)
- Delete: old `docs/tutorial/` files replaced by guides/
- Delete: `docs/design/index.md` (or update to stub)
- Keep: `docs/reference/` directory for auto-generated API docs

**Step 1: Update docs/index.md** — new nav links, remove stale roadmap

**Step 2: Create solutions/index.md** — categorized table of all 30+ solution docs

**Step 3: Delete old reference pages** replaced by concepts/

**Step 4: Delete old tutorial pages** replaced by guides/

**Step 5: Verify no broken internal links**

```bash
uv run mkdocs build --strict 2>&1
```

**Step 6: Commit**

```bash
jj describe -m "docs: clean up old pages, add solutions index, update landing page"
```

---

## Task 17: Update Architecture Docs for Engine Refactors

**Files:**
- Modify: `docs/architecture/engine.md`
- Modify: `docs/architecture/execution.md`
- Modify: `docs/architecture/watch.md`
- Modify: `docs/architecture/tui.md`
- Modify: `docs/architecture/overview.md`

**Approach:**
1. Read each doc, cross-reference with current code
2. Fix class names, file paths, component relationships
3. Key changes: Engine+Scheduler split (#409), WorkerPool/sink supervision (#434), WatchCoordinator extraction (#436), TUI is pure RPC client (#427)

```bash
jj describe -m "docs(architecture): update for engine refactors"
```

---

## Task 18: Final Verification

**Step 1: Full strict build**

```bash
uv run mkdocs build --strict 2>&1
```

**Step 2: Check for YAML-first patterns**

```bash
# No concepts/guides page should lead with YAML before Python
grep -rn "^stages:" docs/concepts/ docs/guides/ docs/getting-started/
# Should return empty
```

**Step 3: Check all output/dep types documented**

```bash
grep -l "PlaceholderDep\|DirectoryOut\|IncrementalOut\|Metric\|Plot\b" docs/concepts/
```

**Step 4: Verify no broken links to old reference/ pages**

```bash
grep -rn "reference/pipelines\|reference/dependencies\|reference/outputs\|reference/parameters\|reference/discovery\|reference/watch" docs/ mkdocs.yml
```

**Step 5: Commit**

```bash
jj describe -m "docs: final verification pass"
```

---

## Execution Order

```
Task 1: Fix broken build              ─── FIRST (unblocks everything)
Task 2: Restructure nav               ─── SECOND (sets up skeleton)
│
├── Tasks 3-10: Concepts pages         ─── SEQUENTIAL (each builds on previous)
│   3: Artifacts & DAG
│   4: Dependencies
│   5: Outputs
│   6: Loaders
│   7: Parameters
│   8: Pipelines
│   9: Fingerprinting
│   10: Caching
│
├── Tasks 11-13: Guides                ─── PARALLEL (independent of each other)
│   11: Cross-repo import
│   12: Watch, multi-pipeline, remote
│   13: CI integration, TUI
│
├── Task 14: CLI reference update      ─── PARALLEL with guides
├── Task 15: README rewrite            ─── PARALLEL with guides
├── Task 16: Landing page + cleanup    ─── AFTER concepts (deletes old pages)
├── Task 17: Architecture update       ─── PARALLEL with everything
│
└── Task 18: Final verification        ─── LAST
```

## Summary

| Task | Scope | Effort |
|------|-------|--------|
| 1. Fix broken build | 4 files | Small |
| 2. Restructure nav | mkdocs.yml + placeholders | Small |
| 3. Artifacts & DAG | 1 new page | Medium |
| 4. Dependencies | 1 new page | Medium |
| 5. Outputs | 1 new page | Medium-Large |
| 6. Loaders | 1 new page | Medium |
| 7. Parameters | 1 new page | Small-Medium |
| 8. Pipelines | 1 new page | Medium-Large |
| 9. Fingerprinting | 1 new page | Medium |
| 10. Caching | 1 new page | Medium |
| 11. Cross-repo import | 1 new page | Small-Medium |
| 12. Watch/multi-pipeline/remote | 3 rewrites | Medium |
| 13. CI integration + TUI | 2 new pages | Medium |
| 14. CLI reference update | 1 file | Small-Medium |
| 15. README rewrite | 1 file | Medium |
| 16. Landing page + cleanup | deletions + updates | Small-Medium |
| 17. Architecture update | 5 files | Medium-Large |
| 18. Final verification | Checks only | Small |
