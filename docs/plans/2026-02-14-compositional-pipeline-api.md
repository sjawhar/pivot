# Compositional Pipeline API

## Problem

The current Pivot API for defining pipelines is verbose, stage-centric, and brittle.
Evidence from the eval-pipeline codebase:

- **Path duplication**: `"data/horizon/wrangled/bootstrap/headline.csv"` appears as an
  `Out` in one stage and a `Dep` in 5+ others. Rename it and you're grep-and-replacing
  across files.
- **Override machinery is the main complexity driver**: `ModelReportConfig.build_model_report`
  is 240 lines of `dep_path_overrides` / `out_path_overrides`. That's plumbing, not
  pipeline logic.
- **Loaders repeated on every annotation**: every consumer of `filtered_runs` restates
  `DataFrameJSONL()`. The format is a property of the artifact, not something each
  consumer should declare.
- **Stage-centric, not artifact-centric**: the user thinks "this artifact feeds that
  artifact." The current API makes them think "this stage has these deps and these outs
  at these paths with these loaders."

The `report.py` builder (832 lines) exists solely to manage path routing for stage
reuse across pipelines. A new model report requires copy-pasting hundreds of lines of
registration boilerplate.

## Design Goals

1. **Easy to define and understand** — confusion means mistakes.
2. **Artifact-centric** — the DAG emerges from data flow, not explicit wiring.
3. **No paths in pipeline definitions** — Pivot manages all file organization.
4. **Composable** — same stage function reusable across pipelines without overrides.
5. **Fast DAG reads** — tab-complete and TUI in <10ms via cached manifest.
6. **Eager validation** — all wiring errors reported together before any stage executes.

## Core Model

### Stage Functions

Stage functions are plain Python with a `@stage` decorator. No `Dep()`, `Out()`,
`Annotated` on parameters, or path strings. The function is a pure data transform:

```python
@stage
def wrangle_bootstrap(
    filtered_runs: pd.DataFrame,
    params: BootstrapParams,
) -> pd.DataFrame:
    ...
```

The decorator is lightweight — it does not change the function's behavior when called
with real data (tests, notebooks call the function directly). When called within a
pipeline context, it returns an `ArtifactHandle` instead of executing.

Parameters are either:
- **Artifact inputs** — any parameter whose value is an `ArtifactHandle` at pipeline
  definition time.
- **Stage params** — a `StageParams` subclass, passed directly.

The distinction is positional: if the caller passes a handle, it's a dep. If they pass
a params object, it's params. No marker annotations needed.

### Serialization Format

Pivot infers a default format from the return type:

| Return type      | Default format  |
|------------------|-----------------|
| `pd.DataFrame`   | JSONL           |
| `dict`           | YAML            |
| `str`            | Text            |
| `pathlib.Path`   | PathOnly        |
| `Figure`         | PNG             |

Override only when the default is wrong, on the return type:

```python
@stage
def wrangle_logistic_regression(...) -> Annotated[pd.DataFrame, CSV()]:
    ...
```

This is the only place `Annotated` appears in the new API, and only when overriding
defaults.

### Plots and Metrics

Plots and metrics are regular outputs with a tag that controls how Pivot handles them
(persistence format, inclusion in `pivot metrics` / `pivot plots`, TUI display, CI
diffing):

```python
@stage
def generate_benchmark_results(...) -> Annotated[dict, metric]:
    ...

@stage
def plot_bootstrap_ci(...) -> Annotated[Figure, plot]:
    ...
```

For multi-output stages that mix data and metrics:

```python
class TaskWeightsResult(TypedDict):
    filtered_runs: pd.DataFrame
    task_weights: Annotated[dict, metric]

@stage
def compute_task_weights(runs: pd.DataFrame) -> TaskWeightsResult:
    return {"filtered_runs": df, "task_weights": metrics}
```

## Pipeline Definition

Pipelines are defined by composing stage calls within a `with Pipeline()` block. The
block sets an active pipeline via context variable. Stage calls record DAG nodes and
return `ArtifactHandle` objects. The DAG emerges from passing handles between stages.

```python
with Pipeline("eval_pipeline_horizon") as p:
    # External data — the ONLY place names/paths appear
    release_dates = p.input("release_dates")
    annotations = p.input("annotations")

    # Stages are function calls. Handles in, handles out.
    swe_bench_runs = fetch_swe_bench_runs(annotations)
    raw_runs = fetch_agent_runs(params=FetchParams(...))
    filtered = filter_runs(raw_runs, params=FilterParams(...))

    # Multi-output: attribute access on the handle
    tw = compute_task_weights(filtered)
    weighted_runs = tw.filtered_runs

    # Variants are just Python loops
    bootstraps = {}
    logistic = {}
    for name, cfg in LOGISTIC_VARIANTS.items():
        bootstraps[name] = wrangle_bootstrap(
            weighted_runs,
            params=BootstrapParams(variant=cfg),
        )
        logistic[name] = wrangle_logistic_regression(
            weighted_runs, bootstraps[name], release_dates,
            params=LogisticParams(variant=cfg),
        )

    # Plots consume handles directly
    for fig, scale in product(BOOTSTRAP_FIG_NAMES, ["log", "linear"]):
        plot_bootstrap_ci(
            logistic["headline"], bootstraps["headline"], release_dates,
            params=PlotParams(fig_name=fig, y_scale=scale),
        )

    generate_benchmark_results(
        logistic["headline"], bootstraps["headline"],
        weighted_runs, release_dates,
        params=BenchmarkParams(...),
    )
```

### No Restrictions on Pipeline Definition Code

The pipeline definition is ordinary Python. Loops, conditionals, dict comprehensions,
helper functions — all work because they're just Python that produces `@stage` calls.
Pivot executes the `with` block to build the DAG; it does not statically parse it.

### Stage Naming

The stage name is the function name. When the same function is called multiple times
(variants), Pivot disambiguates automatically (e.g. `wrangle_bootstrap@0`,
`wrangle_bootstrap@1`). An explicit `name=` keyword can be passed for human-friendly
labels when needed.

### Multi-Output Stages

Stages returning a `TypedDict` produce multiple artifacts. Each field is accessible
via attribute access on the returned handle:

```python
tw = compute_task_weights(filtered)
weighted_runs = tw.filtered_runs
weights = tw.task_weights
```

These sub-handles are independent artifacts that can be passed to different downstream
stages.

## Inputs and External Data

Inputs are artifacts not produced by any stage. They are declared by name only in
pipeline code:

```python
release_dates = p.input("release_dates")
annotations = p.input("annotations")
```

### Discovery via `.pvt` Sidecar Files

Pivot discovers inputs by scanning for `.pvt` marker files under `data/raw/` and
`data/external/`. Each `.pvt` file registers one input:

```
data/
├── raw/                                # p.input() artifacts
│   ├── release_dates.yaml              # actual data
│   ├── release_dates.pvt               # marker: "this is a Pivot input"
│   ├── annotations.csv
│   └── annotations.pvt
└── external/                           # pivot import (cross-repo, S3)
    ├── swe_bench_scores.csv            # downloaded data
    └── swe_bench_scores.pvt            # metadata: source, checksum
```

The `.pvt` file serves as:
- **Registration**: its existence means "Pivot knows about this input."
- **Name**: derived from filename — `release_dates.pvt` → `p.input("release_dates")`.
- **Metadata**: optional format override, source URL, checksum.

```yaml
# data/raw/release_dates.pvt (minimal — just a marker)
{}

# data/external/swe_bench_scores.pvt (imported — tracks source)
source: s3://bucket/path/scores.csv
checksum: a3f2c1...
```

Format is inferred from the data file's extension (`.yaml` → YAML, `.csv` → CSV,
`.jsonl` → DataFrameJSONL). Override in the `.pvt` file only when needed.

### Workflows

```bash
# Register a local file as input (creates the .pvt sidecar)
pivot input add release_dates data/raw/release_dates.yaml

# Import from S3 (downloads + creates .pvt with source tracking)
pivot import swe_bench_scores s3://bucket/scores.csv

# Re-import (re-downloads, updates checksum in .pvt)
pivot import --refresh swe_bench_scores
```

### Version Control

`.pvt` files are committed to version control. Data files in `data/raw/` may also be
committed (small config YAMLs) or gitignored (large datasets, with the `.pvt` tracking
the source for reproducibility).

### Resolution

`p.input("release_dates")` resolves by scanning `data/raw/**/*.pvt` and
`data/external/**/*.pvt` for a file named `release_dates.pvt`. The adjacent data file
(same base name, any extension) is the artifact. No central registry file to maintain.

## Cross-Pipeline Dependencies

Cross-pipeline references are Python imports. Artifact handles are module-level
variables — importable like any other Python object:

```python
# eval_pipeline/base/pipeline.py
with Pipeline("base") as p:
    raw = fetch_agent_runs(params=FetchParams(...))
    all_runs = create_virtual_rebench_tasks(raw)
# all_runs is a module-level ArtifactHandle

# eval_pipeline/horizon/pipeline.py
from eval_pipeline.base.pipeline import all_runs

with Pipeline("horizon") as p:
    filtered = filter_runs(all_runs, ...)
```

No string-based lookup. The handle IS the reference — it knows which pipeline produced
it, which stage, what type. Pivot sees the cross-pipeline edge and ensures the upstream
stage runs first.

This works because:
- `with` blocks don't create a Python scope — variables defined inside are module-level.
- Importing a pipeline module is cheap — `@stage` calls return handles, no stages
  execute.

## Stage Reuse Across Pipelines

The same stage function is reusable across pipelines without any override mechanism.
Different pipelines simply pass different handles:

```python
# Main horizon pipeline
with Pipeline("eval_pipeline_horizon") as p:
    filtered = filter_runs(base_all_runs, params=FilterParams(...))
    bootstraps["headline"] = wrangle_bootstrap(filtered, params=BootstrapParams(variant=headline_cfg))

# Model report pipeline — same functions, different inputs
def build_model_report(config: ModelReportConfig) -> Pipeline:
    with Pipeline(config.pipeline_name) as p:
        filtered_raw = p.input("filtered_runs_raw")
        tw = compute_task_weights(filtered_raw)
        bootstraps["headline"] = wrangle_bootstrap(tw.filtered_runs, params=BootstrapParams(variant=headline_cfg))
    return p
```

No `dep_path_overrides`. No `out_path_overrides`. The function is called with different
arguments — the same way you reuse any Python function.

## Artifact Storage

Pivot manages all file paths. The user never specifies where artifacts are stored.

Pivot derives storage locations from:
- Pipeline name
- Stage name (function name + disambiguation suffix)
- Output key (for multi-output stages)
- A configured data directory

The concrete layout is an implementation detail. Possible schemes:
- `{data_dir}/{pipeline}/{stage}/{output_key}.{ext}`
- Content-addressed storage with human-readable symlinks
- Configurable via `.pivot/config.yaml`

Artifacts are identified by their alias (a human-readable name) and optionally a
variant key for families of related artifacts:

```python
bootstraps["headline"]           # Python dict key — natural variant handling
bootstraps["swe_bench"]          # another variant, same family
tw.filtered_runs                 # named output of a multi-output stage
```

Variant keys are arbitrary — not constrained to fixed dimensions. They can come from
nested loops, arbitrary parameter sets, or computed labels. Pivot treats them as opaque
labels for display and retrieval.

### On-Disk Layout

The human-readable file tree is a **view** — symlinks into content-addressed storage.
Pivot auto-categorizes artifacts based on DAG topology and output tags:

```
project/
├── data/
│   ├── raw/                               # p.input() artifacts
│   │   ├── release_dates.yaml
│   │   └── annotations.csv
│   ├── external/                          # pivot import (cross-repo, S3)
│   ├── interim/                           # has downstream consumers
│   │   ├── filtered_runs.jsonl            → .pivot/cache/objects/a3f2c1...
│   │   └── bootstrap/
│   │       ├── headline.csv               → .pivot/cache/objects/b7d4e2...
│   │       └── swe_bench.csv              → .pivot/cache/objects/c9a1f3...
│   └── processed/                         # leaf nodes (no downstream consumers)
│       └── benchmark_results.yaml         → .pivot/cache/objects/d2e5a7...
├── metrics/                               # tagged: metric
│   └── horizon/
│       ├── task_weights.yaml              → .pivot/cache/objects/e1b3c4...
│       └── logistic_fits/
│           └── headline.yaml              → .pivot/cache/objects/f5a2d8...
├── plots/                                 # tagged: plot
│   └── horizon/
│       └── bootstrap_ci_headline_log.png  → .pivot/cache/objects/a8c3e1...
└── .pivot/
    └── cache/
        └── objects/                       # content-addressed store
            ├── a3f2c1...
            └── ...
```

Categorization rules:

| Signal                                   | Directory          |
|------------------------------------------|--------------------|
| `p.input()`                              | `data/raw/`        |
| `pivot import` (cross-repo, S3, etc.)    | `data/external/`   |
| Has downstream consumers                 | `data/interim/`    |
| No downstream consumers (leaf), untagged | `data/processed/`  |
| Tagged `metric`                          | `metrics/`         |
| Tagged `plot`                            | `plots/`           |

Within each category, named by `{stage_name}.{ext}` or `{stage_name}/{key}.{ext}` for
variant families.

The symlink tree is disposable — Pivot regenerates it from the cache after each run.
The content-addressed cache is the source of truth, providing:

- **Human browsability**: `cat data/interim/filtered_runs.jsonl` to debug.
- **Dedup**: identical content stored once.
- **History**: old versions stay in cache until GC'd; symlinks point to current.
- **Portability**: push cache to S3, pull elsewhere, regenerate symlinks.

## Validation

Pivot validates eagerly but reports all errors together. During the `with Pipeline()`
block, each `@stage` call records a DAG node and checks types. After the block
completes, Pivot reports all validation errors at once:

```
Pipeline "mr_time_horizon_1_1" has 3 validation errors:

  wrangle_bootstrap (call #2):
    Parameter 'filtered_runs' expects pd.DataFrame, got ArtifactHandle[dict]

  plot_bootstrap_ci (call #5):
    Parameter 'agent_summaries' expects pd.DataFrame, got ArtifactHandle[dict[str, Any]]

  compute_trendline_ci (call #8):
    Missing required parameter 'release_dates' (no default, no handle provided)
```

One pass, fix everything. No errors deferred to execution time.

## Performance and Fingerprinting

### Pipeline Definition Overhead

Importing `pipeline.py` executes the `with` block, which imports stage modules (for
the `@stage` decorator). This loads their module-level dependencies. Mitigation:

- Stage functions should use in-body imports for heavy libraries (pandas, matplotlib).
- Pivot could support lazy function resolution — register a module path + function
  name at DAG-build time, resolve the actual function at execution time.

### Fingerprinting

Same content-addressed model as today. A stage needs re-running when any of these
change:

| Input                | Hash source                              |
|----------------------|------------------------------------------|
| Function code        | Source code of the `@stage` function     |
| Params               | Serialized `StageParams` object          |
| Input artifacts      | Content hash of upstream artifacts       |

The compositional model doesn't add fingerprinting complexity — dependencies are
explicit handle references rather than path-matched strings.

### DAG Build Cost

The current system builds the DAG via path matching across registries with three-tier
external discovery and string comparisons. The new system builds the DAG directly from
handle references recorded during the `with` block — no post-hoc matching. Should be
faster.

## Cached Manifest for Fast Reads

The pipeline definition is arbitrary Python and cannot be statically parsed for the
DAG. Fast reads come from a cached manifest.

### Write Step

After any `pipeline.py` changes:
1. Import the module, execute the `with` block, build the DAG.
2. Serialize to `.pivot/cache/manifest.json`: stage names, function references,
   dependency edges, param hashes, output keys, tags (metric/plot).

### Read Step

Tab-complete, TUI, `pivot status`:
1. Read the manifest — sub-millisecond.
2. No Python import needed.

### Invalidation

Only `pipeline.py` files determine DAG structure. Stage function code, config files,
and data changes affect fingerprinting and freshness — not the DAG shape.

Invalidation check: have any `pipeline.py` files changed (by mtime)? If not, the
manifest is fresh. This is a handful of stat calls — essentially free.

| Concern          | What determines it     | When evaluated           |
|------------------|------------------------|--------------------------|
| DAG structure    | `pipeline.py` files    | Manifest build (cached)  |
| Stage staleness  | function code + params + input hashes | Execution time |
| Input resolution | `.pivot/config.yaml`   | Execution time           |

## What Changes vs Current API

### Removed

- `Dep()`, `Out()`, `Metric()`, `DirectoryOut()` annotations on function parameters
- `Annotated[T, Dep("path", Loader())]` on every input
- `dep_path_overrides` / `out_path_overrides` dictionaries
- Path strings in pipeline definitions
- Loader/format repeated on every consumer of an artifact
- TypedDict output classes (for single-output stages)
- `Pipeline.register()` with override kwargs
- Report builder classes (e.g. `ModelReportConfig.build_model_report`)

### Added

- `@stage` decorator on stage functions
- `with Pipeline("name") as p:` context manager
- `p.input("name")` for external data
- `ArtifactHandle` objects (returned by stage calls, passed between stages)
- Cached manifest (`.pivot/cache/manifest.json`)
- `metric` / `plot` output tags

### Preserved

- Stage functions as pure, serializable, module-level functions
- `StageParams` for stage configuration
- Content-addressed fingerprinting (function code + params + inputs)
- Per-pipeline state directories and lock files
- `Pipeline.include()` for pipeline composition
- `pivot repro` / `pivot run` CLI semantics

## Example: Before and After

### Stage Definition

```python
# BEFORE
def compute_task_weights(
    filtered_runs_without_weights: Annotated[
        pd.DataFrame,
        Dep("data/horizon/interim/filtered_runs_without_weights.jsonl", DataFrameJSONL()),
    ],
) -> ComputeTaskWeightsOutputs:
    ...

class ComputeTaskWeightsOutputs(TypedDict):
    filtered_runs: Annotated[
        pd.DataFrame,
        Out("data/horizon/interim/filtered_runs.jsonl", DataFrameJSONL()),
    ]
    task_weights: Annotated[
        dict[str, Any],
        Metric("metrics/horizon/task_weights.yaml", YAML()),
    ]

# AFTER
@stage
def compute_task_weights(
    filtered_runs_without_weights: pd.DataFrame,
) -> TaskWeightsResult:
    ...

class TaskWeightsResult(TypedDict):
    filtered_runs: pd.DataFrame
    task_weights: Annotated[dict, metric]
```

### Pipeline Definition

```python
# BEFORE (report.py — 240 lines of this pattern)
p.register(
    compute_task_weights.compute_task_weights,
    name="compute_task_weights",
    dep_path_overrides={
        "filtered_runs_without_weights": _p("data/interim/filtered_runs_without_weights.jsonl"),
    },
    out_path_overrides={
        "filtered_runs": _p("data/interim/filtered_runs.jsonl"),
        "task_weights": _p("metrics/task_weights.yaml"),
    },
)
p.register(
    generate_agent_summary.generate_agent_summary,
    name="generate_agent_summary",
    dep_path_overrides={
        "filtered_runs": _p("data/interim/filtered_runs.jsonl"),
    },
    out_path_overrides={
        "agent_summary": _p("metrics/agent_summary.csv"),
        "scores_by_agent_and_task": _p("metrics/scores_by_agent_and_task.yaml"),
    },
    params=GenerateAgentSummaryParams(focus_agents=config.agent_summary_agents),
)

# AFTER
tw = compute_task_weights(filtered_raw)
summary = generate_agent_summary(tw.filtered_runs, params=GenerateAgentSummaryParams(focus_agents=config.agent_summary_agents))
```

### Model Report Builder

```python
# BEFORE: 240 lines (report.py ModelReportConfig.build_model_report)
# AFTER: ~30 lines
def build_model_report(config: ModelReportConfig) -> Pipeline:
    with Pipeline(config.pipeline_name) as p:
        filtered_raw = p.input("filtered_runs_raw")
        release_dates = p.input("release_dates")

        tw = compute_task_weights(filtered_raw)

        bootstraps = {}
        logistic = {}
        for name, cfg in config.logistic_variants.items():
            bootstraps[name] = wrangle_bootstrap(
                tw.filtered_runs,
                params=BootstrapParams(variant=cfg),
            )
            logistic[name] = wrangle_logistic_regression(
                tw.filtered_runs, bootstraps[name], release_dates,
                params=LogisticParams(variant=cfg),
            )

        for fig, scale in product(config.bootstrap_fig_names, config.y_scales):
            plot_bootstrap_ci(
                logistic["headline"], bootstraps["headline"], release_dates,
                params=PlotParams(fig_name=fig, y_scale=scale, styling=config.styling),
            )

        generate_benchmark_results(
            logistic["headline"], bootstraps["headline"],
            tw.filtered_runs, release_dates,
            params=BenchmarkParams(
                benchmark_name=config.benchmark_results.benchmark_name,
            ),
        )

        for name, tcv in config.trendline_ci_variants.items():
            compute_trendline_ci(
                bootstraps[tcv.data_file], logistic[tcv.data_file], release_dates,
                params=TrendlineCIParams(
                    after_date=tcv.after_date,
                    before_date=tcv.before_date,
                ),
            )

        compare_trend(
            bootstraps["headline"], release_dates,
            params=config.compare_trend_params,
        )

    return p
```

## Migration

Clean break. Pivot is pre-alpha — no compatibility shims, no dual API support. The
current `Dep()` / `Out()` / `Pipeline.register()` API is removed entirely and replaced
by the compositional model.

## Watch Mode

Three things to watch, cleanly separated:

| What changes             | How detected                          | What happens                                     |
|--------------------------|---------------------------------------|--------------------------------------------------|
| External input file      | Watch paths from input config         | Re-fingerprint → re-run stale subgraph           |
| Stage function code      | Watch `@stage` function source files  | Re-fingerprint → re-run that stage + downstream  |
| Pipeline definition      | Watch import graph of `pipeline.py`   | Re-import → rebuild DAG → diff → re-run affected |

### Detecting Pipeline Definition Changes

Pipeline definitions can import config from other modules (e.g. variant configs,
agent lists). Watching only `pipeline.py` would miss those. Instead, track the
import graph:

```python
before = set(sys.modules.keys())
import pipeline_module   # executes the with-block, builds DAG
after = set(sys.modules.keys())
contributing_modules = after - before  # → source files to watch
```

Filter to project-local modules only (ignore third-party libraries).

### DAG Diffing on Re-Import

When `pipeline.py` is re-imported during watch mode, Pivot builds a new DAG and
diffs it against the previous one. Stage identity is `(function, params, inputs)` —
stable across re-imports even though Python objects differ. Only the delta triggers
re-runs: new stages, removed stages, changed wiring, changed params.

### No Intermediate Artifact Watching

Pivot controls when intermediate artifacts are written. No filesystem watches needed
on the data directory — eliminates spurious triggers from managed storage.

## Resolved Decisions

- **Lazy imports**: No. Eager importing at DAG build time. Catches import errors early.
  Convention: heavy dependencies (pandas, matplotlib, etc.) should be imported inside
  function bodies, not at module level. This keeps DAG builds fast without deferred
  surprises.
- **Storage**: Content-addressed object store (`.pivot/cache/objects/`) with a
  human-readable symlink tree (`data/`, `metrics/`, `plots/`) auto-categorized by
  DAG topology and output tags. See "On-Disk Layout" section.
- **Manifest format**: JSON. Human-readable, debuggable, fast enough for DAG-sized
  data (kilobytes). Optimize to msgpack only if profiling shows it matters.
- **`Pipeline.include()`**: Kept but simplified. Cross-pipeline dependencies are
  handled by importing artifact handles (Python imports). `include()` remains useful
  for merging pipelines into one for a combined `pivot repro`. No stage renaming
  needed — stages have unique identity from `(function, params, inputs)`.
