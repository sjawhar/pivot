# Common errors and anti-patterns

Read this when loading or running a pipeline fails, or when you port a DVC pipeline.

## Anti-patterns

| Don't | Do |
|---|---|
| One shared `params.py` (or config module) holding every stage's settings | One `StageParams` subclass beside each stage |
| A half-migration from DVC: `dvc.api.params_show()`, reading `params.yaml` in a stage, `dvc.yaml` or a YAML stage list next to `pipeline.py`, path strings passed between stages, `pd.read_csv`/`to_csv` inside stages | Port every stage to `@stage`, every input to `pipeline.input()`, every edge to a handle, every setting to `StageParams`; delete the DVC files |
| Mutable configuration in module-level constants (dicts of settings, lists of variants) | `StageParams` fields; loop over values inside the `with` block |
| Loader annotations on parameters, `Dep(...)`/`Out(...)` markers, path overrides, a `register` call | The function signature: handles in, data out |

## Common errors

| Error | Fix |
|---|---|
| `cannot import name 'Out' from 'pivot'` or `module 'pivot' has no attribute 'Out'` (or `Dep`, `Metric`) | Those markers do not exist; use the return type and `metric`/`plot` tags |
| `does not define a 'pipeline' variable` | `with Pipeline("name") as pipeline:` |
| `No module named '<yours>'` | Import from the project root: `from report.helpers import x`, not `from helpers import x`, in `report/pipeline.py` |
| `parameter 'x' has unsupported type` | Pass handles or `StageParams`; move plain values into `StageParams` |
| `received handle from multi-output stage` | Select a field: `handle.field` |
| `Cannot infer serialization format` | Return a supported type or `Annotated[T, loaders.X()]` |
| `captures non-scalar variable 'X'` | Move `X` into `StageParams` |
| `depends on input '<name>' (path data/raw/<name>), which does not exist` | Put the file at `data/raw/<name>` or pass `path=` |
| `must have a return type annotation` | Annotate the return; `-> None` stages have no outputs |
| `Invalid variant name 'big one': use only letters, digits and the characters _ . -` | Rename the variant; `/`, `@` and spaces are not allowed |
| `pipeline.map() needs a directory output` | Map a stage's `directory` output, or a directory field (`out.parts`) |
| `cannot expand over the files of '<stage>'` (a run, or "not yet expanded" in `pivot status`) | Rename the producer's files so each key (path without extension) is a valid variant name and unique |
| `'<stage>' is mapped over the files of '<producer>'` | Use `pivot repro <stage>`; `pivot run` cannot expand a mapped stage |
| `has the name of an instance of mapped stage '<stage>'` | Call the mapped stage, or the stage beside it, inside a `pipeline.variant(...)` block |
| `is tagged directory but its type is <type>` | A `directory` output's type is `dict[str, T]`, from file path to value |
| `whose loader <loader> cannot read it back` | Give that output a full `Loader`, or stop passing its handle to a stage |
| `depends on file 'x' of stage '<stage>': key must include file extension` | Index a directory handle by the file's path as the producer writes it, extension included: `parts["train.jsonl"]` |
| `Input binding conflict: inputs 'a' and 'b' are both bound to '<path>'` | Use one `pipeline.input()` name per file |
| `Output path conflict: 'p/x/y' and 'p/x:y' both write data/p/x/y.jsonl` | Rename a stage, pipeline or `TypedDict` field so each output has its own path |
| `wrote its YAML output in a form Pivot cannot read back` (a warning) | Convert the value before returning it: `df.to_dict()`, `dataclasses.asdict(x)`, or pick a format with `Annotated[T, loaders.X()]` |
