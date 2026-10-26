# Fingerprinting

Pivot fingerprints your stage functions so it knows when code changes.
Unlike file-level hashing (which triggers on whitespace or comment edits),
Pivot's fingerprinting is Abstract Syntax Tree (AST) based — it captures the
*structure* of your code, ignoring cosmetic changes.

## What Gets Tracked

When Pivot builds a stage, it computes a **manifest**: a dict mapping
logical keys to hashes:

| Manifest key | What it represents |
|--------------|--------------------|
| `self:<name>` | The stage function's own AST |
| `func:<module>.<qualname>` | Helper functions called by the stage |
| `class:<module>.<qualname>` | User-defined classes referenced by the stage |
| `const:<module>.<name>` | Immutable module-level values, read as a global or as `module.NAME`; a helper read as `module.helper` is a `func:` entry, a library function an `ext:` entry |
| `const:<module>.<qualname>.<name>` | Immutable values in a closure cell, keyed under the function that holds the cell, so two closures from one factory (`h2 = make(2)`, `h5 = make(5)`) keep their own values |
| `default:<module>.<qualname>.<parameter>` | Default argument values the AST does not record |
| `ext:<module>.<name>` | Library functions and classes a stage reads (`agg = np.mean`), recorded by name (`numpy.mean`); library code is not hashed |
| `schema:<module>.<qualname>` | Pydantic model JSON schemas |
| `wire:<parameter>` | Call-site artifact wiring: a direct identity, or a collection's kind and ordered identities; input path bindings use `wire:<parameter>:path` or `:paths` |
| `dep:…:loader:<module>.<class>:…`, `out:…:loader:<module>.<class>:…`, `out:…:tag` | Reader/Writer method ASTs and config for each input and output, plus each output's placement and write semantics |

### How It Works

For each stage function, Pivot:

1. **Parses the AST** — `inspect.getsource()` → `ast.parse()` →
   normalize (strip docstrings, rename `def` to `func`) → `ast.dump()` →
   `xxhash64`
2. **Walks closure variables** — `inspect.getclosurevars()` finds globals
   and nonlocals the function references
3. **Recurses transitively** — every user-code callable found in step 2
   gets the same treatment, and *its* dependencies are merged into the
   manifest. A user class the stage reaches has its callable members walked:
   methods, descriptors such as properties, partial members, and nested user
   classes. A module value or helper read only inside a method
   (`ScopeWalk().step(...)` reading a module-level regex) is tracked
4. **Follows imports in the function body** — a user module discovery already
   loaded is walked like a module-level dependency, together with its packages
   and a submodule a `from` import names (`import pkg.sub`, `from pkg import
   sub`). A body import of a user module discovery did not load is refused
   when the pipeline is defined, without running it: Pivot would have to
   execute the module to fingerprint it. The error names the module; import
   it at module level (or in `pipeline.py`) so discovery loads it, or move the
   value into the stage's `StageParams`
5. **Inspects type hints** — user-defined classes in annotations
   (including Pydantic models) are fingerprinted
6. **Hashes Pydantic schemas** — `model_json_schema()` captures field
   types, defaults, and validators

The manifest is stored in the per-stage lock file. On the next run, the
worker recomputes the manifest and compares it key-by-key against the
locked version. Any difference triggers re-execution.

While a stage runs, its worker records the user modules it imports through
Python's `__import__` and `importlib.import_module` entry points, including a
module a warm worker already holds, and compares the `sys.modules` keys before
and after the stage to catch a newly registered user module loaded from a
source file. What a covered module's first import starts belongs to that
import, as it would for a module-level import: every import on the thread
running it, every hooked import on a thread it started (and on threads those
start) for the rest of the stage, already resident or not, and every module
any thread registers without the hooks while such an import or thread runs.
Every thread Python starts counts: each start function `_thread` exposes
(`start_new_thread`, `start_new`, `start_joinable_thread`), which
`threading.Thread`, `threading.Timer` and `ThreadPoolExecutor` start through.
A thread native code starts outside `_thread` is not covered.
A hooked import on any other thread is counted, already resident or not, and
so is a module registered without the hooks at any other time, even when a
covered import later imports it. A stage that imported a user module its
fingerprint does not cover still succeeds, with one warning naming the module,
but its result is never reused:
Pivot files no run-cache entry, removes any entry an earlier run filed for the
same inputs, never restores one for that stage, and writes a lock the next run
cannot match; `pivot status --explain` and `pivot repro --dry-run --explain`
show the stage as stale with that reason, and `pivot verify` reports it as
failed with that reason until it reruns without an uncovered import.
`pivot commit` keeps that mark when it
records the same outputs, so the stage still reruns. A user class reached only
through a loaded input (an object in a `loaders.Pickle()` input whose class the
stage never names) counts as a module the stage imported; name the class in the
parameter's annotation (`Annotated[Box, loaders.Pickle()]`) to cover it. To
clear the warning, import the module statically and read it by name in the
stage's own code (`import pkg.mod`, then `pkg.mod.NAME`, or pass `pkg.mod`
itself to the call that needs it), so the walk covers it, or move the value
into `StageParams`. An import in `pipeline.py` alone does not cover a module
for a stage that never reads it, and a body import that only silences the
warning while the stage still reads values through the object a dynamic import
returned leaves those values untracked. A `@no_fingerprint` stage is not
checked: its declared `code_deps` are its code.

That check does not see every way a stage can reach code. In most of these
cases Pivot reuses a result whose fingerprint missed the code, as it would
without the check. In three it reruns a stage it could have reused: a thread
that existed before a covered import, a thread left from an earlier stage (a
covered thread of that stage included) that imports while a later stage runs,
and a thread native code starts outside `_thread`:

- a read of a module already in `sys.modules` without an import, or
  `importlib.reload`: import the module where the stage reads it;
- a warm `EntryPoint.load()` or `importlib.__import__` of a user module another
  stage already made resident: it creates no new `sys.modules` key, and the
  loader may hold an importer before the hook, so it is not fingerprinted, as
  on main; import the module statically;
- private import machinery, or a loader that replaces a module already in
  `sys.modules` or never registers one, such as `runpy.run_path("helpers/lib.py")`
  or `runpy.run_module("toplib")` for a top-level module: import it statically,
  or move the value into `StageParams`;
- an import in a child process: declare the code the child runs as an input,
  or list it in `@no_fingerprint(code_deps=[...])`;
- an import from a thread that outlives its stage: the import is charged to
  whichever stage is running when it happens, which then warns and reruns, or
  to none; a covered thread still running in a later stage is not covered for
  it; join the thread before the stage returns;
- a covered module that hands work at import time to a thread that existed
  before its import, such as a pool an earlier import created: that thread is
  not covered, so its imports are counted and the stage reruns. When an earlier
  stage in the same worker started the thread, whether the stage warns depends
  on which stage ran first there: where the covered import itself starts the
  thread, the thread is covered and the stage caches. Start the thread or pool
  inside the module, or import the module statically;
- a thread native code starts outside `_thread`, such as one a C extension
  starts or `pthread_create` called through `ctypes`: no start hook sees it, so
  its imports are counted and the stage warns and reruns; import the module
  statically;
- a user module a covered thread imports that has nothing to do with the
  covered module: it is excused and not fingerprinted, as on main, so an edit
  to it is seen only through values the stage reads from the covered module,
  or when the covered module changes; import user modules statically;
- a user module another thread registers without the hooks while a covered
  import or covered thread runs: it is excused with that import's own, so an
  edit to it is seen only through values the stage reads from the covered
  module, or when the covered module changes; import user modules statically,
  or before starting threads;
- a user module a covered module loads at import time through `importlib.util`
  and `exec_module`, `EntryPoint.load()` or `importlib.__import__`: it is
  excused, as on main, and not fingerprinted itself, so an edit to it is seen
  only through values the stage reads from the covered module, or when the
  covered module changes; import it statically;
- a package resource read through `importlib.resources` (for example
  `files(helpers).joinpath("table.txt")`): it is data, not code; declare it
  with `pipeline.input`;
- a module created without a source file (CPython's `pyexpat.errors`,
  Cython's `_cython_3_3_0`): it has no source to fingerprint;
- `pivot repro --no-commit` followed by `pivot commit`: no lock exists to carry
  the mark, so `pivot commit` records the outputs on disk against the static
  fingerprint, as it records any output; run `pivot repro` without
  `--no-commit` for a stage that imports dynamically;
- a result cached by an earlier Pivot version: if its static fingerprint is
  unchanged, it may be reused because `_CACHE_SCHEMA_VERSION` keys only the
  state.lmdb caches, not the lock. A stage whose static walk reaches a user
  class method or a user module gains entries in its manifest, so it reruns
  once as Code changed after the upgrade. That accepted one-time rerun writes
  the current manifest; later runs reuse it until code or inputs change. Run
  `pivot repro --force <stage>` once to re-check a stage you know imports
  dynamically. Upgrading from 0.5.6 or earlier, every stage with outputs reruns
  once as Code changed, because each output's tag (`out:<key>:tag`) joined the
  manifest; that run is checked, so it marks a stage that imports dynamically.


### Transitive Tracking

If `train()` calls `normalize()` which calls `clip_outliers()`, all three
functions appear in the manifest. Changing `clip_outliers` triggers a
re-run of `train` even though `train` itself didn't change.

Module-level scalars, exact tuples and frozensets of scalars, and compiled
regular expressions are tracked by value. If your stage does `from myproject
import config` and then accesses `config.THRESHOLD`, Pivot captures the value
or the called function. A stage reading any other module-level value, such as a
list, dict, set, nested collection, enum member, frozen model, or instance,
raises `StageDefinitionError`; use [StageParams](parameters.md) instead.

## What Triggers a Re-Run

| Change | Triggers re-run? |
|--------|:----------------:|
| Rename a variable | Yes |
| Change a numeric constant | Yes |
| Add/remove a function argument | Yes |
| Change a helper function your stage calls | Yes |
| Change a Pydantic field type or default | Yes |
| Edit a docstring | **No** |
| Add/remove comments | **No** |
| Change whitespace/formatting | **No** |
| Change an unused import | **No** |

## Surprises and Pitfalls

### Lambdas

A lambda in a source file is hashed by its AST like any function. One created
dynamically (via `eval`, `exec`, or an interactive session) has no source, so
Pivot hashes its code object (`__code__`: bytecode and constants, without the
source path). Its fingerprint is the same from run to run, and editing the
expression re-runs the stage. Only a callable with neither source nor a code
object falls back to `id()`, which changes every run; Pivot warns when it does.
Prefer named, module-level functions, whose AST hash ignores comments and
formatting:

```python
# Works, hashed by its code object
normalize = eval("lambda df: df / df.max()")


# Better - AST fingerprint
def normalize(df):
    return df / df.max()
```

### Module-Level Values

A stage may read a module-level `bool`, `int`, `float`, `str`, `bytes` or
`None`, a `tuple` or `frozenset` of such values, or a compiled regular
expression; editing it re-runs the stage. A regex is tracked by its pattern and
flags. Reading any other module-level value (list, dict, set, a tuple holding a
tuple, list or object, enum member, class instance, or a method bound to an
instance such as `np.random.default_rng(SEED).normal`), directly or as
`module.NAME`, is a `StageDefinitionError` that names the variable:

```python
COLUMNS = {"score": float}  # Module-level dict


@stage
def process(data: pd.DataFrame) -> pd.DataFrame:
    return data.astype(COLUMNS)  # Error!
```

The same rule covers a default argument that reads a module value, such as
`def process(data, cols=COLUMNS)`, on the stage or on a helper it calls. A
default is evaluated when the module loads, so Pivot checks its value: a scalar,
a tuple or frozenset of scalars, or a compiled regex is tracked, and anything
else is refused. A literal default such as `cols=("a", "b")` is part of the
function's code and is tracked with it. A default that is a pipeline input
handle (`raw=RAW` with `RAW = pipeline.input("t.csv")`) is wired as an input
and tracked by its content.

Fix: move a mutable or nested value into the stage's [StageParams](parameters.md).
Configuration in module-level mutable collections is not supported.

To suppress the check (at your own risk), set
`core.unsafe_fingerprinting = true` in your Pivot config
(`pivot config set core.unsafe_fingerprinting true`) or
`PIVOT_UNSAFE_FINGERPRINTING=1` in the environment. A stage whose fingerprint
needed that downgrade still runs, but its result is never filed in the run
cache or restored from it: the captured value may differ next time while the
fingerprint stays the same.

Pivot rejects patterns that bypass static analysis:

- `globals()` / `locals()` — runtime namespace access
- `getattr(obj, variable)` — dynamic attribute lookup
- `importlib.import_module()` — dynamic imports

All of these silently introduce dependencies that fingerprinting can't
track. Use direct attribute access and static imports instead. Other dynamic
import spellings (`__import__`, an alias of `import_module`, `exec`) are not
refused: a user module one of them loads while the stage runs makes its result
not reusable, as described above, and a library module costs nothing.

## @pivot.no_fingerprint()

For stages where AST fingerprinting doesn't work (C extensions, generated
code, complex metaprogramming), opt out with `@no_fingerprint()`:

```python
from pivot import no_fingerprint, stage


@stage
@no_fingerprint()
def external_model_stage(data: pd.DataFrame) -> pd.DataFrame: ...


@stage
@no_fingerprint(code_deps=["scripts/train.sh", "configs/model.yaml"])
def shell_stage(data: pd.DataFrame) -> dict: ...
```

With `@no_fingerprint()`, Pivot falls back to **file-level hashing**: it
hashes the entire source file containing the function. The optional
`code_deps` argument lists additional files (relative to the project root)
that are part of the stage's code; editing one marks the stage "Code changed".

**Use sparingly.** File-level hashing is less precise: any change anywhere
in the file triggers a re-run.


## @pivot.fingerprint_by_name

A helper function that cannot change a stage's output — process bookkeeping,
file locks, log rotation — can carry this mark so Pivot records it by its
module and qualified name only, never its body:

```python
from pivot import fingerprint_by_name


@fingerprint_by_name
def process_start():
    """Read /proc for a running process."""
    ...
```

A stage that calls a marked helper does not rerun when the helper's body
changes. It reruns once when the helper moves to another module, when the
helper gets a new name, or when someone removes the mark.

The decorator accepts a module-level function, and a `functools.wraps`,
`contextlib.contextmanager` or class-based wrapper of one (an instance that
`functools.update_wrapper` made). It refuses a class, a method, a lambda, a
nested function, a `functools.partial`, a helper that imports a user module in
its body, and a helper that uses `importlib.import_module` or `__import__`.
Import the module at the top of the helper's file instead.

The fingerprint walk never reads the body of a marked function. A decorator
above the mark is user code: Pivot adds a function decorator's own code to the
fingerprint and records the marked function inside it by name. This is also
true when `functools.cache` wraps that decorator. A library wrapper directly
above the mark (`functools.cache`, `functools.lru_cache`,
`contextlib.contextmanager`) keeps the record by name.

Pivot refuses a class-based decorator above the mark, however the stage
reaches the helper (a call, a class member, a partial argument, a default, a
pydantic `default_factory`). Pivot cannot fingerprint the code and state of a
class instance, and the mark records only the function below it.
`core.unsafe_fingerprinting` does not turn this refusal into a warning. To
clear it, put `@fingerprint_by_name` above the class-based decorator, so Pivot
records the whole helper by name; or replace the class with a function
decorator built with `functools.wraps`; or remove the mark.

Without a mark below it, Pivot fingerprints a `functools.wraps` decorator as
in earlier versions. It follows `__wrapped__` to the wrapped function, so it
does not see what the decorator's own lines read through a module (for
example `settings.FACTOR`), or the decorator's body when `functools.cache`
wraps the decorator.

A stage cannot carry `@fingerprint_by_name`: Pipeline refuses a stage with
the mark above `@stage`, below it, or below another decorator. Use
`@no_fingerprint()` to opt a stage out of AST fingerprinting.

Pivot does not walk code that only a marked helper calls. If that code
imports a user module in its body, the run-time check counts the import, and
the stage warns and reruns each run. Move that import to the top of its
module to clear it.

Soundness is the user's choice: if a marked helper can change a stage's
output, Pivot does not see the change.

## Performance

Fingerprinting runs once during pipeline discovery (single-threaded, in
the coordinator process). Results are cached at two levels:

1. **In-memory** — `WeakKeyDictionary` cache avoids re-parsing the same
   function within a single process
2. **Persistent** — AST hashes and full manifests are cached in StateDB,
   keyed by each source file's path and the digest of its content. If the
   source file's content hasn't changed, the cached hash is reused without
   parsing; an edit that keeps the file's mtime, size and inode is still
   seen

For a project with 125 stages, fingerprinting typically completes in under
100ms on subsequent runs.

## Python Version Dependency

AST fingerprinting means the Abstract Syntax Tree — not bytecode — is what
gets hashed. AST structure can vary between Python versions (e.g., 3.12 vs
3.13 may parse some constructs differently), so the same source code can
produce different fingerprints under different Python versions. This can
cause unnecessary stage re-runs when switching versions.

To avoid surprises:

- **Pin your Python version** in your project (e.g., `.python-version`)
- **Use uv** to manage your Python environment consistently
- **All team members and CI should use the same Python version**

## Relationship to Other Concepts

The fingerprint manifest is one of three inputs to the
[caching](caching.md) skip-detection algorithm, alongside
[parameters](parameters.md) and dependency hashes. A stage skips only
when all three match.
