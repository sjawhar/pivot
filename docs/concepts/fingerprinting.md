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
| `dep:…:loader:<module>.<class>:…`, `out:…:loader:<module>.<class>:…` | Reader/Writer method ASTs and config for each input and output |

### How It Works

For each stage function, Pivot:

1. **Parses the AST** — `inspect.getsource()` → `ast.parse()` →
   normalize (strip docstrings, rename `def` to `func`) → `ast.dump()` →
   `xxhash64`
2. **Walks closure variables** — `inspect.getclosurevars()` finds globals
   and nonlocals the function references
3. **Recurses transitively** — every user-code callable found in step 2
   gets the same treatment, and *its* dependencies are merged into the
   manifest. A user class the stage reaches has each method its source
   defines walked the same way, so a module value or helper read only
   inside a method (`ScopeWalk().step(...)` reading a module-level regex)
   is tracked
4. **Follows imports in the function body** — `import helpers` or
   `from helpers import scale` inside a stage or helper is walked as if it
   were a module-level import; a library module adds nothing
5. **Inspects type hints** — user-defined classes in annotations
   (including Pydantic models) are fingerprinted
6. **Hashes Pydantic schemas** — `model_json_schema()` captures field
   types, defaults, and validators

The manifest is stored in the per-stage lock file. On the next run, the
worker recomputes the manifest and compares it key-by-key against the
locked version. Any difference triggers re-execution.

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

Lambdas created dynamically (via `eval`, `exec`, or an interactive session)
have no source to hash, so their fingerprint changes every run. Use named,
module-level functions for stages and the helpers they call:

```python
# Bad - no stable source
normalize = eval("lambda df: df / df.max()")


# Good - stable fingerprint
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

### Dynamic Name Access

Pivot rejects patterns that bypass static analysis:

- `globals()` / `locals()` — runtime namespace access
- `getattr(obj, variable)` — dynamic attribute lookup
- `importlib.import_module()` under any name (`from importlib import
  import_module as load`) and `__import__()` — dynamic imports

All of these silently introduce dependencies that fingerprinting can't
track. Use direct attribute access and static imports instead.

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
