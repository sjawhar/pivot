# Fingerprinting

Pivot fingerprints your stage functions so it knows when code changes.
Unlike file-level hashing (which triggers on whitespace or comment edits),
Pivot's fingerprinting is Abstract Syntax Tree (AST) based — it captures the
*structure* of your code, ignoring cosmetic changes.

## What Gets Tracked

When you register a stage, Pivot builds a **manifest** — a dict mapping
logical keys to hashes:

| Manifest key | What it represents |
|--------------|--------------------|
| `self:<name>` | The stage function's own AST |
| `func:<name>` | Helper functions called by the stage |
| `class:<name>` | User-defined classes referenced by the stage |
| `mod:<module>.<attr>` | Attributes accessed on imported user modules |
| `const:<name>` | Global constants (primitives, frozen collections) |
| `schema:<name>` | Pydantic model JSON schemas |
| `loader:<class>:<method>` | Reader/Writer method ASTs and config |

### How It Works

For each stage function, Pivot:

1. **Parses the AST** — `inspect.getsource()` → `ast.parse()` →
   normalize (strip docstrings, rename `def` to `func`) → `ast.dump()` →
   `xxhash64`
2. **Walks closure variables** — `inspect.getclosurevars()` finds globals
   and nonlocals the function references
3. **Recurses transitively** — every user-code callable found in step 2
   gets the same treatment, and *its* dependencies are merged into the
   manifest
4. **Inspects type hints** — user-defined classes in annotations
   (including Pydantic models) are fingerprinted
5. **Hashes Pydantic schemas** — `model_json_schema()` captures field
   types, defaults, and validators

The manifest is stored in the per-stage lock file. On the next run, the
worker recomputes the manifest and compares it key-by-key against the
locked version. Any difference triggers re-execution.

### Transitive Tracking

If `train()` calls `normalize()` which calls `clip_outliers()`, all three
functions appear in the manifest. Changing `clip_outliers` triggers a
re-run of `train` even though `train` itself didn't change.

Module-level attributes are tracked too. If your stage does
`from myproject import config` and then accesses `config.THRESHOLD`, the
value of `THRESHOLD` is captured (via `repr()` for primitives, or AST hash
for callables).

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

Lambda functions have no stable source location across Python runs. Pivot
falls back to `id(func)`, which is **non-deterministic** — your stage
will re-run every time. Always use named functions:

```python
# Bad — re-runs every time
pipeline.register(lambda data: data.dropna(), name="clean")

# Good — stable fingerprint
def clean(data):
    return data.dropna()
pipeline.register(clean)
```

### Mutable Captured Variables

Pivot cannot track mutations to mutable objects captured from an enclosing
scope. If your stage closes over a list, dict, or mutable instance, Pivot
raises `StageDefinitionError` at registration time:

```python
config = {"threshold": 0.5}  # Mutable dict

def process(data):
    return data[data["score"] > config["threshold"]]  # Error!
```

Fix: pass the value via [StageParams](parameters.md) or declare it as a
`Dep` input. For truly static config, use a frozen dataclass or
`frozenset`.

To suppress the check (at your own risk), set
`core.unsafe_fingerprinting = true` in your Pivot config or
`PIVOT_UNSAFE_FINGERPRINTING=1` in the environment.

### Dynamic Name Access

Pivot rejects patterns that bypass static analysis:

- `globals()` / `locals()` — runtime namespace access
- `getattr(obj, variable)` — dynamic attribute lookup
- `importlib.import_module()` — dynamic imports

All of these silently introduce dependencies that fingerprinting can't
track. Use direct attribute access and static imports instead.

## @pivot.no_fingerprint()

For stages where AST fingerprinting doesn't work (C extensions, generated
code, complex metaprogramming), opt out with the `@pivot.no_fingerprint()`
decorator:

```python
import pivot

@pivot.no_fingerprint()
def external_model_stage(data):
    ...

@pivot.no_fingerprint(code_deps=["scripts/train.sh", "configs/model.yaml"])
def shell_stage(data):
    ...
```

With `@pivot.no_fingerprint()`, Pivot falls back to **file-level hashing** —
it hashes the entire source file containing the function. The optional
`code_deps` argument lets you list additional files that should be
considered part of the stage's code.

**Use sparingly.** File-level hashing is less precise: any change anywhere
in the file triggers a re-run.

## Performance

Fingerprinting runs once during pipeline discovery (single-threaded, in
the coordinator process). Results are cached at two levels:

1. **In-memory** — `WeakKeyDictionary` cache avoids re-parsing the same
   function within a single process
2. **Persistent** — AST hashes and full manifests are cached in StateDB,
   keyed by `(file_path, mtime, size, inode)`. If the source file hasn't
   changed, the cached hash is reused without parsing

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
