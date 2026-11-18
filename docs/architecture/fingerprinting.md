# Code Fingerprinting

Pivot automatically detects when your Python code changes using a combination of AST hashing and `getclosurevars()` analysis.

## How It Works

```python
def helper(x):
    return x * 2  # Change this...

def process():
    return helper(load_data())  # ...and Pivot knows to re-run!
```

Pivot's fingerprinting system:

1. **Parses your function** into an Abstract Syntax Tree (AST)
2. **Discovers dependencies** using `inspect.getclosurevars()`
3. **Recursively fingerprints** all transitive dependencies
4. **Hashes the result** for fast comparison

## What Gets Tracked

| Pattern | Tracked? | How |
|---------|----------|-----|
| Same-module helpers | Yes | `getclosurevars()` discovers reference |
| Module imports | Yes | AST scanning for attribute access |
| Imports in a function body | Yes, or error | Pivot never executes a body import while defining a pipeline. A user module discovery loaded is walked like a module-level import, and the manifest covers it, its packages and a submodule a `from` import names (`module:<name>` keys). A body import of a user module discovery did not load is refused with `StageDefinitionError` naming the module: import it at module level (or in `pipeline.py`), or move the value into `StageParams`. See [Imports a Stage Makes While It Runs](#imports-a-stage-makes-while-it-runs) for the worker's check |
| Class definitions | Yes | Full class AST hashed. A class with no source of its own (`collections.namedtuple`, `dataclasses.make_dataclass`, a functional `enum.Enum`, `type(...)`, `pydantic.create_model`; one without `__firstlineno__`) is hashed by its module, name, bases and namespace: fields, members, and the code of the functions it holds (by source when they have one, by code object when `exec` made them, by name when a library defines them), so its hash is the same in every process (`fingerprint._sourceless_class_hash`). Its hash is computed in each process, never cached in `state.lmdb` by its module's file, because what defines it may come from another file (a factory's arguments, a default read from a config module); and reached by a name it is keyed `class:<module>.<name>@<hash>`, because one factory makes many classes with one module and one name (inside a value, `@<identity>`: see below). A pydantic model is hashed by what its author declared: each field's `FieldInfo` (annotation, default, default factory, alias, constraints), its config, the code and settings of its validators, serializers and computed fields, its private attributes and class variables; never by the core schema, validator or serializer pydantic builds, whose reprs hold addresses. No text in the form is rewritten, so a string keeps its content (an address-like `"0xabcdef"` from a settings module included). An `object()` sentinel or a `__slots__` object in an entry is written by its type and its slots, with no address, so a value another module built reaches the hash through its state. The hash never refuses: a value whose state no form reads (a `random.Random`, whose state is in C) is written by its type. The class's walk reads each value of its namespace (`ns:` steps) and refuses each stage that uses the class, naming the class and the entry |
| Callable members of user classes | Yes | User-defined methods, classmethods, staticmethods, property accessors, cached properties, `functools.partial`/`partialmethod` members, nested user classes, and dataclass default factories are walked like stage helpers. Callable instances and other user-defined descriptors are refused because their state is not reliably fingerprintable. Generated and library members are skipped. A dataclass's own class-level field defaults (a `ClassVar`'s too) are hashed by their form, with no memory address and no repr of user code run: a string, `bytes` or a path by its content (identity-looking text such as `at 0x...`, `Py(0x...)` or `<object at 0x...>` included), an exception by its type, `args`, `__dict__` and value descriptors (`OSError.filename` included), an `object()` sentinel or a `__slots__` object by its type and its slots, an instance of a user class by its attributes and slots, so it is the same in every process and changes with what it holds; a base's defaults are the base's, walked as its own class. A user function or method held in a default or in an instance's state (a callback a `__slots__` object holds) gets a stage helper's walk, so the module values it reads count and refuse as they would; a classmethod is walked as its underlying function. A callback is a node of the stage's walk, so one that reads its enclosing dataclass meets that class's node and stops there. One `__wrapped__` follower handles every callable route, including native caches held in partial arguments, defaults or state. It stops at a by-name mark and refuses identity loops and chains past `sys.getrecursionlimit()` above that boundary, even in unsafe mode: break the loop, shorten the chain or move the value into `StageParams`. A terminating chain inside the limit keeps every Python function's own code and reads; C cache wrappers include the settings exposed by the stdlib `cache_parameters` closure, read without calling it (see the stacked-cache limit below); library helpers are recorded by name. A default whose state no form reads (a `random.Random` or a numpy `Generator`, whose state is in C; a `threading.Lock`) refuses the stage, naming the field: for a field, build the value with `dataclasses.field(default_factory=...)`; a `ClassVar` takes no factory, so create its value in the method that uses it; or move the value into the stage's `StageParams`. A pivot loader in a default reports such a value in its config through its form |
| StageParams subclasses | Yes | Type hint analysis + full class AST + Pydantic field defaults + `default_factory` functions |
| Custom loaders | Yes | Loader method ASTs hashed. A user loader's `load()`, `save()` and `empty()`, the user members their code reaches through `self` in a static form, transitively, and functions and classes in its config fields, are walked like stage helpers: module values they read are tracked or refused. The static forms are `self.<name>`, `getattr(self, "<name>")` (and `hasattr`, `setattr`, `delattr`), `type(self).<name>`, `self.__class__.<name>`, `super().<name>`, and the dunders a builtin call on `self` runs (`str(self)` and `f"{self}"` run `__str__`); the body a `functools.wraps` helper wraps is read too, along the chain `wrapper_chain` reads. A class constant read so (`self.SUFFIX`) is recorded or refused as a global is. When reached code uses `self` in any other way (`self` given to a function, `vars(self)`, an operator, `type(self)(...)`), every user member of the class is walked; `getattr(self, name)` with a computed name is refused, as any computed `getattr` is. A method whose `__wrapped__` chain loops is refused by name. Otherwise a member no such code reads is not walked, so it adds no entry and no refusal; a dataclass field is config, not a member. Methods inherited from `pivot.loaders` are hashed only. A loader that its own walk reaches again (a dataclass default its `load` reads holds it) is not walked again |
| Loaders held in bound values | Yes (as a top-level loader's class) | A pivot loader (an instance of `loaders.Reader` or `loaders.Writer`) that a loader's config, a stage-level `functools.partial`, the field default of a dataclass the walk reaches, or a `StageParams` default holds, at any depth of the value's form (`Outer(Inner())`, a second field, a list, tuple, set or dict value, a partial's arguments or keywords, a dataclass or pydantic model in a field), is recorded in the holder's form (its class's key and its config), and its class gets the entries a top-level loader of that class gets: its `load`, `save` and `empty` hashes and the walk of the code they reach (`loader:<class key>:load`). The class is keyed as a class inside a value is (below), so two classes one factory made keep apart. A loader class is walked once for a stage's loaders, however many loaders of it the values hold. A held loader whose class defines `__call__` is refused: a call runs code the loader walk does not read. A cache hit reads the holder's form again, so a new config, class or position is a change |
| User class instances held in bound values | Yes (as the class a stage names) | An instance of a user class that is not a pivot loader, held where a held loader is (above), is recorded by its form: its class's key, its instance `__dict__`, its slots and the value of a built-in base (`["inst", key, {...}]`), read through C descriptors, so no code of the instance, its class or its metaclass runs; a class whose metaclass defines `__getattribute__` or `__getattr__` in user code is refused, naming the class and the metaclass, since reading anything of it would run that code. Its class gets the walk a named class gets: its AST hash, each member of its own namespace walked as a stage helper, each user base an edge walked for its own members (a `TypedDict`'s or `NamedTuple`'s bases as its class statement names them, `__orig_bases__`), its own dataclass and pydantic field defaults, and the class attributes its members read through `self`. A class is walked once per stage, and a class the walk reaches again (a cycle) is not walked again. Values are formed on an explicit stack, and class identities computed inside one another nest no deeper than a fixed limit, so neither a value nested deeper than the recursion limit nor a class on a chain of 1,100 bases fails the walk. A class reached inside a value keeps its name as its key when it has source and its module's name holds it, else it is `<name>@<identity>` (`_class_key`), the start of a digest of its code and its namespace's forms, so two classes one factory made keep apart wherever they are held. The walk's refusals are a named class's (a member that reads a non-scalar global fails the stage, though the stage may never call it), and each is named once with up to three of the ways the stage reaches the value (`Reached through: ...`). A value whose layout keeps state no descriptor shows (`random.Random`, a lock, a numpy `Generator`) is refused. A manifest cache hit reads each value's form again and replays the reads its classes' walks recorded, under the path through the value (`pa:0`, `at:NAME`, `cls`), so it walks no class again: a thousand distinct classes in one partial (ten thousand instances) cost a warm `pivot status` about what a stage that names them costs, and add three thousand lock entries. An instance held as a global, or in a non-partial global value, is refused |
| Unknown library objects in bound values | No — error | A value in a partial argument or keyword, a loader config field, a field or `StageParams` default, or a nested container there must have a built-in value form or expose all of its state through attributes and slots that C descriptors read. Pivot refuses one that keeps unreadable C state (`random.Random`, a lock, a numpy `Generator`, `lazy_object_proxy.Proxy`), one whose class hides its instance dict behind a `__dict__` property, and a library proxy, an instance of a library class that defines `__getattr__` or `__getattribute__` (`celery.local.Proxy`, a `unittest.mock.MagicMock`), since what it forwards to is built by code no form reads (pydantic's `BaseModel.__getattr__`, which answers from the fields the model's slots hold, does not count). Pivot runs none of the value's code, so it never calls `model_dump()`. Move the value into `StageParams` or read it from a pipeline input. A user class that defines `__getattr__` is formed by its state and walked as a class |
| Native values with Python subclasses | Yes, or error | Exact known type expressions (`types.GenericAlias`, `types.UnionType`, and the stdlib alias type returned by `collections.abc.Callable[[int], str]`) use their origin and arguments; a user `GenericAlias` subclass is refused because those native fields omit its Python behavior and state. A numpy scalar subclass records its native bytes, attributes and readable slots, with replay through the same slot descriptors. An `OrderedDict` preserves logical order only for exact scalar keys (`bool`, `int`, `float`, `complex`, `str`, `bytes`, `None`); its native iterator hashes keys, so other key types refuse before iteration, including through a subclass or mapping proxy. |
| Module-level immutable values | Yes (value) | Scalars, exact tuples and frozensets of scalars, and compiled regexes are recorded by value |
| Other module-level values | No — error | Use `StageParams` instead. Lists, dicts, sets, nested collections, enum members, and class instances are refused. This includes a method bound to an instance (`apply = Scaler(3).apply`, `normal = np.random.default_rng(SEED).normal`): the instance's state is not hashed. A method bound to a module, a class, a scalar or a `logging.Logger` is allowed |
| Closure variables | Yes (value) | `getclosurevars()` nonlocals checked like module-level values, keyed under the binding the closure was reached through, so two closures from one factory (`h2 = make(2)`, `h5 = make(5)`) keep their own values |
| Default arguments | Yes (value) | `__defaults__`/`__kwdefaults__` checked like module-level values; a literal default is covered by the AST hash |
| `functools.partial` | Yes | Bound args + underlying function; functions and classes in bound args walked like stage helpers |
| Library functions and classes | Yes (name) | A global, closure cell, default or partial holding `np.mean` records `ext:<binding>` → `numpy.mean`, so rebinding it to `np.median` reruns the stage; the walk does not hash the callable's code (when it comes from a library installed from git, a local path or a local archive, the library's installed files are hashed: see the `dist:` row below) |
| Library modules | Yes (name, not in the lock) | A library module read as a global (`np` in `np.mean(x)`) or imported in a function body, and the module that defines each library callable, is recorded by the walk as `byname:<module>`, which the worker-import check and the manifest cache read. So is the module of the library code a stage reaches where no other entry names it (`fingerprint._reached_library_module`): a library base class of a user class, a library class in a type hint or a class-body annotation, a library class, instance, enum member or exception inside a value (a partial, a default, a loader's config), each library class on a pivot loader class's MRO, a stage function that is library code, and a library callable whose `__module__` is not a string (recorded as the library module that binds it under its own name). At those places a standard-library module (`builtins` included) and a module in one of pivot's own packages are not recorded. The worker-import check covers these entries too, so a project file that shadows such a module is refused as one that shadows a module read as a global is. The stage fingerprint a lock records leaves these entries out (`fingerprint.without_by_name`), so locks written before they existed stay current. Its attributes and code are not read |
| Libraries installed from git, a local path or a local archive | Yes (installed files, `dist:<name>`) | Each distribution that owns the module of one of the stage's `byname:` entries (a loader's included) and whose `direct_url.json` (PEP 610) has `vcs_info`, `dir_info` without `editable: true`, or `archive_info` with a `file:` URL (a local wheel or sdist), and each such distribution a reached one requires, directly or through others, gives the stage's manifest a `dist:<name>` key: a hash of the distribution's own installed files, read from disk. The lock records it, so a change to those files reruns the stage, and a reinstall of the same files from another commit, path, installer or virtual environment reruns nothing. See [Libraries Installed from Git or a Local Path](#libraries-installed-from-git-or-a-local-path) |
| Libraries from a package index or a remote archive; editable installs; pivot | No key | A distribution with no `direct_url.json` (a package-index or `--find-links` install), with `archive_info` and a URL whose scheme is not `file`, or with `dir_info` and `editable: true` gets no `dist:` key, so a version bump of a package-index library reruns what it reran before this key existed: only a change to an entry the walk records for it (an `ext:` definition name, the `self:` hash of a stage function it defines, a loader's `load`, `save` and `empty`). Nor does pivot's own distribution (one with a RECORD file in one of pivot's package directories) get a key, and its requirements are not followed. An editable install whose source lies outside the interpreter's prefix and outside every `site-packages` or `dist-packages` directory is user code, walked like the project's own |
| Secrets outside a stage's params | Yes (digest) | A `SecretStr`, `SecretBytes` or `Secret[T]` in a `StageParams` default on a helper, a loader config field or a partial argument hashes as its keyed `hmac:` digest, the one the lock file records (`parameters.secret_digest`), so changing it reruns the stage without its value in any manifest |
| Class attributes a stage reads | Yes (value) | `Cls.NAME` read directly on a user class, through a user module (`settings.Config.SCALE`), or from a user class member through `self.NAME`, `cls.NAME`, `type(self).NAME`, `self.__class__.NAME`, or literal `getattr` follows the module-level rule. The value is the one the class body holds, read from the namespace of the nearest class on the real MRO that holds it, so no descriptor `__get__` and no metaclass code runs. Scalars, exact tuples and frozensets of scalars, and compiled regexes are recorded; mutable or nested values (a dataclass's or a `StageParams` class's `ClassVar` list or dict too) and instances, a descriptor object included, are refused; a plain function is walked; and a member of a user Enum is recorded by name and value. A pydantic or `StageParams` class read directly is covered by its schema hash, field defaults included; its `model_config`, read through `self` by a method, is recorded by its form (`default:<class>.model_config`), since the schema leaves parts of it out (`title`). Attributes a library base or the language defines, and descriptors (property, classmethod, staticmethod), are left to the class Abstract Syntax Tree (AST) hash. |
| Other class-level and module reads | No | Not tracked: `Outer.Inner.X`, a `ClassVar` on a `StageParams` class read as `Params.NAME` (through `self` it follows the row above), a module read through a local alias (`s = settings; s.SCALE`) or through the object a dynamic import returns (`load("settings").SCALE`), a literal `getattr(settings, "SCALE")`, and a module held in a class attribute (`Holder.lib.SCALE`). Put such configuration in the stage's `StageParams`. |

**Note:** The class AST and callable-member walk cover a class's code and the module values callable members read. Non-callable class attributes are tracked only in the direct and static receiver forms above, plus dataclass and Pydantic field defaults. A value read as `module.NAME` follows the rule it follows as a global: a library function is an `ext:` entry, a `functools.partial` is walked, and a logger is ignored.

Receiver-attribute analysis reads each Python method body reached through a wrapper chain,
including cached methods, partialmethod targets and the function beneath a forwarding decorator.
It stops at a `@fingerprint_by_name` mark.

Library helpers and their wrappers are recorded by name without inspecting private closure
state. A library-defined stage root keeps its globals, closure values and defaults; a native
cache on that root follows the terminal Python body for those reads. User wrapper code is
walked even when `functools.wraps` copied a library target's module metadata.

For stacked native caches, only the settings visible on the outermost wrapper are recorded.
CPython's `functools.update_wrapper` copies the inner cache's `cache_parameters` onto the
outer cache, so an edit to the outer `typed` or `maxsize` is not seen. This shape remains
accepted; put cache configuration in the stage's `StageParams` when it must trigger a rerun.
Native cache settings are recorded only for caches over user code. A cache over library code
(`functools.lru_cache(...)(pow)`) is recorded by name, so changes to its `typed` or `maxsize`
do not trigger a rerun.

For Pydantic models, both static defaults and `default_factory` functions are tracked:

- **Static defaults** (e.g., `threshold: float = 0.5`): The value is hashed by its form (`_form`), the same value form a partial's arguments take, with the model's schema
- **Factory functions** (e.g., `items: list[str] = Field(default_factory=list)`): The factory function's AST is hashed, so changes to the factory logic trigger re-runs

Schema hashes and default forms are reused only within `fingerprint_pass()`, keyed by object
identity. The pass owns their model references; nothing in a process-lifetime schema cache
keeps a model or its validator alive. The next pass reads a rebuilt model and its live defaults
again.

## Normalization

AST normalization ensures stable hashing:

- **Function names** → Normalized to `"func"`
- **Docstrings** → Removed
- **Whitespace** → Ignored (AST doesn't capture it)
- **Comments** → Ignored (AST doesn't capture them)

This means:

```python
# These two functions have DIFFERENT fingerprints
def process(x): return x + 1
def process(x): return x + 2

# These two have THE SAME fingerprint
def process(x):
    return x + 1

def process(x): return x + 1  # Same logic
```

## Transitive Dependencies

Pivot tracks the entire dependency chain:

```python
def level_3(): return 1
def level_2(): return level_3() + 1
def level_1(): return level_2() + 1

def main():
    return level_1()
```

If `level_3()` changes, Pivot detects it and re-runs `main`.

## Import Patterns

Both import styles are supported:

```python
# Direct import
from helpers import process_data
# Tracked via getclosurevars()

# Module import (Google style)
import helpers
result = helpers.process_data(x)
# Tracked via AST scanning for module.attr patterns
```

## Known Limitations

Some patterns cannot be tracked automatically:

### Imports by a Runtime Name

```python
def process():
    return importlib.import_module(name).run()  # Module chosen at run time
```

An import whose module is named at run time cannot be resolved statically. A call
`x.import_module(...)` is refused with `StageDefinitionError`, as are `getattr()` with a
computed name and `globals()`/`locals()` (`fingerprint._check_dynamic_name_access`). For a
`functools.wraps` wrapper these refusals read the source `inspect.getsource` returns, which is
the wrapped function's, so a computed `getattr` in the wrapper's own body is not refused and one
in the wrapped function is, even when the wrapper reaches it only through `__wrapped__`. Other
spellings (`__import__`, an alias of `importlib.import_module`, a `functools.partial` of it,
`vars(module)[name]`, `operator.attrgetter`, `exec`, and `importlib.util` loader APIs) are not
refused at definition. A user module one of them loads and registers while the stage runs is
caught by the worker's check below, and a library module costs nothing. `runpy.run_path` and
`runpy.run_module` of a top-level module run the file without registering it, so the check does
not see them (see its limits); an attribute read on a module already loaded is not seen either.

**Workaround:** Import the module statically.

### Imports a Stage Makes While It Runs

The worker initializer (`executor.core._init_worker`) wraps `builtins.__import__`,
`importlib.import_module`, and every thread start function `_thread` exposes: `start_new_thread`,
`start_new` and `start_joinable_thread`, the last also at `threading._start_joinable_thread`, the
binding `threading.Thread.start` calls (`worker.install_import_hooks`). Importing
`pivot.executor.worker` alone changes none of them, and a stage executed in a process without
that initializer gets the hooks for its call only. For a relative `builtins.__import__`, the hook
uses CPython 3.13's `_calc___package__` order (`__package__`, `__spec__.parent`, then
`__name__`/`__path__`) before it records the absolute module. Pivot-owned threads start through
`worker._start_pivot_thread`, which bypasses coverage: a pipe reader, parent watcher or
stop-grace timer is not user work and cannot excuse a later user module registration. While a
stage runs, the import hooks record each user module (and its user packages) the stage requests,
including one already in `sys.modules`, and the worker compares the `sys.modules` keys before and
after the stage, counting a newly resident module only when it has a `__file__` that
`is_user_code` accepts. The modules a manifest covers come from one function,
`fingerprint.covered_user_modules`: `module:<name>` keys, and the `dep:<parameter>:module:<name>`
and `out:<output>:module:<name>` keys a user loader's walk adds. The prefix of a loader's key ends
where the dependency's name (a parameter, which holds no `:`) or one of the stage's output keys
(`<stage>` or `<stage>:<field>`) ends, never at a word in the key, so an output field named
`module` (whose tag is `out:<stage>:module:tag`) covers no module `tag`
(`fingerprint.manifest_entries`).

What a covered module's first import starts belongs to that import, whichever thread runs it and
whenever it runs during the stage, as a module-level import of the module would have made it
before the stage ran (the covered thread rule, `_StageImports`):

- **Covered threads.** A thread is covered while it is inside a covered module's first import. A
  thread started while its starting thread is covered is covered for the rest of the stage, and so
  is a thread such a thread starts, through any of those start functions: `threading.Thread` and
  its subclasses, `threading.Timer`, `ThreadPoolExecutor` workers and raw `_thread` threads count.
  A thread native code starts outside `_thread` is not covered (see the limits below).
  A thread covered for one stage is not covered for a later stage in the same worker.
- **Hooked imports** (`__import__`, `importlib.import_module`). One made on a covered thread is
  excused, whether or not the module was already resident, so a warm worker and a cold one give
  the same answer. One made on any other thread is counted, whether or not the module was already
  resident: the stage's own thread outside a covered import, and threads the stage started outside
  one.
- **Unhooked registrations** (`importlib.util` and `exec_module`, `EntryPoint.load()`,
  `importlib.__import__`, a manual `sys.modules` insertion), seen only by the `sys.modules`
  comparison. A module that becomes newly resident while any thread is inside a covered import,
  or any covered thread of the stage is running, is excused; one that becomes resident at any
  other time is counted, even when a covered import later imports it. The comparison cannot tell
  which thread registered a key, so this part is by time.

A covered module is never imported ahead of the stage: discovery pickles `pipeline.py` and other
modules under the project root by value, and importing one of them in the worker would run it
again there.

A recorded module outside the covered set makes the result not reusable: the stage succeeds and
logs one warning naming the modules; its lock carries an `uncovered-import:<modules>` key no
fingerprint matches, so the next `repro` reruns it and `status --explain` and
`repro --dry-run --explain` report "Previous run imported modules its fingerprint does not cover";
and no run-cache entry is filed. A marked lock is never satisfied from the run cache
(`skip.check_stage` returns `run_cache_allowed=False` for it), and the flagged run deletes any
run-cache entry at its own input hash, so an entry an earlier run filed (0.5.2, or `pivot commit`)
cannot restore an output the check never saw. `pivot commit` carries that key into the lock it
writes when the static fingerprint is otherwise unchanged, files no run-cache entry for it, and
deletes any entry at its input hash (`skip.carry_rerun_marks`). `pivot verify` reports a
marked stage as failed, with the uncovered-import reason, until it reruns without an uncovered
import. Under `--no-commit` the warning adds that `pivot commit` keeps the mark when it records
these outputs. A user class reached only through a loaded input (an object in a
`loaders.Pickle()` input whose class the stage never names) counts as a module the stage imported,
because unpickling imports its module while the stage runs; naming the class in the parameter's
annotation (`Annotated[Box, loaders.Pickle()]`) covers it.

The warning's fix is a static import with the module read by name in the stage's own code
(`pkg.mod.NAME`, or `pkg.mod` passed itself), which the walk then covers, or `StageParams`. An import
in `pipeline.py` alone does not add the module to a stage that never reads it, and a body import
that only silences the warning leaves values read through the object a dynamic import returned
untracked, like the local-alias read in [What Gets Tracked](#what-gets-tracked). A
`@no_fingerprint` stage is not checked (`WorkerStageInfo.no_fingerprint`): its `code_deps`
declaration is its contract.

The check does not see these paths. Most behave as they would without it; three rerun a stage that
would otherwise be reused: a thread that existed before a covered import, a thread left from an
earlier stage (a covered thread of that stage included) that imports while a later stage runs, and
a thread native code starts outside `_thread`. Each has a remedy:

| Path | Remedy |
|------|--------|
| Reading a module already in `sys.modules` without an import; `importlib.reload` | Import the module where the stage reads it |
| A warm `EntryPoint.load()` or `importlib.__import__` of a user module another stage already made resident | It does not create a new `sys.modules` key, and the loader may hold an importer before the hook: it is not fingerprinted, as on main. Import the module statically |
| Private import machinery; a loader that replaces a key already in `sys.modules` or never registers one, such as `runpy.run_path("helpers/lib.py")` and `runpy.run_module("toplib")` for a top-level module | Import statically, or move the value into `StageParams` |
| An import in a child process | Declare the code the child runs as an input, or use `@no_fingerprint(code_deps=[...])` |
| An import from a thread that outlives its stage | Join the thread before the stage returns. Its import is charged to whichever stage is running, which then warns and reruns where it would otherwise be reused, or to none. A covered thread still running in a later stage is not covered for that stage |
| A covered module hands work at import time to a thread that existed before its import, such as a pool an earlier import created | That thread is not a covered thread, so its imports are counted: the stage warns and reruns. When an earlier stage in the same worker started the thread, whether the stage warns depends on which stage ran first there: where the covered import itself starts the thread, the thread is covered and the stage caches. Start the thread or pool inside the module, or import the module statically |
| A thread native code starts outside `_thread`, such as one a C extension starts or `pthread_create` called through `ctypes` | No start hook sees it, so it is not a covered thread: its imports are counted, and the stage warns and reruns where it would otherwise be reused. Import the module statically |
| A covered thread imports a user module unrelated to the covered module | It is excused and not fingerprinted, as on main: an edit to it is seen only through values the stage reads from the covered module, or when the covered module changes. Import user modules statically |
| While any thread is inside a covered import, or any covered thread of the stage runs, another thread registers a user module without the hooks (`importlib.util` and `exec_module`, `EntryPoint.load()`, `importlib.__import__`) | It is excused with the covered import's own, so an edit to it is seen only through values the stage reads from the covered module, or when the covered module changes. A hooked import on that thread is counted. Import user modules statically, or before starting threads |
| A user module that a covered module loads at import time through `importlib.util` and `exec_module`, `EntryPoint.load()` or `importlib.__import__` | It is excused, as on main, and not fingerprinted itself: an edit to it is seen only through values the stage reads from the covered module, or when the covered module changes. Import it statically |
| A package resource read through `importlib.resources` (`files(helpers).joinpath("table.txt")`) | It is data, not code: declare it with `pipeline.input` |
| A module created without a source file (CPython's `pyexpat.errors`, Cython's `_cython_3_3_0`) | It has no source to fingerprint; a change to it is a change to the library or interpreter that created it |
| `pivot repro --no-commit`, then `pivot commit` of outputs edited by hand since that run | `pivot commit` records them against the static fingerprint, the "trust me" path. Outputs the run wrote, unchanged since, are recorded with the manifest the run kept in StateDB (`nocommit:<stage>`), its mark included, so the stage still reruns |
| A result cached by an earlier Pivot version | If its static fingerprint is unchanged, it may be reused, from its lock, which Pivot reads as it is, or from its run-cache entry, which Pivot finds under the key that version filed it under (unmarked, with each directory the stage reads whole hashed from its listing the way that version hashed it: `worker.older_pivot_input_hash`, `listing.legacy_digest`). `_CACHE_SCHEMA_VERSION` keys only the fingerprint caches in state.lmdb, not the lock or run cache. A stage whose static walk reaches a user class method or a user module gains entries in its manifest, so it reruns once as Code changed after the upgrade. That accepted one-time rerun writes the current manifest; later runs reuse it until code or inputs change. Run `pivot repro --force <stage>` once to re-check a stage you know imports dynamically. Upgrading from 0.10.4, a stage that reaches a library installed from git, a local path or a local archive gains that library's `dist:` key, so it reruns once as Code changed, and a stage that reaches none keeps its lock. The same upgrade moves `_CACHE_SCHEMA_VERSION` from 43 to 44, so each stage's manifest is computed once more instead of read from the manifest cache; that reruns nothing, because the `byname:` entries the new version records never reach a lock. Upgrading from 0.5.6 or earlier, every stage with outputs reruns once as Code changed, because each output's tag (`out:<key>:tag`) joined the manifest; that run is checked, so it marks a stage that imports dynamically |

### Code Edited After Discovery

A stage's fingerprint describes the code its worker runs, and that code is the code discovery
loaded, not the files as they are when the stage starts. Discovery pickles `pipeline.py` and the
other modules under the project root by value, so a worker runs the functions the coordinator
compiled when it imported them. Fingerprinting reads source through `inspect.getsource`, which
reads `linecache`, and `linecache` reads a file again when its size or mtime changes. Outside
watch mode a stage is fingerprinted when it is ready to run, so a file edited after discovery (by
an upstream stage, or in an editor while the run continues) would be hashed as edited while the
worker ran the loaded version, and the next run would report the old output as cached.

So the record is about the module, not the loader. `discovery._ProjectSourceLoader` records the
bytes each user module was compiled from (`loaded_source.record`). Every other user-code module
(`fingerprint.is_user_code`) that an import finds passes the recorder, which wraps its loader
(`discovery._RecordingLoader`): an editable install's meta-path finder, a zip on `sys.path`,
`importlib.util.spec_from_file_location`, and the loader any finder on `sys.meta_path` returns. A
module the recorder loaded without recording it is recorded at the end of discovery
(`loaded_source.record_user_modules`) and in a worker at each import its hooks see
(`loaded_source.record_modules`, also run for the modules unpickling a stage imports) and when a
stage returns.

The recorder trusts a loader's source only where Pivot can show the code that ran is the code it
read (`discovery._RecordingLoader.exec_module`), and records every other loader's module as
unreadable. A list of loaders to distrust leaks: each loader shape it does not name (a proxy
around CPython's loader, a loader that runs CPython's inside it, CPython's own `SourceFileLoader`
definition on its private bases, a `FileLoader` that caches its own bytecode, a zip holding a
`.pyc` beside its `.py`) serves a stale result as cached. So the rule names what is trusted, and
a new loader shape is unreadable by default:

- CPython's exact `SourceFileLoader`, which accepts mtime-checked bytecode for an edit that keeps
  the size and whole-second mtime, is replaced with `_ProjectSourceLoader`, which records the bytes
  it compiled or validated checked-hash bytecode against.
- A zip importer (`zipimport.zipimporter`) whose spec names the `.py` member as its origin: Pivot
  compiles the member's bytes it read and runs them itself, as zipimport compiles a source member
  (line endings normalized), and records those bytes. The spec's origin is the word of whoever
  built the spec, not the member zipimport runs: zipimport's `exec_module` searches the archive
  again, tries a `.pyc` member first (CPython 3.13 `zipimport.py:290`) and runs it when its header
  matches the `.py` member's zip time, to within 1 s, and size (`:658-660`, `:690-695`), or when it
  is hash-based and either unchecked or matching the source's hash. A finder that builds the spec
  itself (`spec_from_file_location(name, "<archive>/m.py", loader=zipimporter(archive))`) gets a
  `.py` origin while zipimport would run the `.pyc`; running the bytes Pivot read closes that, and
  the gap between Pivot's read and zipimport's. A spec zipimport's own finder built with a `.pyc`
  origin (it found such a member) is unreadable.
- A module Pivot's own loader recorded during this execution, inside another loader that wraps it:
  a finder modelled on dd-trace-py's ModuleWatchdog finds the module through the rest of
  `sys.meta_path`, Pivot's recorder among it, and wraps the loader it gets. The record that inner
  recorder made stands: readable where Pivot's loader compiled the source, and unreadable, with
  the reason that names the real loader, where it judged one Pivot does not trust. A record from
  before the execution (an earlier import, before `importlib.reload`) does not count, nor the one
  a worker's import hook makes while the module's body imports a submodule. Pivot trusts that such
  a wrapper runs the code its own loader compiled for the module, and nothing else: a wrapper that
  asks Pivot's loader for the code (which records it) and then runs other code is not detected, a
  stated limit no known tool reaches.

An unreadable module's record carries a reason that names its loader, and the first stage in each
process that uses it logs that reason with the fix: import the module without that loader, or
accept that the stage reruns on every run. Once a run Pivot records has used the module, the
stage's lock carries the `unreadable-source:` mark, so every later run reruns it and none reports
it as cached; a lock written before that (by an earlier Pivot, or before the loader was installed)
can still be reported cached until a run uses the module. Unreadable does not make the output
current: each rerun runs what that loader runs, which can still be the stale bytecode. A `pivot
commit` of outputs no run recorded (the "trust me" commit) carries the mark the stage's previous
lock holds; with no marked previous lock it writes a lock without the mark, and the next run
reports the stage as cached. A loader with only PEP 302's `load_module` and no `exec_module`,
which CPython 3.13 still imports with an `ImportWarning` (its import system calls `load_module`
"slated for removal in Python 3.15"), is left unwrapped, so the import works as without Pivot; no
recorder sees its module compile, and it is judged as below, where its loader makes it
unreadable. A loader a module calls itself (`SourceFileLoader(...).load_module()`, or
`spec_from_loader` and `exec_module`) passes no import hook, so its module is judged as one no
recorder saw compile.

A user module that no recorder saw compile was imported before discovery installed the recorder,
by `sitecustomize` or a `.pth` import line (in the coordinator, and in each worker before its
initializer runs), by the program that runs Pivot or by a finder ahead of the recorder, or it was
loaded by a loader a module called itself. `loaded_source.record_unseen_modules` runs at the end
of discovery, in `core._init_worker`, and in a worker after each stage
(`worker._untrusted_modules`), so a worker judges every user module it loaded, whatever loaded
it, as the coordinator does, and marks a stage that used one whose record there is unreadable or
differs from the coordinator's. Each such module is judged by `loaded_source._verified_source`,
which records the module's file when those bytes are shown to be the bytes it was compiled from:

- its loader is CPython's own `SourceFileLoader` (a subclass may run transformed text);
- the file's change time (`st_ctime`) is at least 2 s older than the process start. The start is
  read on Linux from `/proc/self/stat` against the boot clock, added to the boot's wall-clock
  time, the smaller of its value when Pivot was imported and now, so a step of the wall clock
  after the import cannot move the start later (psutil elsewhere). The three errors that can make
  an edit look older than it is share those 2 s: a file system that keeps whole seconds (ext4
  with 128-byte inodes stamps a change up to 1 s before it happened), the kernel's coarse
  timestamps, and a forward step of the wall clock between an edit and Pivot's import, which the
  margin covers up to 2 s where change times are exact and to about 1 s where they are whole
  seconds. A file system that keeps change times in 2-second steps (FAT, Linux's `vfat`) spends
  the whole margin: on a FAT32 loop mount every change time is an even second, so an edit reads
  as up to just under 2 s early (1.7 s in eight edits measured). There the margin covers no clock
  step and no kernel tick, and an edit made in the first milliseconds of a process, before the
  interpreter's own startup has reached any import, can read as older than the start. Linux's FAT
  driver also keeps no change time on disk: it sets an inode's change time to its modification
  time when it reads the inode (`fs/fat/inode.c`, `fat_fill_inode`), so once the inode is read
  again, an edit whose modification time a tool put back has its change time put back too, and
  only the bytecode check below can catch it. Both are stated limits for a project on such a file
  system. Every import in the process then read these bytes;
- the bytecode that loader may have run instead is the code of these bytes. A cached file that
  has not changed since the start is the one the import saw: either CPython's check rejects it
  for this source, or its code (`marshal`) equals the source compiled now, which catches the
  edit that keeps the size within the second its predecessor was cached. A cached file written
  since the start must be mtime-checked bytecode of these bytes: CPython's import writes that
  only after rejecting the file it found for this same source. With no cached file, its directory
  must not have changed since the start.

Otherwise the module is recorded as unreadable, with the reason in its record
(`path\0\0reason`), and `check_stage` logs the reason once per process when a stage uses it, in
addition to the stage's own warning. The cases that fall there: a file edited after
the process started or less than 2 s before it (an edit during startup, a module a long-lived
process loaded and then edited, or an edit just before `pivot` ran, which costs that one run), a
module loaded by any other loader, stale or replaced bytecode, a cached file that cannot be read,
a platform that does not report the process start, and a process forked without exec after it
imported Pivot (`os.register_at_fork`), whose modules its parent loaded before it started. Each
costs the same as a sourceless module: the stages that use it rerun on every run in that process
and file no run-cache entry.
Two cases are not detected. A process forked without exec from a parent that imported the module
before it imported Pivot: start Pivot in a process of its own, as the `pivot` command and its
workers are. A forward step of the system clock between an edit and the start of a pivot process
can make that edit look older than the start once it passes what the 2 s margin leaves after the
file system's timestamp granularity (about 1 s where change times are whole seconds, 2 s where
they are exact, none where they are in 2-second steps); so can a file system that stamps change
times with another machine's clock (a network file system's server) that far behind this one.
These records belong to the import session: `loaded_source.prune` drops them when it ends, so
the next discovery judges the modules again.

Discovery's path hook and the module recorder are installed only while a discovery imports;
`discovery._remove_import_hooks` gives CPython's machinery back when it returns, fails or is
interrupted. The session itself (its modules, its `sys.path` entry and its records) lasts until
the next discovery ends it. A worker installs its own loader and recorder for its life
(`discovery.install_user_code_loader`).

Fingerprinting reads the recorded bytes, not the file: `loaded_source.getsource` puts their lines
into `linecache` as an entry without an mtime, which `linecache.checkcache` leaves alone, before
it calls `inspect.getsource`; `fingerprint._source_digest`, which keys the `fp:` and `sm:` caches,
returns their digest; and a `@no_fingerprint` stage's `file:` hash is their digest. A file no
module was recorded from (a `code_deps` entry that is not a module) is read from disk, as before.
The lines are split as `linecache` splits a file (`tokenize.open`, universal newlines, a final
newline added), so an unedited project keeps its recorded fingerprints.

A worker imports by name each module discovery did not register by value, with the `sys.path`
loky gives it (the coordinator's, `""` replaced by the directory the coordinator started in; it
starts in the coordinator's working directory, and its `PathFinder` skips an entry that is not a
`str`), where an import session puts the project root first. A program that runs Pivot in its own
process (a notebook, a script, a test suite) may hold a module of that name loaded from another
file, which `import` in `pipeline.py` returns from `sys.modules`; the stage would then be
fingerprinted
from one module while its worker runs another, and an edit to the project's file would leave a
stale `cached`. So when a stage's fingerprint is computed, `loaded_source.check_worker_imports`
takes each module the manifest names (`fingerprint.by_name_modules`: the `module:` entries and
the `byname:` entries, a loader's included) that `sys.modules` held when the latest import session
started (`loaded_source.note_held_modules`; kept after the session ends, so a pipeline fingerprinted
after a later discovery found nothing is still compared) and still holds, leaving out a module
cloudpickle
sends by value and the program's `__main__` (a worker gets its functions by value). For each it
predicts, without importing anything, where a worker would import it (`loaded_source.worker_origin`,
`import_origin.find_spec`): a built-in or frozen module first, which no file shadows, then each
finder of `sys.meta_path` in order, `PathFinder` searching the worker's `sys.path` (its `str`
entries, a relative one joined to the working directory), a submodule in its package's spec's
directories. Origins compare by resolved file, by kind for a built-in or frozen module, and for a
namespace package by its sorted resolved directories. Only where the prediction differs from the
held module does it start an interpreter as loky starts a worker (the same executable,
environment and working directory; `loky.backend.popen_loky_posix` imported before it is given
the worker's `sys.path`). That interpreter runs what a worker's start runs, `site` with
`sitecustomize` and the `.pth` files and loky's own imports, so their code runs once more; it
imports no project module. It reports the module its start already imported or else the spec an
import would find, on a file descriptor of its own, since startup code may write to stdout
(`import_origin.report_worker_origins`). The first such interpreter of a session, for one
`sys.path` and working directory, also confirms every other held module the prediction places
elsewhere (`loaded_source._held_suspects`), so one interpreter serves every stage of the session.
The check refuses the stage only when that interpreter also places the module elsewhere, naming
both origins, with the stage's other refusals (`fingerprint.refuse`); the remedy is renaming one
of the modules, or a new process when the program imported its own module of that name. When the
interpreter cannot run, the prediction decides and the error says so. Predictions and
confirmations are kept per `sys.path` and working directory until the session ends. A module
nothing would find, or whose finder raised, is left alone: the worker's import fails loudly. Pivot
changes nothing in the program's `sys.modules`.

Cost: every by-name module of every stage is predicted once per import session (memoized by
name, `sys.path` and working directory). On a 1,000-stage project with three by-name modules per
stage, `pivot status` took 3.0% longer (95% interval 2.5% to 3.6%) and an all-cached `pivot repro`
1.7% longer than on main; on `fan-1000` (1,000 stages, each reading one library module, `zlib`),
`pivot status` took 2.3% longer (95% interval 1.6% to 2.7%; 10 paired runs against main
64c5db6c). An interpreter start costs about 80 ms, once per session and only when a held module is
suspect.

The manifest cache records where each of those modules resolved when the walk ran (`o`, the
prediction above), and a hit re-predicts them: a project file that takes over a name a library
held (vendoring `helpers.py` where `helpers` was installed) changes no file the walk read and no
value it re-reads, so without this the cached manifest of the library would answer for the
project's code.

Limits of the comparison:

- A module the session imported itself resolved on the worker's `sys.path` when it was imported,
  so it is not compared; a file that appears or changes after that, during a run, is the worker's
  to report (below). A module a program loads by file location or through its own finder while
  the session runs (another thread of a notebook, between discovery's start and `pipeline.py`'s
  import) is not compared, and the worker's own check does not see it either, because the worker
  imports it when it unpickles the stage, before the stage's window opens.
- A finder the program inserted in `sys.meta_path` after its interpreter started is assumed to be
  in the worker too; one a `.pth` file or a `sitecustomize` installs is.
- The prediction reads a package's spec, so a package whose `__init__` changes its `__path__`
  (`pkgutil.extend_path`, or an append) has its submodules predicted in the directories of the
  spec, not of the loaded package. A submodule found elsewhere is then confirmed in a new
  interpreter, which reads specs too.
- A worker of a warm pool keeps the `sys.path` it started with, while the check reads the current
  one; the two differ only when the program changes `sys.path` between runs and loky reuses the
  pool. In the same way, a pipeline object's fingerprints are kept (`ensure_fingerprint`), so
  running the same pipeline object again after the program changes `sys.path` checks nothing
  again; a new discovery does.
- The parent's `PathFinder` keeps a finder per `sys.path` entry (`sys.path_importer_cache`), made
  for a relative entry in the working directory of its first use. A relative entry with a `chdir`
  between two stages of one session is predicted for the new working directory, while the
  program's own imports still search the old one.

A worker imports some user modules itself, from their files: a stage's module discovery does not
send by value (outside the root, or under it in a `src` layout), a module outside the root that
one sent by value refers to, and any module a stage body imports. Such a module holds the file as
the worker read it, which may be before or after an edit. Each worker gets the coordinator's
records (`loaded_source.digests()`: each module's name, resolved path and digest) once, in its
initializer arguments (`core._init_worker`), and each stage carries only their token
(`WorkerStageInfo.loaded_sources_token`), plus the records themselves when the coordinator loaded
user code after the workers started. Loky replaces a pool whose initializer arguments differ, and
`WorkerPool.start` replaces workers that started with another token.

After the stage function returns, the worker judges the user modules no recorder saw compile
(above), then checks the modules this stage used (`loaded_source.check_stage`): the modules whose
code or values its manifest hashes (`func:`, `class:`, `const:`, `default:`, `schema:`,
`partial:` and `loader:` entries, not an `ext:` by-name entry or a `module:` coverage entry; an
entry's kind is its first segment once its loader prefixes are stripped, as above, so no
dependency, output, field or stage named like a kind is read as one), the module that defines
the stage, and the modules it imported while it ran. For a `@no_fingerprint`
stage, the modules recorded from its `file:` paths count instead of its imports. A module another
stage on the worker used is not this stage's concern. Records compare by module name, path and
digest, so a new package that shadows a module mid-run differs too. A stage sent with a token
other than the one its worker started with, and without the records, cannot be compared, so
every module it used that the worker loaded counts.
A pipeline a program builds itself and passes to `executor.run(pipeline=...)` is never
discovered, so the coordinator holds no record of its modules and sends none: the worker compares
no module (`check_stage` skips a module the coordinator sent no record for), and the stage is
fingerprinted from the files when it is ready to run, as before this check existed. An edit to
its code during that run is not detected. This is by design for that programmatic path: the
`pivot` command, the engine's watch and server modes, and `executor.run` without `pipeline`
always discover the pipeline, so their modules are recorded.
When an import session ends, the entries of the modules it removed are dropped, with the
`linecache` lines pinned for them, and so are those `record_unseen_modules` made
(`loaded_source.prune`).

A difference means the stage ran code its fingerprint does not describe: the stage succeeds, logs
one warning naming the modules, and its lock carries an `edited-source:<modules>` key no
fingerprint matches, so the next run reruns it ("Previous run imported modules edited after pivot
loaded them"). A used module that either side recorded as unreadable gives the warning "uses user
modules whose source pivot cannot read" and an `unreadable-source:<modules>` key: the stage
reruns on every run. Both marks behave as the `uncovered-import:` mark does: no run-cache entry is
filed or used, and `pivot commit` carries them (`skip.rerun_mark_reason`,
`skip.carry_rerun_marks`). Under `--no-commit` the run keeps the manifest its lock would have held,
marks included, with the output hashes (`nocommit:<stage>` in StateDB); `pivot commit` writes that
manifest while the outputs on disk are still those, and computes the input hash of the run-cache
entry from it. Applying that record replaces only these two keys and keeps every other key, as
the `*.pvt` members another feature writes into the same record during the run. A record without
them (one that holds only another feature's key) reads as present, with that key, so
`pivot commit` treats the stage as one with no record. A warm worker that
imported the module before the edit holds the version the coordinator loaded, so its stages are
not marked.

A watch reload imports the modules under the root again. A module outside the root stays loaded
in the watch or server process, so its edit is loaded when pivot starts again. Until then, a
worker that imports the module after the edit (a fresh worker, or one a reload started) runs the
edit: the stages that use the module are marked on every run, and rerun, until pivot restarts.

### Libraries Installed from Git or a Local Path

The walk records library code by name (`byname:`, `ext:`) and does not hash it, but for a
stage function a library defines (`self:`) and a loader's `load`, `save` and `empty`
(`loader:`). A library installed from git or from a local path can change with no new version,
so for such a distribution Pivot also hashes the files it installed.

**Where.** `registry._compute_fingerprint` calls `fingerprint.add_distribution_keys` once the
stage's manifest is whole (the walk or its manifest-cache hit, the call-site wiring and every
loader's entries) and the worker-import check has run; a `@no_fingerprint` stage gets keys too,
from its loaders' `byname:` entries. For each `byname:` module it takes the origin of the module
this process holds (`import_origin.of_module`), or else of the module a worker would import,
found without importing it, and beside it the file Pivot judges the module's file by
(`fingerprint.judged_file` of the module's file as the import located it,
`import_origin.file_of_module`): that file, its directory resolved and its last name kept, or,
for a link outside every library path, the first file of its chain of links that is a library
path, as [Files Linked into site-packages](#files-linked-into-site-packages) says;
`loaded_source.worker_spec` predicts the origin and the file at once for a
module this process does not hold. It looks up neither when no distribution on the worker's
`sys.path` is installed from git, a local path or a local archive, or has a `direct_url.json`
Pivot cannot classify (`installed_distributions.needs_origins`). The keys join the manifest
before the lock leaves out the `byname:` entries (`fingerprint.without_by_name`), so the lock
records them.

**Which distributions.** `installed_distributions` lists the `*.dist-info` and `*.egg-info`
directories of each directory on the worker's `sys.path`, in order, as `importlib.metadata`
finds them, and maps each module to the distributions that own it by file, never by name: a
module's file belongs to every distribution whose RECORD lists it, compared as resolved paths; a
package's `__init__` file also to every distribution with a file under the package's directory,
which is the one the import searched (the directory of the file Pivot judged the `__init__` file
by; where an installer links each file into its cache, that is still the package's directory in
`site-packages`, see [Files Linked into site-packages](#files-linked-into-site-packages)); a
namespace package to
every distribution with a file under one of its directories; a built-in or frozen module, one
with no file and one nothing finds, to none. From each owner Pivot follows
its requirements transitively, through package-index distributions too: the `Requires-Dist`
headers of its metadata, or, for an `.egg-info` directory whose metadata has none, the
requirement lines of its `requires.txt`, as `importlib.metadata` reads them. Each name resolves
to the first installed distribution of that PEP 503 name on the path; a cycle ends. Each
distribution met gets the key `dist:<name>` when it has a RECORD and its `direct_url.json` (PEP
610) has `vcs_info` (any version control system), `dir_info` whose `editable` is not `true`, or
`archive_info` whose `url` has the scheme `file` (a local wheel or sdist). A distribution with no
`direct_url.json` (a package-index or `--find-links` install), with `dir_info` and
`editable: true` (`pip install -e`, `uv pip install -e`), or with `archive_info` and another URL
scheme (a remote archive) gets none. Nor does pivot's own distribution (one with a RECORD file in
one of pivot's package directories, `fingerprint._PIVOT_PACKAGE_DIRS`), whose requirements are
not followed.

**Which files.** The key hashes the files the distribution's RECORD lists that stay, normalised,
inside the directory that holds its `.dist-info` directory, each read from disk, not taken from
RECORD's hashes, so a byte edited in place in `site-packages` changes it. It leaves out every
file of a `.dist-info` directory but the distribution's own `entry_points.txt`, which decides
which plugin code runs (others of them differ by installer or build backend, and `METADATA`
carries the version), and every path with a `__pycache__` part, which the installer compiles
(pip lists the bytecode it writes). A `.pyc` file outside `__pycache__` came from the wheel, so
it is hashed, and a change to a sourceless module is seen. A file listed twice is hashed once.
So the same files installed from another commit or path, by pip or by uv, into another virtual
environment, give the same value, and a reinstall that changes only the version changes
nothing. A listed file that is missing or cannot be read (deleted, a directory, a FIFO, no read
permission) is hashed as `null`: the value changes and nothing is refused. Two installs of one
name on the path that a stage both reaches give one key, whose value covers both.

Limits:

| Limit | What happens |
|-------|--------------|
| Files a distribution installs outside the directory that holds its `.dist-info` directory: console scripts, data files, headers, and a platlib directory apart from purelib | Not hashed: RECORD lists them with a path that leaves that directory, and a console script's first line names its own virtual environment. A change to them alone reruns nothing |
| Markers and extras of a requirement (`Requires-Dist`, or a `requires.txt` line, whose `[extra]` and `[:marker]` section headers are not evaluated either) | Not evaluated: only each requirement's leading name is read, so a requirement behind a false marker or an extra that names an installed git or local distribution still adds its key. That can rerun a stage when a library it never imports changes; it never hides a change |
| A package or namespace package that a stage reads by its name (`import company.vendor` binds `company`) | Every git or local distribution with a file under its directories gives its key, though the stage may use only one of them |
| A reinstall while a Pivot process runs | Pivot reads the installed distributions, and each keyed distribution's files, once per process for one worker `sys.path` (again when that path changes), so the running process keeps the values it read: `pivot repro --watch`, with or without `--serve`, sees a reinstall when Pivot starts again. The next process reads the new files, and a stage that reaches a changed one reruns once |
| An editable install | No key. Its source is user code when it lies outside the interpreter's prefix and outside every `site-packages` or `dist-packages` directory, and the walk hashes it as it hashes the project's code. pip's checkout of a `pip install -e git+…` package, in `<venv>/src`, lies under the interpreter's prefix, so it is library code with no key, recorded by name and not walked: a change to its files is seen only where the walk hashes library code (a stage function it defines, a loader's `load`, `save` and `empty`). Where a `.pth` line puts that checkout itself on `sys.path` and its build left `<name>.egg-info` there, with no RECORD, which requires a library installed from git, a local path or a local archive, a stage that reaches a module of the checkout is refused: no RECORD lists that module's file beside that metadata (the refusal of a file that no RECORD lists, below). Reinstalling it editable, as the refusal's fix says, leaves that `<name>.egg-info` there, and the refusal stays: install the checkout non-editable instead, or set `core.unsafe_fingerprinting`. Where its build left no `<name>.egg-info` in an entry of `sys.path`, as a build backend that writes no metadata into the source tree leaves it (hatchling, flit-core, pdm, poetry-core, uv_build), nothing is refused, and a library installed from git, a local path or a local archive that the checkout requires gets no key through it: the checkout's editable `.dist-info` has a RECORD that lists its `.pth` file and no file of the checkout, so no distribution owns the checkout's modules and Pivot follows none of its requirements, where it follows the requirements of a package-index library that a stage reaches. A change to such a library then reruns no stage that reaches it only through the checkout. Install the checkout non-editable, or keep it outside the virtual environment: there its code is user code, which the walk hashes as it hashes the project's code, and the libraries it calls get their keys as the project's code's do |
| A distribution in a zip archive on `sys.path`, or behind any entry Pivot cannot list as a directory | Not read: it owns no module and gets no key. pip and uv install into a directory |
| A git or local library with compiled extensions | Its extension modules have platform-tagged names (`*.cpython-313-x86_64-linux-gnu.so`) and bytes that differ by platform, compiler and build, so its `dist:` value differs between machines, and can differ when the same commit is built again: a lock written on one machine reads stale on another |
| A version file that setuptools-scm or hatch-vcs writes into the installed package | It holds the commit (or a version made from it), so every new commit of the library changes the key, and every stage that reaches the library reruns, whatever else changed |
| One library installed from git on one machine and from a package index on another | The first machine's lock has a `dist:` key for it and the second's has none, so each machine reads the other's lock as stale (before this key, both had only `byname:` entries, which a lock leaves out) |

Refusals: when Pivot cannot tell which installed code a stage runs, it refuses the stage with an
error that names the stage once, first (`Stage '<name>': fingerprinting failed: `, then each
refusal, a blank line between two), and names the distribution, its `.dist-info` directory, the
reason and the fix (reinstall it); the warning that unsafe fingerprinting gives in its place
starts with the stage's name (`Stage '<name>': `), as every refusal it can downgrade starts. It
looks only when at least one distribution on the worker's `sys.path` is installed from git, a
local path or a local archive, or has a `direct_url.json` Pivot cannot classify, and only at the
distributions a stage needs: each one it reaches and each one those require, directly or through
others. Pivot's own distribution refuses nothing. Each other is refused for the first of these
that holds:

1. Its `direct_url.json` cannot be read, is not JSON, is nested deeper than Python's JSON parser
   recurses, is not a JSON object, names none or more than one of `vcs_info`, `dir_info` and
   `archive_info`, has a `dir_info` that is not a JSON object, has `archive_info` and no `url`, or
   has a `url` that cannot be parsed: Pivot cannot classify it, so it cannot tell whether to hash
   its files. Such a file anywhere on the path refuses only the stages that need its
   distribution.
2. It is installed from git, a local path or a local archive, and its RECORD is missing, cannot
   be read, is not UTF-8, or has a row Python's `csv` module cannot read (such as a field over
   its size limit): Pivot cannot tell its files. A distribution a stage reaches owns a module
   through its RECORD, so this refuses one that a reached distribution requires.
3. Its requirements cannot be read: its METADATA, or its PKG-INFO where it has no METADATA (an
   `.egg-info` directory), is missing, cannot be read, is not UTF-8, has a line in its headers
   (up to the first empty line, each line ended by CR LF, CR or LF) that is neither a header
   (`<name>:<value>`, as Python's `email` parser reads one) nor the continuation of one (a line
   that starts with a space or a tab, after a header), has no `Metadata-Version` header, or has a
   `Requires-Dist` header that names no distribution; or, for an `.egg-info` directory whose
   metadata has no `Requires-Dist` header, its `requires.txt` cannot be read, is not UTF-8, or
   has a line that is not blank, a comment or a section header and names no distribution. Pivot
   cannot tell what it depends on. This holds for a package-index distribution the stage needs
   too.

Pivot also refuses the stage, with a message naming, after the stage's name, the module, its
file and the distribution, when a module the stage reaches has a file that no RECORD in that
file's directories of the worker's `sys.path` lists, while a distribution in one of those
directories has no RECORD Pivot can read and is one Pivot looks at: one installed from git, a
local path or a local archive, one it cannot classify, one whose requirements it cannot read, or
one that requires such a distribution, directly or through others, of any install form (a
package-index `.egg-info` directory in a `site-packages` or `dist-packages` directory or under the
interpreter's prefix, which never has a RECORD, included). Such a distribution owns no file, so
Pivot cannot tell whether the module is that distribution's. A project's own metadata is not
one: a distribution with no RECORD and no `direct_url.json` in an entry of `sys.path` that is no
library location (`installed_distributions.in_library_location`: outside every `site-packages`
and `dist-packages` directory and the interpreter's prefix; the project root, a `src/`
directory, a `PYTHONPATH` entry, a checkout outside the interpreter's prefix), as the
`<name>.egg-info` that setuptools writes beside the project's code when it builds the project
(`uv build`, or `uv pip install -e` with build isolation), refuses nothing; one there with a
`direct_url.json` (a git or local install made with `--target` that has lost its RECORD) still
does, and so does the `<name>.egg-info` of a library's checkout under the interpreter's prefix
(pip's `pip install -e git+...` puts one in `<venv>/src`, and its develop build writes that
metadata there), whose files are library code, recorded by name and not walked. Nor does a file
or a directory of user code meet this check: Pivot gives it only the library code of the modules
the stage reaches by name (`fingerprint.add_distribution_keys`), so a module whose file is user
code (`fingerprint.is_library_file`), which the walk hashes or the user names by name
(`@pivot.fingerprint_by_name`), is not checked, and of a namespace package only its directories
in a `site-packages` or `dist-packages` directory or in one of pivot's own packages are, the rule
by which Pivot takes a namespace package as library code (a directory under the interpreter's
prefix alone, as pip's checkout of a `pip install -e git+...` package in `<venv>/src`, is the
user's part). So a project's `@pivot.fingerprint_by_name` helper whose file is user code, and
the user's part of a namespace package shared with a library, give no `dist:` key and no
refusal, as in pivot 0.10.4, whatever metadata lies beside them. A namespace package's portion
under the interpreter's prefix but outside `site-packages` is the user's by that rule also where
a non-editable install put it there (a git install made with `--target` into `<prefix>/vendor`,
its RECORD kept): a stage that reaches the namespace package only by its top name (`company`)
gets no key of that install, and one that reaches its module by name (`company.v`) still does,
since that module's file is library code. A file's
directories are the deepest entries of the worker's `sys.path` that hold its places, and the
deepest entry that holds its resolved path, where a link points; a RECORD in any of them lists
the file when one of its paths resolves to the file's resolved path. A module's own file lies
where Pivot judged it (`fingerprint.judged_file`: as the import located it, its directory
resolved, or, for a link outside every library path, the first file of its chain of links that
is a library path) and in every later file of that chain, library path or not; a file under a
package lies where the walk lists it in the package's directory and, for a symbolic link, in
every later file of its chain (`installed_distributions.link_chain`, which follows at most 40
links). So a module's own file linked from one entry into another that holds such a distribution
is refused, naming the file in that entry, also where that file is itself a link into uv's cache
or the entry is no library path (a `--target` directory), and a module or package file linked
into an entry whose RECORD lists the file is not refused; a walked package file is checked only in
the entries the walk covers (below), so a link from it into another entry that holds such a
distribution is not refused there. Where an installer links each file into its cache, where that
cache lies outside every entry of the path, no entry holds the resolved path, and the file is
judged where it lies; where the cache lies inside an entry (`UV_CACHE_DIR` under the project
root), that entry holds the resolved path too, and a project's own metadata there refuses
nothing for it (another distribution there with no RECORD that Pivot looks at does; see the
Limits of [Files Linked into site-packages](#files-linked-into-site-packages)). The files checked
are the module's own and, for a package's `__init__` file or a namespace package, the files under
its directories but those under a `__pycache__`
directory, which the key leaves out too (a `.pyc` file outside one is checked). Pivot walks such
a directory only when the deepest entry of `sys.path` above it, or an entry at or under it,
holds such a distribution, and stops once it has found, in each of those entries, a file that no
RECORD lists. The walk enters no subdirectory that is a symbolic link, so a file under one is
not checked, and a file under the directory that is a symbolic link into an entry other than
those, which holds such a distribution, is not checked there. Each process runs the check once
for each module. Only this check reads every RECORD of a directory.

`core.unsafe_fingerprinting` (or `PIVOT_UNSAFE_FINGERPRINTING=1`) turns each of these refusals
into a warning with the same message, so the warning names the stage it lets run, and the
stage's result is kept out of the run cache, as for the other refusals it downgrades; they are
reported with the stage's other refusals. A distribution no stage needs refuses nothing.

### Files Linked into site-packages

Pivot takes a module's file as library code, recorded by name and not walked, when its resolved
path lies under the interpreter's prefix, in a `site-packages` or `dist-packages` directory, or in
one of pivot's own packages (`fingerprint.is_library_path`). A virtual environment that uv builds
with `UV_LINK_MODE=symlink` (`uv pip install --link-mode symlink`) holds each installed file as a
symbolic link into uv's cache, inside real directories, so every resolved path there lies in the
cache and every library would be user code: the walk would hash pydantic's source and refuse a
class whose metaclass is pydantic's `ModelMetaclass`, whose `__getattribute__` it would take as
user code. So `fingerprint.is_library_file` decides a file in this order:

1. Its resolved path is a library path: library code. A copied environment's files are decided
   here.
2. Else Pivot follows the file's chain of symbolic links one link at a time, from the file as the
   import found it (its path with its directory resolved, whatever links its directory was
   reached through, and its last name kept) to the file each link names (the link's target,
   joined to the link's directory when relative, its directory resolved and its last name kept),
   and judges the file by the first file of the chain that is a library path
   (`fingerprint.judged_file`): library code, as the same file copied there would be, unless the
   RECORD of an editable distribution (`direct_url.json`, PEP 610, with `dir_info` and
   `editable: true`) in that file's site directory lists it. Its site directory is the deepest
   entry of the worker's `sys.path` that holds it, and the RECORD lists it when one of its paths,
   compared as the RECORD writes it, normalised, never resolved, is the file's path relative to
   that entry (`installed_distributions.listed_by_editable`). A file that such an editable
   distribution's RECORD lists keeps the decision of its resolved path, as in pivot 0.10.4: user
   code.
3. Else user code: a file of the chain that is no library path and no symbolic link comes before
   one that is a library path, or the chain reaches no library path within 40 links, the most
   that a Linux path lookup follows (a loop of links among them).

The keys and the refusals of
[Libraries Installed from Git or a Local Path](#libraries-installed-from-git-or-a-local-path)
use the places of a module's file, found by the chain of rule 2 for every file, also one that
rule 1 decides: `fingerprint.reached_module_of` gives `installed_distributions`, beside the
module's origin, the file Pivot judges it by (`fingerprint.judged_file`) and every later file of
its chain of links, library path or not, so the site directory and the package directory they
name are those where the chain reaches a library path, in a copied environment as in a linked
one. A file whose own path, its directory resolved, is a library path already, as every file of
a virtual environment is, is its own judged file. The check of a file that no RECORD lists sees
the entry of each place, and a package file it walks that is a link lies in each file of its
chain too, so `A/site-packages/m.py`, a link made by hand to `B/site-packages/m.py` or to
`P/libs/m.py` (a `--target` directory on the path), lies in both, whether that file is a copy or
a link into uv's cache (`test_installed_distributions_snapshot.py`,
`test_installed_distributions_unlisted.py`). A virtual environment that uv builds with
`UV_LINK_MODE=symlink` so fingerprints as one whose files uv copied, and gives the same `dist:`
keys and the same refusals. A distribution owns each
directory above each path its RECORD lists inside its site directory, named as the RECORD names
it, besides each directory above the file that path resolves to, so a namespace package, whose
directories stay in `site-packages`, keeps its owner there, and so does a regular package, whose
directory Pivot takes from its judged `__init__` file, not from that file's resolved path in the
cache. A second distribution that puts a file
inside another's package (`pkg/contrib/extra.py`, where an index library holds
`pkg/__init__.py`) is so reached through `pkg` as in a copied environment, and refuses the stage
as there once it has lost its RECORD. A linked file that no RECORD lists, a link of an editable
install whose RECORD is lost or that Pivot cannot read, and a link that only a distribution Pivot
cannot classify, or one that is not editable, lists, are library code by rule 2, so the checks
of [Libraries Installed from Git or a Local Path](#libraries-installed-from-git-or-a-local-path)
judge such a file as they judge the copied file (`test_installed_distributions_link_mode.py`): a
reached distribution that has lost its RECORD refuses the stage with the message that names the
module's file where it lies in `site-packages`, one whose `direct_url.json` Pivot cannot classify
refuses it naming the distribution, as does a dependency of either kind, and under
`core.unsafe_fingerprinting` each refusal is the copied environment's warning. By rule 2 a
library package linked into the project root (`mylib -> .venv/lib/python3.13/site-packages/mylib`),
an entry of `sys.path` that is a link to `site-packages`, and a module file linked by hand into
the project root to a file of `site-packages` (`mod.py -> .venv/lib/python3.13/site-packages/mod.py`)
are judged in `site-packages` too, as their resolved paths are where the files are copies, and
their keys and refusals name `site-packages` in both environments
(`test_installed_distributions_snapshot.py`: a git library that has lost its RECORD refuses the
stage once, naming the module's file in `site-packages`, also beside a project's own `.egg-info`
with no RECORD; a project file linked to a module file there, which an index library's RECORD
lists or which no RECORD lists, is not refused beside that `.egg-info` and a local install with no
RECORD in the project root; a project package whose files are linked to the
files of a git library's package gives the key of each git library with a file in that package
in `site-packages`). A project file linked
to a file outside every library path, and a loop of links, are user code by rule 3
(`test_is_library_file.py`). flit's `install --symlink` (flit 4.2.0) links a checkout's module
file, or its package's directory, into `site-packages`, with a `direct_url.json` that says
editable and a RECORD that lists the link; its code stays user code, walked as the project's own,
as in pivot 0.10.4, so an edit to the checkout reruns the stage: the module file by rule 2, since
an editable install's RECORD lists it, also through a project file linked to it, and a file
under the linked package directory by rule 3, since that directory resolves into the checkout,
where the file is no link. `uv pip install -e --link-mode symlink` of a setuptools project links
the `.pth` file and the finder module that setuptools writes into uv's cache, and the editable
install's RECORD lists both: the finder module is user code by rule 2, where its copy in a copied
environment is library code by rule 1, and the checkout it imports is user code by its resolved
path in both; a stage that imports the editable package does not reach the finder module, so the
two environments give one lock (`test_installed_distributions_link_mode.py`). A directory (a
namespace package's, an entry of `sys.path`) is judged by its resolved path alone: uv keeps
directories real. The decision is made once per file name per process: each file of the chain
costs one `os.readlink`, and for a file whose chain reaches a library path Pivot reads, once, the
RECORD of each editable distribution of that path's site directory and no other, so a virtual
environment whose files are copies or hard links reads nothing more, and one with no editable
distribution reads no RECORD for it; a worker, whose import recorder asks only this question,
reads no requirement and builds no index of the distributions that can give a key.

Limits:

| Limit | What happens |
|-------|--------------|
| A file linked into `site-packages` that no editable distribution's RECORD lists: a link made by hand to a user's file (a new name, or in place of a library's file), a link of an editable install whose RECORD is lost or that Pivot cannot read, a link that only a distribution Pivot cannot classify, or one that is not editable, lists, and a module file that flit before 3.0 linked with `install --symlink` (flit wrote no `direct_url.json`, PEP 610, before 3.0, so its distribution reads as a package-index install, which is not editable); also when the stage reaches it through a link to it | The linked file is library code, recorded by name and refused as a copied file would be (rule 2), so an edit to the file it points to reruns nothing: an edit to the checkout of an editable install whose RECORD is lost reruns nothing either. Pivot 0.10.4 resolved the link and walked its target as user code, so that edit reran the stage. Pivot reads no second sign of an editable install. A package directory that flit links is not affected: its files resolve into the checkout (rule 3). Fix: reinstall with a tool that writes `direct_url.json` (flit 3.0 or later, `pip install -e`, `uv pip install -e`), reinstall an editable install whose RECORD is lost, or install a user's code editable in place of a link made by hand |
| One entry of uv's cache linked from two entries of `sys.path`: the same wheel installed with `--link-mode symlink` into a virtual environment and into the environment it sees through `--system-site-packages`, or into a `--target` directory | The RECORDs of both installs resolve to the same files, so both own a module the stage reaches in either (a file is owned by its resolved path, a package's directory by where the import searched). For a git or local library the key's value then covers both installs, where copied files give the value of the one the stage imports, and a second install Pivot cannot read (a `direct_url.json` it cannot classify, requirements it cannot read) refuses the stage. That adds reruns and refusals only: no change is hidden |
| uv's cache inside an entry of `sys.path`: `UV_CACHE_DIR` under the project root, which is on `sys.path`, for a virtual environment that uv builds with `--link-mode symlink` | A linked file's resolved path, in the cache, then lies in that entry too, so the check above judges the file there as well as where Pivot judged it. Where that entry holds a distribution with no RECORD that Pivot looks at and that is no project's own metadata (one installed into the project root from a local directory, `dir_info`, or from git, that has lost its RECORD), a file the check covers that no RECORD lists (one of a distribution that has lost its RECORD) refuses the stage, named as it lies in the cache; with the files copied, or the cache outside every entry, that distribution refuses nothing for it. A project's own `.egg-info` there refuses nothing in either layout. That fails closed: it adds a refusal only, for installs that are already broken, and hides no change |

### Module-Level Instances

```python
processor = DataProcessor()

def process():
    return processor.transform(data)  # State held by a module-level instance
```

A user class instance held at module level is refused: its attributes are state no code hash
records. An instance the stage creates (`DataProcessor().transform(data)`) is tracked through
the class's methods. So is an instance a stage-level `functools.partial`, a loader's config, a
dataclass field default or a `StageParams` default holds (`functools.partial(run,
DataProcessor())`): its fields are recorded by value, and its class is walked as if the stage
named it (see "User class instances held in bound values" above).

**Workaround:** Create the instance in the stage, with its settings from `StageParams`.

### Lambda Functions

```python
process = eval("lambda x: x + 1")  # No source: hashed by its code object
```

A lambda without accessible source is hashed by its code object: `marshal` of `__code__` with the source path removed, so bytecode and constants (`x + 1` against `x + 100`) count. The hash is the same in every interpreter session. Only a callable with neither source nor `__code__` is hashed by its `id()`, which differs between sessions; Pivot logs a warning for a user callable hashed that way.

**Workaround:** Prefer named functions, whose AST hash ignores formatting and comments.

### Dynamic Patterns

```python
func_name = "process"
func = getattr(module, func_name)  # Dynamic lookup
result = eval("process(data)")     # String execution
```

**Workaround:** Not supported; restructure code.

### Module-Level Values

A stage may read a module-level `bool`, `int`, `float`, `str`, `bytes` or `None`; its current value is recorded with `repr()`, so editing it re-runs the stage. A `tuple` or `frozenset` whose members are all such scalars, and a compiled regular expression, are recorded by value too: the tuple keeps member order, the frozenset sorts members by canonical JSON text (independent of `PYTHONHASHSEED`), and the regex records its full pattern and flags. Members are checked one level deep only. Reading any other module-level value (list, dict, set, a tuple or frozenset holding a tuple, list, dict or object, enum member, class instance, or a method bound to an instance such as `np.random.default_rng(SEED).normal`), directly or as `module.NAME`, is a `StageDefinitionError` that names the variable and points to `StageParams`. A default argument such as `cols=COLUMNS` or `factor=settings.FACTOR` on the stage or on a helper it calls follows the same rule, because a default is evaluated once when the module loads and its value never appears in the AST:

```python
THRESHOLD = 0.5            # allowed: tracked by value
COLUMNS = ("a", "b")       # allowed: tracked by value
STMT = re.compile(r"\s+")  # allowed: tracked by pattern and flags
PAIRS = (("a", 1),)        # error when a stage reads it
CONFIG = load_config()     # error when a stage reads it (a dict)
```

Configuration belongs in the stage's `StageParams`:

```python
class ProcessParams(StageParams):
    threshold: float = 0.5
    columns: list[str] = ["a", "b"]


@stage
def process(data: pd.DataFrame, params: ProcessParams) -> pd.DataFrame:
    return data.loc[data["value"] > params.threshold, params.columns]
```

For an explicitly unsafe migration path, `core.unsafe_fingerprinting=true` or
`PIVOT_UNSAFE_FINGERPRINTING=1` downgrades the error to a warning. A stage whose
fingerprint needed that downgrade is marked (`RegistryStageInfo.unsafe_fingerprint`),
and neither `pivot repro` nor `pivot commit` files its result in the run cache or
restores one from it.

## Why AST + getclosurevars?

Other approaches were tested and rejected:

| Approach | Problem |
|----------|---------|
| **Bytecode** | `co_code` doesn't include constants |
| **cloudpickle** | Pickles functions by reference, not content |
| **Comprehensive hashing** | Too slow, over-sensitive |

The AST + getclosurevars combination provides:

- **Accuracy** - Detects real code changes
- **Performance** - Fast hashing with xxhash64
- **Transitive tracking** - Follows dependency chains
- **Stability** - Ignores formatting changes

## Performance: Caching

Fingerprinting uses several `WeakKeyDictionary` caches to avoid repeated expensive operations:

| Cache | What it stores | Why it matters |
|-------|---------------|----------------|
| `_hash_function_ast_cache` | AST hash per function | Avoids re-parsing the same function |
| `_getclosurevars_cache` | Closure vars per function | `getclosurevars()` costs ~0.5ms per call |
| `_is_user_code_cache` | Boolean per object | Called 10K+ times, mostly repeats |
| `_get_type_hints_cache` | Type hints per function | `get_type_hints()` is expensive |
| `_module_attr_cache` | Module attribute usage per function | Avoids re-parsing AST for `module.attr` patterns |

These caches use weak references, so entries are automatically cleaned up when functions are garbage collected. They have no lock. Stages are fingerprinted in the `pivot` process, never in a worker (each worker process has its own memory), on the thread that asks: `pivot status` and `pivot commit` on their main thread, the engine's skip check and watched startup snapshot in anyio worker threads (`anyio.to_thread.run_sync`), a stage that the skip check did not fingerprint (under `--force`, or with no lock yet) on the engine's event loop when it is sent to a worker (`executor.core.prepare_worker_info`), a watch reload in a daemon thread; each of these asks for one stage at a time. An engine that serves RPC (`pivot repro --tui`, `pivot repro --watch --serve`, `pivot run --tui`) is the exception: it fingerprints for an `explain` request on its event loop (`engine.stage_queries.explain`) and for a `commit` request in a worker thread, waiting only for the startup snapshot and a reload (`Engine.wait_for_fingerprints`), so either can fingerprint while a run's skip check, or the other request, fingerprints on another thread, and the two share these caches.

The `dist:` keys are not in the manifest cache (`sm:`): `registry._compute_fingerprint` adds them after `fingerprint.get_stage_fingerprint_cached` returns, on a hit and a miss alike, because a reinstall to the same path changes no source file, read or by-name origin that a hit re-checks. What they need is read once per process by `installed_distributions`, under a lock, for one worker `sys.path` and one set of pivot's package directories: the metadata directories on the path and each `direct_url.json`; when a stage's keys are first asked for, and only when one of those distributions is installed from git, a local path or a local archive, or cannot be classified, every distribution's requirements (its `Requires-Dist` headers, and an `.egg-info` directory's `requires.txt` when its metadata has none) and RECORD for the distributions that can give a key or a refusal (those, each one whose requirements cannot be read, and each one that requires one of them, directly or through others); every RECORD of a directory, and the files under a reached package's or namespace package's directory, only for the refusal check of [Libraries Installed from Git or a Local Path](#libraries-installed-from-git-or-a-local-path), when that section says; the RECORD of each editable distribution of a directory, its paths not resolved, and no other RECORD, once, when `fingerprint.is_library_file` first asks about a file linked there ([Files Linked into site-packages](#files-linked-into-site-packages)); and a keyed distribution's files only when a stage first reaches it. Each RECORD is read once, whichever of these needs it first. A process that asks only about linked files, as a worker does while it imports, reads no requirement and builds no index of those distributions. Each path a RECORD lists for a key or a refusal, and each file under such a directory, is resolved to the path `os.path.realpath` gives, a name at a time from the nearest directory above it already resolved (an entry of the path, or one resolved for an earlier path): each new name, of a directory or of the file, costs one `os.path.islink`, and only a name that is a symbolic link, or is empty, `.` or `..`, is given to `realpath`, with the path up to it. The exception is a RECORD path of PATH_MAX bytes or more in its directory: it is given to `realpath` whole, which keeps nothing from one path to the next; the index still keeps every directory above each listed file, so one RECORD row costs memory of the order of its names times its length, bounded only by installed metadata, which Pivot trusts. Such a path can still name a file, a long name in a shorter directory (as a venv moved into a deep directory leaves it), which the walk lists and which can be a module's origin: it owns that file, and that file is not refused as one that no RECORD lists; its digest, where it is hashed, is `null` when `os.open` refuses the path as that long, so a change to its bytes changes no key. A RECORD path that can name no file (one with a NUL character or a name the filesystem encoding cannot encode) is not resolved: it owns no file, and its digest, where it is hashed, is `null`, as a missing file's is. A manifest with no `byname:` entry reads nothing, and on a path with no distribution installed from git, a local path or a local archive, and none Pivot cannot classify, `fingerprint.add_distribution_keys` looks up no module's origin.

## Manifest Key Prefixes

The fingerprint manifest uses prefixed keys to distinguish different types of tracked items:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `self:` | The stage function itself | `self:train` |
| `func:` | Helper functions (user code callables) | `func:myproject.helpers.process_data` |
| `class:` | Class definitions | `class:myproject.params.TrainParams` |
| `const:` | Immutable values read as globals or as `module.NAME` | `const:myproject.settings.THRESHOLD` |
| `default:` | Default argument values that the AST does not record | `default:myproject.stages.scale.factor` |
| `wire:` | Call-site artifact wiring: parameter bindings, collection kind/order, input paths | `wire:left` → `raw-a`, `wire:items` → `{"kind":"list","identities":["a","b"]}`, `wire:file:path` → `old.txt` |
| `partial:` | `functools.partial` bound arguments | `partial:myproject.stage.bound.args` |
| `ext:` | A library function or class a stage reads, named by its definition (the walk does not hash its code) | `ext:myproject.stages.agg` → `numpy.mean` |
| `byname:` | A module a stage's worker imports by name to run code the walk does not hash: a library module read as a global or imported in a function body, the module that defines each `ext:` callable, and the module of the library code a stage reaches through a base class, a type hint or class-body annotation, a value, a pivot loader class, its own stage function, or a callable whose `__module__` is not a string (see [What Gets Tracked](#what-gets-tracked)). Compared with the module the worker would import (`loaded_source.check_worker_imports`), and re-resolved on a manifest cache hit. A loader's `byname:` entries join the stage's own without the `dep:`/`out:` prefix (`fingerprint.merge_loader_fingerprint`), so only a key that starts with `byname:` is one, and a dependency or output named `byname` keeps its entries. Its modules are the ones `dist:` keys come from. Recorded by the walk and the manifest cache only: a lock leaves it out (`fingerprint.without_by_name`) | `byname:numpy` |
| `dist:` | `dist:<name>`: the installed files of a distribution installed from git, a local path or a local archive, not editable, that owns the module of a `byname:` entry, or that such a distribution requires, directly or through others. `<name>` is the PEP 503 name (lower case, each run of `-`, `_` and `.` made one `-`) read from its `.dist-info` directory's name. The value is the xxh64 hex of the compact JSON array, sorted, of the content digest of each install of that name that gives the stage this key; a content digest is the xxh64 hex of the compact JSON array, in path order, of `[path, digest]` for each hashed file, `digest` being the xxh64 hex of the file's bytes read from disk, or `null` for a file that is missing or cannot be read, and the distribution's own `entry_points.txt` being named `.dist-info/entry_points.txt` (`installed_distributions.content_digest`, `key_value`). Added after the manifest cache and recorded in the lock: see [Libraries Installed from Git or a Local Path](#libraries-installed-from-git-or-a-local-path) | `dist:fix-lib` |
| `loader:` | Loader class methods and config | `loader:pivot.loaders.CSV:load` |
| `schema:` | Pydantic model JSON schema | `schema:myproject.params.TrainParams` |
| `builtin:` | Built-in types (for deterministic hashing) | `builtin:list` |

## Comparison with Other Tools

| Tool | Code Change Detection |
|------|----------------------|
| **DVC** | Hashes shell command string only |
| **Kedro** | No incremental execution based on code |
| **Hamilton** | No persistent caching between runs |
| **Prefect** | Caches by inputs, not code changes |
| **Dagster** | Hashes op config, not code |
| **Pivot** | AST + getclosurevars with transitive tracking |

## Debugging Fingerprints

Use explain mode to see what changed:

```bash
$ pivot status --explain

Stage: demo/train
  Status: WILL RUN
  Reason: Code changed

  Code Changes:
    func:helper_a
      Old: 5995c853
      New: a1b2c3d4
```

## Manual Invalidation

If automatic detection fails, manually invalidate:

```bash
# Force re-run of specific stage
pivot repro demo/train --force
```

## Further Reading

- [Test Suite](https://github.com/sjawhar/pivot/tree/main/tests/fingerprint) - Comprehensive behavior tests
- [Architecture Overview](overview.md) - System design and other components
