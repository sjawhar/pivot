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
| Class definitions | Yes | Full class AST hashed |
| Callable members of user classes | Yes | User-defined methods, classmethods, staticmethods, property accessors, cached properties, `functools.partial`/`partialmethod` members, nested user classes, and dataclass default factories are walked like stage helpers. Callable instances and other user-defined descriptors are refused because their state is not reliably fingerprintable. Generated and library members are skipped. |
| StageParams subclasses | Yes | Type hint analysis + full class AST + Pydantic field defaults + `default_factory` functions |
| Custom loaders | Yes | Loader method ASTs hashed. A user loader's own `load()`, `save()` and `empty()`, and functions and classes in its config fields, are walked like stage helpers: module values they read are tracked or refused. Methods inherited from `pivot.loaders` are hashed only |
| Module-level immutable values | Yes (value) | Scalars, exact tuples and frozensets of scalars, and compiled regexes are recorded by value |
| Other module-level values | No — error | Use `StageParams` instead. Lists, dicts, sets, nested collections, enum members, and class instances are refused. This includes a method bound to an instance (`apply = Scaler(3).apply`, `normal = np.random.default_rng(SEED).normal`): the instance's state is not hashed. A method bound to a module, a class, a scalar or a `logging.Logger` is allowed |
| Closure variables | Yes (value) | `getclosurevars()` nonlocals checked like module-level values, keyed under the binding the closure was reached through, so two closures from one factory (`h2 = make(2)`, `h5 = make(5)`) keep their own values |
| Default arguments | Yes (value) | `__defaults__`/`__kwdefaults__` checked like module-level values; a literal default is covered by the AST hash |
| `functools.partial` | Yes | Bound args + underlying function; functions and classes in bound args walked like stage helpers |
| Library functions and classes | Yes (name) | A global, closure cell, default or partial holding `np.mean` records `ext:<binding>` → `numpy.mean`, so rebinding it to `np.median` reruns the stage; library code itself is not hashed |
| Secrets outside a stage's params | Yes (digest) | A `SecretStr`, `SecretBytes` or `Secret[T]` in a `StageParams` default on a helper, a loader config field or a partial argument hashes as its keyed `hmac:` digest, the one the lock file records (`parameters.secret_digest`), so changing it reruns the stage without its value in any manifest |
| Class attributes a stage reads | Yes (value) | `Cls.NAME` read directly on a user class that the function names (a global, cell or default holding the class), or reaches through a user module (`settings.Config.SCALE`). When `NAME` is written in a user class body, it follows the module-level rule: scalars, exact tuples and frozensets of scalars, and compiled regexes are recorded; mutable or nested values are refused; a plain function is walked; and a member of a user Enum is recorded by name and value. A pydantic or StageParams class read this way is covered by its schema hash, field defaults included. Attributes a library base or the language defines (an inherited stdlib dict, `NamedTuple._fields`, an Enum's `_member_names_`), and descriptors (property, classmethod, staticmethod), are left to the class AST hash |
| Other class-level and module reads | No | Not tracked: `cls.SCALE` inside a classmethod, `Outer.Inner.X`, an attribute read through an instance (`Config().SCALE`), a `ClassVar` on a `StageParams` class, a module read through a local alias (`s = settings; s.SCALE`) or through the object a dynamic import returns (`load("settings").SCALE`), a literal `getattr(settings, "SCALE")`, and a module held in a class attribute (`Holder.lib.SCALE`). Put such configuration in the stage's `StageParams` |

**Note:** The class AST and callable-member walk cover a class's code and the module values callable members read. Non-callable class attributes are only tracked in the direct `Cls.NAME` and `module.Cls.NAME` shapes above, plus dataclass and Pydantic field defaults. A value read as `module.NAME` follows the rule it follows as a global: a library function is an `ext:` entry, a `functools.partial` is walked, and a logger is ignored.

For Pydantic models, both static defaults and `default_factory` functions are tracked:

- **Static defaults** (e.g., `threshold: float = 0.5`): The value is hashed via `repr()`
- **Factory functions** (e.g., `items: list[str] = Field(default_factory=list)`): The factory function's AST is hashed, so changes to the factory logic trigger re-runs

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
and `out:<output>:module:<name>` keys a user loader's walk adds.

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
| A result cached by an earlier Pivot version | If its static fingerprint is unchanged, it may be reused, from its lock, which Pivot reads as it is, or from its run-cache entry, which Pivot finds under the key that version filed it under (unmarked, with each directory the stage reads whole hashed from its listing the way that version hashed it: `worker.older_pivot_input_hash`, `listing.legacy_digest`). `_CACHE_SCHEMA_VERSION` keys only the fingerprint caches in state.lmdb, not the lock or run cache. A stage whose static walk reaches a user class method or a user module gains entries in its manifest, so it reruns once as Code changed after the upgrade. That accepted one-time rerun writes the current manifest; later runs reuse it until code or inputs change. Run `pivot repro --force <stage>` once to re-check a stage you know imports dynamically. Upgrading from 0.5.6 or earlier, every stage with outputs reruns once as Code changed, because each output's tag (`out:<key>:tag`) joined the manifest; that run is checked, so it marks a stage that imports dynamically |

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

A session puts the project root first on `sys.path`, so project code's `import helpers` loads the
root's `helpers.py`, as it does in a `pivot` process and in a worker. A program that runs Pivot
in its own process (`executor.run`, a notebook) may already hold a user module of that name from
elsewhere, which `sys.modules` would return instead: the stage would be fingerprinted from that
module while its worker, which imports a module discovery did not send by value from its file,
ran the root's. So a session displaces each user module loaded from outside the root whose
top-level module or package the root provides, with its submodules, and puts them back when it
ends (`discovery._displace_modules_the_root_provides`). Libraries stay, as Pivot and its workers
hold them too, and so does the program's main module.

Fingerprinting reads the recorded bytes, not the file: `loaded_source.getsource` puts their lines
into `linecache` as an entry without an mtime, which `linecache.checkcache` leaves alone, before
it calls `inspect.getsource`; `fingerprint._source_digest`, which keys the `fp:` and `sm:` caches,
returns their digest; and a `@no_fingerprint` stage's `file:` hash is their digest. A file no
module was recorded from (a `code_deps` entry that is not a module) is read from disk, as before.
The lines are split as `linecache` splits a file (`tokenize.open`, universal newlines, a final
newline added), so an unedited project keeps its recorded fingerprints.

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
`partial:` and `loader:` entries, not an `ext:` by-name entry or a `module:` coverage entry), the
module that defines the stage, and the modules it imported while it ran. For a `@no_fingerprint`
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

### Module-Level Instances

```python
processor = DataProcessor()

def process():
    return processor.transform(data)  # State held by a module-level instance
```

A user class instance held at module level is refused: its attributes are state no code hash
records. An instance the stage creates (`DataProcessor().transform(data)`) is tracked through
the class's methods.

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

These caches use weak references, so entries are automatically cleaned up when functions are garbage collected. The caches are not thread-safe, but fingerprinting runs single-threaded per process (multiprocessing uses separate memory spaces).

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
| `ext:` | A library function or class a stage reads, named by its definition (its code is not hashed) | `ext:myproject.stages.agg` → `numpy.mean` |
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
