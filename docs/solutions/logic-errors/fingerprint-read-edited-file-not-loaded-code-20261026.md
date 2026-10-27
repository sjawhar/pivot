---
module: Fingerprinting
date: 2026-10-26
problem_type: logic_error
component: fingerprint
symptoms:
  - "A stage that runs after an upstream stage edits a helper writes the old helper's output"
  - "The next pivot repro reports that stage cached, with the old output, though the helper file holds the new code"
  - "A helper edited in an editor while a long run continues gives the same stale cache hit"
root_cause: logic_error
resolution_type: code_fix
severity: critical
tags: [fingerprint, linecache, inspect, cloudpickle, pickle-by-value, warm-workers, soundness, pivot]
---

# Troubleshooting: Fingerprints Read the Edited File, Workers Run the Loaded Code

## Problem

A stage's lock recorded a fingerprint of a helper as edited on disk, while the worker ran the
helper as discovery had loaded it. The next run compared the lock with the edited file, found them
equal, and reported the stale output as cached (Dispatch BET-89).

## Root Cause

Two views of one function disagreed:

- **What the worker runs.** Discovery registers every module under the project root with
  `cloudpickle.register_pickle_by_value`, so a worker receives the code objects the coordinator
  compiled at import. A module the worker imports itself (outside the root) is whatever the file
  held when that worker imported it, which a warm worker may have done before the edit.
- **What the fingerprint hashed.** Outside watch mode, `registry.ensure_fingerprint` runs when a
  stage is ready, after its upstream stages. `inspect.getsource` reads through `linecache`, and
  `inspect.findsource` calls `linecache.checkcache`, which drops an entry whose size or mtime
  changed, so the hash was of the file at that moment. `fingerprint._source_digest` even popped the
  `linecache` entry on purpose.

The in-memory `WeakKeyDictionary` of function hashes hid the bug when an earlier stage had hashed
the same function before the edit, so whether a stage was stale depended on stage order.

## Solution

Record the bytes each user module was compiled from, and fingerprint those:

- `discovery._ProjectSourceLoader.get_code` captures the source bytes its own `get_data` read (the
  bytes the checked-hash bytecode was validated against, or compiled from) and calls
  `loaded_source.record`. Every other user-code module is recorded from its own loader at the end
  of discovery and at each import a worker's hooks see (editable-install finders, zip importers,
  `spec_from_file_location`). A module whose loader gives no source is recorded as unreadable.
- `loaded_source.getsource` puts the recorded lines into `linecache` as an entry without an mtime
  (CPython's form for loader-supplied source, which `checkcache` leaves alone) and calls
  `inspect.getsource`. It restores the entry if something cleared `linecache`; `prune` removes it
  with the record when an import session ends.
- `fingerprint._source_digest` and a `@no_fingerprint` stage's `file:` hash use the recorded
  digest, so the `fp:`/`sm:` caches stay keyed by the content they hashed.
- Workers record what they import too. Each worker gets the coordinator's records once, in its
  loky initializer arguments, and each stage carries their token. A stage that used a module (its
  manifest hashes the module's code, or it imported it) whose record differs gets an
  `edited-source:` lock key and one warning; one that used an unreadable module gets
  `unreadable-source:` and reruns every run.
- A `--no-commit` run records the manifest its lock would have held; `pivot commit` writes it while
  the outputs on disk are still the ones that run wrote.
- A user module imported before Pivot installs its recorder (by `sitecustomize` or a `.pth` import
  line, in the coordinator and in every worker) was compiled before any recorder could see it, and
  a module that a module loads with a loader it calls itself (`SourceFileLoader(...).load_module()`)
  passes no recorder. `loaded_source.record_unseen_modules`, at the end of discovery, in each
  worker's initializer and in a worker after each stage, reads its file and records it only where
  the file and its cached bytecode show they hold the module's code: CPython's own source file
  loader imported it, the file's change time is at least 2 s older than the process start (a
  file system with whole-second times stamps a change up to 1 s early, and a forward step of the
  wall clock moves the start later), and the bytecode the loader may have run instead compiles to
  the same code. Any other such module is recorded as unreadable, with the reason, which a stage
  that uses it logs once per process; a worker whose record differs from the coordinator's marks
  the stage.
- The path hook and the module recorder are installed only while a discovery imports; a worker
  keeps its own for its life. Left installed after discovery, the recorder recorded modules that
  a later caller (or a later test) loaded, and changed what that caller fingerprinted.

The recorded lines are split exactly as `linecache.updatecache` splits a file, so an unedited
project keeps every recorded fingerprint.

## Prevention

- A fingerprint must be computed from the same snapshot of code that the worker executes. Any
  read of a source file at fingerprint time is a second snapshot, and so is a commit that
  fingerprints the code again after the run.
- `inspect.getsource` is not a pure function of the function object: it reads the file system.
- The rule is about the module, not the loader that imported it: a fix inside one loader misses
  every other way a module is imported.
- Check per stage, not per worker: a worker's other stages did not use the module.
- Test code-change detection with the edit made *after* the import and *before* the fingerprint,
  and with the stage order that leaves the function unhashed until after the edit.
- "Every module is recorded" includes modules that exist before Pivot runs any code. Check what
  `sys.modules` holds when the recorder starts, and decide for each module whether its file can
  be shown to hold its code; never fall back to reading the file silently.
- A time comparison that proves a file unchanged needs a margin for every way a timestamp can
  read early: a file system that keeps whole seconds, and a step of the wall clock between the
  edit and the moment the start is computed; pin the clock's offset when Pivot is imported.
- A module can appear after the last check: a worker loads modules while its stages run, by
  loaders no hook sees. Judge again at each point a process can have loaded one.
- A process-wide hook (`sys.meta_path`, `sys.path_hooks`, a `linecache` pin, a patched
  `importlib.util` function) must end with the session that installed it, in the tests too: one
  pytest-xdist worker runs many tests in one process, so a leaked hook makes the suite pass or
  fail by schedule. The suite's `restore_import_machinery` fixture restores them after each test
  and ends the discovery session the test started.
