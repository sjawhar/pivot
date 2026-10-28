# CHANGELOG

<!-- version list -->

## v0.5.15 (2026-10-28)

### Bug Fixes

- **fingerprint**: Record the code a worker ran when user code is edited during a run
  ([#518](https://github.com/sjawhar/pivot/pull/518),
  [`900e30c`](https://github.com/sjawhar/pivot/commit/900e30c273ffc2d1ce1f64b8f19ddde8ceeda9a1))

### Testing

- **fingerprint**: Write queued fingerprint cache entries only into the project that queued them
  ([#536](https://github.com/sjawhar/pivot/pull/536),
  [`6c54f8d`](https://github.com/sjawhar/pivot/commit/6c54f8ddd07d6ad730df377495b541d5de57932e))


## v0.5.14 (2026-10-27)

### Performance Improvements

- **storage**: Stat unchanged directories instead of listing them when finding .pvt files
  ([#513](https://github.com/sjawhar/pivot/pull/513),
  [`7fbdfa3`](https://github.com/sjawhar/pivot/commit/7fbdfa31cbdcb01832a935f5a57335239db5ed57))


## v0.5.13 (2026-10-27)

### Bug Fixes

- **executor**: Send a stopped worker's SIGTERM to its main thread, where its stage runs
  ([#528](https://github.com/sjawhar/pivot/pull/528),
  [`e463210`](https://github.com/sjawhar/pivot/commit/e46321080a2a95a8dd803bb1753c0efd79e850b4))

- **track**: Restore a .pvt's file only inside the project
  ([#533](https://github.com/sjawhar/pivot/pull/533),
  [`361a6e8`](https://github.com/sjawhar/pivot/commit/361a6e876be9771de8ca642826e8756a1b885827))


## v0.5.12 (2026-10-27)

### Bug Fixes

- **cli**: Restore get targets from remote ([#522](https://github.com/sjawhar/pivot/pull/522),
  [`f3b8a3f`](https://github.com/sjawhar/pivot/commit/f3b8a3f9b01fb572711eec54f73ea38ea5090917))

### Continuous Integration

- Run no suite on a draft and cancel superseded pull-request runs
  ([#526](https://github.com/sjawhar/pivot/pull/526),
  [`7aba10b`](https://github.com/sjawhar/pivot/commit/7aba10b480c5bc52432f6c609ca8c20c299afc7a))

- Run the benchmark on a pull request only when it carries the benchmark label
  ([#531](https://github.com/sjawhar/pivot/pull/531),
  [`1d362ee`](https://github.com/sjawhar/pivot/commit/1d362eee9076a32bb22a0d5d8a335cf26a4c4c36))

### Testing

- Read watch outputs once so a rerunning stage cannot race the check
  ([#529](https://github.com/sjawhar/pivot/pull/529),
  [`25759f8`](https://github.com/sjawhar/pivot/commit/25759f8d43ede6bd9fe2ec7d38b56f8c2c05c5b8))

- Restore permissions a test removes so a rerun can delete its temporary tree
  ([#532](https://github.com/sjawhar/pivot/pull/532),
  [`8eae480`](https://github.com/sjawhar/pivot/commit/8eae48051222fc24800916de787fe3f7e61087d0))


## v0.5.11 (2026-10-27)

### Bug Fixes

- **storage**: Keep a backslash in a file or directory name instead of reading it as a separator
  ([#520](https://github.com/sjawhar/pivot/pull/520),
  [`0ab4b1f`](https://github.com/sjawhar/pivot/commit/0ab4b1f8d2ddd566d4a7e098a9452afd29e2d9cc))

### Testing

- Wait on what the serve and worker-termination tests check, not on timing
  ([#527](https://github.com/sjawhar/pivot/pull/527),
  [`efbd180`](https://github.com/sjawhar/pivot/commit/efbd180345a01048b62756d369be2f9e0b1bc832))


## v0.5.10 (2026-10-26)

### Bug Fixes

- **cli**: Keep pivot status --json output to one JSON document
  ([#517](https://github.com/sjawhar/pivot/pull/517),
  [`a9c6280`](https://github.com/sjawhar/pivot/commit/a9c62800d67f12e012b6e8d413e1ef161c1ba864))

### Build System

- **ruff**: Keep ruff's default excludes so a fixer never enters .venv
  ([#515](https://github.com/sjawhar/pivot/pull/515),
  [`d3bb6d5`](https://github.com/sjawhar/pivot/commit/d3bb6d569eda3f9f891f6d6ac0e798719da05176))

### Testing

- **benchmarks**: Find the measured session without racing /proc or killing session 0
  ([#524](https://github.com/sjawhar/pivot/pull/524),
  [`959c922`](https://github.com/sjawhar/pivot/commit/959c922681d5814505960cdd8497be8614dd3a87))


## v0.5.9 (2026-10-25)

### Bug Fixes

- **engine**: A watch reload on loky 3.7 stops the previous run's workers, where 0.5.8 kept them
  alive beside the new ones ([#502](https://github.com/sjawhar/pivot/pull/502),
  [`d770e2e`](https://github.com/sjawhar/pivot/commit/d770e2ef074119e3731e8badbddf3eb9e9398e74))

- **executor**: Idle workers of a one-shot run exit after a 10-second idle timeout, so a run that
  waits on a long stage no longer keeps every worker it started alive
  ([#502](https://github.com/sjawhar/pivot/pull/502),
  [`d770e2e`](https://github.com/sjawhar/pivot/commit/d770e2ef074119e3731e8badbddf3eb9e9398e74))

- **executor**: Reclaim idle one-shot workers ([#502](https://github.com/sjawhar/pivot/pull/502),
  [`d770e2e`](https://github.com/sjawhar/pivot/commit/d770e2ef074119e3731e8badbddf3eb9e9398e74))

### Continuous Integration

- **bench**: Accept either stop status when two stop signals arrive together
  ([#511](https://github.com/sjawhar/pivot/pull/511),
  [`825fa47`](https://github.com/sjawhar/pivot/commit/825fa474b9fe3434bdb70410e7d9fc577b60f91f))

- **bench**: Accept either stop status when two stop signals arrive together, so the benchmark
  harness test no longer fails when both signals are pending
  ([#511](https://github.com/sjawhar/pivot/pull/511),
  [`825fa47`](https://github.com/sjawhar/pivot/commit/825fa474b9fe3434bdb70410e7d9fc577b60f91f))

### Performance Improvements

- **engine**: A run's first submission sizes its workers by replaying its schedule within resource
  limits, so four stages under local_grade: 2 start two workers, not four
  ([#502](https://github.com/sjawhar/pivot/pull/502),
  [`d770e2e`](https://github.com/sjawhar/pivot/commit/d770e2ef074119e3731e8badbddf3eb9e9398e74))


## v0.5.8 (2026-10-23)

### Bug Fixes

- **cli**: An absolute path typed through a symlink to the project root names the file it points to
  in `diff`, `checkout`, `repro`, `dag`, `push`, `fetch`, `pull` and `track`, instead of matching
  nothing or being refused as outside the project
  ([#500](https://github.com/sjawhar/pivot/pull/500),
  [`b24d007`](https://github.com/sjawhar/pivot/commit/b24d007a68f774e12c61e2534a50f2da8eb68835))

- **compose**: After upgrading, stages with outputs rerun once because output tags are part of their
  fingerprint; stages with no outputs remain cached
  ([#500](https://github.com/sjawhar/pivot/pull/500),
  [`b24d007`](https://github.com/sjawhar/pivot/commit/b24d007a68f774e12c61e2534a50f2da8eb68835))

- **compose**: Harden directory output handling ([#500](https://github.com/sjawhar/pivot/pull/500),
  [`b24d007`](https://github.com/sjawhar/pivot/commit/b24d007a68f774e12c61e2534a50f2da8eb68835))

- **diff**: A stage whose output path changes with its loader, such as `loaders.CSV()` to
  `loaders.DataFrameJSONL()`, no longer exits 1 with `ValueError('Invalid JSON …')`
  ([#500](https://github.com/sjawhar/pivot/pull/500),
  [`b24d007`](https://github.com/sjawhar/pivot/commit/b24d007a68f774e12c61e2534a50f2da8eb68835))

- **diff**: An entry whose git HEAD content is not in the local cache is reported with unknown
  counts and a `pivot fetch` warning instead of aborting the diff, and `--json` entries carry
  `change_type` ([#500](https://github.com/sjawhar/pivot/pull/500),
  [`b24d007`](https://github.com/sjawhar/pivot/commit/b24d007a68f774e12c61e2534a50f2da8eb68835))

- **diff**: Key git HEAD and the workspace by path, so an output that moves is removed at its HEAD
  path and added at its new one, and a path whose kind changed is named with both kinds and not
  compared ([#500](https://github.com/sjawhar/pivot/pull/500),
  [`b24d007`](https://github.com/sjawhar/pivot/commit/b24d007a68f774e12c61e2534a50f2da8eb68835))

- **diff**: One selection decides, for git HEAD and the workspace, which stage outputs and directory
  members each target names, and a target that names no CSV, JSON or JSONL file is named in a
  warning ([#500](https://github.com/sjawhar/pivot/pull/500),
  [`b24d007`](https://github.com/sjawhar/pivot/commit/b24d007a68f774e12c61e2534a50f2da8eb68835))

- **rpc**: `diff_output` answers a modified file whose two sides are cached instead of dropping the
  connection on numpy or datetime values, and refuses a request naming neither hash
  ([#500](https://github.com/sjawhar/pivot/pull/500),
  [`b24d007`](https://github.com/sjawhar/pivot/commit/b24d007a68f774e12c61e2534a50f2da8eb68835))

- **storage**: A lock records and reads back every directory member name it can hold, including
  spaces and format characters such as a macOS screenshot's; a control character is recorded with a
  warning that a later release refuses it, and a backslash fails the stage naming the file
  ([#500](https://github.com/sjawhar/pivot/pull/500),
  [`b24d007`](https://github.com/sjawhar/pivot/commit/b24d007a68f774e12c61e2534a50f2da8eb68835))

- **storage**: Refuse committed directory manifests with unsafe paths before `pivot diff` resolves
  them into workspace files ([#500](https://github.com/sjawhar/pivot/pull/500),
  [`b24d007`](https://github.com/sjawhar/pivot/commit/b24d007a68f774e12c61e2534a50f2da8eb68835))


## v0.5.7 (2026-10-23)

### Bug Fixes

- **cache**: `--checkout-mode hardlink`, and a `cache.checkout_mode` order without copy, now copy a
  file whose cache object is at the filesystem's hard-link limit in checkout, pull and repro,
  instead of failing with `[Errno 31] Too many links`
  ([#510](https://github.com/sjawhar/pivot/pull/510),
  [`7faf1e7`](https://github.com/sjawhar/pivot/commit/7faf1e7920dad13f50516ceac9d845a1a53cbc75))

- **cache**: Checkout, get, pull, repro's restore and a stage's own outputs copy a file whose cache
  object already has as many hard links as the filesystem allows (65,000 on ext4), instead of
  symlinking it into the cache; other files still hardlink, and a cache on another device still gets
  symlinks as before ([#510](https://github.com/sjawhar/pivot/pull/510),
  [`7faf1e7`](https://github.com/sjawhar/pivot/commit/7faf1e7920dad13f50516ceac9d845a1a53cbc75))

- **cache**: Copy a cache object that has reached the filesystem's hard-link limit instead of
  linking it ([#510](https://github.com/sjawhar/pivot/pull/510),
  [`7faf1e7`](https://github.com/sjawhar/pivot/commit/7faf1e7920dad13f50516ceac9d845a1a53cbc75))


## v0.5.6 (2026-10-23)

### Bug Fixes

- **fingerprint**: A result marked as not reusable is never restored from the run cache, and pivot
  commit keeps the mark ([#496](https://github.com/sjawhar/pivot/pull/496),
  [`23a7d5e`](https://github.com/sjawhar/pivot/commit/23a7d5ee1d2faa368ebf63184214c6e900e25696))

- **fingerprint**: A stage reruns when a method of a user class it uses changes, including dataclass
  methods ([#496](https://github.com/sjawhar/pivot/pull/496),
  [`23a7d5e`](https://github.com/sjawhar/pivot/commit/23a7d5ee1d2faa368ebf63184214c6e900e25696))

- **fingerprint**: A stage reruns when a user module imported inside the stage body changes, whether
  the import is absolute, relative or inside a function
  ([#496](https://github.com/sjawhar/pivot/pull/496),
  [`23a7d5e`](https://github.com/sjawhar/pivot/commit/23a7d5ee1d2faa368ebf63184214c6e900e25696))

- **fingerprint**: A stage whose run imports a user module its fingerprint does not cover warns
  once, naming the module, and is not reused ([#496](https://github.com/sjawhar/pivot/pull/496),
  [`23a7d5e`](https://github.com/sjawhar/pivot/commit/23a7d5ee1d2faa368ebf63184214c6e900e25696))

- **fingerprint**: After upgrading, each stage whose fingerprint now covers more code, such as one
  that uses a user class or imports a user module in its body, reruns once; later runs reuse results
  as before ([#496](https://github.com/sjawhar/pivot/pull/496),
  [`23a7d5e`](https://github.com/sjawhar/pivot/commit/23a7d5ee1d2faa368ebf63184214c6e900e25696))

- **fingerprint**: Imports made by a covered import, on any thread it starts, count as covered;
  threads that pivot starts never do ([#496](https://github.com/sjawhar/pivot/pull/496),
  [`23a7d5e`](https://github.com/sjawhar/pivot/commit/23a7d5ee1d2faa368ebf63184214c6e900e25696))

- **fingerprint**: Refuse at definition a stage whose body imports a user module that is not loaded,
  with the import to add ([#496](https://github.com/sjawhar/pivot/pull/496),
  [`23a7d5e`](https://github.com/sjawhar/pivot/commit/23a7d5ee1d2faa368ebf63184214c6e900e25696))

- **fingerprint**: Track user-class methods and in-body imports, refuse dynamic imports, keep unsafe
  results out of the run cache ([#496](https://github.com/sjawhar/pivot/pull/496),
  [`23a7d5e`](https://github.com/sjawhar/pivot/commit/23a7d5ee1d2faa368ebf63184214c6e900e25696))

### Continuous Integration

- Benchmark wall time and CPU against main on every pull request
  ([#501](https://github.com/sjawhar/pivot/pull/501),
  [`600ad58`](https://github.com/sjawhar/pivot/commit/600ad58cd27abd30b39106ec39d459d7839cc5b6))

- **bench**: An interrupted run.py stops the measured pivot run, sweeps its session and restores the
  caller's signal mask ([#501](https://github.com/sjawhar/pivot/pull/501),
  [`600ad58`](https://github.com/sjawhar/pivot/commit/600ad58cd27abd30b39106ec39d459d7839cc5b6))

- **bench**: Derive thresholds from recorded A/A runs with a 2% floor, and exit 2 on any input the
  comparator cannot read ([#501](https://github.com/sjawhar/pivot/pull/501),
  [`600ad58`](https://github.com/sjawhar/pivot/commit/600ad58cd27abd30b39106ec39d459d7839cc5b6))

- **bench**: Measure each pull request's merge commit against its first parent, for wall time and
  CPU at pipeline and task scope ([#501](https://github.com/sjawhar/pivot/pull/501),
  [`600ad58`](https://github.com/sjawhar/pivot/commit/600ad58cd27abd30b39106ec39d459d7839cc5b6))

- **bench**: Report a regression only when the median change is over the calibrated threshold and a
  paired bootstrap interval lies wholly above 1 ([#501](https://github.com/sjawhar/pivot/pull/501),
  [`600ad58`](https://github.com/sjawhar/pivot/commit/600ad58cd27abd30b39106ec39d459d7839cc5b6))


## v0.5.5 (2026-10-22)

### Bug Fixes

- **watch**: A reload that adds a stage whose input is missing reports the pipeline invalid and
  keeps watching, instead of ending the session ([#509](https://github.com/sjawhar/pivot/pull/509),
  [`e0f8f6e`](https://github.com/sjawhar/pivot/commit/e0f8f6e7590f5ea559816687c9d2102cb28e4ce7))

- **watch**: Creating the missing input, including inside a new directory or through a symlink in
  the project, runs the stopped stages once ([#509](https://github.com/sjawhar/pivot/pull/509),
  [`e0f8f6e`](https://github.com/sjawhar/pivot/commit/e0f8f6e7590f5ea559816687c9d2102cb28e4ce7))

- **watch**: Keep watching when a reload adds a stage whose input is missing
  ([#509](https://github.com/sjawhar/pivot/pull/509),
  [`e0f8f6e`](https://github.com/sjawhar/pivot/commit/e0f8f6e7590f5ea559816687c9d2102cb28e4ce7))

- **watch**: While an input is missing, only a change at that input or a directory above it
  re-checks the pipeline; other changes are handled as before
  ([#509](https://github.com/sjawhar/pivot/pull/509),
  [`e0f8f6e`](https://github.com/sjawhar/pivot/commit/e0f8f6e7590f5ea559816687c9d2102cb28e4ce7))


## v0.5.4 (2026-10-22)

### Bug Fixes

- Serve and TUI at any project path; run the suite without uv on PATH
  ([#493](https://github.com/sjawhar/pivot/pull/493),
  [`a17c062`](https://github.com/sjawhar/pivot/commit/a17c0626ce8c695717dfbb7cbf0249c3ab5b7998))

- **rpc**: A stopping serve removes only the socket it bound, and a cancelled bind still cleans up
  ([#493](https://github.com/sjawhar/pivot/pull/493),
  [`a17c062`](https://github.com/sjawhar/pivot/commit/a17c0626ce8c695717dfbb7cbf0249c3ab5b7998))

- **serve**: A worker stopped by the same group stop is not reported as a failed stage
  ([#493](https://github.com/sjawhar/pivot/pull/493),
  [`a17c062`](https://github.com/sjawhar/pivot/commit/a17c0626ce8c695717dfbb7cbf0249c3ab5b7998))

- **serve**: The pivot command takes over a stop signal only when it is at its default, and leaves a
  launcher's disposition alone ([#493](https://github.com/sjawhar/pivot/pull/493),
  [`a17c062`](https://github.com/sjawhar/pivot/commit/a17c0626ce8c695717dfbb7cbf0249c3ab5b7998))

- **serve**: The pivot command's serve mode stops cleanly on SIGTERM, repeated SIGTERM or a
  process-group SIGTERM, and removes its socket ([#493](https://github.com/sjawhar/pivot/pull/493),
  [`a17c062`](https://github.com/sjawhar/pivot/commit/a17c0626ce8c695717dfbb7cbf0249c3ab5b7998))

- **socket**: Serve and the TUI work when the project's agent socket path is longer than a Unix
  socket address ([#493](https://github.com/sjawhar/pivot/pull/493),
  [`a17c062`](https://github.com/sjawhar/pivot/commit/a17c0626ce8c695717dfbb7cbf0249c3ab5b7998))

- **tui**: Pivot-tui carries its own long-path socket code, so its wheel works beside any pivot
  release ([#493](https://github.com/sjawhar/pivot/pull/493),
  [`a17c062`](https://github.com/sjawhar/pivot/commit/a17c0626ce8c695717dfbb7cbf0249c3ab5b7998))

### Testing

- Replace the help timing test with a check that help imports no heavy modules
  ([#493](https://github.com/sjawhar/pivot/pull/493),
  [`a17c062`](https://github.com/sjawhar/pivot/commit/a17c0626ce8c695717dfbb7cbf0249c3ab5b7998))

- Run the test suite without uv on PATH, and check it in CI
  ([#493](https://github.com/sjawhar/pivot/pull/493),
  [`a17c062`](https://github.com/sjawhar/pivot/commit/a17c0626ce8c695717dfbb7cbf0249c3ab5b7998))


## v0.5.3 (2026-10-22)

### Bug Fixes

- **engine**: Say a deferred reload covers a data change only if it succeeds
  ([#508](https://github.com/sjawhar/pivot/pull/508),
  [`2da40c2`](https://github.com/sjawhar/pivot/commit/2da40c2e2d5ca95e966b839fb6745ca0a16333aa))

- **watch**: Hand a batch's data to the data handler when its reload starts no re-run
  ([#508](https://github.com/sjawhar/pivot/pull/508),
  [`2da40c2`](https://github.com/sjawhar/pivot/commit/2da40c2e2d5ca95e966b839fb6745ca0a16333aa))

- **watch**: Run each stage once when one watch batch edits code and data
  ([#508](https://github.com/sjawhar/pivot/pull/508),
  [`2da40c2`](https://github.com/sjawhar/pivot/commit/2da40c2e2d5ca95e966b839fb6745ca0a16333aa))

### Chores

- Exclude .worktrees/ from basedpyright ([#506](https://github.com/sjawhar/pivot/pull/506),
  [`38238f8`](https://github.com/sjawhar/pivot/commit/38238f8f0e7baffdf4b663861cf14b657defee93))

- Ignore agent worktrees under .worktrees/ ([#506](https://github.com/sjawhar/pivot/pull/506),
  [`38238f8`](https://github.com/sjawhar/pivot/commit/38238f8f0e7baffdf4b663861cf14b657defee93))

### Continuous Integration

- Hold releases while main carries a known regression
  ([#505](https://github.com/sjawhar/pivot/pull/505),
  [`cf66ba9`](https://github.com/sjawhar/pivot/commit/cf66ba99c6234037f6e64a1fec79a1ecf1e5d26a))

- Run known-regressions on pushes to main only, as intended
  ([#505](https://github.com/sjawhar/pivot/pull/505),
  [`cf66ba9`](https://github.com/sjawhar/pivot/commit/cf66ba99c6234037f6e64a1fec79a1ecf1e5d26a))

- The known-regressions header asks for the fix's PR only once one is open
  ([#505](https://github.com/sjawhar/pivot/pull/505),
  [`cf66ba9`](https://github.com/sjawhar/pivot/commit/cf66ba99c6234037f6e64a1fec79a1ecf1e5d26a))

### Documentation

- **engine**: Define CodeOrConfigChanged.data_paths without the handler policy
  ([#508](https://github.com/sjawhar/pivot/pull/508),
  [`2da40c2`](https://github.com/sjawhar/pivot/commit/2da40c2e2d5ca95e966b839fb6745ca0a16333aa))

- **watch**: Describe the no-stages branch of a batch's data paths
  ([#508](https://github.com/sjawhar/pivot/pull/508),
  [`2da40c2`](https://github.com/sjawhar/pivot/commit/2da40c2e2d5ca95e966b839fb6745ca0a16333aa))

### Refactoring

- **engine**: State the reloads that start no re-run in one docstring
  ([#508](https://github.com/sjawhar/pivot/pull/508),
  [`2da40c2`](https://github.com/sjawhar/pivot/commit/2da40c2e2d5ca95e966b839fb6745ca0a16333aa))

### Testing

- **engine**: Cover a code change's batch data while a run is in progress
  ([#508](https://github.com/sjawhar/pivot/pull/508),
  [`2da40c2`](https://github.com/sjawhar/pivot/commit/2da40c2e2d5ca95e966b839fb6745ca0a16333aa))


## v0.5.2 (2026-10-20)

### Bug Fixes

- **cli**: Ignore third-party warning conditions ([#497](https://github.com/sjawhar/pivot/pull/497),
  [`88596c4`](https://github.com/sjawhar/pivot/commit/88596c45af80c5f763363bcd597b843eef41c23c))

- **cli**: Make TUI warning replay atomic and preserve current errors
  ([#497](https://github.com/sjawhar/pivot/pull/497),
  [`88596c4`](https://github.com/sjawhar/pivot/commit/88596c45af80c5f763363bcd597b843eef41c23c))

- **cli**: Print each TUI-time warning once, across watch runs and thread orders
  ([#497](https://github.com/sjawhar/pivot/pull/497),
  [`88596c4`](https://github.com/sjawhar/pivot/commit/88596c45af80c5f763363bcd597b843eef41c23c))

- **cli**: Print each warning a TUI run logs once, after the TUI exits
  ([#497](https://github.com/sjawhar/pivot/pull/497),
  [`88596c4`](https://github.com/sjawhar/pivot/commit/88596c45af80c5f763363bcd597b843eef41c23c))

### Refactoring

- **cli**: Simplify TUI warning deferral ([#497](https://github.com/sjawhar/pivot/pull/497),
  [`88596c4`](https://github.com/sjawhar/pivot/commit/88596c45af80c5f763363bcd597b843eef41c23c))

### Testing

- **cli**: Restore O3 engine-state regression lock
  ([#497](https://github.com/sjawhar/pivot/pull/497),
  [`88596c4`](https://github.com/sjawhar/pivot/commit/88596c45af80c5f763363bcd597b843eef41c23c))


## v0.5.1 (2026-10-20)

### Bug Fixes

- **fingerprint**: Keep the user's part of a namespace package user code
  ([#499](https://github.com/sjawhar/pivot/pull/499),
  [`58c7384`](https://github.com/sjawhar/pivot/commit/58c73847c645b3b0a152f022137eb231a4d9d2dd))


## v0.5.0 (2026-10-20)

### Features

- **remote**: Gs:// remotes through a RemoteStorage protocol
  ([#498](https://github.com/sjawhar/pivot/pull/498),
  [`f98e332`](https://github.com/sjawhar/pivot/commit/f98e332061e5dff39687e457a4ad4f99d6c1280c))


## v0.4.1 (2026-10-19)

### Bug Fixes

- **discovery**: Treat pivot's own packages as library code when installed from a checkout
  ([#491](https://github.com/sjawhar/pivot/pull/491),
  [`47ab55c`](https://github.com/sjawhar/pivot/commit/47ab55c3d40fdaca333d66590c9cf4c5fcf7bf34))

### Documentation

- **agents**: Removing something users reach needs sjawhar's approval
  ([#492](https://github.com/sjawhar/pivot/pull/492),
  [`4ce2069`](https://github.com/sjawhar/pivot/commit/4ce2069a96fb678e2cd40d92b499be5bb5ccdc7b))

- **fingerprint**: Name the setuptools layouts that keep pivot_tui user code
  ([#491](https://github.com/sjawhar/pivot/pull/491),
  [`47ab55c`](https://github.com/sjawhar/pivot/commit/47ab55c3d40fdaca333d66590c9cf4c5fcf7bf34))

- **fingerprint**: Say which pivot-tui installs count as pivot's own
  ([#491](https://github.com/sjawhar/pivot/pull/491),
  [`47ab55c`](https://github.com/sjawhar/pivot/commit/47ab55c3d40fdaca333d66590c9cf4c5fcf7bf34))

### Testing

- **engine**: Check the pool size against stage-DAG order and every step budget
  ([#494](https://github.com/sjawhar/pivot/pull/494),
  [`11567a3`](https://github.com/sjawhar/pivot/commit/11567a34517546c0db6fa17a958e8ce6bfe46427))

- **engine**: Sweep step budgets until every order finishes, not to a fixed count
  ([#494](https://github.com/sjawhar/pivot/pull/494),
  [`11567a3`](https://github.com/sjawhar/pivot/commit/11567a34517546c0db6fa17a958e8ce6bfe46427))


## v0.4.0 (2026-10-19)

### Features

- **compose**: Restore directory outputs that a stage returns as a dict of files
  ([#489](https://github.com/sjawhar/pivot/pull/489),
  [`961e174`](https://github.com/sjawhar/pivot/commit/961e1742ac20fcee71ef17cba3db47924a4b2a2e))


## v0.3.11 (2026-10-19)

### Bug Fixes

- **cli**: Pre-warm the TUI's pool over the stages the engine runs
  ([#488](https://github.com/sjawhar/pivot/pull/488),
  [`014ab27`](https://github.com/sjawhar/pivot/commit/014ab2766756218b2e9f5ab0529120cbe26194aa))

- **engine**: Bound the width search so sizing the pool stays fast below the cap
  ([#488](https://github.com/sjawhar/pivot/pull/488),
  [`014ab27`](https://github.com/sjawhar/pivot/commit/014ab2766756218b2e9f5ab0529120cbe26194aa))

- **engine**: Restart and pre-warm the pool at the size the run uses, and stop sizing it at the cap
  ([#488](https://github.com/sjawhar/pivot/pull/488),
  [`014ab27`](https://github.com/sjawhar/pivot/commit/014ab2766756218b2e9f5ab0529120cbe26194aa))

- **engine**: Size the pool the same whatever order the run's stages come in
  ([#488](https://github.com/sjawhar/pivot/pull/488),
  [`014ab27`](https://github.com/sjawhar/pivot/commit/014ab2766756218b2e9f5ab0529120cbe26194aa))

- **engine**: Size the worker pool to the pipeline's width so a chain reuses one warm worker
  ([#488](https://github.com/sjawhar/pivot/pull/488),
  [`014ab27`](https://github.com/sjawhar/pivot/commit/014ab2766756218b2e9f5ab0529120cbe26194aa))

### Documentation

- Keep stage settings in StageParams; say DirectoryOut's return is tracked
  ([#490](https://github.com/sjawhar/pivot/pull/490),
  [`9f02ec1`](https://github.com/sjawhar/pivot/commit/9f02ec191c3cdfc096a59cf3bd5a8dd6a476a31e))

### Testing

- **engine**: Drop the Hypothesis order test that cannot fail
  ([#488](https://github.com/sjawhar/pivot/pull/488),
  [`014ab27`](https://github.com/sjawhar/pivot/commit/014ab2766756218b2e9f5ab0529120cbe26194aa))


## v0.3.10 (2026-10-18)

### Bug Fixes

- **engine**: Decide a cached stage's skip under shared locks on its outputs
  ([#487](https://github.com/sjawhar/pivot/pull/487),
  [`1c2d1c2`](https://github.com/sjawhar/pivot/commit/1c2d1c27961dad54dd6e4afcc0261ec372cf7c54))


## v0.3.9 (2026-10-18)

### Bug Fixes

- **discovery**: Load all user code through the checked loader, never the stdlib or a library
  ([#482](https://github.com/sjawhar/pivot/pull/482),
  [`d0a5ad2`](https://github.com/sjawhar/pivot/commit/d0a5ad2bb17c88458aec657c03bbd2904bf971cb))

- **discovery**: Load and fingerprint an edit to user code even when it keeps the file's size and
  mtime ([#482](https://github.com/sjawhar/pivot/pull/482),
  [`d0a5ad2`](https://github.com/sjawhar/pivot/commit/d0a5ad2bb17c88458aec657c03bbd2904bf971cb))

- **discovery**: Load edited user code in workers, keep never-mode bytecode current, leave symlinked
  libraries alone ([#482](https://github.com/sjawhar/pivot/pull/482),
  [`d0a5ad2`](https://github.com/sjawhar/pivot/commit/d0a5ad2bb17c88458aec657c03bbd2904bf971cb))

- **discovery**: Load project code from bytecode checked by content, not mtime
  ([#482](https://github.com/sjawhar/pivot/pull/482),
  [`d0a5ad2`](https://github.com/sjawhar/pivot/commit/d0a5ad2bb17c88458aec657c03bbd2904bf971cb))

- **fingerprint**: Key cached function hashes and manifest sources by file content
  ([#482](https://github.com/sjawhar/pivot/pull/482),
  [`d0a5ad2`](https://github.com/sjawhar/pivot/commit/d0a5ad2bb17c88458aec657c03bbd2904bf971cb))

### Documentation

- **outputs**: Say that restoring a lock with no empty directories onto an empty path leaves them
  out ([#484](https://github.com/sjawhar/pivot/pull/484),
  [`aa33a3a`](https://github.com/sjawhar/pivot/commit/aa33a3ac8cd46426a498e1826b96052e91d356e0))

- **outputs**: Say what pivot does with a lock that records no empty directories, and how to repair
  a stripped output ([#484](https://github.com/sjawhar/pivot/pull/484),
  [`aa33a3a`](https://github.com/sjawhar/pivot/commit/aa33a3ac8cd46426a498e1826b96052e91d356e0))

- **outputs**: Say when pivot runs a stage over unrecorded empty directories, and how to repair a
  stripped one ([#484](https://github.com/sjawhar/pivot/pull/484),
  [`aa33a3a`](https://github.com/sjawhar/pivot/commit/aa33a3ac8cd46426a498e1826b96052e91d356e0))


## v0.3.8 (2026-10-18)

### Bug Fixes

- **engine**: Let an interrupted pivot exit when a worker outlives the grace
  ([#486](https://github.com/sjawhar/pivot/pull/486),
  [`098d8b2`](https://github.com/sjawhar/pivot/commit/098d8b2d6fcd7b012da41a8c43995abfae8cd846))

- **executor**: Keep every worker's cleanup alive on Ctrl-C, kill stage children, bound parent death
  ([#486](https://github.com/sjawhar/pivot/pull/486),
  [`098d8b2`](https://github.com/sjawhar/pivot/commit/098d8b2d6fcd7b012da41a8c43995abfae8cd846))

- **executor**: Let a stopped worker's running stage unwind before the worker exits
  ([#486](https://github.com/sjawhar/pivot/pull/486),
  [`098d8b2`](https://github.com/sjawhar/pivot/commit/098d8b2d6fcd7b012da41a8c43995abfae8cd846))


## v0.3.7 (2026-10-18)

### Bug Fixes

- **engine**: Compare user-code stamps with stamps, never with a clock
  ([#485](https://github.com/sjawhar/pivot/pull/485),
  [`ac80b6f`](https://github.com/sjawhar/pivot/commit/ac80b6f8fd9e9e2be4f88e8c1dc085c11c76b5eb))

- **engine**: Drop the unused #482 initializer shim and allow file-clock slack in the staleness
  guard ([#485](https://github.com/sjawhar/pivot/pull/485),
  [`ac80b6f`](https://github.com/sjawhar/pivot/commit/ac80b6f8fd9e9e2be4f88e8c1dc085c11c76b5eb))

- **engine**: Exit the output-queue manager when pivot dies
  ([#485](https://github.com/sjawhar/pivot/pull/485),
  [`ac80b6f`](https://github.com/sjawhar/pivot/commit/ac80b6f8fd9e9e2be4f88e8c1dc085c11c76b5eb))

- **engine**: Keep the pool usable after a failed manager start; let EMFILE propagate
  ([#485](https://github.com/sjawhar/pivot/pull/485),
  [`ac80b6f`](https://github.com/sjawhar/pivot/commit/ac80b6f8fd9e9e2be4f88e8c1dc085c11c76b5eb))

- **engine**: Keep warm workers across runs and restart them when user code changes
  ([#485](https://github.com/sjawhar/pivot/pull/485),
  [`ac80b6f`](https://github.com/sjawhar/pivot/commit/ac80b6f8fd9e9e2be4f88e8c1dc085c11c76b5eb))

- **engine**: Preserve warm capacity and child cleanup
  ([#485](https://github.com/sjawhar/pivot/pull/485),
  [`ac80b6f`](https://github.com/sjawhar/pivot/commit/ac80b6f8fd9e9e2be4f88e8c1dc085c11c76b5eb))

- **engine**: Replace an output-queue manager that died between runs
  ([#485](https://github.com/sjawhar/pivot/pull/485),
  [`ac80b6f`](https://github.com/sjawhar/pivot/commit/ac80b6f8fd9e9e2be4f88e8c1dc085c11c76b5eb))

- **engine**: Replace an output-queue manager whose socket is gone
  ([#485](https://github.com/sjawhar/pivot/pull/485),
  [`ac80b6f`](https://github.com/sjawhar/pivot/commit/ac80b6f8fd9e9e2be4f88e8c1dc085c11c76b5eb))


## v0.3.6 (2026-10-17)

### Bug Fixes

- **executor**: Run a stage whose old lock cannot record the empty directories its output holds
  ([#483](https://github.com/sjawhar/pivot/pull/483),
  [`8bb8e4f`](https://github.com/sjawhar/pivot/commit/8bb8e4f4ec80a44e4edb1aeb3a21fe63449bda31))

### Documentation

- **storage**: State where flock leaves temporary entries unprotected; pin the vanished-temp test
  ([#481](https://github.com/sjawhar/pivot/pull/481),
  [`acf266b`](https://github.com/sjawhar/pivot/commit/acf266b63027e7995e4f15e78103ef765367706b))


## v0.3.5 (2026-10-17)

### Bug Fixes

- **storage**: Never change a published cache object; reclaim killed tree publishers' copies
  ([#480](https://github.com/sjawhar/pivot/pull/480),
  [`a7b286c`](https://github.com/sjawhar/pivot/commit/a7b286c0b8c186a9348e5032823c8669acdfd7bb))

- **storage**: Shared-cache lock files, serialised tree repair, sweep every cache temporary
  ([#480](https://github.com/sjawhar/pivot/pull/480),
  [`a7b286c`](https://github.com/sjawhar/pivot/commit/a7b286c0b8c186a9348e5032823c8669acdfd7bb))

- **storage**: Stage without the shared lock where flock is refused; check a tree before marking it
  ([#480](https://github.com/sjawhar/pivot/pull/480),
  [`a7b286c`](https://github.com/sjawhar/pivot/commit/a7b286c0b8c186a9348e5032823c8669acdfd7bb))

- **storage**: Stop parallel stages committing the same new bytes from failing with ENOENT
  ([#480](https://github.com/sjawhar/pivot/pull/480),
  [`a7b286c`](https://github.com/sjawhar/pivot/commit/a7b286c0b8c186a9348e5032823c8669acdfd7bb))

- **storage**: Sweep killed tree publishers' copies under a prefix flock; repair unmarked trees
  ([#480](https://github.com/sjawhar/pivot/pull/480),
  [`a7b286c`](https://github.com/sjawhar/pivot/commit/a7b286c0b8c186a9348e5032823c8669acdfd7bb))

### Testing

- **storage**: Cover copy_to_cache on a filesystem without hard links
  ([#480](https://github.com/sjawhar/pivot/pull/480),
  [`a7b286c`](https://github.com/sjawhar/pivot/commit/a7b286c0b8c186a9348e5032823c8669acdfd7bb))

- **storage**: Cover parallel commits of the same new cache object
  ([#480](https://github.com/sjawhar/pivot/pull/480),
  [`a7b286c`](https://github.com/sjawhar/pivot/commit/a7b286c0b8c186a9348e5032823c8669acdfd7bb))


## v0.3.4 (2026-10-17)

### Bug Fixes

- **storage**: Honor 0.3.3 directory hashes in checkout; check a cache tree's layout once
  ([#478](https://github.com/sjawhar/pivot/pull/478),
  [`36c883d`](https://github.com/sjawhar/pivot/commit/36c883dc413dd66e87d8310154ca68c2adc69e62))

- **storage**: Keep a 0.3.3 directory's empty dirs through --only-missing; best-effort layout marker
  ([#478](https://github.com/sjawhar/pivot/pull/478),
  [`36c883d`](https://github.com/sjawhar/pivot/commit/36c883dc413dd66e87d8310154ca68c2adc69e62))

- **storage**: Keep user symlinks out of the SYMLINK cache tree
  ([#478](https://github.com/sjawhar/pivot/pull/478),
  [`36c883d`](https://github.com/sjawhar/pivot/commit/36c883dc413dd66e87d8310154ca68c2adc69e62))

- **storage**: Restore a directory output's empty directories
  ([#478](https://github.com/sjawhar/pivot/pull/478),
  [`36c883d`](https://github.com/sjawhar/pivot/commit/36c883dc413dd66e87d8310154ca68c2adc69e62))


## v0.3.3 (2026-10-16)

### Bug Fixes

- **storage**: Add the missing path argument to the track --force tip
  ([#477](https://github.com/sjawhar/pivot/pull/477),
  [`a223886`](https://github.com/sjawhar/pivot/commit/a223886017411c54614dac7302d83ba95feb3d0d))

- **storage**: Close path traversal and repair gaps from re-review
  ([#477](https://github.com/sjawhar/pivot/pull/477),
  [`a223886`](https://github.com/sjawhar/pivot/commit/a223886017411c54614dac7302d83ba95feb3d0d))

- **storage**: Close three review gaps in the exec-bit checkout guard
  ([#477](https://github.com/sjawhar/pivot/pull/477),
  [`a223886`](https://github.com/sjawhar/pivot/commit/a223886017411c54614dac7302d83ba95feb3d0d))

- **storage**: Guard checkout against stale or colliding cache-object permissions
  ([#477](https://github.com/sjawhar/pivot/pull/477),
  [`a223886`](https://github.com/sjawhar/pivot/commit/a223886017411c54614dac7302d83ba95feb3d0d))

- **storage**: Preserve a directory output's executable bit through checkout
  ([#477](https://github.com/sjawhar/pivot/pull/477),
  [`a223886`](https://github.com/sjawhar/pivot/commit/a223886017411c54614dac7302d83ba95feb3d0d))

### Testing

- **storage**: Cover directory-output executable bit through checkout
  ([#477](https://github.com/sjawhar/pivot/pull/477),
  [`a223886`](https://github.com/sjawhar/pivot/commit/a223886017411c54614dac7302d83ba95feb3d0d))


## v0.3.2 (2026-10-16)

### Bug Fixes

- **checkout**: Preserve outputs on cache misses ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

- **checkout**: Retain tracked directory ignore semantics
  ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

- **cli**: Honour --quiet for the left-out symlink warning in track and commit
  ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

- **storage**: Atomically restore get fallbacks ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

- **storage**: Count only links into the configured cache as directory content
  ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

- **storage**: Hash cache-derived directory links
  ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

- **storage**: Hash produced directories consistently
  ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

- **storage**: Keep checkout and get from losing files; count only Pivot cache links in directory
  artifacts; warn on left-out symlinks ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

- **storage**: Preserve complete directory artifacts
  ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

- **storage**: Recover interrupted get staging files
  ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

- **storage**: Refuse user symlinks in directory artifacts on top of main
  ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

- **storage**: Warn instead of refusing symlinks in directory artifacts
  ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

### Documentation

- Clarify directory artifact symlink rules ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))

### Testing

- **storage**: Keep main's symlink-skip test unchanged
  ([#476](https://github.com/sjawhar/pivot/pull/476),
  [`c941ad8`](https://github.com/sjawhar/pivot/commit/c941ad8b371c9ed7cf18822be1613ed91fc7d79b))


## v0.3.1 (2026-10-15)

### Bug Fixes

- Params comparison, cached-run lock refresh, watch routing for enclosing and aliased inputs,
  explain and doctor output ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

- **cli**: Display falsy explain values ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

- **doctor**: Name the invalid remote entry ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

- **doctor**: Report invalid remote configuration
  ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

- **engine**: Refresh stage lock after artifact wait
  ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

- **skip**: Compare params canonically ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

- **watch**: Match symlinked input paths ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

- **watch**: Preserve output path semantics ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

- **watch**: Recognize symlinked output parents ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

- **watch**: Rerun all enclosing input consumers ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

### Chores

- Format post-skip regression coverage ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

### Performance Improvements

- **watch**: Index resolved input aliases ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

### Testing

- **watch**: Assert output routing for either watcher spelling
  ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))

- **watch**: Cover physical output watchfiles events
  ([#475](https://github.com/sjawhar/pivot/pull/475),
  [`6af2e32`](https://github.com/sjawhar/pivot/commit/6af2e322940e16cc7518c6c18fe00b70efb99649))


## v0.3.0 (2026-10-15)

### Bug Fixes

- **agent-rpc**: Scope normal disconnect handling to socket I/O
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **cache**: Hash and cache every file in a directory artifact
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **cli**: Clear stage manifests in fingerprint reset
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **cli**: Exit non-zero when the TUI crashes
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **cli**: Fail `pivot repro --tui --watch` when the engine crashes
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **cli**: Render grouped missing inputs with workspace paths
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **cli**: Stop the TUI event poller when the engine or TUI finishes
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **cli**: Suggest pivot repro, and pivot pull for a clone missing outputs (F1)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: Compare a stage's closure and default values by canonical text
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: Compare a stage's params by their canonical encoding
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: Compare input and output formats when one stage is reached twice
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: Complete incremental output typing
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: Name the return value in p.previous() and handle-lookup errors
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: Name the return value, not '_single', in output errors (F11)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: One stage reached twice means same definition and wiring, not same object
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: Refuse a pipeline input bound to another stage's output path
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: Refuse two pipelines with one name instead of sharing their stages
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: Reject incremental loaders without empty values
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: Reject two input names bound to one file at load (F12)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **deps**: Require pydantic 2.13, whose by-value pickled models validate
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **discovery**: --all skips a pipeline.py that re-exports another file's pipeline
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **discovery**: Pickle session-owned project modules by value for reusable workers
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **discovery**: Scope by-value modules to root-relative imports
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **discovery**: Scope worker by-value registration to root-relative modules
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **engine**: A params.yaml error during watch is an invalid pipeline
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **engine**: Preserve capacities for RPC runs
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **explain**: Name an edited or missing metric instead of 'Outputs missing from cache' (F6)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Aggregate nested capture errors
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Allow ArtifactHandle default arguments
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Allow library ufuncs that lack __module__
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Allow library-exported callable objects
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Apply the class attribute rule only to what a user class body wrote
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Avoid caching unsafe captures
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Avoid evaluating decorator owners
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Bump the cache schema version for StageParams extra="forbid"
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Check default argument values like captured module values
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Check every live value a cached manifest read
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Drop bare @stage decorators from stage code hashes
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: End a self-referencing value in a stable cycle token
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Hash a dataclass instance field by field
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Hash a default-repr object by its class and attributes
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Hash a function value by its name and code
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Hash a pydantic model class as a class
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Hash a secret outside a stage's params by its keyed digest
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Hash and record set params in a canonical order
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Key the manifest cache by stage function identity
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: List each unsafe-capture refusal once
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Name a function by its closure values and defaults
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Qualify portable dependency manifests
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Record library functions a stage reads by name
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Refuse a partial whose function is bound to an instance
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Refuse methods bound to an instance
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Refuse user callable instances bound in loader config and partials
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Reject non-function callable instances
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Resolve resource decorators exactly
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Resolve resource decorators semantically
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Satisfy type check for sys.modules scan
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Track the class attributes a stage reads
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Treat nonlocal modules as module dependencies
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Validate cached manifests against every project module
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Walk a custom loader's own load, save and empty
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Walk functions and classes bound in loader config and partials
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Walk the function behind functools.cache helpers
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **init**: Ignore the runtime files pivot creates under .pivot (F2, F17, F19)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **loaders**: Round-trip empty CSV incremental values
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **lock**: Require every recorded hash to be a cache key
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **lock**: Treat a lock whose fields have the wrong type as missing
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **outputs**: Never move a received dependency path
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **outputs**: Reject symlinked and directory-member path escapes
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **outputs**: Stages returning pathlib.Path hand pivot the file or directory they wrote
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: A params.yaml override reaches a field with a pydantic alias
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: Key secret digests with a per-project HMAC key
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: Mask secrets declared through a PEP 695 type alias in errors
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: Match params.yaml keys to the validated value, not to annotations
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: Never echo a secret's value in a params error
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: Only the process that creates the secret key appends to .gitignore
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: Record a secret param by a digest of its value
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: Refuse a params.yaml typo inside a nested plain BaseModel
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: Refuse a params.yaml value its StageParams field rejects at load
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: Refuse params.yaml keys that name no loaded stage
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: StageParams refuses unknown fields, from code or params.yaml
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **plots**: Write the gallery relative to the project root, not cwd
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **plots**: Write the gallery under .pivot/ with relative image paths (F5)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **registry**: Refuse outputs that resolve to one workspace path (F9)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **rpc**: Commit the engine's pipeline instead of reading it from Click context
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **rpc**: Refuse commit while the pipeline is invalid or the engine is starting
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **rpc**: Refuse run with -32004 while the engine is starting instead of waiting
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **rpc**: Refuse run with an error while the watched pipeline is invalid
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **rpc**: Send every artifact identity as a {producer, key} object
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **rpc**: Wait for the startup snapshot before deciding a run request
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **skip**: Compare recorded and current params by canonical JSON text
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **status**: Count downstream stages pull restores in a plain clone
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **status**: Count pullable stages from explain's reason, without re-hashing
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **status**: Don't suggest pivot pull for a metric-only output mismatch
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **status**: Don't suggest pivot pull for stages downstream of a stale stage
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **status**: Pull count passes through committed stages and keeps -q --json
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **storage**: Hash every directory artifact file
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **tui**: Read engine_state_changed's state field so watch status reaches the app
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **tui**: Rebuild artifact identities in explanations received over RPC
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **tui**: Send output_summary identities as {producer, key} objects
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **tui**: Send run and commit from Textual workers so slow replies never freeze the app
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **tui**: Show a closed engine connection as a notification, not a crash
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **tui**: Signal socket readiness only once the RPC server listens
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **verify**: Resolve lock dependency paths through the workspace store
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: A restart replays the cancelled run's stages too
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: A run request while a run is cancelling waits for it
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: An explicit cancel after a queued restart keeps the run cancelled
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Ctrl-C stops plain watch at once, even mid-reload
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Drop stale bytecode for changed sources before a reload
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Keep redirected stderr log records in their file
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Listen on the RPC socket while the startup fingerprint snapshot runs
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Print log records above the live console area
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Queue the startup run ahead of RPC run requests made during the snapshot
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Reload the edited pipeline off the event loop
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Report each watch run on its own in the console
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Report pipeline edits made before any run as modified stages
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Report why the pipeline is invalid, and show it in the TUI
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Run stale stages when watch mode starts
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Treat a pipeline as invalid whenever a reload rejects it
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Treat a stage fingerprint error as an invalid pipeline and wait for a fix
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Warn and keep reloading when stale bytecode cannot be removed
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **worker**: Run the stage outside the joblib ImportError handler (F3)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

### Code Style

- Apply formatter output after worker serialization tests
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Format incremental guard test
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Format incremental loader validation
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Format incremental review fixes
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **outputs**: Format PathOnly ownership check
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

### Continuous Integration

- Measure coverage on the pivot test row so fail_under = 90 binds
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

### Documentation

- Clarify immutable capture fingerprinting
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Clarify incremental empty loader requirement
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Explain compositional incremental outputs
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Remove legacy output plumbing references
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- State manifest cache validation rule
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **agents**: Structure listing matches the package (no pipeline/, dag/ renders, outputs.py holds
  DirectoryOut)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Name current framework classes in the is_user_code note
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Say why the project-module scan is a memo
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **outputs**: DirectoryOut docstring states that compose rejects it
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **params**: A container serializer can still put a secret in the record
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **pipelines**: Name fingerprint settings in the same-stage rule
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **plans**: Incremental outputs in the compositional API
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **readme**: Package README uses the compositional API
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **resources**: Document stage scheduling limits
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **skill**: Quote the error messages users actually see (F10)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **stage_def**: A subclass keeps extra="forbid" unless it sets extra
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **stage_def**: Describe directory-output keys without DirectoryOut
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **tui**: Describe the poller stop event and engine-crash shutdown
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Only RPC cancel is an explicit cancel in watch mode
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

### Features

- **compose**: Add incremental outputs
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **engine**: Schedule stages with resources
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

### Performance Improvements

- **fingerprint**: Name each function once per fingerprint pass
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Scan sys.modules once per fingerprint pass
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

### Refactoring

- Delete stale stage-function type docs and StageReturn
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Delete unreachable output trie
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Delete unused Console progress and stage output methods
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Delete unused IgnoreMatcher.filter_entries
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Delete unused is_tui_status_message and is_tui_log_message
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Delete unused legacy output classes Out, Metric and Plot
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Delete unused path_utils module
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Delete unused Pipeline.restore and StageRegistry.restore
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Delete unused project.try_resolve_path and to_absolute_path
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Delete unused StageRegistry.get_all_output_paths
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Delete unused StateDB.write_run_cache
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Delete unused write_run_cache_entry
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **exceptions**: Delete unused DVC compatibility errors
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **outputs**: Remove dead legacy output plumbing
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

### Testing

- Adapt to the integrated head (82886c4)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Give three lock fixtures valid hashes so each fails for its own reason
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Stop building the legacy output classes
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- Type incremental review regressions
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **cli**: Cover uncached incremental flag orthogonality
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **compose**: Cover uncached incremental artifact guard
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **coverage**: Measure code run in worker threads
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **deps**: Add hypothesis and mutmut to the pivot dev extra
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **discovery**: Isolate caller-imported stages and surface load errors
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **engine**: Allow worker spawn time in stop_accepting test
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **engine**: Type RPC capacity event
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **execution**: Add resource scheduling coverage
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **execution**: Strengthen resource dependency timing assertion
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **executor**: Cover incremental directory copy restoration
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **executor**: Cover normal incremental skip restoration
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **executor**: Preserve executable bits in incremental restores
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **executor**: Run(checkout_missing=True) restores a deleted tracked input or reports it uncached
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Compare params as the run-cache key serializes them
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Cover callable capture boundaries
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **fingerprint**: Hypothesis properties for fresh-process determinism, formatting insensitivity,
  one-token sensitivity and param overrides in the run-cache key
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **graph**: Retain missing dependency diagnostics for mock-only stores
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **graph**: Use a typed Store fake for missing input diagnostics
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **import**: Pivot import/update end to end against a local git source and moto S3; red test for
  Task 9
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **import**: Port restored import tests to identity-addressed imports
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **import**: Truncation behaviour of the available list; red test for Task 9
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **integration**: Fresh-venv wheel install smoke of init/repro/status/dag (CI only)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **integration**: Rendezvous the disjoint concurrent-run stages
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **lock**: Red tests for malformed lock shapes (Task 27)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **properties**: Hypothesis round-trips for loaders, lock files, identities, DAG order and input
  hash
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **resources**: Cover scheduler capacities
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **resources**: Rendezvous concurrent stages
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **skill**: Match each Common errors quote against the message it stands for
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **skip**: Kill surviving mutants for output-path, param add/remove and empty-reason cases
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **skip**: Narrow optional param_changes before reading it (basedpyright error)
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **skip**: Stateful model of the skip contract through executor.run
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **skip**: Type the stateful skip-contract machine
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **socket**: Wait for Unix listeners to accept connections
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **status**: Missing tracked files, real-remote push counts, unusable and absent remotes
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **storage**: Deterministic Task 8 red test; drop the racy restore stress test
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **storage**: Multi-process stress of shared StateDB writes and directory restores
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **storage**: Name the restore-race owner in the xfail reason
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **tui**: Remove obsolete mutex helper API
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **tui**: Wait for workers before asserting a rerun guard sent nothing
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **watch**: Pipeline.py edits reload a running Engine; red test for unreported edits before any run
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **worker**: A restore check lists every unavailable output and writes nothing
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **worker**: A restore keeps checking outputs after restoring one
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **worker**: Assert what messages must say, not their exact wording
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **worker**: Generation tier, output restore, dependency hashing, run-cache key and generation bump
  against a real store
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **worker**: Lock-comparison skip reason, stage without joblib; red test for split UTF-8 in
  captured subprocess output
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **worker**: Logs, captured output, lock waiting, params overrides, skip/run reasons, seeding,
  nested parallelism and value plumbing
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **worker**: Name Task 9 as owner of the split-UTF-8 red test; drop a duplicate reason test
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))

- **worker**: Return-shape errors, missing outputs, early-failure input hash, project root reset,
  author-facing messages and a leaked child holding stdout
  ([`4cd4a2d`](https://github.com/sjawhar/pivot/commit/4cd4a2dbd446dbdf8da9bd00131af16d425246d0))


## v0.2.4 (2026-10-15)

### Bug Fixes

- **cli**: Clear stage manifests in fingerprint reset
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **cli**: Exit non-zero when the TUI crashes ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **cli**: Fail `pivot repro --tui --watch` when the engine crashes
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **cli**: Stop the TUI event poller when the engine or TUI finishes
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **cli**: Suggest pivot repro, and pivot pull for a clone missing outputs (F1)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Carry stage resources on the stage node, not the shared function
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Compare a stage's closure and default values by canonical text
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Compare a stage's params by their canonical encoding
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Compare input and output formats when one stage is reached twice
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Name the return value in p.previous() and handle-lookup errors
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Name the return value, not '_single', in output errors (F11)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: One stage reached twice means same definition and wiring, not same object
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Refuse a non-integer resource request when the stage is declared
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Refuse a pipeline input bound to another stage's output path
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Refuse two pipelines with one name instead of sharing their stages
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Reject stage signatures workers cannot call
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Reject stage signatures workers cannot invoke
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Reject two input names bound to one file at load (F12)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Retain direct foreign input bindings
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Show a loader's function fields by closure values in a refusal
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **deps**: Require pydantic 2.13, whose by-value pickled models validate
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **discovery**: --all skips a pipeline.py that re-exports another file's pipeline
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **discovery**: Accept a symlinked project state directory
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **discovery**: Never own site-packages modules; reuse the completion pipeline
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **discovery**: Refuse a pipeline whose root= is not the project root
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **engine**: A params.yaml error during watch is an invalid pipeline
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **engine**: Finalize cancellation after final dispatch
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **engine**: Finalize cancellation before worker startup
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **engine**: Size the cpu resource by the worker count, not the stages in the run
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **explain**: Name an edited or missing metric instead of 'Outputs missing from cache' (F6)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Allow ArtifactHandle default arguments
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Allow library ufuncs that lack __module__
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Allow library-exported callable objects
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Apply the class attribute rule only to what a user class body wrote
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Bump the cache schema version for StageParams extra="forbid"
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Check default argument values like captured module values
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Check every live value a cached manifest read
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Drop bare @stage decorators from stage code hashes
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: End a self-referencing value in a stable cycle token
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Hash a dataclass instance field by field
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Hash a default-repr object by its class and attributes
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Hash a function value by its name and code
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Hash a pydantic model class as a class
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Hash a secret outside a stage's params by its keyed digest
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Hash and record set params in a canonical order
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Include call-site input wiring in every skip key
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Key the manifest cache by stage function identity
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: List each unsafe-capture refusal once
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Name a function by its closure values and defaults
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Read module.NAME through the classifier globals use
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Record library functions a stage reads by name
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Refuse a partial whose function is bound to an instance
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Refuse methods bound to an instance
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Refuse user callable instances bound in loader config and partials
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Satisfy type check for sys.modules scan
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Track the class attributes a stage reads
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Treat nonlocal modules as module dependencies
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Validate cached manifests against every project module
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Walk a custom loader's own load, save and empty
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Walk functions and classes bound in loader config and partials
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Walk the function behind functools.cache helpers
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **get**: Preserve keyed revision target selection
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **get**: Recover files after a root-refused pipeline
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **get**: Restore files while pipeline.py fails to import
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **get**: Restore revision outputs to paths recorded in their locks
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **import**: Recurse through typed GitHub directories named .lock
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **incremental**: Restore previous outputs as writable copies
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **init**: Ignore the runtime files pivot creates under .pivot (F2, F17, F19)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **lock**: Require every recorded hash to be a cache key
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **lock**: Treat a lock whose fields have the wrong type as missing
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: A params.yaml override reaches a field with a pydantic alias
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Key secret digests with a per-project HMAC key
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Mask secrets declared through a PEP 695 type alias in errors
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Match params.yaml keys to the validated value, not to annotations
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Never echo a secret's value in a params error
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Only the process that creates the secret key appends to .gitignore
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Preserve excluded fields during override validation
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Record a secret param by a digest of its value
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Refuse a params.yaml typo inside a nested plain BaseModel
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Refuse a params.yaml value its StageParams field rejects at load
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Refuse a symlinked .pivot/.gitignore when gitignoring the key
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Refuse a symlinked .pivot/secret.key instead of following it
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: Refuse params.yaml keys that name no loaded stage
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: StageParams refuses unknown fields, from code or params.yaml
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **plots**: Write the gallery relative to the project root, not cwd
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **plots**: Write the gallery under .pivot/ with relative image paths (F5)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **registry**: Refuse outputs that resolve to one workspace path (F9)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **rpc**: Commit the engine's pipeline instead of reading it from Click context
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **rpc**: Refuse commit while the pipeline is invalid or the engine is starting
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **rpc**: Refuse run with -32004 while the engine is starting instead of waiting
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **rpc**: Refuse run with an error while the watched pipeline is invalid
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **rpc**: Send every artifact identity as a {producer, key} object
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **rpc**: Wait for the startup snapshot before deciding a run request
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **run-cache**: Build every stage input hash with one per-identity builder
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **skip**: Compare recorded and current params by canonical JSON text
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **state**: Skip oversized output file-hash cache keys
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **status**: Count downstream stages pull restores in a plain clone
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **status**: Count pullable stages from explain's reason, without re-hashing
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **status**: Don't suggest pivot pull for a metric-only output mismatch
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **status**: Don't suggest pivot pull for stages downstream of a stale stage
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **status**: Pull count passes through committed stages and keeps -q --json
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **status**: Write explain dep_changes identities as {producer, key}
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **storage**: Preserve lexical containment through data symlinks
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **storage**: Refuse absolute artifact output identities
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **store**: Give pipeline inputs their own tag instead of guessing from disk
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **store**: Key the dependency collision check on the logical path
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **tui**: Read engine_state_changed's state field so watch status reaches the app
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **tui**: Rebuild artifact identities in explanations received over RPC
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **tui**: Send output_summary identities as {producer, key} objects
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **tui**: Send run and commit from Textual workers so slow replies never freeze the app
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **tui**: Show a closed engine connection as a notification, not a crash
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **tui**: Signal socket readiness only once the RPC server listens
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **verify**: Resolve lock dependency paths through the workspace store
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: A restart replays the cancelled run's stages too
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: A run request while a run is cancelling waits for it
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: An explicit cancel after a queued restart keeps the run cancelled
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Avoid submitting after cancellation ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Ctrl-C stops plain watch at once, even mid-reload
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Drop stale bytecode for changed sources before a reload
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Keep redirected stderr log records in their file
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Listen on the RPC socket while the startup fingerprint snapshot runs
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Print log records above the live console area
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Queue the startup run ahead of RPC run requests made during the snapshot
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Reload the edited pipeline off the event loop
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Report each watch run on its own in the console
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Report pipeline edits made before any run as modified stages
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Report why the pipeline is invalid, and show it in the TUI
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Run stale stages when watch mode starts
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Treat a pipeline as invalid whenever a reload rejects it
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Treat a stage fingerprint error as an invalid pipeline and wait for a fix
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Warn and keep reloading when stale bytecode cannot be removed
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **worker**: Run the stage outside the joblib ImportError handler (F3)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

### Chores

- **compose**: Drop the file-level pyright pragma
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **pyright**: Remove 24 redundant #452 file pragmas
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

### Code Style

- **status**: Format integrated source ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **types**: Format output path lock fields ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

### Continuous Integration

- Measure coverage on the pivot test row so fail_under = 90 binds
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

### Documentation

- State manifest cache validation rule ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **agents**: Structure listing matches the package (no pipeline/, dag/ renders, outputs.py holds
  DirectoryOut) ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **caching**: State the stat-cache boundary for committed outputs
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: List module reads the walk does not follow
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Name current framework classes in the is_user_code note
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Say why the project-module scan is a memo
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprinting**: Name the current module-helper manifest key
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **outputs**: DirectoryOut docstring states that compose rejects it
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **params**: A container serializer can still put a secret in the record
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **pipelines**: Name fingerprint settings in the same-stage rule
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **readme**: Package README uses the compositional API
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **resources**: Guide joblib process pools ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **resources**: Identify internally parallel stages
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **skill**: Quote the error messages users actually see (F10)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **skills**: Point RC-3 at the current type-hint walk
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **stage_def**: A subclass keeps extra="forbid" unless it sets extra
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **stage_def**: Describe directory-output keys without DirectoryOut
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **tui**: Describe the poller stop event and engine-crash shutdown
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Only RPC cancel is an explicit cancel in watch mode
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

### Performance Improvements

- **commit**: Hash each output once and link it by the configured checkout mode
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **discovery**: Memoize the root check per module file for a session
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Cache loader method AST hashes
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Cache loader method AST hashes; document joblib process pools; fix a watch
  cancellation race ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Name each function once per fingerprint pass
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Scan sys.modules once per fingerprint pass
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **state**: Record output file hashes in the StateDB when outputs are committed or restored
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **status**: Read each stage lock once, and only what RPC explain needs
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **store**: Stop importing pandas on every CLI path
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Decide each output root's own-write check once per event batch
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

### Refactoring

- Delete stale stage-function type docs and StageReturn
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- Delete unused legacy output classes Out, Metric and Plot
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- Delete unused path_utils module ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **cli**: One shared --resources option for repro and run
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Drop test-only stage wrapper attributes; import loaders at module level
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: One output-spec lookup for handles, deps and p.previous
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: One output-tag mapping for deps and outs
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **core**: Build WorkerStageInfo with its constructor
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **core**: Drop core.py's last file-wide pyright suppressions
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **engine**: Delete the merkle IDs nothing reads
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **engine**: Let discovery's import session own reload module eviction
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **exceptions**: Delete unused DVC compatibility errors
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **graph**: Parse artifact node identities with types.identity_from_key
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **lock**: Drop the constant outs[].tag and the never-written display field
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **registry**: Require input paths in every stage record
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **store**: Drop the unreachable absolute input binding branch
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

### Testing

- Adapt to the integrated head (82886c4) ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- Give three lock fixtures valid hashes so each fails for its own reason
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- Stop building the legacy output classes ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Cover direct foreign input binding metadata
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **compose**: Type direct foreign input binding guard
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **deps**: Add hypothesis and mutmut to the pivot dev extra
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **engine**: Assert cancelled skip is not submitted
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **engine**: Cover cancellation during coordinator skip
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **executor**: Run(checkout_missing=True) restores a deleted tracked input or reports it uncached
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Compare params as the run-cache key serializes them
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **fingerprint**: Hypothesis properties for fresh-process determinism, formatting insensitivity,
  one-token sensitivity and param overrides in the run-cache key
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **import**: Pivot import/update end to end against a local git source and moto S3; red test for
  Task 9 ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **import**: Truncation behaviour of the available list; red test for Task 9
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **integration**: Fresh-venv wheel install smoke of init/repro/status/dag (CI only)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **integration**: Rendezvous the disjoint concurrent-run stages
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **lock**: Red tests for malformed lock shapes (Task 27)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **properties**: Hypothesis round-trips for loaders, lock files, identities, DAG order and input
  hash ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **skill**: Match each Common errors quote against the message it stands for
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **skip**: Kill surviving mutants for output-path, param add/remove and empty-reason cases
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **skip**: Narrow optional param_changes before reading it (basedpyright error)
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **skip**: Stateful model of the skip contract through executor.run
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **skip**: Type the stateful skip-contract machine
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **status**: Missing tracked files, real-remote push counts, unusable and absent remotes
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **storage**: Deterministic Task 8 red test; drop the racy restore stress test
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **storage**: Multi-process stress of shared StateDB writes and directory restores
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **storage**: Name the restore-race owner in the xfail reason
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **tui**: Wait for workers before asserting a rerun guard sent nothing
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **watch**: Pipeline.py edits reload a running Engine; red test for unreported edits before any run
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **worker**: A restore check lists every unavailable output and writes nothing
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **worker**: A restore keeps checking outputs after restoring one
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **worker**: Assert what messages must say, not their exact wording
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **worker**: Generation tier, output restore, dependency hashing, run-cache key and generation bump
  against a real store ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **worker**: Lock-comparison skip reason, stage without joblib; red test for split UTF-8 in
  captured subprocess output ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **worker**: Logs, captured output, lock waiting, params overrides, skip/run reasons, seeding,
  nested parallelism and value plumbing ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **worker**: Match the cache restore override signature
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **worker**: Name Task 9 as owner of the split-UTF-8 red test; drop a duplicate reason test
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))

- **worker**: Return-shape errors, missing outputs, early-failure input hash, project root reset,
  author-facing messages and a leaked child holding stdout
  ([#474](https://github.com/sjawhar/pivot/pull/474),
  [`9219c81`](https://github.com/sjawhar/pivot/commit/9219c81fbb9447f22c3b2fa7d7dcfb975b259e3a))


## v0.2.3 (2026-10-15)

### Bug Fixes

- **cache**: Clean staging left by interrupted saves
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **cache**: Lock file restores and clean what a killed one staged
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **cache**: Serialize save staging with restores
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **cache**: Stage file checkouts in a read-only parent
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **checkout**: Retain crash cleanup under restore lock
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **cli**: Clear stage manifests in fingerprint reset
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **cli**: Exit non-zero when the TUI crashes ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **cli**: Fail `pivot repro --tui --watch` when the engine crashes
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **cli**: Stop the TUI event poller when the engine or TUI finishes
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Carry stage resources on the stage node, not the shared function
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Compare a stage's closure and default values by canonical text
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Compare a stage's params by their canonical encoding
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Compare input and output formats when one stage is reached twice
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Name the return value in p.previous() and handle-lookup errors
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Name the return value, not '_single', in output errors (F11)
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: One stage reached twice means same definition and wiring, not same object
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Refuse a non-integer resource request when the stage is declared
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Refuse a pipeline input bound to another stage's output path
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Refuse two pipelines with one name instead of sharing their stages
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Reject stage signatures workers cannot call
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Reject stage signatures workers cannot invoke
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Reject two input names bound to one file at load (F12)
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Retain direct foreign input bindings
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Show a loader's function fields by closure values in a refusal
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **deps**: Require pydantic 2.13, whose by-value pickled models validate
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **discovery**: --all skips a pipeline.py that re-exports another file's pipeline
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **discovery**: Accept a symlinked project state directory
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **discovery**: Never own site-packages modules; reuse the completion pipeline
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **discovery**: Refuse a pipeline whose root= is not the project root
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **engine**: A params.yaml error during watch is an invalid pipeline
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **engine**: Size the cpu resource by the worker count, not the stages in the run
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **explain**: Name an edited or missing metric instead of 'Outputs missing from cache' (F6)
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Allow ArtifactHandle default arguments
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Allow library ufuncs that lack __module__
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Apply the class attribute rule only to what a user class body wrote
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Bump the cache schema version for StageParams extra="forbid"
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Check default argument values like captured module values
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Check every live value a cached manifest read
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Drop bare @stage decorators from stage code hashes
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: End a self-referencing value in a stable cycle token
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Hash a dataclass instance field by field
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Hash a default-repr object by its class and attributes
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Hash a function value by its name and code
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Hash a pydantic model class as a class
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Hash a secret outside a stage's params by its keyed digest
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Hash and record set params in a canonical order
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Include call-site input wiring in every skip key
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Key the manifest cache by stage function identity
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: List each unsafe-capture refusal once
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Name a function by its closure values and defaults
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Read module.NAME through the classifier globals use
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Record library functions a stage reads by name
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Refuse a partial whose function is bound to an instance
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Refuse methods bound to an instance
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Refuse user callable instances bound in loader config and partials
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Satisfy type check for sys.modules scan
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Track the class attributes a stage reads
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Validate cached manifests against every project module
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Walk a custom loader's own load, save and empty
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Walk functions and classes bound in loader config and partials
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Walk the function behind functools.cache helpers
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **get**: Preserve keyed revision target selection
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **get**: Recover files after a root-refused pipeline
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **get**: Restore files while pipeline.py fails to import
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **get**: Restore revision outputs to paths recorded in their locks
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **import**: Recurse through typed GitHub directories named .lock
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **incremental**: Restore previous outputs as writable copies
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **lock**: Require every recorded hash to be a cache key
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **lock**: Treat a lock whose fields have the wrong type as missing
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: A params.yaml override reaches a field with a pydantic alias
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Key secret digests with a per-project HMAC key
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Mask secrets declared through a PEP 695 type alias in errors
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Match params.yaml keys to the validated value, not to annotations
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Never echo a secret's value in a params error
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Only the process that creates the secret key appends to .gitignore
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Preserve excluded fields during override validation
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Record a secret param by a digest of its value
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Refuse a params.yaml typo inside a nested plain BaseModel
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Refuse a params.yaml value its StageParams field rejects at load
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Refuse a symlinked .pivot/.gitignore when gitignoring the key
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Refuse a symlinked .pivot/secret.key instead of following it
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: Refuse params.yaml keys that name no loaded stage
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: StageParams refuses unknown fields, from code or params.yaml
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **plots**: Write the gallery relative to the project root, not cwd
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **registry**: Refuse outputs that resolve to one workspace path (F9)
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **remote**: Never replace an existing cache entry when pulling
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **remote**: Pull without hard links; give repair steps that work
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **remote**: Verify downloaded cache hashes before promotion
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **restore**: Defer forced removal until cache fallback
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **rpc**: Commit the engine's pipeline instead of reading it from Click context
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **rpc**: Refuse commit while the pipeline is invalid or the engine is starting
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **rpc**: Refuse run with -32004 while the engine is starting instead of waiting
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **rpc**: Refuse run with an error while the watched pipeline is invalid
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **rpc**: Send every artifact identity as a {producer, key} object
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **rpc**: Wait for the startup snapshot before deciding a run request
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **run-cache**: Build every stage input hash with one per-identity builder
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **skip**: Compare recorded and current params by canonical JSON text
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **state**: Skip oversized output file-hash cache keys
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **status**: Count downstream stages pull restores in a plain clone
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **status**: Count pullable stages from explain's reason, without re-hashing
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **status**: Don't suggest pivot pull for a metric-only output mismatch
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **status**: Don't suggest pivot pull for stages downstream of a stale stage
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **status**: Pull count passes through committed stages and keeps -q --json
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **status**: Write explain dep_changes identities as {producer, key}
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Complete project restore state context
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Concurrent restores stop racing; crash leftovers are cleaned; pull refuses corrupt
  remote objects ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Keep generic store commit contract
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Make concurrent restores crash-safe (#461, #228)
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Preserve lexical containment through data symlinks
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Refuse absolute artifact output identities
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Share restore locks across cache commands
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **store**: Give pipeline inputs their own tag instead of guessing from disk
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **store**: Key the dependency collision check on the logical path
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **tui**: Read engine_state_changed's state field so watch status reaches the app
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **tui**: Rebuild artifact identities in explanations received over RPC
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **tui**: Send output_summary identities as {producer, key} objects
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **tui**: Send run and commit from Textual workers so slow replies never freeze the app
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **tui**: Show a closed engine connection as a notification, not a crash
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **tui**: Signal socket readiness only once the RPC server listens
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: A restart replays the cancelled run's stages too
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: A run request while a run is cancelling waits for it
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: An explicit cancel after a queued restart keeps the run cancelled
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Ctrl-C stops plain watch at once, even mid-reload
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Drop stale bytecode for changed sources before a reload
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Keep redirected stderr log records in their file
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Listen on the RPC socket while the startup fingerprint snapshot runs
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Print log records above the live console area
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Queue the startup run ahead of RPC run requests made during the snapshot
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Reload the edited pipeline off the event loop
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Report each watch run on its own in the console
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Report pipeline edits made before any run as modified stages
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Report why the pipeline is invalid, and show it in the TUI
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Run stale stages when watch mode starts
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Treat a pipeline as invalid whenever a reload rejects it
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Treat a stage fingerprint error as an invalid pipeline and wait for a fix
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Warn and keep reloading when stale bytecode cannot be removed
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **worker**: Run the stage outside the joblib ImportError handler (F3)
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

### Chores

- **compose**: Drop the file-level pyright pragma
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **pyright**: Remove 24 redundant #452 file pragmas
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

### Code Style

- **status**: Format integrated source ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **types**: Format output path lock fields ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

### Continuous Integration

- Measure coverage on the pivot test row so fail_under = 90 binds
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

### Documentation

- State manifest cache validation rule ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **agents**: Structure listing matches the package (no pipeline/, dag/ renders, outputs.py holds
  DirectoryOut) ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **caching**: State the stat-cache boundary for committed outputs
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: List module reads the walk does not follow
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Name current framework classes in the is_user_code note
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Say why the project-module scan is a memo
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprinting**: Name the current module-helper manifest key
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **outputs**: DirectoryOut docstring states that compose rejects it
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **params**: A container serializer can still put a secret in the record
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **pipelines**: Name fingerprint settings in the same-stage rule
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **readme**: Package README uses the compositional API
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **remote**: Give repair guidance that reuploads corrupt hashes
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **remote**: Say pull verifies each download against its hash
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **skill**: Quote the error messages users actually see (F10)
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **skills**: Point RC-3 at the current type-hint walk
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **stage_def**: A subclass keeps extra="forbid" unless it sets extra
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **stage_def**: Describe directory-output keys without DirectoryOut
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **tui**: Describe the poller stop event and engine-crash shutdown
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Only RPC cancel is an explicit cancel in watch mode
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

### Performance Improvements

- **cache**: Reuse restore token during checkout ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **commit**: Hash each output once and link it by the configured checkout mode
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **discovery**: Memoize the root check per module file for a session
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Name each function once per fingerprint pass
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Scan sys.modules once per fingerprint pass
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **state**: Record output file hashes in the StateDB when outputs are committed or restored
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **status**: Read each stage lock once, and only what RPC explain needs
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **store**: Stop importing pandas on every CLI path
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Decide each output root's own-write check once per event batch
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

### Refactoring

- Delete stale stage-function type docs and StageReturn
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- Delete unused legacy output classes Out, Metric and Plot
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- Delete unused path_utils module ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **cli**: One shared --resources option for repro and run
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Drop test-only stage wrapper attributes; import loaders at module level
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: One output-spec lookup for handles, deps and p.previous
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: One output-tag mapping for deps and outs
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **core**: Build WorkerStageInfo with its constructor
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **core**: Drop core.py's last file-wide pyright suppressions
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **engine**: Delete the merkle IDs nothing reads
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **engine**: Let discovery's import session own reload module eviction
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **exceptions**: Delete unused DVC compatibility errors
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **graph**: Parse artifact node identities with types.identity_from_key
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **lock**: Drop the constant outs[].tag and the never-written display field
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **registry**: Require input paths in every stage record
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **store**: Drop the unreachable absolute input binding branch
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

### Testing

- Adapt to the integrated head (82886c4) ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- Give three lock fixtures valid hashes so each fails for its own reason
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- Stop building the legacy output classes ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **checkout**: Crash the restore directly in the orphan-cleanup test
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Cover direct foreign input binding metadata
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **compose**: Type direct foreign input binding guard
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **executor**: Run(checkout_missing=True) restores a deleted tracked input or reports it uncached
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Compare params as the run-cache key serializes them
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **fingerprint**: Hypothesis properties for fresh-process determinism, formatting insensitivity,
  one-token sensitivity and param overrides in the run-cache key
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **import**: Drop strict xfail now that update keeps the last good file
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **import**: Pivot import/update end to end against a local git source and moto S3; red test for
  Task 9 ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **import**: Truncation behaviour of the available list; red test for Task 9
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **integration**: Rendezvous the disjoint concurrent-run stages
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **integration**: Wait for producer lock before consumer
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **remote**: Key moto fixtures by their content hash
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **skill**: Match each Common errors quote against the message it stands for
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **skip**: Narrow optional param_changes before reading it (basedpyright error)
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **state**: Cover readonly opens of incomplete state.lmdb
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **status**: Missing tracked files, real-remote push counts, unusable and absent remotes
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Deterministic Task 8 red test; drop the racy restore stress test
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Drop strict xfail on restore lock-file test
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Preserve shared-cache lock regression
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Spawn concurrent restore workers ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **storage**: Use spawn manager in concurrency fixture
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **tui**: Wait for workers before asserting a rerun guard sent nothing
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **watch**: Pipeline.py edits reload a running Engine; red test for unreported edits before any run
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **worker**: A restore check lists every unavailable output and writes nothing
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **worker**: A restore keeps checking outputs after restoring one
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **worker**: Assert what messages must say, not their exact wording
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **worker**: Generation tier, output restore, dependency hashing, run-cache key and generation bump
  against a real store ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **worker**: Lock-comparison skip reason, stage without joblib; red test for split UTF-8 in
  captured subprocess output ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **worker**: Logs, captured output, lock waiting, params overrides, skip/run reasons, seeding,
  nested parallelism and value plumbing ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **worker**: Match the cache restore override signature
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **worker**: Name Task 9 as owner of the split-UTF-8 red test; drop a duplicate reason test
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))

- **worker**: Return-shape errors, missing outputs, early-failure input hash, project root reset,
  author-facing messages and a leaked child holding stdout
  ([#473](https://github.com/sjawhar/pivot/pull/473),
  [`35c075e`](https://github.com/sjawhar/pivot/commit/35c075eec766d46504ecc29376c33b1257151b5b))


## v0.2.2 (2026-10-15)

### Bug Fixes

- **cli**: Fail `pivot repro --tui --watch` when the engine crashes
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Carry stage resources on the stage node, not the shared function
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Compare a stage's closure and default values by canonical text
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Compare a stage's params by their canonical encoding
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Compare input and output formats when one stage is reached twice
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Name the return value in p.previous() and handle-lookup errors
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Name the return value, not '_single', in output errors (F11)
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: One stage reached twice means same definition and wiring, not same object
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Refuse a non-integer resource request when the stage is declared
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Refuse a pipeline input bound to another stage's output path
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Refuse two pipelines with one name instead of sharing their stages
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Reject stage signatures workers cannot call
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Reject stage signatures workers cannot invoke
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Reject two input names bound to one file at load (F12)
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Retain direct foreign input bindings
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Show a loader's function fields by closure values in a refusal
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **console**: Always report lossy-output warnings; count truncated ones
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **console**: Collapse repeated stage warnings into one line with a count
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **data**: Load Pivot JSONL metadata through DataFrameJSONL
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **data**: Stream Pivot JSONL schema and row counts
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **deps**: Require pydantic 2.13, whose by-value pickled models validate
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **discovery**: --all skips a pipeline.py that re-exports another file's pipeline
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **discovery**: Accept a symlinked project state directory
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **discovery**: Never own site-packages modules; reuse the completion pipeline
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **discovery**: Refuse a pipeline whose root= is not the project root
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **engine**: A params.yaml error during watch is an invalid pipeline
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **engine**: Size the cpu resource by the worker count, not the stages in the run
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **explain**: Name an edited or missing metric instead of 'Outputs missing from cache' (F6)
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Allow ArtifactHandle default arguments
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Allow library ufuncs that lack __module__
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Apply the class attribute rule only to what a user class body wrote
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Bump the cache schema version for StageParams extra="forbid"
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Check default argument values like captured module values
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Check every live value a cached manifest read
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Drop bare @stage decorators from stage code hashes
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: End a self-referencing value in a stable cycle token
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Hash a dataclass instance field by field
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Hash a default-repr object by its class and attributes
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Hash a function value by its name and code
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Hash a pydantic model class as a class
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Hash a secret outside a stage's params by its keyed digest
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Hash and record set params in a canonical order
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Include call-site input wiring in every skip key
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Key the manifest cache by stage function identity
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: List each unsafe-capture refusal once
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Name a function by its closure values and defaults
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Read module.NAME through the classifier globals use
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Record library functions a stage reads by name
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Refuse a partial whose function is bound to an instance
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Refuse methods bound to an instance
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Refuse user callable instances bound in loader config and partials
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Track the class attributes a stage reads
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Walk a custom loader's own load, save and empty
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Walk functions and classes bound in loader config and partials
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Walk the function behind functools.cache helpers
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **get**: Preserve keyed revision target selection
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **get**: Recover files after a root-refused pipeline
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **get**: Restore files while pipeline.py fails to import
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **get**: Restore revision outputs to paths recorded in their locks
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **import**: Recurse through typed GitHub directories named .lock
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **incremental**: Restore previous outputs as writable copies
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Accept equivalent IANA timezone round trips
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Keep columns and dtypes of zero-row DataFrames
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Keep integer column labels of DataFrameJSONL outputs
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Name the nanosecond range for out-of-range date objects
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Never write an out-of-range datetime as a different date
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Outputs load back as written (dates, exact floats, numpy metrics, line endings)
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Preserve DataFrame temporal dtypes explicitly
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Read headerless JSONL exactly as pandas did
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Reject non-roundtrippable timezones
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Restore all-null float column dtypes
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Round-trip a column named like the DataFrameJSONL header
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Round-trip DataFrame values faithfully
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Round-trip Text line endings exactly
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Write mixed-timezone datetime columns as before, with a warning
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **lock**: Require every recorded hash to be a cache key
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **lock**: Treat a lock whose fields have the wrong type as missing
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **metrics**: Name a single YAML output without printing None
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **metrics**: Reject NumPy temporal YAML values ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **metrics**: Reject unrepresentable extended NumPy scalars
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **metrics**: Write only safe YAML metric data ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **outputs**: Warn instead of refusing outputs the exact formats cannot hold
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: A params.yaml override reaches a field with a pydantic alias
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Key secret digests with a per-project HMAC key
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Mask secrets declared through a PEP 695 type alias in errors
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Match params.yaml keys to the validated value, not to annotations
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Never echo a secret's value in a params error
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Only the process that creates the secret key appends to .gitignore
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Preserve excluded fields during override validation
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Record a secret param by a digest of its value
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Refuse a params.yaml typo inside a nested plain BaseModel
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Refuse a params.yaml value its StageParams field rejects at load
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Refuse a symlinked .pivot/.gitignore when gitignoring the key
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Refuse a symlinked .pivot/secret.key instead of following it
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: Refuse params.yaml keys that name no loaded stage
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: StageParams refuses unknown fields, from code or params.yaml
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **registry**: Refuse outputs that resolve to one workspace path (F9)
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **rpc**: Commit the engine's pipeline instead of reading it from Click context
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **rpc**: Refuse commit while the pipeline is invalid or the engine is starting
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **rpc**: Refuse run with -32004 while the engine is starting instead of waiting
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **rpc**: Refuse run with an error while the watched pipeline is invalid
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **rpc**: Send every artifact identity as a {producer, key} object
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **rpc**: Wait for the startup snapshot before deciding a run request
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **run-cache**: Build every stage input hash with one per-identity builder
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **security**: Warn when loading Pickle artifacts
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **skip**: Compare recorded and current params by canonical JSON text
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **state**: Skip oversized output file-hash cache keys
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **status**: Count downstream stages pull restores in a plain clone
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **status**: Count pullable stages from explain's reason, without re-hashing
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **status**: Pull count passes through committed stages and keeps -q --json
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **status**: Write explain dep_changes identities as {producer, key}
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **storage**: Preserve lexical containment through data symlinks
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **storage**: Refuse absolute artifact output identities
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **store**: Give pipeline inputs their own tag instead of guessing from disk
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **store**: Key the dependency collision check on the logical path
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **tui**: Read engine_state_changed's state field so watch status reaches the app
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **tui**: Send output_summary identities as {producer, key} objects
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **tui**: Send run and commit from Textual workers so slow replies never freeze the app
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **tui**: Show a closed engine connection as a notification, not a crash
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **tui**: Signal socket readiness only once the RPC server listens
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: A restart replays the cancelled run's stages too
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: A run request while a run is cancelling waits for it
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: An explicit cancel after a queued restart keeps the run cancelled
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Ctrl-C stops plain watch at once, even mid-reload
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Drop stale bytecode for changed sources before a reload
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Keep redirected stderr log records in their file
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Listen on the RPC socket while the startup fingerprint snapshot runs
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Print log records above the live console area
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Queue the startup run ahead of RPC run requests made during the snapshot
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Reload the edited pipeline off the event loop
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Report each watch run on its own in the console
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Report pipeline edits made before any run as modified stages
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Report why the pipeline is invalid, and show it in the TUI
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Run stale stages when watch mode starts
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Treat a pipeline as invalid whenever a reload rejects it
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Treat a stage fingerprint error as an invalid pipeline and wait for a fix
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Warn and keep reloading when stale bytecode cannot be removed
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **worker**: Run the stage outside the joblib ImportError handler (F3)
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

### Chores

- **compose**: Drop the file-level pyright pragma
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Type metadata restoration tests ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **pyright**: Remove 24 redundant #452 file pragmas
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

### Code Style

- **status**: Format integrated source ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **test**: Format compose metric tests ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **test**: Separate DataFrameJSONL tests from incremental-value tests
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **types**: Format output path lock fields ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

### Continuous Integration

- Measure coverage on the pivot test row so fail_under = 90 binds
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

### Documentation

- Document DataFrameJSONL metadata header contract
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **agents**: Structure listing matches the package (no pipeline/, dag/ renders, outputs.py holds
  DirectoryOut) ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **caching**: State the stat-cache boundary for committed outputs
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: List module reads the walk does not follow
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Name current framework classes in the is_user_code note
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Say why the project-module scan is a memo
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprinting**: Name the current module-helper manifest key
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: DataFrameJSONL header lists every column's name and dtype (loader-fixes b09dfc0)
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **outputs**: DirectoryOut docstring states that compose rejects it
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **params**: A container serializer can still put a secret in the record
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **pipelines**: Name fingerprint settings in the same-stage rule
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **readme**: Package README uses the compositional API
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **skill**: Numpy values in YAML outputs now load; name the new refusal
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **skill**: Quote the error messages users actually see (F10)
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **skills**: Point RC-3 at the current type-hint walk
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **stage_def**: A subclass keeps extra="forbid" unless it sets extra
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **stage_def**: Describe directory-output keys without DirectoryOut
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **tui**: Describe the poller stop event and engine-crash shutdown
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Only RPC cancel is an explicit cancel in watch mode
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

### Performance Improvements

- **commit**: Hash each output once and link it by the configured checkout mode
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **discovery**: Memoize the root check per module file for a session
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Name each function once per fingerprint pass
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Scan sys.modules once per fingerprint pass
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **state**: Record output file hashes in the StateDB when outputs are committed or restored
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **status**: Read each stage lock once, and only what RPC explain needs
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **store**: Stop importing pandas on every CLI path
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Decide each output root's own-write check once per event batch
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

### Refactoring

- Delete stale stage-function type docs and StageReturn
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- Delete unused legacy output classes Out, Metric and Plot
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- Delete unused path_utils module ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **cli**: One shared --resources option for repro and run
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Drop test-only stage wrapper attributes; import loaders at module level
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: One output-spec lookup for handles, deps and p.previous
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: One output-tag mapping for deps and outs
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **core**: Build WorkerStageInfo with its constructor
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **core**: Drop core.py's last file-wide pyright suppressions
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **engine**: Delete the merkle IDs nothing reads
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **engine**: Let discovery's import session own reload module eviction
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **exceptions**: Delete unused DVC compatibility errors
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **graph**: Parse artifact node identities with types.identity_from_key
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **lock**: Drop the constant outs[].tag and the never-written display field
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **registry**: Require input paths in every stage record
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **store**: Drop the unreachable absolute input binding branch
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

### Testing

- Adapt to the integrated head (82886c4) ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- Give three lock fixtures valid hashes so each fails for its own reason
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- Stop building the legacy output classes ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **cli**: Compare DataFrame outputs as loaded rows, not raw JSONL bytes
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **cli**: Compare merged manifest-cache outputs as loaded rows
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **cli**: Load DataFrameJSONL outputs in final-review tests
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Cover direct foreign input binding metadata
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **compose**: Type direct foreign input binding guard
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **executor**: Run(checkout_missing=True) restores a deleted tracked input or reports it uncached
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Compare params as the run-cache key serializes them
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **fingerprint**: Hypothesis properties for fresh-process determinism, formatting insensitivity,
  one-token sensitivity and param overrides in the run-cache key
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **import**: Pivot import/update end to end against a local git source and moto S3; red test for
  Task 9 ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **import**: Truncation behaviour of the available list; red test for Task 9
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **integration**: Rendezvous the disjoint concurrent-run stages
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **integration**: Start the contention consumer once the producer holds its lock
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Assert Pickle runtime warning ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Drop Task 10 strict xfails the loader fixes turn green
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Expect DataFrameJSONL metadata record
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **loaders**: Expect float dtype metadata ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **skill**: Match each Common errors quote against the message it stands for
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **skip**: Narrow optional param_changes before reading it (basedpyright error)
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **status**: Missing tracked files, real-remote push counts, unusable and absent remotes
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **storage**: Deterministic Task 8 red test; drop the racy restore stress test
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **tui**: Wait for workers before asserting a rerun guard sent nothing
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **watch**: Pipeline.py edits reload a running Engine; red test for unreported edits before any run
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **worker**: A restore check lists every unavailable output and writes nothing
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **worker**: A restore keeps checking outputs after restoring one
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **worker**: Assert what messages must say, not their exact wording
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **worker**: Generation tier, output restore, dependency hashing, run-cache key and generation bump
  against a real store ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **worker**: Lock-comparison skip reason, stage without joblib; red test for split UTF-8 in
  captured subprocess output ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **worker**: Logs, captured output, lock waiting, params overrides, skip/run reasons, seeding,
  nested parallelism and value plumbing ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **worker**: Match the cache restore override signature
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **worker**: Name Task 9 as owner of the split-UTF-8 red test; drop a duplicate reason test
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))

- **worker**: Return-shape errors, missing outputs, early-failure input hash, project root reset,
  author-facing messages and a leaked child holding stdout
  ([#470](https://github.com/sjawhar/pivot/pull/470),
  [`faf413b`](https://github.com/sjawhar/pivot/commit/faf413b776dd53e8f99d87b8a5888a112926e5fe))


## v0.2.1 (2026-10-15)

### Bug Fixes

- **checkout**: --only-missing never overwrites files inside existing directories
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **checkout**: Don't delete staged kept entries in a restore that carries none
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **checkout**: Keep hash-ignored entries safe if a restore is killed mid-swap
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **checkout**: Keep hash-ignored entries when replacing a directory
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **checkout**: Reject directory divergence before only-missing restore
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **checkout**: Replace a directory whose differing files are all cached
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **cli**: Clear stage manifests in fingerprint reset
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **cli**: Exit non-zero when the TUI crashes ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **cli**: Fail `pivot repro --tui --watch` when the engine crashes
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **cli**: Stop the TUI event poller when the engine or TUI finishes
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **cli**: Verify, pull, checkout and import stop losing or misreporting data
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Carry stage resources on the stage node, not the shared function
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Compare a stage's closure and default values by canonical text
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Compare a stage's params by their canonical encoding
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Compare input and output formats when one stage is reached twice
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Name the return value in p.previous() and handle-lookup errors
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Name the return value, not '_single', in output errors (F11)
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: One stage reached twice means same definition and wiring, not same object
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Refuse a non-integer resource request when the stage is declared
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Refuse a pipeline input bound to another stage's output path
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Refuse two pipelines with one name instead of sharing their stages
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Reject stage signatures workers cannot call
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Reject stage signatures workers cannot invoke
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Reject two input names bound to one file at load (F12)
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Retain direct foreign input bindings
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Show a loader's function fields by closure values in a refusal
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **deps**: Require pydantic 2.13, whose by-value pickled models validate
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **discovery**: --all skips a pipeline.py that re-exports another file's pipeline
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **discovery**: Accept a symlinked project state directory
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **discovery**: Never own site-packages modules; reuse the completion pipeline
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **discovery**: Refuse a pipeline whose root= is not the project root
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **engine**: A params.yaml error during watch is an invalid pipeline
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **engine**: Size the cpu resource by the worker count, not the stages in the run
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **explain**: Name an edited or missing metric instead of 'Outputs missing from cache' (F6)
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Allow ArtifactHandle default arguments
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Allow library ufuncs that lack __module__
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Allow library-exported callable objects
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Apply the class attribute rule only to what a user class body wrote
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Bump the cache schema version for StageParams extra="forbid"
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Check default argument values like captured module values
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Check every live value a cached manifest read
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Drop bare @stage decorators from stage code hashes
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: End a self-referencing value in a stable cycle token
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Hash a dataclass instance field by field
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Hash a default-repr object by its class and attributes
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Hash a function value by its name and code
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Hash a pydantic model class as a class
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Hash a secret outside a stage's params by its keyed digest
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Hash and record set params in a canonical order
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Include call-site input wiring in every skip key
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Key the manifest cache by stage function identity
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: List each unsafe-capture refusal once
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Name a function by its closure values and defaults
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Read module.NAME through the classifier globals use
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Record library functions a stage reads by name
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Refuse a partial whose function is bound to an instance
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Refuse methods bound to an instance
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Refuse user callable instances bound in loader config and partials
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Satisfy type check for sys.modules scan
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Track the class attributes a stage reads
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Treat nonlocal modules as module dependencies
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Validate cached manifests against every project module
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Walk a custom loader's own load, save and empty
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Walk functions and classes bound in loader config and partials
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Walk the function behind functools.cache helpers
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **get**: Preserve keyed revision target selection
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **get**: Recover files after a root-refused pipeline
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **get**: Restore files while pipeline.py fails to import
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **get**: Restore revision outputs to paths recorded in their locks
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **import**: Recurse through typed GitHub directories named .lock
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **import**: Refuse a non-string default_remote in the source remote config
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **import**: Verify a download before it replaces the imported file
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **import**: Warn on a non-string default_remote instead of refusing
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **incremental**: Restore previous outputs as writable copies
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **lock**: Require every recorded hash to be a cache key
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **lock**: Treat a lock whose fields have the wrong type as missing
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: A params.yaml override reaches a field with a pydantic alias
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Key secret digests with a per-project HMAC key
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Mask secrets declared through a PEP 695 type alias in errors
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Match params.yaml keys to the validated value, not to annotations
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Never echo a secret's value in a params error
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Only the process that creates the secret key appends to .gitignore
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Preserve excluded fields during override validation
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Record a secret param by a digest of its value
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Refuse a params.yaml typo inside a nested plain BaseModel
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Refuse a params.yaml value its StageParams field rejects at load
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Refuse a symlinked .pivot/.gitignore when gitignoring the key
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Refuse a symlinked .pivot/secret.key instead of following it
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: Refuse params.yaml keys that name no loaded stage
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: StageParams refuses unknown fields, from code or params.yaml
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **plots**: Write the gallery relative to the project root, not cwd
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **plots**: Write the gallery under .pivot/ with relative image paths (F5)
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **registry**: Refuse outputs that resolve to one workspace path (F9)
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **remote**: Don't fetch pivot import inputs on pull
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **remote**: Don't fetch raw-input deps on pull ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **rpc**: Commit the engine's pipeline instead of reading it from Click context
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **rpc**: Refuse commit while the pipeline is invalid or the engine is starting
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **rpc**: Refuse run with -32004 while the engine is starting instead of waiting
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **rpc**: Refuse run with an error while the watched pipeline is invalid
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **rpc**: Send every artifact identity as a {producer, key} object
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **rpc**: Wait for the startup snapshot before deciding a run request
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **run-cache**: Build every stage input hash with one per-identity builder
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **skip**: Compare recorded and current params by canonical JSON text
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **state**: Skip oversized output file-hash cache keys
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **status**: Count downstream stages pull restores in a plain clone
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **status**: Count pullable stages from explain's reason, without re-hashing
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **status**: Don't suggest pivot pull for a metric-only output mismatch
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **status**: Don't suggest pivot pull for stages downstream of a stale stage
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **status**: Pull count passes through committed stages and keeps -q --json
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **status**: Write explain dep_changes identities as {producer, key}
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **storage**: Preserve lexical containment through data symlinks
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **storage**: Refuse absolute artifact output identities
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **store**: Give pipeline inputs their own tag instead of guessing from disk
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **store**: Key the dependency collision check on the logical path
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **tui**: Read engine_state_changed's state field so watch status reaches the app
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **tui**: Rebuild artifact identities in explanations received over RPC
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **tui**: Send output_summary identities as {producer, key} objects
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **tui**: Send run and commit from Textual workers so slow replies never freeze the app
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **tui**: Show a closed engine connection as a notification, not a crash
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **tui**: Signal socket readiness only once the RPC server listens
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **verify**: Name stale producer dependencies under --allow-missing
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: A restart replays the cancelled run's stages too
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: A run request while a run is cancelling waits for it
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: An explicit cancel after a queued restart keeps the run cancelled
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Ctrl-C stops plain watch at once, even mid-reload
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Drop stale bytecode for changed sources before a reload
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Keep redirected stderr log records in their file
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Listen on the RPC socket while the startup fingerprint snapshot runs
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Print log records above the live console area
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Queue the startup run ahead of RPC run requests made during the snapshot
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Reload the edited pipeline off the event loop
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Report each watch run on its own in the console
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Report pipeline edits made before any run as modified stages
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Report why the pipeline is invalid, and show it in the TUI
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Run stale stages when watch mode starts
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Treat a pipeline as invalid whenever a reload rejects it
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Treat a stage fingerprint error as an invalid pipeline and wait for a fix
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Warn and keep reloading when stale bytecode cannot be removed
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **worker**: Keep multi-byte characters split across captured pipe reads
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **worker**: Run the stage outside the joblib ImportError handler (F3)
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

### Chores

- **compose**: Drop the file-level pyright pragma
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **pyright**: Remove 24 redundant #452 file pragmas
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

### Code Style

- **status**: Format integrated source ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **types**: Format output path lock fields ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

### Continuous Integration

- Measure coverage on the pivot test row so fail_under = 90 binds
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

### Documentation

- Describe --only-missing directory refusal and what pull transfers
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- State manifest cache validation rule ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **agents**: Structure listing matches the package (no pipeline/, dag/ renders, outputs.py holds
  DirectoryOut) ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **caching**: State the stat-cache boundary for committed outputs
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **checkout**: Name the commands that put staged kept entries back
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: List module reads the walk does not follow
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Name current framework classes in the is_user_code note
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Say why the project-module scan is a memo
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprinting**: Name the current module-helper manifest key
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **outputs**: DirectoryOut docstring states that compose rejects it
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **params**: A container serializer can still put a secret in the record
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **pipelines**: Name fingerprint settings in the same-stage rule
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **readme**: Package README uses the compositional API
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **skill**: Quote the error messages users actually see (F10)
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **skills**: Point RC-3 at the current type-hint walk
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **stage_def**: A subclass keeps extra="forbid" unless it sets extra
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **stage_def**: Describe directory-output keys without DirectoryOut
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **tui**: Describe the poller stop event and engine-crash shutdown
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Only RPC cancel is an explicit cancel in watch mode
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

### Performance Improvements

- **checkout**: Prove retained directory files unchanged without reading them
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **commit**: Hash each output once and link it by the configured checkout mode
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **discovery**: Memoize the root check per module file for a session
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Name each function once per fingerprint pass
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Scan sys.modules once per fingerprint pass
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **state**: Record output file hashes in the StateDB when outputs are committed or restored
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **status**: Read each stage lock once, and only what RPC explain needs
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **store**: Stop importing pandas on every CLI path
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Decide each output root's own-write check once per event batch
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

### Refactoring

- Delete stale stage-function type docs and StageReturn
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- Delete unused legacy output classes Out, Metric and Plot
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- Delete unused path_utils module ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **cli**: One shared --resources option for repro and run
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Drop test-only stage wrapper attributes; import loaders at module level
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: One output-spec lookup for handles, deps and p.previous
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: One output-tag mapping for deps and outs
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **core**: Build WorkerStageInfo with its constructor
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **core**: Drop core.py's last file-wide pyright suppressions
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **engine**: Delete the merkle IDs nothing reads
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **engine**: Let discovery's import session own reload module eviction
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **exceptions**: Delete unused DVC compatibility errors
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **graph**: Parse artifact node identities with types.identity_from_key
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **lock**: Drop the constant outs[].tag and the never-written display field
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **registry**: Require input paths in every stage record
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **store**: Drop the unreachable absolute input binding branch
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

### Testing

- Adapt to the integrated head (82886c4) ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- Give three lock fixtures valid hashes so each fails for its own reason
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- Stop building the legacy output classes ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Cover direct foreign input binding metadata
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **compose**: Type direct foreign input binding guard
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **executor**: Run(checkout_missing=True) restores a deleted tracked input or reports it uncached
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Compare params as the run-cache key serializes them
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **fingerprint**: Hypothesis properties for fresh-process determinism, formatting insensitivity,
  one-token sensitivity and param overrides in the run-cache key
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **import**: Pivot import/update end to end against a local git source and moto S3; red test for
  Task 9 ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **import**: Truncation behaviour of the available list; red test for Task 9
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **integration**: Fresh-venv wheel install smoke of init/repro/status/dag (CI only)
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **integration**: Rendezvous the disjoint concurrent-run stages
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **lock**: Red tests for malformed lock shapes (Task 27)
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **locking**: Wait for the writer lock before starting the contending reader
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **properties**: Hypothesis round-trips for loaders, lock files, identities, DAG order and input
  hash ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **remote**: Pull-with-deps tests use an upstream stage dep
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **skill**: Match each Common errors quote against the message it stands for
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **skip**: Kill surviving mutants for output-path, param add/remove and empty-reason cases
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **skip**: Narrow optional param_changes before reading it (basedpyright error)
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **skip**: Stateful model of the skip contract through executor.run
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **skip**: Type the stateful skip-contract machine
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **status**: Missing tracked files, real-remote push counts, unusable and absent remotes
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **storage**: Deterministic Task 8 red test; drop the racy restore stress test
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **storage**: Multi-process stress of shared StateDB writes and directory restores
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **storage**: Name the restore-race owner in the xfail reason
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **tui**: Wait for workers before asserting a rerun guard sent nothing
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **watch**: Pipeline.py edits reload a running Engine; red test for unreported edits before any run
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **worker**: A restore check lists every unavailable output and writes nothing
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **worker**: A restore keeps checking outputs after restoring one
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **worker**: Assert what messages must say, not their exact wording
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **worker**: Generation tier, output restore, dependency hashing, run-cache key and generation bump
  against a real store ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **worker**: Lock-comparison skip reason, stage without joblib; red test for split UTF-8 in
  captured subprocess output ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **worker**: Logs, captured output, lock waiting, params overrides, skip/run reasons, seeding,
  nested parallelism and value plumbing ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **worker**: Match the cache restore override signature
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **worker**: Name Task 9 as owner of the split-UTF-8 red test; drop a duplicate reason test
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))

- **worker**: Return-shape errors, missing outputs, early-failure input hash, project root reset,
  author-facing messages and a leaked child holding stdout
  ([#472](https://github.com/sjawhar/pivot/pull/472),
  [`4c55bf2`](https://github.com/sjawhar/pivot/commit/4c55bf2194815d16a3713fb193ad620123bb10cb))


## v0.2.0 (2026-10-15)

### Refactoring

- Compositional pipeline API, path-free engine, and pipeline unification
  ([#452](https://github.com/sjawhar/pivot/pull/452),
  [`0a50002`](https://github.com/sjawhar/pivot/commit/0a500027fa5a02c10c7719418b0314ecff01a9af))

### Breaking Changes

- Pivot.yaml, Pipeline.register, Dep/Out/DirectoryOut annotations, matrix expansion and mutex
  declarations are removed; pipelines are written with @stage and `with Pipeline(...)`. See
  skills/writing-pivot-stages/SKILL.md and docs/migrating-from-dvc.md.


## v0.1.0 (2026-07-13)

- Initial Release
