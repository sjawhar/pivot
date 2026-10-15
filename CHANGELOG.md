# CHANGELOG

<!-- version list -->

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
