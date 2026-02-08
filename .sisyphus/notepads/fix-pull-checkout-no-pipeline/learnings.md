
## Task 2: Fix pull command (2026-02-08)

### Implementation Summary
- Added early failure check in `pull` command: if no pipeline and no targets (and not using `--all`), raise clear error
- Added default `only_missing=True` for checkout invocation when neither `--force` nor `--only-missing` was explicitly passed
- Updated tests to pass targets instead of relying on implicit behavior

### Key Learnings

1. **Context-based flag detection**: The `--all` flag is consumed by the decorator and stored in context via `_store_all_pipelines_in_context()`. Use `cli_decorators.get_all_pipelines_from_context()` to check if `--all` was passed.

2. **Test updates required**: When adding early failure checks, tests that relied on implicit behavior need to be updated. Tests without a pipeline and without targets now need to either:
   - Pass explicit targets (e.g., `["pull", "output.csv"]`)
   - Use `--all` flag (e.g., `["pull", "--all"]`)

3. **Decorator behavior**: The `pivot_command` decorator with `allow_all=True` automatically:
   - Adds `--all` flag to the command
   - Calls `discover_pipeline(all_pipelines=use_all)` before the command runs
   - Stores the flag in context for later retrieval

4. **Error message clarity**: The error message "No pipeline found. Use --all to pull from all pipelines, or specify file targets." guides users to the correct solution.

### Files Modified
- `src/pivot/cli/remote.py`: Added early failure check and default `only_missing` logic
- `tests/remote/test_cli_remote.py`: Updated 4 tests to pass targets or use `--all` flag

### Test Results
- All 22 tests in `tests/remote/test_cli_remote.py` pass
- Regression tests verified: `test_pull_success` and `test_pull_with_stages` both pass

### Commits Created
1. `621782a` - fix(checkout): handle missing pipeline gracefully
2. `c15d30d` - fix(pull): default to --only-missing for checkout, fail early without pipeline

