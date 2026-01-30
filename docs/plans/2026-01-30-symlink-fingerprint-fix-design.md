# Symlink Fingerprinting Fix Design

## Problem

On macOS with Homebrew Python (and similar setups where Python's installation path contains symlinks), stdlib modules are incorrectly classified as user code. This causes fingerprints to include stdlib functions that should be excluded.

**Root cause:** In `_is_stdlib_path()`, the `module_file` parameter is fully resolved (symlinks followed), but `sys.prefix` and `sys.base_prefix` are used without resolution. When comparing paths, a resolved path won't match an unresolved parent containing symlinks.

## Solution

Add `_init_stdlib_paths()` function that:
1. Resolves `sys.prefix` and `sys.base_prefix` for symlink-safe comparison
2. Handles `OSError` gracefully (falls back to unresolved path)
3. Deduplicates paths when not in a venv (`sys.prefix == sys.base_prefix`)
4. Returns an immutable tuple for consistency with `_SITE_PACKAGE_PATHS`

Update `_is_stdlib_path()` to use the cached `_STDLIB_PATHS` constant.

## Testing

Existing test coverage validates stdlib detection (`test_google_style.py:46`). All 106 fingerprint-related tests pass.
