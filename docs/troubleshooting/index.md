# Troubleshooting

Quick reference for common issues. Each links to detailed solutions in the relevant documentation.

## Quick Reference

| Issue | Solution Location |
|-------|-------------------|
| Stage reruns unexpectedly | [Caching & Skip Detection](../concepts/caching.md) |
| Stage runs but output not cached | [Outputs](../concepts/outputs.md) |
| "Cannot pickle..." errors | [Pipelines](../concepts/pipelines.md) |
| Parameters not taking effect | [Parameters](../concepts/parameters.md) |
| Watch mode not detecting changes | [Watch Mode Guide](../guides/watch-mode.md) |
| Lambda causes unnecessary re-runs | [Fingerprinting](../concepts/fingerprinting.md) |
| Remote push/pull fails | [Remote Storage Guide](../guides/remote-storage.md) |
| CI fails but local passes | [CI Integration Guide](../guides/ci-integration.md) |

## Debugging Commands

```bash
# See all stages and their status
pivot list

# Detailed explanation for why a stage will/won't run
pivot status --explain stage_name

# Show what would run without executing
pivot repro --dry-run

# Force a stage to run regardless of cache
pivot repro stage_name --force

# Verbose logging for more detail
PIVOT_LOG_LEVEL=DEBUG pivot repro
```

## Inspect Lock Files

Lock files are YAML and human-readable:

```bash
cat .pivot/stages/train.lock
```

Contains: code fingerprint, parameter hash, dependency hashes, output hashes.

## Common Patterns

### "Could not pickle the task to send it to the workers"

Your stage function captures a variable from its enclosing scope. Move the function to module level and pass values through parameters.

See: [Pipelines](../concepts/pipelines.md)

### Stage Re-runs Every Time

1. Check if you're using lambdas (non-deterministic fingerprints)
2. Check if a helper function changed
3. Run `pivot status --explain stage_name` to see what changed

See: [Fingerprinting](../concepts/fingerprinting.md) and [Caching](../concepts/caching.md)

### Cache Not Shared in CI

1. Ensure lock files are committed: `git add .pivot/stages/*.lock`
2. Ensure remote is configured: `pivot config list`
3. Run `pivot pull` before `pivot repro` in CI

See: [CI Integration Guide](../guides/ci-integration.md) and [Remote Storage Guide](../guides/remote-storage.md)
