# Troubleshooting

Quick reference for common issues. Each links to detailed solutions in the relevant documentation.

## Quick Reference

| Issue | Solution Location |
|-------|-------------------|
| Stage reruns unexpectedly | [Outputs & Caching](../reference/outputs.md#stage-reruns-unexpectedly) |
| Stage runs but output not cached | [Outputs & Caching](../reference/outputs.md#stage-runs-but-output-not-cached) |
| "Cannot pickle..." errors | [Defining Pipelines](../reference/pipelines.md#function-requirements) |
| Parameters not taking effect | [Parameters](../reference/parameters.md#troubleshooting) |
| Watch mode not detecting changes | [Watch Mode](../reference/watch.md#troubleshooting) |
| Lambda causes unnecessary re-runs | [Watch Mode](../reference/watch.md#lambda-causes-unnecessary-re-runs) |
| Remote push/pull fails | [Configuration](../reference/configuration.md#troubleshooting) |
| CI fails but local passes | [Configuration](../reference/configuration.md#ci-fails-but-local-passes) |

## Debugging Commands

```bash
# See all stages and their status
pivot list

# Detailed explanation for why a stage will/won't run
pivot explain stage_name

# Show what would run without executing
pivot run --dry-run

# Force a stage to run regardless of cache
pivot run stage_name --force

# Verbose logging for more detail
PIVOT_LOG_LEVEL=DEBUG pivot run
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

See: [Function Requirements](../reference/pipelines.md#function-requirements)

### Stage Re-runs Every Time

1. Check if you're using lambdas (non-deterministic fingerprints)
2. Check if a helper function changed
3. Run `pivot explain stage_name` to see what changed

See: [Stage Reruns Unexpectedly](../reference/outputs.md#stage-reruns-unexpectedly)

### Cache Not Shared in CI

1. Ensure lock files are committed: `git add .pivot/stages/*.lock`
2. Ensure remote is configured: `pivot config list`
3. Run `pivot pull` before `pivot run` in CI

See: [CI Fails but Local Passes](../reference/configuration.md#ci-fails-but-local-passes)
