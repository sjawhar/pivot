# Troubleshooting

Common issues and solutions when using Pivot.

## Choosing Output Types

Use this decision tree when deciding which output type to use:

```
Is your output a visualization (chart, graph, image)?
├── Yes → Plot
└── No → Is it computed numbers for tracking over time?
         ├── Yes → Metric (git-tracked JSON/YAML)
         └── No → Out (cached, not git-tracked)
```

| Type | Git-Tracked | Cached | Use Case |
|------|-------------|--------|----------|
| `Out` | No | Yes | Large files, intermediate data, models |
| `Metric` | Yes | No | Accuracy, loss, F1 scores for comparison |
| `Plot` | No | Yes | Visualizations with optional display |

See [Output Types](outputs.md) for detailed documentation.

## Common Issues

### Pipeline Reruns Unexpectedly

**Symptom:** A stage runs even though you didn't change it.

**Cause:** Pivot detected a change in code, parameters, or dependencies.

**Solution:** Use `pivot explain` to see what changed:

```bash
$ pivot explain train

Stage: train
  Status: WILL RUN
  Reason: Code dependency changed

  Code changes:
    func:helper_a
      Old: 5995c853
      New: a1b2c3d4
      File: src/utils.py:15
```

Common triggers:
- Modified a helper function the stage calls
- Changed default argument values
- Updated a module the stage imports

### CI Fails but Local Passes

**Symptom:** Pipeline works locally but stages re-run in CI.

**Cause:** Lock files not committed to git.

**Solution:** Commit `.pivot/*.lock.yaml` files:

```bash
git add .pivot/stages/*.lock.yaml
git commit -m "Update lock files"
git push
```

Lock files record the fingerprint (code + params + deps + output hashes) that allow cache hits.

### "Stage function must be module-level"

**Symptom:** Error when registering a stage.

**Cause:** Stage function is a lambda, closure, or defined in `__main__`.

**Solution:** Define functions at module level in importable files:

```python
# Bad - lambda
process = stage(deps=['x'], outs=['y'])(lambda: ...)

# Bad - closure
def make_stage(threshold):
    @stage(deps=['x'], outs=['y'])
    def process():
        if value > threshold:  # Captures threshold!
            pass
    return process

# Bad - in __main__
if __name__ == '__main__':
    @stage(deps=['x'], outs=['y'])
    def my_stage():
        pass

# Good - module-level function
@stage(deps=['x'], outs=['y'])
def process():
    pass
```

### "Cannot pickle..." / "Could not pickle the task"

**Symptom:** Error when running stages in parallel.

**Cause:** Stage function or its dependencies aren't serializable.

**Solution:** Avoid closures; use parameters instead:

```python
# Bad - closure captures local variable
THRESHOLD = 0.5

@stage(deps=['data.csv'])
def process():
    if value > THRESHOLD:  # Global variable captured
        pass

# Good - use parameters
class ProcessParams(pydantic.BaseModel):
    threshold: float = 0.5

@stage(deps=['data.csv'], params=ProcessParams)
def process(params: ProcessParams):
    if value > params.threshold:
        pass
```

**Why?** Pivot uses `ProcessPoolExecutor` with separate worker processes. Functions and their data must be serialized (pickled) to send to workers.

### Stage Runs But Output Not Cached

**Symptom:** Stage executes successfully but runs again next time.

**Cause:** Output file not declared or declared incorrectly.

**Solution:** Verify outputs are declared and paths match:

```python
@stage(deps=['input.csv'], outs=['output.csv'])  # Must match actual path
def process():
    import pandas as pd
    df = pd.read_csv('input.csv')
    df.to_csv('output.csv')  # Must write to declared path
```

Check that the file was actually created:

```bash
ls -la output.csv
```

### Remote Push/Pull Fails

**Symptom:** `pivot push` or `pivot pull` errors.

**Cause:** Missing or incorrect AWS credentials, or remote not configured.

**Solution:**

1. Check remote configuration:
   ```bash
   pivot remote list
   ```

2. Verify AWS credentials:
   ```bash
   aws sts get-caller-identity
   ```

3. Check S3 bucket permissions:
   ```bash
   aws s3 ls s3://your-bucket/pivot-cache/
   ```

### Watch Mode Not Detecting Changes

**Symptom:** `pivot watch` doesn't re-run when files change.

**Cause:** File not in dependencies or in a non-watched directory.

**Solution:**

1. Verify the file is declared as a dependency:
   ```python
   @stage(deps=['config.yaml', 'data/'])  # Both file and directory
   def process():
       pass
   ```

2. Check that the file is within the project directory

3. Some editors use atomic saves (write to temp file, then rename) which may need a brief delay

## Debugging Tips

### Check Stage Status

```bash
# See all stages and their status
pivot list

# Detailed explanation for specific stage
pivot explain stage_name
```

### Force Re-run

```bash
# Force a specific stage to run regardless of cache
pivot run stage_name --force
```

### Verbose Logging

Set the log level for more detail:

```bash
PIVOT_LOG_LEVEL=DEBUG pivot run
```

### Inspect Lock Files

Lock files are YAML and human-readable:

```bash
cat .pivot/stages/train.lock.yaml
```

Contains: code fingerprint, parameter hash, dependency hashes, output hashes.

## See Also

- [Code Fingerprinting](../architecture/fingerprinting.md) - How change detection works
- [Caching & Remote Storage](caching.md) - Cache behavior details
- [CLI Reference](../cli/index.md) - All available commands
