# Troubleshooting

Common issues and solutions when using Pivot.

## Choosing Output Types

Use this decision tree when deciding which output type to use:

```
Is your output a visualization (chart, graph, image)?
├── Yes → Plot
└── No → Is it computed numbers for tracking over time?
         ├── Yes → Metric (git-tracked JSON/YAML)
         └── No → Regular output (cached, not git-tracked)
```

| Type | Git-Tracked | Cached | Use Case |
|------|-------------|--------|----------|
| Regular | No | Yes | Large files, intermediate data, models |
| `metrics` | Yes | No | Accuracy, loss, F1 scores for comparison |
| `plots` | No | Yes | Visualizations with optional display |

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

**Solution:** Commit `.pivot/stages/*.lock` files:

```bash
git add .pivot/stages/*.lock
git commit -m "Update lock files"
git push
```

Lock files record the fingerprint (code + params + deps + output hashes) that allow cache hits.

### "Stage function must be module-level"

**Symptom:** Error when running a stage.

**Cause:** Stage function is a lambda, closure, or defined in `__main__`.

**Solution:** Define functions at module level in importable files:

```python
# Bad - lambda
process = lambda: ...

# Bad - closure
def make_stage(threshold):
    def process():
        if value > threshold:  # Captures threshold!
            pass
    return process

# Bad - in __main__
if __name__ == '__main__':
    def my_stage():
        pass

# Good - module-level function
def process():
    pass
```

### "Cannot pickle..." / "Could not pickle the task"

**Symptom:** Error when running stages in parallel.

**Cause:** Stage function or its dependencies aren't serializable.

**Solution:** Avoid closures; use parameters instead:

```python
# stages.py
from pivot.stage_def import StageParams


class ProcessParams(StageParams):
    threshold: float = 0.5


def process(params: ProcessParams):
    if value > params.threshold:
        pass
```

```yaml
# pivot.yaml
stages:
  process:
    python: stages.process
    params:
      threshold: 0.5
```

**Why?** Pivot uses `ProcessPoolExecutor` with separate worker processes. Functions and their data must be serialized (pickled) to send to workers.

### Stage Runs But Output Not Cached

**Symptom:** Stage executes successfully but runs again next time.

**Cause:** Output file not declared or declared incorrectly.

**Solution:** Verify outputs are declared in the function's return type and paths match:

```python
# stages.py
import pathlib
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs


class ProcessOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.csv", loaders.PathOnly())]


def process(
    input_data: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV())],
) -> ProcessOutputs:
    out_path = pathlib.Path("output.csv")
    input_data.to_csv(out_path)
    return {"output": out_path}  # Must return the declared output
```

```yaml
# pivot.yaml
stages:
  process:
    python: stages.process
    deps:
      input_data: input.csv
    outs:
      output: output.csv
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

**Symptom:** `pivot run --watch` doesn't re-run when files change.

**Cause:** File not in dependencies or in a non-watched directory.

**Solution:**

1. Verify the file is declared as a dependency:
   ```yaml
   stages:
     process:
       python: stages.process
       deps:
         config: config.yaml
         data: data/
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
cat .pivot/stages/train.lock
```

Contains: code fingerprint, parameter hash, dependency hashes, output hashes.

## See Also

- [Code Fingerprinting](../architecture/fingerprinting.md) - How change detection works
- [Caching & Remote Storage](caching.md) - Cache behavior details
- [CLI Reference](../cli/index.md) - All available commands
