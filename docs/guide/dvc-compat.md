# DVC Compatibility

Pivot can export pipelines to DVC YAML format for code reviews and gradual migration.

## Exporting to DVC

```bash
# Generate dvc.yaml
pivot export

# Custom output path
pivot export --output my-pipeline.yaml
```

This creates a DVC-compatible `dvc.yaml` file:

```yaml
stages:
  preprocess:
    cmd: python -c "from pipeline import preprocess; preprocess()"
    deps:
      - data.csv
    outs:
      - processed.parquet

  train:
    cmd: python -c "from pipeline import train; train()"
    deps:
      - processed.parquet
    outs:
      - model.pkl
```

## Export Specific Stages

```bash
# Export only selected stages
pivot export preprocess train
```

## Use Cases

### Code Reviews

Export for team members who review via DVC YAML:

```bash
# Before PR
pivot export
git add dvc.yaml
git commit -m "Update pipeline definition"
```

### Migration from DVC

Run Pivot and DVC side-by-side during migration:

```bash
# Run with Pivot
pivot run

# Validate outputs match DVC
pivot export
dvc repro --dry  # Should show nothing to run
```

### CI/CD Integration

Some CI systems expect DVC YAML:

```yaml
# .github/workflows/pipeline.yml
- name: Export and validate
  run: |
    pivot export
    dvc repro --dry
```

## Limitations

The export captures:

- Stage commands
- Dependencies
- Outputs (with cache/persist settings)
- Metrics and plots

Not exported:

- Automatic code fingerprinting (DVC doesn't support this)
- Mutex groups
- Pydantic parameter types (exported as plain values)

## See Also

- [API Reference: dvc_compat](../reference/pivot/dvc_compat.md) - Full API documentation
