# Remote Storage & CI

This tutorial shows how to share cached outputs across machines and integrate Pivot into CI pipelines.

## Prerequisites

- Complete the previous tutorials
- AWS credentials configured (for S3)
- A GitHub repository (for CI example)

## Configure a Remote

Add an S3 bucket as your cache remote:

```bash
# Add a remote
pivot config set remotes.origin s3://my-bucket/pivot-cache

# Set as default
pivot config set default_remote origin
```

Verify configuration:

```bash
pivot config list
```

Output:

```yaml
remotes:
  origin: s3://my-bucket/pivot-cache
default_remote: origin
```

## Push Cached Outputs

After running your pipeline locally:

```bash
# Run the pipeline
pivot run

# Push all cached outputs to S3
pivot push
```

Push shows what's being uploaded:

```
Pushing 2 files to s3://my-bucket/pivot-cache
  processed.parquet (2.4 MB)
  model.pkl (156 KB)
Done in 1.2s
```

### Selective Push

```bash
# Push specific stages only
pivot push train

# Dry run to see what would be pushed
pivot push --dry-run

# More parallel uploads for large caches
pivot push --jobs 40
```

## Pull Cached Outputs

On another machine or in CI:

```bash
# Clone the repo
git clone https://github.com/you/your-project.git
cd your-project

# Pull cached outputs from S3
pivot pull

# Run - stages with cached outputs skip
pivot run
```

Pull output:

```
Pulling 2 files from s3://my-bucket/pivot-cache
  processed.parquet (2.4 MB)
  model.pkl (156 KB)
Done in 0.8s
```

### Selective Pull

```bash
# Pull specific stages
pivot pull train

# Dry run
pivot pull --dry-run
```

## What Gets Transferred

| Data | Pushed | Pulled |
|------|--------|--------|
| Cache files (`.pivot/cache/files/`) | Yes | Yes |
| Lock files (`.pivot/stages/*.lock`) | No | No |
| Config (`.pivot/config.yaml`) | No | No |

Lock files should be committed to git. They reference cached content by hash, enabling `pivot pull` to download the right files.

## GitHub Actions Integration

Create `.github/workflows/pipeline.yml`:

```yaml
name: Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  run:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Pull cache
        run: uv run pivot pull || true  # OK if nothing cached yet

      - name: Run pipeline
        run: uv run pivot run

      - name: Push cache
        run: uv run pivot push
        if: github.ref == 'refs/heads/main'  # Only push from main
```

### Key Points

1. **Pull before run** - Download any previously cached outputs
2. **Run the pipeline** - Stages with cached outputs skip
3. **Push after run** - Upload new outputs (only on main to avoid conflicts)
4. **Secrets** - Store AWS credentials as GitHub secrets

## AWS Credentials

Pivot uses the standard AWS credential chain:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM roles (EC2, ECS, Lambda)

For CI, use environment variables via secrets.

## Multiple Remotes

Configure backup or staging remotes:

```bash
# Add a backup remote
pivot config set remotes.backup s3://backup-bucket/pivot-cache

# Push to specific remote
pivot push --remote backup

# Change default
pivot config set default_remote backup
```

## Typical Workflow

**Developer machine:**

```bash
# Work on feature branch
git checkout -b feature/improve-model

# Run and iterate
pivot run

# Push cache before PR
pivot push
git add .pivot/stages/*.lock
git commit -m "Update model with new approach"
git push origin feature/improve-model
```

**After PR merge to main:**

```bash
# CI pulls existing cache
pivot pull

# CI runs - only changed stages execute
pivot run

# CI pushes updated cache
pivot push
```

**Team member pulls latest:**

```bash
git pull origin main
pivot pull  # Download cached outputs
pivot run   # Instant - everything cached
```

## Troubleshooting

**Push/pull fails with access denied:**

```bash
# Check AWS credentials
aws sts get-caller-identity

# Check bucket access
aws s3 ls s3://your-bucket/pivot-cache/
```

**Cache miss after pull:**

Lock files may not match. Ensure `.pivot/stages/*.lock` files are committed:

```bash
git add .pivot/stages/*.lock
git commit -m "Update lock files"
```

## Next Steps

- [Caching Reference](../reference/outputs.md#caching) - Cache behavior details
- [Configuration Reference](../reference/configuration.md) - All config options
