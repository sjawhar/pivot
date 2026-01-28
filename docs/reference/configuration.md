# Configuration

Pivot stores project configuration in `.pivot/config.yaml` for settings like remote storage.

## Configuration File

```yaml
# .pivot/config.yaml
remotes:
  origin: s3://my-bucket/pivot-cache
  backup: s3://backup-bucket/pivot-cache
default_remote: origin
```

## Remote Storage

Share cached outputs across machines and CI environments using S3.

### Setup

```bash
# Add a remote
pivot config set remotes.origin s3://my-bucket/pivot-cache

# Set as default
pivot config set default_remote origin
```

### Push to Remote

```bash
# Push all cached outputs
pivot push

# Push specific stages
pivot push train_model evaluate_model

# Dry run (show what would be pushed)
pivot push --dry-run

# Parallel uploads (default: 20)
pivot push --jobs 40
```

### Pull from Remote

```bash
# Pull all available outputs
pivot pull

# Pull specific stages
pivot pull train_model

# Dry run
pivot pull --dry-run
```

### What Gets Transferred

| Data | Pushed | Pulled |
|------|--------|--------|
| Cache files (`.pivot/cache/files/`) | Yes | Yes |
| Lock files (`.pivot/stages/*.lock`) | No | No |
| Config (`.pivot/config.yaml`) | No | No |
| State DB (`.pivot/state.lmdb/`) | No | No |

!!! note "Lock Files in Git"
    Lock files should be committed to git. They reference cached content by hash, enabling `pivot pull` to download the right files.

### AWS Credentials

Pivot uses the standard AWS credential chain:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM roles (EC2, ECS, Lambda)

## Managing Configuration

```bash
# List all config
pivot config list

# Get a specific value
pivot config get remotes.origin
pivot config get default_remote

# Set a value
pivot config set remotes.origin s3://new-bucket/cache
pivot config set default_remote backup

# Remove a value
pivot config unset remotes.backup
```

## Multiple Remotes

Configure separate remotes for different purposes:

```yaml
remotes:
  origin: s3://primary-bucket/pivot-cache     # Main remote
  backup: s3://backup-bucket/pivot-cache      # Backup
  staging: s3://staging-bucket/pivot-cache    # Staging environment
default_remote: origin
```

Push/pull to specific remotes:

```bash
# Push to non-default remote
pivot push --remote backup

# Pull from specific remote
pivot pull --remote staging
```

## Project Structure

Pivot stores state in the `.pivot/` directory:

```
.pivot/
├── config.yaml          # Project configuration
├── cache/
│   └── files/           # Content-addressable cache
│       ├── ab/
│       │   └── cdef0123...
│       └── ...
├── stages/
│   ├── preprocess.lock  # Per-stage lock files
│   └── train.lock
└── state.lmdb/          # State database
```

### What to Git-Track

| Item | Git-Track | Reason |
|------|-----------|--------|
| `.pivot/stages/*.lock` | Yes | Enables cache sharing |
| `.pivot/config.yaml` | Optional | Share remote config with team |
| `.pivot/cache/` | No | Large cached files |
| `.pivot/state.lmdb/` | No | Local state |

Typical `.gitignore`:

```
.pivot/cache/
.pivot/state.lmdb/
```

## Troubleshooting

### Remote Push/Pull Fails

**Symptom:** `pivot push` or `pivot pull` errors.

**Solutions:**

1. Check remote configuration:
   ```bash
   pivot config list
   ```

2. Verify AWS credentials:
   ```bash
   aws sts get-caller-identity
   ```

3. Check S3 bucket permissions:
   ```bash
   aws s3 ls s3://your-bucket/pivot-cache/
   ```

### CI Fails but Local Passes

**Symptom:** Pipeline works locally but stages re-run in CI.

**Cause:** Lock files not committed to git.

**Solution:** Commit lock files:

```bash
git add .pivot/stages/*.lock
git commit -m "Update lock files"
```

## See Also

- [Outputs & Caching](outputs.md) - Cache behavior
- [Remote Storage Tutorial](../tutorial/remote.md) - Complete CI setup guide
