# Caching & Remote Storage

Pivot uses content-addressable storage for efficient caching and supports S3 for sharing cached outputs across machines.

## Local Cache

Pivot stores cached outputs in `.pivot/cache/`:

```
.pivot/
├── cache/
│   └── files/
│       ├── ab/
│       │   └── cdef0123...  # File content keyed by hash
│       └── ...
└── stages/
    ├── preprocess.lock      # Per-stage lock file
    └── train.lock
```

### How Caching Works

1. **Stage runs** - outputs are created
2. **Hash computed** - xxhash64 of file content
3. **Stored in cache** - content-addressable by hash
4. **Lock file updated** - records fingerprint (code + params + deps + output hashes)

On subsequent runs:

1. **Fingerprint compared** - code, params, deps checked
2. **If match** - restore outputs from cache (skip execution)
3. **If changed** - re-run stage

### Checkout Modes

When restoring from cache:

| Mode | Description | Use Case |
|------|-------------|----------|
| `hardlink` | Hard link to cache (default) | Fast, space-efficient |
| `symlink` | Symbolic link to cache | Visual clarity |
| `copy` | Full copy from cache | When modification needed |

```bash
pivot checkout --checkout-mode copy
```

!!! warning "IncrementalOut Uses Copy Mode"
    `IncrementalOut` always uses copy mode internally. Since the stage modifies the file in-place, using hardlinks or symlinks would corrupt the cache. Pivot handles this automatically.

### Why xxhash64?

Pivot uses xxhash64 for content hashing:

- **10x faster** than MD5 with equivalent collision resistance for caching
- Non-cryptographic (not for security, just deduplication)
- 64-bit hash provides sufficient uniqueness for cache keys

## Remote Storage (S3)

Share cached outputs across machines and CI environments.

### Setup

```bash
# Add a remote
pivot remote add origin s3://my-bucket/pivot-cache

# Set as default
pivot remote default origin

# Or combine
pivot remote add origin s3://my-bucket/pivot-cache --default
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

### Remote Configuration

Configuration is stored in `.pivot/config.yaml`:

```yaml
remotes:
  origin: s3://my-bucket/pivot-cache
  backup: s3://backup-bucket/pivot-cache
default_remote: origin
```

### AWS Credentials

Pivot uses the standard AWS credential chain:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM roles (EC2, ECS, Lambda)

### What Gets Transferred

| Data | Pushed | Pulled |
|------|--------|--------|
| Cache files (`.pivot/cache/files/`) | Yes | Yes |
| Lock files (`.pivot/stages/*.lock`) | No | No |
| Config (`.pivot/config.yaml`) | No | No |
| State DB (`.pivot/state.lmdb/`) | No | No |

!!! note "Lock Files in Git"
    Lock files should be committed to git. They reference cached content by hash, enabling `pivot pull` to download the right files.

### Typical Workflow

**Developer machine:**

```bash
pivot run            # Run pipeline
pivot push           # Upload cached outputs
git commit -m "..."  # Commit lock files
git push
```

**CI or another machine:**

```bash
git pull
pivot pull           # Download cached outputs
pivot run            # Stages with cached outputs skip
```

## Managing Remotes

```bash
# List configured remotes
pivot remote list

# Remove a remote
pivot remote remove backup

# Change default
pivot remote default backup
```

## See Also

- [API Reference: cache](../reference/pivot/cache.md) - Full API documentation
