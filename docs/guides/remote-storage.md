# Remote Storage

Share cached pipeline outputs across machines using S3. Push outputs from your laptop, pull them in CI or on a teammate's machine — stages with matching [cached](../concepts/caching.md) outputs skip instead of re-running.

## Prerequisites

Install Pivot with S3 support:

```bash
uv add "pivot[s3]"
```

Pivot uses `aioboto3` under the hood. Standard AWS credentials (env vars, `~/.aws/credentials`, IAM roles) work automatically.

## Configure a Remote

```bash
# Add an S3 remote
pivot config set remotes.origin s3://my-bucket/pivot-cache

# Set it as default
pivot config set default_remote origin
```

Verify with:

```bash
pivot config list
```

The remote URL must be an S3 path (`s3://bucket/prefix`). Pivot stores cached files under this prefix, organized by content hash.

## Push Outputs

After running your pipeline locally, upload cached outputs to S3:

```bash
pivot repro
pivot push
```

Output:

```
Pushed to 'origin': 3 transferred, 0 skipped, 0 failed
```

### Selective Push

```bash
# Push outputs from specific stages only
pivot push train

# Push a specific file
pivot push model/best.pkl

# Preview what would be uploaded
pivot push --dry-run

# More parallel uploads for large caches
pivot push --jobs 40
```

### Options

| Option | Description |
|--------|-------------|
| `-r`, `--remote` | Target a named remote instead of the default |
| `-n`, `--dry-run` | Show what would be pushed without uploading |
| `-j`, `--jobs` | Number of parallel upload threads |

## Pull Outputs

On another machine, download cached outputs and restore them to the workspace:

```bash
pivot pull
```

`pivot pull` combines two operations: **fetch** (download from S3 to local cache) and **checkout** (restore files from local cache to workspace). This mirrors `git pull` = `git fetch` + `git merge`.

```
Fetched from 'origin': 3 transferred, 0 skipped, 0 failed
Restored 3 file(s)
```

After pulling, `pivot repro` skips stages whose outputs are already present and match their lock file hashes.

### Selective Pull

```bash
# Pull outputs for specific stages
pivot pull train

# Pull a specific file
pivot pull model/best.pkl

# Preview what would be downloaded
pivot pull --dry-run
```

### Options

| Option | Description |
|--------|-------------|
| `-r`, `--remote` | Pull from a named remote instead of the default |
| `-n`, `--dry-run` | Show what would be downloaded |
| `-j`, `--jobs` | Number of parallel download threads |
| `-f`, `--force` | Overwrite existing workspace files |
| `--only-missing` | Only restore files that don't exist on disk |
| `--checkout-mode` | `symlink`, `hardlink`, or `copy` (default: project config) |

## Fetch and Checkout Separately

For more control, split the two steps:

```bash
# Download to local cache only (no workspace changes)
pivot fetch

# Restore from local cache to workspace
pivot checkout
```

This is useful when you want to pre-populate the cache without touching the working tree, or restore specific files after fetching.

```bash
# Fetch everything, then selectively checkout
pivot fetch
pivot checkout model/best.pkl
pivot checkout --only-missing
```

## What Gets Transferred

| Data | Pushed/Pulled | Version controlled |
|------|---------------|-------------------|
| Cache files (`.pivot/cache/files/`) | Yes | No (`.gitignore`) |
| Lock files (`.pivot/stages/*.lock`) | No | Yes (commit these) |
| Config (`.pivot/config.yaml`) | No | Yes (commit this) |

**Lock files are the bridge.** They map output paths to content hashes. Push uploads the files referenced by those hashes; pull downloads them. Always commit lock files to version control so other machines know what to pull.

## Multiple Remotes

```bash
# Add a second remote
pivot config set remotes.backup s3://backup-bucket/pivot-cache

# Push to a specific remote
pivot push --remote backup

# Change the default
pivot config set default_remote backup
```

List configured remotes:

```bash
pivot remote list
```

## AWS Credentials

Pivot uses the standard AWS credential chain:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM roles (EC2, ECS, Lambda)
4. SSO/profile via `AWS_PROFILE`

No Pivot-specific credential configuration needed — if `aws s3 ls` works, `pivot push` works.

## GitHub Actions

For a complete GitHub Actions workflow, see the [CI Integration guide](ci-integration.md).
The key remote commands for CI are:

```bash
# In CI: pull cached outputs before running
pivot pull || true

# After successful run on main: push to remote
pivot push
```

## Typical Team Workflow

**Developer (local):**

```bash
# Iterate on a feature branch
pivot repro
pivot push
git add .pivot/stages/*.lock
git commit -m "Update pipeline outputs"
git push
```

**CI (after merge to main):**

```bash
pivot pull
pivot repro       # Only changed stages run
pivot push        # Update cache for the team
```

**Teammate (pulling latest):**

```bash
git pull
pivot pull        # Download cached outputs
pivot repro       # Instant — everything is cached
```

## Troubleshooting

**Push/pull fails with access denied:**

```bash
aws sts get-caller-identity     # Verify credentials
aws s3 ls s3://your-bucket/     # Verify bucket access
```

**Cache miss after pull — stages re-run unexpectedly:**

Lock files may be out of sync. Ensure `.pivot/stages/*.lock` files are committed:

```bash
git add .pivot/stages/*.lock
git commit -m "Sync lock files"
```

**Slow transfers:**

Increase parallelism:

```bash
pivot push --jobs 40
pivot pull --jobs 40
```

Or set the default: `pivot config set remote.jobs 40`.

## Related

- [Caching](../concepts/caching.md) — content-addressed caching and skip detection
- [Cross-Repo Import](./cross-repo-import.md) — import artifacts from other Pivot projects
- [CI Integration](./ci-integration.md) — more CI patterns and providers
