# Remote Storage

Share cached pipeline outputs across machines using S3 or Google Cloud Storage (GCS). Push outputs from your laptop, pull them in CI or on a teammate's machine — stages with matching [cached](../concepts/caching.md) outputs skip instead of re-running.

## Prerequisites

Install Pivot with the extra for your storage:

```bash
uv add "pivot[s3]"   # S3, through aioboto3
uv add "pivot[gcs]"  # GCS, through gcloud-aio-storage
```

Credentials come from each cloud's standard chain; see [Credentials](#credentials).

## Configure a Remote

```bash
# Add an S3 remote
pivot config set remotes.origin s3://my-bucket/pivot-cache

# Or a GCS remote
pivot config set remotes.origin gs://my-bucket/pivot-cache

# Set it as default
pivot config set default_remote origin
```

Verify with:

```bash
pivot config list
```

The remote URL is `s3://bucket/prefix` or `gs://bucket/prefix`; any other scheme is rejected when the config is set or loaded. Pivot stores cached files under this prefix, organized by content hash (`<prefix>/files/ab/cdef…` for hash `abcdef…`), the same layout on both clouds.

## Push Outputs

After running your pipeline locally, upload cached outputs to the remote:

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

`pivot pull` combines two operations: **fetch** (download from the remote to local cache) and **checkout** (restore files from local cache to workspace). This mirrors `git pull` = `git fetch` + `git merge`.

```
Fetched from 'origin': 3 transferred, 0 skipped, 0 failed
Restored 3 file(s)
```

After pulling, `pivot repro` skips stages whose outputs are already present and match their lock file hashes.

Every downloaded file is checked against the hash it is stored under before it enters the local cache. A file that does not match is discarded and counted as failed (`Hash mismatch: expected …, got …`), and `pivot pull` exits without restoring anything. To repair the remote, delete that object from it, then run `pivot push` on a machine whose cache has a good copy and has not run `pivot push`, `pivot pull`, `pivot fetch`, or `pivot status --remote` against this remote for this object. Pivot records hashes those commands upload, download, or find, then skips recorded hashes. If every machine with a good copy has recorded it, delete `.pivot/state.lmdb/` before pushing; this also clears run history and local caches. The check catches corruption in storage or transfer. It does not authenticate who wrote the file.

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
| `--only-missing` | Only restore files that don't exist on disk (the default without `--force`); a directory is replaced when every changed or extra file is pivot's own earlier output (an extra empty directory holds nothing to lose), and refused (naming the files) when one holds a local change the cache lacks |
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
| Cache files and directory listing objects (`.pivot/cache/files/`) | Yes | No (`.gitignore`) |
| Lock files and directory listings (`.pivot/stages/**/*.lock`, `.pivot/stages/**/*.listing`) | No | Yes (commit both) |
| Config (`.pivot/config.yaml`) | No | Yes (commit this) |

**Lock files and listings are the bridge.** A lock maps each stage output to a content hash; a
directory's members are in its committed listing file. Push uploads the listing object and the
files it names; pull downloads them. Always commit lock files and their listing files so other
machines know what to pull.

Pulling a stage also fetches the outputs of the upstream stages it depends on, and any dependency tracked with `pivot track`. It never fetches raw inputs committed to git (`data/raw/...`), metric outputs (git carries both), or files brought in with `pivot import`: those come from the source repo's remote, and `pivot update` refreshes them.

## Multiple Remotes

```bash
# Add a second remote (remotes may be on different clouds)
pivot config set remotes.backup gs://backup-bucket/pivot-cache

# Push to a specific remote
pivot push --remote backup

# Change the default
pivot config set default_remote backup
```

List configured remotes:

```bash
pivot remote list
```

## Credentials

No Pivot-specific credential configuration is needed: each cloud's standard chain applies.

**S3** uses the standard AWS credential chain:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM roles (EC2, ECS, Lambda)
4. SSO/profile via `AWS_PROFILE`

If `aws s3 ls` works, `pivot push` works.

**GCS** uses Application Default Credentials:

1. A service account key file named by `GOOGLE_APPLICATION_CREDENTIALS`
2. Your user credentials from `gcloud auth application-default login`
3. The attached service account on GCE, GKE, or Cloud Run (metadata server)

Pivot needs `storage.objects.get` and `storage.objects.list` to pull (Storage Object Viewer grants both). Pushing also needs `storage.objects.create` and `storage.objects.delete`: a push can overwrite an object that is already there (two machines pushing the same output at once, or a retry after a reply was lost), and an upload that overwrites an object requires `storage.objects.delete` ([IAM permissions for JSON methods](https://cloud.google.com/storage/docs/access-control/iam-json), Objects: insert). A create-only grant such as Storage Object Creator is therefore not enough; Storage Object User grants all four ([IAM roles](https://cloud.google.com/storage/docs/access-control/iam-roles)). `STORAGE_EMULATOR_HOST` points Pivot at a GCS emulator such as fake-gcs-server, which needs no credentials.

`remote.retries`, `remote.connect_timeout`, and `remote.jobs` apply to both clouds.

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
git add .pivot/stages
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
aws sts get-caller-identity              # S3: verify credentials
aws s3 ls s3://your-bucket/              # S3: verify bucket access
gcloud auth application-default login    # GCS: refresh Application Default Credentials
gcloud storage ls gs://your-bucket/      # GCS: verify bucket access
```

A GCS permission error names the object and the missing permission, for example `Permission 'storage.objects.create' denied`.

**Cache miss after pull — stages re-run unexpectedly:**

Lock files may be out of sync. Ensure the lock files under `.pivot/stages/` are committed:

```bash
git add .pivot/stages
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
