# Cross-Repo Import

Import artifacts produced by another Pivot project without copying pipelines or re-running stages. Pivot downloads the file from the source project's remote storage and tracks it with a `.pvt` metadata file so you can update it later.

## When to Use This

- A model-training repo produces `model.pkl` and your serving repo needs it
- A shared feature-engineering repo produces datasets consumed by multiple downstream projects
- You want to pin to a specific git ref and update deliberately

## Prerequisites

The **source** repo must:

1. Have a Pivot project with at least one stage that produces the file you want
2. Have lock files committed (`.pivot/stages/*.lock`)
3. Have a [remote configured](./remote-storage.md) with cached outputs pushed to it

## Import an Artifact

```bash
pivot import <repo_url> <path> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `repo_url` | Git URL of the source project (HTTPS or SSH) |
| `path` | Path to the output file within the source project |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--rev` | `main` | Git ref to import from (branch, tag, or commit SHA) |
| `--out` | Same as source path | Local path to write the downloaded file |
| `--force` | off | Overwrite existing files and `.pvt` metadata |
| `--no-download` | off | Create `.pvt` metadata only, skip downloading the file |

### Basic Example

```bash
# Import a model from a training repo
pivot import https://github.com/team/ml-training model/best.pkl

# Import from a specific tag
pivot import https://github.com/team/ml-training model/best.pkl --rev v2.1.0

# Import to a different local path
pivot import https://github.com/team/ml-training model/best.pkl --out models/prod.pkl
```

### What Happens

1. Pivot resolves the git ref to a commit SHA
2. Reads the source repo's `.pivot/config.yaml` to find its remote storage URL
3. Reads lock files to find which stage produces the requested path and its content hash
4. Downloads the file from the source's S3 remote (hash-verified)
5. Creates a `.pvt` file tracking the import source

After importing `model/best.pkl`, your project has:

```
my-project/
├── model/
│   ├── best.pkl          # The downloaded artifact
│   └── best.pkl.pvt      # Import metadata
```

### Metadata-Only Import

Use `--no-download` when you want to record the dependency without downloading yet (useful in CI where you'll `pivot pull` later):

```bash
pivot import https://github.com/team/ml-training model/best.pkl --no-download
```

This creates the `.pvt` file but skips the download. Run `pivot checkout model/best.pkl` later to restore from cache, or `pivot update` to download.

## The `.pvt` File

The `.pvt` file is a YAML sidecar that tracks where the artifact came from. Commit it to version control — it's how Pivot knows this file is an import and where to update it from.

The `.pvt` file records: the source repo URL, the git ref and locked commit SHA, the stage and path that produce it, the content hash, and the remote storage URL.

> **Tip:** `.pvt` files work with `pivot checkout` — if the data file is missing but the `.pvt` exists and the hash is in your local cache, `pivot checkout` restores it.

## Update Imports

When the source repo produces new outputs, update your local copies:

```bash
# Update all imports in the project
pivot update

# Update a specific file
pivot update model/best.pkl

# You can also target the .pvt file directly
pivot update model/best.pkl.pvt

# Check what would change without modifying anything
pivot update --dry-run

# Override the git ref (e.g., switch from main to a release tag)
pivot update model/best.pkl --rev v3.0.0
```

**Options:**

| Option | Description |
|--------|-------------|
| `--rev` | Override the git ref for this update |
| `--dry-run` | Show available updates without applying them |

### Update Behavior

- Pivot re-resolves the tracked git ref to its current commit SHA
- If the resolved SHA differs from the locked one, it checks whether the content hash changed
- If the hash changed, it downloads the new file
- If only the SHA changed (same content), it updates the `.pvt` metadata without re-downloading

### Dry Run Output

```
$ pivot update --dry-run
Update available: model/best.pkl.pvt (a1b2c3d4 → e5f6g7h8)
Up to date: data/features.csv.pvt
```

## Pull Imported Files

On a fresh clone, imported data files won't exist — only the `.pvt` metadata is in version control. Restore them:

```bash
# Restore all tracked files (imports + stage outputs) from cache
pivot checkout

# Restore only missing files (safe if you have local modifications)
pivot checkout --only-missing

# Restore a specific import
pivot checkout model/best.pkl
```

If the file isn't in your local cache, you need the source remote configured, or you can use `pivot update` to re-download from the source.

## Workflow Example

**Team A** (ML training):

```bash
# Train and push outputs
pivot repro
pivot push
git add .pivot/stages/*.lock && git commit -m "Update model"
git push
```

**Team B** (serving, different repo):

```bash
# One-time setup: import the model
pivot import https://github.com/team-a/ml-training model/best.pkl --rev main
git add model/best.pkl.pvt && git commit -m "Track model import"

# Later: pull latest model
pivot update
git add model/best.pkl.pvt && git commit -m "Update model to latest"
```

**CI** (Team B's pipeline):

```bash
git clone https://github.com/team-b/serving
cd serving
pivot update            # Download latest imports
pivot repro             # Run serving pipeline
```

## Limitations

- **Individual files only.** You cannot import a directory output directly. Import individual files within the directory instead (e.g., `data/features/train.csv` not `data/features/`).
- **Requires source remote access.** The importing project needs network access to the source's S3 bucket.
- **GitHub and generic git.** GitHub repos use the API for metadata; other git hosts fall back to `git archive`.

## Related

- [Remote Storage](./remote-storage.md) — configuring S3 remotes for push/pull
- [Caching](../concepts/caching.md) — how content-addressed caching works
- [Artifacts & DAG](../concepts/artifacts-and-dag.md) — the artifact-first dependency model
