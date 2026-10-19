# Cross-Repo Import

Import a stage output produced by another Pivot project without copying its pipeline or re-running its stages. Pivot downloads the file from the source project's remote storage and tracks it with a `.pvt` metadata file so you can update it later.

## When to Use This

- A model-training repo produces a model and your serving repo needs it
- A shared data-preparation repo produces datasets consumed by several downstream projects
- You want to pin to a specific git ref and update deliberately

## Prerequisites

The **source** repo must:

1. Have a Pivot project with a stage that produces the artifact you want
2. Have its lock files committed (`.pivot/stages/`)
3. Have a [remote configured](./remote-storage.md) with cached outputs pushed to it

## Import an Artifact

```bash
pivot import <repo_url> <artifact> --out <path> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `repo_url` | Git URL of the source project (HTTPS, SSH, or `file://`) |
| `artifact` | Identity of the output in the source project: `pipeline/stage` for a single-output stage, or `pipeline/stage:key` for one field of a multi-output stage (`pivot list --deps` in the source repo prints these) |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--rev` | `main` | Git ref to import from (branch, tag, or commit SHA) |
| `--out` | the artifact's name, with `/` for `:` (`training/evaluate/scores`) | Local path to write the downloaded file |
| `--force` | off | Overwrite existing files and `.pvt` metadata |
| `--no-download` | off | Create `.pvt` metadata only, skip downloading the file |

Always pass `--out`: the downloaded file is an input for your own pipeline, so put it under `data/raw/` (or `data/external/`) and read it with `pipeline.input()`.

### Basic Example

```bash
# Import the output of training/train
pivot import https://github.com/team/ml-training training/train --out data/raw/model.pkl

# Import one field of a multi-output stage, pinned to a tag
pivot import https://github.com/team/ml-training training/evaluate:scores --rev v2.1.0 --out data/raw/scores.yaml
```

Then, in your `pipeline.py`:

```python
model = pipeline.input("model.pkl")
```

### What Happens

1. Pivot resolves the git ref to a commit SHA
2. Reads the source repo's `.pivot/config.yaml` to find its remote storage URL
3. Reads the source lock files to find the artifact and its content hash
4. Downloads the file from the source's S3 remote (hash-verified)
5. Creates a `.pvt` file tracking the import source

After the first import above, your project has:

```
my-project/
├── data/raw/
│   ├── model.pkl          # The downloaded artifact
│   └── model.pkl.pvt      # Import metadata
```

### Metadata-Only Import

Use `--no-download` to record the import without downloading yet:

```bash
pivot import https://github.com/team/ml-training training/train --out data/raw/model.pkl --no-download
```

This creates the `.pvt` file but skips the download. Run `pivot update` later to download it.

## The `.pvt` File

The `.pvt` file is a YAML sidecar that tracks where the artifact came from. Commit it to version control; it is how Pivot knows this file is an import and where to update it from.

The `.pvt` file records: the source repo URL, the git ref and locked commit SHA, the producing stage and artifact identity, the content hash, and the remote storage URL.

> **Tip:** `.pvt` files work with `pivot checkout`: if the data file is missing but the `.pvt` exists and the hash is in your local cache, `pivot checkout` restores it.

## Update Imports

When the source repo produces new outputs, update your local copies:

```bash
# Update all imports in the project
pivot update

# Update a specific file
pivot update data/raw/model.pkl

# You can also target the .pvt file directly
pivot update data/raw/model.pkl.pvt

# Check what would change without modifying anything
pivot update --dry-run

# Override the git ref (e.g., switch from main to a release tag)
pivot update data/raw/model.pkl --rev v3.0.0
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

## Pull Imported Files

On a fresh clone, imported data files won't exist; only the `.pvt` metadata is in version control. Restore them:

```bash
# Restore all tracked files (imports + stage outputs) from cache
pivot checkout

# Restore only missing files (safe if you have local modifications)
pivot checkout --only-missing

# Restore a specific import
pivot checkout data/raw/model.pkl
```

If the file isn't in your local cache, run `pivot update` to re-download it from the source.

## Workflow Example

**Team A** (ML training):

```bash
# Train and push outputs
pivot repro
pivot push
git add .pivot/stages && git commit -m "Update model"
git push
```

**Team B** (serving, different repo):

```bash
# One-time setup: import the model
pivot import https://github.com/team-a/ml-training training/train --rev main --out data/raw/model.pkl
git add data/raw/model.pkl.pvt && git commit -m "Track model import"

# Later: pull latest model
pivot update
git add data/raw/model.pkl.pvt && git commit -m "Update model to latest"
```

**CI** (Team B's pipeline):

```bash
git clone https://github.com/team-b/serving
cd serving
pivot update            # Download latest imports
pivot repro             # Run serving pipeline
```

## Limitations

- **Individual files only.** A directory output cannot be imported whole; import one file of it by the name `pivot list` gives it there (`pipeline/stage:path`, or `pipeline/stage:field/path` for a `TypedDict` field). An import made before Pivot used these names records a file of a stage's return value as `pipeline/stage/path`. Such a `.pvt` fails `pivot status --check-imports`, `pivot update --dry-run` and `pivot update`, each naming the new name and the `pivot import … --force` command that re-imports it under the new name.
- **Requires source remote access.** The importing project needs network access to the source's S3 bucket.
- **GitHub and generic git.** GitHub repos use the API for metadata; other git hosts fall back to `git archive`.

## Related

- [Remote Storage](./remote-storage.md): configuring S3 remotes for push/pull
- [Caching](../concepts/caching.md): how content-addressed caching works
- [Artifacts & DAG](../concepts/artifacts-and-dag.md): artifact identities
