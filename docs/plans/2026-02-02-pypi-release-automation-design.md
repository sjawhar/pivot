# PyPI Release Automation Design

## Overview

Automate releases to PyPI using python-semantic-release with uv integration. Releases trigger automatically after tests pass on main when commits include releasable changes (`feat:`, `fix:`, breaking changes).

## Components

### 1. python-semantic-release Configuration

Add to `pyproject.toml`:

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
build_command = "uv lock --upgrade-package pivot && git add uv.lock && uv build"
```

Note: Other settings use defaults (remote.type="github", publish.upload_to_vcs_release=true, changelog file="CHANGELOG.md").

Also change version from `0.1.0-dev` to `0.1.0`.

### 2. GitHub Actions Workflow

Create `.github/workflows/release.yaml`:

```yaml
name: Release

on:
  workflow_run:
    workflows: ["Tests"]
    types: [completed]
    branches: [main]

concurrency:
  group: release
  cancel-in-progress: false

env:
  UV_VERSION: 0.9.18

jobs:
  release:
    runs-on: ubuntu-24.04
    environment: pypi
    if: >-
      github.event.workflow_run.conclusion == 'success' &&
      !startsWith(github.event.workflow_run.head_commit.message || '', 'chore(release):')
    permissions:
      contents: write
      id-token: write

    steps:
      - uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd # v5
        with:
          fetch-depth: 0
          ref: ${{ github.event.workflow_run.head_sha }}
          ssh-key: ${{ secrets.DEPLOY_KEY }}
          persist-credentials: true

      - name: Configure git for semantic-release
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Install uv
        uses: astral-sh/setup-uv@85856786d1ce8acfbcc2f13a5f3fbd6b938f9f41 # v7.1.2
        with:
          version: "${{ env.UV_VERSION }}"
          enable-cache: true
          cache-dependency-glob: uv.lock

      - name: Set up Python
        uses: actions/setup-python@e797f83bcb11b83ae66e0230d6156d7c80228e7c # v6.0.0
        with:
          python-version-file: pyproject.toml

      - name: Run semantic-release
        id: release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          set -uo pipefail
          uvx --from="python-semantic-release@10" semantic-release version || exit $?
          uvx --from="python-semantic-release@10" semantic-release publish || exit $?
          if [ -d dist ] && [ -n "$(ls -A dist/)" ]; then
            echo "released=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Publish to PyPI
        if: steps.release.outputs.released == 'true'
        run: uv publish
```

### 3. Deploy Key Setup (Manual)

Required for semantic-release to push version commits to protected main branch.

1. Generate SSH key pair (no passphrase):
   ```bash
   ssh-keygen -t ed25519 -C "github-actions" -N "" -f deploy_key
   ```

2. Add public key as Deploy Key (Settings → Deploy keys):
   - Title: `semantic-release`
   - Key: contents of `deploy_key.pub`
   - Check "Allow write access"

3. Add private key as secret (Settings → Secrets and variables → Actions):
   - Name: `DEPLOY_KEY`
   - Value: contents of `deploy_key`

4. Configure ruleset (Settings → Rules → Rulesets → New):
   - Target: main branch
   - Bypass list: Add "Deploy keys"
   - Rules: Require PR, status checks, etc.

5. Delete local key files after setup:
   ```bash
   rm deploy_key deploy_key.pub
   ```

### 4. PyPI Trusted Publishing Setup (Manual)

1. Go to https://pypi.org/manage/account/publishing/
2. Add a "Pending Publisher":
   - PyPI Project Name: `pivot`
   - Owner: GitHub username/org
   - Repository: repo name
   - Workflow name: `release.yaml`
   - Environment: `pypi`

3. Create GitHub environment:
   - Repo Settings → Environments → New environment
   - Name: `pypi`
   - Optionally add reviewers for manual approval

## Commit Convention

After implementation, use conventional commits:

| Prefix | Version Bump | Example |
|--------|--------------|---------|
| `feat:` | Minor (0.1.0 → 0.2.0) | `feat: add watch mode` |
| `fix:` | Patch (0.1.0 → 0.1.1) | `fix: handle empty input` |
| `feat!:` or `BREAKING CHANGE:` | Major (0.1.0 → 1.0.0) | `feat!: change API` |
| `docs:`, `chore:`, `ci:`, `refactor:`, `test:` | No release | `docs: update README` |

## References

- [python-semantic-release uv integration](https://python-semantic-release.readthedocs.io/en/latest/configuration/configuration-guides/uv_integration.html)
- [uv GitHub Actions guide](https://docs.astral.sh/uv/guides/integration/github/)
- [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/)
