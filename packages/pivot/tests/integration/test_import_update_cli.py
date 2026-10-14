"""`pivot import` and `pivot update` against a real local git source repo and a moto S3 remote.

The source repo holds only what a Pivot project commits (``.pivot/config.yaml`` and stage lock
files); the remote holds the content-addressed files those locks point at. Every command runs
through the CLI in a separate consumer project.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from typing import TYPE_CHECKING

import boto3
import pytest
import xxhash
import yaml

import conftest
from pivot import cli, project, types
from pivot.remote import storage as remote_storage
from pivot.storage import lock, track

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Mapping

    import click.testing


class _Bucket:
    """A moto S3 bucket laid out as a Pivot remote under the ``cache/`` prefix."""

    url: str
    _name: str

    def __init__(self) -> None:
        self._name = f"pivot-import-{uuid.uuid4().hex[:8]}"
        self._client().create_bucket(Bucket=self._name)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
        self.url = f"s3://{self._name}/cache"

    @staticmethod
    def _client() -> object:
        return boto3.Session(region_name="us-east-1").client("s3")

    def put(self, content: bytes, *, as_hash: str | None = None) -> str:
        """Store content under its own hash (or under ``as_hash``) and return that hash."""
        file_hash = as_hash or xxhash.xxh64(content).hexdigest()
        self._client().put_object(  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
            Bucket=self._name, Key=remote_storage._hash_to_key("cache/", file_hash), Body=content
        )
        return file_hash


class _Source:
    """A git repo holding a Pivot project's committed state: remote config and stage locks."""

    root: pathlib.Path

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        root.mkdir()
        conftest.init_git_repo(root)
        # `git archive --remote` by commit SHA requires this for local repos
        self._git("config", "uploadArchive.allowUnreachable", "true")
        self._git("checkout", "-b", "main")

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    def write(self, rel_path: str, content: str) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def write_config(self, remote_url: str) -> None:
        self.write(".pivot/config.yaml", yaml.safe_dump({"remotes": {"origin": remote_url}}))

    def write_lock(self, stage_name: str, outputs: Mapping[str | None, types.HashInfo]) -> None:
        """Write the lock file Pivot records for a stage with these outputs."""
        lock.StageLock(stage_name, lock.get_stages_dir(self.root / ".pivot")).write(
            types.LockData(
                code_manifest={"self:stage": "0123456789abcdef"},
                params={},
                dep_hashes={},
                output_hashes={
                    types.ArtifactIdentity(stage_name, key): hash_info
                    for key, hash_info in outputs.items()
                },
            )
        )

    def commit(self, message: str) -> str:
        self._git("add", ".")
        self._git("commit", "--allow-empty", "-m", message)
        return self._git("rev-parse", "HEAD")


@pytest.fixture
def bucket(moto_patch_session: None) -> _Bucket:
    return _Bucket()


@pytest.fixture
def source(tmp_path: pathlib.Path) -> _Source:
    return _Source(tmp_path.resolve() / "source")


@pytest.fixture
def consumer(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    root = tmp_path.resolve() / "consumer"
    (root / ".pivot").mkdir(parents=True)
    (root / ".git").mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", root)
    return root


def _helper_publish(
    source: _Source, bucket: _Bucket, stage_name: str, content: bytes, message: str
) -> str:
    """Upload content, point the stage's lock at it, commit, and return the new HEAD."""
    source.write_config(bucket.url)
    source.write_lock(stage_name, {None: types.FileHash(hash=bucket.put(content))})
    return source.commit(message)


def _helper_import(
    runner: click.testing.CliRunner, source: _Source, artifact: str, out: str
) -> pathlib.Path:
    result = runner.invoke(cli.cli, ["import", str(source.root), artifact, "--out", out])
    assert result.exit_code == 0, result.output
    assert "Imported" in result.output
    return project.get_project_root() / out


def _helper_rev_lock(data_path: pathlib.Path) -> str:
    pvt_data = track.read_pvt_file(track.get_pvt_path(data_path))
    assert pvt_data is not None and "source" in pvt_data, "an import .pvt should exist"
    return pvt_data["source"]["rev_lock"]


# =============================================================================
# import
# =============================================================================


def test_import_downloads_artifact_and_records_its_source(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    head = _helper_publish(source, bucket, "smoke/total", b"sum: 6\n", "v1")

    data_path = _helper_import(runner, source, "smoke/total", "data/total.yaml")

    assert data_path.read_bytes() == b"sum: 6\n"
    pvt_data = track.read_pvt_file(track.get_pvt_path(data_path))
    assert pvt_data is not None
    assert pvt_data["hash"] == xxhash.xxh64(b"sum: 6\n").hexdigest()
    assert pvt_data["size"] == len(b"sum: 6\n")
    assert "source" in pvt_data
    assert pvt_data["source"]["rev_lock"] == head
    assert pvt_data["source"]["remote"] == bucket.url


def test_import_file_inside_directory_output(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    """A directory output is imported one file at a time, by ``<identity>/<relpath>``."""
    content = b"a,b\n1,2\n"
    file_hash = bucket.put(content)
    source.write_config(bucket.url)
    source.write_lock(
        "smoke/images",
        {
            None: types.DirHash(
                hash="0123456789abcdef",
                manifest=[
                    types.DirManifestEntry(
                        relpath="part/a.csv", hash=file_hash, size=len(content), isexec=False
                    )
                ],
            )
        },
    )
    source.commit("directory output")

    whole = runner.invoke(cli.cli, ["import", str(source.root), "smoke/images"])
    assert whole.exit_code != 0
    assert "Cannot import directory output 'smoke/images'" in whole.output
    assert "smoke/images/{filename}" in whole.output

    data_path = _helper_import(runner, source, "smoke/images/part/a.csv", "a.csv")
    assert data_path.read_bytes() == content


@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        pytest.param(None, "missing .pivot/config.yaml", id="no-config"),
        pytest.param("remotes: [unclosed\n", "Invalid YAML in remote config", id="bad-yaml"),
        pytest.param("- just\n- a list\n", "Remote config is not a mapping", id="not-mapping"),
        pytest.param("remotes: {}\n", "Remote config has no remotes", id="no-remotes"),
        pytest.param(
            "remotes:\n  a: s3://bucket-a/c\n  b: s3://bucket-b/c\n",
            "multiple remotes but no default_remote",
            id="ambiguous-remote",
        ),
        pytest.param(
            "remotes:\n  a: s3://bucket-a/c\ndefault_remote: b\n",
            "Default remote 'b' not found",
            id="unknown-default",
        ),
        pytest.param(
            "remotes:\n  a: https://example.com/c\n", "Invalid S3 URL", id="non-s3-remote"
        ),
    ],
)
def test_import_rejects_source_without_usable_remote(
    runner: click.testing.CliRunner,
    source: _Source,
    consumer: pathlib.Path,
    config_text: str | None,
    message: str,
) -> None:
    if config_text is not None:
        source.write(".pivot/config.yaml", config_text)
    source.write_lock("smoke/total", {None: types.FileHash(hash="0123456789abcdef")})
    source.commit("broken remote config")

    result = runner.invoke(cli.cli, ["import", str(source.root), "smoke/total"])

    assert result.exit_code != 0
    assert message in result.output
    assert not (consumer / "smoke").exists(), "a failed import writes nothing"


@pytest.mark.xfail(
    strict=True,
    reason="red test for Task 9: non-string default_remote must be refused",
)
def test_import_refuses_non_string_default_remote(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    """``default_remote: 7`` is a config mistake; it must not silently mean the sole remote."""
    source.write(
        ".pivot/config.yaml",
        yaml.safe_dump({"remotes": {"origin": bucket.url}, "default_remote": 7}),
    )
    source.write_lock("smoke/total", {None: types.FileHash(hash=bucket.put(b"only remote"))})
    source.commit("non-string default")

    result = runner.invoke(cli.cli, ["import", str(source.root), "smoke/total"])

    assert result.exit_code != 0
    assert "default_remote" in result.output, "the error must name the bad config key"
    assert not (consumer / "smoke").exists(), "a refused import writes nothing"


@pytest.mark.parametrize(
    ("stage_files", "message"),
    [
        pytest.param({}, "Remote repo has no .pivot/stages directory", id="no-stages-dir"),
        pytest.param(
            {".pivot/stages/README.txt": "notes\n"},
            "Remote repo has no lock files",
            id="no-lock-files",
        ),
        pytest.param(
            {
                ".pivot/stages/smoke/broken.lock": "outs: [unclosed\n",
                ".pivot/stages/smoke/partial.lock": "code_manifest: {}\n",
            },
            "Available outputs: (none)",
            id="unreadable-locks",
        ),
    ],
)
def test_import_reports_source_without_importable_outputs(
    runner: click.testing.CliRunner,
    source: _Source,
    bucket: _Bucket,
    consumer: pathlib.Path,
    stage_files: dict[str, str],
    message: str,
) -> None:
    source.write_config(bucket.url)
    for rel_path, content in stage_files.items():
        source.write(rel_path, content)
    source.commit("nothing importable")

    result = runner.invoke(cli.cli, ["import", str(source.root), "smoke/total"])

    assert result.exit_code != 0
    assert message in result.output


def test_import_unknown_artifact_lists_some_available_and_counts_the_rest(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    """A long list is cut short; the identities shown come first in order and the rest are
    counted, so shown + counted accounts for every available identity."""
    total = 30
    names = [f"smoke/s{index:02d}" for index in range(total)]
    source.write_config(bucket.url)
    for name in names:
        source.write_lock(name, {None: types.FileHash(hash="0123456789abcdef")})
    source.commit("many stages")

    result = runner.invoke(cli.cli, ["import", str(source.root), "smoke/missing"])

    assert result.exit_code != 0
    shown = [name for name in names if name in result.output]
    more = re.search(r"\(and (\d+) more\)", result.output)
    assert more is not None, "a cut-short list must say how many identities it left out"
    assert 0 < len(shown) < total
    assert shown == names[: len(shown)], "the identities shown must be the first in order"
    assert len(shown) + int(more.group(1)) == total


def test_import_refuses_path_produced_by_two_stages(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    """``smoke/images/a.csv`` names both a file in a directory output and a stage."""
    source.write_config(bucket.url)
    source.write_lock(
        "smoke/images",
        {
            None: types.DirHash(
                hash="0123456789abcdef",
                manifest=[
                    types.DirManifestEntry(
                        relpath="a.csv", hash="1111111111111111", size=1, isexec=False
                    )
                ],
            )
        },
    )
    source.write_lock("smoke/images/a.csv", {None: types.FileHash(hash="2222222222222222")})
    source.commit("colliding identities")

    result = runner.invoke(cli.cli, ["import", str(source.root), "smoke/images/a.csv"])

    assert result.exit_code != 0
    assert "produced by multiple stages: smoke/images, smoke/images/a.csv" in result.output


def test_import_never_writes_outside_project_or_over_existing_files(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    _helper_publish(source, bucket, "smoke/total", b"remote", "v1")
    repo = str(source.root)

    escape = runner.invoke(cli.cli, ["import", repo, "smoke/total", "--out", "../escape.txt"])
    assert escape.exit_code != 0
    assert "must be within project root" in escape.output
    assert not (consumer.parent / "escape.txt").exists()

    existing = consumer / "total.txt"
    existing.write_text("local edit")
    over_data = runner.invoke(cli.cli, ["import", repo, "smoke/total", "--out", "total.txt"])
    assert over_data.exit_code != 0
    assert "already exists. Use --force" in over_data.output
    assert existing.read_text() == "local edit"

    existing.unlink()
    track.get_pvt_path(existing).write_text("path: total.txt\n")
    over_pvt = runner.invoke(cli.cli, ["import", repo, "smoke/total", "--out", "total.txt"])
    assert over_pvt.exit_code != 0
    assert "total.txt.pvt' already exists" in over_pvt.output
    assert not existing.exists()

    directory = consumer / "folder"
    (directory / "keep.txt").parent.mkdir()
    (directory / "keep.txt").write_text("kept")
    over_dir = runner.invoke(cli.cli, ["import", repo, "smoke/total", "--out", "folder", "--force"])
    assert over_dir.exit_code != 0
    assert "Cannot overwrite directory with import" in over_dir.output
    assert (directory / "keep.txt").read_text() == "kept"

    forced = runner.invoke(
        cli.cli, ["import", repo, "smoke/total", "--out", "total.txt", "--force"]
    )
    assert forced.exit_code == 0, forced.output
    assert existing.read_bytes() == b"remote"


def test_import_rejects_remote_content_that_does_not_match_its_hash(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    claimed = xxhash.xxh64(b"what the lock promises").hexdigest()
    bucket.put(b"tampered bytes", as_hash=claimed)
    source.write_config(bucket.url)
    source.write_lock("smoke/total", {None: types.FileHash(hash=claimed)})
    source.commit("corrupt remote")

    result = runner.invoke(cli.cli, ["import", str(source.root), "smoke/total", "--out", "t.txt"])

    assert result.exit_code != 0
    assert "Downloaded file hash mismatch" in result.output
    assert not (consumer / "t.txt").exists(), "corrupt content must not stay in the workspace"
    assert not (consumer / "t.txt.pvt").exists()


# =============================================================================
# update
# =============================================================================


def test_update_leaves_current_import_untouched(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    head = _helper_publish(source, bucket, "smoke/total", b"v1", "v1")
    data_path = _helper_import(runner, source, "smoke/total", "data/total.txt")
    pvt_path = track.get_pvt_path(data_path)
    pvt_before = pvt_path.read_text()

    check = runner.invoke(cli.cli, ["update", "--dry-run"])
    assert check.exit_code == 0, check.output
    assert f"Up to date: {pvt_path}" in check.output

    by_pvt = runner.invoke(cli.cli, ["update", "data/total.txt.pvt"])
    assert by_pvt.exit_code == 0, by_pvt.output
    assert f"Up to date: {data_path}" in by_pvt.output
    assert pvt_path.read_text() == pvt_before
    assert _helper_rev_lock(data_path) == head


def test_update_records_new_revision_without_download_when_content_is_unchanged(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    old_head = _helper_publish(source, bucket, "smoke/total", b"v1", "v1")
    data_path = _helper_import(runner, source, "smoke/total", "total.txt")
    source.write("README.md", "unrelated change\n")
    new_head = source.commit("docs only")

    check = runner.invoke(cli.cli, ["update", "--dry-run"])
    assert f"({old_head[:8]} → {new_head[:8]})" in check.output
    result = runner.invoke(cli.cli, ["update", "total.txt"])

    assert result.exit_code == 0, result.output
    assert f"Metadata updated: {data_path} ({old_head[:8]} → {new_head[:8]})" in result.output
    assert data_path.read_bytes() == b"v1"
    assert _helper_rev_lock(data_path) == new_head


def test_update_downloads_changed_content_and_restores_a_deleted_file(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    _helper_publish(source, bucket, "smoke/total", b"v1", "v1")
    data_path = _helper_import(runner, source, "smoke/total", "total.txt")
    new_head = _helper_publish(source, bucket, "smoke/total", b"version two", "v2")

    changed = runner.invoke(cli.cli, ["update"])
    assert changed.exit_code == 0, changed.output
    assert f"Updated: {data_path}" in changed.output
    assert data_path.read_bytes() == b"version two"
    pvt_data = track.read_pvt_file(track.get_pvt_path(data_path))
    assert pvt_data is not None
    assert pvt_data["hash"] == xxhash.xxh64(b"version two").hexdigest()
    assert pvt_data["size"] == len(b"version two")

    data_path.unlink()
    restored = runner.invoke(cli.cli, ["update"])
    assert restored.exit_code == 0, restored.output
    assert f"Updated: {data_path} ({new_head[:8]} → {new_head[:8]})" in restored.output
    assert data_path.read_bytes() == b"version two"


@pytest.mark.parametrize(
    "dry_run", [pytest.param(True, id="dry-run"), pytest.param(False, id="update")]
)
def test_update_with_unreachable_source_fails_naming_the_import(
    runner: click.testing.CliRunner,
    source: _Source,
    bucket: _Bucket,
    consumer: pathlib.Path,
    dry_run: bool,
) -> None:
    _helper_publish(source, bucket, "smoke/total", b"v1", "v1")
    data_path = _helper_import(runner, source, "smoke/total", "total.txt")
    shutil.rmtree(source.root)

    result = runner.invoke(cli.cli, ["update", *(["--dry-run"] if dry_run else [])])

    assert result.exit_code == 1
    assert f"Error updating {track.get_pvt_path(data_path)}: " in result.output
    assert "resolve ref 'main'" in result.output
    assert data_path.read_bytes() == b"v1"


def test_update_failure_of_one_import_does_not_stop_the_others(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    source.write_config(bucket.url)
    source.write_lock("smoke/a", {None: types.FileHash(hash=bucket.put(b"a1"))})
    source.write_lock("smoke/b", {None: types.FileHash(hash=bucket.put(b"b1"))})
    source.commit("v1")
    a_path = _helper_import(runner, source, "smoke/a", "a.txt")
    b_path = _helper_import(runner, source, "smoke/b", "b.txt")
    source.write_lock("smoke/a", {None: types.FileHash(hash=bucket.put(b"a2"))})
    source.write_lock("smoke/b", {None: types.FileHash(hash=bucket.put(b"b2"))})
    source.commit("v2")
    a_path.unlink()
    a_path.mkdir()  # a directory now sits where a's file is restored

    result = runner.invoke(cli.cli, ["update"])

    assert result.exit_code == 1
    assert f"Unexpected error updating {track.get_pvt_path(a_path)}: IsADirectoryError" in (
        result.output
    )
    assert f"Updated: {b_path}" in result.output
    assert b_path.read_bytes() == b"b2"


def test_update_skips_unreadable_and_non_import_targets(
    runner: click.testing.CliRunner, consumer: pathlib.Path
) -> None:
    (consumer / "junk.csv.pvt").write_text("path: [unclosed\n")
    track.write_pvt_file(
        consumer / "local.csv.pvt", track.PvtData(path="local.csv", hash="cafebabe", size=1)
    )

    result = runner.invoke(cli.cli, ["update", "junk.csv", "local.csv.pvt"])

    assert result.exit_code == 0, result.output
    assert f"Skipping invalid .pvt: {consumer / 'junk.csv.pvt'}" in result.output
    assert f"Skipping non-import: {consumer / 'local.csv.pvt'}" in result.output
    assert "No imports found to update." in result.output


@pytest.mark.xfail(
    strict=True,
    reason="red test for Task 9 (cli-fixes Slice 2 PR): failed pivot update deletes the last "
    + "good import",
)
def test_update_with_corrupt_remote_keeps_last_good_file(
    runner: click.testing.CliRunner, source: _Source, bucket: _Bucket, consumer: pathlib.Path
) -> None:
    """A failed update returns to the last known good state (AGENTS.md, Error Handling)."""
    _helper_publish(source, bucket, "smoke/total", b"v1", "v1")
    data_path = _helper_import(runner, source, "smoke/total", "total.txt")
    claimed = xxhash.xxh64(b"v2").hexdigest()
    bucket.put(b"tampered", as_hash=claimed)
    source.write_lock("smoke/total", {None: types.FileHash(hash=claimed)})
    source.commit("v2 with corrupt remote copy")

    result = runner.invoke(cli.cli, ["update"])

    assert result.exit_code == 1
    assert "Hash mismatch" in result.output
    assert data_path.exists() and data_path.read_bytes() == b"v1", (
        "the previously imported file must survive a failed update"
    )
