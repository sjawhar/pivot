from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli, exceptions, types
from pivot.storage import cache, listing, track

if TYPE_CHECKING:
    import click.testing

    from pivot.registry import PipelineLike


# =============================================================================
# Path Traversal Security Tests
# =============================================================================


def test_track_null_byte_in_path_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Paths with null bytes rejected."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Create a file that we'll try to track with a null byte in path
        pathlib.Path("data.txt").write_text("content")

        # Try to track with null byte in path - should fail
        result = runner.invoke(cli.cli, ["track", "data\x00.txt"])

        # Must fail - null bytes are dangerous for C libraries and filesystem operations
        # The file "data\x00.txt" doesn't exist (we created "data.txt"), so this must error
        assert result.exit_code != 0, "Null byte path should be rejected"


def test_checkout_manifest_path_traversal_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A directory listing with ../ is rejected during checkout."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # A malicious sidecar has a valid record hash but an unsafe member path.
        malicious_listing = b"f" * 16 + b" 10 - ../outside.txt\n"
        pvt_path = pathlib.Path("data_dir.pvt")
        pvt_data = track.PvtData(
            path="data_dir",
            hash=listing.digest(malicious_listing),
            size=10,
            nfiles=1,
        )
        track.write_pvt_file(pvt_path, pvt_data)
        listing.tracked_listing_path(pvt_path).write_bytes(malicious_listing)

        result = runner.invoke(cli.cli, ["checkout", "data_dir"])

        # Must either fail OR not create files outside the project directory
        # The critical security property: no file created at ../outside.txt
        assert not (tmp_path.parent / "outside.txt").exists(), (
            "Path traversal should not create files outside project"
        )
        # Command should also indicate failure for a malicious directory listing.
        assert (
            result.exit_code != 0
            or "error" in result.output.lower()
            or "invalid" in result.output.lower()
            or "not found" in result.output.lower()
        )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("..%2foutside.txt", id="url-encoded-slash"),
        pytest.param("..%252foutside.txt", id="double-encoded-slash"),
        pytest.param("..\\outside.txt", id="backslash"),
    ],
)
def test_track_keeps_a_name_spelled_like_a_step_up_inside_the_project(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, name: str
) -> None:
    """A name that only spells ``../outside.txt`` is a file in the project.

    Pivot decodes no ``%`` escape, and a backslash is part of a name, so ``track`` takes the
    file in the project, writes its ``.pvt`` beside it, and writes nothing next to the project,
    where ``outside.txt`` waits for a reading that steps out.
    """
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / name).write_text("inside\n")
        (cwd.parent / "outside.txt").write_text("outside\n")
        beside_project = sorted(cwd.parent.iterdir())

        result = runner.invoke(cli.cli, ["track", name])
        sidecar = track.read_pvt_file(cwd / f"{name}.pvt")
        beside_project_after = sorted(cwd.parent.iterdir())

    assert result.exit_code == 0, result.output
    assert sidecar is not None
    assert sidecar["path"] == name
    assert beside_project_after == beside_project, "track wrote outside the project"


# =============================================================================
# Stage Name Validation Tests
# =============================================================================


def test_stage_name_path_injection_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Stage names with / or .. rejected."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Try to run with malicious stage name
        result = runner.invoke(cli.cli, ["run", "../../../etc/passwd"])

        # Should not create files outside project
        assert result.exit_code != 0
        # Should not have created the malicious path
        assert not pathlib.Path("../../../etc/passwd.lock").exists()


# =============================================================================
# Hash Validation Tests
# =============================================================================


def test_cache_path_rejects_invalid_characters(tmp_path: pathlib.Path) -> None:
    """get_cache_path rejects hashes with non-hex characters."""
    cache_dir = tmp_path / "cache" / "files"
    cache_dir.mkdir(parents=True)

    # All hashes must be exactly 16 chars to test character validation (not length)
    invalid_hashes = [
        "abcd/efg12345678",  # forward slash
        "abcdefgh12345GHI",  # uppercase letters
        "abcdefgh1234567!",  # special character
        "abcd efgh1234567",  # space character
        "..%2fabcdef12345",  # URL encoded traversal
    ]

    for bad_hash in invalid_hashes:
        with pytest.raises(exceptions.SecurityValidationError, match="invalid characters"):
            cache.get_cache_path(cache_dir, bad_hash)


def test_cache_path_rejects_invalid_length(tmp_path: pathlib.Path) -> None:
    """get_cache_path rejects hashes with wrong length."""
    cache_dir = tmp_path / "cache" / "files"
    cache_dir.mkdir(parents=True)

    invalid_lengths = [
        "abc",  # too short (3 chars)
        "abcdef12",  # too short (8 chars)
        "abcdef1234567890abcdef",  # too long (22 chars)
        "",  # empty string
    ]

    for bad_hash in invalid_lengths:
        with pytest.raises(exceptions.SecurityValidationError, match="exactly 16 characters"):
            cache.get_cache_path(cache_dir, bad_hash)


def test_cache_path_structure(tmp_path: pathlib.Path) -> None:
    """Cache paths use hash prefix directory structure."""
    cache_dir = tmp_path / "cache" / "files"
    cache_dir.mkdir(parents=True)

    # Valid hash should create subdirectory structure
    valid_hash = "abcd1234567890ef"
    cache_path = cache.get_cache_path(cache_dir, valid_hash)

    # Should be cache_dir/ab/cd1234567890ef (first 2 chars as subdir, rest as filename)
    assert cache_path.name == valid_hash[2:]  # Filename is rest of hash
    assert cache_path.parent.name == valid_hash[:2]  # Parent dir is first 2 chars
    assert cache_path.parent.parent == cache_dir  # Grandparent is cache_dir


# =============================================================================
# YAML Security Tests
# =============================================================================


def test_pvt_listing_large_file_handled(tmp_path: pathlib.Path) -> None:
    """A large tracked-directory listing parses without an oversized .pvt file."""
    pvt_path = tmp_path / "large.pvt"
    members = listing.DirListing(
        files=[
            types.DirManifestEntry(
                relpath=f"file_{index:04d}.txt",
                hash="a" * 16,
                size=100,
                isexec=False,
            )
            for index in range(100)
        ],
        empty_dirs=[],
    )
    members_data = listing.encode(members)
    record = listing.record_of(members, members_data)
    pvt_data = track.PvtData(
        path="large",
        hash=record["hash"],
        size=record["size"],
        nfiles=record["nfiles"],
    )
    track.write_pvt_file(pvt_path, pvt_data)
    listing.write(listing.tracked_listing_path(pvt_path), members_data)

    result = track.read_pvt_file(pvt_path)

    assert result is not None and result == pvt_data
    restored_members = track.tracked_members(tmp_path / "large", result, None)
    assert len(restored_members["files"]) == 100


def test_pvt_file_yaml_invalid_type_rejected(tmp_path: pathlib.Path) -> None:
    """Invalid YAML types in .pvt file rejected gracefully."""
    pvt_path = tmp_path / "invalid.pvt"

    # Create invalid pvt content (wrong types)
    invalid_contents = [
        # List instead of dict at root
        "- item1\n- item2",
        # String instead of dict
        "just a string",
        # Missing required fields
        "path: test.txt\n",  # Missing hash
    ]

    for content in invalid_contents:
        pvt_path.write_text(content)
        result = track.read_pvt_file(pvt_path)
        # Should return None for invalid content, not raise
        assert result is None


# =============================================================================
# Symlink Security Tests
# =============================================================================


def test_directory_track_processes_valid_files(
    runner: click.testing.CliRunner,
    mock_discovery: PipelineLike,
) -> None:
    """Track on directory processes valid regular files."""
    # Create directory with valid file
    data_dir = pathlib.Path("data_dir")
    data_dir.mkdir()
    (data_dir / "valid.txt").write_text("content")

    result = runner.invoke(cli.cli, ["track", "data_dir"])

    # Should succeed and track the directory
    assert result.exit_code == 0
    assert "Tracked: data_dir" in result.output


def test_directory_track_creates_pvt_file(
    runner: click.testing.CliRunner,
    mock_discovery: PipelineLike,
) -> None:
    """Track on a directory writes its compact record and sidecar listing."""
    # Create directory with files
    data_dir = pathlib.Path("data_dir")
    data_dir.mkdir()
    (data_dir / "file1.txt").write_text("content1")
    (data_dir / "file2.txt").write_text("content2")

    result = runner.invoke(cli.cli, ["track", "data_dir"])

    assert result.exit_code == 0

    # Check pvt file and adjacent listing were created.
    pvt_path = pathlib.Path("data_dir.pvt")
    assert pvt_path.exists()
    listing_path = listing.tracked_listing_path(pvt_path)
    assert listing_path.exists()

    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    members_data = listing_path.read_bytes()
    members = listing.decode(members_data)
    record = listing.record_of(members, members_data)
    assert set(pvt_data) == {"path", "hash", "size", "nfiles"}
    assert pvt_data["path"] == "data_dir"
    assert pvt_data["hash"] == record["hash"]
    assert pvt_data["size"] == record["size"]
    assert "nfiles" in pvt_data and pvt_data["nfiles"] == record["nfiles"] == 2
    assert [entry["relpath"] for entry in members["files"]] == ["file1.txt", "file2.txt"]
