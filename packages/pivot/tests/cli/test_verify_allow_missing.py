"""`pivot verify --allow-missing` on compositional-API pipelines."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot import cli, compose, config, types
from pivot.storage import cache, lock, track

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture


@compose.stage
def process(input_file: str) -> str:
    return input_file.upper()


@compose.stage
def produce(input_file: str) -> str:
    return input_file.upper()


@compose.stage
def consume(produced: str) -> str:
    return f"consumed: {produced}"


@compose.stage
def walk_a(source: pathlib.Path) -> str:
    return str(len(list(source.iterdir())))


@compose.stage
def walk_b(source: pathlib.Path) -> str:
    return str(len(list(source.iterdir())))


@compose.stage
def walk_c(source: pathlib.Path) -> str:
    return str(len(list(source.iterdir())))


def _helper_mock_remote(mocker: MockerFixture, *, exists: bool) -> None:
    """Mock the remote boundary: configured remote plus S3 existence checks."""
    mocker.patch("pivot.remote.config.list_remotes", return_value={"origin": "s3://bucket/cache"})
    mocker.patch("pivot.remote.config.get_default_remote", return_value="origin")
    mocker.patch("pivot.remote.config.get_remote_url", return_value="s3://bucket/cache")
    remote = mocker.patch("pivot.remote.storage.S3Remote").return_value

    async def _bulk_exists(hashes: list[str], concurrency: int = 20) -> dict[str, bool]:
        return dict.fromkeys(hashes, exists)

    remote.bulk_exists = _bulk_exists


def test_verify_allow_missing_finds_present_raw_input_at_its_workspace_path(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """A raw input on disk at data/raw/<name> is not missing, even if it is on no remote."""
    root = mock_discovery.root
    raw = root / "data" / "raw" / "input.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("hello")
    with mock_discovery:
        process(mock_discovery.input("input.txt", t=str))
    stage_name = mock_discovery.list_stages()[0]

    output = root / "data" / "test" / "process.txt"
    output.parent.mkdir(parents=True)
    output.write_text("HELLO")
    out_hash = cache.save_to_cache(
        output, config.get_cache_dir() / "files", checkout_mode=config.CheckoutMode.COPY
    )
    raw_hash, _ = cache.hash_file(raw)
    lock.StageLock(stage_name, lock.get_stages_dir(config.get_state_dir())).write(
        types.LockData(
            code_manifest=mock_discovery.ensure_fingerprint(stage_name),
            params={},
            dep_hashes={types.ArtifactIdentity("input.txt", None): types.FileHash(hash=raw_hash)},
            output_hashes={types.ArtifactIdentity(stage_name, None): out_hash},
        )
    )
    _helper_mock_remote(mocker, exists=False)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert "Missing files" not in result.output, result.output
    assert result.exit_code == 0, result.output


def test_verify_allow_missing_names_stale_producer_dependency(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """A lock-only consumer must fail against the producer lock and name that dependency."""
    root = mock_discovery.root
    raw = root / "data" / "raw" / "input.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("v2")
    with mock_discovery:
        intermediate = produce(mock_discovery.input("input.txt", t=str))
        consume(intermediate)
    producer_stage, consumer_stage = mock_discovery.list_stages()
    producer_hash = types.FileHash(hash="aaaaaaaaaaaaaaaa")
    consumer_hash = types.FileHash(hash="bbbbbbbbbbbbbbbb")
    state_dir = config.get_state_dir()
    raw_hash, _ = cache.hash_file(raw)
    lock.StageLock(producer_stage, lock.get_stages_dir(state_dir)).write(
        types.LockData(
            code_manifest=mock_discovery.ensure_fingerprint(producer_stage),
            params={},
            dep_hashes={types.ArtifactIdentity("input.txt", None): types.FileHash(hash=raw_hash)},
            output_hashes={types.ArtifactIdentity(producer_stage, None): producer_hash},
        )
    )
    lock.StageLock(consumer_stage, lock.get_stages_dir(state_dir)).write(
        types.LockData(
            code_manifest=mock_discovery.ensure_fingerprint(consumer_stage),
            params={},
            dep_hashes={types.ArtifactIdentity(producer_stage, None): consumer_hash},
            output_hashes={
                types.ArtifactIdentity(consumer_stage, None): types.FileHash(
                    hash="cccccccccccccccc"
                )
            },
        )
    )
    _helper_mock_remote(mocker, exists=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code == 1, result.output
    assert f"✗ {consumer_stage}: failed" in result.output
    assert f"Input dependencies changed: {producer_stage}" in result.output


def test_verify_allow_missing_scans_the_project_for_pvt_files_once(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """Every stage with a directory input asks which tracked data holds it, from one scan of the
    project for .pvt files per command, not one per stage: status's explanations scan once, and
    verify once."""
    root = mock_discovery.root
    state_dir = config.get_state_dir()
    with mock_discovery:
        for name, walk in (("a", walk_a), ("b", walk_b), ("c", walk_c)):
            source = root / "data" / "raw" / name
            source.mkdir(parents=True)
            (source / "f.txt").write_text(name)
            walk(mock_discovery.input(name, t=pathlib.Path))
    for stage_name, name in zip(mock_discovery.list_stages(), ("a", "b", "c"), strict=True):
        output = root / "data" / "test" / f"{stage_name.rpartition('/')[2]}.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("1")
        out_hash = cache.save_to_cache(
            output, config.get_cache_dir() / "files", checkout_mode=config.CheckoutMode.COPY
        )
        lock.StageLock(stage_name, lock.get_stages_dir(state_dir)).write(
            types.LockData(
                code_manifest=mock_discovery.ensure_fingerprint(stage_name),
                params={},
                dep_hashes={
                    types.ArtifactIdentity(name, None): cache.hash_directory(
                        root / "data" / "raw" / name
                    )["record"]
                },
                output_hashes={types.ArtifactIdentity(stage_name, None): out_hash},
            )
        )
    _helper_mock_remote(mocker, exists=True)
    scans = mocker.spy(track, "discover_pvt_files")

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code == 0, result.output
    assert scans.call_count == 2, "one scan for status's explanations and one for verify"
