from __future__ import annotations

import asyncio
import json
import pathlib
from typing import TYPE_CHECKING, TypedDict

import click

from pivot import config, exceptions
from pivot import status as status_mod
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.remote import config as remote_config
from pivot.remote import storage as remote_mod
from pivot.remote import sync as transfer
from pivot.storage import lock
from pivot.types import OutputHash, PipelineStatus, PipelineStatusInfo, is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class StageVerifyInfo(TypedDict):
    """Verification info for a single stage."""

    name: str
    status: str
    reason: str
    missing_files: list[str]


class VerifyOutput(TypedDict):
    """JSON output structure for verify command."""

    passed: bool
    stages: list[StageVerifyInfo]


def _extract_file_hashes(hash_infos: Mapping[str, OutputHash]) -> dict[str, str]:
    """Extract individual file hashes from a hash_info dict.

    Tree hashes (directory hashes) are computed, not cached - only individual
    file hashes are stored in the cache. For directories with manifests,
    extracts each manifest entry's hash.
    """
    result = dict[str, str]()
    for path, hash_info in hash_infos.items():
        if hash_info is None:
            continue
        if is_dir_hash(hash_info):
            for entry in hash_info["manifest"]:
                entry_path = str(pathlib.Path(path) / entry["relpath"])
                result[entry_path] = entry["hash"]
        else:
            result[path] = hash_info["hash"]
    return result


def _get_stage_lock_hashes(
    stage_name: str, state_dir: Path
) -> tuple[dict[str, str], dict[str, str]]:
    """Get output and dep file hashes from a stage's lock file.

    Returns (output_hashes, dep_hashes) where each is {path: hash}.

    For both outputs and deps, includes manifest entry hashes for directories.
    """
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
    lock_data = stage_lock.read()
    if lock_data is None:
        return {}, {}

    return (
        _extract_file_hashes(lock_data["output_hashes"]),
        _extract_file_hashes(lock_data["dep_hashes"]),
    )


def _verify_stage_files(
    stage_name: str,
    state_dir: Path,
    local_hashes: set[str],
    allow_missing: bool,
    remote: remote_mod.S3Remote | None,
) -> list[str]:
    """Verify all files for a stage exist locally or on remote.

    When allow_missing=True, also verifies deps on remote (if missing locally).
    Returns list of missing file paths.
    """
    output_hashes, dep_hashes = _get_stage_lock_hashes(stage_name, state_dir)

    # Track unique missing hashes and all paths that map to them
    missing_hashes = set[str]()
    hash_to_paths = dict[str, list[str]]()

    # Check outputs: must be in local cache or (with allow_missing) on remote
    for path, hash_val in output_hashes.items():
        if hash_val not in local_hashes:
            missing_hashes.add(hash_val)
            hash_to_paths.setdefault(hash_val, []).append(path)

    # Check deps only when allow_missing: must exist locally OR on remote
    # (deps are in working dir, not cache, so we check file existence)
    if allow_missing:
        for path, hash_val in dep_hashes.items():
            # Skip deps that exist locally (in working directory)
            if pathlib.Path(path).exists():
                continue
            # Dep missing locally - must be on remote
            missing_hashes.add(hash_val)
            hash_to_paths.setdefault(hash_val, []).append(path)

    if not missing_hashes:
        return []

    if not allow_missing or remote is None:
        # Return all paths for all missing hashes
        return [p for paths in hash_to_paths.values() for p in paths]

    # Check remote for missing hashes (deduplicated to avoid redundant HEAD requests)
    try:
        remote_exists = asyncio.run(remote.bulk_exists(list(missing_hashes)))
    except exceptions.RemoteError as e:
        raise exceptions.RemoteError(f"Failed to check remote for missing files: {e}") from e

    missing_files = list[str]()
    for hash_val, paths in hash_to_paths.items():
        if not remote_exists.get(hash_val, False):
            missing_files.extend(paths)

    return missing_files


def _create_remote_if_needed(allow_missing: bool) -> remote_mod.S3Remote | None:
    """Create remote connection if allow_missing mode requires it."""
    if not allow_missing:
        return None

    remotes = remote_config.list_remotes()
    if not remotes:
        raise exceptions.RemoteNotConfiguredError(
            "No remotes configured. --allow-missing requires a remote to check for files."
        )

    try:
        remote, _ = transfer.create_remote_from_name(None)
    except exceptions.RemoteError as e:
        raise exceptions.RemoteError(f"Failed to connect to remote: {e}") from e
    return remote


def _verify_stages(
    pipeline_status: list[PipelineStatusInfo],
    state_dir: Path,
    cache_dir: Path,
    allow_missing: bool,
) -> tuple[bool, list[StageVerifyInfo]]:
    """Verify all stages and return pass/fail status with details."""
    remote = _create_remote_if_needed(allow_missing)
    local_hashes = transfer.get_local_cache_hashes(cache_dir)

    results = list[StageVerifyInfo]()
    all_passed = True

    for stage_info in pipeline_status:
        stage_name = stage_info["name"]
        status = stage_info["status"]
        reason = stage_info["reason"]

        # Check if stage is stale (code/params/deps changed)
        if status == PipelineStatus.STALE:
            results.append(
                StageVerifyInfo(
                    name=stage_name,
                    status="failed",
                    reason=reason or "Stage is stale",
                    missing_files=[],
                )
            )
            all_passed = False
            continue

        # Check if files exist locally or on remote
        missing_files = _verify_stage_files(
            stage_name, state_dir, local_hashes, allow_missing, remote
        )

        if missing_files:
            results.append(
                StageVerifyInfo(
                    name=stage_name,
                    status="failed",
                    reason=f"Missing files: {', '.join(missing_files)}",
                    missing_files=missing_files,
                )
            )
            all_passed = False
        else:
            results.append(
                StageVerifyInfo(
                    name=stage_name,
                    status="passed",
                    reason="",
                    missing_files=[],
                )
            )

    return all_passed, results


def _output_text(passed: bool, results: list[StageVerifyInfo], quiet: bool) -> None:
    """Output verification results as formatted text."""
    if quiet:
        return

    if passed:
        click.echo("Verification passed")
    else:
        click.echo("Verification failed")

    click.echo()
    for stage in results:
        status_icon = "✓" if stage["status"] == "passed" else "✗"
        click.echo(f"  {status_icon} {stage['name']}: {stage['status']}")
        if stage["reason"]:
            click.echo(f"      {stage['reason']}")


def _output_json(passed: bool, results: list[StageVerifyInfo]) -> None:
    """Output verification results as JSON."""
    output = VerifyOutput(passed=passed, stages=results)
    click.echo(json.dumps(output, indent=2))


@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option("--allow-missing", is_flag=True, help="Allow missing local files if on remote")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def verify(
    ctx: click.Context,
    stages: tuple[str, ...],
    allow_missing: bool,
    output_json: bool,
) -> None:
    """Verify pipeline was reproduced and outputs are available.

    Checks that all stages are cached (code, params, deps match lock files)
    and output files exist locally or on remote.

    With --allow-missing, both stage dependencies and outputs are verified
    to exist on the remote cache, enabling CI verification without local data.

    Use in CI pre-merge gates to ensure pipeline is reproducible.

    Exit codes:
      0 - Verification passed
      1 - Verification failed (stale stages or missing files)
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    # Validate stages exist
    stages_list = cli_helpers.stages_to_list(stages)
    cli_helpers.validate_stages_exist(stages_list)

    # Check if any stages are registered
    all_stages = cli_helpers.get_all_stages()
    if not all_stages:
        raise click.ClickException("No stages registered. Nothing to verify.")

    # Resolve directories - use same state_dir as get_pipeline_status for consistency
    state_dir = config.get_state_dir()
    cache_dir = config.get_cache_dir()

    # Get pipeline status (uses default state directory internally)
    pipeline_status, _ = status_mod.get_pipeline_status(
        stages_list,
        single_stage=False,
        all_stages=all_stages,
        stage_registry=cli_helpers.get_registry(),
        allow_missing=allow_missing,
    )

    if not pipeline_status:
        raise click.ClickException("No stages to verify.")

    # Verify stages
    passed, results = _verify_stages(pipeline_status, state_dir, cache_dir, allow_missing)

    # Output results
    if output_json:
        _output_json(passed, results)
    else:
        _output_text(passed, results, quiet)

    # Set exit code
    if not passed:
        raise SystemExit(1)
