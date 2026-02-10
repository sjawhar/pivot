from __future__ import annotations

import asyncio
import functools
import logging
import pathlib
from typing import TYPE_CHECKING, TypedDict, cast

import aiohttp
import yaml

from pivot import exceptions, project
from pivot.remote import config as remote_config
from pivot.remote import git_archive, github
from pivot.remote import storage as remote_storage
from pivot.storage import cache, track
from pivot.storage import lock as storage_lock

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.types import DirManifestEntry, OutEntry, StorageLockData

logger = logging.getLogger(__name__)


class ResolvedImport(TypedDict):
    stage: str
    path: str
    hash: str
    size: int
    remote_url: str
    rev_lock: str


class ImportResult(TypedDict):
    pvt_path: str
    data_path: str
    downloaded: bool


def _maybe_token(repo_url: str, token: str | None) -> str | None:
    if token is not None:
        return token
    if github.is_github_url(repo_url):
        return github.get_token()
    return None


async def _run_blocking[T](func: Callable[..., T], *args: object) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args))


async def _read_remote_file(
    repo_url: str,
    path: str,
    rev: str,
    token: str | None,
    *,
    session: aiohttp.ClientSession | None = None,
) -> bytes | None:
    if github.is_github_url(repo_url):
        owner, repo = github.parse_github_url(repo_url)
        return await github.read_file(
            owner, repo, path, rev, _maybe_token(repo_url, token), session=session
        )
    return await _run_blocking(git_archive.read_file_from_remote_repo, repo_url, path, rev)


async def _resolve_ref(
    repo_url: str,
    ref: str,
    token: str | None,
    *,
    session: aiohttp.ClientSession | None = None,
) -> str | None:
    if github.is_github_url(repo_url):
        owner, repo = github.parse_github_url(repo_url)
        return await github.resolve_ref(
            owner, repo, ref, _maybe_token(repo_url, token), session=session
        )
    return await _run_blocking(git_archive.resolve_ref_from_remote_repo, repo_url, ref)


async def read_remote_config(
    repo_url: str,
    rev: str,
    token: str | None,
    *,
    session: aiohttp.ClientSession | None = None,
) -> str:
    raw = await _read_remote_file(repo_url, ".pivot/config.yaml", rev, token, session=session)
    if raw is None:
        raise exceptions.RemoteError("Remote repo is missing .pivot/config.yaml")

    try:
        data = cast("object", yaml.safe_load(raw))
    except yaml.YAMLError as e:
        raise exceptions.RemoteError(f"Invalid YAML in remote config: {e}") from e

    if not isinstance(data, dict):
        raise exceptions.RemoteError("Remote config is not a mapping")

    config_data = cast("dict[str, object]", data)
    remotes_raw = config_data.get("remotes")
    if not isinstance(remotes_raw, dict) or not remotes_raw:
        raise exceptions.RemoteError("Remote config has no remotes")

    remotes_data = cast("dict[object, object]", remotes_raw)
    remotes = {str(k): str(v) for k, v in remotes_data.items()}

    default_remote = config_data.get("default_remote")
    if not isinstance(default_remote, str):
        default_remote = None
    if default_remote is None:
        if len(remotes) == 1:
            default_remote = next(iter(remotes.keys()))
        else:
            raise exceptions.RemoteError("Remote config has multiple remotes but no default_remote")

    if default_remote not in remotes:
        raise exceptions.RemoteError(f"Default remote '{default_remote}' not found in config")

    url = str(remotes[default_remote])
    _ = remote_config.validate_s3_url(url)
    logger.info("Using source remote: %s", url)
    return url


async def list_remote_lock_files(
    repo_url: str,
    rev: str,
    token: str | None,
    *,
    session: aiohttp.ClientSession | None = None,
) -> list[str]:
    if github.is_github_url(repo_url):
        owner, repo = github.parse_github_url(repo_url)
        entries = await github.list_directory(
            owner,
            repo,
            ".pivot/stages",
            rev,
            _maybe_token(repo_url, token),
            session=session,
        )
    else:
        entries = await _run_blocking(
            git_archive.list_directory_from_remote_repo,
            repo_url,
            ".pivot/stages",
            rev,
        )

    if entries is None:
        raise exceptions.RemoteError("Remote repo has no .pivot/stages directory")

    stage_names = list[str]()
    for entry in [str(e) for e in entries]:
        name = pathlib.Path(entry).name
        if not name.endswith(".lock"):
            continue
        stage_names.append(name[: -len(".lock")])

    if not stage_names:
        raise exceptions.RemoteError("Remote repo has no lock files")

    return stage_names


async def read_remote_lock_file(
    repo_url: str,
    stage_name: str,
    rev: str,
    token: str | None,
    *,
    session: aiohttp.ClientSession | None = None,
) -> StorageLockData | None:
    path = f".pivot/stages/{stage_name}.lock"
    raw = await _read_remote_file(repo_url, path, rev, token, session=session)
    if raw is None:
        return None
    try:
        data = cast("object", yaml.safe_load(raw))
    except yaml.YAMLError:
        logger.warning("Failed to parse lock file for stage '%s'", stage_name)
        return None
    if not storage_lock.is_lock_data(data):
        logger.warning("Invalid lock file structure for stage '%s'", stage_name)
        return None
    return data


def _entry_size(entry: OutEntry) -> int:
    if "size" in entry:
        return int(entry["size"])
    if "manifest" in entry:
        return sum(int(m["size"]) for m in entry["manifest"])
    entry_path = entry["path"] if "path" in entry else "unknown"
    logger.warning("Lock entry '%s' has no size information", entry_path)
    return 0


def _iter_manifest_paths(entry: OutEntry) -> list[tuple[str, DirManifestEntry]]:
    if "manifest" not in entry:
        return []
    base = entry["path"].rstrip("/")
    return [(f"{base}/{m['relpath']}", m) for m in entry["manifest"]]


async def resolve_remote_path(
    repo_url: str,
    path: str,
    rev: str,
    token: str | None,
) -> ResolvedImport:
    async with aiohttp.ClientSession() as session:
        return await _resolve_remote_path_with_session(repo_url, path, rev, token, session)


async def _resolve_remote_path_with_session(
    repo_url: str,
    path: str,
    rev: str,
    token: str | None,
    session: aiohttp.ClientSession,
) -> ResolvedImport:
    gh_session = session if github.is_github_url(repo_url) else None
    rev_lock = await _resolve_ref(repo_url, rev, token, session=gh_session)
    if rev_lock is None:
        raise exceptions.RemoteError(f"Unable to resolve ref '{rev}'")

    remote_url = await read_remote_config(repo_url, rev_lock, token, session=gh_session)

    # Use resolved SHA for all subsequent reads to avoid TOCTOU race
    stage_names = await list_remote_lock_files(repo_url, rev_lock, token, session=gh_session)
    lock_datas = await asyncio.gather(
        *[
            read_remote_lock_file(repo_url, stage, rev_lock, token, session=gh_session)
            for stage in stage_names
        ]
    )

    matches = list[tuple[str, str, str, int]]()
    available = set[str]()

    for stage, lock_data in zip(stage_names, lock_datas, strict=True):
        if lock_data is None:
            continue
        for entry in lock_data["outs"]:
            available.add(entry["path"])
            for full_path, _ in _iter_manifest_paths(entry):
                available.add(full_path)

            if entry["path"] == path:
                if "manifest" in entry:
                    message = (
                        "Cannot import directory output "
                        + f"'{path}'. Import individual files instead, "
                        + f"e.g., '{path.rstrip('/')}/{{filename}}'"
                    )
                    raise exceptions.PivotError(message)
                matches.append((stage, path, entry["hash"], _entry_size(entry)))
                continue

            for full_path, manifest_entry in _iter_manifest_paths(entry):
                if full_path == path:
                    matches.append(
                        (
                            stage,
                            path,
                            manifest_entry["hash"],
                            int(manifest_entry["size"]),
                        )
                    )

    if not matches:
        max_shown = 10
        sorted_available = sorted(available)
        if len(sorted_available) > max_shown:
            shown = ", ".join(sorted_available[:max_shown])
            available_list = f"{shown} (and {len(sorted_available) - max_shown} more)"
        elif sorted_available:
            available_list = ", ".join(sorted_available)
        else:
            available_list = "(none)"
        raise exceptions.PivotError(
            f"Path '{path}' not found in remote outputs. Available outputs: {available_list}"
        )

    if len(matches) > 1:
        stages = ", ".join(sorted({stage for stage, _, _, _ in matches}))
        raise exceptions.PivotError(f"Path '{path}' is produced by multiple stages: {stages}")

    stage, matched_path, hash_, size = matches[0]
    return ResolvedImport(
        stage=stage,
        path=matched_path,
        hash=hash_,
        size=size,
        remote_url=remote_url,
        rev_lock=rev_lock,
    )


async def import_artifact(
    repo_url: str,
    path: str,
    *,
    rev: str = "main",
    out: str | None = None,
    force: bool = False,
    no_download: bool = False,
    project_root: pathlib.Path | None = None,
) -> ImportResult:
    resolved = await resolve_remote_path(repo_url, path, rev, None)

    if project_root is None:
        project_root = project.get_project_root()

    dest = out if out is not None else path
    data_path = project.normalize_path(dest, base=project_root)
    if not data_path.is_relative_to(project_root.absolute()):
        raise exceptions.PivotError(
            f"Import destination must be within project root. Got: {data_path}"
        )
    pvt_path = track.get_pvt_path(data_path)

    if data_path.exists() and not force:
        raise exceptions.PivotError(f"'{data_path}' already exists. Use --force to overwrite.")
    if pvt_path.exists() and not force:
        raise exceptions.PivotError(f"'{pvt_path}' already exists. Use --force to overwrite.")

    if force and data_path.exists() and data_path.is_dir():
        raise exceptions.PivotError("Cannot overwrite directory with import")

    pvt_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = False
    file_size = resolved["size"]
    if not no_download:
        remote = remote_storage.S3Remote(resolved["remote_url"])
        await remote.download_file(resolved["hash"], data_path)
        downloaded = True
        actual_hash, _ = cache.hash_file(data_path)
        if actual_hash != resolved["hash"]:
            data_path.unlink(missing_ok=True)
            raise exceptions.RemoteError(
                f"Downloaded file hash mismatch: expected {resolved['hash']}, got {actual_hash}"
            )
        file_size = data_path.stat().st_size

    source = track.ImportSource(
        repo=repo_url,
        rev=rev,
        rev_lock=resolved["rev_lock"],
        stage=resolved["stage"],
        path=resolved["path"],
        remote=resolved["remote_url"],
    )
    pvt_data = track.PvtData(
        path=data_path.name,
        hash=resolved["hash"],
        size=file_size,
        source=source,
    )

    track.write_pvt_file(pvt_path, pvt_data)

    return ImportResult(
        pvt_path=str(pvt_path),
        data_path=str(data_path),
        downloaded=downloaded,
    )


class UpdateCheck(TypedDict):
    available: bool
    current_rev: str
    latest_rev: str


class UpdateResult(TypedDict):
    downloaded: bool
    metadata_updated: bool
    updated: bool
    old_rev: str
    new_rev: str
    path: str


async def check_for_update(pvt_data: track.PvtData) -> UpdateCheck:
    """Check if import has updates. Resolves current ref, compares to rev_lock."""
    if "source" not in pvt_data:
        raise exceptions.PivotError("Not an import")
    source = pvt_data["source"]
    rev_lock = await _resolve_ref(source["repo"], source["rev"], None)
    if rev_lock is None:
        raise exceptions.RemoteError(f"Cannot resolve ref '{source['rev']}' from {source['repo']}")
    return UpdateCheck(
        available=rev_lock != source["rev_lock"],
        current_rev=source["rev_lock"],
        latest_rev=rev_lock,
    )


async def update_import(pvt_path: pathlib.Path, *, new_rev: str | None = None) -> UpdateResult:
    """Update an imported artifact. Re-resolves ref, re-downloads if hash changed."""
    pvt_data = track.read_pvt_file(pvt_path)
    if pvt_data is None:
        raise exceptions.PivotError(f"Invalid .pvt file: {pvt_path}")
    if "source" not in pvt_data:
        raise exceptions.PivotError(f"Not an import: {pvt_path}")

    source = pvt_data["source"]
    rev = new_rev if new_rev is not None else source["rev"]
    old_rev = source["rev_lock"]

    # Re-resolve at new/current rev
    resolved = await resolve_remote_path(source["repo"], source["path"], rev, None)

    downloaded = False
    data_path = track.get_data_path(pvt_path)
    need_download = resolved["hash"] != pvt_data["hash"] or not data_path.exists()

    file_size = resolved["size"]
    if need_download:
        remote = remote_storage.S3Remote(resolved["remote_url"])
        await remote.download_file(resolved["hash"], data_path)
        actual_hash, _ = cache.hash_file(data_path)
        if actual_hash != resolved["hash"]:
            data_path.unlink(missing_ok=True)
            raise exceptions.RemoteError(
                f"Hash mismatch: expected {resolved['hash']}, got {actual_hash}"
            )
        downloaded = True
        file_size = data_path.stat().st_size
    elif data_path.exists():
        file_size = data_path.stat().st_size

    metadata_changed = (
        resolved["rev_lock"] != old_rev
        or resolved["hash"] != pvt_data["hash"]
        or rev != source["rev"]
    )
    if metadata_changed:
        new_source = track.ImportSource(
            repo=source["repo"],
            rev=rev,
            rev_lock=resolved["rev_lock"],
            stage=resolved["stage"],
            path=resolved["path"],
            remote=resolved["remote_url"],
        )
        new_pvt = track.PvtData(
            path=pvt_data["path"],
            hash=resolved["hash"],
            size=file_size,
            source=new_source,
        )
        track.write_pvt_file(pvt_path, new_pvt)

    return UpdateResult(
        downloaded=downloaded,
        metadata_updated=metadata_changed,
        updated=downloaded,
        old_rev=old_rev,
        new_rev=resolved["rev_lock"],
        path=str(data_path),
    )
