from __future__ import annotations

import base64
import logging
import os
import re
import typing

import aiohttp

from pivot import exceptions

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=30)
_GITHUB_HTTPS_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/.]+?)(?:\.git)?/?$")
_GITHUB_SSH_RE = re.compile(r"^git@github\.com:([^/]+)/([^/.]+?)(?:\.git)?$")
_API_BASE = "https://api.github.com"


def get_token() -> str | None:
    """Get GitHub auth token from environment.

    Checks ``GH_TOKEN`` first, then falls back to ``GITHUB_TOKEN``.
    Returns ``None`` when neither is set.
    """
    token = (os.environ.get("GH_TOKEN") or "").strip() or (
        os.environ.get("GITHUB_TOKEN") or ""
    ).strip()
    return token or None


def is_github_url(url: str) -> bool:
    """Check if *url* points to a GitHub repository."""
    return bool(_GITHUB_HTTPS_RE.match(url) or _GITHUB_SSH_RE.match(url))


def parse_github_url(url: str) -> tuple[str, str]:
    """Extract ``(owner, repo)`` from a GitHub URL.

    Raises :class:`ValueError` if *url* is not a recognised GitHub URL.
    """
    m = _GITHUB_HTTPS_RE.match(url) or _GITHUB_SSH_RE.match(url)
    if not m:
        raise ValueError(f"Not a GitHub URL: {url}")
    return m.group(1), m.group(2)


def _auth_headers(token: str | None) -> dict[str, str]:
    """Build request headers for the GitHub REST API."""
    headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _raise_for_error_status(status: int, headers: dict[str, str], body: str) -> None:
    if status == 429:
        raise exceptions.RemoteError(
            "GitHub API rate limited. Set GH_TOKEN environment variable for higher limits."
        )
    if status == 403:
        remaining = headers.get("X-RateLimit-Remaining", "")
        if remaining == "0":
            raise exceptions.RemoteError(
                "GitHub API rate limited. Set GH_TOKEN environment variable for higher limits."
            )
        if "rate limit" in body.lower() or "abuse" in body.lower():
            raise exceptions.RemoteError(
                "GitHub API secondary rate limit. Retry after a short wait."
            )
        raise exceptions.RemoteError(
            "GitHub API access denied (403). Set GH_TOKEN environment variable if the repo is private."
        )


async def _get(
    session: aiohttp.ClientSession,
    url: str,
    token: str | None,
) -> tuple[int, object]:
    async with session.get(url, headers=_auth_headers(token), timeout=_DEFAULT_TIMEOUT) as resp:
        if resp.status == 404:
            if token is None:
                logger.debug("GitHub 404 (no token set — repo may be private)")
            return 404, None
        if resp.status in (403, 429):
            body = await resp.text()
            resp_headers = dict(resp.headers.items())
            _raise_for_error_status(resp.status, resp_headers, body)
        resp.raise_for_status()
        return resp.status, await resp.json()


async def read_file(
    owner: str,
    repo: str,
    path: str,
    ref: str,
    token: str | None = None,
    *,
    session: aiohttp.ClientSession | None = None,
) -> bytes | None:
    url = f"{_API_BASE}/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    if session is not None:
        status, data = await _get(session, url, token)
    else:
        async with aiohttp.ClientSession() as s:
            status, data = await _get(s, url, token)
    if status == 404:
        return None
    if isinstance(data, dict) and "content" in data:
        data_dict = typing.cast("dict[str, object]", data)
        content = data_dict["content"]
        if not isinstance(content, str):
            return None
        return base64.b64decode(content)
    return None


async def list_directory(
    owner: str,
    repo: str,
    path: str,
    ref: str,
    token: str | None = None,
    *,
    session: aiohttp.ClientSession | None = None,
) -> list[str] | None:
    url = f"{_API_BASE}/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    if session is not None:
        status, data = await _get(session, url, token)
    else:
        async with aiohttp.ClientSession() as s:
            status, data = await _get(s, url, token)
    if status == 404:
        return None
    if isinstance(data, list):
        data_list = typing.cast("list[object]", data)
        names: list[str] = []
        for entry in data_list:
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            entry_dict = typing.cast("dict[str, object]", entry)
            name = entry_dict["name"]
            if isinstance(name, str):
                names.append(name)
        return names
    return None


async def resolve_ref(
    owner: str,
    repo: str,
    ref: str,
    token: str | None = None,
    *,
    session: aiohttp.ClientSession | None = None,
) -> str | None:
    url = f"{_API_BASE}/repos/{owner}/{repo}/commits/{ref}"
    if session is not None:
        status, data = await _get(session, url, token)
    else:
        async with aiohttp.ClientSession() as s:
            status, data = await _get(s, url, token)
    if status == 404:
        return None
    if isinstance(data, dict) and "sha" in data:
        data_dict = typing.cast("dict[str, object]", data)
        sha_value = data_dict["sha"]
        if isinstance(sha_value, str):
            return sha_value
    return None
