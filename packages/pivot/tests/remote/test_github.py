from __future__ import annotations

import asyncio
import base64
import logging
import urllib.parse
from typing import TYPE_CHECKING, Any

import pytest
import yaml
import yarl

from pivot.remote import github

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from pytest_mock import MockerFixture

# parse_github_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        pytest.param(
            "https://github.com/org/repo",
            ("org", "repo"),
            id="https",
        ),
        pytest.param(
            "https://github.com/org/repo.git",
            ("org", "repo"),
            id="https-git-suffix",
        ),
        pytest.param(
            "git@github.com:org/repo.git",
            ("org", "repo"),
            id="ssh",
        ),
        pytest.param(
            "https://github.com/org/repo/",
            ("org", "repo"),
            id="https-trailing-slash",
        ),
        pytest.param(
            "http://github.com/org/repo",
            ("org", "repo"),
            id="http",
        ),
    ],
)
def test_parse_github_url(url: str, expected: tuple[str, str]) -> None:
    """GitHub URL variants are parsed to (owner, repo) tuple."""
    assert github.parse_github_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("https://gitlab.com/org/repo", id="gitlab"),
        pytest.param("s3://bucket/path", id="s3"),
        pytest.param("not-a-url", id="junk"),
    ],
)
def test_parse_github_url_invalid_raises(url: str) -> None:
    """Non-GitHub URLs raise ValueError."""
    with pytest.raises(ValueError, match="Not a GitHub URL"):
        github.parse_github_url(url)


# is_github_url


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/org/repo",
        "https://github.com/org/repo.git",
        "git@github.com:org/repo.git",
    ],
)
def test_is_github_url_true(url: str) -> None:
    """GitHub URLs are detected."""
    assert github.is_github_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/org/repo",
        "s3://bucket/path",
        "not-a-url",
    ],
)
def test_is_github_url_false(url: str) -> None:
    """Non-GitHub URLs are rejected."""
    assert github.is_github_url(url) is False


# get_token


def test_get_token_from_gh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """GH_TOKEN is preferred over GITHUB_TOKEN."""
    monkeypatch.setenv("GH_TOKEN", "gh-tok")
    monkeypatch.setenv("GITHUB_TOKEN", "github-tok")
    assert github.get_token() == "gh-tok"


def test_get_token_from_github_token_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """GITHUB_TOKEN is used when GH_TOKEN is not set."""
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "github-tok")
    assert github.get_token() == "github-tok"


def test_get_token_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns None when no token env vars are set."""
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert github.get_token() is None


# Helpers for mocking aiohttp


def _mock_aiohttp_session(
    mocker: MockerFixture,
    status: int,
    json_data: object = None,
    read_data: bytes = b"",
    text_data: str = "",
    headers: dict[str, str] | None = None,
) -> None:
    mock_response = mocker.AsyncMock()
    mock_response.status = status
    mock_response.json = mocker.AsyncMock(return_value=json_data)
    mock_response.read = mocker.AsyncMock(return_value=read_data)
    mock_response.text = mocker.AsyncMock(return_value=text_data)
    mock_response.headers = headers or {}
    mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = mocker.AsyncMock(return_value=False)
    mock_response.raise_for_status = mocker.Mock()

    mock_session = mocker.AsyncMock()
    mock_session.get = mocker.Mock(return_value=mock_response)
    mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("pivot.remote.github.aiohttp.ClientSession", return_value=mock_session)


# read_file


def test_read_file_success(mocker: MockerFixture) -> None:
    """200 response with base64 content returns decoded bytes."""
    content = b"hello world"
    encoded = base64.b64encode(content).decode()
    _mock_aiohttp_session(mocker, 200, json_data={"content": encoded})

    result = asyncio.run(github.read_file("org", "repo", "path/file.txt", "main"))

    assert result == content


def test_read_file_not_found(mocker: MockerFixture) -> None:
    """404 response returns None."""
    _mock_aiohttp_session(mocker, 404)

    result = asyncio.run(github.read_file("org", "repo", "missing.txt", "main"))

    assert result is None


def test_read_file_403_access_denied(mocker: MockerFixture) -> None:
    """403 without rate limit indicators raises access denied error."""
    _mock_aiohttp_session(mocker, 403, text_data="Forbidden", headers={})

    with pytest.raises(Exception, match="access denied"):
        asyncio.run(github.read_file("org", "repo", "file.txt", "main"))


def test_read_file_403_rate_limit_via_header(mocker: MockerFixture) -> None:
    """403 with X-RateLimit-Remaining: 0 raises rate limit error."""
    _mock_aiohttp_session(mocker, 403, text_data="", headers={"X-RateLimit-Remaining": "0"})

    with pytest.raises(Exception, match="[Rr]ate limit"):
        asyncio.run(github.read_file("org", "repo", "file.txt", "main"))


def test_read_file_403_secondary_rate_limit(mocker: MockerFixture) -> None:
    """403 with 'abuse' in body raises secondary rate limit error."""
    _mock_aiohttp_session(mocker, 403, text_data='{"message": "abuse detection"}')

    with pytest.raises(Exception, match="secondary rate limit"):
        asyncio.run(github.read_file("org", "repo", "file.txt", "main"))


def test_read_file_rate_limited(mocker: MockerFixture) -> None:
    """429 response raises RemoteError mentioning rate limit."""
    _mock_aiohttp_session(mocker, 429)

    with pytest.raises(Exception, match="[Rr]ate limit"):
        asyncio.run(github.read_file("org", "repo", "file.txt", "main"))


# resolve_ref


def test_resolve_ref_success(mocker: MockerFixture) -> None:
    """Commits API response returns SHA string."""
    sha = "abc123def456789012345678901234567890abcd"
    _mock_aiohttp_session(mocker, 200, json_data={"sha": sha})

    result = asyncio.run(github.resolve_ref("org", "repo", "main"))

    assert result == sha


def test_resolve_ref_not_found(mocker: MockerFixture) -> None:
    """404 response returns None."""
    _mock_aiohttp_session(mocker, 404)

    result = asyncio.run(github.resolve_ref("org", "repo", "nonexistent"))

    assert result is None


def test_resolve_ref_auth_error(mocker: MockerFixture) -> None:
    """403 response raises RemoteError mentioning access denied."""
    _mock_aiohttp_session(mocker, 403, text_data="Forbidden", headers={})

    with pytest.raises(Exception, match="access denied"):
        asyncio.run(github.resolve_ref("org", "repo", "main"))


# list_directory


def test_list_directory_success(mocker: MockerFixture) -> None:
    """Contents API list response preserves names and entry types."""
    _mock_aiohttp_session(
        mocker,
        200,
        json_data=[
            {"name": "train.csv", "type": "file"},
            {"name": "test.csv", "type": "file"},
            {"name": "subdir", "type": "dir"},
        ],
    )

    result = asyncio.run(github.list_directory("org", "repo", "data", "main"))

    assert result == [
        {"name": "train.csv", "type": "file"},
        {"name": "test.csv", "type": "file"},
        {"name": "subdir", "type": "dir"},
    ]


def test_list_directory_not_found(mocker: MockerFixture) -> None:
    """404 response returns None."""
    _mock_aiohttp_session(mocker, 404)

    result = asyncio.run(github.list_directory("org", "repo", "missing", "main"))

    assert result is None


def test_list_directory_auth_error(mocker: MockerFixture) -> None:
    """403 response raises RemoteError mentioning access denied."""
    _mock_aiohttp_session(mocker, 403, text_data="Forbidden", headers={})

    with pytest.raises(Exception, match="access denied"):
        asyncio.run(github.list_directory("org", "repo", "data", "main"))


# GitHub Contents URL encoding.
#
# A source config controls a GitHub Contents path. The reader must request exactly its owner/repo
# contents prefix, segment-encode every path component, and send the imported revision as the sole
# query value. These tests use yarl, the parser aiohttp uses when it sends the recorded URL.

_API = "https://api.github.com/repos/o/r/contents"


_REF = "rev/with?query#fragment"


_STATE_DIRS = [
    "%2e%2e/%2e%2e/%2e%2e/o2/r2/contents/s",
    "x?ref=main&a=",
    "x#y",
    "a%2Fb",
]


class _Response:
    status: int = 404
    headers: dict[str, str] = {}

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False

    async def json(self) -> None:
        return None


class _Session:
    urls: list[str]

    def __init__(self) -> None:
        self.urls = list[str]()

    def get(self, url: str, **_: object) -> _Response:
        self.urls.append(url)
        return _Response()


def _expected_raw_path(path: str) -> str:
    encoded = "/".join(urllib.parse.quote(segment, safe="") for segment in path.split("/"))
    return yarl.URL(f"{_API}/{encoded}").raw_path


@pytest.mark.parametrize("state_dir", _STATE_DIRS)
@pytest.mark.parametrize("reader", [github.read_file, github.list_directory])
def test_contents_reader_encodes_each_path_segment_and_ref(
    state_dir: str,
    reader: Callable[..., Coroutine[Any, Any, object]],
) -> None:
    session = _Session()
    path = f"{state_dir}/stages"

    result = asyncio.run(reader("o", "r", path, _REF, session=session))

    assert result is None
    assert len(session.urls) == 1
    requested = yarl.URL(session.urls[0])
    assert requested.raw_path == _expected_raw_path(path)
    assert dict(requested.query) == {"ref": _REF}


@pytest.mark.parametrize("state_dir", _STATE_DIRS)
def test_github_import_falls_back_without_requesting_another_contents_prefix(
    state_dir: str, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """The encoded failing state-dir request stays in the source contents namespace."""
    from pivot import import_artifact

    config = yaml.safe_dump(
        {"remotes": {"origin": "s3://source-cache"}, "core": {"state_dir": state_dir}}
    ).encode()
    lock_data = {
        "code_manifest": {},
        "params": {},
        "deps": [],
        "outs": [{"key": None, "tag": "data", "hash": "0123456789abcdef", "size": 5}],
    }
    lock = yaml.safe_dump(lock_data).encode()
    requested = list[yarl.URL]()

    async def get(_: object, url: str, __: str | None) -> tuple[int, object]:
        parsed = yarl.URL(url)
        requested.append(parsed)
        path = parsed.path
        if path.endswith("/.pivot/config.yaml"):
            return 200, {"content": base64.b64encode(config).decode()}
        if path.endswith("/.pivot/stages"):
            return 200, [{"name": "test", "type": "dir"}]
        if path.endswith("/.pivot/stages/test"):
            return 200, [{"name": "process.lock", "type": "file"}]
        if path.endswith("/.pivot/stages/test/process.lock"):
            return 200, {"content": base64.b64encode(lock).decode()}
        return 404, None

    mocker.patch.object(github, "resolve_ref", autospec=True, return_value="f" * 40)
    mocker.patch.object(github, "_get", autospec=True, side_effect=get)

    with caplog.at_level(logging.WARNING):
        resolved = asyncio.run(
            import_artifact.resolve_remote_path(
                "https://github.com/o/r", "test/process", "main", None
            )
        )

    assert resolved["hash"] == "0123456789abcdef"
    assert caplog.text.count("Ignoring core.state_dir") == 1
    expected_prefix = "/repos/o/r/contents/"
    expected_state_path = _expected_raw_path(f"{state_dir}/stages")
    assert requested
    assert any(url.raw_path == expected_state_path for url in requested)
    assert all(url.raw_path.startswith(expected_prefix) for url in requested)
    assert all(dict(url.query) == {"ref": "f" * 40} for url in requested)
