from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING

import pytest

from pivot.remote import github

if TYPE_CHECKING:
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
    """Contents API list response returns file names."""
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

    assert result is not None
    assert sorted(result) == ["subdir", "test.csv", "train.csv"]


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
