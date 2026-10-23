"""Round 16 G1 red tests for GitHub Contents URL encoding.

A source config controls a GitHub Contents path. The reader must request exactly its owner/repo
contents prefix, segment-encode every path component, and send the imported revision as the sole
query value. These tests use yarl, the parser aiohttp uses when it sends the recorded URL.
"""

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
