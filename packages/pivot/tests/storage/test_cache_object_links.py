"""Which symlinks a directory artifact counts as the cache object they name.

Every case is checked against ``_helper_resolved_key``, the check as it stood at
sjawhar/pivot#516's head (``resolve()`` on the link and on the cache directory, for every link):
the cheap check must give the same answer for every tree.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import tempfile
from typing import TYPE_CHECKING

import pytest
import xxhash

from pivot.storage import cache, state
from pivot.types import is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    _Maker = Callable[[pathlib.Path, pathlib.Path, pathlib.Path, bytes], None]

_CONTENT = b"bytes held in the cache\n"
_NOT_HEX = "zz" + "z" * 14


def _helper_key(content: bytes) -> str:
    return xxhash.xxh64(content).hexdigest()


def _helper_resolved_symlink_hash(path: pathlib.Path, cache_dir: pathlib.Path) -> str | None:
    """``_get_symlink_cache_hash`` at sjawhar/pivot#516's head 0ea51e77, verbatim."""
    if not path.is_symlink():
        return None
    try:
        target = path.resolve()
        cache_resolved = cache_dir.resolve()
        if not target.is_relative_to(cache_resolved):
            return None
        rel = target.relative_to(cache_resolved)
        if len(rel.parts) != 2:
            return None
        reconstructed = rel.parts[0] + rel.parts[1]
        if len(reconstructed) != cache.XXHASH64_HEX_LENGTH:
            return None
        return reconstructed
    except (OSError, ValueError):
        return None


def _helper_resolved_key(path: pathlib.Path, cache_dir: pathlib.Path) -> str | None:
    """``_cache_object_key`` at sjawhar/pivot#516's head 0ea51e77, verbatim."""
    key = _helper_resolved_symlink_hash(path, cache_dir)
    if key is None or not cache.is_valid_hash(key) or not path.is_file():
        return None
    return key


def _helper_object(cache_dir: pathlib.Path, content: bytes) -> pathlib.Path:
    """Publish ``content`` as its cache object, read-only as pivot leaves objects."""
    obj = cache.get_cache_path(cache_dir, _helper_key(content))
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_bytes(content)
    os.chmod(obj, 0o444)
    return obj


def _helper_slot(cache_dir: pathlib.Path, content: bytes) -> pathlib.Path:
    """Where ``content``'s object would be, under the cache's real path, with nothing there."""
    return cache.get_cache_path(cache_dir.resolve(), _helper_key(content))


# Each maker builds one kind of link at ``link`` to ``content``'s object, in its own scratch
# directory ``tmp``, sharing ``cache_dir`` with the others.


def _direct(tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes) -> None:
    """The link a symlink checkout writes: the object's real path."""
    link.symlink_to(_helper_object(cache_dir, content).resolve())


def _chain(tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes) -> None:
    middle = tmp / "middle"
    middle.symlink_to(_helper_object(cache_dir, content).resolve())
    link.symlink_to(middle)


def _relative(
    tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes
) -> None:
    link.symlink_to(os.path.relpath(_helper_object(cache_dir, content), link.parent))


def _through_symlinked_cache_path(
    tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes
) -> None:
    """The object reached through a link to the cache's parent: not the real path's text."""
    obj = _helper_object(cache_dir, content)
    alias = tmp / "alias"
    alias.symlink_to(cache_dir.parent, target_is_directory=True)
    link.symlink_to(alias / cache_dir.name / obj.parent.name / obj.name)


def _inside_prefix_directory(
    tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes
) -> None:
    link.symlink_to(_helper_object(cache_dir, content).parent.resolve())


def _inside_not_a_key(
    tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes
) -> None:
    """A file in the cache at a path that is no object's: three levels down."""
    ref = cache_dir.resolve() / "refs" / "stage" / "_single"
    ref.parent.mkdir(parents=True)
    ref.write_bytes(content)
    link.symlink_to(ref)


def _inside_not_hex(
    tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes
) -> None:
    """Two levels down and 16 characters, but not a hex key."""
    named = cache_dir.resolve() / _NOT_HEX[:2] / _NOT_HEX[2:]
    named.parent.mkdir(parents=True)
    named.write_bytes(content)
    link.symlink_to(named)


def _object_name_links_out(
    tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes
) -> None:
    """The link's text is an object's real path, but the object's name is a link out of the
    cache: the first step is inside the cache, the end is not."""
    outside = tmp / "outside.bin"
    outside.write_bytes(content)
    slot = _helper_slot(cache_dir, content)
    slot.parent.mkdir(parents=True, exist_ok=True)
    slot.symlink_to(outside)
    link.symlink_to(slot)


def _prefix_directory_links_out(
    tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes
) -> None:
    """The link's text is an object's real path, but its two-character directory is a link to a
    directory outside the cache holding a file of that name and content."""
    slot = _helper_slot(cache_dir, content)
    outside = tmp / "outside-prefix"
    outside.mkdir()
    (outside / slot.name).write_bytes(content)
    slot.parent.parent.mkdir(parents=True, exist_ok=True)
    slot.parent.symlink_to(outside, target_is_directory=True)
    link.symlink_to(slot)


def _outside(
    tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes
) -> None:
    _helper_object(cache_dir, content)
    outside = tmp / "outside.bin"
    outside.write_bytes(content)
    link.symlink_to(outside)


def _outside_named_like_an_object(
    tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes
) -> None:
    obj = _helper_object(cache_dir, content)
    outside = tmp / "elsewhere" / obj.parent.name / obj.name
    outside.parent.mkdir(parents=True)
    outside.write_bytes(content)
    link.symlink_to(outside)


def _broken(tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes) -> None:
    """A checkout link whose object is not in the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    link.symlink_to(_helper_slot(cache_dir, content))


def _directory(
    tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes
) -> None:
    """A link to a directory at an object's path, as a whole-directory SYMLINK checkout makes."""
    tree = _helper_slot(cache_dir, content)
    tree.mkdir(parents=True)
    (tree / "member.txt").write_bytes(content)
    link.symlink_to(tree, target_is_directory=True)


def _loop(tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes) -> None:
    _helper_object(cache_dir, content)
    other = tmp / "other"
    other.symlink_to(link)
    link.symlink_to(other)


def _hard_link(
    tmp: pathlib.Path, cache_dir: pathlib.Path, link: pathlib.Path, content: bytes
) -> None:
    os.link(_helper_object(cache_dir, content), link)


_OWN = "own key"  # the expected answer is the key of the case's own content
# kind -> (maker, _cache_object_key's answer, _get_symlink_cache_hash's answer)
_CASES: dict[str, tuple[_Maker, str | None, str | None]] = {
    "direct": (_direct, _OWN, _OWN),
    "link-to-link": (_chain, _OWN, _OWN),
    "relative": (_relative, _OWN, _OWN),
    "through-symlinked-cache-path": (_through_symlinked_cache_path, _OWN, _OWN),
    "inside-cache-prefix-directory": (_inside_prefix_directory, None, None),
    "inside-cache-not-a-key": (_inside_not_a_key, None, None),
    "inside-cache-not-hex": (_inside_not_hex, None, _NOT_HEX),
    "first-step-inside-object-name-links-out": (_object_name_links_out, None, None),
    "first-step-inside-prefix-directory-links-out": (_prefix_directory_links_out, None, None),
    "outside": (_outside, None, None),
    "outside-named-like-an-object": (_outside_named_like_an_object, None, None),
    "broken": (_broken, None, _OWN),
    "directory": (_directory, None, _OWN),
    "loop": (_loop, None, None),
    "hard-link": (_hard_link, None, None),
}


def _helper_expected(answer: str | None, content: bytes) -> str | None:
    return _helper_key(content) if answer == _OWN else answer


@pytest.mark.parametrize("kind", list(_CASES))
def test_cache_object_key_gives_the_resolved_answer(tmp_path: pathlib.Path, kind: str) -> None:
    """Each kind of link gets the answer resolving the whole path gave."""
    make, object_key, symlink_hash = _CASES[kind]
    cache_dir = tmp_path / "cache" / "files"
    link = tmp_path / "work" / "member.bin"
    link.parent.mkdir()
    make(tmp_path, cache_dir, link, _CONTENT)
    cache_root = os.path.realpath(cache_dir)

    found_key = cache._cache_object_key(link, cache_root)
    found_symlink_hash = cache._get_symlink_cache_hash(link, cache_dir)

    assert found_key == _helper_expected(object_key, _CONTENT)
    assert found_symlink_hash == _helper_expected(symlink_hash, _CONTENT)
    assert found_key == _helper_resolved_key(link, cache_dir)
    assert found_symlink_hash == _helper_resolved_symlink_hash(link, cache_dir)
    if link.is_symlink():
        assert cache._resolve_link(link, cache_root) == os.path.realpath(link)


def test_checkout_link_is_settled_without_resolving_the_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The link a checkout writes is settled from its text and two lstats: realpath, which
    walks every component of both paths, is never called for it."""
    cache_dir = tmp_path / "cache" / "files"
    link = tmp_path / "member.bin"
    _direct(tmp_path, cache_dir, link, _CONTENT)
    cache_root = os.path.realpath(cache_dir)

    def no_realpath(path: object, *, strict: bool = False) -> str:
        raise AssertionError(f"realpath({path!r}) called")

    monkeypatch.setattr(os.path, "realpath", no_realpath)

    assert cache._cache_object_key(link, cache_root) == _helper_key(_CONTENT)


def test_hash_directory_counts_the_links_it_counted_before(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a directory artifact, the links that end at their object count as that file; every
    other link is left out and named, and is never read; a hard link to an object is a plain
    file. Only a link to a misnamed object is read, to find that its bytes are not its name's."""
    cache_dir = tmp_path / "cache" / "files"
    root = tmp_path / "tree"
    root.mkdir()
    contents = {kind: f"{kind}\n".encode() for kind in _CASES}
    # The prefix-directory case replaces a two-character directory of the cache with a link,
    # so its content gets a prefix no other object here has.
    prefix_case = "first-step-inside-prefix-directory-links-out"
    taken = {_helper_key(content)[:2] for kind, content in contents.items() if kind != prefix_case}
    taken |= {_NOT_HEX[:2], _helper_key(b"what the name says\n")[:2]}
    suffix = 0
    while _helper_key(contents[prefix_case])[:2] in taken:
        suffix += 1
        contents[prefix_case] = f"prefix {suffix}\n".encode()
    for kind, (make, _, _) in _CASES.items():
        (tmp_path / kind).mkdir()
        make(tmp_path / kind, cache_dir, root / kind, contents[kind])
    misnamed = _helper_slot(cache_dir, b"what the name says\n")
    misnamed.parent.mkdir(parents=True, exist_ok=True)
    misnamed.write_bytes(b"other bytes\n")
    (root / "misnamed-object").symlink_to(misnamed)
    read = list[str]()
    original_hash_contents = cache._hash_contents

    def recording_hash_contents(path: str | os.PathLike[str], size: int) -> str:
        read.append(os.path.relpath(path, root))
        return original_hash_contents(path, size)

    monkeypatch.setattr(cache, "_hash_contents", recording_hash_contents)

    with caplog.at_level(logging.WARNING):
        result = cache.hash_directory(root, cache_dir=cache_dir, warn_left_out_symlinks=True)

    members = {entry["relpath"]: entry["hash"] for entry in result["manifest"]}
    counted = ["direct", "link-to-link", "relative", "through-symlinked-cache-path", "hard-link"]
    assert members == {kind: _helper_key(contents[kind]) for kind in counted}
    left_out = sorted([*(kind for kind in _CASES if kind not in counted), "misnamed-object"])
    (record,) = [r for r in caplog.records if "leaves out" in r.getMessage()]
    assert f"symlinks {', '.join(left_out)} (" in record.getMessage()
    assert sorted(read) == sorted([*counted, "misnamed-object"])


@pytest.fixture
def other_filesystem(tmp_path: pathlib.Path) -> Generator[pathlib.Path]:
    """A directory on another filesystem than ``tmp_path`` (tmpfs ``/dev/shm``) where one is
    writable, else one beside ``tmp_path``. The check goes by path, never by device or inode,
    so the answer is the same either way; the case reviewers asked for crosses a filesystem."""
    shm = pathlib.Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK) and shm.stat().st_dev != tmp_path.stat().st_dev:
        other = pathlib.Path(tempfile.mkdtemp(prefix="pivot-test-", dir=shm))
        try:
            yield other
        finally:
            shutil.rmtree(other)
    else:
        other = tmp_path / "other-filesystem"
        other.mkdir()
        yield other


@pytest.mark.parametrize("repointed", ["member-link", "prefix-directory"])
def test_a_link_repointed_at_another_filesystem_stops_counting(
    tmp_path: pathlib.Path, other_filesystem: pathlib.Path, repointed: str
) -> None:
    """A member link, or the cache's two-character directory, re-pointed at a twin of the
    object on another filesystem (same names, same bytes) is left out on the next hash: the
    check reads the link again and follows it, and keeps nothing from the hash before."""
    cache_dir = tmp_path / "cache" / "files"
    root = tmp_path / "tree"
    root.mkdir()
    obj = _helper_object(cache_dir, _CONTENT)
    link = root / "member.bin"
    link.symlink_to(obj.resolve())
    twin = other_filesystem / obj.parent.name / obj.name
    twin.parent.mkdir()
    twin.write_bytes(_CONTENT)

    with state.StateDB(tmp_path / "state") as db:
        before = cache.hash_directory(root, db, cache_dir=cache_dir)
        if repointed == "member-link":
            link.unlink()
            link.symlink_to(twin)
        else:
            prefix = obj.resolve().parent
            prefix.rename(tmp_path / "moved-prefix")
            prefix.symlink_to(twin.parent, target_is_directory=True)
        after = cache.hash_directory(root, db, cache_dir=cache_dir)

    assert [entry["relpath"] for entry in before["manifest"]] == ["member.bin"]
    assert after["manifest"] == [], "the hash through the link before is not reused"
    assert _helper_resolved_key(link, cache_dir) is None, "and resolve() refuses it too"


def _helper_source(tmp_path: pathlib.Path, files: int) -> tuple[pathlib.Path, pathlib.Path]:
    """A directory to save, and the cache directory to save it in."""
    cache_dir = tmp_path / "cache" / "files"
    source = tmp_path / "source"
    source.mkdir()
    for index in range(files):
        (source / f"f{index:02d}.txt").write_text(f"{index}\n")
    return source, cache_dir


def _helper_recording_realpath(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    resolved = list[str]()
    original_realpath = os.path.realpath

    def recording_realpath(path: str | os.PathLike[str], *, strict: bool = False) -> str:
        resolved.append(os.fspath(path))
        return original_realpath(path, strict=strict)

    monkeypatch.setattr(os.path, "realpath", recording_realpath)
    return resolved


def test_symlink_checkout_names_each_object_without_resolving_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file-by-file symlink restore links each member to its object's real path, the text it
    wrote before, resolving the cache directory once instead of each object's path."""
    source, cache_dir = _helper_source(tmp_path, 40)
    output_hash = cache.save_to_cache(source, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK)
    assert is_dir_hash(output_hash)
    restored = tmp_path / "restored"
    resolved = _helper_recording_realpath(monkeypatch)

    assert cache.restore_from_cache(
        restored, output_hash, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK
    )

    under_cache = [path for path in resolved if path.startswith(os.fspath(cache_dir))]
    assert under_cache == [os.fspath(cache_dir)]
    assert not restored.is_symlink(), "No tree was cached, so each member is its own link"
    for entry in output_hash["manifest"]:
        expected = cache.get_cache_path(cache_dir, entry["hash"]).resolve()
        assert os.readlink(restored / entry["relpath"]) == os.fspath(expected)


def test_hash_directory_of_a_symlink_checkout_resolves_only_the_cache_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hashing a file-by-file symlink checkout resolves the cache directory once, not each link:
    at 180,000 links the per-link resolve was most of a status after the checkout."""
    source, cache_dir = _helper_source(tmp_path, 40)
    output_hash = cache.save_to_cache(source, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK)
    restored = tmp_path / "restored"
    assert cache.restore_from_cache(
        restored, output_hash, cache_dir, checkout_mode=cache.CheckoutMode.SYMLINK
    )
    assert all(member.is_symlink() for member in restored.iterdir())
    resolved = _helper_recording_realpath(monkeypatch)

    result = cache.hash_directory(restored, cache_dir=cache_dir)

    assert result["hash"] == output_hash["hash"]
    assert resolved == [os.fspath(cache_dir)]


@pytest.mark.parametrize(
    "kind",
    [
        "direct",
        "link-to-link",
        "relative",
        "through-symlinked-cache-path",
        "broken",
        "first-step-inside-object-name-links-out",
        "first-step-inside-prefix-directory-links-out",
    ],
)
def test_object_paths_resolve_as_before(tmp_path: pathlib.Path, kind: str) -> None:
    """``checkout --only-missing`` asks whether a member link resolves where its object's path
    does, and a symlink checkout writes the object's real path: both answers are resolve()'s,
    symlinks inside the cache included, for the object the link names and for another."""
    make, _, _ = _CASES[kind]
    cache_dir = tmp_path / "cache" / "files"
    link = tmp_path / "member.bin"
    make(tmp_path, cache_dir, link, _CONTENT)
    cache_root = os.path.realpath(cache_dir)
    own = cache.get_cache_path(cache_dir, _helper_key(_CONTENT))
    other = _helper_object(cache_dir, b"another object\n")

    assert cache._links_to_object(link, own, cache_root) == (link.resolve() == own.resolve())
    assert cache._links_to_object(link, own, cache_root)
    assert not cache._links_to_object(link, other, cache_root)
    assert cache._object_realpath(own, cache_root) == os.fspath(own.resolve())
    assert cache._object_realpath(own, None) == os.fspath(own.resolve())


def test_a_cache_directory_repointed_between_commands_is_resolved_again(
    tmp_path: pathlib.Path, other_filesystem: pathlib.Path
) -> None:
    """A cache directory reached through a symlink, re-pointed at another filesystem between two
    commands, is resolved again by the second: a link into the old cache, with the same names and
    bytes as the new cache's object, no longer counts, and ``--only-missing`` no longer treats it
    as pointing at the object."""
    old_cache = tmp_path / "old-cache" / "files"
    obj = _helper_object(old_cache, _CONTENT)
    cache_alias = tmp_path / "cache-alias"
    cache_alias.symlink_to(old_cache.parent, target_is_directory=True)
    cache_dir = cache_alias / "files"
    root = tmp_path / "tree"
    root.mkdir()
    link = root / "member.bin"
    link.symlink_to(obj.resolve())
    _helper_object(other_filesystem / "files", _CONTENT)
    before = cache.hash_directory(root, cache_dir=cache_dir)

    cache_alias.unlink()
    cache_alias.symlink_to(other_filesystem, target_is_directory=True)
    after = cache.hash_directory(root, cache_dir=cache_dir)
    own = cache.get_cache_path(cache_dir, _helper_key(_CONTENT))

    assert [entry["relpath"] for entry in before["manifest"]] == ["member.bin"]
    assert after["manifest"] == []
    assert _helper_resolved_key(link, cache_dir) is None
    assert link.resolve() != own.resolve()
    assert not cache._links_to_object(link, own, os.path.realpath(cache_dir))
