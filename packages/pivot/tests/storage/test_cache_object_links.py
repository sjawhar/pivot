"""Which symlinks a directory artifact counts as the cache object they name.

Every case is checked against ``_helper_resolved_key``, the check as it stood at
sjawhar/pivot#516's head (``resolve()`` on the link and on the cache directory, for every link):
the cheap check must give the same answer for every tree it can read. A link it cannot read
raises (the tests at the end of the file): #516's ``resolve()`` hid a readlink error that came
after a good lstat, and the link was left out of the manifest as a user symlink.
"""

from __future__ import annotations

import errno
import logging
import os
import pathlib
import shutil
import tempfile
from typing import TYPE_CHECKING

import pytest
import xxhash

from pivot import exceptions
from pivot.show import selection
from pivot.storage import cache, listing, state
from pivot.types import DirHash, HashInfo, is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    _Maker = Callable[[pathlib.Path, pathlib.Path, pathlib.Path, bytes], None]
    _LoopBack = Callable[[pathlib.Path, pathlib.Path], None]

_CONTENT = b"bytes held in the cache\n"
_NOT_HEX = "zz" + "z" * 14


def _helper_key(content: bytes) -> str:
    return xxhash.xxh64(content).hexdigest()


def _with_members(cache_dir: pathlib.Path, record: HashInfo) -> tuple[DirHash, listing.DirListing]:
    """Return a record and its cached listing for tests that inspect members."""
    assert is_dir_hash(record)
    members = cache.cached_listing(cache_dir, record)
    assert members is not None
    return record, members


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
    cache_root = cache.resolve_cache_root(cache_dir)

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
    cache_root = cache.resolve_cache_root(cache_dir)

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

    members = {entry["relpath"]: entry["hash"] for entry in result["listing"]["files"]}
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

    assert [entry["relpath"] for entry in before["listing"]["files"]] == ["member.bin"]
    assert after["listing"]["files"] == [], "the hash through the link before is not reused"
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
    output_hash, output_members = _with_members(
        cache_dir, cache.save_to_cache(source, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK)
    )
    assert is_dir_hash(output_hash)
    restored = tmp_path / "restored"
    resolved = _helper_recording_realpath(monkeypatch)

    assert cache.restore_from_cache(
        restored,
        output_hash,
        cache_dir,
        checkout_mode=cache.CheckoutMode.SYMLINK,
        members=output_members,
    )

    under_cache = [path for path in resolved if path.startswith(os.fspath(cache_dir))]
    assert under_cache == [os.fspath(cache_dir)]
    assert not restored.is_symlink(), "No tree was cached, so each member is its own link"
    for entry in output_members["files"]:
        expected = cache.get_cache_path(cache_dir, entry["hash"]).resolve()
        assert os.readlink(restored / entry["relpath"]) == os.fspath(expected)


def test_hash_directory_of_a_symlink_checkout_resolves_only_the_cache_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hashing a file-by-file symlink checkout resolves the cache directory once, not each link:
    at 180,000 links the per-link resolve was most of a status after the checkout."""
    source, cache_dir = _helper_source(tmp_path, 40)
    output_hash, output_members = _with_members(
        cache_dir, cache.save_to_cache(source, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK)
    )
    restored = tmp_path / "restored"
    assert cache.restore_from_cache(
        restored,
        output_hash,
        cache_dir,
        checkout_mode=cache.CheckoutMode.SYMLINK,
        members=output_members,
    )
    assert all(member.is_symlink() for member in restored.iterdir())
    resolved = _helper_recording_realpath(monkeypatch)

    result = cache.hash_directory(restored, cache_dir=cache_dir)

    assert result["record"]["hash"] == output_hash["hash"]
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
    cache_root = cache.resolve_cache_root(cache_dir)
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

    assert [entry["relpath"] for entry in before["listing"]["files"]] == ["member.bin"]
    assert after["listing"]["files"] == []
    assert _helper_resolved_key(link, cache_dir) is None
    assert link.resolve() != own.resolve()
    assert not cache._links_to_object(link, own, cache.resolve_cache_root(cache_dir))


def test_a_member_link_ending_in_a_loop_inside_the_cache_is_left_out(
    tmp_path: pathlib.Path,
) -> None:
    """A link holding its object's real path, where the object's name and another name beside it
    link to each other, ends in a loop inside the cache: it is no object, so the walk leaves it
    out. (The ``loop`` case above loops outside the cache, where the prefix test refuses it
    before the loop matters.)"""
    cache_dir = tmp_path / "cache" / "files"
    root = tmp_path / "tree"
    root.mkdir()
    obj = _helper_object(cache_dir, _CONTENT)
    (root / "member.bin").symlink_to(obj.resolve())
    other = obj.parent / ("f" * len(obj.name))
    obj.unlink()
    obj.symlink_to(other)
    other.symlink_to(obj)

    assert cache.hash_directory(root, cache_dir=cache_dir)["listing"]["files"] == []
    assert _helper_resolved_key(root / "member.bin", cache_dir) is None, "resolve() refused it"


def test_the_cache_root_is_a_path_boundary_not_a_string_prefix(tmp_path: pathlib.Path) -> None:
    """A link to ``<cache_root>XX/NAME``, a file in a directory beside the cache's object root
    whose name is the root's name followed by two characters, is outside the cache although its
    text starts with the root's text: the cache is tested by path components, as
    ``is_relative_to`` tested it."""
    cache_dir = tmp_path / "cache" / "files"
    root = tmp_path / "tree"
    root.mkdir()
    obj = _helper_object(cache_dir, _CONTENT)
    sibling = pathlib.Path(os.path.realpath(cache_dir) + obj.parent.name, obj.name)
    sibling.parent.mkdir()
    sibling.write_bytes(_CONTENT)
    (root / "member.bin").symlink_to(sibling)

    assert cache.hash_directory(root, cache_dir=cache_dir)["listing"]["files"] == []
    assert _helper_resolved_key(root / "member.bin", cache_dir) is None, "resolve() refused it"


def test_link_text_with_dotdot_under_the_cache_root_is_resolved_in_full(
    tmp_path: pathlib.Path,
) -> None:
    """``<cache_root>/XX/../XX/NAME`` ends at the object, so it counts, as resolve() said: only
    the text ``<cache_root>/XX/NAME`` itself is taken as the link's real path; text with more
    components after the cache root is resolved in full."""
    cache_dir = tmp_path / "cache" / "files"
    root = tmp_path / "tree"
    root.mkdir()
    obj = _helper_object(cache_dir, _CONTENT)
    prefix = obj.parent.name
    (root / "member.bin").symlink_to(
        f"{os.path.realpath(cache_dir)}/{prefix}/../{prefix}/{obj.name}"
    )

    result = cache.hash_directory(root, cache_dir=cache_dir)

    assert [(e["relpath"], e["hash"]) for e in result["listing"]["files"]] == [
        ("member.bin", _helper_key(_CONTENT))
    ]
    assert _helper_resolved_key(root / "member.bin", cache_dir) == _helper_key(_CONTENT)


def _helper_cache_through_symlink(tmp_path: pathlib.Path) -> pathlib.Path:
    """The cache's object root, reached through a symlinked cache directory."""
    real = tmp_path / "real-cache"
    (real / "files").mkdir(parents=True)
    alias = tmp_path / "cache-alias"
    alias.symlink_to(real, target_is_directory=True)
    return alias / "files"


def _helper_symlink_checkout(
    tmp_path: pathlib.Path, cache_dir: pathlib.Path, restored: pathlib.Path, suffix: str = ".txt"
) -> tuple[DirHash, listing.DirListing]:
    """Three files saved by hardlink, then restored at ``restored`` file by file as symlinks."""
    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        (source / f"f{index}{suffix}").write_text(f'{{"i": {index}}}\n')
    output_hash, output_members = _with_members(
        cache_dir, cache.save_to_cache(source, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK)
    )
    restored.parent.mkdir(parents=True, exist_ok=True)
    assert cache.restore_from_cache(
        restored,
        output_hash,
        cache_dir,
        checkout_mode=cache.CheckoutMode.SYMLINK,
        members=output_members,
    )
    assert all(member.is_symlink() for member in restored.iterdir())
    return output_hash, output_members


# Each loop-back maker replaces a part of the cache object's path ``obj`` with a symlink back to
# ``member``, the member link that names the object, so following the member returns to it.


def _object_name_back_to(member: pathlib.Path, obj: pathlib.Path) -> None:
    """member -> object name -> member."""
    os.chmod(obj.parent, 0o755)
    obj.unlink()
    obj.symlink_to(member)


def _prefix_directory_back_to(member: pathlib.Path, obj: pathlib.Path) -> None:
    """member -> two-character directory -> member."""
    prefix = obj.parent
    os.chmod(prefix, 0o755)
    prefix.rename(prefix.parent.parent / "moved-prefix")
    prefix.symlink_to(member)


_LOOP_BACK = [
    pytest.param(_object_name_back_to, id="object-name"),
    pytest.param(_prefix_directory_back_to, id="prefix-directory"),
]


@pytest.mark.parametrize("loop_back", _LOOP_BACK)
def test_links_to_object_answers_as_resolve_for_a_member_looping_through_its_object(
    tmp_path: pathlib.Path, loop_back: _LoopBack
) -> None:
    """A member link holding its object's real path, the text a checkout writes, whose object
    name or two-character directory links back to the member, does not resolve where the
    object's path does: resolve() stops at a different link in each. The text settles it only
    when neither is a symlink, so ``_links_to_object`` gives resolve()'s answer."""
    cache_dir = tmp_path / "cache" / "files"
    restored = tmp_path / "restored"
    output_hash, output_members = _helper_symlink_checkout(tmp_path, cache_dir, restored)
    looped = output_members["files"][0]
    member = restored / looped["relpath"]
    obj = cache.get_cache_path(cache_dir, looped["hash"])
    loop_back(member, obj)
    cache_root = cache.resolve_cache_root(cache_dir)

    assert os.readlink(member) == os.path.join(cache_root, obj.parent.name, obj.name)
    assert member.resolve() != obj.resolve(), "resolve() stops at a different link in each"
    assert not cache._links_to_object(member, obj, cache_root)


@pytest.mark.parametrize("loop_back", _LOOP_BACK)
def test_only_missing_calls_a_member_looping_through_its_object_a_local_change(
    tmp_path: pathlib.Path, loop_back: _LoopBack
) -> None:
    """``checkout --only-missing``: a member link holding its object's real path, whose object
    name or two-character directory links back to that member, does not link to its object, so
    it is a local change the cache cannot restore, and the missing member is not restored over
    it."""
    cache_dir = tmp_path / "cache" / "files"
    restored = tmp_path / "restored"
    output_hash, output_members = _helper_symlink_checkout(tmp_path, cache_dir, restored)
    looped, deleted = output_members["files"][0], output_members["files"][1]
    loop_back(restored / looped["relpath"], cache.get_cache_path(cache_dir, looped["hash"]))
    (restored / deleted["relpath"]).unlink()

    result = cache.restore_missing_in_directory(
        restored,
        output_members,
        cache_dir,
        [cache.CheckoutMode.SYMLINK],
    )

    assert result == cache.MissingRestore(
        restored=0, missing=0, local_changes=[looped["relpath"]], replace=False
    )
    assert not (restored / deleted["relpath"]).is_symlink(), "nothing is written"


def test_only_missing_resolves_a_symlinked_cache_directory(tmp_path: pathlib.Path) -> None:
    """``checkout --only-missing`` with the cache reached through a symlink: a missing member is
    restored as a link to its object's real path, and a member re-pointed at another cached
    object is a change the cache holds, so the directory may be replaced."""
    cache_dir = _helper_cache_through_symlink(tmp_path)
    restored = tmp_path / "restored"
    output_hash, output_members = _helper_symlink_checkout(tmp_path, cache_dir, restored)
    first, second, third = output_members["files"]

    (restored / third["relpath"]).unlink()
    result = cache.restore_missing_in_directory(
        restored,
        output_members,
        cache_dir,
        [cache.CheckoutMode.SYMLINK],
    )
    assert result == cache.MissingRestore(restored=1, missing=0, local_changes=[], replace=False)
    third_object = cache.get_cache_path(cache_dir, third["hash"])
    assert os.readlink(restored / third["relpath"]) == os.fspath(third_object.resolve())

    member = restored / first["relpath"]
    member.unlink()
    member.symlink_to(cache.get_cache_path(cache_dir, second["hash"]).resolve())
    result = cache.restore_missing_in_directory(
        restored,
        output_members,
        cache_dir,
        [cache.CheckoutMode.SYMLINK],
    )
    assert result == cache.MissingRestore(restored=0, missing=0, local_changes=[], replace=True)


def test_diff_resolves_a_symlinked_cache_directory(tmp_path: pathlib.Path) -> None:
    """``pivot diff`` walks a directory output's workspace members with the record-time walker;
    with the cache reached through a symlink, a symlink checkout's member links are members."""
    cache_dir = _helper_cache_through_symlink(tmp_path)
    proj = tmp_path / "proj"
    output_hash, output_members = _helper_symlink_checkout(
        tmp_path, cache_dir, proj / "data" / "out", suffix=".jsonl"
    )
    output = selection._DiffOutput(
        path="data/out",
        head=output_hash,
        head_listing=output_members,
        in_workspace=True,
        is_directory=True,
        changed_kind=None,
        state_dir=proj / ".pivot",
    )

    members = selection._workspace_members(output, None, proj, cache_dir, None, [])

    assert sorted(members) == [f"data/out/{e['relpath']}" for e in output_members["files"]]


def test_publish_tree_resolves_a_symlinked_cache_directory(tmp_path: pathlib.Path) -> None:
    """A file-by-file symlink checkout published as a whole-directory tree through a symlinked
    cache has every member (not copied as a user link). Publishing leaves its tree unmarked; the
    commit's `_ensure_tree_complete` verifies the layout and writes the content-checked marker."""
    cache_dir = _helper_cache_through_symlink(tmp_path)
    restored = tmp_path / "restored"
    output_hash, output_members = _helper_symlink_checkout(tmp_path, cache_dir, restored)
    tree = cache.tree_path(cache_dir, output_hash["hash"])

    members = output_members
    cache._publish_tree(restored, tree, members, cache_dir)

    assert sorted(p.name for p in tree.iterdir()) == [e["relpath"] for e in output_members["files"]]
    assert not cache._tree_layout_marker(tree).exists()
    assert cache._ensure_tree_complete(tree, members, cache_dir, repair=True)
    assert cache._tree_layout_marker(tree).exists()


def _helper_refusal(call: Callable[[], str]) -> exceptions.UnreadableLinkError:
    """The ``UnreadableLinkError`` ``call`` raises. Without one, the test fails with what
    ``call`` says it returned (at #516's head, the members it recorded)."""
    try:
        outcome = call()
    except exceptions.UnreadableLinkError as error:
        return error
    pytest.fail(outcome)


def _helper_hash(restored: pathlib.Path, cache_dir: pathlib.Path) -> str:
    result = cache.hash_directory(restored, cache_dir=cache_dir, warn_left_out_symlinks=True)
    return f"recorded {[e['relpath'] for e in result['listing']['files']]}"


def _helper_assert_unreadable(
    error: exceptions.UnreadableLinkError, link: pathlib.Path, cause: type[OSError], code: int
) -> None:
    """The refusal names ``link``, says what to do, and keeps the OSError behind it."""
    assert error.path == os.fspath(link)
    assert isinstance(error.__cause__, cause)
    assert error.__cause__.errno == code
    assert str(error).startswith(f"Cannot read symlink '{link}': ")
    assert str(error).endswith("readable, or remove the link.")


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches any directory")
def test_hash_directory_raises_for_a_member_link_into_an_unsearchable_prefix_directory(
    tmp_path: pathlib.Path,
) -> None:
    """walk_artifact_members: an error that leaves an entry unclassified propagates, since a
    partial hash would be wrong. A symlink checkout's member link whose object this user cannot
    stat (its two-character directory lost search permission, as in a shared cache another user
    owns) raises, as the base raised ``PermissionError`` from ``Path.is_file()``; it is not left
    out of the manifest as a user symlink."""
    cache_dir = tmp_path / "cache" / "files"
    restored = tmp_path / "restored"
    output_hash, output_members = _helper_symlink_checkout(tmp_path, cache_dir, restored)
    prefix = cache.get_cache_path(cache_dir, output_members["files"][0]["hash"]).parent
    os.chmod(prefix, 0)
    try:
        error = _helper_refusal(lambda: _helper_hash(restored, cache_dir))
    finally:
        os.chmod(prefix, 0o755)
    link = restored / output_members["files"][0]["relpath"]
    _helper_assert_unreadable(error, link, PermissionError, errno.EACCES)


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches any directory")
def test_hash_directory_raises_for_member_links_in_a_listable_unsearchable_directory(
    tmp_path: pathlib.Path,
) -> None:
    """A directory inside the artifact this user can list but not search (mode r--): its member
    links cannot be read, so the hash raises, as the base raised ``PermissionError`` from
    ``Path.is_symlink()``; the links are not left out of the manifest as user symlinks."""
    cache_dir = tmp_path / "cache" / "files"
    source = tmp_path / "source"
    (source / "sub").mkdir(parents=True)
    for index in range(2):
        (source / "sub" / f"f{index}.txt").write_text(f"{index}\n")
    output_hash, output_members = _with_members(
        cache_dir, cache.save_to_cache(source, cache_dir, checkout_mode=cache.CheckoutMode.HARDLINK)
    )
    restored = tmp_path / "restored"
    assert cache.restore_from_cache(
        restored,
        output_hash,
        cache_dir,
        checkout_mode=cache.CheckoutMode.SYMLINK,
        members=output_members,
    )
    sub = restored / "sub"
    assert all(member.is_symlink() for member in sub.iterdir())
    os.chmod(sub, 0o400)
    try:
        error = _helper_refusal(lambda: _helper_hash(restored, cache_dir))
    finally:
        os.chmod(sub, 0o755)
    _helper_assert_unreadable(
        error, sub / pathlib.Path(error.path).name, PermissionError, errno.EACCES
    )


def _helper_fail_readlink(monkeypatch: pytest.MonkeyPatch, victim: pathlib.Path) -> None:
    """``os.readlink`` raises EIO for ``victim``, as on a faulty FUSE or NFS mount. ``os.lstat``
    is not patched, so ``victim`` is still a symlink to ``Path.is_symlink()`` and the walk."""
    real_readlink = os.readlink
    target = os.fspath(victim)

    def failing_readlink(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> str:
        if os.fspath(path) == target:
            raise OSError(errno.EIO, os.strerror(errno.EIO), target)
        return real_readlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "readlink", failing_readlink)
    assert os.path.islink(target), "lstat still works"


def test_hash_of_a_checkout_link_that_cannot_be_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink checkout's member link that can be lstat'd but not read raises, so ``pivot
    commit`` refuses the output. #516's ``resolve()`` hid the readlink error, so the link was
    left out as a user symlink and the commit recorded the other members only."""
    cache_dir = tmp_path / "cache" / "files"
    restored = tmp_path / "restored"
    output_hash, output_members = _helper_symlink_checkout(tmp_path, cache_dir, restored)
    victim = restored / output_members["files"][0]["relpath"]
    _helper_fail_readlink(monkeypatch, victim)
    error = _helper_refusal(lambda: _helper_hash(restored, cache_dir))
    _helper_assert_unreadable(error, victim, OSError, errno.EIO)


def test_only_missing_with_a_checkout_link_that_cannot_be_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``checkout --only-missing`` over the same tree, with another member deleted: the link it
    cannot read raises. #516 called that member a local change the cache cannot restore."""
    cache_dir = tmp_path / "cache" / "files"
    restored = tmp_path / "restored"
    output_hash, output_members = _helper_symlink_checkout(tmp_path, cache_dir, restored)
    deleted = restored / output_members["files"][1]["relpath"]
    deleted.unlink()
    victim = restored / output_members["files"][0]["relpath"]
    _helper_fail_readlink(monkeypatch, victim)

    def only_missing() -> str:
        result = cache.restore_missing_in_directory(
            restored,
            output_members,
            cache_dir,
            [cache.CheckoutMode.SYMLINK],
        )
        return f"returned {result}"

    error = _helper_refusal(only_missing)
    _helper_assert_unreadable(error, victim, OSError, errno.EIO)
    assert not deleted.is_symlink(), "nothing is written"


def test_hash_of_a_user_symlink_that_cannot_be_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user symlink, to a file outside the cache, that can be lstat'd but not read raises too:
    without its text the check cannot tell it from a checkout link. #516 left it out, which was
    the right answer for this link."""
    cache_dir = tmp_path / "cache" / "files"
    restored = tmp_path / "restored"
    _helper_symlink_checkout(tmp_path, cache_dir, restored)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n")
    user_link = restored / "u.txt"
    user_link.symlink_to(outside)
    _helper_fail_readlink(monkeypatch, user_link)
    error = _helper_refusal(lambda: _helper_hash(restored, cache_dir))
    _helper_assert_unreadable(error, user_link, OSError, errno.EIO)
