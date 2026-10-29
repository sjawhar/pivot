"""`.pivotignore` and the hash of a directory input: what is left out, and what never is."""

from __future__ import annotations

import os
import pathlib
import tempfile

import hypothesis
import pytest
from hypothesis import strategies as st

from pivot import ignore, loaders, types
from pivot.storage import cache, listing, store

_PYC_PATTERNS = "*.pyc\n__pycache__/\n"
_SETTINGS = hypothesis.settings(max_examples=200, deadline=None, derandomize=True, database=None)


def _helper_member_filter(
    project: pathlib.Path, directory: str, patterns: str
) -> ignore.MemberFilter:
    (project / ".pivotignore").write_text(patterns)
    member_filter = ignore.IgnoreFilter(project).member_filter(directory)
    assert member_filter is not None
    return member_filter


def _helper_source_tree(root: pathlib.Path, *, bytecode: bool) -> None:
    """A source directory a Python process imported from: with ``bytecode``, it wrote .pyc."""
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "m.py").write_text("m = 1\n")
    (root / "grade.py").write_text("print('grade')\n")
    (root / "notes").mkdir()
    if bytecode:
        (root / "pkg" / "__pycache__").mkdir()
        (root / "pkg" / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"\x00pyc m")
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"\x00pyc grade")
        (root / "notes" / "stray.pyc").write_bytes(b"\x00stray")


def _helper_ref(producer: str, key: str | None, tag: types.ArtifactTag) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=producer, key=key),
        format=loaders.PathOnly(),
        python_type=pathlib.Path,
        tag=tag,
    )


# =============================================================================
# cache.hash_directory with a member filter
# =============================================================================


def test_hash_directory_with_a_filter_hashes_the_tree_without_the_ignored_files(
    tmp_path: pathlib.Path,
) -> None:
    """Bytecode written into a source directory leaves its filtered hash as it was; a
    directory left holding only ignored files is kept as an empty directory."""
    root = tmp_path / "src"
    _helper_source_tree(root, bytecode=True)
    clean = tmp_path / "clean"
    _helper_source_tree(clean, bytecode=False)

    filtered = cache.hash_directory(
        root, ignored=_helper_member_filter(tmp_path, "src", _PYC_PATTERNS)
    )

    assert [entry["relpath"] for entry in filtered["listing"]["files"]] == ["grade.py", "pkg/m.py"]
    assert filtered == cache.hash_directory(clean), "Same hash as the tree with no bytecode"
    assert cache.hash_directory(root) != filtered, "Without the filter every file counts"


def test_hash_directory_with_a_filter_that_names_nothing_gives_the_whole_hash(
    tmp_path: pathlib.Path,
) -> None:
    """A project whose inputs hold no ignored file keeps every hash: nothing reruns."""
    root = tmp_path / "src"
    _helper_source_tree(root, bytecode=False)

    filtered = cache.hash_directory(
        root, ignored=_helper_member_filter(tmp_path, "src", _PYC_PATTERNS)
    )

    assert filtered == cache.hash_directory(root)


def test_hash_directory_never_enters_an_ignored_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ignored directory (a virtualenv inside an input) is never listed, so it costs nothing."""
    root = tmp_path / "src"
    _helper_source_tree(root, bytecode=False)
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "site.py").write_text("")
    listed = list[str]()
    real_scandir = os.scandir

    def recording_scandir(path: str) -> object:
        listed.append(os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", recording_scandir)
    member_filter = _helper_member_filter(tmp_path, "src", ".venv/\n")

    _ = cache.hash_directory(root, ignored=member_filter)

    assert listed, "The walk lists the directory"
    assert not [path for path in listed if ".venv" in path], listed


def test_hash_directory_leaves_out_an_ignored_symlink_without_warning(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A symlink .pivotignore names is neither stored nor reported as left out."""
    root = tmp_path / "src"
    _helper_source_tree(root, bytecode=False)
    (root / "link.pyc").symlink_to(root / "grade.py")

    with caplog.at_level("WARNING"):
        filtered = cache.hash_directory(
            root,
            ignored=_helper_member_filter(tmp_path, "src", _PYC_PATTERNS),
            warn_left_out_symlinks=True,
        )

    assert "link.pyc" not in caplog.text
    assert "link.pyc" not in [entry["relpath"] for entry in filtered["listing"]["files"]]


# =============================================================================
# cache.filter_listing: the same listing and hash from a record as from a walk
# =============================================================================

_TREE_NAMES = ("a", "b.pyc", "__pycache__", "build", "keep.txt", "x.log", "caf\u00e9")
_TREE_PATTERNS = (
    "*.pyc",
    "__pycache__/",
    "build/",
    "!build/keep.txt",
    "!keep.txt",
    "/inputs/a/",
    "a/x.log",
    "*.log",
    "b.pyc/",
    "a/**",
    "caf\u0065\u0301",
)
_relpaths = st.lists(st.sampled_from(_TREE_NAMES), min_size=1, max_size=3).map("/".join)


def _helper_build_tree(root: pathlib.Path, files: list[str], empty_dirs: list[str]) -> None:
    """Create each file and empty directory unless a name on its way is already the other kind."""
    root.mkdir()
    for relpath, is_dir in [(path, False) for path in files] + [
        (path, True) for path in empty_dirs
    ]:
        target = root / relpath
        if any(parent.is_file() for parent in target.parents) or target.exists():
            continue
        if is_dir:
            target.mkdir(parents=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(relpath)


@_SETTINGS
@hypothesis.given(
    st.lists(_relpaths, max_size=8),
    st.lists(_relpaths, max_size=3),
    st.lists(st.sampled_from(_TREE_PATTERNS), min_size=1, max_size=5),
)
def test_filter_listing_of_the_whole_listing_equals_the_filtered_walk(
    files: list[str], empty_dirs: list[str], patterns: list[str]
) -> None:
    """An input's dependency hash taken from a record (a .pvt, the whole hash) is the one a
    walk with the filter gives the same tree, so the two can never disagree."""
    with tempfile.TemporaryDirectory() as tmp:
        project = pathlib.Path(tmp)
        root = project / "inputs"
        _helper_build_tree(root, files, empty_dirs)
        (project / ".pivotignore").write_text("\n".join(patterns) + "\n")
        member_filter = ignore.IgnoreFilter(project).member_filter("inputs")
        hypothesis.assume(member_filter is not None)
        assert member_filter is not None

        walked = cache.hash_directory(root, ignored=member_filter)
        recorded = cache.filter_listing(cache.hash_directory(root)["listing"], member_filter)

        assert recorded == walked


# =============================================================================
# store.WorkspaceStore: which dependency hashes leave files out
# =============================================================================


@pytest.mark.parametrize("linked", [False, True], ids=["local", "symlinked-from-outside"])
def test_workspace_store_leaves_ignored_files_out_of_a_directory_input(
    tmp_path: pathlib.Path, linked: bool
) -> None:
    """A directory input is matched by its path inside the project, never by a symlink's
    target, so a directory linked in from outside gets the hash a local one gets."""
    project = tmp_path / "project"
    (project / "tasks").mkdir(parents=True)
    source = tmp_path / "elsewhere" / "server" if linked else project / "tasks" / "server"
    _helper_source_tree(source, bytecode=True)
    (source / "scratch").mkdir()
    (source / "scratch" / "notes.txt").write_text("scratch")
    if linked:
        (project / "tasks" / "server").symlink_to(source, target_is_directory=True)
    (project / ".pivotignore").write_text(_PYC_PATTERNS + "/tasks/server/scratch/\n")
    clean = tmp_path / "clean"
    _helper_source_tree(clean, bytecode=False)
    workspace = store.WorkspaceStore(project, "p", {"server": "tasks/server"})

    dep_hash = workspace.hash_artifact(_helper_ref("server", None, types.ArtifactTag.INPUT))

    assert dep_hash == cache.hash_directory(clean)["record"]


def test_workspace_store_hashes_a_directory_output_whole_as_a_dependency(
    tmp_path: pathlib.Path,
) -> None:
    """An upstream stage's directory output is a dependency by the hash its producer records,
    every file included: .pivotignore never changes an output's hash."""
    (tmp_path / ".pivotignore").write_text(_PYC_PATTERNS)
    workspace = store.WorkspaceStore(tmp_path, "p", {})
    ref = _helper_ref("p/build", None, types.ArtifactTag.DIRECTORY)
    output = workspace.prepare_output(ref)
    _helper_source_tree(output, bytecode=True)

    dep_hash = workspace.hash_artifact(ref)
    recorded = workspace.commit(ref, output)

    assert dep_hash == recorded == cache.hash_directory(output)["record"]
    assert types.is_dir_hash(dep_hash)
    assert "pkg/__pycache__/m.cpython-312.pyc" in [
        entry["relpath"] for entry in cache.hash_directory(output)["listing"]["files"]
    ]


def test_recorded_input_hash_is_the_hash_of_the_input_holding_that_record(
    tmp_path: pathlib.Path,
) -> None:
    """A tracked input missing from the workspace is judged by its .pvt: the whole listing
    recorded there, less what .pivotignore names, gives the hash the restored input would."""
    (tmp_path / ".pivotignore").write_text(_PYC_PATTERNS)
    source = tmp_path / "data" / "raw" / "src"
    _helper_source_tree(source, bytecode=True)
    workspace = store.WorkspaceStore(tmp_path, "p", {})
    ref = _helper_ref("src", None, types.ArtifactTag.INPUT)
    tracked = cache.hash_directory(source)

    from_record = workspace.recorded_input_hash(ref, tracked["record"], lambda: tracked["listing"])

    assert from_record == workspace.hash_artifact(ref)
    assert from_record != tracked["record"]


def test_recorded_input_hash_keeps_an_output_hash_without_reading_its_listing(
    tmp_path: pathlib.Path,
) -> None:
    """A directory no filter applies to keeps its record, and its listing is never read."""
    (tmp_path / ".pivotignore").write_text(_PYC_PATTERNS)
    workspace = store.WorkspaceStore(tmp_path, "p", {})
    directory = tmp_path / "tree"
    _helper_source_tree(directory, bytecode=True)
    whole = cache.hash_directory(directory)["record"]

    def unread() -> listing.DirListing:
        raise AssertionError("a directory with no filter must not read its listing")

    assert (
        workspace.recorded_input_hash(
            _helper_ref("p/o", None, types.ArtifactTag.DIRECTORY), whole, unread
        )
        == whole
    )
