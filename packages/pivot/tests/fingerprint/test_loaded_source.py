"""`loaded_source`: the source each user module was compiled from, as fingerprinting reads it."""

import inspect
import linecache
import pathlib
import sys
import types
import uuid

import pytest
import xxhash

from pivot import loaded_source


def _load(path: pathlib.Path, text: str, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Write `text` to `path`, compile and run it as a module, and record it as loaded."""
    data = text.encode()
    path.write_bytes(data)
    name = f"loaded_{uuid.uuid4().hex}"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(data, str(path), "exec"), vars(module))
    loaded_source.record(str(path), name, data)
    return module


def test_getsource_reads_the_loaded_text_after_the_file_is_edited(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`inspect.getsource` re-reads an edited file; `loaded_source.getsource` returns the text
    the function was compiled from."""
    path = tmp_path / "helper.py"
    module = _load(path, 'def value():\n    return "alpha"\n', monkeypatch)
    path.write_text('def value():\n    return "bravo, edited after the load"\n')

    assert "bravo" in inspect.getsource(module.value), "inspect reads the edited file"
    assert loaded_source.getsource(module.value) == 'def value():\n    return "alpha"\n'
    assert "alpha" in inspect.getsource(module.value), "the loaded text stays in linecache"


def test_getsource_follows_a_functools_cache_wrapper_to_its_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `functools.cache` helper has no code object of its own; its source is the wrapped
    function's, read from the loaded text as for any function."""
    path = tmp_path / "cached.py"
    loaded = "import functools\n\n\n@functools.cache\ndef table(n):\n    return tuple(range(n))\n"
    module = _load(path, loaded, monkeypatch)
    path.write_text(loaded.replace("range(n)", "range(n + 1000)"))

    assert loaded_source.getsource(module.table) == (
        "@functools.cache\ndef table(n):\n    return tuple(range(n))\n"
    )


def test_getsource_restores_the_loaded_text_after_linecache_is_cleared(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "helper.py"
    module = _load(path, "class Settings:\n    SCALE = 2\n", monkeypatch)
    path.write_text("class Settings:\n    SCALE = 3000\n")
    assert loaded_source.getsource(module.Settings) == "class Settings:\n    SCALE = 2\n"

    linecache.clearcache()

    assert loaded_source.getsource(module.Settings) == "class Settings:\n    SCALE = 2\n"


def test_a_file_loaded_again_replaces_its_text(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "helper.py"
    first = _load(path, 'def value():\n    return "alpha"\n', monkeypatch)
    assert loaded_source.getsource(first.value) == 'def value():\n    return "alpha"\n'
    token = loaded_source.token()

    again = _load(path, 'def value():\n    return "bravo"\n', monkeypatch)

    assert loaded_source.getsource(again.value) == 'def value():\n    return "bravo"\n'
    assert loaded_source.digest(str(path)) == (
        xxhash.xxh64(b'def value():\n    return "bravo"\n').hexdigest()
    )
    assert loaded_source.token() != token, "the token follows the record"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"def f():\r\n    return 1\r\n", id="crlf"),
        pytest.param(b"def f():\n    return 1", id="no-final-newline"),
        pytest.param(b"", id="empty"),
        pytest.param(b"\xef\xbb\xbfdef f():\n    return '\xc3\xa9'\n", id="utf-8-bom"),
        pytest.param(b"# -*- coding: latin-1 -*-\nX = '\xe9'\n", id="latin-1-cookie"),
    ],
)
def test_loaded_lines_are_the_lines_linecache_reads_from_the_file(
    tmp_path: pathlib.Path, data: bytes
) -> None:
    """The text fingerprinting hashes for an unedited file is the text `inspect` reads from it,
    so recorded fingerprints do not change."""
    path = tmp_path / "source.py"
    path.write_bytes(data)

    assert loaded_source._source_lines(data) == linecache.updatecache(str(path))


def test_digest_answers_by_module_file_and_by_resolved_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    _load(link / "mod.py", "VALUE = 1\n", monkeypatch)

    expected = xxhash.xxh64(b"VALUE = 1\n").hexdigest()
    resolved = str((real / "mod.py").resolve())
    assert loaded_source.digest(str(link / "mod.py")) == expected
    assert loaded_source.digest(resolved) == expected
    assert loaded_source.digests()[resolved] == expected
    assert loaded_source.digest(str(tmp_path / "never_loaded.py")) is None


def test_prune_forgets_files_whose_module_is_no_longer_loaded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An import session's end removes its modules; their entries go with them, so a process
    that discovers many projects (a test run) keeps only what is loaded."""
    kept = _load(tmp_path / "kept.py", "VALUE = 1\n", monkeypatch)
    gone = _load(tmp_path / "gone.py", "VALUE = 2\n", monkeypatch)
    token = loaded_source.token()
    monkeypatch.delitem(sys.modules, gone.__name__)

    loaded_source.prune()

    assert loaded_source.digest(str(tmp_path / "gone.py")) is None
    assert loaded_source.digest(str(tmp_path / "kept.py")) is not None
    assert kept.__name__ in sys.modules
    assert loaded_source.token() != token


def test_edited_modules_compares_with_the_coordinators_digests(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker names each module it loaded from a file the coordinator loaded with other
    content, given the digests it started with and the token the stage carries."""
    monkeypatch.setattr(loaded_source, "_coordinator", None)
    shared = _load(tmp_path / "shared.py", "VALUE = 2\n", monkeypatch)
    resolved = str((tmp_path / "shared.py").resolve())
    assert loaded_source.edited_modules("any", None) == [], "the coordinator compares nothing"

    other = {resolved: xxhash.xxh64(b"VALUE = 1\n").hexdigest()}
    loaded_source.set_coordinator(other)
    assert loaded_source.edited_modules(loaded_source._token_of(other), None) == [shared.__name__]

    same = {resolved: xxhash.xxh64(b"VALUE = 2\n").hexdigest()}
    loaded_source.set_coordinator(same)
    assert loaded_source.edited_modules(loaded_source._token_of(same), None) == []

    elsewhere = {str(tmp_path / "elsewhere.py"): "0" * 16}
    loaded_source.set_coordinator(elsewhere)
    assert loaded_source.edited_modules(loaded_source._token_of(elsewhere), None) == []


def test_a_stage_that_carries_digests_is_compared_with_those(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coordinator loaded user code after these workers started, so the stage carries the
    digests it was fingerprinted against, and the worker compares with them."""
    monkeypatch.setattr(loaded_source, "_coordinator", None)
    shared = _load(tmp_path / "shared.py", "VALUE = 2\n", monkeypatch)
    resolved = str((tmp_path / "shared.py").resolve())
    started_with = {resolved: xxhash.xxh64(b"VALUE = 2\n").hexdigest()}
    loaded_source.set_coordinator(started_with)
    sent = {resolved: xxhash.xxh64(b"VALUE = 1\n").hexdigest(), "/other.py": "0" * 16}

    assert loaded_source.edited_modules(loaded_source._token_of(sent), sent) == [shared.__name__]
    assert loaded_source.edited_modules(loaded_source._token_of(started_with), None) == []


def test_edited_modules_counts_every_own_module_when_the_coordinator_moved_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage sent with a token other than the one the worker started with, and without the
    digests, cannot be checked: every module the worker loaded itself counts, so the stage
    reruns rather than reusing."""
    monkeypatch.setattr(loaded_source, "_coordinator", None)
    shared = _load(tmp_path / "shared.py", "VALUE = 2\n", monkeypatch)
    same = {str((tmp_path / "shared.py").resolve()): xxhash.xxh64(b"VALUE = 2\n").hexdigest()}
    loaded_source.set_coordinator(same)

    assert shared.__name__ in loaded_source.edited_modules("a-later-token", None)


def test_a_stage_carries_digests_only_after_the_record_changed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers get the digests once, when they start; a stage carries only the token until the
    coordinator loads more user code."""
    monkeypatch.setattr(loaded_source, "_sent_token", None)
    _load(tmp_path / "one.py", "VALUE = 1\n", monkeypatch)

    assert loaded_source.digests_for_workers() == loaded_source.digests()
    assert loaded_source.digests_for_stage() is None

    _load(tmp_path / "two.py", "VALUE = 2\n", monkeypatch)

    assert loaded_source.digests_for_stage() == loaded_source.digests()
