"""The pure functions of `pivot.installed_distributions` behind the `dist:` key of an installed
distribution, called in this process on plain data: on `direct_url.json`, METADATA,
`requires.txt` and RECORD as uv, pip and setuptools write them and on wrong and boundary inputs,
on the paths a RECORD lists, on digests, on requirement graphs, on origins and on the readings
that decide a distribution's key and its refusals, and on the install forms that decide whether a
file linked into a site directory is library code; the resolution of a RECORD path, on a tree of
symbolic links this test makes; the walk of a reached package's directory, on a tree of
files this test makes; and `fingerprint.is_library_file`, on site directories this test lays out
by hand as uv and flit lay them out (`tests.installed_dists`), reached through a link to a
package's directory and through a path entry that is a link. No test starts a process.

The CLI tests of the same decisions are in `test_installed_distributions.py`, and
`distribution_keys` over real installs is in `test_installed_distributions_snapshot.py`.
"""

import errno
import itertools
import os
import pathlib
import sys

import pytest
import pytest_mock
import xxhash
from tests import installed_dists

from pivot import fingerprint, installed_distributions

_FORM = installed_distributions.InstallForm
_SITE = "/home/user/venv/lib/python3.13/site-packages"
# uv's cache of unpacked wheels, into which `--link-mode symlink` links each installed file.
_CACHE = "/home/user/.cache/uv/archive-v0"

# `direct_url.json` (PEP 610) as uv and pip write it for each install form, in their own key order
# and spacing (uv's for a local wheel holds no hash).
_UV_DIRECTORY_URL = b'{"url":"file:///home/user/src/fixlib","dir_info":{}}'
_PIP_DIRECTORY_URL = b'{"dir_info": {}, "url": "file:///home/user/src/fixlib"}'
_UV_EDITABLE_URL = b'{"url":"file:///home/user/src/fixlib","dir_info":{"editable":true}}'
_UV_GIT_URL = (
    b'{"url":"file:///home/user/src/fixgit","vcs_info":{"vcs":"git",'
    b'"commit_id":"6964203a125157369c715bd7ba6b130944b8329e",'
    b'"requested_revision":"6964203a125157369c715bd7ba6b130944b8329e"}}'
)
_UV_WHEEL_URL = (
    b'{"url":"file:///home/user/wheels/fixwheel-0.1-py3-none-any.whl","archive_info":{}}'
)
_PIP_WHEEL_HASH = b"935cd5a593c2ac5031e455ed1aa1cdbcd43b2bb26275a8c361e59578b4203e3c"
_PIP_WHEEL_URL = (
    b'{"archive_info": {"hash": "sha256='
    + _PIP_WHEEL_HASH
    + b'", "hashes": {"sha256": "'
    + _PIP_WHEEL_HASH
    + b'"}}, "url": "file:///home/user/wheels/fixwheel-0.1-py3-none-any.whl"}'
)

# METADATA as setuptools writes it, its `Requires-Dist` headers with an extra, a marker and a name
# that is not normalised.
_SETUPTOOLS_METADATA = (
    b"Metadata-Version: 2.4\nName: Fix_Lib\nVersion: 0.1\n"
    b'Requires-Dist: fixdep[cli]>=1.0; python_version < "3"\n'
    b"Requires-Dist: Other.Name__x\n"
)
# A header continued over lines, one of them only white space, as loky's `License` header is,
# before the `Requires-Dist` headers, and a description after the headers that spells one more.
_CONTINUED_METADATA = (
    b"Metadata-Version: 2.4\nName: fix-lib\nVersion: 0.1\nLicense: BSD 3-Clause License\n"
    b"        \n        Copyright (c) 2017.\n        All rights reserved.\n"
    b"Requires-Dist: cloudpickle\nRequires-Dist: fixdep ; extra == 'test'\n"
    b"\nRequires-Dist: in-the-description\n"
)
_CRLF_METADATA = (
    b"Metadata-Version: 2.1\r\nName: fix-lib\r\nRequires-Dist: fixdep (>=1.0)\r\n"
    b"\r\nRequires-Dist: in-the-description\r\n"
)
# `requires.txt` as setuptools 80 writes it into an `.egg-info` directory for `dependencies =
# ["fixdep>=1.0", "Other.Name__x", 'legacy; python_version < "3"']` and the extras `cli =
# ["click"]` and `win = ['pywin32; sys_platform == "win32"']`: a section for a marker, one for each
# extra, and one for an extra with a marker.
_SETUPTOOLS_REQUIRES_TXT = (
    b'fixdep>=1.0\nOther.Name__x\n\n[:python_version < "3"]\nlegacy\n\n[cli]\nclick\n\n[win]\n\n'
    b'[win:sys_platform == "win32"]\npywin32\n'
)

# RECORD as uv writes it, with a console script, and as pip writes it: CR LF line ends, and
# bytecode with no hash.
_UV_RECORD = (
    b"../../../bin/fix-lib,sha256=v-5vszzj0FJZ29t3YnJ4yzhoyL3Na1GiXDBymD6G_K8,331\n"
    b"fix_lib-0.1.dist-info/RECORD,,\n"
    b"fixlib/__init__.py,sha256=Crrh4K5yghbuRJk8Wjp1X4scOH2Uf8TE9yyrDkqEIUs,6\n"
)
_PIP_RECORD = (
    b"fix_lib-0.1.dist-info/INSTALLER,sha256=zuuue4knoyJ-UwPPXg8fezS7VCrXJQrAP7zeNuwvFQg,4\r\n"
    b"fix_lib-0.1.dist-info/RECORD,,\r\n"
    b"fixlib/__init__.py,sha256=Crrh4K5yghbuRJk8Wjp1X4scOH2Uf8TE9yyrDkqEIUs,6\r\n"
    b"fixlib/__pycache__/__init__.cpython-313.pyc,,\r\n"
)

# The refusal `decide` gives the distribution `_helper_installed` describes.
_DECIDE_REFUSAL = (
    f"Distribution 'fix-lib' ({_SITE}/fix_lib-0.1.dist-info), which {{how}}, {{reason}}. Pivot "
    "cannot fingerprint the installed code the stage runs, so the stage could skip a run after "
    "that code changes. Fix: reinstall 'fix-lib'."
)
_REACHES = "this stage reaches"
_DEPENDED_ON = "a distribution this stage reaches depends on"
_FIXLIB_RECORD = installed_distributions.Record(("fixlib/__init__.py",), "")
_NO_RECORD = installed_distributions.Record(None, "has no RECORD file")
_NO_METADATA = installed_distributions.Requirements(frozenset(), "has no METADATA file")
_NO_REQUIREMENT = installed_distributions.Requirements(frozenset(), "")
# A sourceless package as its wheel holds it, bytecode in place of source, with the bytecode an
# installer compiled from it.
_SOURCELESS_PACKAGE = ("pkg/__init__.pyc", "pkg/mod.pyc", "pkg/__pycache__/mod.cpython-313.pyc")


def _helper_installed(
    form: installed_distributions.InstallForm, reason: str = "", site_dir: str = _SITE
) -> installed_distributions.Installed:
    """`fix-lib` in `site_dir`, installed in `form` (`reason`: why it is unclassified)."""
    return installed_distributions.Installed(
        dist_info=f"{site_dir}/fix_lib-0.1.dist-info",
        site_dir=site_dir,
        name="fix-lib",
        classification=installed_distributions.Classification(form, reason),
    )


def _helper_mismatches(
    paths: list[str], found: list[str], expected: list[str]
) -> list[tuple[str, str, str]]:
    """Each of `paths` whose resolution in `found` is not the one in `expected`, with both."""
    return [
        (path, got, want)
        for path, got, want in zip(paths, found, expected, strict=True)
        if got != want
    ]


def _helper_record_path(site: pathlib.Path, length: int, top: str = "fixlib") -> str:
    """A RECORD path under `top/` whose path in `site` is `length` bytes long: directories of
    one letter, then a file name of one or two letters."""
    rest = length - len(os.fsencode(f"{site}/{top}/"))
    directories = (rest - 1) // 2
    return f"{top}/" + "d/" * directories + "m" * (rest - 2 * directories)


@pytest.mark.parametrize(
    ("direct_url", "form", "reason"),
    [
        pytest.param(None, _FORM.INDEX, "", id="none"),
        pytest.param(_UV_DIRECTORY_URL, _FORM.LOCAL_DIRECTORY, "", id="uv-directory"),
        pytest.param(_PIP_DIRECTORY_URL, _FORM.LOCAL_DIRECTORY, "", id="pip-directory"),
        pytest.param(
            _UV_DIRECTORY_URL.replace(b"{}", b'{"editable":false}'),
            _FORM.LOCAL_DIRECTORY,
            "",
            id="editable-false",
        ),
        pytest.param(_UV_EDITABLE_URL, _FORM.EDITABLE, "", id="editable-true"),
        pytest.param(_UV_GIT_URL, _FORM.VCS, "", id="uv-git"),
        pytest.param(_UV_WHEEL_URL, _FORM.LOCAL_ARCHIVE, "", id="uv-local-wheel"),
        pytest.param(_PIP_WHEEL_URL, _FORM.LOCAL_ARCHIVE, "", id="pip-local-wheel"),
        pytest.param(
            _PIP_WHEEL_URL.replace(b"file:///home/user/wheels", b"https://files.example.org"),
            _FORM.REMOTE_ARCHIVE,
            "",
            id="https-archive",
        ),
        pytest.param(
            b"[]", _FORM.UNCLASSIFIED, "its direct_url.json is not a JSON object", id="array"
        ),
        pytest.param(
            _UV_DIRECTORY_URL[:-1],
            _FORM.UNCLASSIFIED,
            "its direct_url.json is not JSON (Expecting ',' delimiter: line 1 column 52 (char 51))",
            id="truncated",
        ),
        pytest.param(
            _UV_DIRECTORY_URL.replace(b"fixlib", b"caf\xe9"),
            _FORM.UNCLASSIFIED,
            "its direct_url.json is not JSON ('utf-8' codec can't decode byte 0xe9 in position "
            + "33: invalid continuation byte)",
            id="not-utf-8",
        ),
        pytest.param(
            b'{"url":"file:///home/user/src/fixlib"}',
            _FORM.UNCLASSIFIED,
            "its direct_url.json names 0 of vcs_info, dir_info and archive_info, where PEP 610 "
            + "requires exactly one",
            id="no-form",
        ),
        pytest.param(
            _UV_GIT_URL.replace(b'"vcs_info"', b'"dir_info":{},"vcs_info"'),
            _FORM.UNCLASSIFIED,
            "its direct_url.json names 2 of vcs_info, dir_info and archive_info, where PEP 610 "
            + "requires exactly one",
            id="two-forms",
        ),
        pytest.param(
            _UV_DIRECTORY_URL.replace(b"{}", b"true"),
            _FORM.UNCLASSIFIED,
            "its direct_url.json's dir_info is not a JSON object",
            id="dir-info-not-an-object",
        ),
        pytest.param(
            b'{"archive_info":{}}',
            _FORM.UNCLASSIFIED,
            "its direct_url.json has archive_info and no url",
            id="archive-without-url",
        ),
        pytest.param(
            _UV_WHEEL_URL.replace(b"file:///home/user/wheels", b"https://[::1"),
            _FORM.UNCLASSIFIED,
            "its direct_url.json's url cannot be parsed (Invalid IPv6 URL)",
            id="unparseable-url",
        ),
        pytest.param(
            b"[" * 100_000 + b"]" * 100_000,
            _FORM.UNCLASSIFIED,
            "its direct_url.json is nested too deeply to parse (maximum recursion depth exceeded "
            + "while decoding a JSON array from a unicode string)",
            id="nested-deeper-than-the-parser-recurses",
        ),
        pytest.param(
            PermissionError(errno.EACCES, "Permission denied", f"{_SITE}/direct_url.json"),
            _FORM.UNCLASSIFIED,
            "its direct_url.json cannot be read ([Errno 13] Permission denied: "
            + f"'{_SITE}/direct_url.json')",
            id="unreadable",
        ),
    ],
)
def test_classify_gives_each_direct_url_json_its_install_form(
    direct_url: bytes | OSError | None, form: installed_distributions.InstallForm, reason: str
) -> None:
    """Each form uv and pip write gets its form: no file is the index form, `dir_info` is a local
    directory unless `editable` is true, `vcs_info` is version control, and `archive_info` is a
    local archive for a `file:` URL and a remote one for another. A file that cannot be read, is
    not JSON or not a JSON object, names no form or two, has a `dir_info` that is not an object,
    an archive with no URL or a URL that cannot be parsed is unclassified, with that reason."""
    classification = installed_distributions.classify(direct_url)

    assert classification == installed_distributions.Classification(form, reason), classification


@pytest.mark.parametrize(
    ("metadata", "names", "problem"),
    [
        pytest.param(
            _SETUPTOOLS_METADATA, {"fixdep", "other-name-x"}, "", id="extras-markers-unnormalised"
        ),
        pytest.param(
            _CONTINUED_METADATA, {"cloudpickle", "fixdep"}, "", id="after-a-continued-header"
        ),
        pytest.param(_CRLF_METADATA, {"fixdep"}, "", id="crlf"),
        pytest.param(
            b"Metadata-Version: 2.1\rName: fix-lib\rRequires-Dist: fixdep (>=1.0)\r"
            + b"\rRequires-Dist: in-the-description\r",
            {"fixdep"},
            "",
            id="lone-cr",
        ),
        pytest.param(
            b"Metadata-Version: 2.4\nName: fix-lib\nRequires-Dist: fixdep\n\t(>=1.0) otherdep\n",
            {"fixdep"},
            "",
            id="continued-requires-dist",
        ),
        pytest.param(
            b"Metadata-Version: 2.1\r\nName: fix-lib\r\nRequires-Dist: fixdep\r\n (>=1.0)\r\n"
            + b"Requires-Dist: otherdep\r\n",
            {"fixdep", "otherdep"},
            "",
            id="continued-requires-dist-crlf",
        ),
        pytest.param(
            b"Metadata-Version: 2.4\nName: fix-lib\nRequires-Dist:\n  fixdep\n",
            {"fixdep"},
            "",
            id="name-on-the-continuation-line",
        ),
        pytest.param(
            b"metadata-version: 2.4\nname: fix-lib\nrequires-dist: fixdep\n",
            {"fixdep"},
            "",
            id="lower-case-names",
        ),
        pytest.param(
            b"Metadata-Version: 2.4\nName: fix-lib\n\nRequires-Dist: fixdep\n",
            set[str](),
            "",
            id="requires-dist-in-the-description",
        ),
        pytest.param(
            b"Metadata-Version: 2.4\nName: fix-lib\nSummary: one\x0ctwo\nRequires-Dist: fixdep\n",
            {"fixdep"},
            "",
            id="form-feed-in-a-value",
        ),
        pytest.param(
            b"Metadata-Version: 2.4\nName: fixdep\nVersion: 0.1\n", set[str](), "", id="none"
        ),
        pytest.param(
            b"Name: fix-lib\nVersion: 0.1\nRequires-Dist: fixdep\n",
            set[str](),
            "has a METADATA file with no Metadata-Version header",
            id="no-metadata-version",
        ),
        pytest.param(
            b"Metadata-Version: 2.4\nName: fix-lib\nnot a header\nRequires-Dist: fixdep\n",
            set[str](),
            "has a METADATA file whose headers Pivot cannot read (a line that is neither a "
            + "header nor the continuation of one: 'not a header')",
            id="non-header-line",
        ),
        pytest.param(
            b" Metadata-Version: 2.4\nName: fix-lib\nRequires-Dist: fixdep\n",
            set[str](),
            "has a METADATA file whose headers Pivot cannot read (a line that is neither a "
            + "header nor the continuation of one: ' Metadata-Version: 2.4')",
            id="first-line-a-continuation",
        ),
        pytest.param(
            b'Metadata-Version: 2.4\nName: fix-lib\nRequires-Dist: ; python_version < "3"\n',
            set[str](),
            "has a Requires-Dist header that names no distribution ('; python_version < \"3\"')",
            id="no-name",
        ),
        pytest.param(
            b"Metadata-Version: 2.4\nName: fix-lib\nAuthor: Jos\xe9 Doe\n",
            set[str](),
            "has a METADATA file that is not UTF-8 ('utf-8' codec can't decode byte 0xe9 in "
            + "position 47: invalid continuation byte)",
            id="not-utf-8",
        ),
        pytest.param(None, set[str](), "has no METADATA file", id="no-file"),
        pytest.param(
            PermissionError(errno.EACCES, "Permission denied", f"{_SITE}/METADATA"),
            set[str](),
            "has a METADATA file Pivot cannot read ([Errno 13] Permission denied: "
            + f"'{_SITE}/METADATA')",
            id="unreadable",
        ),
    ],
)
def test_requirement_names_reads_the_leading_name_of_each_requires_dist_header(
    metadata: bytes | OSError | None, names: set[str], problem: str
) -> None:
    """Each `Requires-Dist` header, its name in any case, gives its leading name, PEP 503
    normalised, with no extra or marker evaluated: after a continued header, from the first line
    of a continued one (or the next, when the first has none), with CR LF or lone CR line ends,
    and with a form feed inside a value, which ends no line; a description after the headers is
    not read. A file that cannot be read, is not UTF-8, has a line in its headers that is neither
    a header nor the continuation of one (a first line that starts with a space included), has no
    `Metadata-Version` or a `Requires-Dist` header that names nothing gives no names and that
    problem."""
    requirements = installed_distributions.requirement_names(metadata)

    assert requirements == installed_distributions.Requirements(frozenset(names), problem), (
        requirements
    )


@pytest.mark.parametrize(
    ("requires", "names", "problem"),
    [
        pytest.param(
            _SETUPTOOLS_REQUIRES_TXT,
            {"fixdep", "other-name-x", "click", "legacy", "pywin32"},
            "",
            id="setuptools-sections",
        ),
        pytest.param(
            b"# pinned by hand\r\n\r\n   fixdep (>=1.0)  \r\n", {"fixdep"}, "", id="comment-crlf"
        ),
        pytest.param(b"", set[str](), "", id="empty"),
        pytest.param(None, set[str](), "", id="no-file"),
        pytest.param(
            b"fixdep\n[cli\nclick\n",
            set[str](),
            "has a requires.txt line that names no distribution ('[cli')",
            id="unclosed-section-header",
        ),
        pytest.param(
            b"fixdep\ncaf\xe9\n",
            set[str](),
            "has a requires.txt file that is not UTF-8 ('utf-8' codec can't decode byte 0xe9 in "
            + "position 10: invalid continuation byte)",
            id="not-utf-8",
        ),
        pytest.param(
            PermissionError(errno.EACCES, "Permission denied", f"{_SITE}/requires.txt"),
            set[str](),
            "has a requires.txt file Pivot cannot read ([Errno 13] Permission denied: "
            + f"'{_SITE}/requires.txt')",
            id="unreadable",
        ),
    ],
)
def test_requires_txt_names_reads_the_leading_name_of_each_requirement_line(
    requires: bytes | OSError | None, names: set[str], problem: str
) -> None:
    """Each requirement line of an `.egg-info` directory's `requires.txt` gives its leading name,
    PEP 503 normalised, those under an `[extra]`, `[:marker]` or `[extra:marker]` section header
    too, with no extra or marker evaluated; a blank line, a comment and a section header give
    none, whatever the line ends. No file, or an empty one, requires nothing. A file that cannot
    be read, is not UTF-8 or has a line that names nothing gives no names and that problem."""
    requirements = installed_distributions.requires_txt_names(requires)

    assert requirements == installed_distributions.Requirements(frozenset(names), problem), (
        requirements
    )


@pytest.mark.parametrize(
    ("record", "paths", "problem"),
    [
        pytest.param(
            _UV_RECORD,
            ("../../../bin/fix-lib", "fix_lib-0.1.dist-info/RECORD", "fixlib/__init__.py"),
            "",
            id="uv",
        ),
        pytest.param(
            _PIP_RECORD,
            (
                "fix_lib-0.1.dist-info/INSTALLER",
                "fix_lib-0.1.dist-info/RECORD",
                "fixlib/__init__.py",
                "fixlib/__pycache__/__init__.cpython-313.pyc",
            ),
            "",
            id="pip-crlf",
        ),
        pytest.param(
            b'"fixlib/a,b.py",sha256=abc,3\n\n,sha256=abc,3\nfixlib/__init__.py,,\n',
            ("fixlib/a,b.py", "fixlib/__init__.py"),
            "",
            id="quoted-comma-and-empty-rows",
        ),
        pytest.param(
            b"fixlib/caf\xe9.py,,\n",
            None,
            "has a RECORD file Pivot cannot read ('utf-8' codec can't decode byte 0xe9 in "
            + "position 10: invalid continuation byte)",
            id="not-utf-8",
        ),
        pytest.param(
            b"a" * 131073,
            None,
            "has a RECORD file Pivot cannot read (field larger than field limit (131072))",
            id="field-over-the-csv-limit",
        ),
        pytest.param(None, None, "has no RECORD file", id="no-file"),
        pytest.param(
            PermissionError(errno.EACCES, "Permission denied", f"{_SITE}/RECORD"),
            None,
            f"has a RECORD file Pivot cannot read ([Errno 13] Permission denied: '{_SITE}/RECORD')",
            id="unreadable",
        ),
    ],
)
def test_record_paths_lists_the_first_field_of_each_row(
    record: bytes | OSError | None, paths: tuple[str, ...] | None, problem: str
) -> None:
    """Each row gives its first field, a quoted one with its comma, whatever the line ends; an
    empty line or an empty first field gives none. A file that cannot be read, is not UTF-8 or
    that `csv` cannot read gives no paths and that problem."""
    found = installed_distributions.record_paths(record)

    assert found == installed_distributions.Record(paths, problem), found


def test_hashed_files_keep_each_installed_file_once_but_the_installers_and_pycache() -> None:
    """Of a RECORD's paths, the key hashes each relative one that stays in the site directory,
    once, under its normalised name, and the distribution's own `entry_points.txt` under one name
    whatever its version: not an absolute path, a path that leaves the directory, the console
    script, the other files of a `.dist-info` directory, another distribution's or a nested
    `.dist-info` directory, nor a path under `__pycache__`, which the installer compiles. A `.pyc`
    file outside `__pycache__` is kept, and so is a directory entry, to hash as a file that cannot
    be read."""
    paths = [
        "../../../bin/fix-lib",
        "/usr/lib/python3.13/site-packages/fixlib/__init__.py",
        "fixlib/../../outside.py",
        "fixlib/..",
        "./fixlib/__init__.py",
        "fixlib/__init__.py",
        "fixlib/__init__.py",
        "fixlib/",
        "fixlib/data/table.csv",
        "fixlib/__pycache__/__init__.cpython-313.pyc",
        "fixlib/legacy.pyc",
        "fixlib/__pycache__/notes.txt",
        "fix_lib-0.1.dist-info/entry_points.txt",
        "fix_lib-0.1.dist-info/METADATA",
        "fix_lib-0.1.dist-info/RECORD",
        "other-1.0.dist-info/entry_points.txt",
        "fixlib/_vendor/inner-2.0.dist-info/METADATA",
        "fixlib/_vendor/INNER-2.0.DIST-INFO/entry_points.txt",
        "fixlib/_vendor/inner.py",
    ]

    files = installed_distributions.hashed_files(paths, "fix_lib-0.1.dist-info")

    assert files == [
        (".dist-info/entry_points.txt", "fix_lib-0.1.dist-info/entry_points.txt"),
        ("fixlib", "fixlib"),
        ("fixlib/__init__.py", "fixlib/__init__.py"),
        ("fixlib/_vendor/inner.py", "fixlib/_vendor/inner.py"),
        ("fixlib/data/table.csv", "fixlib/data/table.csv"),
        ("fixlib/legacy.pyc", "fixlib/legacy.pyc"),
    ], files


def test_the_key_and_the_walk_keep_a_sourceless_packages_pyc_files_but_pycache(
    tmp_path: pathlib.Path,
) -> None:
    """A sourceless package: its wheel holds `pkg/__init__.pyc` and `pkg/mod.pyc` in place of
    source, and the installer compiled `pkg/__pycache__/mod.cpython-313.pyc`. Of these RECORD
    paths the key hashes exactly the two `.pyc` files outside `__pycache__`, and the walk of the
    reached package's directory (`_files_under`), over those files on disk, gives the same two,
    each as the walk lists it and resolved: the check of a file that no RECORD lists reads the
    files the key would hash."""
    for path in _SOURCELESS_PACKAGE:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(b"")

    files = installed_distributions.hashed_files(_SOURCELESS_PACKAGE, "pkg-0.1.dist-info")
    walked = sorted(installed_distributions._files_under(str(tmp_path / "pkg"), dict[str, str]()))

    kept = ["pkg/__init__.pyc", "pkg/mod.pyc"]
    assert files == [(path, path) for path in kept], files
    assert walked == [(str(tmp_path / path), os.path.realpath(tmp_path / path)) for path in kept], (
        walked
    )


def test_content_digest_and_key_value_hash_json_arrays_in_name_order() -> None:
    """A content digest is the xxh64 of the JSON array of `[name, digest]` pairs in name order,
    null for a file that cannot be read, which no digest equals; a key's value is the xxh64 of the
    JSON array of its installs' digests, sorted. JSON keeps a name apart from its digest."""
    first = xxhash.xxh64(b"A = 1\n").hexdigest()
    second = xxhash.xxh64(b"B = 1\n").hexdigest()

    digest = installed_distributions.content_digest(
        [("fixlib/b.py", second), ("fixlib/a.py", first)]
    )
    missing = installed_distributions.content_digest([("fixlib/a.py", None)])
    others = {
        installed_distributions.content_digest([("fixlib/a.py", text)])
        for text in ("null", "", first)
    }
    value = installed_distributions.key_value([second, first])

    stream = f'[["fixlib/a.py","{first}"],["fixlib/b.py","{second}"]]'
    assert digest == xxhash.xxh64(stream.encode()).hexdigest(), digest
    assert missing == xxhash.xxh64(b'[["fixlib/a.py",null]]').hexdigest(), missing
    assert missing not in others and len(others) == 3, (missing, others)
    assert installed_distributions.content_digest([("ab", "c")]) != (
        installed_distributions.content_digest([("a", "bc")])
    ), "a name and its digest must stay apart"
    ordered = sorted([first, second])
    assert value == xxhash.xxh64(f'["{ordered[0]}","{ordered[1]}"]'.encode()).hexdigest(), value


def test_candidates_are_the_seeds_and_every_distribution_that_requires_one() -> None:
    """Each distribution that requires a seed, directly or through others, is a candidate, through
    a cycle too; one that only a candidate requires is not."""
    requires = {
        "fix-lib": {"fix-egg", "requests"},
        "fix-egg": {"fixdep"},
        "fixdep": set[str](),
        "requests": {"urllib3"},
        "urllib3": set[str](),
        "cycle-a": {"cycle-b"},
        "cycle-b": {"cycle-a", "fixdep"},
    }

    found = installed_distributions.candidates(["fixdep"], requires)

    assert found == {"fixdep", "fix-egg", "fix-lib", "cycle-a", "cycle-b"}, found


def test_reach_meets_each_requirement_once_and_follows_none_of_a_stop() -> None:
    """The closure meets each distribution once, in the order met, so a cycle ends; a stop
    (pivot's own distribution) is met, but its requirements are not followed."""
    requires = {
        "fix-lib": ("fix-egg",),
        "fix-egg": ("fixdep", "fix-lib"),
        "fixdep": (),
        "pivot": ("pivot-only",),
        "pivot-only": (),
    }

    assert installed_distributions.reach(["fix-lib"], requires, ()) == [
        "fix-lib",
        "fix-egg",
        "fixdep",
    ]
    assert installed_distributions.reach(["fix-lib"], requires, ("fix-egg",)) == [
        "fix-lib",
        "fix-egg",
    ]
    assert installed_distributions.reach(["pivot", "fix-lib"], requires, ("pivot",)) == [
        "pivot",
        "fix-lib",
        "fix-egg",
        "fixdep",
    ]


@pytest.mark.parametrize(
    ("origin", "real", "expected"),
    [
        pytest.param(
            ("file", f"{_SITE}/fixlib/core.py"),
            f"{_SITE}/fixlib/core.py",
            {"fix-lib", "fix-lib-extra"},
            id="file",
        ),
        pytest.param(
            ("file", f"{_SITE}/company/__init__.py"),
            f"{_SITE}/company/__init__.py",
            {"company-core", "company-vendor"},
            id="package-init",
        ),
        pytest.param(
            ("file", f"{_SITE}/company/__init__.cpython-313-x86_64-linux-gnu.so"),
            f"{_SITE}/company/__init__.cpython-313-x86_64-linux-gnu.so",
            {"company-core", "company-vendor"},
            id="extension-package-init",
        ),
        pytest.param(
            ("file", f"{_SITE}/stray.py"),
            f"{_SITE}/stray.py",
            set[str](),
            id="file-no-record-lists",
        ),
        pytest.param(
            ("file", f"{_CACHE}/core/linked/__init__.py"),
            f"{_SITE}/linked/__init__.py",
            {"linked-core", "linked-plugin"},
            id="linked-package-init-owned-through-its-directory-in-site-packages",
        ),
        pytest.param(
            ("file", f"{_CACHE}/core/linked/__init__.py"),
            None,
            {"linked-core"},
            id="linked-package-init-with-no-located-file-owned-through-its-resolved-directory",
        ),
        pytest.param(
            ("file", f"{_CACHE}/core/linked/core.py"),
            f"{_SITE}/linked/core.py",
            {"linked-core"},
            id="linked-file-owned-by-its-resolved-path",
        ),
        pytest.param(
            ("file", f"{_CACHE}/twice/twice.py"),
            f"{_SITE}/twice.py",
            {"twice-in-venv", "twice-in-target"},
            id="file-linked-from-two-site-directories-owned-by-both-installs",
        ),
        pytest.param(
            ("namespace", f"{_SITE}/nsco", "/home/user/other/nsco"),
            None,
            {"nsco-local", "nsco-other"},
            id="namespace",
        ),
        pytest.param(("built-in",), None, set[str](), id="built-in"),
        pytest.param(("frozen",), None, set[str](), id="frozen"),
        pytest.param(("no file",), None, set[str](), id="no-file"),
        pytest.param(("unknown", "ImportError: boom"), None, set[str](), id="unknown"),
        pytest.param(None, None, set[str](), id="none"),
    ],
)
def test_owners_maps_an_origin_to_the_candidates_by_file(
    origin: tuple[str, ...] | None, real: str | None, expected: set[str]
) -> None:
    """A file maps to every candidate whose RECORD lists it, resolved, and a package's `__init__`
    of any suffix also to every candidate with a file under its directory, the directory of the
    file as the import located it (`real`); a namespace package maps to every candidate with a
    file under one of its directories; any other origin maps to none. Where an installer links
    each file into its cache (`linked-core`'s files, and `linked-plugin`'s `linked/plugin.py`, in
    two cache entries), a package's `__init__` maps through its directory in the site directory
    to each candidate that lists a file there, and, with no located file, only through the
    directory of its resolved path, to the one candidate whose cache entry that is; a linked
    module file maps to the candidate that lists its resolved path, and one cache entry linked
    from two site directories (`twice.py`) to both installs, whose RECORDs resolve to it."""
    file_owners = {
        f"{_SITE}/fixlib/core.py": {"fix-lib", "fix-lib-extra"},
        f"{_SITE}/company/__init__.py": {"company-core"},
        f"{_SITE}/company/vendor.py": {"company-vendor"},
        f"{_SITE}/nsco/local/__init__.py": {"nsco-local"},
        "/home/user/other/nsco/other/__init__.py": {"nsco-other"},
        f"{_CACHE}/core/linked/__init__.py": {"linked-core"},
        f"{_CACHE}/core/linked/core.py": {"linked-core"},
        f"{_CACHE}/plugin/linked/plugin.py": {"linked-plugin"},
        f"{_CACHE}/twice/twice.py": {"twice-in-venv", "twice-in-target"},
    }
    directory_owners = {
        f"{_SITE}/fixlib": {"fix-lib", "fix-lib-extra"},
        f"{_SITE}/company": {"company-core", "company-vendor"},
        f"{_SITE}/nsco": {"nsco-local"},
        f"{_SITE}/nsco/local": {"nsco-local"},
        "/home/user/other/nsco": {"nsco-other"},
        "/home/user/other/nsco/other": {"nsco-other"},
        f"{_CACHE}/core/linked": {"linked-core"},
        f"{_CACHE}/plugin/linked": {"linked-plugin"},
        f"{_SITE}/linked": {"linked-core", "linked-plugin"},
    }

    found = installed_distributions.owners(origin, real, file_owners, directory_owners)

    assert found == expected, found


@pytest.mark.parametrize(
    ("installed", "record", "requirements", "pivot_own", "owner", "decision"),
    [
        pytest.param(
            _helper_installed(_FORM.UNCLASSIFIED, "its direct_url.json is not a JSON object"),
            _NO_RECORD,
            _NO_METADATA,
            True,
            True,
            installed_distributions.Decision(keyed=False, refusal=""),
            id="pivots-own-first",
        ),
        pytest.param(
            _helper_installed(_FORM.UNCLASSIFIED, "its direct_url.json is not a JSON object"),
            _FIXLIB_RECORD,
            _NO_METADATA,
            False,
            True,
            installed_distributions.Decision(
                keyed=False,
                refusal=_DECIDE_REFUSAL.format(
                    how=_REACHES,
                    reason="is installed in a form Pivot cannot classify: its direct_url.json "
                    + "is not a JSON object",
                ),
            ),
            id="unclassified-before-requirements",
        ),
        pytest.param(
            _helper_installed(_FORM.VCS),
            _NO_RECORD,
            _NO_METADATA,
            False,
            False,
            installed_distributions.Decision(
                keyed=False,
                refusal=_DECIDE_REFUSAL.format(
                    how=_DEPENDED_ON,
                    reason="is installed from version control (vcs_info) and has no RECORD file, "
                    + "so Pivot cannot read its files",
                ),
            ),
            id="keyed-without-record-before-requirements",
        ),
        pytest.param(
            _helper_installed(_FORM.LOCAL_ARCHIVE),
            _FIXLIB_RECORD,
            _NO_METADATA,
            False,
            True,
            installed_distributions.Decision(
                keyed=True,
                refusal=_DECIDE_REFUSAL.format(
                    how=_REACHES,
                    reason="has no METADATA file, so Pivot cannot tell what it depends on",
                ),
            ),
            id="keyed-requirements-unreadable",
        ),
        pytest.param(
            _helper_installed(_FORM.INDEX),
            _NO_RECORD,
            _NO_METADATA,
            False,
            False,
            installed_distributions.Decision(
                keyed=False,
                refusal=_DECIDE_REFUSAL.format(
                    how=_DEPENDED_ON,
                    reason="has no METADATA file, so Pivot cannot tell what it depends on",
                ),
            ),
            id="index-requirements-unreadable",
        ),
        pytest.param(
            _helper_installed(_FORM.LOCAL_DIRECTORY),
            _FIXLIB_RECORD,
            installed_distributions.Requirements(frozenset({"fixdep"}), ""),
            False,
            True,
            installed_distributions.Decision(keyed=True, refusal=""),
            id="keyed",
        ),
        pytest.param(
            _helper_installed(_FORM.INDEX),
            _NO_RECORD,
            _NO_REQUIREMENT,
            False,
            False,
            installed_distributions.Decision(keyed=False, refusal=""),
            id="index-without-record",
        ),
        pytest.param(
            _helper_installed(_FORM.EDITABLE),
            _FIXLIB_RECORD,
            _NO_REQUIREMENT,
            False,
            True,
            installed_distributions.Decision(keyed=False, refusal=""),
            id="editable",
        ),
        pytest.param(
            _helper_installed(_FORM.REMOTE_ARCHIVE),
            _FIXLIB_RECORD,
            _NO_REQUIREMENT,
            False,
            True,
            installed_distributions.Decision(keyed=False, refusal=""),
            id="remote-archive",
        ),
    ],
)
def test_decide_gives_a_key_and_refusals_in_their_order(
    installed: installed_distributions.Installed,
    record: installed_distributions.Record,
    requirements: installed_distributions.Requirements,
    pivot_own: bool,
    owner: bool,
    decision: installed_distributions.Decision,
) -> None:
    """Pivot's own distribution gives nothing, whatever its defects; then an unclassified one is
    refused, then one of a keyed form with no RECORD, then one whose requirements cannot be read,
    each refusal naming the distribution, its `.dist-info` directory, how the stage meets it, the
    reason and the fix. A distribution of a keyed form with a RECORD gets a key, refused or not;
    an index, editable or remote-archive one gets none."""
    found = installed_distributions.decide(
        installed, record, requirements, pivot_own=pivot_own, owner=owner
    )

    assert found == decision, found


@pytest.mark.parametrize(
    ("forms", "library"),
    [
        pytest.param([], False, id="no-distribution"),
        pytest.param([_FORM.INDEX], True, id="index"),
        pytest.param([_FORM.VCS], True, id="vcs"),
        pytest.param([_FORM.LOCAL_DIRECTORY], True, id="local-directory"),
        pytest.param([_FORM.LOCAL_ARCHIVE], True, id="local-archive"),
        pytest.param([_FORM.REMOTE_ARCHIVE], True, id="remote-archive"),
        pytest.param([_FORM.EDITABLE], False, id="editable"),
        pytest.param([_FORM.UNCLASSIFIED], False, id="unclassified"),
        pytest.param([_FORM.INDEX, _FORM.VCS], True, id="two-not-editable"),
        pytest.param([_FORM.INDEX, _FORM.EDITABLE], False, id="one-editable-of-two"),
        pytest.param([_FORM.UNCLASSIFIED, _FORM.LOCAL_DIRECTORY], False, id="one-unclassified"),
    ],
)
def test_owned_by_libraries_takes_a_linked_file_listed_only_by_known_forms_not_editable(
    forms: list[installed_distributions.InstallForm], library: bool
) -> None:
    """A file linked into a site directory is library code by the RECORDs that list it when at
    least one distribution lists it and each one that does is installed from an index, version
    control, a local directory, a local archive or a remote archive; one editable or unclassified
    distribution among them, or none, leaves the file to its resolved path."""
    found = installed_distributions.owned_by_libraries(forms)

    assert found is library, f"forms {forms} must give {library}, not {found}"


def test_a_record_path_resolves_to_what_realpath_gives(tmp_path: pathlib.Path) -> None:
    """A RECORD path is resolved a name at a time from the real path of the nearest directory
    above it already resolved, memoised, and gives what `os.path.realpath` gives, from an empty
    memo, from the memo it fills and from that memo filled: under a package directory that is a
    symbolic link, for a file that is a symbolic link, by an absolute target or by a relative one
    through `..`, a dangling link, a link to itself, a loop of two links crossed by `..`
    (`a -> b`, `b -> a`, `a/../b/z.py`), a missing directory, a name under a regular file, a
    `..`, a `.`, a trailing slash, doubled slashes, a console script's `../../../bin` path and the
    root; and every path of one to three names under the site directory, each name one of its
    links (`fixlib` to a directory, `core.py` to a file, `dangling.py`, `loop.py` to itself, `a`,
    `b`, `into`, which points into the loop, and `through`, which crosses the loop and leaves it
    by `..`), `..`, `.` or an empty name. After `through`, `realpath` resolves `b` by what it
    learnt of the loop, as after `a/..`, so the real path of a directory that is a link must be
    `realpath` of the directory's own path, not of its name joined to the real path of the
    directory above."""
    package = tmp_path / "elsewhere" / "fixlib"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    target = tmp_path / "other" / "core.py"
    target.parent.mkdir()
    target.write_text("")
    (package / "core.py").symlink_to(target)
    (package / "relative.py").symlink_to(pathlib.Path("..", "..", "other", "core.py"))
    site = tmp_path / "site"
    site.mkdir()
    (site / "fixlib").symlink_to(package, target_is_directory=True)
    (site / "plain.py").write_text("")
    (site / "core.py").symlink_to(pathlib.Path("..", "other", "core.py"))
    (site / "dangling.py").symlink_to(tmp_path / "missing.py")
    (site / "loop.py").symlink_to("loop.py")
    (site / "a").symlink_to("b")
    (site / "b").symlink_to("a")
    (site / "into").symlink_to("a")
    (site / "through").symlink_to(pathlib.Path("a", os.pardir))
    named = [
        f"{site}/fixlib/__init__.py",
        f"{site}/fixlib/core.py",
        f"{site}/fixlib/relative.py",
        f"{site}/dangling.py",
        f"{site}/loop.py",
        f"{site}/missing/module.py",
        f"{site}/plain.py/under-a-file",
        f"{site}/a/../b/z.py",
        f"{site}/b/../a/z.py",
        f"{site}/fixlib/../plain.py",
        f"{site}/./plain.py",
        f"{site}/fixlib/",
        f"{site}//fixlib//core.py",
        f"{site}/../../../bin/fix-lib",
        "/",
    ]
    links = ["fixlib", "core.py", "dangling.py", "loop.py", "a", "b", "into", "through"]
    names = [*links, os.pardir, os.curdir, ""]
    paths = named + [
        "/".join([str(site), *parts])
        for count in (1, 2, 3)
        for parts in itertools.product(names, repeat=count)
    ]
    real_dirs = dict[str, str]()

    fresh = [installed_distributions._real_path(path, dict[str, str]()) for path in paths]
    first = [installed_distributions._real_path(path, real_dirs) for path in paths]
    again = [installed_distributions._real_path(path, real_dirs) for path in paths]

    expected = [os.path.realpath(path) for path in paths]
    assert expected[1:3] == [str(target.resolve())] * 2, f"the links must resolve: {expected[:3]}"
    loop = expected[paths.index(f"{site}/a/../b/z.py")]
    assert loop == f"{site.resolve()}/a/z.py", f"realpath must stop the loop at its link: {loop}"
    through = expected[paths.index(f"{site}/through/b/core.py")]
    assert through == f"{site.resolve()}/a/core.py", (
        f"realpath must resolve b by the loop that through crossed: {through}"
    )
    assert _helper_mismatches(paths, fresh, expected) == [], (
        "an empty memo must give realpath's paths"
    )
    assert _helper_mismatches(paths, first, expected) == [], (
        "the memo being filled must give realpath's paths"
    )
    assert _helper_mismatches(paths, again, expected) == [], (
        "the filled memo must give realpath's paths"
    )


def test_resolved_leaves_out_each_record_path_that_names_no_file_and_resolves_a_long_one(
    tmp_path: pathlib.Path,
) -> None:
    """A RECORD path that holds a NUL character, or a name the filesystem encoding cannot encode,
    for both of which `os.path.realpath` raises a `ValueError`, names no file: `_resolved` leaves
    each out. It resolves to what `realpath` gives the installed `fixlib/__init__.py`, a path one
    byte shorter than PATH_MAX in its site directory, one PATH_MAX bytes long through `linked`, a
    link to `fixlib`, which `os.lstat` refuses as too long but which can name a file (a long name
    in a shorter directory), and one twice that long; the memo it fills holds no directory of
    PATH_MAX bytes or more. The name is a lone surrogate, which the filesystem encoding cannot
    encode under any locale (its error handler, surrogateescape, encodes only U+DC80 to U+DCFF);
    in use it is a non-ASCII name under an ASCII locale. `_file_digest`, which reads a keyed
    distribution's RECORD paths as they are, gives each left-out path, and the one PATH_MAX bytes
    long, no digest."""
    site = tmp_path.resolve() / "site"
    (site / "fixlib").mkdir(parents=True)
    (site / "fixlib" / "__init__.py").write_text("")
    (site / "linked").symlink_to("fixlib", target_is_directory=True)
    installed = _helper_installed(_FORM.LOCAL_DIRECTORY, site_dir=str(site))
    path_max = os.pathconf("/", "PC_PATH_MAX")
    too_long = _helper_record_path(site, path_max, "linked")
    longest = _helper_record_path(site, path_max - 1)
    twice = _helper_record_path(site, 2 * path_max)
    lengths = [len(os.fsencode(f"{site}/{path}")) for path in (too_long, longest, twice)]
    assert lengths == [path_max, path_max - 1, 2 * path_max], (
        f"the long paths must be built to length: {lengths}"
    )
    left_out = ["fixlib/\0.py", "fixlib/\ud800.py"]
    kept = ["fixlib/__init__.py", too_long, longest, twice]
    real_dirs = dict[str, str]()

    found = list(installed_distributions._resolved(installed, [*left_out, *kept], real_dirs))
    digests = [
        installed_distributions._file_digest(f"{site}/{path}") for path in [*left_out, too_long]
    ]

    through_link = f"{site}/fixlib/{too_long.removeprefix('linked/')}"
    assert os.path.realpath(f"{site}/{too_long}") == through_link, "realpath must follow linked"
    assert found == [os.path.realpath(f"{site}/{path}") for path in kept], found
    memoised = max(len(os.fsencode(directory)) for directory in real_dirs)
    assert memoised < path_max, f"the memo must hold no directory of PATH_MAX bytes: {memoised}"
    assert digests == [None, None, None], digests


@pytest.mark.parametrize(
    "linked",
    [
        pytest.param(False, id="copied"),
        pytest.param(True, id="linked-into-a-cache"),
    ],
)
def test_is_library_file_takes_a_library_file_reached_through_a_link_to_its_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, linked: bool
) -> None:
    """`idxlib`, laid out by hand in `site-packages` as uv installs from a package index
    (`installed_dists.index_install_by_hand`), its files copies or, as `--link-mode symlink`
    leaves them, links into a cache from real directories. `project/idxlib` is a link to its
    package's directory, as a library package linked into a project root, and `libs` a link to
    `site-packages`. `fingerprint.is_library_file` takes the module's file as library code where
    it is installed, through `project/idxlib` with `site-packages` on the path, and through
    `libs` with `libs` on the path in its place, whether the file is a copy or a link: a linked
    file is judged where its directory resolves, in `site-packages`, as a copy is by its resolved
    path. A project file beside the link stays user code."""
    site = tmp_path / "venv" / "site-packages"
    installed_dists.index_install_by_hand(
        site,
        "idxlib",
        {"idxlib/__init__.py": "LIB = 1\n"},
        cache=tmp_path / "cache" if linked else None,
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / "idxlib").symlink_to(site / "idxlib", target_is_directory=True)
    (project / "helpers.py").write_text("HELP = 1\n")
    libs = tmp_path / "libs"
    libs.symlink_to(site, target_is_directory=True)
    installed = site / "idxlib" / "__init__.py"
    assert installed.is_symlink() is linked, f"{installed} must be a link only when linked"
    assert not installed.parent.is_symlink(), f"{installed.parent} must be a real directory"

    monkeypatch.setattr(sys, "path", [str(project), str(site)])
    found = {
        "in site-packages": fingerprint.is_library_file(str(installed)),
        "through project/idxlib": fingerprint.is_library_file(
            str(project / "idxlib" / "__init__.py")
        ),
        "project file": fingerprint.is_library_file(str(project / "helpers.py")),
    }
    monkeypatch.setattr(sys, "path", [str(project), str(libs)])
    found["through libs"] = fingerprint.is_library_file(str(libs / "idxlib" / "__init__.py"))

    assert found == {
        "in site-packages": True,
        "through project/idxlib": True,
        "project file": False,
        "through libs": True,
    }, found


@pytest.mark.parametrize(
    ("link", "source", "asked"),
    [
        pytest.param("flitlib", "flitlib/__init__.py", False, id="package-directory"),
        pytest.param("flitlib.py", "flitlib.py", True, id="module-file"),
    ],
)
def test_is_library_file_leaves_flits_editable_link_user_code_through_a_linked_path_entry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
    link: str,
    source: str,
    asked: bool,
) -> None:
    """`flitlib`, laid out as flit's `install --symlink` installs it
    (`installed_dists.flit_symlink_install`): a link in `site-packages` to the checkout's package
    directory, or to its module file, which the RECORD of an editable install lists. `libs`, a
    link to `site-packages`, is the one entry of the path. `fingerprint.is_library_file` takes
    the checkout's source as user code, in `site-packages` and through `libs`. A file under the
    linked package directory is judged in the checkout, where its directory resolves, and the
    RECORDs are not asked about it; the linked module file is judged in `site-packages`, and the
    RECORDs are asked about it there, where the one that lists it is an editable install's."""
    site = tmp_path / "venv" / "site-packages"
    site.mkdir(parents=True)
    checkout = tmp_path / "flitlib"
    (checkout / source).parent.mkdir(parents=True, exist_ok=True)
    (checkout / source).write_text(installed_dists.FIXLIB_ALONE)
    installed_dists.flit_symlink_install(site, checkout, link)
    libs = tmp_path / "libs"
    libs.symlink_to(site, target_is_directory=True)
    monkeypatch.setattr(sys, "path", [str(libs)])
    listed = mocker.spy(installed_distributions, "listed_as_library")

    found = [fingerprint.is_library_file(str(entry / source)) for entry in (site, libs)]

    assert found == [False, False], f"flit's source must be user code in both: {found}"
    files_asked = [call.args[0] for call in listed.call_args_list]
    in_site = os.path.join(os.path.realpath(site), source)
    assert files_asked == ([in_site, in_site] if asked else []), files_asked
