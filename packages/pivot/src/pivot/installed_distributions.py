"""The `dist:` keys of a stage's code manifest (`fingerprint.add_distribution_keys`).

A stage reaches a distribution when a module that its manifest names by name (a `byname:` entry)
comes from a file that the distribution's RECORD lists, or is a package or a namespace package
whose directory, as the import searched it, holds one (`owners`). For each reached distribution
installed from version control, from a local directory or from a local archive, and for each such
distribution that a reached one depends on, the manifest gets the key `dist:<name>`, `<name>`
normalised as PEP 503 says. Its value hashes the distribution's own installed files, read from
disk, so a change to them makes the stage stale, and a reinstall of the same files from another
commit, path, installer or environment changes nothing. Package-index, remote-archive and
editable distributions and pivot's own get no key.

It also tells `fingerprint.is_library_file` whether a file that an installer linked into a site
directory, as uv's `--link-mode symlink` links each file into its cache, is library code by the
RECORDs there (`listed_as_library`): some list it, and each that does is installed in a known form
that is not editable (`owned_by_libraries`).

A leaf: it imports nothing of pivot's at run time, so `fingerprint` imports it and no import cycle
forms. It takes plain data (each reached module's origin and its file as the import located it,
`ReachedModule`; the worker's `sys.path`; pivot's package directories) and returns plain data (the
keys and the refusal messages). Each decision on what a distribution's files say (`classify`,
`requirement_names`, `requires_txt_names`, `record_paths`, `hashed_files`, `content_digest`,
`key_value`, `candidates`, `reach`, `owners`, `decide`, `owned_by_libraries`) is a public function
of plain data. What reads the disk is private: the scan of the path, the index of the candidates,
the digest of a keyed distribution's files, the refusal of a reached file that no RECORD lists,
which reads the RECORDs of a site directory and the files under a package, and the index of the
paths the RECORDs of a site directory list (`_Snapshot`).

What it reads, once per process for one worker path, so a reinstall while Pivot runs is seen at the
next start: the metadata directories on the path and each `direct_url.json`; when a stage's keys
are first asked for (`needs_origins`, `distribution_keys`), and only when one of those
distributions is installed from version control, a local directory or a local archive, or cannot
be classified, the requirements of every distribution (its `Requires-Dist` headers, and an
`.egg-info` directory's `requires.txt` when its metadata has none) and RECORD for the candidates
(those distributions, each distribution whose requirements cannot be read, and every distribution
that requires one of them, directly or through others); every RECORD of a site directory only when
a candidate there has no RECORD and a stage reaches a file whose site directory it is (the deepest
entry of the path that holds the file), and, apart from that, when `listed_as_library` is first
asked about a file whose site directory it is; the files under the directory of a reached
package or namespace package only when such a site directory is the deepest entry of the path
above that directory or lies at or under it, and only until each such site directory has a file
there that no RECORD lists, once for each reached module; and the files of a keyed distribution
only when a stage reaches it. Each RECORD is read once, whichever of these needs it first. A
process that asks only `listed_as_library`, as a worker's import recorder does, reads no
requirement and builds no index of the candidates. Each path a RECORD lists that can name a file
is resolved for the index of the candidates and for the refusal of a reached file that no RECORD
lists (`_resolved`), and so is each file under such a directory, to the path `os.path.realpath`
gives: a RECORD path of PATH_MAX bytes or more in its site directory by `realpath` alone, and
every other path a name at a time from the nearest directory above it already resolved (a site
directory, or one resolved for an earlier path): one `os.path.islink` for each new name, and
`realpath` only for a name that is a symbolic link, empty, `.` or `..` (`_real_path`).
`listed_as_library` resolves no RECORD path: it compares each as written, normalised, with the
file's path in its site directory.
"""

from __future__ import annotations

import csv
import dataclasses
import enum
import functools
import json
import os
import posixpath
import re
import stat
import threading
import urllib.parse
from typing import TYPE_CHECKING, NamedTuple, cast

import xxhash

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence

    from pivot import import_origin

# The prefix of every key this module makes.
_KEY_PREFIX = "dist:"

# How much of a file is read at a time to hash it.
_CHUNK_SIZE = 1024 * 1024

# The name a distribution's own `entry_points.txt` is hashed under: without the name of its
# `.dist-info` directory, which carries the version.
_OWN_ENTRY_POINTS = ".dist-info/entry_points.txt"

# A PEP 508 name at the start of a requirement.
_REQUIREMENT_NAME = re.compile(r"[A-Z0-9](?:[A-Z0-9._-]*[A-Z0-9])?", re.IGNORECASE)

# The empty line that ends a metadata file's headers.
_END_OF_HEADERS = re.compile(r"\r?\n\r?\n")

# A line end in a metadata file's headers, as `email` reads one: CR LF, CR or LF, not each
# character `str.splitlines` breaks at (`\x0b`, `\x0c`, `\x1c`, ...).
_LINE_END = re.compile(r"\r\n|\r|\n")

# A header line: its name, of the characters `email` allows in one (printable ASCII but the
# colon), and its value after the colon and any spaces and tabs.
_HEADER = re.compile(r"([\x21-\x39\x3b-\x7e]+):[ \t]*(.*)")

# The names in a path that `os.path.realpath` reads specially: an empty one, `.` and `..`.
_SPECIAL_NAMES = ("", os.curdir, os.pardir)

# PATH_MAX: `os.lstat` refuses a path of this many bytes or more as too long (ENAMETOOLONG), as the
# limit counts the NUL that ends a path in C (4096 on Linux, which looks up a path of 4095 bytes).
_PATH_MAX = os.pathconf("/", "PC_PATH_MAX")


class InstallForm(enum.Enum):
    """How a distribution was installed, read from its `direct_url.json` (PEP 610). Each value
    completes the phrase "installed ..." in a message."""

    VCS = "from version control (vcs_info)"
    LOCAL_DIRECTORY = "from a local directory (dir_info)"
    LOCAL_ARCHIVE = "from a local archive (archive_info with a file: URL)"
    INDEX = "with no direct_url.json, as from a package index"
    EDITABLE = "editable (dir_info with editable true)"
    REMOTE_ARCHIVE = "from a remote archive (archive_info with a URL that is not file:)"
    UNCLASSIFIED = "in a form Pivot cannot classify"


# The forms whose distributions get a key.
_KEYED_FORMS = frozenset({InstallForm.VCS, InstallForm.LOCAL_DIRECTORY, InstallForm.LOCAL_ARCHIVE})

# The forms that make a distribution a candidate by itself: the keyed forms, and the unclassified
# form, which may be one of them.
_SEED_FORMS = _KEYED_FORMS | {InstallForm.UNCLASSIFIED}

# The forms of a distribution whose RECORD can make a file it lists library code wherever a link
# there points (`owned_by_libraries`): each known form but the editable one. Named one by one, so
# a form added later leaves such a file to its resolved path.
_LIBRARY_FORMS = frozenset(
    {
        InstallForm.INDEX,
        InstallForm.VCS,
        InstallForm.LOCAL_DIRECTORY,
        InstallForm.LOCAL_ARCHIVE,
        InstallForm.REMOTE_ARCHIVE,
    }
)

# The `direct_url.json` fields that name a form; PEP 610 requires exactly one.
_FORM_FIELDS = ("vcs_info", "dir_info", "archive_info")


class Classification(NamedTuple):
    """A distribution's install form, and why it is `UNCLASSIFIED` (empty for every other form)."""

    form: InstallForm
    reason: str


class Requirements(NamedTuple):
    """The PEP 503 names a distribution's requirements name, and why they cannot be read (empty
    when they were read)."""

    names: frozenset[str]
    problem: str


class Record(NamedTuple):
    """The paths a distribution's RECORD lists, or None and why it has none."""

    paths: tuple[str, ...] | None
    problem: str


class Decision(NamedTuple):
    """What a distribution the closure meets gives a stage: a key or not, and a refusal message
    (empty when nothing is refused)."""

    keyed: bool
    refusal: str


class DistributionKeys(NamedTuple):
    """The `dist:` keys of one stage, and the messages of its refusals."""

    keys: dict[str, str]
    refusals: list[str]


class ReachedModule(NamedTuple):
    """A module a stage reaches by name: where it comes from (`origin`, as `import_origin` gives
    it: a file's path resolved), and its file as the import located it (`located`: its spec's
    origin or `__file__`, not resolved; None when it has none). In a venv whose installer links
    each file into its cache from a real directory (uv's `--link-mode symlink`), the resolved
    path lies in the cache, and only the located one names the directory the import searched."""

    origin: import_origin.Origin | None
    located: str | None


@dataclasses.dataclass(frozen=True)
class Installed:
    """A distribution's metadata directory on the worker's path, found as `importlib.metadata`
    finds it."""

    dist_info: str  # its `.dist-info` or `.egg-info` directory, in the resolved `site_dir`
    site_dir: str  # the directory that holds it, resolved
    name: str  # its PEP 503 name, read from the directory's name as `importlib.metadata` reads it
    classification: Classification


def _normalize(name: str) -> str:
    """`name` normalised as PEP 503 says: lower case, each run of `-`, `_` and `.` made one `-`."""
    return re.sub(r"[-_.]+", "-", name).lower()


def classify(direct_url: bytes | OSError | None) -> Classification:
    """The install form a `direct_url.json` gives (PEP 610): its bytes, None when the
    distribution has none, or the error that reading it raised.

    `vcs_info`, `dir_info` whose `editable` is not `true`, and `archive_info` whose `url` has the
    scheme `file` get a key. No `direct_url.json` (an index or a `--find-links` install),
    `dir_info` with `editable` true and `archive_info` with another scheme get none. A file that
    cannot be read, is not JSON, is nested deeper than the JSON parser recurses, is not a JSON
    object, or names none or more than one of the three forms is unclassified: a stage that
    reaches it is refused (`decide`)."""
    if direct_url is None:
        return Classification(InstallForm.INDEX, "")
    if isinstance(direct_url, OSError):
        return _unclassified(f"its direct_url.json cannot be read ({direct_url})")
    try:
        data: object = json.loads(direct_url)
    except ValueError as exc:  # also a UnicodeDecodeError
        return _unclassified(f"its direct_url.json is not JSON ({exc})")
    except RecursionError as exc:  # `[[[...`, nested deeper than the parser's recursion limit
        return _unclassified(f"its direct_url.json is nested too deeply to parse ({exc})")
    if not isinstance(data, dict):
        return _unclassified("its direct_url.json is not a JSON object")
    fields = cast("dict[str, object]", data)
    forms = [field for field in _FORM_FIELDS if field in fields]
    if len(forms) != 1:
        return _unclassified(
            f"its direct_url.json names {len(forms)} of vcs_info, dir_info and archive_info, "
            + "where PEP 610 requires exactly one"
        )
    if forms[0] == "vcs_info":
        return Classification(InstallForm.VCS, "")
    if forms[0] == "dir_info":
        info = fields["dir_info"]
        if not isinstance(info, dict):
            return _unclassified("its direct_url.json's dir_info is not a JSON object")
        # PEP 610: an absent `editable` means false.
        editable = cast("dict[str, object]", info).get("editable")
        form = InstallForm.EDITABLE if editable is True else InstallForm.LOCAL_DIRECTORY
        return Classification(form, "")
    url = fields.get("url")  # PEP 610 requires it; absent, the archive's place is unknown
    if not isinstance(url, str):
        return _unclassified("its direct_url.json has archive_info and no url")
    try:
        scheme = urllib.parse.urlsplit(url).scheme
    except ValueError as exc:
        return _unclassified(f"its direct_url.json's url cannot be parsed ({exc})")
    form = InstallForm.LOCAL_ARCHIVE if scheme.lower() == "file" else InstallForm.REMOTE_ARCHIVE
    return Classification(form, "")


def _unclassified(reason: str) -> Classification:
    return Classification(InstallForm.UNCLASSIFIED, reason)


def requirement_names(metadata: bytes | OSError | None) -> Requirements:
    """The PEP 503 names that a metadata file's `Requires-Dist` headers start with: its bytes,
    None when the distribution has none, or the error that reading it raised.

    Only the headers are read, up to the first empty line, as `importlib.metadata` reads them with
    `email`'s parser, read here by hand since importing `email.parser` and `email.policy` adds
    some 20 modules to the start of every worker, which imports this module: a line ends at CR LF,
    CR or LF; one that starts with a space or a tab continues the header before it; every other
    line is a header, `<name>:<value>`, its name of printable ASCII characters but the colon,
    compared without case. A continued header's value is its first line's followed by each
    continuation line, so a requirement's name is read from its first line, or from the next when
    the first holds none. Only each requirement's leading PEP 508 name is read: markers and extras
    are not evaluated, so a requirement behind a false marker or an extra is followed too. A file
    that cannot be read, is not UTF-8, has a line in its headers that is neither a header nor the
    continuation of one (a first line that starts with a space or a tab is not one), has no
    `Metadata-Version` header (not metadata), or has a `Requires-Dist` header that does not start
    with a name gives no names and a problem: Pivot cannot tell what the distribution depends on.
    So no names and no problem means the file has no `Requires-Dist` header."""
    if metadata is None:
        return Requirements(frozenset(), "has no METADATA file")
    if isinstance(metadata, OSError):
        return Requirements(frozenset(), f"has a METADATA file Pivot cannot read ({metadata})")
    try:
        text = metadata.decode("utf-8")
    except UnicodeDecodeError as exc:
        return Requirements(frozenset(), f"has a METADATA file that is not UTF-8 ({exc})")
    # Each header's name, in lower case, and the lines of its value.
    headers = list[tuple[str, list[str]]]()
    for line in _LINE_END.split(_END_OF_HEADERS.split(text, maxsplit=1)[0]):
        # An empty line ends the headers: one that starts the file, one that `_END_OF_HEADERS`
        # does not match because a lone CR makes it (between lone CRs, a CR LF then a lone CR, a
        # lone CR then a CR LF), or the one after the last line end of a file that is only headers.
        if not line:
            break
        if headers and line.startswith((" ", "\t")):
            headers[-1][1].append(line)
            continue
        header = _HEADER.fullmatch(line)
        if header is None:
            return Requirements(
                frozenset(),
                "has a METADATA file whose headers Pivot cannot read (a line that is neither a "
                + f"header nor the continuation of one: {line!r})",
            )
        headers.append((header[1].lower(), [header[2]]))
    if all(name != "metadata-version" for name, _ in headers):
        return Requirements(frozenset(), "has a METADATA file with no Metadata-Version header")
    return _leading_names(
        ("\n".join(lines) for name, lines in headers if name == "requires-dist"),
        "a Requires-Dist header",
    )


def requires_txt_names(requires: bytes | OSError | None) -> Requirements:
    """The PEP 503 names that the requirement lines of an `.egg-info` directory's `requires.txt`
    start with, which `importlib.metadata` reads when the directory's metadata has no
    `Requires-Dist` header: its bytes, None when the directory has none, or the error that
    reading it raised.

    Every line is a requirement but a blank one, a comment (`#`) and a section header (`[extra]`,
    `[:marker]`, `[extra:marker]`), as `importlib.metadata` reads the file; the lines under a
    section header are requirements too. As for `Requires-Dist`, only each requirement's leading
    PEP 508 name is read: no extra or marker is evaluated. No file gives no names and no problem:
    the distribution requires nothing. A file that cannot be read, is not UTF-8, or has a
    requirement line that does not start with a name gives no names and a problem."""
    if requires is None:
        return Requirements(frozenset(), "")
    if isinstance(requires, OSError):
        return Requirements(frozenset(), f"has a requires.txt file Pivot cannot read ({requires})")
    try:
        text = requires.decode("utf-8")
    except UnicodeDecodeError as exc:
        return Requirements(frozenset(), f"has a requires.txt file that is not UTF-8 ({exc})")
    requirements = list[str]()
    for line in text.splitlines():
        value = line.strip()
        if not value or value.startswith("#") or (value.startswith("[") and value.endswith("]")):
            continue
        requirements.append(value)
    return _leading_names(requirements, "a requires.txt line")


def _leading_names(requirements: Iterable[str], what: str) -> Requirements:
    """The PEP 503 names that `requirements` start with, each read as its leading PEP 508 name once
    stripped, or no names and a problem naming the first that starts with no name, as `what` (`a
    Requires-Dist header`, `a requires.txt line`)."""
    names = set[str]()
    for requirement in requirements:
        match = _REQUIREMENT_NAME.match(requirement.strip())
        if match is None:
            return Requirements(
                frozenset(), f"has {what} that names no distribution ({requirement!r})"
            )
        names.add(_normalize(match.group()))
    return Requirements(frozenset(names), "")


def record_paths(record: bytes | OSError | None) -> Record:
    """The paths a RECORD file lists, the first field of each CSV row, as
    `importlib.metadata.Distribution.files` reads them, but with every listed path, a missing
    file's too: its bytes, None when the distribution has none, or the error that reading it
    raised."""
    if record is None:
        return Record(None, "has no RECORD file")
    if isinstance(record, OSError):
        return Record(None, f"has a RECORD file Pivot cannot read ({record})")
    try:
        rows = list(csv.reader(record.decode("utf-8").splitlines()))
    except (UnicodeDecodeError, csv.Error) as exc:
        return Record(None, f"has a RECORD file Pivot cannot read ({exc})")
    return Record(tuple(row[0] for row in rows if row and row[0]), "")


def hashed_files(paths: Iterable[str], dist_info_name: str) -> list[tuple[str, str]]:
    """The files of a distribution whose RECORD lists `paths` that its key hashes, each as
    `(name in the hash, path in the site directory)`, in name order.

    Kept: each relative path that, normalised (`posixpath.normpath`), stays inside the directory
    that holds the `.dist-info` directory `dist_info_name`. Left out: every path in a
    `*.dist-info` directory but the distribution's own `entry_points.txt` (hashed under one name
    whatever the version in its directory's name), and every path with a `__pycache__` part,
    which the installer compiles (pip lists the bytecode it writes). A `.pyc` file outside
    `__pycache__` came from the wheel, the code of a sourceless module, and is kept. A path listed
    twice is hashed once."""
    own_entry_points = posixpath.join(dist_info_name, "entry_points.txt")
    kept = dict[str, str]()
    for path in paths:
        normal = _inside(path)
        if normal is None:
            continue
        parts = normal.split("/")
        if normal == own_entry_points:
            kept[_OWN_ENTRY_POINTS] = normal
        elif not (
            any(part.lower().endswith(".dist-info") for part in parts) or "__pycache__" in parts
        ):
            kept[normal] = normal
    return sorted(kept.items())


def _inside(path: str) -> str | None:
    """A RECORD path normalised (`posixpath.normpath`), when it stays inside the directory that
    holds the `.dist-info` directory; None for an absolute path, and for one that, normalised, is
    that directory or leaves it (a console script's `../../../bin/<name>`)."""
    if posixpath.isabs(path):
        return None
    normal = posixpath.normpath(path)
    return None if normal.split("/")[0] in (".", "..") else normal


def content_digest(files: Iterable[tuple[str, str | None]]) -> str:
    """The value one install gives its key: the xxh64 hex of the JSON array of each hashed file's
    `[name, xxh64 hex of its bytes]` in name order, null in place of the digest of a file that is
    missing or cannot be read. JSON keeps each name and digest apart from the next."""
    ordered = sorted(files, key=lambda file: file[0])
    return _json_digest(ordered)


def key_value(digests: Iterable[str]) -> str:
    """The value of a key from the `content_digest` of each reached install of its name: two
    installs of one name on the path, both reached, give one key."""
    return _json_digest(sorted(digests))


def _json_digest(value: object) -> str:
    """The xxh64 hex of `value` written as compact JSON."""
    return xxhash.xxh64(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def candidates(seeds: Iterable[str], requires: Mapping[str, Collection[str]]) -> frozenset[str]:
    """`seeds` and every distribution that requires one of them, directly or through others:
    `requires` maps each distribution to the distributions its requirements resolve to."""
    required_by = dict[str, list[str]]()
    for distribution, targets in requires.items():
        for target in targets:
            required_by.setdefault(target, []).append(distribution)
    found = set(seeds)
    queue = list(found)
    while queue:
        for requirer in required_by.get(queue.pop(), ()):
            if requirer not in found:
                found.add(requirer)
                queue.append(requirer)
    return frozenset(found)


def reach(
    start: Iterable[str], requires: Mapping[str, Collection[str]], stops: Collection[str]
) -> list[str]:
    """`start` and every distribution their requirements (`requires`) resolve to, transitively,
    each once (a cycle ends), in the order met; the requirements of a distribution in `stops`
    (pivot's own) are not followed."""
    met = dict.fromkeys(start)
    queue = list(met)
    while queue:
        distribution = queue.pop()
        if distribution in stops:
            continue
        for target in requires.get(distribution, ()):
            if target not in met:
                met[target] = None
                queue.append(target)
    return list(met)


def owners(
    origin: import_origin.Origin | None,
    real: str | None,
    file_owners: Mapping[str, Collection[str]],
    directory_owners: Mapping[str, Collection[str]],
) -> set[str]:
    """The candidates a reached module maps to, by file, never by name: every candidate whose
    RECORD lists the file its origin covers, resolved, and every candidate with a file under one
    of the directories it covers (`_reached_paths`, with `real` the module's file as the import
    located it, its directory resolved and its last name kept, or None). `file_owners` maps each
    resolved file a candidate's RECORD lists to those candidates, and `directory_owners` each
    directory above such a file, or above a path the RECORD lists, as it names it
    (`_build_index`)."""
    files, directories = _reached_paths(origin, real)
    found = set[str]()
    for _, file in files:
        found.update(file_owners.get(file, ()))
    for directory in directories:
        found.update(directory_owners.get(directory, ()))
    return found


def _reached_paths(
    origin: import_origin.Origin | None, real: str | None
) -> tuple[list[tuple[str, str]], list[str]]:
    """The files and the directories a reached module covers. A `("file", path)` origin covers
    its file, as the pair of where the import found it and `path`, resolved: where is `real`, the
    module's file as the import located it with its directory resolved and its last name kept
    (`path` when None). For a package's `__init__` file it also covers the package's directory,
    the directory of `real`, as the import searched it (the walk records only the top name of
    `import a.b`). Where an installer links each file into its cache from a real directory of a
    site directory (uv's symlink link mode), `path` and the directory above it lie in the cache,
    while `real` and its directory lie in the site directory, where each distribution with a file
    in the package lists its own (`_listed_directories`), as where the files are copies. A
    `("namespace", *directories)` origin covers its directories, which `import_origin` resolves.
    Any other origin (built-in, frozen, no file, unknown, None) covers nothing."""
    match origin:
        case ("file", path):
            where = path if real is None else real
            return [(where, path)], [os.path.dirname(where)] if _is_package_init(where) else []
        case ("namespace", *directories):
            return [], directories
        case _:
            return [], []


def decide(
    installed: Installed,
    record: Record,
    requirements: Requirements,
    *,
    pivot_own: bool,
    owner: bool,
) -> Decision:
    """What a distribution the closure meets gives the stage: a reached owner of a module
    (`owner`), or one that a met distribution requires.

    In this order: pivot's own distribution (`pivot_own`) gives nothing. An unclassified one is
    refused, since Pivot cannot tell whether to hash its files. One installed from version control,
    a local directory or a local archive with no RECORD is refused, since Pivot cannot tell its
    files. One whose requirements cannot be read is refused, since Pivot cannot tell what it
    depends on. A distribution of a keyed form with a RECORD gets a key, refused or not."""
    if pivot_own:
        return Decision(keyed=False, refusal="")
    form = installed.classification.form
    keyed = form in _KEYED_FORMS and record.paths is not None
    if form is InstallForm.UNCLASSIFIED:
        reason = f"is installed {form.value}: {installed.classification.reason}"
    elif form in _KEYED_FORMS and record.paths is None:
        reason = f"is installed {form.value} and {record.problem}, so Pivot cannot read its files"
    elif requirements.problem:
        reason = f"{requirements.problem}, so Pivot cannot tell what it depends on"
    else:
        return Decision(keyed=keyed, refusal="")
    how = "this stage reaches" if owner else "a distribution this stage reaches depends on"
    return Decision(
        keyed=keyed,
        refusal=f"Distribution '{installed.name}' ({installed.dist_info}), which {how}, {reason}. "
        + "Pivot cannot fingerprint the installed code the stage runs, so the stage could skip "
        + f"a run after that code changes. Fix: reinstall '{installed.name}'.",
    )


def _unlisted_refusal(module: str, file: str, installed: Installed, record: Record) -> str:
    """The refusal for a reached module whose `file` lies in the site directory of a candidate
    with no RECORD (`installed`, `record`) while no RECORD in that directory lists the file:
    Pivot cannot tell whether the module is that distribution's."""
    return (
        f"Module '{module}' ({file}) lies in {installed.site_dir}, where distribution "
        + f"'{installed.name}' ({installed.dist_info}), installed "
        + f"{installed.classification.form.value}, {record.problem}, and no RECORD file there "
        + "lists that file. Pivot cannot tell whether the stage runs that distribution's code, "
        + f"so the stage could skip a run after it changes. Fix: reinstall '{installed.name}'."
    )


def owned_by_libraries(forms: Collection[InstallForm]) -> bool:
    """Whether a file linked into a site directory is library code by the distributions there
    whose RECORDs list it, given each one's install form (`forms`, one for each): at least one
    lists it, and every one is installed in a known form that is not editable
    (`_LIBRARY_FORMS`). An editable or unclassified one, or none, leaves the file to its resolved
    path (`fingerprint.is_library_file`): flit's `install --symlink` links a module file of a
    checkout into a site directory as an editable install, which stays user code."""
    return bool(forms) and all(form in _LIBRARY_FORMS for form in forms)


def needs_origins(path: Sequence[str], pivot_dirs: Collection[str]) -> bool:
    """Whether a stage's modules can give a key or a refusal with `path` as the worker's
    `sys.path` and `pivot_dirs` as pivot's own resolved package directories: False when no
    distribution on the path is installed from version control, a local directory or a local
    archive, or unclassified, so the caller need not find its modules' origins. Reads the
    distributions as `distribution_keys` does, the index of the candidates included, once per
    process for one `path` and `pivot_dirs`."""
    with _lock:
        return _snapshot_of(path, pivot_dirs).index is not None


def distribution_keys(
    modules: Mapping[str, ReachedModule],
    path: Sequence[str],
    pivot_dirs: Collection[str],
) -> DistributionKeys:
    """The `dist:` keys of a stage that reaches `modules` (each name's origin and file as its
    import located it), with `path` as the worker's `sys.path` and `pivot_dirs` as pivot's own
    resolved package directories, and the messages of the stage's refusals.

    No module, no read. The installed distributions are read once per process for one `path` and
    `pivot_dirs` (`reset` forgets them), under a lock, since two threads of one process can
    fingerprint stages at once."""
    if not modules:
        return DistributionKeys({}, [])
    with _lock:
        return _snapshot_of(path, pivot_dirs).keys_for(modules)


def listed_as_library(file: str, path: Sequence[str], pivot_dirs: Collection[str]) -> bool:
    """Whether `file`, an absolute path whose directory is resolved and whose last name is not,
    is library code by the RECORDs of its site directory (`owned_by_libraries`), with `path` as
    the worker's `sys.path` and `pivot_dirs` as pivot's own resolved package directories. Its site
    directory is the deepest entry of the path that the scan listed and that holds it; the
    distributions there whose RECORD lists it are those with a row whose path, normalised, is
    `file`'s path relative to that directory. No RECORD path is resolved: in a venv uv links into
    its cache, every listed file is a link. A file that no entry of the path holds is listed by
    none. Scans the path as `distribution_keys` does, once per process for one `path` and
    `pivot_dirs`, and reads every RECORD of the file's site directory once; it reads no
    requirement and builds no index of the candidates, which a worker never needs."""
    with _lock:
        listing = _snapshot_of(path, pivot_dirs).listed_by(file)
    return owned_by_libraries([installed.classification.form for installed in listing])


def reset() -> None:
    """Forget what was read of the installed distributions, so the next stage reads them again."""
    global _snapshot
    with _lock:
        _snapshot = None


def _snapshot_of(path: Sequence[str], pivot_dirs: Collection[str]) -> _Snapshot:
    """The snapshot of `path` and `pivot_dirs`: the last one, when it read them, else a new one.
    The caller holds `_lock`."""
    global _snapshot
    path_key, dirs_key = tuple(path), frozenset(pivot_dirs)
    if _snapshot is None or (_snapshot.path, _snapshot.pivot_dirs) != (path_key, dirs_key):
        _snapshot = _Snapshot(path_key, dirs_key)
    return _snapshot


@dataclasses.dataclass(frozen=True)
class _Index:
    """The candidates of one snapshot and what decides their keys."""

    installed: Mapping[str, Installed]  # every distribution, by `dist_info`
    requirements: Mapping[str, Requirements]  # every distribution's
    requires: Mapping[str, frozenset[str]]  # each candidate's requirements, resolved to candidates
    records: Mapping[str, Record]  # each candidate's
    pivot_own: frozenset[str]  # the candidates with a RECORD file in one of pivot's directories
    file_owners: Mapping[str, frozenset[str]]
    directory_owners: Mapping[str, frozenset[str]]
    # Each site directory's candidates with no RECORD, which own no file.
    recordless: Mapping[str, tuple[str, ...]]


class _Scan(NamedTuple):
    """What a scan of the worker's path found."""

    installed: tuple[Installed, ...]  # every distribution, in the order found
    site_dirs: frozenset[str]  # each entry of the path that could be listed, resolved


class _Snapshot:
    """What one process read of the distributions on one worker path: the distributions when it
    is made, and, each when first needed, the index of the candidates, each distribution's
    RECORD, the files a site directory's RECORDs list, resolved, and the paths they list, as
    written, the refusals of a reached module with a file that no RECORD lists, and the digest of
    a keyed distribution's files."""

    def __init__(self, path: tuple[str, ...], pivot_dirs: frozenset[str]) -> None:
        self.path: tuple[str, ...] = path  # the worker's `sys.path`
        self.pivot_dirs: frozenset[str] = pivot_dirs  # pivot's own package directories
        scan = _scan(path)
        self._installed: tuple[Installed, ...] = scan.installed
        self._site_dirs: frozenset[str] = scan.site_dirs
        # Each directory's real path (`_real_dir`); a site directory is resolved already.
        self._real_dirs: dict[str, str] = {site_dir: site_dir for site_dir in scan.site_dirs}
        self._records: dict[str, Record] = {}  # `dist_info` -> its RECORD (`_record`)
        # A reached module's file as its import located it, absolute, -> that file with its
        # directory resolved (`_real_file`).
        self._real_files: dict[str, str] = {}
        # Site directory -> every resolved file that a RECORD of a distribution there lists.
        self._listed: dict[str, frozenset[str]] = {}
        # Site directory -> each path a RECORD there lists, normalised, -> the distributions whose
        # RECORD lists it (`_record_index`).
        self._record_indexes: dict[str, dict[str, tuple[Installed, ...]]] = {}
        self._digests: dict[str, str] = {}  # `dist_info` -> its `content_digest`
        # (name, module) -> `_unlisted`'s refusals for that reached module.
        self._unlisted_refusals: dict[tuple[str, ReachedModule], tuple[str, ...]] = {}

    @functools.cached_property
    def index(self) -> _Index | None:
        """The index of the candidates (`_build_index`), built when first asked for, by
        `needs_origins` or `keys_for`: None when no distribution on the path is of a seed form,
        and then no stage gets a key or a refusal, and no requirement is read."""
        return _build_index(self._installed, self.pivot_dirs, self._real_dirs, self._record)

    def keys_for(self, modules: Mapping[str, ReachedModule]) -> DistributionKeys:
        index = self.index
        if index is None:
            return DistributionKeys({}, [])
        refusals = set[str]()
        reached = dict[str, None]()
        for name in sorted(modules):
            module = modules[name]
            real = self._real_file(module.located)
            found = owners(module.origin, real, index.file_owners, index.directory_owners)
            for distribution in sorted(found):
                reached[distribution] = None
            if index.recordless:
                refusals.update(self._unlisted(name, module, real, index))
        digests = dict[str, list[str]]()
        for distribution in reach(reached, index.requires, index.pivot_own):
            installed = index.installed[distribution]
            decision = decide(
                installed,
                index.records[distribution],
                index.requirements[distribution],
                pivot_own=distribution in index.pivot_own,
                owner=distribution in reached,
            )
            if decision.refusal:
                refusals.add(decision.refusal)
            if decision.keyed:
                digests.setdefault(installed.name, []).append(self._digest(installed, index))
        keys = {f"{_KEY_PREFIX}{name}": key_value(digests[name]) for name in sorted(digests)}
        return DistributionKeys(keys, sorted(refusals))

    def _real_file(self, located: str | None) -> str | None:
        """A reached module's file as its import `located` it, made absolute, with its directory
        resolved and its last name kept, as `fingerprint.is_library_file` names a file, found
        once for each such path; None for None."""
        if located is None:
            return None
        absolute = os.path.abspath(located)
        if absolute not in self._real_files:
            directory, name = os.path.split(absolute)
            self._real_files[absolute] = os.path.join(os.path.realpath(directory), name)
        return self._real_files[absolute]

    def _digest(self, installed: Installed, index: _Index) -> str:
        """The `content_digest` of a keyed distribution's files, read from disk once."""
        if installed.dist_info not in self._digests:
            paths = index.records[installed.dist_info].paths
            assert paths is not None, "decide keys only a distribution with a RECORD"
            files = hashed_files(paths, os.path.basename(installed.dist_info))
            self._digests[installed.dist_info] = content_digest(
                (name, _file_digest(os.path.join(installed.site_dir, relative)))
                for name, relative in files
            )
        return self._digests[installed.dist_info]

    def _unlisted(
        self, name: str, module: ReachedModule, real: str | None, index: _Index
    ) -> tuple[str, ...]:
        """The refusals for reached module `name` with a file that no RECORD in its site directory
        lists, where that directory holds a candidate with no RECORD (`_Index.recordless`), which
        owns no file, so Pivot cannot tell whether the module is that candidate's: one for each
        such candidate, naming the first such file of that directory. Found once for each module.

        The files are those the module covers (`_reached_paths`, with `real` its file as
        `_real_file` gives it): the module's own file, and, since such a candidate may hold a
        submodule, the files under the directory of a package's `__init__` file or of a namespace
        package but those under a `__pycache__` directory (`_files_under`). A file's site
        directory is the deepest entry of the path that the scan listed and that holds the file
        where the import found it or the walk listed it (`_site_dir`), and a RECORD there lists
        the file when one of its paths resolves to the file's resolved path (`_listed_in`): in a
        venv whose installer links each file into its cache, the first lies in the site directory
        and the second in the cache, as a resolved RECORD path does. A directory is walked only
        for the site directories with such a candidate that can hold a file under it (the deepest
        entry of the path above the directory, and each entry at or under it), and only until
        each of them has its file. The walk enters no directory that is a symbolic link, so a
        file under such a link is not checked."""
        key = (name, module)
        if key not in self._unlisted_refusals:
            files, directories = _reached_paths(module.origin, real)
            first = dict[str, str]()  # site directory -> its first file that no RECORD lists
            self._find_unlisted(files, set(index.recordless), first)
            for directory in directories:
                above = self._site_dir(directory)
                pending = {
                    site_dir
                    for site_dir in index.recordless
                    if site_dir not in first
                    and (site_dir in (above, directory) or directory in _ancestors(site_dir))
                }
                if pending:
                    self._find_unlisted(_files_under(directory, self._real_dirs), pending, first)
            self._unlisted_refusals[key] = tuple(
                _unlisted_refusal(
                    name, file, index.installed[distribution], index.records[distribution]
                )
                for site_dir, file in first.items()
                for distribution in index.recordless[site_dir]
            )
        return self._unlisted_refusals[key]

    def _find_unlisted(
        self, files: Iterable[tuple[str, str]], pending: set[str], first: dict[str, str]
    ) -> None:
        """Put in `first`, for each site directory of `pending`, the first of `files` (each the
        pair of where the import found it or the walk listed it, a path whose directory is
        resolved, and its resolved path) that lies in it where found (`_site_dir`) and that no
        RECORD there lists resolved (`_listed_in`), named where found, and take that directory
        out of `pending`; once `pending` is empty, read no more of `files`."""
        for where, resolved in files:
            site_dir = self._site_dir(where)
            if site_dir in pending and resolved not in self._listed_in(site_dir):
                first[site_dir] = where
                pending.remove(site_dir)
                if not pending:
                    return

    def listed_by(self, file: str) -> tuple[Installed, ...]:
        """The distributions whose RECORD lists `file`, a path whose directory is resolved, in its
        site directory (`_site_dir`): those of that directory with a RECORD path that, normalised,
        is `file`'s path relative to it (`_record_index`). No distribution when no entry of the
        path holds `file`."""
        site_dir = self._site_dir(file)
        if site_dir is None:
            return ()
        return self._record_index(site_dir).get(os.path.relpath(file, site_dir), ())

    def _site_dir(self, file: str) -> str | None:
        """The deepest entry of the path that the scan listed and that holds `file`, a path whose
        directory is resolved, or None when none holds it: the entry an import of the file's
        module searches."""
        return next((up for up in _ancestors(file) if up in self._site_dirs), None)

    def _listed_in(self, site_dir: str) -> frozenset[str]:
        """Every resolved file that a RECORD of a distribution in `site_dir` lists (`_record`)."""
        if site_dir not in self._listed:
            listed = set[str]()
            for installed in self._installed:
                if installed.site_dir == site_dir:
                    paths = self._record(installed).paths or ()
                    listed.update(_resolved(installed, paths, self._real_dirs))
            self._listed[site_dir] = frozenset(listed)
        return self._listed[site_dir]

    def _record_index(self, site_dir: str) -> dict[str, tuple[Installed, ...]]:
        """Each path that a RECORD of a distribution in `site_dir` lists (`_record`) inside it,
        normalised (`_inside`) and not resolved, with the distributions whose RECORD lists it,
        each once."""
        if site_dir not in self._record_indexes:
            listing = dict[str, dict[str, Installed]]()
            for installed in self._installed:
                if installed.site_dir == site_dir:
                    for path in self._record(installed).paths or ():
                        normal = _inside(path)
                        if normal is not None:
                            listing.setdefault(normal, {})[installed.dist_info] = installed
            self._record_indexes[site_dir] = {
                path: tuple(found.values()) for path, found in listing.items()
            }
        return self._record_indexes[site_dir]

    def _record(self, installed: Installed) -> Record:
        """A distribution's RECORD (`_read_record`), read once, whether the index of the
        candidates (`_build_index`), the refusal of a file that no RECORD lists (`_listed_in`) or
        the question about a linked file (`_record_index`) asks for it first."""
        if installed.dist_info not in self._records:
            self._records[installed.dist_info] = _read_record(installed)
        return self._records[installed.dist_info]


def _build_index(
    installed: Sequence[Installed],
    pivot_dirs: frozenset[str],
    real_dirs: dict[str, str],
    record_of: Callable[[Installed], Record],
) -> _Index | None:
    """The candidates and their ownership, or None when no distribution on the path is of a seed
    form (`_SEED_FORMS`): then no stage gets a key or a refusal, and no requirement or RECORD is
    read. A candidate's RECORD is `record_of` it (the snapshot's `_record`, which reads each
    once). Each RECORD path is resolved by `_resolved`, with `real_dirs` as `_real_path`'s memo. A
    candidate owns each file its RECORD lists, resolved, and each directory above such a file or
    above a path its RECORD lists inside its site directory, named as the RECORD names it
    (`_listed_directories`)."""
    seeds_by_form = [
        entry.dist_info for entry in installed if entry.classification.form in _SEED_FORMS
    ]
    if not seeds_by_form:
        return None
    by_dist_info = {entry.dist_info: entry for entry in installed}
    first = dict[str, str]()  # name -> the first distribution of that name on the path
    for entry in installed:
        first.setdefault(entry.name, entry.dist_info)
    requirements = {entry.dist_info: _read_requirements(entry) for entry in installed}
    resolved = {
        dist_info: frozenset(first[name] for name in found.names if name in first)
        for dist_info, found in requirements.items()
    }
    unknown = [dist_info for dist_info, found in requirements.items() if found.problem]
    found_candidates = candidates([*seeds_by_form, *unknown], resolved)
    records = {dist_info: record_of(by_dist_info[dist_info]) for dist_info in found_candidates}
    file_owners = dict[str, set[str]]()
    directory_owners = dict[str, set[str]]()
    pivot_own = set[str]()
    recordless = dict[str, list[str]]()
    for dist_info in sorted(found_candidates):
        entry = by_dist_info[dist_info]
        paths = records[dist_info].paths
        if paths is None:
            recordless.setdefault(entry.site_dir, []).append(dist_info)
            continue
        directories = set[str]()
        for file in _resolved(entry, paths, real_dirs):
            file_owners.setdefault(file, set()).add(dist_info)
            directories.update(_ancestors(file))
        directories.update(_listed_directories(entry.site_dir, paths))
        for directory in directories:
            directory_owners.setdefault(directory, set()).add(dist_info)
        if not pivot_dirs.isdisjoint(directories):
            pivot_own.add(dist_info)
    return _Index(
        installed=by_dist_info,
        requirements=requirements,
        requires={
            dist_info: resolved[dist_info] & found_candidates for dist_info in found_candidates
        },
        records=records,
        pivot_own=frozenset(pivot_own),
        file_owners={file: frozenset(found) for file, found in file_owners.items()},
        directory_owners={
            directory: frozenset(found) for directory, found in directory_owners.items()
        },
        recordless={site_dir: tuple(found) for site_dir, found in recordless.items()},
    )


def _scan(path: Sequence[str]) -> _Scan:
    """Every distribution on `path`, in the order `importlib.metadata` finds them: each
    directory on it in order, and in each its `*.dist-info` and `*.egg-info` directories in
    listing order, named from the directory's name as `importlib.metadata` names them; and each
    entry it listed. An entry that Pivot cannot list as a directory (a zip archive, a missing or
    unreadable directory, a path with a NUL character) adds none: pip and uv install a
    distribution into a directory."""
    seen = set[str]()
    found = list[Installed]()
    site_dirs = set[str]()
    for entry in path:
        try:
            with os.scandir(entry or ".") as children:
                names = [child.name for child in children if _is_metadata_directory(child)]
        except (OSError, ValueError):  # ValueError: a NUL character in the path
            continue
        site_dir = os.path.realpath(entry or ".")
        site_dirs.add(site_dir)
        for name in names:
            dist_info = os.path.join(site_dir, name)
            if dist_info in seen:
                continue
            seen.add(dist_info)
            stem = name.lower().rpartition(".")[0].partition("-")[0]
            classification = classify(_read(os.path.join(dist_info, "direct_url.json")))
            found.append(Installed(dist_info, site_dir, _normalize(stem), classification))
    return _Scan(tuple(found), frozenset(site_dirs))


def _is_metadata_directory(child: os.DirEntry[str]) -> bool:
    if not child.name.lower().endswith((".dist-info", ".egg-info")):
        return False
    try:
        return child.is_dir()
    except OSError:
        return False


def _read(path: str) -> bytes | OSError | None:
    """The bytes of the file at `path`; None when there is none; the error that reading raised."""
    try:
        with open(path, "rb") as file:
            return file.read()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return exc


def _read_requirements(installed: Installed) -> Requirements:
    """A distribution's requirements, read as `importlib.metadata` reads them: the `Requires-Dist`
    headers of its METADATA, or of its PKG-INFO when it has no METADATA (an `.egg-info`
    directory), and for an `.egg-info` directory whose metadata has none, its `requires.txt`."""
    metadata = _read(os.path.join(installed.dist_info, "METADATA"))
    if metadata is None:
        metadata = _read(os.path.join(installed.dist_info, "PKG-INFO"))
    found = requirement_names(metadata)
    if found.names or found.problem or not installed.dist_info.lower().endswith(".egg-info"):
        return found
    return requires_txt_names(_read(os.path.join(installed.dist_info, "requires.txt")))


def _read_record(installed: Installed) -> Record:
    return record_paths(_read(os.path.join(installed.dist_info, "RECORD")))


def _resolved(
    installed: Installed, paths: Iterable[str], real_dirs: dict[str, str]
) -> Iterator[str]:
    """Each path a RECORD lists, located in the distribution's site directory and resolved to the
    path `os.path.realpath` gives. A path that holds a NUL character or a name the filesystem
    encoding cannot encode names no file and is left out: `realpath` raises a `ValueError` for
    it. One shorter than `_PATH_MAX` bytes, located and encoded, is resolved by `_real_path`, with
    `real_dirs` as its memo. A longer one can still name a file (a long name in a directory
    shorter than that, which `os.walk` lists and which can be a module's origin, as
    `import_origin` resolves it by `realpath` too), and is resolved by `realpath` alone, which
    takes a name that `os.lstat` refuses as too long for one that is no link and keeps nothing
    from one call to the next. So each path given to `_real_path` from here, and each directory
    above it that `_real_dir` memoises, is shorter than `_PATH_MAX` bytes. The index still keeps
    every directory above each listed file (`_build_index`), so one row costs memory of the order
    of its names times its length; only installed metadata, which Pivot trusts, bounds it."""
    for path in paths:
        located = os.path.join(installed.site_dir, path)
        try:
            encoded = os.fsencode(located)
        except UnicodeEncodeError:
            continue
        if b"\0" in encoded:
            continue
        if len(encoded) < _PATH_MAX:
            yield _real_path(located, real_dirs)
        else:
            yield os.path.realpath(located)


def _real_path(path: str, real_dirs: dict[str, str]) -> str:
    """`os.path.realpath(path)`, equal to it for every absolute path with no NUL character whose
    names the filesystem encoding can encode, found with one `os.path.islink` once `real_dirs`
    holds the real path of the directory that holds `path` (`_real_dir` finds and memoises it).
    The memo does not follow the working directory a relative path depends on, and for a NUL or
    a name the encoding cannot encode `realpath` raises a `ValueError`, which `os.path.islink`
    takes as no link. Each path this module resolves is one: `_resolved` leaves out every other
    RECORD path, and `_files_under` joins the names `os.walk` lists, which it decoded with
    surrogateescape and so can encode back, to a reached module's directory, which `realpath`
    resolved (`_Snapshot._real_file`, or `import_origin` for a namespace package).

    `realpath` resolves a path a name at a time from the left, so before it reads the last name
    it holds the real path of the directory that name is in, and it then takes the name as it is
    unless the name is a symbolic link or one it reads specially (`_SPECIAL_NAMES`). So this joins
    the last name to the directory's real path, and gives `realpath` the whole path for such a
    name or a link (`_real_child`)."""
    directory, name = os.path.split(path)
    if name in _SPECIAL_NAMES:
        return os.path.realpath(path)
    return _real_child(path, name, _real_dir(directory, real_dirs))


def _real_dir(directory: str, real_dirs: dict[str, str]) -> str:
    """`os.path.realpath(directory)`, memoised in `real_dirs` (a directory -> its real path), with
    every directory walked to find it. The walk goes up to the nearest directory that `real_dirs`
    holds, or to the nearest whose last name `realpath` reads specially (`_SPECIAL_NAMES`; the
    root's is empty), which `realpath` resolves; then each name below it costs one
    `os.path.islink` (`_real_child`), so a directory below one already resolved costs one. A
    loop, not a recursion: a path shorter than `_PATH_MAX` bytes can hold some 2,000 names on
    Linux, twice Python's default recursion limit."""
    walked = list[tuple[str, str]]()  # (a directory, its last name), from `directory` up
    head = directory
    while head not in real_dirs:
        parent, name = os.path.split(head)
        if name in _SPECIAL_NAMES:
            real_dirs[head] = os.path.realpath(head)
            break
        walked.append((head, name))
        head = parent
    real = real_dirs[head]
    for below, name in reversed(walked):
        real = _real_child(below, name, real)
        real_dirs[below] = real
    return real


def _real_child(path: str, name: str, real_parent: str) -> str:
    """`os.path.realpath(path)`, for a `path` whose last name, `name`, is not special
    (`_SPECIAL_NAMES`), from `real_parent`, the real path of the directory that holds it: the
    name joined to `real_parent`, or `realpath(path)` when that joined path is a symbolic link.
    Not `realpath` of the joined path: `realpath` keeps what each link it met earlier in `path`
    resolved to, and across a loop of links that differs from resolving the link alone (with
    `a -> b` and `b -> a` side by side, `realpath` gives `a` for `a/../b`, and `b` for `b`)."""
    joined = os.path.join(real_parent, name)
    return os.path.realpath(path) if os.path.islink(joined) else joined


def _ancestors(path: str) -> Iterator[str]:
    """Each directory above `path`, nearest first, up to the root."""
    parent = os.path.dirname(path)
    while parent != path:
        yield parent
        path, parent = parent, os.path.dirname(parent)


def _listed_directories(site_dir: str, paths: Iterable[str]) -> Iterator[str]:
    """Each directory above each of `paths`, the paths a RECORD in `site_dir` lists, that stays
    inside `site_dir` (`_inside`), named as the RECORD names it under `site_dir`, not resolved.
    Where an installer links each file into its cache from a real directory of `site_dir` (uv's
    symlink link mode), the directories above a listed file, resolved, lie in the cache, while a
    namespace package's origin names its directories in `site_dir`, resolved, and a regular
    package's directory is named as the import searched it, in `site_dir` (`_reached_paths`):
    these directories let the distribution own either package there, as it does where its files
    are copies, whose resolved directories they are."""
    for path in paths:
        normal = _inside(path)
        if normal is not None:
            yield from _ancestors(os.path.join(site_dir, normal))


def _is_package_init(path: str) -> bool:
    """Whether `path` is a package's `__init__` module, of any suffix (`.py`, `.pyc`, an
    extension module)."""
    return os.path.basename(path).partition(".")[0] == "__init__"


def _files_under(directory: str, real_dirs: dict[str, str]) -> Iterator[tuple[str, str]]:
    """Each file under `directory`, a reached module's directory, as `os.walk` lists them, but
    those under a `__pycache__` directory, which `hashed_files` leaves out too: a `.pyc` file
    outside one is a file like any other. Each is the pair of its path as the walk lists it and
    that path resolved (`_real_path`, with `real_dirs` as its memo). A subdirectory it cannot
    list, or a symbolic link to one, adds none."""
    for root, directories, files in os.walk(directory):
        directories[:] = [name for name in directories if name != "__pycache__"]
        for name in files:
            path = os.path.join(root, name)
            yield path, _real_path(path, real_dirs)


def _file_digest(path: str) -> str | None:
    """The xxh64 hex of the bytes of the file at `path`, read from disk; None when it is missing,
    cannot be read, has a NUL character or a name the filesystem encoding cannot encode in its
    path, or is not a regular file (a directory, or a FIFO, which opening without O_NONBLOCK would
    wait on). The descriptor is closed on every path."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except (OSError, ValueError):  # ValueError: a NUL character, or an unencodable name
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        hasher = xxhash.xxh64()
        while chunk := os.read(descriptor, _CHUNK_SIZE):
            hasher.update(chunk)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    return hasher.hexdigest()


# One process's snapshot of the distributions on the worker's path (`_snapshot_of`), which `reset`
# forgets. A process fingerprints a stage on the thread that asks for it, and two threads can ask
# at once (the comment on `fingerprint._hash_function_ast_cache` lists them), so `_lock` guards it.
_lock = threading.Lock()
_snapshot: _Snapshot | None = None
