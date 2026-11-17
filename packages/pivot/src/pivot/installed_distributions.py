"""The `dist:` keys of a stage's code manifest (`fingerprint.add_distribution_keys`).

A stage reaches a distribution when a module that its manifest names by name (a `byname:` entry)
comes from a file that the distribution's RECORD lists, or is a package or a namespace package
whose directory holds one (`owners`). For each reached distribution installed from version
control, from a local directory or from a local archive, and for each such distribution that a
reached one depends on, the manifest gets the key `dist:<name>`, `<name>` normalised as PEP 503
says. Its value hashes the distribution's own installed files, read from disk, so a change to them
makes the stage stale, and a reinstall of the same files from another commit, path, installer or
environment changes nothing. Package-index, remote-archive and editable distributions and pivot's
own get no key.

A leaf: it imports nothing of pivot's at run time, so `fingerprint` imports it and no import cycle
forms. It takes plain data (the reached modules' origins, the worker's `sys.path`, pivot's package
directories) and returns plain data (the keys and the refusal messages). Each decision on what a
distribution's files say (`classify`, `requirement_names`, `requires_txt_names`, `record_paths`,
`hashed_files`, `content_digest`, `key_value`, `candidates`, `reach`, `owners`, `decide`) is a
public function of plain data. What reads the disk is private: the scan of the path, the index of
the candidates, the digest of a keyed distribution's files, and the refusal of a reached file that
no RECORD lists, which reads the RECORDs of a site directory and the files under a package
(`_Snapshot`).

What it reads, once per process for one worker path, so a reinstall while Pivot runs is seen at the
next start: the metadata directories on the path and each `direct_url.json`; only when one of them
is installed from version control, a local directory or a local archive, or cannot be classified,
the requirements of every distribution (its `Requires-Dist` headers, and an `.egg-info`
directory's `requires.txt` when its metadata has none); RECORD only for the candidates (those
distributions, each distribution whose requirements cannot be read, and every distribution that
requires one of them, directly or through others); every RECORD of a site directory only when a
candidate there has no RECORD and a stage reaches a file whose site directory it is (the deepest
entry of the path that holds the file); the files under the directory of a reached package or
namespace package only when such a site directory is the deepest entry of the path above that
directory or lies at or under it, and only until each such site directory has a file there that
no RECORD lists, once for each reached module; and the files of a keyed distribution only when a
stage reaches it. Each path a RECORD lists, and each file under such a directory, is resolved
with one `os.path.islink` once the real path of the directory that holds it is known, which
`os.path.realpath` finds once for each directory (`_real_path`).
"""

from __future__ import annotations

import csv
import dataclasses
import email.parser
import email.policy
import enum
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
    from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence

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

    Only the headers are read, up to the first empty line, as `importlib.metadata` reads them
    (`email`), and only each requirement's leading PEP 508 name: markers and extras are not
    evaluated, so a requirement behind a false marker or an extra is followed too. A file that
    cannot be read, whose headers `email` reads with a defect (a line that is not a header ends
    them early), that has no `Metadata-Version` header (not metadata), or that has a
    `Requires-Dist` header that does not start with a name gives no names and a problem: Pivot
    cannot tell what the distribution depends on. So no names and no problem means the file has
    no `Requires-Dist` header."""
    if metadata is None:
        return Requirements(frozenset(), "has no METADATA file")
    if isinstance(metadata, OSError):
        return Requirements(frozenset(), f"has a METADATA file Pivot cannot read ({metadata})")
    try:
        text = metadata.decode("utf-8")
    except UnicodeDecodeError as exc:
        return Requirements(frozenset(), f"has a METADATA file that is not UTF-8 ({exc})")
    headers = email.parser.HeaderParser(policy=email.policy.compat32).parsestr(
        _END_OF_HEADERS.split(text, maxsplit=1)[0]
    )
    if headers.defects:
        return Requirements(
            frozenset(), f"has a METADATA file whose headers Pivot cannot read ({headers.defects})"
        )
    if "Metadata-Version" not in headers:
        return Requirements(frozenset(), "has a METADATA file with no Metadata-Version header")
    return _leading_names(headers.get_all("Requires-Dist") or [], "a Requires-Dist header")


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
    whatever the version in its directory's name), every `.pyc` file and every path with a
    `__pycache__` part. A path listed twice is hashed once."""
    own_entry_points = posixpath.join(dist_info_name, "entry_points.txt")
    kept = dict[str, str]()
    for path in paths:
        if posixpath.isabs(path):
            continue
        normal = posixpath.normpath(path)
        parts = normal.split("/")
        if parts[0] in (".", ".."):
            continue
        if normal == own_entry_points:
            kept[_OWN_ENTRY_POINTS] = normal
        elif not (
            any(part.lower().endswith(".dist-info") for part in parts)
            or normal.endswith(".pyc")
            or "__pycache__" in parts
        ):
            kept[normal] = normal
    return sorted(kept.items())


def content_digest(files: Iterable[tuple[str, str | None]]) -> str:
    """The value one install gives its key: the xxh64 hex of the JSON array of each hashed file's
    `[name, xxh64 hex of its bytes]` in name order, null in place of the digest of a file that is
    missing or cannot be read. JSON keeps each name and digest apart from the next."""
    ordered = sorted(files, key=lambda file: file[0])
    return xxhash.xxh64(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest()


def key_value(digests: Iterable[str]) -> str:
    """The value of a key from the `content_digest` of each reached install of its name: two
    installs of one name on the path, both reached, give one key."""
    return xxhash.xxh64(json.dumps(sorted(digests), separators=(",", ":")).encode()).hexdigest()


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
    file_owners: Mapping[str, Collection[str]],
    directory_owners: Mapping[str, Collection[str]],
) -> set[str]:
    """The candidates a reached module's origin maps to, by file, never by name: every candidate
    whose RECORD lists one of the files the origin covers, and every candidate with a file under
    one of the directories it covers (`_reached_paths`). `file_owners` maps each resolved file a
    candidate's RECORD lists to those candidates, and `directory_owners` each directory above
    such a file."""
    files, directories = _reached_paths(origin)
    found = set[str]()
    for file in files:
        found.update(file_owners.get(file, ()))
    for directory in directories:
        found.update(directory_owners.get(directory, ()))
    return found


def _reached_paths(origin: import_origin.Origin | None) -> tuple[list[str], list[str]]:
    """The files and the directories a reached module's origin covers. A `("file", path)` origin
    covers its file, and, for a package's `__init__` file, the package's directory too (the walk
    records only the top name of `import a.b`). A `("namespace", *directories)` origin covers its
    directories. Any other origin (built-in, frozen, no file, unknown, None) covers nothing."""
    match origin:
        case ("file", path):
            return [path], [os.path.dirname(path)] if _is_package_init(path) else []
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


def needs_origins(path: Sequence[str], pivot_dirs: Collection[str]) -> bool:
    """Whether a stage's modules can give a key or a refusal with `path` as the worker's
    `sys.path` and `pivot_dirs` as pivot's own resolved package directories: False when no
    distribution on the path is installed from version control, a local directory or a local
    archive, or unclassified, so the caller need not find its modules' origins. Reads the
    distributions as `distribution_keys` does, once per process for one `path` and
    `pivot_dirs`."""
    with _lock:
        return _snapshot_of(path, pivot_dirs).index is not None


def distribution_keys(
    origins: Mapping[str, import_origin.Origin | None],
    path: Sequence[str],
    pivot_dirs: Collection[str],
) -> DistributionKeys:
    """The `dist:` keys of a stage that reaches the modules of `origins` (each name's
    `import_origin` origin), with `path` as the worker's `sys.path` and `pivot_dirs` as pivot's
    own resolved package directories, and the messages of the stage's refusals.

    No module, no read. The installed distributions are read once per process for one `path` and
    `pivot_dirs` (`reset` forgets them), under a lock, since two threads of one process can
    fingerprint stages at once."""
    if not origins:
        return DistributionKeys({}, [])
    with _lock:
        return _snapshot_of(path, pivot_dirs).keys_for(origins)


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
    """What one process read of the distributions on one worker path: the distributions and the
    index of the candidates when it is made, and, each when first needed, the files a site
    directory's RECORDs list, the refusals of a reached module with a file that no RECORD lists,
    and the digest of a keyed distribution's files."""

    def __init__(self, path: tuple[str, ...], pivot_dirs: frozenset[str]) -> None:
        self.path: tuple[str, ...] = path  # the worker's `sys.path`
        self.pivot_dirs: frozenset[str] = pivot_dirs  # pivot's own package directories
        scan = _scan(path)
        self._installed: tuple[Installed, ...] = scan.installed
        self._site_dirs: frozenset[str] = scan.site_dirs
        # Each directory's real path (`_real_path`); a site directory is resolved already.
        self._real_dirs: dict[str, str] = {site_dir: site_dir for site_dir in scan.site_dirs}
        # None when no distribution on the path is of a seed form: then no stage gets a key or a
        # refusal, and no requirement or RECORD is read.
        self.index: _Index | None = _build_index(scan.installed, pivot_dirs, self._real_dirs)
        # Site directory -> every resolved file that a RECORD of a distribution there lists.
        self._listed: dict[str, frozenset[str]] = {}
        self._digests: dict[str, str] = {}  # `dist_info` -> its `content_digest`
        # (module, origin) -> `_unlisted`'s refusals for that reached module.
        self._unlisted_refusals: dict[tuple[str, import_origin.Origin | None], tuple[str, ...]] = {}

    def keys_for(self, origins: Mapping[str, import_origin.Origin | None]) -> DistributionKeys:
        index = self.index
        if index is None:
            return DistributionKeys({}, [])
        refusals = set[str]()
        reached = dict[str, None]()
        for module, origin in sorted(origins.items()):
            for distribution in sorted(owners(origin, index.file_owners, index.directory_owners)):
                reached[distribution] = None
            if index.recordless:
                refusals.update(self._unlisted(module, origin, index))
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
        self, module: str, origin: import_origin.Origin | None, index: _Index
    ) -> tuple[str, ...]:
        """The refusals for a reached module with a file that no RECORD in its site directory
        lists, where that directory holds a candidate with no RECORD (`_Index.recordless`), which
        owns no file, so Pivot cannot tell whether the module is that candidate's: one for each
        such candidate, naming the first such file of that directory. Found once for each module
        and origin.

        The files are those the origin covers (`_reached_paths`): the module's own file, and the
        files under the directory of a package's `__init__` file or of a namespace package, but
        bytecode, since such a candidate may hold a submodule. A file's site directory is the
        deepest entry of the path that the scan listed and that holds it (`_site_dir`). A
        directory is walked only for the site directories with such a candidate that can hold a
        file under it other than through a symbolic link (the deepest entry of the path above the
        directory, and each entry at or under it), and only until each of them has its file. So a
        file under the directory that is a symbolic link into another site directory with such a
        candidate is not checked."""
        key = (module, origin)
        if key not in self._unlisted_refusals:
            files, directories = _reached_paths(origin)
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
                    module, file, index.installed[distribution], index.records[distribution]
                )
                for site_dir, file in first.items()
                for distribution in index.recordless[site_dir]
            )
        return self._unlisted_refusals[key]

    def _find_unlisted(
        self, files: Iterable[str], pending: set[str], first: dict[str, str]
    ) -> None:
        """Put in `first`, for each site directory of `pending`, the first of the resolved `files`
        that lies in it (`_site_dir`) and that no RECORD there lists, and take that directory out
        of `pending`; once `pending` is empty, read no more of `files`."""
        for file in files:
            site_dir = self._site_dir(file)
            if site_dir in pending and file not in self._listed_in(site_dir):
                first[site_dir] = file
                pending.remove(site_dir)
                if not pending:
                    return

    def _site_dir(self, file: str) -> str | None:
        """The deepest entry of the path that the scan listed and that holds the resolved `file`,
        or None when none holds it: the entry an import of the file's module searches."""
        return next((up for up in _ancestors(file) if up in self._site_dirs), None)

    def _listed_in(self, site_dir: str) -> frozenset[str]:
        """Every resolved file that a RECORD of a distribution in `site_dir` lists."""
        if site_dir not in self._listed:
            listed = set[str]()
            for installed in self._installed:
                if installed.site_dir == site_dir:
                    paths = _read_record(installed).paths or ()
                    listed.update(_resolved(installed, paths, self._real_dirs))
            self._listed[site_dir] = frozenset(listed)
        return self._listed[site_dir]


def _build_index(
    installed: Sequence[Installed], pivot_dirs: frozenset[str], real_dirs: dict[str, str]
) -> _Index | None:
    """The candidates and their ownership, or None when no distribution on the path is of a seed
    form (`_SEED_FORMS`): then no stage gets a key or a refusal, and no requirement or RECORD is
    read. Each RECORD path is resolved with `real_dirs` as `_real_path`'s memo."""
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
    records = {dist_info: _read_record(by_dist_info[dist_info]) for dist_info in found_candidates}
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
    """Each path a RECORD lists, located in the distribution's site directory and resolved
    (`_real_path`, with `real_dirs` as its memo); a path that holds a NUL character, which
    `os.path.realpath` refuses, names no file and is left out."""
    for path in paths:
        if "\0" not in path:
            yield _real_path(os.path.join(installed.site_dir, path), real_dirs)


def _real_path(path: str, real_dirs: dict[str, str]) -> str:
    """`os.path.realpath(path)`, equal to it for every path, found with one `os.path.islink` once
    `real_dirs` holds the real path of the directory that holds `path`.

    `realpath` resolves the directory a path's last name is in before it reads that name, then
    takes the name as it is unless the name is a symbolic link or one it reads specially (empty,
    `.` or `..`). So this joins the last name to the directory's real path (`realpath`, memoised
    in `real_dirs`), and gives `realpath` of the whole path for such a name or a link."""
    directory, name = os.path.split(path)
    if name in ("", os.curdir, os.pardir):
        return os.path.realpath(path)
    if directory not in real_dirs:
        real_dirs[directory] = os.path.realpath(directory)
    joined = os.path.join(real_dirs[directory], name)
    return os.path.realpath(path) if os.path.islink(joined) else joined


def _ancestors(path: str) -> Iterator[str]:
    """Each directory above `path`, nearest first, up to the root."""
    parent = os.path.dirname(path)
    while parent != path:
        yield parent
        path, parent = parent, os.path.dirname(parent)


def _is_package_init(path: str) -> bool:
    """Whether `path` is a package's `__init__` module, of any suffix (`.py`, `.pyc`, an
    extension module)."""
    return os.path.basename(path).partition(".")[0] == "__init__"


def _files_under(directory: str, real_dirs: dict[str, str]) -> Iterator[str]:
    """Each file under `directory`, resolved (`_real_path`, with `real_dirs` as its memo), but
    bytecode (`.pyc` files, `__pycache__`), as `os.walk` lists them: a subdirectory it cannot
    list, or a symbolic link to one, adds none."""
    for root, directories, files in os.walk(directory):
        directories[:] = [name for name in directories if name != "__pycache__"]
        for name in files:
            if not name.endswith(".pyc"):
                yield _real_path(os.path.join(root, name), real_dirs)


def _file_digest(path: str) -> str | None:
    """The xxh64 hex of the bytes of the file at `path`, read from disk; None when it is missing,
    cannot be read, has a NUL character in its path, or is not a regular file (a directory, or a
    FIFO, which opening without O_NONBLOCK would wait on). The descriptor is closed on every
    path."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except (OSError, ValueError):  # ValueError: a NUL character in the path
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
