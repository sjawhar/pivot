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
directories) and returns plain data (the keys and the refusal messages). Each decision but the one
that lists a package's files on disk (`_unlisted_files`) is a public function of plain data.

What it reads, once per process for one worker path, so a reinstall while Pivot runs is seen at the
next start: the metadata directories on the path and each `direct_url.json`; only when one of them
is installed from version control, a local directory or a local archive, or cannot be classified,
the `Requires-Dist` headers of every distribution; RECORD only for the candidates (those
distributions and every distribution that requires one, directly or through others), and every
RECORD of a site directory where a candidate has none; and the files of a keyed distribution only
when a stage reaches it.
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
    from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence

    from pivot import import_origin

# The prefix of every key this module makes.
KEY_PREFIX = "dist:"

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
KEYED_FORMS = frozenset({InstallForm.VCS, InstallForm.LOCAL_DIRECTORY, InstallForm.LOCAL_ARCHIVE})

# The `direct_url.json` fields that name a form; PEP 610 requires exactly one.
_FORM_FIELDS = ("vcs_info", "dir_info", "archive_info")


class Classification(NamedTuple):
    """A distribution's install form, and why it is `UNCLASSIFIED` (empty for every other form)."""

    form: InstallForm
    reason: str


class Requirements(NamedTuple):
    """The PEP 503 names a distribution's `Requires-Dist` headers name, and why they cannot be read
    (empty when they were read)."""

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


def normalize(name: str) -> str:
    """`name` normalised as PEP 503 says: lower case, each run of `-`, `_` and `.` made one `-`."""
    return re.sub(r"[-_.]+", "-", name).lower()


def classify(direct_url: bytes | OSError | None) -> Classification:
    """The install form a `direct_url.json` gives (PEP 610): its bytes, None when the
    distribution has none, or the error that reading it raised.

    `vcs_info`, `dir_info` whose `editable` is not `true`, and `archive_info` whose `url` has the
    scheme `file` get a key. No `direct_url.json` (an index or a `--find-links` install),
    `dir_info` with `editable` true and `archive_info` with another scheme get none. A file that
    cannot be read, is not a JSON object, or names none or more than one of the three forms is
    unclassified: a stage that reaches it is refused (`decide`)."""
    if direct_url is None:
        return Classification(InstallForm.INDEX, "")
    if isinstance(direct_url, OSError):
        return _unclassified(f"its direct_url.json cannot be read ({direct_url})")
    try:
        data: object = json.loads(direct_url)
    except ValueError as exc:  # also a UnicodeDecodeError
        return _unclassified(f"its direct_url.json is not JSON ({exc})")
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
    cannot tell what the distribution depends on."""
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
    names = set[str]()
    for value in headers.get_all("Requires-Dist") or []:
        match = _REQUIREMENT_NAME.match(value.strip())
        if match is None:
            return Requirements(
                frozenset(), f"has a Requires-Dist header that names no distribution ({value!r})"
            )
        names.add(normalize(match.group()))
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
    """The candidates a reached module's origin maps to, by file, never by name.

    `file_owners` maps each resolved file a candidate's RECORD lists to those candidates, and
    `directory_owners` each directory above such a file. A `("file", path)` origin maps to every
    candidate whose RECORD lists the file, and, for a package's `__init__` file, every candidate
    with a file under the package's directory (the walk records only the top name of `import a.b`).
    A `("namespace", *directories)` origin maps to every candidate with a file under one of them.
    Any other origin (built-in, frozen, no file, unknown, None) maps to none."""
    match origin:
        case ("file", path):
            found = set(file_owners.get(path, ()))
            if _is_package_init(path):
                found.update(directory_owners.get(os.path.dirname(path), ()))
            return found
        case ("namespace", *directories):
            return {
                distribution
                for directory in directories
                for distribution in directory_owners.get(directory, ())
            }
        case _:
            return set()


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
    keyed = form in KEYED_FORMS and record.paths is not None
    if form is InstallForm.UNCLASSIFIED:
        reason = f"is installed {form.value}: {installed.classification.reason}"
    elif form in KEYED_FORMS and record.paths is None:
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


def distribution_keys(
    origins: Mapping[str, import_origin.Origin | None],
    path: Sequence[str],
    pivot_dirs: Collection[str],
) -> DistributionKeys:
    """The `dist:` keys of a stage that reaches the modules of `origins` (each name's
    `import_origin` origin), with `path` as the worker's `sys.path` and `pivot_dirs` as pivot's
    own resolved package directories, and the messages of the stage's refusals.

    No module, no read. The installed distributions are read once per process for one `path` and
    `pivot_dirs` (`reset` forgets them), under a lock, since stages are fingerprinted in anyio
    worker threads."""
    global _snapshot
    if not origins:
        return DistributionKeys({}, [])
    key = (tuple(path), frozenset(pivot_dirs))
    with _lock:
        if _snapshot is None or _snapshot.key != key:
            _snapshot = _Snapshot(key)
        return _snapshot.keys_for(origins)


def reset() -> None:
    """Forget what was read of the installed distributions, so the next stage reads them again."""
    global _snapshot
    with _lock:
        _snapshot = None


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
    recordless: Mapping[str, tuple[str, ...]]  # each site directory's candidates with no RECORD


class _Snapshot:
    """What one process read of the distributions on one worker path, each part when first
    needed."""

    def __init__(self, key: tuple[tuple[str, ...], frozenset[str]]) -> None:
        self.key: tuple[tuple[str, ...], frozenset[str]] = key  # the worker path, pivot's dirs
        self._installed: tuple[Installed, ...] = _scan(key[0])
        self._index: _Index | None = None
        self._indexed: bool = False
        # Site directory -> every resolved file that a RECORD of a distribution there lists.
        self._listed: dict[str, frozenset[str]] = {}
        self._digests: dict[str, str] = {}  # `dist_info` -> its `content_digest`

    def keys_for(self, origins: Mapping[str, import_origin.Origin | None]) -> DistributionKeys:
        index = self._get_index()
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
        keys = {f"{KEY_PREFIX}{name}": key_value(digests[name]) for name in sorted(digests)}
        return DistributionKeys(keys, sorted(refusals))

    def _get_index(self) -> _Index | None:
        if not self._indexed:
            self._index = _build_index(self._installed, self.key[1])
            self._indexed = True
        return self._index

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
    ) -> list[str]:
        """The refusals for a reached module with a file in a site directory where a candidate
        has no RECORD, while no RECORD there lists the file: its own file, and, for a package's
        `__init__` file or a namespace package, every file under its directories, since the walk
        records only the top name of `import a.b` and such a candidate may hold a submodule."""
        refusals = list[str]()
        for site_dir, file in _unlisted_files(origin, index.recordless, self._listed_in):
            for distribution in index.recordless[site_dir]:
                installed = index.installed[distribution]
                refusals.append(
                    _unlisted_refusal(module, file, installed, index.records[distribution])
                )
        return refusals

    def _listed_in(self, site_dir: str) -> frozenset[str]:
        """Every resolved file that a RECORD of a distribution in `site_dir` lists."""
        if site_dir not in self._listed:
            listed = set[str]()
            for installed in self._installed:
                if installed.site_dir == site_dir:
                    paths = _read_record(installed).paths or ()
                    listed.update(_resolved(installed, paths))
            self._listed[site_dir] = frozenset(listed)
        return self._listed[site_dir]


def _unlisted_files(
    origin: import_origin.Origin | None,
    recordless: Collection[str],
    listed_in: Callable[[str], frozenset[str]],
) -> Iterator[tuple[str, str]]:
    """Each `(site directory, file)` where a reached module's file lies in a site directory of
    `recordless` while no RECORD there lists it (`listed_in`), the first such file for each site
    directory: the module's own file, and every file under the directory of a package's
    `__init__` file or of a namespace package (`owners` counts those as reached), but bytecode."""
    match origin:
        case ("file", path):
            files = [path]
            directories = [os.path.dirname(path)] if _is_package_init(path) else []
        case ("namespace", *names):
            files = list[str]()
            directories = names
        case _:
            return
    seen = set[str]()
    for top in [*files, *directories]:
        site_dir = next((up for up in _ancestors(top) if up in recordless), None)
        if site_dir is None or site_dir in seen:
            continue
        listed = listed_in(site_dir)
        reached = _files_under(top) if top in directories else iter((top,))
        found = next((file for file in reached if file not in listed), None)
        if found is not None:
            seen.add(site_dir)
            yield site_dir, found


def _build_index(installed: Sequence[Installed], pivot_dirs: frozenset[str]) -> _Index | None:
    """The candidates and their ownership, or None when no distribution on the path is installed
    from version control, a local directory or a local archive, or unclassified: then no stage
    gets a key or a refusal, and no requirement or RECORD is read."""
    seeds_by_form = [
        entry.dist_info
        for entry in installed
        if entry.classification.form in KEYED_FORMS
        or entry.classification.form is InstallForm.UNCLASSIFIED
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
        for file in _resolved(entry, paths):
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


def _scan(path: Sequence[str]) -> tuple[Installed, ...]:
    """Every distribution on `path`, in the order `importlib.metadata` finds them: each
    directory on it in order, and in each its `*.dist-info` and `*.egg-info` directories in
    listing order, named from the directory's name as `importlib.metadata` names them. An entry
    that Pivot cannot list as a directory (a zip archive, a missing or unreadable directory) adds
    none: pip and uv install a distribution into a directory."""
    seen = set[str]()
    found = list[Installed]()
    for entry in path:
        try:
            with os.scandir(entry or ".") as children:
                names = [child.name for child in children if _is_metadata_directory(child)]
        except OSError:
            continue
        site_dir = os.path.realpath(entry or ".")
        for name in names:
            dist_info = os.path.join(site_dir, name)
            if dist_info in seen:
                continue
            seen.add(dist_info)
            stem = name.lower().rpartition(".")[0].partition("-")[0]
            classification = classify(_read(os.path.join(dist_info, "direct_url.json")))
            found.append(Installed(dist_info, site_dir, normalize(stem), classification))
    return tuple(found)


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
    """A distribution's requirements from its METADATA, or from PKG-INFO when it has no METADATA
    (an `.egg-info` directory), the file `importlib.metadata` reads as its metadata."""
    metadata = _read(os.path.join(installed.dist_info, "METADATA"))
    if metadata is None:
        metadata = _read(os.path.join(installed.dist_info, "PKG-INFO"))
    return requirement_names(metadata)


def _read_record(installed: Installed) -> Record:
    return record_paths(_read(os.path.join(installed.dist_info, "RECORD")))


def _resolved(installed: Installed, paths: Iterable[str]) -> Iterator[str]:
    """Each path a RECORD lists, located in the distribution's site directory and resolved."""
    for path in paths:
        yield os.path.realpath(os.path.join(installed.site_dir, path))


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


def _files_under(directory: str) -> Iterator[str]:
    """Each file under `directory`, resolved, but bytecode (`.pyc` files, `__pycache__`), as
    `os.walk` lists them: a subdirectory it cannot list, or a symbolic link to one, adds none."""
    for root, directories, files in os.walk(directory):
        directories[:] = [name for name in directories if name != "__pycache__"]
        for name in files:
            if not name.endswith(".pyc"):
                yield os.path.realpath(os.path.join(root, name))


def _file_digest(path: str) -> str | None:
    """The xxh64 hex of the bytes of the file at `path`, read from disk; None when it is missing,
    cannot be read or is not a regular file (a directory, or a FIFO, which opening without
    O_NONBLOCK would wait on)."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    with os.fdopen(descriptor, "rb") as file:
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return None
            hasher = xxhash.xxh64()
            while chunk := file.read(_CHUNK_SIZE):
                hasher.update(chunk)
        except OSError:
            return None
    return hasher.hexdigest()


# One process's snapshot of the distributions on the worker's path (`distribution_keys`), which
# `reset` forgets. Fingerprints run in anyio worker threads, so `_lock` guards it.
_lock = threading.Lock()
_snapshot: _Snapshot | None = None
