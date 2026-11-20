"""A differential test of `pivot repro --all --keep-going` (leave-out mode): on small projects
generated at random, the dry run (`--dry-run --jsonl`) on one copy of a tree and the run
(`--jsonl`) on another copy of it must agree, as spec R5 says
(``keep_going_agreement.disagreements``).

A seed gives one case (``_generate``): a project of one or two pipelines of plain stages,
directory producers and mapped stages over them (``Pipeline.map``, and chains of them), reading
raw files and directories, some tracked with `pivot track`, and of two pipelines the first may
read the second's stages through their handles (``_compose_across``); the changes made after a
first full run (``_Change``); and the run's options and targets. A failure names the seed and
prints the case and its pipelines, and the seed alone builds the same case again. Not
generated: a cache whose object store is a mount of its own, which takes a mount the test
cannot make for a command it starts (`test_tracked_keep_going_and_dry_run_keep_going_refuse_a_
cache_whose_store_is_its_own_mount` in tests/integration/test_all_pipelines.py fakes the mount
table in process).

The fixed seeds run in the normal test job. ``PIVOT_DIFFERENTIAL_SEEDS=N`` runs seeds 0 to N-1
instead, and ``PIVOT_DIFFERENTIAL_SEEDS=A:B`` seeds A to B-1, for a long run.
"""

from __future__ import annotations

import os
import pathlib
import random
import shutil
import stat
from typing import TYPE_CHECKING, Literal, NamedTuple

import pytest
import yaml

from pivot.storage import cache, listing, track
from tests import helpers
from tests.cli import keep_going_agreement

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Callable, Collection

_SEEDS_VARIABLE = "PIVOT_DIFFERENTIAL_SEEDS"

# Chosen so that, between them, they generate every kind of change, both run options, every
# kind of target and each way the restore of a deleted tracked path that --checkout-missing
# reaches meets each cache object, listing and directory it reads, a file, a member and a whole
# directory that it brings back among them
# (``test_dry_run_keep_going_fixed_seeds_cover_every_kind``), with the seeds that found the
# disagreements this test was written for
_FIXED_SEEDS = (
    *(4, 13, 14, 18, 39, 48, 52, 61, 76, 80, 106, 117, 193, 380, 522, 729),
    # A pulled producer whose directory a --no-commit dry run read
    *(435, 480, 1126),
    # A target hiding the instances it is mapped over
    2651,
    # A target naming a mapped stage not expanded, which hid a stage the run schedules upstream
    *(32, 211, 265),
    # A whole tracked directory the restore brings back, from its cached listing (8697) or its
    # committed listing (2747), and one whose cached listing's prefix directory this user cannot
    # search (1083)
    *(8697, 2747, 1083),
    # A chained stage mapped over another pipeline's chained stage, which pivot 0.10.4 registered
    # before the mapped stage it is mapped over and left not expanded
    817,
)

# The keys every producer writes, and the key a dropping producer wrote in the first run alone
_KEYS = ("a", "b")
_DROPPED_KEY = "z"

_REMOTE = "s3://bucket/differential"


class _Input(NamedTuple):
    """A raw input under data/raw/."""

    path: str
    """Project-relative: data/raw/f<i>.txt, or data/raw/d<i> for a directory of x.txt and y.txt."""
    directory: bool
    tracked: str
    """"no"; "whole": `pivot track` of the path; "member": of the directory's file x.txt."""


def _tracked_path(item: _Input) -> str:
    """The path `pivot track` tracks for input ``item`` (one that is tracked): the input itself,
    or, for a "member", its directory's file x.txt."""
    return f"{item.path}/x.txt" if item.tracked == "member" else item.path


_StageKind = Literal["plain", "producer", "mapped", "chained"]


class _Stage(NamedTuple):
    """A stage of a generated pipeline."""

    name: str
    """``<pipeline>/<function>``."""
    pipeline: str
    function: str
    kind: _StageKind
    """"plain" (one text output), "producer" (a directory output of one file per key),
    "mapped" (over a producer's files) or "chained" (over the instances of a mapped or chained
    stage)."""
    reads: tuple[str, ...]
    """Input paths, and names of plain stages of its pipeline or, for a stage of p0 composed
    across pipelines (``_compose_across``), of p1, that it reads."""
    over: str
    """The stage it is mapped over, of its pipeline or, composed across pipelines, of p1; "" for
    a plain stage or a producer."""
    params: bool
    """Whether it takes a StageParams, whose field ``k`` params.yaml can set."""


class _Project(NamedTuple):
    inputs: tuple[_Input, ...]
    stages: tuple[_Stage, ...]
    remote: bool
    """Whether a remote is configured: a stage whose outputs the cache lost is then one a pull
    brings up to date."""
    dropping: tuple[str, ...]
    """Producers that also wrote key z in the first run, and then ran alone without it, so the
    instances of key z are orphans."""


# Each kind of change a case makes after the first run (``_APPLY`` makes it)
_ChangeKind = Literal[
    "lose_outputs",
    "empty_cache",
    "delete_listing",
    "edit_listed",
    "delete_output",
    "track_output",
    "delete_lock",
    "delete_tracked",
    "delete_input",
    "change_input",
    "change_code",
    "change_params",
]


class _Change(NamedTuple):
    """One change made to the tree after the first run (``_APPLY``)."""

    kind: _ChangeKind
    subject: str
    """An input path, a stage name, an instance name, or "" for the whole cache."""
    detail: dict[str, str]
    """How the change is made, by what it changes: for `delete_tracked`, what happens to each
    cache object the restore reads (``_OBJECTS``), to the committed listing and to the
    directory holding the path (``_tracked_detail``); for `lose_outputs`, what else the stage
    loses (``_lost_detail``); for `edit_listed`, the edit; for any other, nothing."""


class _Case(NamedTuple):
    seed: int
    project: _Project
    changes: tuple[_Change, ...]
    options: tuple[str, ...]
    targets: tuple[str, ...]


def _seeds() -> list[int]:
    """The fixed seeds, or the seeds ``PIVOT_DIFFERENTIAL_SEEDS`` names (``N`` or ``A:B``)."""
    value = os.environ.get(_SEEDS_VARIABLE)
    if value is None:
        return list(_FIXED_SEEDS)
    start, colon, stop = value.partition(":")
    return list(range(int(start), int(stop))) if colon else list(range(int(value)))


# =============================================================================
# The generator
# =============================================================================


def _generate(seed: int, *, locked_out: bool) -> _Case:
    """The case of ``seed``. ``locked_out``: whether a mode can lock this user out of a path
    (not root), so a change may make a cache object or directory unreadable or a directory
    read-only."""
    rng = random.Random(seed)
    inputs = [
        _Input(f"data/raw/f{index}.txt", False, rng.choice(("no", "whole")))
        for index in range(rng.randint(1, 3))
    ]
    inputs += [
        _Input(f"data/raw/d{index}", True, rng.choice(("no", "whole", "member")))
        for index in range(rng.randint(0, 2))
    ]
    paths = [item.path for item in inputs]
    stages = list[_Stage]()
    for pipeline in (f"p{index}" for index in range(rng.randint(1, 2))):
        chain = rng.choice((0, 2, 3, 3))
        plain = list[str]()
        for index in range(rng.randint(2, 3) if chain == 0 else rng.randint(0, 2)):
            candidates = [*paths, *plain]
            name = f"{pipeline}/s{index}"
            reads = tuple(rng.sample(candidates, rng.randint(1, min(2, len(candidates)))))
            stages.append(
                _Stage(name, pipeline, f"s{index}", "plain", reads, "", rng.random() < 0.3)
            )
            plain.append(name)
        candidates = [*paths, *plain]
        functions = ("plan0", "cut0", "build0")[:chain]
        over = ""
        for function, kind in zip(functions, ("producer", "mapped", "chained"), strict=False):
            reads = tuple(rng.sample(candidates, rng.randint(0, 1))) if kind != "chained" else ()
            name = f"{pipeline}/{function}"
            stages.append(_Stage(name, pipeline, function, kind, reads, over, False))
            over = name
    stages = _compose_across(random.Random(f"{seed}:across"), stages)
    producers = [item.name for item in stages if item.kind == "producer"]
    project = _Project(
        inputs=tuple(inputs),
        stages=tuple(stages),
        remote=rng.random() < 0.5,
        dropping=tuple(name for name in producers if rng.random() < 0.25),
    )
    changes = _generate_changes(rng, project, locked_out=locked_out)
    options = tuple(
        option
        for option, chance in (("--no-commit", 0.4), ("--checkout-missing", 0.5))
        if rng.random() < chance
    )
    targets = list[str]()
    if rng.random() < 0.5:
        named = [
            *(item.name for item in stages),
            *(f"{item.name}@{key}" for item in stages if item.over for key in _KEYS),
        ]
        targets = rng.sample(named, rng.randint(1, min(2, len(named))))
    return _Case(seed, project, changes, options, tuple(targets))


def _compose_across(rng: random.Random, stages: list[_Stage]) -> list[_Stage]:
    """``stages``, or, for half the cases of two pipelines, ``stages`` with pipeline p0 composed
    from p1's handles, which its pipeline.py imports: each plain stage of p0 may also read a
    plain stage of p1, and p0's chained stage is mapped over the instances of a mapped or
    chained stage of p1 instead of p0's mapped stage. ``rng`` is this alone's, and a stage keeps
    its kind, so a case composed of no cross-pipeline read is the one its seed gave before these
    were generated. With p0's chained stage over p1's chained stage, pivot 0.10.4 registered
    p1's chained stage before the mapped stage it is mapped over, and so never expanded it
    outside a run."""
    if len({item.pipeline for item in stages}) == 1 or rng.random() < 0.5:
        return stages
    plain = [item.name for item in stages if item.pipeline == "p1" and item.kind == "plain"]
    mapped = [item.name for item in stages if item.pipeline == "p1" and item.over]
    composed = list[_Stage]()
    for item in stages:
        if item.pipeline == "p0" and item.kind == "plain" and plain and rng.random() < 0.5:
            composed.append(item._replace(reads=(*item.reads, rng.choice(plain))))
        elif item.pipeline == "p0" and item.kind == "chained" and mapped:
            composed.append(item._replace(over=rng.choice(mapped)))
        else:
            composed.append(item)
    return composed


# How likely each subject is to be changed, by its kind: a producer and a mapped stage have the
# most ways to go wrong, the whole cache the fewest
_WEIGHTS = {"input": 1.0, "plain": 1.0, "producer": 3.0, "mapped": 2.0, "chained": 2.0}


def _generate_changes(
    rng: random.Random, project: _Project, *, locked_out: bool
) -> tuple[_Change, ...]:
    """One to three changes, each to its own input, its own stage or the whole cache, a
    subject drawn by its weight (``_WEIGHTS``)."""
    choices = list[tuple[float, list[_Change]]]()
    for item in project.inputs:
        if item.tracked == "no":
            choices.append(
                (
                    _WEIGHTS["input"],
                    [
                        _Change("delete_input", item.path, {}),
                        _Change("change_input", item.path, {}),
                    ],
                )
            )
            continue
        path = _tracked_path(item)
        detail = _tracked_detail(rng, directory=item.directory and item.tracked == "whole")
        if not locked_out:
            # Root reads and writes whatever the modes say
            detail = {
                key: "kept"
                if key in _OBJECTS and value in ("unreadable", "unsearchable")
                else value
                for key, value in detail.items()
            } | {"parent": "writable"}
        choices.append((_WEIGHTS["input"], [_Change("delete_tracked", path, detail)]))
    for item in project.stages:
        options = [_Change("change_code", item.name, {})]
        if item.params:
            options.append(_Change("change_params", item.name, {}))
        if item.kind in ("plain", "producer"):
            producer = item.kind == "producer"
            options += [
                _Change("delete_output", item.name, {}),
                _Change("delete_lock", item.name, {}),
                # A stage whose cache objects are gone, a pull-only one with a remote, twice
                _Change("lose_outputs", item.name, _lost_detail(rng, producer=producer)),
                _Change("lose_outputs", item.name, _lost_detail(rng, producer=producer)),
            ]
        if item.kind == "producer":
            edit = rng.choice(("add c.txt", "add c d.txt", "remove a.txt"))
            options += [
                _Change("edit_listed", item.name, {"edit": edit}),
                _Change("delete_listing", item.name, {}),
            ]
        if item.over:
            instance = f"{item.name}@{rng.choice(_KEYS)}"
            options += [
                _Change("delete_output", instance, {}),
                _Change("delete_lock", instance, {}),
                _Change("lose_outputs", instance, {"listing": "kept", "workspace": "kept"}),
                _Change("track_output", instance, {}),
                _Change("track_output", instance, {}),
            ]
        choices.append((_WEIGHTS[item.kind], options))
    choices.append((0.5, [_Change("empty_cache", "", {})]))
    count = rng.randint(1, min(3, len(choices)))
    # Drawn without replacement by weight: the largest of random() ** (1 / weight)
    ranked = sorted(choices, key=lambda choice: rng.random() ** (1 / choice[0]), reverse=True)
    picked = [rng.choice(options) for _, options in ranked[:count]]
    if any(change.kind == "empty_cache" for change in picked):
        # Every object is gone, so a tracked path's restore finds none, locked out or not
        picked = [
            change._replace(
                detail={
                    key: "removed" if key in _OBJECTS else value
                    for key, value in change.detail.items()
                }
            )
            if change.kind == "delete_tracked"
            else change
            for change in picked
        ]
    return tuple(sorted(picked, key=lambda change: _ORDER.index(change.kind)))


# The keys of a `delete_tracked` detail that each say what happens to a cache object the
# restore reads (``_OBJECT_FATES``)
_OBJECTS = ("object", "listing", "members")

# What deleting a tracked path does to a cache object its restore reads: kept, removed,
# unreadable, or in a prefix directory this user cannot search
_OBJECT_FATES = ("kept", "removed", "unreadable", "unsearchable")
# The same for a tracked directory's member objects: each but a prefix directory locked
_MEMBER_FATES = _OBJECT_FATES[:3]
# What it does to a tracked directory's committed listing
_COMMITTED = ("intact", "stale", "empty", "missing")
# What it does to the directory that holds the tracked path
_PARENTS = ("writable", "read_only")


def _tracked_detail(rng: random.Random, *, directory: bool) -> dict[str, str]:
    """What deleting a tracked path does to what its restore reads: its object (a file), or its
    cached listing object, committed listing and member objects (a directory), each as
    ``_OBJECT_FATES``, ``_COMMITTED`` and ``_MEMBER_FATES`` say (``_OBJECTS``); and whether the
    directory that holds it is left writable or made read-only (``_PARENTS``)."""
    parent = rng.choice(_PARENTS)
    if not directory:
        return {"object": rng.choice(_OBJECT_FATES), "parent": parent}
    return {
        "listing": rng.choice(_OBJECT_FATES),
        "committed": rng.choice(_COMMITTED),
        "members": rng.choice(_MEMBER_FATES),
        "parent": parent,
    }


def _lost_detail(rng: random.Random, *, producer: bool) -> dict[str, str]:
    """What else a stage that lost its cache objects lost: its committed listing (a producer),
    and its output in the workspace, removed or, for a producer, edited."""
    workspace = ("kept", "removed")
    if not producer:
        return {"listing": "kept", "workspace": rng.choice(workspace)}
    edits = (*workspace, "add c.txt", "add c d.txt", "remove a.txt")
    return {"listing": rng.choice(("kept", "removed")), "workspace": rng.choice(edits)}


# =============================================================================
# Building the tree
# =============================================================================


def _pipeline_code(
    project: _Project, pipeline: str, *, changed: Collection[str], first_run: bool
) -> str:
    """Pipeline ``pipeline``'s pipeline.py: each stage's text names its stage, its version (1 for
    a stage in ``changed``, else 0) and what it read. In the ``first_run`` a dropping producer
    writes key z too. A stage of another pipeline that it reads or is mapped over is read
    through the handle that pipeline's pipeline.py names it by, imported."""
    directories = {item.path for item in project.inputs if item.directory}
    own = [item for item in project.stages if item.pipeline == pipeline]
    # The import of the handle of each stage of another pipeline read, by the stage's name
    imports = dict[str, str]()

    def _handle(name: str) -> str:
        other, function = _stage_parts(name)
        if other == pipeline:
            return f"h_{function}"
        imports[name] = f"from {other}.pipeline import h_{function} as {other}_{function}"
        return f"{other}_{function}"

    lines = list[str]()
    calls = list[str]()
    inputs = dict[str, str]()
    for item in own:
        handles = list[str]()
        for read in item.reads:
            if read.startswith("data/raw/"):
                variable = inputs.setdefault(read, f"i{len(inputs)}")
                handles.append(variable)
            else:
                handles.append(_handle(read))
        arguments = [
            f"a{index}: {'pathlib.Path' if read in directories else 'str'}"
            for index, read in enumerate(item.reads)
        ]
        texts = [
            "\",\".join(sorted(e.name + '=' + e.read_text() for e in " + f"a{index}.iterdir()))"
            if read in directories
            else f"a{index}"
            for index, read in enumerate(item.reads)
        ]
        version = 1 if item.name in changed else 0
        label = f'"{item.name}:v{version}"'
        read_text = " + '|' + ".join([label, *texts])
        lines += ["", ""]
        if item.params:
            lines += [
                f"class {item.function.capitalize()}Params(StageParams):",
                "    k: int = 0",
                "",
                "",
            ]
            arguments.append(f"params: {item.function.capitalize()}Params")
            read_text += " + '|' + str(params.k)"
            handles.append(f"params={item.function.capitalize()}Params()")
        lines.append("@stage")
        if item.kind == "plain":
            lines += [
                f"def {item.function}({', '.join(arguments)}) -> str:",
                f"    return {read_text}",
            ]
        elif item.kind == "producer":
            keys = (*_KEYS, _DROPPED_KEY) if first_run and item.name in project.dropping else _KEYS
            files = ", ".join(f'"{key}.txt": "{key}|" + {read_text}' for key in keys)
            lines += [
                f"def {item.function}({', '.join(arguments)}) -> Annotated[dict[str, str], directory]:",
                f"    return {{{files}}}",
            ]
        else:
            lines += [
                f"def {item.function}({', '.join(['entry: str', *arguments])}) -> str:",
                f"    return entry + '|' + {read_text}",
            ]
            over = _handle(item.over)
            handles.insert(0, f"pipeline.map({over})" if item.kind == "mapped" else over)
        calls.append(f"    h_{item.function} = {item.function}({', '.join(handles)})")
    lines += ["", "", f'with Pipeline("{pipeline}") as pipeline:']
    lines += [
        f'    {variable} = pipeline.input("{path.removeprefix("data/raw/")}", '
        + f"t={'pathlib.Path' if path in directories else 'str'})"
        for path, variable in inputs.items()
    ]
    lines += calls
    header = [
        "import pathlib",
        "from typing import Annotated",
        "",
        "from pivot import Pipeline, StageParams, directory, stage",
        *imports.values(),
        "",
    ]
    return "\n".join([*header, *lines]) + "\n"


def _write_pipelines(
    root: pathlib.Path, project: _Project, *, changed: Collection[str] = (), first_run: bool = False
) -> None:
    for pipeline in sorted({item.pipeline for item in project.stages}):
        path = root / pipeline / "pipeline.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(_pipeline_code(project, pipeline, changed=changed, first_run=first_run))


def _build(case: _Case, root: pathlib.Path, home: pathlib.Path) -> None:
    """``case``'s project at ``root`` (a copy of an initialized one), after a first full run."""
    project = case.project
    for item in project.inputs:
        path = root / item.path
        if item.directory:
            path.mkdir(parents=True)
            for name in ("x.txt", "y.txt"):
                (path / name).write_text(f"{item.path}/{name}\n")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{item.path}\n")
    _write_pipelines(root, project, first_run=True)
    if project.remote:
        (root / ".pivot" / "config.yaml").write_text(
            yaml.safe_dump({"remotes": {"origin": _REMOTE}})
        )
    tracked = [_tracked_path(item) for item in project.inputs if item.tracked != "no"]
    if tracked:
        _ok(helpers.run_pivot(root, home, "track", *tracked), case)
    _ok(helpers.run_pivot(root, home, "repro", "--all"), case)
    if project.dropping:
        _write_pipelines(root, project)
        _ok(helpers.run_pivot(root, home, "repro", "--all", *project.dropping), case)


def _copy_tree(source: pathlib.Path, target: pathlib.Path) -> None:
    """Copy ``source`` to ``target`` as the same tree: modes and times kept, each set of hard
    links inside it linked again (a checked-out output shares its cache object's inode), and
    each symlink copied as a link, pointing into ``target`` where it pointed into ``source``."""
    linked = dict[tuple[int, int], pathlib.Path]()
    directories = list[tuple[pathlib.Path, pathlib.Path]]()
    for top, names, files in os.walk(source):
        here = pathlib.Path(top)
        there = target / here.relative_to(source)
        there.mkdir()
        directories.append((here, there))
        for name in [*names, *files]:
            entry = here / name
            info = entry.lstat()
            if stat.S_ISLNK(info.st_mode):
                pointed = pathlib.Path(os.readlink(entry))
                if pointed.is_absolute() and pointed.is_relative_to(source):
                    pointed = target / pointed.relative_to(source)
                (there / name).symlink_to(pointed)
            elif stat.S_ISREG(info.st_mode):
                identity = (info.st_dev, info.st_ino)
                if identity in linked:
                    (there / name).hardlink_to(linked[identity])
                else:
                    shutil.copy2(entry, there / name)
                    linked[identity] = there / name
            else:
                assert stat.S_ISDIR(info.st_mode), f"{entry} is no file, directory or symlink"
        # A symlink to a directory is copied as a link above, not walked
        names[:] = [name for name in names if not (here / name).is_symlink()]
    for here, there in reversed(directories):
        shutil.copystat(here, there)


def _object(root: pathlib.Path, digest: str) -> pathlib.Path:
    return cache.get_cache_path(root / ".pivot" / "cache" / "files", digest)


def _stage_parts(name: str) -> tuple[str, str]:
    """(pipeline, function[@key]) of a stage or instance name."""
    pipeline, own = name.split("/", 1)
    return pipeline, own


def _output_path(root: pathlib.Path, project: _Project, name: str) -> pathlib.Path:
    """The workspace output of stage or instance ``name``."""
    pipeline, own = _stage_parts(name)
    producer = any(item.name == name and item.kind == "producer" for item in project.stages)
    return root / "data" / pipeline / (own if producer else f"{own}.txt")


def _lock_path(root: pathlib.Path, name: str) -> pathlib.Path:
    pipeline, own = _stage_parts(name)
    return root / ".pivot" / "stages" / pipeline / f"{own}.lock"


def _listing_path(root: pathlib.Path, name: str) -> pathlib.Path:
    pipeline, own = _stage_parts(name)
    return root / ".pivot" / "stages" / pipeline / f"{own}=.listing"


def _edit_listed(directory: pathlib.Path, edit: str) -> None:
    """Edit a producer's directory in the workspace: ``add <file>`` or ``remove <file>``."""
    verb, name = edit.split(" ", 1)
    if verb == "add":
        (directory / name).write_text(name)
    else:
        (directory / name).unlink()


# What a change made by ``_APPLY`` leaves for ``_prepare`` to do last: each (path, mode) to set
type _Modes = list[tuple[pathlib.Path, int]]


def _apply_lose_outputs(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    """Remove from the cache every output object stage or instance ``change.subject``'s lock
    names (with a remote, a pull brings them back), and its committed listing or workspace
    output as ``change.detail`` says (``_lost_detail``)."""
    name, details = change.subject, change.detail
    recorded = yaml.safe_load(_lock_path(root, name).read_text())
    # A lock that records a directory, read or written, keeps every entry under `records`
    outs = recorded["records"]["outs"] if "records" in recorded else recorded["outs"]
    for out in outs:
        if "listing" not in out:
            _object(root, out["hash"]).unlink()
            continue
        committed = _listing_path(root, name)
        for member in listing.decode(committed.read_bytes())["files"]:
            _object(root, member["hash"]).unlink()
        _object(root, out["listing"]).unlink()
        if details["listing"] == "removed":
            committed.unlink()
    output = _output_path(root, case.project, name)
    if details["workspace"] == "removed":
        cache.remove_output(output)
    elif details["workspace"] != "kept":
        _edit_listed(output, details["workspace"])
    return []


def _apply_empty_cache(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    for entry in (root / ".pivot" / "cache" / "files").iterdir():
        shutil.rmtree(entry)
    return []


def _apply_delete_listing(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    _listing_path(root, change.subject).unlink()
    return []


def _apply_edit_listed(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    _edit_listed(_output_path(root, case.project, change.subject), change.detail["edit"])
    return []


def _apply_delete_output(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    cache.remove_output(_output_path(root, case.project, change.subject))
    return []


def _apply_track_output(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    output = _output_path(root, case.project, change.subject)
    cache.remove_output(output)
    helpers.track_without_data(output)
    return []


def _apply_delete_lock(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    _lock_path(root, change.subject).unlink()
    return []


def _apply_delete_tracked(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    """Delete tracked path ``change.subject`` from the workspace and change what its restore
    reads, as ``change.detail`` says (``_tracked_detail``)."""
    details = change.detail
    path = root / change.subject
    pvt = track.get_pvt_path(path)
    record = track.read_pvt_file(pvt)
    assert record is not None, pvt
    modes = list[tuple[pathlib.Path, int]]()
    if "object" in details:
        reads = {"object": _object(root, record["hash"])}
    else:
        committed = listing.tracked_listing_path(pvt)
        members = listing.decode(committed.read_bytes())["files"]
        reads = {"listing": _object(root, record["hash"])}
        if details["members"] == "removed":
            for member in members:
                # After `empty_cache`, every object is gone already
                _object(root, member["hash"]).unlink(missing_ok=True)
        else:
            reads["members"] = _object(root, members[0]["hash"])
        if details["committed"] == "stale":
            committed.write_bytes(committed.read_bytes().replace(b"x.txt", b"w.txt"))
        elif details["committed"] == "empty":
            committed.write_bytes(b"")
        elif details["committed"] == "missing":
            committed.unlink()
    cache.remove_output(path)
    for part, found in reads.items():
        if details[part] == "removed":
            # After `empty_cache`, every object is gone already
            found.unlink(missing_ok=True)
        elif details[part] == "unreadable":
            modes.append((found, 0))
        elif details[part] == "unsearchable":
            modes.append((found.parent, 0))
    if details["parent"] == "read_only":
        modes.append((path.parent, 0o555))
    return modes


def _apply_delete_input(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    cache.remove_output(root / change.subject)
    return []


def _apply_change_input(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    path = root / change.subject
    if path.is_dir():
        (path / "z.txt").write_text("z")
    else:
        path.unlink()
        path.write_text(f"{change.subject} changed\n")
    return []


def _apply_change_code(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    # Each pipeline holds every stage whose code the case changes, whichever change writes it
    changed = [other.subject for other in case.changes if other.kind == "change_code"]
    _write_pipelines(root, case.project, changed=changed)
    return []


def _apply_change_params(change: _Change, root: pathlib.Path, case: _Case) -> _Modes:
    params = root / "params.yaml"
    values = yaml.safe_load(params.read_text()) if params.exists() else {}
    values[change.subject] = {"k": 1}
    params.write_text(yaml.safe_dump(values))
    return []


# How ``_prepare`` makes each kind of change in the tree at a root, but for the modes it sets,
# which it returns; in the order it makes them, each reading what the ones before it leave
_APPLY: dict[_ChangeKind, Callable[[_Change, pathlib.Path, _Case], _Modes]] = {
    "lose_outputs": _apply_lose_outputs,
    "empty_cache": _apply_empty_cache,
    "delete_listing": _apply_delete_listing,
    "edit_listed": _apply_edit_listed,
    "delete_output": _apply_delete_output,
    "track_output": _apply_track_output,
    "delete_lock": _apply_delete_lock,
    "delete_tracked": _apply_delete_tracked,
    "delete_input": _apply_delete_input,
    "change_input": _apply_change_input,
    "change_code": _apply_change_code,
    "change_params": _apply_change_params,
}
_ORDER = tuple(_APPLY)


def _prepare(case: _Case, base: pathlib.Path, root: pathlib.Path) -> None:
    """A copy of ``base`` at ``root`` with ``case``'s changes made; the modes they set last, so
    no change meets a path another one locked."""
    _copy_tree(base, root)
    modes = list[tuple[pathlib.Path, int]]()
    for change in case.changes:
        modes += _APPLY[change.kind](change, root, case)
    for path, mode in modes:
        path.chmod(mode)


def _limited(case: _Case) -> set[str]:
    """The stages whose explanation the dry run takes from the files on disk where the run's
    restore changes them (spec R5's limit, written in docs/cli/index.md): with
    --checkout-missing, each stage that reads a directory holding a deleted tracked file, and
    every stage downstream of one. The dry run lists them, as the run does, but its ``would_run``
    and reason for them need not be the run's."""
    if "--checkout-missing" not in case.options:
        return set[str]()
    deleted = {change.subject for change in case.changes if change.kind == "delete_tracked"}
    holding = {
        item.path
        for item in case.project.inputs
        if item.tracked == "member" and _tracked_path(item) in deleted
    }
    limited = {item.name for item in case.project.stages if holding.intersection(item.reads)}
    grew = True
    while grew:
        grew = False
        for item in case.project.stages:
            if item.name not in limited and (
                limited.intersection(item.reads) or item.over in limited
            ):
                limited.add(item.name)
                grew = True
    return limited


# =============================================================================
# The tests
# =============================================================================


def _ok(proc: subprocess.CompletedProcess[str], case: _Case) -> None:
    assert proc.returncode == 0, f"seed {case.seed}: {proc.args}\n{proc.stdout}{proc.stderr}"


@pytest.fixture(scope="module")
def initialized_project(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pathlib.Path, pathlib.Path]:
    """An empty pivot project in a git repository (``helpers.git_project``), made once, and the
    home its commands run with; each case starts from a copy of it."""
    return helpers.git_project(tmp_path_factory.mktemp("initialized"))


def _report(
    case: _Case, root: pathlib.Path, found: list[str], *procs: subprocess.CompletedProcess[str]
) -> str:
    """What a failure prints: the seed, how to run it again, each disagreement, the case, its
    pipelines as the commands ran them, and the commands' output."""
    pipelines = "".join(
        f"--- {path.relative_to(root)}\n{path.read_text()}"
        for path in sorted(root.glob("p*/pipeline.py"))
    )
    outputs = "".join(
        f"--- {' '.join(proc.args[1:])} (exit {proc.returncode})\n{proc.stdout}{proc.stderr}"
        for proc in procs
    )
    uid = os.geteuid()
    user = "root" if uid == 0 else "a user who is not root"
    return (
        f"seed {case.seed}: {len(found)} disagreements; run it again with "
        + f"{_SEEDS_VARIABLE}={case.seed}:{case.seed + 1} as {user} (this run's uid is {uid}): "
        + "for root a seed draws the same case but locks no path, so the case printed below, "
        + "its details, holds for this kind of user alone\n"
        + "".join(f"  - {problem}\n" for problem in found)
        + f"{case}\n{pipelines}{outputs}"
    )


@pytest.mark.parametrize("seed", _seeds())
def test_dry_run_keep_going_agrees_with_the_run_on_generated_projects(
    seed: int,
    tmp_path: pathlib.Path,
    initialized_project: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """R5 on a generated project: the leave-out dry run of a changed tree lists the stages the
    run on a copy of the same tree runs, skips, fails or blocks, as
    ``keep_going_agreement.disagreements`` reads them."""
    case = _generate(seed, locked_out=os.geteuid() != 0)
    initialized, home = initialized_project
    base = tmp_path / "base"
    _copy_tree(initialized, base)
    _build(case, base, home)
    roots = (tmp_path / "dry", tmp_path / "run")
    for root in roots:
        _prepare(case, base, root)

    dry = helpers.run_pivot(
        roots[0], home, *helpers.DRY_RUN_KEEP_GOING, *case.options, *case.targets
    )
    ran, events = helpers.run_keep_going(roots[1], home, *case.options, *case.targets)

    found = keep_going_agreement.disagreements(
        dry, ran, events, roots=roots, limited=_limited(case)
    )
    assert not found, _report(case, roots[0], found, dry, ran)


def _target_kind(case: _Case, target: str) -> str:
    if "@" in target:
        return "instance"
    return next(item.kind for item in case.project.stages if item.name == target)


def _restores(case: _Case) -> bool:
    """Whether ``case``'s commands restore a tracked path, or say why not, from a cache that
    holds objects: they run with --checkout-missing, and no change empties the cache. Without
    it neither reads a tracked path's object or listing."""
    return "--checkout-missing" in case.options and all(
        change.kind != "empty_cache" for change in case.changes
    )


def _restored_read(case: _Case, change: _Change) -> str | None:
    """What a stage of ``case`` reads that the restore of `delete_tracked` ``change`` brings
    back, as ``_COVERED`` names it, or None. The directory that holds the tracked path must be
    writable and a stage must read the input: a file, whose object is kept; a member of a
    directory (``_Input.tracked`` "member"), whose object is kept, read through that directory;
    or a whole directory, whose member objects are kept and whose listing the restore reads: its
    cached listing object is kept, or removed beside an intact committed listing."""
    item = next(
        item
        for item in case.project.inputs
        if item.tracked != "no" and _tracked_path(item) == change.subject
    )
    detail = change.detail
    if detail["parent"] != "writable" or not any(
        item.path in stage.reads for stage in case.project.stages
    ):
        return None
    if item.tracked == "member":
        read = "a restorable member read through its directory"
        restorable = detail["object"] == "kept"
    elif item.directory:
        read = "a restorable directory"
        restorable = detail["members"] == "kept" and (
            detail["listing"] == "kept"
            or (detail["listing"] == "removed" and detail["committed"] == "intact")
        )
    else:
        read = "a restorable file"
        restorable = detail["object"] == "kept"
    return read if restorable else None


def _across(case: _Case) -> set[str]:
    """What ``case`` composes across pipelines (``_compose_across``), as ``_COVERED`` names it."""
    kinds = {item.name: item.kind for item in case.project.stages}
    found = set[str]()
    for item in case.project.stages:
        if any(read in kinds and _stage_parts(read)[0] != item.pipeline for read in item.reads):
            found.add(f"a {item.kind} stage reading another pipeline's")
        if item.over and _stage_parts(item.over)[0] != item.pipeline:
            found.add(f"a {item.kind} stage over another pipeline's {kinds[item.over]} stage")
    return found


def test_dry_run_keep_going_fixed_seeds_cover_every_kind() -> None:
    """Between them, the fixed seeds generate every kind of change, each run option, a target
    naming each kind of stage and an instance, two pipelines, one reading the other's stages
    and mapped over its mapped and chained stages (``_across``), each kind of tracked input, a
    configured remote and a producer whose key left its listing (for a user who is not root,
    as CI's; root gets the same cases but for the modes). What a deleted tracked path's restore
    meets counts only in a case that restores it (``_restores``), so each detail of it is
    compared as the restore reads it: among them a file, a member of a directory and a whole
    directory that the restore brings back for the stage reading it (``_restored_read``)."""
    cases = [_generate(seed, locked_out=True) for seed in _FIXED_SEEDS]
    tracked = [
        (case, change)
        for case in cases
        if _restores(case)
        for change in case.changes
        if change.kind == "delete_tracked"
    ]

    found = {
        *(f"change {change.kind}" for case in cases for change in case.changes),
        *(
            f"{change.kind} {key}={value}"
            for _, change in tracked
            for key, value in change.detail.items()
        ),
        *(read for case, change in tracked if (read := _restored_read(case, change)) is not None),
        *(f"option {option}" for case in cases for option in case.options),
        *(f"target {_target_kind(case, target)}" for case in cases for target in case.targets),
        *(
            f"input {'directory' if item.directory else 'file'} {item.tracked}"
            for case in cases
            for item in case.project.inputs
        ),
        *(
            "two pipelines"
            for case in cases
            if len({item.pipeline for item in case.project.stages}) == 2
        ),
        *(composed for case in cases for composed in _across(case)),
        *("remote" for case in cases if case.project.remote),
        *("dropping" for case in cases if case.project.dropping),
    }

    assert found.issuperset(_COVERED), sorted(_COVERED - found)


# What the fixed seeds generate between them (``test_dry_run_keep_going_fixed_seeds_cover_every_kind``)
_COVERED = {
    *(f"change {kind}" for kind in _ORDER),
    *(f"delete_tracked {key}={value}" for key in ("object", "listing") for value in _OBJECT_FATES),
    *(f"delete_tracked members={value}" for value in _MEMBER_FATES),
    *(f"delete_tracked committed={value}" for value in _COMMITTED),
    *(f"delete_tracked parent={value}" for value in _PARENTS),
    "a restorable file",
    "a restorable member read through its directory",
    "a restorable directory",
    "option --no-commit",
    "option --checkout-missing",
    *(f"target {kind}" for kind in ("plain", "producer", "mapped", "chained", "instance")),
    "input file whole",
    "input directory whole",
    "input directory member",
    "two pipelines",
    "a plain stage reading another pipeline's",
    "a chained stage over another pipeline's mapped stage",
    "a chained stage over another pipeline's chained stage",
    "remote",
    "dropping",
}
