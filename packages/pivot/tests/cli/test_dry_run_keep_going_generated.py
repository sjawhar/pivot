"""A differential test of `pivot repro --all --keep-going` (leave-out mode): on small projects
generated at random, the dry run (`--dry-run --jsonl`) on one copy of a tree and the run
(`--jsonl`) on another copy of it must agree, as spec R5 says (``_disagreements``).

A seed gives one case (``_generate``): a project of one or two pipelines of plain stages,
directory producers and mapped stages over them (``Pipeline.map``, and chains of them), reading
raw files and directories, some tracked with `pivot track`; the changes made after a first full
run (``_Change``); and the run's options and targets. A failure names the seed and prints the
case and its pipelines, and the seed alone builds the same case again. Not generated: a cache
whose object store is a mount of its own, which takes a mount the test cannot make for a
command it starts (`test_tracked_keep_going_and_dry_run_keep_going_refuse_a_cache_whose_store_
is_its_own_mount` in tests/integration/test_all_pipelines.py fakes the mount table in process).

The fixed seeds run in the normal test job. ``PIVOT_DIFFERENTIAL_SEEDS=N`` runs seeds 0 to N-1
instead, and ``PIVOT_DIFFERENTIAL_SEEDS=A:B`` seeds A to B-1, for a long run.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import re
import shutil
import stat
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest
import yaml

from pivot import registry, status
from pivot.storage import cache, listing, track
from tests import helpers

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Collection, Sequence

_SEEDS_VARIABLE = "PIVOT_DIFFERENTIAL_SEEDS"

# Chosen so that, between them, they generate every kind of change, both run options and every
# kind of target (``test_dry_run_keep_going_fixed_seeds_cover_every_kind``), with seeds that
# found two disagreements this test was written for: 435, 480 and 1126 a pulled producer whose
# directory a --no-commit dry run read, 2651 a target hiding the instances it is mapped over
_FIXED_SEEDS = (4, 13, 18, 39, 48, 52, 61, 80, 106, 117, 193, 435, 480, 1126, 2651)

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


class _Stage(NamedTuple):
    """A stage of a generated pipeline."""

    name: str
    """``<pipeline>/<function>``."""
    pipeline: str
    function: str
    kind: str
    """"plain" (one text output), "producer" (a directory output of one file per key),
    "mapped" (over a producer's files) or "chained" (over a mapped stage's instances)."""
    reads: tuple[str, ...]
    """Input paths, and names of plain stages of its pipeline, that it reads."""
    over: str
    """The stage it is mapped over; "" for a plain stage or a producer."""
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


class _Change(NamedTuple):
    """One change made to the tree after the first run (``_apply``)."""

    kind: str
    subject: str
    """An input path, a stage name, an instance name, or "" for the whole cache."""
    detail: str


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
                        _Change("delete_input", item.path, ""),
                        _Change("change_input", item.path, ""),
                    ],
                )
            )
            continue
        path = f"{item.path}/x.txt" if item.tracked == "member" else item.path
        detail = _tracked_detail(rng, directory=item.directory and item.tracked == "whole")
        if not locked_out:
            # Root reads and writes whatever the modes say
            detail = re.sub(r"=(unreadable|unsearchable)", "=kept", detail)
            detail = detail.replace("parent=read_only", "parent=writable")
        choices.append((_WEIGHTS["input"], [_Change("delete_tracked", path, detail)]))
    for item in project.stages:
        options = [_Change("change_code", item.name, "")]
        if item.params:
            options.append(_Change("change_params", item.name, ""))
        if item.kind in ("plain", "producer"):
            producer = item.kind == "producer"
            options += [
                _Change("delete_output", item.name, ""),
                _Change("delete_lock", item.name, ""),
                # A stage whose cache objects are gone, a pull-only one with a remote, twice
                _Change("lose_outputs", item.name, _lost_detail(rng, producer=producer)),
                _Change("lose_outputs", item.name, _lost_detail(rng, producer=producer)),
            ]
        if item.kind == "producer":
            edit = rng.choice(("add c.txt", "add c d.txt", "remove a.txt"))
            options += [
                _Change("edit_listed", item.name, edit),
                _Change("delete_listing", item.name, ""),
            ]
        if item.over:
            instance = f"{item.name}@{rng.choice(_KEYS)}"
            options += [
                _Change("delete_output", instance, ""),
                _Change("delete_lock", instance, ""),
                _Change("lose_outputs", instance, "listing=kept;workspace=kept"),
                _Change("track_output", instance, ""),
                _Change("track_output", instance, ""),
            ]
        choices.append((_WEIGHTS[item.kind], options))
    choices.append((0.5, [_Change("empty_cache", "", "")]))
    count = rng.randint(1, min(3, len(choices)))
    # Drawn without replacement by weight: the largest of random() ** (1 / weight)
    ranked = sorted(choices, key=lambda choice: rng.random() ** (1 / choice[0]), reverse=True)
    picked = [rng.choice(options) for _, options in ranked[:count]]
    if any(change.kind == "empty_cache" for change in picked):
        # Every object is gone, so a tracked path's restore finds none, locked out or not
        picked = [
            change._replace(
                detail=re.sub(r"=(kept|unreadable|unsearchable)", "=removed", change.detail)
            )
            if change.kind == "delete_tracked"
            else change
            for change in picked
        ]
    return tuple(sorted(picked, key=lambda change: _ORDER.index(change.kind)))


# The order ``_apply`` makes changes in: each reads what the ones before it leave
_ORDER = (
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
)


def _tracked_detail(rng: random.Random, *, directory: bool) -> str:
    """What deleting a tracked path does to what its restore reads, as ``key=value;...``: its
    object (a file), or its cached listing object, committed listing and member objects (a
    directory), each kept, removed, unreadable or in a prefix directory this user cannot search;
    and whether the directory that holds it is read-only."""
    parent = rng.choice(("writable", "read_only"))
    objects = ("kept", "removed", "unreadable", "unsearchable")
    if not directory:
        return f"object={rng.choice(objects)};parent={parent}"
    return (
        f"listing={rng.choice(objects)};"
        + f"committed={rng.choice(('intact', 'stale', 'empty', 'missing'))};"
        + f"members={rng.choice(objects[:3])};parent={parent}"
    )


def _lost_detail(rng: random.Random, *, producer: bool) -> str:
    """What else a stage that lost its cache objects lost: its committed listing (a producer),
    and its output in the workspace, removed or, for a producer, edited."""
    workspace = ("kept", "removed")
    if not producer:
        return f"listing=kept;workspace={rng.choice(workspace)}"
    edits = (*workspace, "add c.txt", "add c d.txt", "remove a.txt")
    return f"listing={rng.choice(('kept', 'removed'))};workspace={rng.choice(edits)}"


def _details(change: _Change) -> dict[str, str]:
    """A change's ``key=value;...`` detail, by key."""
    return {
        key: value for key, _, value in (part.partition("=") for part in change.detail.split(";"))
    }


# =============================================================================
# Building the tree
# =============================================================================


def _pipeline_code(
    project: _Project, pipeline: str, *, changed: Collection[str], first_run: bool
) -> str:
    """Pipeline ``pipeline``'s pipeline.py: each stage's text names its stage, its version (1 for
    a stage in ``changed``, else 0) and what it read. In the ``first_run`` a dropping producer
    writes key z too."""
    directories = {item.path for item in project.inputs if item.directory}
    own = [item for item in project.stages if item.pipeline == pipeline]
    lines = [
        "import pathlib",
        "from typing import Annotated",
        "",
        "from pivot import Pipeline, StageParams, directory, stage",
        "",
    ]
    calls = list[str]()
    inputs = dict[str, str]()
    for item in own:
        handles = list[str]()
        for read in item.reads:
            if read.startswith("data/raw/"):
                variable = inputs.setdefault(read, f"i{len(inputs)}")
                handles.append(variable)
            else:
                handles.append(f"h_{read.rsplit('/', 1)[1]}")
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
            over = f"h_{item.over.rsplit('/', 1)[1]}"
            handles.insert(0, f"pipeline.map({over})" if item.kind == "mapped" else over)
        calls.append(f"    h_{item.function} = {item.function}({', '.join(handles)})")
    lines += ["", "", f'with Pipeline("{pipeline}") as pipeline:']
    lines += [
        f'    {variable} = pipeline.input("{path.removeprefix("data/raw/")}", '
        + f"t={'pathlib.Path' if path in directories else 'str'})"
        for path, variable in inputs.items()
    ]
    lines += calls
    return "\n".join(lines) + "\n"


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
    tracked = [
        f"{item.path}/x.txt" if item.tracked == "member" else item.path
        for item in project.inputs
        if item.tracked != "no"
    ]
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


def _apply(change: _Change, root: pathlib.Path, case: _Case) -> list[tuple[pathlib.Path, int]]:
    """Make ``change`` in the tree at ``root``, but for the modes it sets: those it returns, as
    (path, mode), for ``_prepare`` to set once every change is made."""
    project = case.project
    kind, subject = change.kind, change.subject
    if kind == "lose_outputs":
        _lose_outputs(root, project, subject, _details(change))
    elif kind == "empty_cache":
        for entry in (root / ".pivot" / "cache" / "files").iterdir():
            shutil.rmtree(entry)
    elif kind == "delete_listing":
        _listing_path(root, subject).unlink()
    elif kind == "edit_listed":
        _edit_listed(_output_path(root, project, subject), change.detail)
    elif kind == "delete_output":
        cache.remove_output(_output_path(root, project, subject))
    elif kind == "track_output":
        output = _output_path(root, project, subject)
        cache.remove_output(output)
        helpers.track_without_data(output)
    elif kind == "delete_lock":
        _lock_path(root, subject).unlink()
    elif kind == "delete_tracked":
        return _delete_tracked(root, subject, _details(change))
    elif kind == "delete_input":
        cache.remove_output(root / subject)
    elif kind == "change_input":
        path = root / subject
        if path.is_dir():
            (path / "z.txt").write_text("z")
        else:
            path.unlink()
            path.write_text(f"{subject} changed\n")
    elif kind == "change_params":
        params = root / "params.yaml"
        values = yaml.safe_load(params.read_text()) if params.exists() else {}
        values[subject] = {"k": 1}
        params.write_text(yaml.safe_dump(values))
    else:
        # _prepare writes the pipelines of every code change at once
        assert kind == "change_code", kind
    return []


def _lose_outputs(
    root: pathlib.Path, project: _Project, name: str, details: dict[str, str]
) -> None:
    """Remove from the cache every object stage or instance ``name``'s lock names (with a remote,
    a pull brings them back), and its committed listing or workspace output as ``details`` say."""
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
    output = _output_path(root, project, name)
    if details["workspace"] == "removed":
        cache.remove_output(output)
    elif details["workspace"] != "kept":
        _edit_listed(output, details["workspace"])


def _delete_tracked(
    root: pathlib.Path, subject: str, details: dict[str, str]
) -> list[tuple[pathlib.Path, int]]:
    """Delete tracked path ``subject`` from the workspace and change what its restore reads, as
    ``_tracked_detail`` says; return the modes to set."""
    path = root / subject
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


def _prepare(case: _Case, base: pathlib.Path, root: pathlib.Path) -> None:
    """A copy of ``base`` at ``root`` with ``case``'s changes made; the modes they set last, so
    no change meets a path another one locked."""
    _copy_tree(base, root)
    changed = [change.subject for change in case.changes if change.kind == "change_code"]
    if changed:
        _write_pipelines(root, case.project, changed=changed)
    modes = list[tuple[pathlib.Path, int]]()
    for change in case.changes:
        modes += _apply(change, root, case)
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
    holding = {
        change.subject.rsplit("/", 1)[0]
        for change in case.changes
        if change.kind == "delete_tracked" and change.subject.endswith("/x.txt")
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
# The comparison
# =============================================================================


class _Row(NamedTuple):
    """A row of the dry run's document, by what it says the run does with the stage."""

    kind: str
    """"would run"; "after upstream" (would run only because a stage upstream would);
    "unchanged"; "would fail"; "blocked"; "not expanded" (a mapped stage's row standing for the
    instances the run adds)."""
    reason: str


_BLOCKED = re.compile(r"upstream '[^']+' would fail")


def _row(entry: dict[str, Any]) -> _Row:
    reason: str = entry["reason"]
    if entry["would_run"]:
        if reason.startswith(status.NOT_EXPANDED_REASON_PREFIX):
            return _Row("not expanded", reason)
        if reason.startswith(status.UPSTREAM_STALE_REASON_PREFIX):
            return _Row("after upstream", reason)
        return _Row("would run", reason)
    if reason == "unchanged":
        return _Row("unchanged", reason)
    if _BLOCKED.fullmatch(reason):
        return _Row("blocked", reason)
    return _Row("would fail", reason)


def _outcome(event: dict[str, Any]) -> str:
    """What the run did with a stage: "ran" (a run-cache hit counts), "skipped", "failed",
    "blocked", or the status it gave."""
    if event["status"] == "ran" or event["reason"] == "unchanged (run cache)":
        return "ran"
    if event["status"] == "cached":
        return "skipped"
    return event["status"]


# Each kind of row, with what the run may do with its stage
_AGREES = {
    "would run": {"ran"},
    "after upstream": {"ran", "skipped"},
    "unchanged": {"skipped"},
    "would fail": {"failed"},
    "blocked": {"blocked"},
    # With each instance the run adds
    "not expanded": {"ran", "skipped"},
}

# What the run may do with a mapped stage the dry run shows as not yet expanded, which adds no
# instance: fail its expansion, or block it behind one that failed. Which files the producer
# writes is known only once it has run (``test_dry_run_keep_going_limit_a_mapped_stage_its_
# producer_must_run_for`` in tests/cli/test_dry_run_keep_going.py)
_UNEXPANDED = {"failed", "blocked"}


def _error(stderr: str, roots: Sequence[pathlib.Path]) -> str:
    """The error a command ended with: its stderr from the last line starting `Error: `."""
    lines = stderr.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("Error: ")]
    return _normal("\n".join(lines[starts[-1] :]) if starts else "", roots)


def _normal(text: str, roots: Sequence[pathlib.Path]) -> str:
    """``text`` with each copy's root, as given and resolved, read as ``<root>``."""
    for root in roots:
        for form in {str(root), os.path.realpath(root)}:
            text = text.replace(form, "<root>")
    return text


def _restored(stderr: str, roots: Sequence[pathlib.Path]) -> list[str]:
    """The path each `Cannot restore` warning names."""
    return sorted(
        _normal(line.removeprefix("Cannot restore ").split(": ", 1)[0], roots)
        for line in helpers.restore_warnings(stderr)
    )


def _disagreements(
    dry: subprocess.CompletedProcess[str],
    ran: subprocess.CompletedProcess[str],
    *,
    roots: Sequence[pathlib.Path],
    limited: Collection[str],
) -> list[str]:
    """Each way the leave-out dry run ``dry`` and the run ``ran`` of one tree (copies at
    ``roots``) disagree, by spec R5 (version 7), each named.

    The keys of the dry run's `stages` are the stages the run runs, skips, fails or blocks; a
    row of a mapped stage not yet expanded stands for the instances the run adds. A row that
    would run is a stage the run runs; a run-cache hit counts, since the dry run does not
    predict one. A row that would run only because a stage upstream of it would (its reason
    `Upstream stale`) is a stage the run runs or skips: what that stage writes is known only
    once it has run. A row that is unchanged is a stage the run skips; one that would fail, a
    stage the run fails with the same reason; one that is blocked, a stage the run blocks. The
    instances a not-yet-expanded row stands for are each run or skipped, for the same reason;
    or, the limit, the run fails that mapped stage's expansion, or blocks it, once the producer
    has run (``_UNEXPANDED``). `would_remove` names the instances the run removes, but for a
    mapped stage not yet expanded, whose plan is known only once its producer has run. A
    left-out stage is absent from both, and each gives the same `Skipping:` lines and names the
    same paths in its `Cannot restore` warnings. A dry run that stops before its document stops
    as the run does, before any stage, with the same error; else it exits 1 only for a left-out
    target, with the run's error, and the run exits 1 for that or a stage it fails. Not
    compared: the stages of ``limited``, whose rows need only be present (``_limited``), and
    the dry run's notices, which are stderr only."""
    found = list[str]()
    events = [json.loads(line) for line in ran.stdout.splitlines()]
    completed = [event for event in events if event["type"] == "stage_complete"]
    skipping = (helpers.skipping_lines(dry.stderr), helpers.skipping_lines(ran.stderr))
    if _normal("\n".join(skipping[0]), roots) != _normal("\n".join(skipping[1]), roots):
        found.append(f"the same Skipping: lines: {skipping[0]} != {skipping[1]}")
    if _restored(dry.stderr, roots) != _restored(ran.stderr, roots):
        found.append(
            "the same failed restores: "
            + f"{_restored(dry.stderr, roots)} != {_restored(ran.stderr, roots)}"
        )
    errors = (_error(dry.stderr, roots), _error(ran.stderr, roots))
    if not dry.stdout.strip():
        if dry.returncode == 0 or ran.returncode == 0 or completed:
            found.append(
                "a dry run that stops stops the run before any stage: exits "
                + f"{dry.returncode} and {ran.returncode}, {len(completed)} stages completed"
            )
        if errors[0] != errors[1]:
            found.append(f"the same error: {errors[0]!r} != {errors[1]!r}")
        return found
    document = json.loads(dry.stdout)
    rows = {name: _row(entry) for name, entry in document["stages"].items()}
    not_expanded = {name for name, row in rows.items() if row.kind == "not expanded"}
    by_row = dict[str, list[dict[str, Any]]]()
    for event in completed:
        key = registry.template_of(event["stage"], not_expanded) or event["stage"]
        by_row.setdefault(key, []).append(event)
    if rows.keys() != by_row.keys():
        found.append(
            "the same stages: only in the dry run "
            + f"{sorted(rows.keys() - by_row.keys())}, only in the run "
            + f"{sorted(by_row.keys() - rows.keys())}"
        )
    for name in sorted(rows.keys() & by_row.keys()):
        row = rows[name]
        if name in limited or registry.template_of(name, limited) is not None:
            continue
        outcomes = [(event["stage"], _outcome(event), event["reason"]) for event in by_row[name]]
        if row.kind == "not expanded":
            agrees = all(
                outcome in (_UNEXPANDED if stage == name else _AGREES[row.kind])
                for stage, outcome, _ in outcomes
            )
        else:
            agrees = len(outcomes) == 1 and outcomes[0][1] in _AGREES[row.kind]
            if agrees and row.kind == "would fail":
                agrees = _normal(row.reason, roots) == _normal(outcomes[0][2], roots)
        if not agrees:
            found.append(f"{row.kind} <-> the run: {name}: {row.reason!r} vs {outcomes}")
    # `would_remove` is absent from a document that removes nothing, and `removed` from an
    # expansion that removes nothing
    predicted = sorted(document["would_remove"]) if "would_remove" in document else list[str]()
    removed = sorted(
        name
        for event in events
        if event["type"] == "stages_expanded" and event["template"] not in not_expanded
        for name in (event["removed"] if "removed" in event else list[str]())
    )
    if predicted != removed:
        found.append(f"would_remove <-> the run's removals: {predicted} != {removed}")
    refused = any(row.kind == "would fail" for row in rows.values())
    if (dry.returncode != 0) != bool(errors[0]) or dry.returncode not in (0, 1):
        found.append(f"the dry run exits 1 for a left-out target alone: {dry.returncode}")
    unexpanded = any(
        event["stage"] in not_expanded and event["status"] == "failed" for event in completed
    )
    expected = 1 if dry.returncode != 0 or refused or unexpanded else 0
    if ran.returncode != expected:
        found.append(f"the run's exit code: {ran.returncode}, expected {expected}")
    if dry.returncode != 0 and errors[0] != errors[1]:
        found.append(f"the same left-out targets: {errors[0]!r} != {errors[1]!r}")
    return found


# =============================================================================
# The tests
# =============================================================================


def _ok(proc: subprocess.CompletedProcess[str], case: _Case) -> None:
    assert proc.returncode == 0, f"seed {case.seed}: {proc.args}\n{proc.stdout}{proc.stderr}"


@pytest.fixture(scope="module")
def initialized_project(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pathlib.Path, pathlib.Path]:
    """An empty pivot project in a git repository, made once (`git init`, `pivot init`), and the
    home its commands run with; each case starts from a copy of it."""
    base = tmp_path_factory.mktemp("initialized")
    home = base / "home"
    home.mkdir()
    root = base / "project"
    root.mkdir()
    for args in (("git", "init", "-q"), (helpers.PIVOT, "init")):
        proc = helpers.run_cli(root, home, *args)
        assert proc.returncode == 0, proc.stdout + proc.stderr
    return root, home


def _report(case: _Case, root: pathlib.Path, found: list[str], *procs: Any) -> str:
    """What a failure prints: the seed, each disagreement, the case, its pipelines as the
    commands ran them, and the commands' output."""
    pipelines = "".join(
        f"--- {path.relative_to(root)}\n{path.read_text()}"
        for path in sorted(root.glob("p*/pipeline.py"))
    )
    outputs = "".join(
        f"--- {' '.join(proc.args[1:])} (exit {proc.returncode})\n{proc.stdout}{proc.stderr}"
        for proc in procs
    )
    return (
        f"seed {case.seed} ({_SEEDS_VARIABLE}={case.seed}:{case.seed + 1}): "
        + f"{len(found)} disagreements\n"
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
    run on a copy of the same tree runs, skips, fails or blocks, as ``_disagreements`` reads
    them."""
    case = _generate(seed, locked_out=os.geteuid() != 0)
    initialized, home = initialized_project
    base = tmp_path / "base"
    _copy_tree(initialized, base)
    _build(case, base, home)
    roots = (tmp_path / "dry", tmp_path / "run")
    for root in roots:
        _prepare(case, base, root)
    arguments = ("repro", "--all", "--keep-going", "--jsonl", *case.options, *case.targets)

    dry = helpers.run_pivot(roots[0], home, *arguments[:3], "--dry-run", *arguments[3:])
    ran = helpers.run_pivot(roots[1], home, *arguments)

    found = _disagreements(dry, ran, roots=roots, limited=_limited(case))
    assert not found, _report(case, roots[0], found, dry, ran)


def _target_kind(case: _Case, target: str) -> str:
    if "@" in target:
        return "instance"
    return next(item.kind for item in case.project.stages if item.name == target)


def test_dry_run_keep_going_fixed_seeds_cover_every_kind() -> None:
    """Between them, the fixed seeds generate every kind of change, each run option, a target
    naming each kind of stage and an instance, two pipelines, each kind of tracked input, a
    configured remote and a producer whose key left its listing (for a user who is not root,
    as CI's; root gets the same cases but for the modes)."""
    cases = [_generate(seed, locked_out=True) for seed in _FIXED_SEEDS]

    found = {
        *(f"change {change.kind}" for case in cases for change in case.changes),
        *(
            f"{change.kind} {part}"
            for case in cases
            for change in case.changes
            if change.kind == "delete_tracked"
            for part in change.detail.split(";")
        ),
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
        *("remote" for case in cases if case.project.remote),
        *("dropping" for case in cases if case.project.dropping),
    }

    assert found.issuperset(_COVERED), sorted(_COVERED - found)


# What the fixed seeds generate between them (``test_dry_run_keep_going_fixed_seeds_cover_every_kind``)
_COVERED = {
    *(f"change {kind}" for kind in _ORDER),
    *(
        f"delete_tracked object={value}"
        for value in ("kept", "removed", "unreadable", "unsearchable")
    ),
    "delete_tracked parent=read_only",
    *(f"delete_tracked committed={value}" for value in ("intact", "stale", "empty", "missing")),
    "option --no-commit",
    "option --checkout-missing",
    *(f"target {kind}" for kind in ("plain", "producer", "mapped", "chained", "instance")),
    "input file whole",
    "input directory whole",
    "input directory member",
    "two pipelines",
    "remote",
    "dropping",
}
