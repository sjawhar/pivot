"""The cases of the differential test of `pivot repro --all --keep-going` in leave-out mode
(``test_dry_run_keep_going_generated``), and the generator that makes one from a seed.

A seed gives one case (``generate``): a project of one or two pipelines of plain stages,
directory producers and mapped stages over them (``Pipeline.map``, and chains of them), reading
raw files and directories, some tracked with `pivot track`, and of two pipelines the first may
read the second's stages through their handles (``_compose_across``); the changes made after a
first full run (``Change``), among them a FIFO where a deleted tracked path was (``_occupy``);
and the run's options and targets. The seed alone builds the same case again.
"""

from __future__ import annotations

import random
import typing
from typing import Literal, NamedTuple

# The keys every producer writes, and the key a dropping producer wrote in the first run alone
KEYS = ("a", "b")
DROPPED_KEY = "z"


class Input(NamedTuple):
    """A raw input under data/raw/."""

    path: str
    """Project-relative: data/raw/f<i>.txt, or data/raw/d<i> for a directory of x.txt and y.txt."""
    directory: bool
    tracked: str
    """"no"; "whole": `pivot track` of the path; "member": of the directory's file x.txt."""


def tracked_path(item: Input) -> str:
    """The path `pivot track` tracks for input ``item`` (one that is tracked): the input itself,
    or, for a "member", its directory's file x.txt."""
    return f"{item.path}/x.txt" if item.tracked == "member" else item.path


StageKind = Literal["plain", "producer", "mapped", "chained"]


class Stage(NamedTuple):
    """A stage of a generated pipeline."""

    name: str
    """``<pipeline>/<function>``."""
    pipeline: str
    function: str
    kind: StageKind
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


class Project(NamedTuple):
    inputs: tuple[Input, ...]
    stages: tuple[Stage, ...]
    remote: bool
    """Whether a remote is configured: a stage whose outputs the cache lost is then one a pull
    brings up to date."""
    dropping: tuple[str, ...]
    """Producers that also wrote key z in the first run, and then ran alone without it, so the
    instances of key z are orphans."""


# Each kind of change a case makes after the first run, in the order the test makes them
# (``ORDER``), each reading what the ones before it leave
ChangeKind = Literal[
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
ORDER: tuple[ChangeKind, ...] = typing.get_args(ChangeKind)


class Change(NamedTuple):
    """One change made to the tree after the first run (``test_dry_run_keep_going_generated``'s
    ``_APPLY`` makes it)."""

    kind: ChangeKind
    subject: str
    """An input path, a stage name, an instance name, or "" for the whole cache."""
    detail: dict[str, str]
    """How the change is made, by what it changes: for `delete_tracked`, what happens to each
    cache object the restore reads (``_OBJECTS``), to the committed listing and to the
    directory holding the path (``_tracked_detail``), and what stands in the path's place
    (``_occupy``); for `lose_outputs`, what else the stage loses (``_lost_detail``); for
    `edit_listed`, the edit; for any other, nothing."""


class Case(NamedTuple):
    seed: int
    project: Project
    changes: tuple[Change, ...]
    options: tuple[str, ...]
    targets: tuple[str, ...]


def generate(seed: int, *, locked_out: bool) -> Case:
    """The case of ``seed``. ``locked_out``: whether a mode can lock this user out of a path
    (not root), so a change may make a cache object or directory unreadable or a directory
    read-only."""
    rng = random.Random(seed)
    inputs = [
        Input(f"data/raw/f{index}.txt", False, rng.choice(("no", "whole")))
        for index in range(rng.randint(1, 3))
    ]
    inputs += [
        Input(f"data/raw/d{index}", True, rng.choice(("no", "whole", "member")))
        for index in range(rng.randint(0, 2))
    ]
    paths = [item.path for item in inputs]
    stages = list[Stage]()
    for pipeline in (f"p{index}" for index in range(rng.randint(1, 2))):
        chain = rng.choice((0, 2, 3, 3))
        plain = list[str]()
        for index in range(rng.randint(2, 3) if chain == 0 else rng.randint(0, 2)):
            candidates = [*paths, *plain]
            name = f"{pipeline}/s{index}"
            reads = tuple(rng.sample(candidates, rng.randint(1, min(2, len(candidates)))))
            stages.append(
                Stage(name, pipeline, f"s{index}", "plain", reads, "", rng.random() < 0.3)
            )
            plain.append(name)
        candidates = [*paths, *plain]
        functions = ("plan0", "cut0", "build0")[:chain]
        over = ""
        for function, kind in zip(functions, ("producer", "mapped", "chained"), strict=False):
            reads = tuple(rng.sample(candidates, rng.randint(0, 1))) if kind != "chained" else ()
            name = f"{pipeline}/{function}"
            stages.append(Stage(name, pipeline, function, kind, reads, over, False))
            over = name
    stages = _compose_across(random.Random(f"{seed}:across"), stages)
    producers = [item.name for item in stages if item.kind == "producer"]
    project = Project(
        inputs=tuple(inputs),
        stages=tuple(stages),
        remote=rng.random() < 0.5,
        dropping=tuple(name for name in producers if rng.random() < 0.25),
    )
    changes = _occupy(
        random.Random(f"{seed}:occupant"), _generate_changes(rng, project, locked_out=locked_out)
    )
    options = tuple(
        option
        for option, chance in (("--no-commit", 0.4), ("--checkout-missing", 0.5))
        if rng.random() < chance
    )
    targets = list[str]()
    if rng.random() < 0.5:
        named = [
            *(item.name for item in stages),
            *(f"{item.name}@{key}" for item in stages if item.over for key in KEYS),
        ]
        targets = rng.sample(named, rng.randint(1, min(2, len(named))))
    return Case(seed, project, changes, options, tuple(targets))


def _compose_across(rng: random.Random, stages: list[Stage]) -> list[Stage]:
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
    composed = list[Stage]()
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
    rng: random.Random, project: Project, *, locked_out: bool
) -> tuple[Change, ...]:
    """One to three changes, each to its own input, its own stage or the whole cache, a
    subject drawn by its weight (``_WEIGHTS``)."""
    choices = list[tuple[float, list[Change]]]()
    for item in project.inputs:
        if item.tracked == "no":
            choices.append(
                (
                    _WEIGHTS["input"],
                    [
                        Change("delete_input", item.path, {}),
                        Change("change_input", item.path, {}),
                    ],
                )
            )
            continue
        path = tracked_path(item)
        detail = _tracked_detail(rng, directory=item.directory and item.tracked == "whole")
        if not locked_out:
            # Root reads and writes whatever the modes say
            detail = {
                key: "kept"
                if key in _OBJECTS and value in ("unreadable", "unsearchable")
                else value
                for key, value in detail.items()
            } | {"parent": "writable"}
        choices.append((_WEIGHTS["input"], [Change("delete_tracked", path, detail)]))
    for item in project.stages:
        options = [Change("change_code", item.name, {})]
        if item.params:
            options.append(Change("change_params", item.name, {}))
        if item.kind in ("plain", "producer"):
            producer = item.kind == "producer"
            options += [
                Change("delete_output", item.name, {}),
                Change("delete_lock", item.name, {}),
                # A stage whose cache objects are gone, a pull-only one with a remote, twice
                Change("lose_outputs", item.name, _lost_detail(rng, producer=producer)),
                Change("lose_outputs", item.name, _lost_detail(rng, producer=producer)),
            ]
        if item.kind == "producer":
            edit = rng.choice(("add c.txt", "add c d.txt", "remove a.txt"))
            options += [
                Change("edit_listed", item.name, {"edit": edit}),
                Change("delete_listing", item.name, {}),
            ]
        if item.over:
            instance = f"{item.name}@{rng.choice(KEYS)}"
            options += [
                Change("delete_output", instance, {}),
                Change("delete_lock", instance, {}),
                Change("lose_outputs", instance, {"listing": "kept", "workspace": "kept"}),
                Change("track_output", instance, {}),
                Change("track_output", instance, {}),
            ]
        choices.append((_WEIGHTS[item.kind], options))
    choices.append((0.5, [Change("empty_cache", "", {})]))
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
    return tuple(sorted(picked, key=lambda change: ORDER.index(change.kind)))


# The keys of a `delete_tracked` detail that each say what happens to a cache object the
# restore reads (``OBJECT_FATES``)
_OBJECTS = ("object", "listing", "members")

# What deleting a tracked path does to a cache object its restore reads: kept, removed,
# unreadable, or in a prefix directory this user cannot search
OBJECT_FATES = ("kept", "removed", "unreadable", "unsearchable")
# The same for a tracked directory's member objects: each but a prefix directory locked
MEMBER_FATES = OBJECT_FATES[:3]
# What it does to a tracked directory's committed listing
COMMITTED = ("intact", "stale", "empty", "missing")
# What it does to the directory that holds the tracked path
PARENTS = ("writable", "read_only")
# What stands where the tracked path was: nothing, or a FIFO, which the pre-flight counts as
# missing (``track.present_kind``), the restore does not replace, and nothing may open
OCCUPANTS = ("none", "fifo")
# How likely a FIFO is to stand where a deleted tracked path was
_FIFO_CHANCE = 0.1


def _tracked_detail(rng: random.Random, *, directory: bool) -> dict[str, str]:
    """What deleting a tracked path does to what its restore reads: its object (a file), or its
    cached listing object, committed listing and member objects (a directory), each as
    ``OBJECT_FATES``, ``COMMITTED`` and ``MEMBER_FATES`` say (``_OBJECTS``); and whether the
    directory that holds it is left writable or made read-only (``PARENTS``)."""
    parent = rng.choice(PARENTS)
    if not directory:
        return {"object": rng.choice(OBJECT_FATES), "parent": parent}
    return {
        "listing": rng.choice(OBJECT_FATES),
        "committed": rng.choice(COMMITTED),
        "members": rng.choice(MEMBER_FATES),
        "parent": parent,
    }


def _occupy(rng: random.Random, changes: tuple[Change, ...]) -> tuple[Change, ...]:
    """``changes``, with what stands where each `delete_tracked` change's path was in its detail,
    as ``OCCUPANTS`` says: nothing, or, in one case of ``1 / _FIFO_CHANCE``, a FIFO. ``rng`` is
    this alone's, so a case with no FIFO is the one its seed gave before FIFOs were generated."""
    return tuple(
        change._replace(
            detail=change.detail | {"occupant": "fifo" if rng.random() < _FIFO_CHANCE else "none"}
        )
        if change.kind == "delete_tracked"
        else change
        for change in changes
    )


def _lost_detail(rng: random.Random, *, producer: bool) -> dict[str, str]:
    """What else a stage that lost its cache objects lost: its committed listing (a producer),
    and its output in the workspace, removed or, for a producer, edited."""
    workspace = ("kept", "removed")
    if not producer:
        return {"listing": "kept", "workspace": rng.choice(workspace)}
    edits = (*workspace, "add c.txt", "add c d.txt", "remove a.txt")
    return {"listing": rng.choice(("kept", "removed")), "workspace": rng.choice(edits)}
