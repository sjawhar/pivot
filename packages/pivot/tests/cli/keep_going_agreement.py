"""Whether the leave-out dry run and the run of one tree agree, as spec R5 (version 7) says.

``disagreements`` names each way `pivot repro --all --keep-going --dry-run --jsonl` on one copy of
a tree and `pivot repro --all --keep-going --jsonl` on another copy of it disagree. The
differential test (``test_dry_run_keep_going_generated``) asks it of each generated case.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pivot import registry, status
from tests import helpers

if TYPE_CHECKING:
    import pathlib
    import subprocess
    from collections.abc import Collection, Mapping, Sequence


_RowKind = Literal[
    "would run", "after upstream", "unchanged", "would fail", "blocked", "not expanded"
]


class _Row(NamedTuple):
    """A row of the dry run's document, by what it says the run does with the stage."""

    kind: _RowKind
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


def _not_expanded(rows: Mapping[str, _Row]) -> set[str]:
    """The mapped stages the dry run shows as not yet expanded."""
    return {name for name, row in rows.items() if row.kind == "not expanded"}


class _Outcome(NamedTuple):
    """What the run did with a stage, by its `stage_complete` event (``_outcome``)."""

    stage: str
    status: str
    """"ran" (a run-cache hit counts), "skipped", "failed", "blocked", or the status it gave."""
    reason: str


def _outcome(event: Mapping[str, Any]) -> _Outcome:
    if event["status"] == "ran" or event["reason"] == "unchanged (run cache)":
        did = "ran"
    elif event["status"] == "cached":
        did = "skipped"
    else:
        did = event["status"]
    return _Outcome(event["stage"], did, event["reason"])


# Each kind of row, with what the run may do with its stage
_AGREES: dict[_RowKind, set[str]] = {
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
# producer_must_run_for`` in tests/cli/test_dry_run_keep_going_chain.py)
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


def _is_limited(name: str, limited: Collection[str]) -> bool:
    """Whether stage or instance ``name`` is of ``limited`` (``disagreements``')."""
    return name in limited or registry.template_of(name, limited) is not None


def _stderr_disagreements(
    dry: subprocess.CompletedProcess[str],
    ran: subprocess.CompletedProcess[str],
    roots: Sequence[pathlib.Path],
) -> list[str]:
    """Each gives the same `Skipping:` lines, names the same paths in its `Cannot restore`
    warnings, and ends with the same error, its stderr from its last `Error:` line, or with
    none (a run whose only problem is a stage it fails prints none)."""
    found = list[str]()
    dry_skipping = _normal("\n".join(helpers.skipping_lines(dry.stderr)), roots)
    run_skipping = _normal("\n".join(helpers.skipping_lines(ran.stderr)), roots)
    if dry_skipping != run_skipping:
        found.append(f"the same Skipping: lines: {dry_skipping!r} != {run_skipping!r}")
    dry_restores, run_restores = _restored(dry.stderr, roots), _restored(ran.stderr, roots)
    if dry_restores != run_restores:
        found.append(f"the same failed restores: {dry_restores} != {run_restores}")
    dry_error, run_error = _error(dry.stderr, roots), _error(ran.stderr, roots)
    if dry_error != run_error:
        found.append(f"the same error: {dry_error!r} != {run_error!r}")
    return found


def _stop_disagreements(
    dry: subprocess.CompletedProcess[str],
    ran: subprocess.CompletedProcess[str],
    completed: Sequence[_Outcome],
) -> list[str]:
    """A dry run that stops before its document stops as the run does: both exit non-zero, and
    the run before any stage completes."""
    if dry.returncode != 0 and ran.returncode != 0 and not completed:
        return []
    return [
        "a dry run that stops stops the run before any stage: exits "
        + f"{dry.returncode} and {ran.returncode}, {len(completed)} stages completed"
    ]


def _row_disagreements(
    rows: Mapping[str, _Row],
    completed: Sequence[_Outcome],
    *,
    roots: Sequence[pathlib.Path],
    limited: Collection[str],
) -> list[str]:
    """The keys of the dry run's `stages` are the stages the run runs, skips, fails or blocks;
    a row of a mapped stage not yet expanded stands for the instances the run adds, and a
    left-out stage is absent from both. A row that would run is a stage the run runs; a
    run-cache hit counts, since the dry run does not predict one. A row that would run only
    because a stage upstream of it would (its reason `Upstream stale`) is a stage the run runs
    or skips: what that stage writes is known only once it has run. A row that is unchanged is
    a stage the run skips; one that would fail, a stage the run fails with the same reason; one
    that is blocked, a stage the run blocks. The instances a not-yet-expanded row stands for are
    each run or skipped; or, the limit, the run fails that mapped stage's expansion, or blocks
    it, once the producer has run (``_UNEXPANDED``). The rows of the stages of ``limited`` need
    only be present."""
    not_expanded = _not_expanded(rows)
    by_row = dict[str, list[_Outcome]]()
    for outcome in completed:
        key = registry.template_of(outcome.stage, not_expanded) or outcome.stage
        by_row.setdefault(key, []).append(outcome)
    found = list[str]()
    if rows.keys() != by_row.keys():
        found.append(
            "the same stages: only in the dry run "
            + f"{sorted(rows.keys() - by_row.keys())}, only in the run "
            + f"{sorted(by_row.keys() - rows.keys())}"
        )
    for name in sorted(rows.keys() & by_row.keys()):
        row = rows[name]
        if _is_limited(name, limited):
            continue
        outcomes = by_row[name]
        if row.kind == "not expanded":
            agrees = all(
                outcome.status in (_UNEXPANDED if outcome.stage == name else _AGREES[row.kind])
                for outcome in outcomes
            )
        elif len(outcomes) != 1:
            agrees = False
        else:
            (outcome,) = outcomes
            agrees = outcome.status in _AGREES[row.kind] and (
                row.kind != "would fail"
                or _normal(row.reason, roots) == _normal(outcome.reason, roots)
            )
        if not agrees:
            found.append(f"{row.kind} <-> the run: {name}: {row.reason!r} vs {outcomes}")
    return found


def _removal_disagreements(
    document: dict[str, Any], events: Sequence[Mapping[str, Any]], rows: Mapping[str, _Row]
) -> list[str]:
    """`would_remove` names the instances the run removes, but for a mapped stage not yet
    expanded, whose plan is known only once its producer has run."""
    not_expanded = _not_expanded(rows)
    # `would_remove` is absent from a document that removes nothing, and `removed` from an
    # expansion that removes nothing
    predicted = sorted(document["would_remove"]) if "would_remove" in document else list[str]()
    removed = sorted(
        name
        for event in events
        if event["type"] == "stages_expanded" and event["template"] not in not_expanded
        for name in (event["removed"] if "removed" in event else list[str]())
    )
    if predicted == removed:
        return []
    return [f"would_remove <-> the run's removals: {predicted} != {removed}"]


def _exit_disagreements(
    dry: subprocess.CompletedProcess[str],
    ran: subprocess.CompletedProcess[str],
    rows: Mapping[str, _Row],
    completed: Sequence[_Outcome],
    *,
    limited: Collection[str],
) -> list[str]:
    """The dry run's exit code is non-zero exactly when its stderr has an `Error:` line, a
    target it leaves out, and is 1 then. The run exits 1 exactly when the dry run does, a row of
    a stage outside ``limited`` would fail, or the run fails a stage whose failure the dry run
    need not predict: a mapped stage it shows as not yet expanded (``_UNEXPANDED``), or a stage
    of ``limited``, whose row may say it would fail or not either way; it exits 0 otherwise."""
    found = list[str]()
    has_error = bool(_error(dry.stderr, ()))
    if (dry.returncode != 0) != has_error or dry.returncode not in (0, 1):
        found.append(
            "the dry run's exit code is non-zero exactly when its stderr has an Error: line, "
            + f"and is 1 then: exit {dry.returncode}, an Error: line {has_error}"
        )
    not_expanded = _not_expanded(rows)
    refused = any(
        row.kind == "would fail" for name, row in rows.items() if not _is_limited(name, limited)
    )
    unpredicted = any(
        outcome.status == "failed"
        and (outcome.stage in not_expanded or _is_limited(outcome.stage, limited))
        for outcome in completed
    )
    expected = 1 if dry.returncode != 0 or refused or unpredicted else 0
    if ran.returncode != expected:
        found.append(f"the run's exit code: {ran.returncode}, expected {expected}")
    return found


def disagreements(
    dry: subprocess.CompletedProcess[str],
    ran: subprocess.CompletedProcess[str],
    events: Sequence[Mapping[str, Any]],
    *,
    roots: Sequence[pathlib.Path],
    limited: Collection[str],
) -> list[str]:
    """Each way the leave-out dry run ``dry`` and the run ``ran``, whose JSONL events are
    ``events``, of one tree (copies at ``roots``) disagree, each named, rule by rule: their
    stderr (``_stderr_disagreements``); a dry run that stops before its document
    (``_stop_disagreements``); else the rows (``_row_disagreements``), `would_remove`
    (``_removal_disagreements``) and the exit codes (``_exit_disagreements``). ``limited``: the
    stages whose explanation the dry run takes from the files on disk where the run's restore
    changes them (spec R5's limit, written in docs/cli/index.md), a mapped stage standing for
    its instances too; their rows need only be present, and their ``would_run`` and reason need
    not be the run's. Not compared: the dry run's notices, which are stderr only."""
    completed = [_outcome(event) for event in events if event["type"] == "stage_complete"]
    found = _stderr_disagreements(dry, ran, roots)
    if not dry.stdout.strip():
        return [*found, *_stop_disagreements(dry, ran, completed)]
    document = json.loads(dry.stdout)
    rows = {name: _row(entry) for name, entry in document["stages"].items()}
    return [
        *found,
        *_row_disagreements(rows, completed, roots=roots, limited=limited),
        *_removal_disagreements(document, events, rows),
        *_exit_disagreements(dry, ran, rows, completed, limited=limited),
    ]
