"""Time `pivot repro` on generated projects, for several pivot builds interleaved.

Usage: run.py --project NAME --variant NAME=PATH_TO_PIVOT [--variant ...] [--reps 6]
              --work DIR --out-dir DIR

Each variant gets its own copy of the project under --work and writes one JSON line per
command to <out-dir>/<variant>.jsonl. Within each scenario the variants take turns, and the
order flips every rep, so drift in the machine's load falls on both alike.

CPU is the user plus system time of every descendant the command reaped, read from
getrusage(RUSAGE_CHILDREN) before and after it. Processes still alive in the command's session
when it exits are counted in `leftover_procs` and killed; their CPU is not counted.

Commands run with the environment a user has: the harness removes the BLAS thread variables
(OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS) rather than pinning them, because,
once Task 1.10a is merged, pivot sets them for its workers and keeps any value it inherits.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import TypedDict

import gen

KINDS = ("cold", "warm", "touch", "edit")
_BLAS_VARS = frozenset({"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"})
_SUMMARY = re.compile(r"Summary: (\d+) ran, (\d+) cached")
_MARKER = re.compile(r"^    _edit_marker = .*\n", re.MULTILINE)
_KEEP_IN_PIVOT_DIR = frozenset({".gitignore", "config.yaml"})


class Measurement(TypedDict):
    wall_s: float
    cpu_s: float
    exit_code: int
    load1: float
    cpu_count: int
    leftover_procs: int


class Row(TypedDict):
    scenario: str
    variant: str
    rep: int
    wall_s: float
    cpu_s: float
    exit_code: int
    load1: float
    cpu_count: int
    ran: int | None
    cached: int | None
    leftover_procs: int


def _session_members(sid: int) -> list[int]:
    members = list[int]()
    for stat in pathlib.Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        # After the command name: state, ppid, pgrp, session.
        if fields[0] != "Z" and int(fields[3]) == sid:
            members.append(int(stat.parent.name))
    return members


def measure(argv: list[str], cwd: pathlib.Path) -> tuple[Measurement, str]:
    """Run one command to completion; return its measurement and combined output."""
    env = {key: value for key, value in os.environ.items() if key not in _BLAS_VARS}
    load1 = os.getloadavg()[0]
    # Output goes to a file, not a pipe: a leftover process holding the pipe open would
    # otherwise keep the harness waiting after the command itself exits.
    with tempfile.TemporaryFile("w+") as log:
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.perf_counter()
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        proc.wait()
        wall = time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        log.seek(0)
        output = log.read()
    leftovers = _session_members(proc.pid)
    for pid in leftovers:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    measurement = Measurement(
        wall_s=round(wall, 4),
        cpu_s=round(cpu, 4),
        exit_code=proc.returncode,
        load1=load1,
        cpu_count=os.process_cpu_count() or 1,
        leftover_procs=len(leftovers),
    )
    return measurement, output


def _reset(root: pathlib.Path) -> None:
    """Remove every output, lock, cache entry and the StateDB; keep code and raw inputs."""
    for child in (root / ".pivot").iterdir():
        if child.name in _KEEP_IN_PIVOT_DIR:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in (root / "data").iterdir():
        if child.name != "raw":
            shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    for name in ("metrics", "plots"):
        if (root / name).exists():
            shutil.rmtree(root / name)


def _edit(path: pathlib.Path, function: str, stamp: str) -> None:
    """Change one stage's code (an unused assignment) without changing what it returns."""
    source = path.read_text()
    marker = f"    _edit_marker = {stamp!r}\n"
    if _MARKER.search(source):
        source = _MARKER.sub(marker, source, count=1)
    else:
        body = source.index(":\n", source.index(f"def {function}(")) + 2
        source = source[:body] + marker + source[body:]
    path.write_text(source)


def _prepare(kind: str, root: pathlib.Path, scope: gen.Scope, stamp: str) -> None:
    match kind:
        case "cold":
            _reset(root)
        case "warm":
            pass
        case "touch":
            (root / scope["touch"]).write_text(f"name: touched\nv: {stamp}\n")
        case "edit":
            _edit(root / scope["edit_file"], scope["edit_function"], stamp)
        case _:
            raise ValueError(f"unknown kind {kind!r}")


def _parse_variant(text: str) -> tuple[str, pathlib.Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH_TO_PIVOT, got {text!r}")
    return name, pathlib.Path(path).absolute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", required=True, choices=gen.PROJECTS)
    parser.add_argument("--variant", required=True, action="append", type=_parse_variant)
    parser.add_argument("--reps", type=int, default=6)
    parser.add_argument("--work", required=True, type=pathlib.Path)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    variants: list[tuple[str, pathlib.Path]] = args.variant
    project: str = args.project
    reps: int = args.reps
    work: pathlib.Path = args.work.absolute()
    out_dir: pathlib.Path = args.out_dir
    if len({name for name, _ in variants}) != len(variants):
        parser.error("variant names must be unique")

    roots = {name: work / name / project for name, _ in variants}
    specs = {name: gen.write_project(project, roots[name], pivot) for name, pivot in variants}
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = {name: (out_dir / f"{name}.jsonl").open("w") for name, _ in variants}
    failed = 0
    try:
        for scope_index, scope in enumerate(specs[variants[0][0]]["scopes"]):
            for kind in KINDS:
                scenario = f"{project}/{scope['name']}/{kind}"
                for rep in range(reps):
                    order = variants if rep % 2 == 0 else variants[::-1]
                    for name, pivot in order:
                        root = roots[name]
                        own_scope = specs[name]["scopes"][scope_index]
                        _prepare(kind, root, own_scope, f"{scope['name']}-{rep}")
                        measurement, output = measure(
                            [str(pivot), "repro", *own_scope["args"]], root / own_scope["cwd"]
                        )
                        summary = _SUMMARY.search(output)
                        row = Row(
                            scenario=scenario,
                            variant=name,
                            rep=rep,
                            ran=int(summary.group(1)) if summary else None,
                            cached=int(summary.group(2)) if summary else None,
                            **measurement,
                        )
                        outs[name].write(json.dumps(row) + "\n")
                        outs[name].flush()
                        print(
                            f"{scenario} {name} rep={rep} exit={row['exit_code']} "
                            + f"wall={row['wall_s']:.2f}s cpu={row['cpu_s']:.2f}s "
                            + f"ran={row['ran']} cached={row['cached']} load1={row['load1']:.2f} "
                            + f"cores={row['cpu_count']} leftover={row['leftover_procs']}",
                            flush=True,
                        )
                        if row["exit_code"] != 0:
                            failed += 1
                            print(output[-3000:], file=sys.stderr, flush=True)
    finally:
        for out in outs.values():
            out.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
