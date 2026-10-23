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
once Task 1.10a is merged, pivot sets them for its workers and keeps any value it inherits. It
also removes PIVOT_CACHE_DIR, so each variant writes its own project's .pivot/cache.

An interrupted run (Ctrl-C, or SIGTERM from `kill` or `timeout`) stops the command it is
measuring: pivot gets SIGINT, its own Ctrl-C path, and up to 10 s to exit; then it, and anything
left in its session, is killed. Further SIGINTs and SIGTERMs cannot interrupt that stop.
"""

from __future__ import annotations

import argparse
import contextlib
import enum
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


class Kind(enum.StrEnum):
    """What each measured command changes first; the value is the scenario name's last part."""

    COLD = "cold"
    WARM = "warm"
    TOUCH = "touch"
    EDIT = "edit"


_DROPPED_VARS = frozenset(
    {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PIVOT_CACHE_DIR"}
)
_INTERRUPT_GRACE_S = 10

_STOP_SIGNALS = frozenset({signal.SIGINT, signal.SIGTERM})
_first_stop_signal: int | None = None
_stop_signal_read_fd: int | None = None
_stop_signal_write_fd: int | None = None
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


class Row(Measurement):
    scenario: str
    variant: str
    rep: int
    ran: int | None
    cached: int | None


def session_members(sid: int) -> list[int]:
    """Return the live processes in session `sid`, the one `measure()` starts each command in."""
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
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, set[signal.Signals]())
    try:
        env = {key: value for key, value in os.environ.items() if key not in _DROPPED_VARS}
        load1 = os.getloadavg()[0]
        # Output goes to a file, not a pipe: a leftover process holding the pipe open would
        # otherwise keep the harness waiting after the command itself exits.
        with tempfile.TemporaryFile("w+") as log:
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            start = time.perf_counter()
            proc = subprocess.Popen(
                argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                proc.wait()
            except BaseException:
                signal.pthread_sigmask(signal.SIG_BLOCK, _STOP_SIGNALS)
                _stop(proc)
                raise
            wall = time.perf_counter() - start
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            log.seek(0)
            output = log.read()
        leftovers = session_members(proc.pid)
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
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def _reaped(proc: subprocess.Popen[bytes], *, block: bool) -> bool:
    """Reap `proc` directly, avoiding Popen's lock after an interrupt."""
    if proc.returncode is not None:
        return True
    try:
        pid, status = os.waitpid(proc.pid, 0 if block else os.WNOHANG)
    except ChildProcessError:
        return True
    if pid == 0:
        return False
    proc.returncode = os.waitstatus_to_exitcode(status)
    return True


def _discard_pending_stop_signals() -> None:
    """Discard stop signals that arrived after the first one started cleanup."""
    while signal.sigtimedwait(_STOP_SIGNALS, 0) is not None:
        pass


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """Finish a fixed stop sequence despite further SIGINT or SIGTERM deliveries."""
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, _STOP_SIGNALS)
    try:
        if not _reaped(proc, block=False):
            with contextlib.suppress(ProcessLookupError):
                os.kill(proc.pid, signal.SIGINT)
            deadline = time.monotonic() + _INTERRUPT_GRACE_S
            while not _reaped(proc, block=False):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(proc.pid, signal.SIGKILL)
                    break
                time.sleep(min(remaining, 0.05))
        for pid in session_members(proc.pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        _reaped(proc, block=True)
    finally:
        _discard_pending_stop_signals()
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def _first_queued_stop_signal(fallback: int) -> int:
    """Return the first observed stop signal from the wakeup pipe.

    run.py's status is the first stop signal it observes. Two stop signals that are pending together
    have no observable order, and run.py reports the one the wakeup pipe lists first. CPython writes
    the byte in `trip_signal` at
    https://github.com/python/cpython/blob/v3.13.16/Modules/signalmodule.c#L323; Linux documents
    pending-standard-signal delivery order as unspecified at
    https://man7.org/linux/man-pages/man7/signal.7.html.
    """
    if _stop_signal_read_fd is None:
        return fallback
    try:
        queued = os.read(_stop_signal_read_fd, 4_096)
    except BlockingIOError:
        return fallback
    for signum in queued:
        if signum in _STOP_SIGNALS:
            return signum
    return fallback


def _stop_on_signal(signum: int, _frame: object) -> None:
    """Stop once using the first observed stop signal.

    run.py's status is the first stop signal it observes. Two stop signals that are pending together
    have no observable order, and run.py reports the one the wakeup pipe lists first. CPython writes
    the byte in `trip_signal` at
    https://github.com/python/cpython/blob/v3.13.16/Modules/signalmodule.c#L323; Linux documents
    pending-standard-signal delivery order as unspecified at
    https://man7.org/linux/man-pages/man7/signal.7.html.
    """
    global _first_stop_signal
    signal.pthread_sigmask(signal.SIG_BLOCK, _STOP_SIGNALS)
    if _first_stop_signal is not None:
        return
    _first_stop_signal = _first_queued_stop_signal(signum)
    if _first_stop_signal == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + _first_stop_signal)


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


def _prepare(kind: Kind, root: pathlib.Path, scope: gen.Scope, stamp: str) -> None:
    match kind:
        case Kind.COLD:
            _reset(root)
        case Kind.WARM:
            pass
        case Kind.TOUCH:
            (root / scope["touch"]).write_text(f"name: touched\nv: {stamp}\n")
        case Kind.EDIT:
            _edit(root / scope["edit_file"], scope["edit_function"], stamp)


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
    # Every variant's project has the same scopes: a Scope holds paths relative to its root.
    scopes = [gen.write_project(project, roots[name], pivot) for name, pivot in variants][0]
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = {name: (out_dir / f"{name}.jsonl").open("w") for name, _ in variants}
    failed = 0
    try:
        for scope in scopes:
            for kind in Kind:
                scenario = f"{project}/{scope['name']}/{kind}"
                for rep in range(reps):
                    order = variants if rep % 2 == 0 else variants[::-1]
                    for name, pivot in order:
                        root = roots[name]
                        _prepare(kind, root, scope, f"{scope['name']}-{rep}")
                        measurement, output = measure(
                            [str(pivot), "repro", *scope["args"]], root / scope["cwd"]
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
    _stop_signal_read_fd, _stop_signal_write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    signal.set_wakeup_fd(_stop_signal_write_fd)
    signal.signal(signal.SIGINT, _stop_on_signal)
    signal.signal(signal.SIGTERM, _stop_on_signal)
    sys.exit(main())
