"""Steer XXH64 to a chosen digest, so tests can show a cache tree's name vouches for nothing.

XXH64's finalization is invertible step by step: 16 bytes placed right after the last full
32-byte stripe, followed by fewer than 16 bytes, can be solved for any target digest. A forger
picks the first 8 at random from an alphabet and computes the other 8, retrying until those land
in the alphabet too. ``forge_value`` does that for one string a document builder embeds; every
result is checked against the ``xxhash`` library. Adapted from the round-5 reviewer's
``xxcollide.py`` (Dispatch BET-237, #516).
"""

from __future__ import annotations

import random
import string
from typing import TYPE_CHECKING, cast

import xxhash

from pivot.storage import cache

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.types import DirHash

_M = (1 << 64) - 1
_P1 = 0x9E3779B185EBCA87
_P2 = 0xC2B2AE3D27D4EB4F
_P3 = 0x165667B19E3779F9
_P4 = 0x85EBCA77C2B2AE63
_P5 = 0x27D4EB2F165667C5
_P1I = pow(_P1, -1, 1 << 64)
_P2I = pow(_P2, -1, 1 << 64)
_ALPHABET = (string.ascii_letters + string.digits).encode()
_ALLOWED = frozenset(_ALPHABET)
_SENTINEL = "\x00" * 16  # pivot's JSON escapes it, so it is found in the encoded document
_ENCODED_SENTINEL = b"\\u0000" * 16


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & _M


def _rotr(x: int, r: int) -> int:
    return ((x >> r) | (x << (64 - r))) & _M


def _round(acc: int, lane: int) -> int:
    return (_rotl((acc + lane * _P2) & _M, 31) * _P1) & _M


def _stripes_state(data: bytes, total_length: int) -> int:
    """XXH64 (seed 0) state after ``data``'s 32-byte stripes, with the input's total length."""
    v1, v2, v3, v4 = (_P1 + _P2) & _M, _P2, 0, (-_P1) & _M
    for offset in range(0, len(data), 32):
        lanes = [
            int.from_bytes(data[offset + 8 * i : offset + 8 * i + 8], "little") for i in range(4)
        ]
        accs = (v1, v2, v3, v4)
        v1, v2, v3, v4 = (_round(acc, lane) for acc, lane in zip(accs, lanes, strict=True))
    h = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & _M
    for v in (v1, v2, v3, v4):
        h = ((h ^ _round(0, v)) * _P1 + _P4) & _M
    return (h + total_length) & _M


def _unavalanche(h: int) -> int:
    h ^= h >> 32
    h = (h * pow(_P3, -1, 1 << 64)) & _M
    x = h
    for _ in range(3):
        x = h ^ (x >> 29)
    h = (x * _P2I) & _M
    return h ^ (h >> 33)


def _undo_tail(h: int, suffix: bytes) -> int:
    """The state before XXH64's finalization consumed ``suffix``, given the state after it."""
    whole = len(suffix) - len(suffix) % 8
    rest = suffix[whole:]
    singles = rest[4:] if len(rest) >= 4 else rest
    for byte in reversed(singles):
        h = _rotr((h * _P1I) & _M, 11) ^ ((byte * _P5) & _M)
    if len(rest) >= 4:
        word = int.from_bytes(rest[:4], "little")
        h = _rotr(((h - _P3) * _P2I) & _M, 23) ^ ((word * _P1) & _M)
    for offset in reversed(range(0, whole, 8)):
        lane = int.from_bytes(suffix[offset : offset + 8], "little")
        h = _rotr(((h - _P4) * _P1I) & _M, 27) ^ _round(0, lane)
    return h


def forge_value(build: Callable[[str], bytes], target_hex: str, *, prefix: str = "") -> str:
    """A string starting with ``prefix`` for which ``xxh64(build(value))`` is ``target_hex``.

    ``build`` embeds its argument contiguously and unescaped (alphanumeric characters are), with
    fewer than 16 bytes after it, as ``json.dumps`` does for a manifest's last string field.
    """
    probe = build(prefix + _SENTINEL)
    start = probe.index(_ENCODED_SENTINEL) - len(prefix.encode())
    head, suffix = probe[:start], probe[start + len(prefix.encode()) + len(_ENCODED_SENTINEL) :]
    assert len(suffix) < 16, "the forged bytes must end in XXH64's final, partial stripe"
    filler = prefix.encode() + b"a" * ((-start - len(prefix.encode())) % 32)
    stripes = head + filler
    total = len(stripes) + 16 + len(suffix)
    h0 = _stripes_state(stripes, total)
    h2 = _undo_tail(_unavalanche(int(target_hex, 16)), suffix)
    rng = random.Random(0)
    while True:
        first = bytes(rng.choices(_ALPHABET, k=8))
        h1 = (_rotl(h0 ^ _round(0, int.from_bytes(first, "little")), 27) * _P1 + _P4) & _M
        lane = _rotr(((h2 - _P4) * _P1I) & _M, 27) ^ h1
        second = ((_rotr((lane * _P1I) & _M, 31) * _P2I) & _M).to_bytes(8, "little")
        if all(byte in _ALLOWED for byte in second):
            break
    value = (filler + first + second).decode()
    document = build(value)
    assert document == stripes + first + second + suffix, "the value sits where it was solved"
    assert xxhash.xxh64(document).hexdigest() == target_hex, "the library agrees"
    return value


def _encode(entries: list[dict[str, object]], empty_dirs: list[str] | None) -> bytes:
    """A manifest as written, in pivot's own JSON encoding (imported, so a change to it changes
    what the forgery targets): what the round-5 gate hashed."""
    return cache._manifest_json(entries, empty_dirs or [])


def _canonical(entries: list[dict[str, object]], empty_dirs: list[str] | None) -> bytes:
    """What pivot's own XXH64 gate hashes for a manifest: ``cache._canonical_manifest``, imported,
    so the forgery targets exactly the gate's bytes, its order and fields included."""
    output_hash: dict[str, object] = {"hash": "", "manifest": entries}
    if empty_dirs:
        output_hash["empty_dirs"] = empty_dirs
    canonical = cache._canonical_manifest(cast("DirHash", cast("object", output_hash)))
    assert canonical is not None, "a canonical forgery has a canonical form"
    return canonical


def pad_to(
    entries: list[dict[str, object]], target_hex: str, empty_dirs: list[str] | None = None
) -> list[dict[str, object]]:
    """``entries`` with a key pivot never writes added to the last one, padded so the whole
    manifest, that key included, hashes to ``target_hex`` (the round-5 reviewer's forgery: it
    passes a gate that hashes the manifest as written, not one that hashes pivot's four fields)."""

    def build(value: str) -> bytes:
        return _encode([*entries[:-1], {**entries[-1], "zzz": value}], empty_dirs)

    return [*entries[:-1], {**entries[-1], "zzz": forge_value(build, target_hex)}]


def steer_with_relpath(
    entries: list[dict[str, object]],
    extra: dict[str, object],
    target_hex: str,
    empty_dirs: list[str] | None = None,
) -> list[dict[str, object]]:
    """``entries`` plus ``extra`` as their last entry, its relpath ``zz`` and 16 free characters
    chosen so pivot's canonical form of the manifest hashes to ``target_hex``, with no key pivot
    does not write: a forgery that survives dropping unknown keys, but names a file the tree
    lacks. ``zz`` sorts the entry last in the canonical form, where the free characters must be."""

    def build(value: str) -> bytes:
        return _canonical([*entries, {**extra, "relpath": value}], empty_dirs)

    return [*entries, {**extra, "relpath": forge_value(build, target_hex, prefix="zz")}]


def steer_with_size(
    entries: list[dict[str, object]], target_hex: str, empty_dirs: list[str] | None = None
) -> list[dict[str, object]]:
    """``entries``, sorted by relpath as pivot's canonical form lists them, with the last one's
    ``size`` replaced by a string of 16 free characters chosen so that canonical form hashes to
    ``target_hex``: only pivot's four fields, and every relpath one the tree holds, so nothing
    but the tree's marker tells it from the tree's manifest. A `.pvt` accepts a string size; lock
    reading refuses it."""
    ordered = sorted(entries, key=lambda entry: str(entry["relpath"]))

    def build(value: str) -> bytes:
        return _canonical([*ordered[:-1], {**ordered[-1], "size": value}], empty_dirs)

    return [*ordered[:-1], {**ordered[-1], "size": forge_value(build, target_hex)}]
