"""Every file a `.pvt` drives is written inside the project.

Each test drives the installed `pivot` binary in a real project, the way a user does. A `.pvt`
is saved input: a clone carries whatever its author committed, so a sidecar whose ``path:``
leads out of the project, or names no file at all, must make no command write, replace or
delete anything outside the file it names inside the project.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import uuid
from typing import TYPE_CHECKING

import boto3
import pytest
import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """\
from pivot.compose import Pipeline, stage


@stage
def copy(value: str) -> str:
    return value


with Pipeline("p") as pipeline:
    copy(pipeline.input("value.txt", t=str))
"""


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(extra)
    return env


def _pivot(
    cwd: pathlib.Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PIVOT, *args],
        cwd=cwd,
        env=env or _env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _names(paths: Iterable[pathlib.Path]) -> list[str]:
    return sorted(path.name for path in paths)


def _project(root: pathlib.Path, files: dict[str, str] | None = None) -> pathlib.Path:
    """A git project with pivot initialised and ``files`` written in it."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _ok(_pivot(root, "init"))
    for rel, text in (files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _s3_remote(root: pathlib.Path, endpoint: str) -> dict[str, str]:
    """Make a moto bucket the project's default remote, and return the env that reaches it."""
    env = _env(
        AWS_ENDPOINT_URL=endpoint,
        AWS_ACCESS_KEY_ID="x",
        AWS_SECRET_ACCESS_KEY="x",
        AWS_DEFAULT_REGION="us-east-1",
    )
    bucket = f"pvt-escape-{uuid.uuid4().hex[:8]}"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    )
    s3.create_bucket(Bucket=bucket)  # pyright: ignore[reportAttributeAccessIssue] - boto3 ships no stubs
    _ok(_pivot(root, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    _ok(_pivot(root, "config", "set", "default_remote", "origin"))
    return env


def _empty_local_cache(root: pathlib.Path) -> None:
    """Remove every cached object, so a pull has to fetch what it restores."""
    files_dir = root / ".pivot" / "cache" / "files"
    for directory, _subdirs, names in os.walk(files_dir):
        os.chmod(directory, 0o755)
        for name in names:
            os.unlink(os.path.join(directory, name))


def _cached_hash(root: pathlib.Path, content: str, env: dict[str, str] | None = None) -> str:
    """Put ``content`` in the cache through `pivot track` and return its hash.

    With ``env`` the content is pushed to the project's remote while its sidecar still exists,
    and the local cache is emptied: a later pull fetches it from the remote.
    """
    seed = root / "seed.csv"
    seed.write_text(content)
    _ok(_pivot(root, "track", "seed.csv"))
    sidecar = root / "seed.csv.pvt"
    pvt_data = yaml.safe_load(sidecar.read_text())
    if env is not None:
        _ok(_pivot(root, "push", env=env))
    seed.unlink()
    sidecar.unlink()
    if env is not None:
        _empty_local_cache(root)
    hash_: str = pvt_data["hash"]
    return hash_


def _write_pvt(pvt_path: pathlib.Path, data_path: str, hash_: str, size: int) -> None:
    """Write a sidecar by hand, the way a hand-edited or hostile commit carries it."""
    pvt_path.write_text(f"path: {data_path}\nhash: {hash_}\nsize: {size}\n")


def _escape_link(
    tmp_path: pathlib.Path, root: pathlib.Path, env: dict[str, str] | None = None
) -> pathlib.Path:
    """The issue's layout: `link -> ../outside`, and `x.pvt` with `path: link/x.csv`.

    Returns the directory outside the project that the link points to.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    hash_ = _cached_hash(root, "seed\n", env)
    (root / "link").symlink_to(pathlib.Path("..") / "outside", target_is_directory=True)
    _write_pvt(root / "x.pvt", "link/x.csv", hash_, 5)
    return outside


def _nameless_pvt(
    root: pathlib.Path, sidecar: str, env: dict[str, str] | None = None
) -> pathlib.Path:
    """Write ``sidecar`` with `path: .`, beside a file that must survive. Returns its directory."""
    hash_ = _cached_hash(root, "seed\n", env)
    pvt_path = root / sidecar
    pvt_path.parent.mkdir(parents=True, exist_ok=True)
    (pvt_path.parent / "precious.txt").write_text("keep me\n")
    _write_pvt(pvt_path, ".", hash_, 5)
    return pvt_path.parent


# --- checkout ---


def test_checkout_writes_nothing_outside_the_project_through_a_symlinked_directory(
    tmp_path: pathlib.Path,
) -> None:
    """The issue's probe: `path: link/x.csv` through `link -> ../outside` restores nothing."""
    root = _project(tmp_path / "project")
    outside = _escape_link(tmp_path, root)

    proc = _pivot(root, "checkout")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "Skipping x.pvt" in out, out
    assert "outside the project root" in out, out
    assert _names(outside.iterdir()) == [], _names(outside.iterdir())


def test_checkout_force_keeps_a_file_outside_the_project_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    """`--force` replaces a file it may restore; it may not restore one outside the project."""
    root = _project(tmp_path / "project")
    outside = _escape_link(tmp_path, root)
    victim = outside / "x.csv"
    victim.write_text("VICTIM\n")

    proc = _pivot(root, "checkout", "--force")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "Skipping x.pvt" in out, out
    assert victim.read_text() == "VICTIM\n"
    assert _names(outside.iterdir()) == ["x.csv"], _names(outside.iterdir())


@pytest.mark.parametrize(
    "sidecar",
    [
        pytest.param("dot.pvt", id="project-root"),
        pytest.param("sub/dot.pvt", id="subdirectory"),
    ],
)
def test_checkout_force_keeps_the_directory_a_pvt_that_names_no_file_sits_in(
    tmp_path: pathlib.Path, sidecar: str
) -> None:
    """`path: .` names the .pvt's own directory; restoring it would replace that directory."""
    root = _project(tmp_path / "project")
    directory = _nameless_pvt(root, sidecar)

    proc = _pivot(root, "checkout", "--force")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert f"Skipping {sidecar}" in out, out
    assert "names no file" in out, out
    assert directory.is_dir(), f"{directory} is no longer a directory"
    assert (directory / "precious.txt").read_text() == "keep me\n"
    assert (root / ".git").is_dir(), "the project's git directory survives"


def test_checkout_force_keeps_the_directory_a_sidecar_named_dot_pvt_sits_in(
    tmp_path: pathlib.Path,
) -> None:
    root = _project(tmp_path / "project")
    (root / "sub").mkdir()
    (root / "sub" / "precious.txt").write_text("keep me\n")
    hash_ = _cached_hash(root, "seed\n")
    (root / "sub" / ".pvt").write_text(f'path: ""\nhash: {hash_}\nsize: 5\n')

    proc = _pivot(root, "checkout", "--force")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert (root / "sub").is_dir(), out
    assert (root / "sub" / "precious.txt").read_text() == "keep me\n"


def test_checkout_restores_the_file_a_pvt_names_beside_itself_without_a_warning(
    tmp_path: pathlib.Path,
) -> None:
    """The form `pivot track` writes is restored as before, and says nothing about its path."""
    root = _project(tmp_path / "project", {"data/values.csv": "a\n1\n"})
    _ok(_pivot(root, "track", "data/values.csv"))
    (root / "data" / "values.csv").unlink()

    proc = _pivot(root, "checkout")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert (root / "data" / "values.csv").read_text() == "a\n1\n"
    assert "Skipping" not in out, out
    assert "is not the .pvt's own name" not in out, out


def test_checkout_restores_a_file_under_an_in_project_symlink_and_names_the_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    """A path inside the project still restores, and the reader is told which name each command
    uses, because `checkout y.pvt`, `get`, `update` and `push` take the sidecar's own name."""
    root = _project(tmp_path / "project")
    hash_ = _cached_hash(root, "yy\n")
    (root / "data_real").mkdir()
    (root / "dlink").symlink_to("data_real", target_is_directory=True)
    _write_pvt(root / "y.pvt", "dlink/y.csv", hash_, 3)

    proc = _pivot(root, "checkout")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert (root / "data_real" / "y.csv").read_text() == "yy\n"
    assert "y.pvt: its path 'dlink/y.csv' is not the .pvt's own name 'y'" in out, out
    assert out.count("is not the .pvt's own name") == 1, out


# --- pull ---


def test_pull_writes_nothing_outside_the_project_through_a_symlinked_directory(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """`pull` fetches from the remote and then checks out: the same rule holds there."""
    root = _project(tmp_path / "project", {"pipeline.py": _PIPELINE})
    env = _s3_remote(root, moto_services["s3"])
    outside = _escape_link(tmp_path, root, env)

    proc = _pivot(root, "pull", env=env)
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "1 transferred" in out, "the content came from the remote"
    assert "Skipping x.pvt" in out, out
    assert "outside the project root" in out, out
    assert _names(outside.iterdir()) == [], _names(outside.iterdir())


def test_pull_force_keeps_a_file_outside_the_project_unchanged(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """`pull --force` passes force on to checkout, which still may not write outside."""
    root = _project(tmp_path / "project", {"pipeline.py": _PIPELINE})
    env = _s3_remote(root, moto_services["s3"])
    outside = _escape_link(tmp_path, root, env)
    victim = outside / "x.csv"
    victim.write_text("VICTIM\n")

    proc = _pivot(root, "pull", "--force", env=env)
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "Skipping x.pvt" in out, out
    assert victim.read_text() == "VICTIM\n"


def test_pull_force_keeps_the_project_when_a_pvt_names_no_file(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """A `path: .` sidecar reaches checkout through pull as well."""
    root = _project(tmp_path / "project", {"pipeline.py": _PIPELINE})
    env = _s3_remote(root, moto_services["s3"])
    _nameless_pvt(root, "dot.pvt", env)

    proc = _pivot(root, "pull", "--force", env=env)
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "Skipping dot.pvt" in out, out
    assert "names no file" in out, out
    assert root.is_dir(), "the project is still a directory"
    assert (root / "precious.txt").read_text() == "keep me\n"


# --- repro --checkout-missing ---


def test_repro_checkout_missing_writes_nothing_outside_the_project(
    tmp_path: pathlib.Path,
) -> None:
    """The pre-flight restores tracked files that are missing: not one outside the project."""
    root = _project(tmp_path / "project", {"pipeline.py": _PIPELINE, "data/raw/value.txt": "V"})
    outside = _escape_link(tmp_path, root)

    proc = _pivot(root, "repro", "--checkout-missing")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "Skipping x.pvt" in out, out
    assert "outside the project root" in out, out
    assert _names(outside.iterdir()) == [], _names(outside.iterdir())


def test_repro_checkout_missing_keeps_a_file_outside_the_project_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    """A file already outside the project is neither taken for the tracked file nor replaced."""
    root = _project(tmp_path / "project", {"pipeline.py": _PIPELINE, "data/raw/value.txt": "V"})
    outside = _escape_link(tmp_path, root)
    victim = outside / "x.csv"
    victim.write_text("VICTIM\n")

    proc = _pivot(root, "repro", "--checkout-missing")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "Skipping x.pvt" in out, out
    assert victim.read_text() == "VICTIM\n"


def test_repro_checkout_missing_keeps_the_project_when_a_pvt_names_no_file(
    tmp_path: pathlib.Path,
) -> None:
    """`path: .` names the project root here, which the pre-flight must not read or restore."""
    root = _project(tmp_path / "project", {"pipeline.py": _PIPELINE, "data/raw/value.txt": "V"})
    _nameless_pvt(root, "dot.pvt")

    proc = _pivot(root, "repro", "--checkout-missing")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "Skipping dot.pvt" in out, out
    assert "names no file" in out, out
    assert (root / "precious.txt").read_text() == "keep me\n"
    assert (root / ".git").is_dir(), "the project's git directory survives"


# --- track ---


def test_track_refuses_a_file_whose_pvt_would_land_outside_the_project(
    tmp_path: pathlib.Path,
) -> None:
    """`data -> ../ext` puts the sidecar in `ext`, where no pivot command reads it again."""
    root = _project(tmp_path / "project")
    external = tmp_path / "ext" / "raw"
    external.mkdir(parents=True)
    (root / "data").symlink_to(pathlib.Path("..") / "ext", target_is_directory=True)
    (external / "v.csv").write_text("VALUE\n")

    proc = _pivot(root, "track", "data/raw/v.csv")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 1, out
    assert "its .pvt would be written outside the project" in out, out
    assert "Tracked:" not in out, out
    assert list((tmp_path / "ext").rglob("*.pvt")) == []
    assert list(root.rglob("*.pvt")) == []


def test_track_keeps_tracking_through_a_symlinked_directory_inside_the_project(
    tmp_path: pathlib.Path,
) -> None:
    """The sidecar lands inside the project, so this stays a warning, not a refusal."""
    root = _project(tmp_path / "project", {"data_real/raw/v.csv": "VALUE\n"})
    (root / "dlink").symlink_to("data_real", target_is_directory=True)

    proc = _pivot(root, "track", "dlink/raw/v.csv")
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "is inside a symlinked directory" in out, out
    assert (root / "data_real" / "raw" / "v.csv.pvt").is_file()
