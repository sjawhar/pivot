"""The tests of spec requirement N1 (dispatch://BET-1194/spec#n1): a namespace package shared with
an installed library works, and stays keyed, wherever it is installed, checked through the `pivot`
CLI on real installs.

A namespace package's directory under the interpreter's prefix but in no `site-packages`
directory, as a `uv pip install --target` directory that a `.pth` file in `site-packages` puts on
`sys.path`, is keyed as the same install in `site-packages` is: Pivot gives the `dist:` keys each
of a namespace package's directories that is a library path (`fingerprint.is_library_path`), the
rule by which the files in it are library code, which the walk does not hash.

Each test makes its own venv with uv over this test environment (`tests.installed_dists`) and its
own pivot project. The file runs pivot only through its CLI and imports nothing that pivot 0.10.4
lacks, so it collects with `packages/pivot/src` at that release too.
"""

import collections.abc
import importlib.metadata
import json
import pathlib

import pytest
from tests import installed_dists

# The modules of `acme-core` and `acme-b`, the two portions `acme.core` and `acme.b` of the
# namespace package `acme`; the stage writes what `acme.b.value()` returns.
_ACME_CORE = 'def value():\n    return "core"\n'
_ACME_B = 'def value():\n    return "b1"\n'


def _helper_in_site_packages(venv: installed_dists.Venv) -> tuple[pathlib.Path, list[str]]:
    """The venv's site-packages, where `uv pip install` installs with no option, and no option."""
    return venv.site_packages, []


def _helper_in_a_target_directory_under_the_prefix(
    venv: installed_dists.Venv,
) -> tuple[pathlib.Path, list[str]]:
    """`<venv>/vendor`, under the interpreter's prefix and in no `site-packages` directory, which
    the `.pth` file `vendor.pth` in site-packages puts on `sys.path` after site-packages, and the
    options that install there (`--target <venv>/vendor`)."""
    vendor = venv.python.parent.parent / "vendor"
    (venv.site_packages / "vendor.pth").write_text(f"{vendor}\n")
    return vendor, ["--target", str(vendor)]


def _helper_installed_commit(directory: pathlib.Path) -> str:
    """The commit that the `direct_url.json` of `acme-b` in `directory` names; the test fails
    unless `directory` holds one `acme-b`, installed from git."""
    found = list(importlib.metadata.Distribution.discover(name="acme-b", path=[str(directory)]))
    assert len(found) == 1, f"{directory} must hold one acme-b; it holds {len(found)}"
    text = found[0].read_text("direct_url.json")
    assert text is not None, f"acme-b in {directory} must have a direct_url.json"
    return json.loads(text)["vcs_info"]["commit_id"]


@pytest.mark.parametrize(
    "place",
    [
        pytest.param(_helper_in_a_target_directory_under_the_prefix, id="prefix-target-dir"),
        pytest.param(_helper_in_site_packages, id="site-packages"),
    ],
)
def test_a_git_install_in_a_shared_namespace_reruns_its_stage_wherever_it_lies(
    tmp_path: pathlib.Path,
    place: collections.abc.Callable[[installed_dists.Venv], tuple[pathlib.Path, list[str]]],
) -> None:
    """`acme-core`, whose one package is the namespace portion `acme.core`, is installed into
    site-packages from its path. `acme-b`, whose one package is the portion `acme.b`, is installed
    from commit C1 of its git repository, its RECORD kept, into `<venv>/vendor` with `uv pip
    install --target`, a directory under the interpreter's prefix that a `.pth` file in
    site-packages puts on `sys.path` (`prefix-target-dir`), or into site-packages
    (`site-packages`). The stage imports `acme.b` and writes `acme.b.value()`, so it reaches the
    namespace package `acme` by its top name, whose directories are `acme` in site-packages and,
    for `prefix-target-dir`, `acme` in `<venv>/vendor`. `pivot repro` runs the stage, which writes
    `b1`, refusing nothing, and its lock holds the keys of `acme-b` and `acme-core`. After one byte
    of `acme.b` changes at a new commit C2 and `acme-b` is reinstalled from it, `pivot status`
    reports the stage stale on `dist:acme-b` alone, and `pivot repro` writes `b2`. Were the
    directory under the prefix taken as the user's part of `acme`, as `is_user_code` takes a
    namespace directory outside site-packages, `acme-b` would get no key there: the stage would
    stay cached after the change, and `pivot repro` would keep `b1`."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    core = installed_dists.write_setuptools_project(
        tmp_path / "src" / "acme-core",
        "acme-core",
        "0.1",
        packages=["acme.core"],
        files={"acme/core/__init__.py": _ACME_CORE},
    )
    acme_b = installed_dists.write_setuptools_project(
        tmp_path / "src" / "acme-b",
        "acme-b",
        "0.1",
        packages=["acme.b"],
        files={"acme/b/__init__.py": _ACME_B},
    )
    installed_dists.install_directory(venv, core)
    project = installed_dists.make_project_in(tmp_path, venv, installed_dists.pipeline("acme.b"))
    directory, options = place(venv)
    installed_dists.install(
        venv, *options, installed_dists.init_git_requirement(acme_b, project.home, "acme-b")
    )
    imported = installed_dists.module_file(project, "acme.b")
    assert imported == (directory / "acme" / "b" / "__init__.py").resolve(), (
        f"the venv must import acme.b from {directory}, not {imported}"
    )

    manifest = installed_dists.assert_runs_with_no_refusal(project, "b1")
    assert installed_dists.dist_keys(manifest) == ["dist:acme-b", "dist:acme-core"], (
        f"the lock must hold the keys of acme-b and acme-core: {manifest}"
    )
    installed_dists.replace_once(acme_b / "acme" / "b" / "__init__.py", '"b1"', '"b2"')
    commit = installed_dists.commit_all(acme_b, project.home, "C2")
    installed_dists.install(
        venv,
        *options,
        installed_dists.git_requirement("acme-b", acme_b, commit),
        reinstall=True,
    )
    assert _helper_installed_commit(directory) == commit, (
        f"acme-b in {directory} must now be installed from C2 {commit}"
    )

    installed_dists.assert_stale_on(project, installed_dists.STAGE, "dist:acme-b")
    installed_dists.assert_runs_and_writes(project, "b2")
