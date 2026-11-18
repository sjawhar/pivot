"""Library code installed as a symbolic link, checked through the `pivot` CLI.

uv's `--link-mode symlink` (`UV_LINK_MODE=symlink`) installs each file of a distribution as a link
into uv's cache, inside real directories of `site-packages`: such a venv fingerprints as one whose
files uv copied, and gives the same `dist:` keys (`fingerprint.is_library_file`). flit's
`install --symlink` links a checkout's package directory, or its module file, into
`site-packages` as an editable install: its code stays user code, as in pivot 0.10.4, so an edit
to it reruns the stage.

Each test makes its venvs with uv over this test environment (`tests.installed_dists`) and its own
pivot project in each. A test that installs with uv passes the link mode to uv itself (the pod
that runs the tests by hand sets `UV_LINK_MODE=copy`, CI sets nothing). The flit test builds
flit's layout by hand, as flit 4.2.0 wrote it, since CI has no flit; it imports nothing that
pivot 0.10.4 lacks, and so runs with `packages/pivot/src` at that release too.
"""

import pathlib

import pytest
from tests import installed_dists

# `Fix_Lib`'s module `fixlib`: the function the stage calls, which adds `fixdep.base()` to a
# literal, and a class whose metaclass defines `__getattribute__`, as pydantic's `BaseModel` has
# `ModelMetaclass`, which a project's class subclasses.
_FIXLIB_WITH_METACLASS = """\
import fixdep


class Meta(type):
    def __getattribute__(cls, name):
        return super().__getattribute__(name)


class Model(metaclass=Meta):
    pass


def value():
    return "value-1:" + fixdep.base()
"""

# `fixlate`'s module, which nothing imports but the stage's body.
_FIXLATE = 'def value():\n    return ":late"\n'

# A stage that calls `fixlib.value()`, reads a constant of a project class that subclasses
# `fixlib.Model`, and imports `fixlate` in its body, which discovery does not load.
_METACLASS_PIPELINE = """\
import fixlib

from pivot import Pipeline, stage


class Thing(fixlib.Model):
    SUFFIX = ":thing"


@stage
def make() -> str:
    import fixlate

    return fixlib.value() + Thing.SUFFIX + fixlate.value()


with Pipeline("p") as pipeline:
    make()
"""


def _helper_linked_project(
    root: pathlib.Path, sources: tuple[pathlib.Path, ...], link_mode: str
) -> tuple[installed_dists.Project, bool]:
    """Make a venv at `root/venv`, install the projects at `sources` into it from their paths in
    that order with uv's `link_mode`, and make a project at `root/project` whose pipeline is
    `_METACLASS_PIPELINE`. Returns the project, and whether `fixlib`'s installed module file is a
    symbolic link."""
    root.mkdir()
    venv = installed_dists.make_venv(root / "venv")
    for source in sources:
        installed_dists.install_directory(venv, source, link_mode=link_mode)
    linked = (venv.site_packages / "fixlib" / "__init__.py").is_symlink()
    return installed_dists.make_project_in(root, venv, _METACLASS_PIPELINE), linked


def test_a_venv_uv_links_into_its_cache_fingerprints_as_one_it_copied_into(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib` (a function the stage calls, and a class with a metaclass that defines
    `__getattribute__`, which a project class subclasses), its dependency `fixdep`, and `fixlate`,
    which the stage imports in its body alone, all from their paths, installed into one venv with
    `--link-mode copy` and into another with `--link-mode symlink`, each with its own project and
    the same pipeline. In the second venv the installed module is a link into uv's cache, in the
    first it is not. `pivot repro` runs the stage in both, refusing nothing, and the two locks
    record the same code manifest, with the `dist:` keys of the three distributions. Once they
    are reinstalled into the first venv with `--link-mode symlink`, its stage is still cached,
    with no code change."""
    src = tmp_path / "src"
    fixdep = installed_dists.write_library(src, "fixdep", "fixdep", installed_dists.FIXDEP)
    fixlib = installed_dists.write_library(
        src, "Fix_Lib", "fixlib", _FIXLIB_WITH_METACLASS, dependencies=("fixdep",)
    )
    fixlate = installed_dists.write_library(src, "fixlate", "fixlate", _FIXLATE)
    sources = (fixdep, fixlib, fixlate)
    copied, copied_linked = _helper_linked_project(tmp_path / "copy", sources, "copy")
    linked, linked_linked = _helper_linked_project(tmp_path / "symlink", sources, "symlink")
    assert (copied_linked, linked_linked) == (False, True), (
        "fixlib/__init__.py must be a copy in the first venv and a link in the second: "
        + f"links {(copied_linked, linked_linked)}"
    )

    installed_dists.assert_ran_then_cached(copied, "value-1:base-1:thing:late")
    installed_dists.assert_ran_then_cached(linked, "value-1:base-1:thing:late")

    copied_manifest = installed_dists.lock_manifest(copied, installed_dists.STAGE)
    linked_manifest = installed_dists.lock_manifest(linked, installed_dists.STAGE)
    dist_keys = installed_dists.dist_keys(copied_manifest)
    assert dist_keys == ["dist:fix-lib", "dist:fixdep", "dist:fixlate"], (
        f"the copied venv's lock must hold the dist: keys of the three: {copied_manifest}"
    )
    assert linked_manifest == copied_manifest, (
        f"the linked venv's lock must equal the copied one's:\nlinked {linked_manifest}\n"
        + f"copied {copied_manifest}"
    )

    for source in sources:
        installed_dists.install_directory(copied.venv, source, reinstall=True, link_mode="symlink")
    relinked = (copied.venv.site_packages / "fixlib" / "__init__.py").is_symlink()
    assert relinked, "fixlib/__init__.py must be a link once reinstalled with --link-mode symlink"
    installed_dists.assert_marks_nothing(copied, installed_dists.STAGE)


@pytest.mark.parametrize(
    ("link", "source"),
    [
        pytest.param("flitlib", "flitlib/__init__.py", id="package-directory"),
        pytest.param("flitlib.py", "flitlib.py", id="module-file"),
    ],
)
def test_an_edit_to_the_checkout_flit_links_into_site_packages_reruns_the_stage(
    tmp_path: pathlib.Path, link: str, source: str
) -> None:
    """`flitlib`, installed as flit's `install --symlink` installs it
    (`installed_dists.flit_symlink_install`): a link in `site-packages` to the checkout's package
    directory, or to its module file, which the RECORD of an editable install lists. The stage
    calls `flitlib.value()`, which the walk hashes as user code. After one byte of the checkout's
    source changes, the stage is stale on `func:flitlib.value` alone, and `pivot repro` runs it
    and writes the new value."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    checkout = tmp_path / "flitlib"
    (checkout / source).parent.mkdir(parents=True, exist_ok=True)
    (checkout / source).write_text(installed_dists.FIXLIB_ALONE)
    installed_dists.flit_symlink_install(venv.site_packages, checkout, link)
    project = installed_dists.make_project_in(tmp_path, venv, installed_dists.pipeline("flitlib"))
    installed_dists.assert_ran_then_cached(project, "value-1")

    installed_dists.replace_once(checkout / source, '"value-1"', '"value-2"')

    installed_dists.assert_stale_on(project, installed_dists.STAGE, "func:flitlib.value")
    installed_dists.assert_runs_and_writes(project, "value-2", "the second `pivot repro`")


def test_a_linked_library_whose_record_is_lost_is_walked_as_user_code(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib`, installed from its path with `--link-mode symlink`, has lost its RECORD, so no
    RECORD lists its module file, a link into uv's cache, and the file keeps the decision of its
    resolved path: user code, which the walk hashes. `pivot repro` runs the stage, refusing
    nothing; after one byte of the linked file changes (written through the link, into uv's cache
    entry of this test's own build), the stage is stale on `func:fixlib.value` alone, and `pivot
    repro` runs it and writes the new value. In a copied venv the stage is refused instead, as
    `test_a_broken_install_the_stage_needs_is_refused_and_unsafe_fingerprinting_warns`
    (`reached-without-record`) in `test_installed_distributions.py` checks."""
    venv = installed_dists.make_venv(tmp_path / "venv")
    source = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE
    )
    installed_dists.install_directory(venv, source, link_mode="symlink")
    module = venv.site_packages / "fixlib" / "__init__.py"
    assert module.is_symlink(), f"{module} must be a link into uv's cache"
    installed_dists.damage(venv.site_packages / "fix_lib-0.1.dist-info" / "RECORD", None)
    project = installed_dists.make_project_in(tmp_path, venv, installed_dists.pipeline("fixlib"))
    installed_dists.assert_ran_then_cached(project, "value-1")

    installed_dists.replace_once(module, '"value-1"', '"value-2"')

    installed_dists.assert_stale_on(project, installed_dists.STAGE, "func:fixlib.value")
    installed_dists.assert_runs_and_writes(project, "value-2", "the second `pivot repro`")


def _helper_refusal(project: installed_dists.Project, site: pathlib.Path) -> str:
    """The refusal `pivot repro --json` exits 1 with in `project`, from its `Error:` line, with
    the venv's resolved site-packages `site` written `{site}`; the test fails unless it exits 1."""
    ran = installed_dists.run_pivot(project, "repro", "--json")
    shown = installed_dists.shown(ran)
    assert ran.returncode == 1, f"pivot must refuse {installed_dists.STAGE}:\n{shown}"
    errors = [line for line in ran.stderr.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1, f"pivot must print one Error: line:\n{shown}"
    return errors[0].replace(str(site.resolve()), "{site}")


def test_a_linked_library_whose_record_is_lost_is_refused_as_a_copied_one_is(
    tmp_path: pathlib.Path,
) -> None:
    """`Fix_Lib` installed from its path into one venv with `--link-mode copy` and into another
    with `--link-mode symlink`, each with its own project and the same pipeline, and each with its
    RECORD lost. A venv uv links into its cache fingerprints as one it copied into (BET-1194 spec
    v9, "uv's symlink link mode"): the copied venv refuses the stage, since Pivot cannot tell
    which files the reached distribution owns, so the linked venv refuses it with the same
    refusal. Walking the linked library as user code instead hashes its Python source only: a
    change to a package data file it reads, or to any other file it ships that is not Python
    source, leaves the stage cached with the old output."""
    source = installed_dists.write_library(
        tmp_path / "src", "Fix_Lib", "fixlib", installed_dists.FIXLIB_ALONE
    )
    refusals = dict[str, str]()
    for link_mode in ("copy", "symlink"):
        root = tmp_path / link_mode
        root.mkdir()
        venv = installed_dists.make_venv(root / "venv")
        installed_dists.install_directory(venv, source, link_mode=link_mode)
        linked = (venv.site_packages / "fixlib" / "__init__.py").is_symlink()
        assert linked == (link_mode == "symlink"), (
            f"fixlib/__init__.py must be a link only with --link-mode symlink: {link_mode} {linked}"
        )
        installed_dists.damage(venv.site_packages / "fix_lib-0.1.dist-info" / "RECORD", None)
        project = installed_dists.make_project_in(root, venv, installed_dists.pipeline("fixlib"))
        refusals[link_mode] = _helper_refusal(project, venv.site_packages)

    assert refusals["symlink"] == refusals["copy"], (
        f"the linked venv must refuse as the copied one does:\nlinked {refusals['symlink']}\n"
        + f"copied {refusals['copy']}"
    )
