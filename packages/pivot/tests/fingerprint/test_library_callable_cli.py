# pyright: reportImplicitStringConcatenation=false

"""Library callable CLI outcomes checked against main 170a2ebc.

Run this same file with each build's interpreter and installed CLI. The literal expectations
cover the main contract; the library support files live outside the project in site-packages.
"""

import json
import pathlib
import subprocess

import pytest
from tests import helpers

_HEADER = "from pivot import Pipeline, stage\nimport library_api\n"
_FOOTER = '\nwith Pipeline("t") as pipeline:\n    use(pipeline.input("lines.txt", t=str))\n'
_ROOT = _HEADER + "\nuse = stage(library_api.use)\n" + _FOOTER
_HELPER = (
    _HEADER
    + "\nHELPER = library_api.use\n"
    + "\n@stage\ndef use(text: str) -> str:\n    return HELPER(text)\n"
    + _FOOTER
)

# library code, project code, outcome after a project setting edit, output after that edit.
_CASES = {
    "plain-library-root": (
        'def use(text: str) -> str:\n    return text + "!"\n',
        _ROOT,
        "cached",
        "x!",
    ),
    "library-root-project-setting": (
        "import settings\n\ndef use(text: str) -> str:\n    return text + settings.VALUE\n",
        _ROOT,
        "ran",
        "xBB",
    ),
    "cached-library-root-project-setting": (
        "import functools\nimport settings\n\n@functools.cache\n"
        "def use(text: str) -> str:\n    return text + settings.VALUE\n",
        _ROOT,
        "ran",
        "xBB",
    ),
    "lru-library-root-project-setting": (
        "import functools\nimport settings\n\n@functools.lru_cache(maxsize=8)\n"
        "def use(text: str) -> str:\n    return text + settings.VALUE\n",
        _ROOT,
        "ran",
        "xBB",
    ),
    "locked-library-helper": (
        "import functools\nimport threading\n\ndef serialized(fn):\n"
        "    lock = threading.Lock()\n    @functools.wraps(fn)\n"
        "    def wrapper(text: str) -> str:\n        with lock:\n            return fn(text)\n"
        '    return wrapper\n\n@serialized\ndef use(text: str) -> str:\n    return text + "!"\n',
        _HELPER,
        "cached",
        "x!",
    ),
    "library-wraps-library-helper": (
        "import functools\n\ndef original(text: str) -> str:\n    return text\n\n"
        '@functools.wraps(original)\ndef use(text: str) -> str:\n    return original(text) + "!"\n',
        _HELPER,
        "cached",
        "x!",
    ),
    "user-wraps-library-helper": (
        "def echo(text: str) -> str:\n    return text\n",
        _HEADER
        + "import functools\nimport settings\n\nSUFFIX = settings.VALUE\n"
        + "\n@functools.wraps(library_api.echo)\n"
        + "def wrapped(text: str) -> str:\n    return library_api.echo(text) + SUFFIX\n"
        + "wrapped.__module__ = __name__\n\n@stage\ndef use(text: str) -> str:\n"
        + "    return wrapped(text)\n"
        + _FOOTER,
        "ran",
        "xBB",
    ),
    "library-lru-over-user-helper": (
        "import functools\n\ndef cached(fn):\n    return functools.lru_cache(maxsize=8)(fn)\n",
        _HEADER
        + "import settings\n\n@library_api.cached\ndef helper(text: str) -> str:\n"
        + "    return text + settings.VALUE\n\n@stage\ndef use(text: str) -> str:\n"
        + "    return helper(text)\n"
        + _FOOTER,
        "ran",
        "xBB",
    ),
}


def _repro(project: pathlib.Path, home: pathlib.Path, env: dict[str, str]) -> str:
    result = helpers.run_pivot(project, home, "repro", "--json", extra_env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    statuses = [
        event["status"]
        for line in result.stdout.splitlines()
        if line.startswith("{")
        and (event := json.loads(line)).get("type") == "stage_complete"
        and event["stage"] == "t/use"
    ]
    assert len(statuses) == 1, result.stdout + result.stderr
    return statuses[0]


@pytest.mark.parametrize("shape", sorted(_CASES))
def test_library_callable_matches_mains_run_cache_and_rerun(
    tmp_path: pathlib.Path, shape: str
) -> None:
    library, pipeline, edited_status, edited_output = _CASES[shape]
    project = tmp_path / "project"
    support = tmp_path / "support/site-packages"
    home = tmp_path / "home"
    support.mkdir(parents=True)
    (project / "data/raw").mkdir(parents=True)
    (project / "data/raw/lines.txt").write_text("x")
    (project / "pipeline.py").write_text(pipeline)
    (project / "settings.py").write_text('VALUE = "A"\n')
    (support / "library_api.py").write_text(library)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    env = {"PYTHONPATH": f"{project}:{support}", "PYTHONHASHSEED": "1"}
    init = helpers.run_pivot(project, home, "init", extra_env=env)
    assert init.returncode == 0, init.stdout + init.stderr
    output = project / "data/t/use.txt"
    first_output = "xA" if edited_status == "ran" else "x!"

    assert _repro(project, home, env) == "ran"
    assert output.read_text() == first_output
    assert _repro(project, home, env) == "cached"
    (project / "settings.py").write_text('VALUE = "BB"\n')
    assert _repro(project, home, env) == edited_status
    assert output.read_text() == edited_output
    assert _repro(project, home, {**env, "PYTHONHASHSEED": "7"}) == "cached"
    reset = helpers.run_pivot(project, home, "fingerprint", "reset", extra_env=env)
    assert reset.returncode == 0, reset.stdout + reset.stderr
    assert _repro(project, home, env) == "cached"
    forced = helpers.run_pivot(project, home, "repro", "--force", "--json", extra_env=env)
    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert output.read_text() == edited_output

    # Library source edits alone remain cached, including a library root, as on main.
    if '"!"' in library:
        (support / "library_api.py").write_text(library.replace('"!"', '"??"'))
        assert _repro(project, home, env) == "cached"
        assert output.read_text() == "x!"


def test_library_root_keeps_mains_private_lock_refusal(tmp_path: pathlib.Path) -> None:
    project = tmp_path / "project"
    support = tmp_path / "support/site-packages"
    home = tmp_path / "home"
    support.mkdir(parents=True)
    (project / "data/raw").mkdir(parents=True)
    (project / "data/raw/lines.txt").write_text("x")
    (project / "pipeline.py").write_text(_ROOT)
    (support / "library_api.py").write_text(_CASES["locked-library-helper"][0])
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    env = {"PYTHONPATH": f"{project}:{support}"}
    init = helpers.run_pivot(project, home, "init", extra_env=env)
    assert init.returncode == 0, init.stdout + init.stderr
    result = helpers.run_pivot(project, home, "repro", "--json", extra_env=env)
    assert result.returncode != 0
    assert "non-scalar variable 'lock'" in result.stdout + result.stderr
    assert not (project / "data/t/use.txt").exists()


def test_user_wraps_library_keeps_its_own_project_reads(tmp_path: pathlib.Path) -> None:
    """wraps copies a library __module__, but the wrapper's code still belongs to the project."""
    project = tmp_path / "project"
    support = tmp_path / "support/site-packages"
    home = tmp_path / "home"
    support.mkdir(parents=True)
    (project / "data/raw").mkdir(parents=True)
    (project / "data/raw/lines.txt").write_text("x")
    library, pipeline, _, _ = _CASES["user-wraps-library-helper"]
    (project / "pipeline.py").write_text(pipeline.replace("wrapped.__module__ = __name__", ""))
    (project / "settings.py").write_text('VALUE = "A"\n')
    (support / "library_api.py").write_text(library)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    env = {"PYTHONPATH": f"{project}:{support}"}
    init = helpers.run_pivot(project, home, "init", extra_env=env)
    assert init.returncode == 0, init.stdout + init.stderr
    output = project / "data/t/use.txt"
    assert _repro(project, home, env) == "ran"
    assert output.read_text() == "xA"
    assert _repro(project, home, env) == "cached"
    (project / "settings.py").write_text('VALUE = "BB"\n')
    assert _repro(project, home, env) == "ran"
    assert output.read_text() == "xBB"
    reset = helpers.run_pivot(project, home, "fingerprint", "reset", extra_env=env)
    assert reset.returncode == 0, reset.stdout + reset.stderr
    assert _repro(project, home, env) == "cached"
