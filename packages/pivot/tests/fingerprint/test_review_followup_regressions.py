import datetime
import enum
import functools
import json
import pathlib
import subprocess
from typing import Any

import pandas as pd
import pytest
import pytz
from tests import helpers

from pivot import fingerprint

_LABEL = "A"


_Field = enum.Enum("_Field", {"name": "n", "value": "v"})


class _Labelled(enum.Enum):
    A = 1

    @enum.property
    def label(self) -> str:
        return _LABEL


_FIELD_VALUE = functools.partial(str, _Field.name)


def _names_enum() -> str:
    return str(_Field.name)


def _captures_enum() -> str:
    return _FIELD_VALUE()


def _reads_enum_property() -> str:
    return _Labelled.A.label


@pytest.mark.parametrize("function", [_names_enum, _captures_enum, _reads_enum_property])
def test_enum_redirects_and_properties_are_not_opaque_descriptors(function: Any) -> None:
    fingerprint.get_stage_fingerprint(function)


@pytest.mark.parametrize("kind", ["enum-property", "user-generic"])
def test_a_held_type_tracks_its_user_method_after_reload(tmp_path: pathlib.Path, kind: str) -> None:
    """Real CLI imports must see an edited getter or the origin of a typing alias."""
    root, home = tmp_path / "project", tmp_path / "home"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    initialized = helpers.run_pivot(root, home, "init")
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "text.txt").write_text("hello")
    if kind == "enum-property":
        source = (
            "import enum\nLABEL = 'A'\nclass Config(enum.Enum):\n    A = 1\n"
            + "    @enum.property\n    def label(self):\n        return LABEL\n"
        )
        call = "return Config.A.label + text"
        held = ""
    else:
        source = "class Config[T]:\n    def label(self):\n        return 'A'\n"
        call = "return cls().label() + text"
        held = "HELD = functools.partial(use_type, Config[int])\n"
    (root / "helpers.py").write_text(source)
    (root / "pipeline.py").write_text(
        "import functools\nfrom pivot import Pipeline, stage\nfrom helpers import Config\n"
        + f"def use_type({'cls, ' if held else ''}text):\n    {call}\n"
        + held
        + "@stage\ndef use(text: str) -> str:\n"
        + f"    return {'HELD' if held else 'use_type'}(text)\n"
        + 'with Pipeline("t") as pipeline:\n    use(pipeline.input("text.txt", t=str))\n'
    )
    for status in ("ran", "cached", "ran"):
        if status == "ran" and (root / "data" / "t" / "use.txt").exists():
            (root / "helpers.py").write_text(source.replace("'A'", "'B'"))
        result = helpers.run_pivot(root, home, "repro", "--json")
        assert result.returncode == 0, result.stdout + result.stderr
        statuses = [
            json.loads(line)["status"]
            for line in result.stdout.splitlines()
            if line.startswith("{") and json.loads(line).get("type") == "stage_complete"
        ]
        assert statuses == [status], result.stdout + result.stderr
    assert (root / "data" / "t" / "use.txt").read_text() == "Bhello"


@pytest.mark.parametrize(
    "value",
    [
        pd.Timestamp("2020-01-01", tz="Europe/Paris"),
        datetime.datetime(2020, 1, 1, tzinfo=pytz.timezone("Europe/Paris")),
        pd.NaT,
        pd.Timestamp("2020-01-01", tz=pytz.FixedOffset(60)),
        datetime.datetime(2020, 1, 1, tzinfo=pytz.FixedOffset(60)),
    ],
    ids=[
        "pandas-pytz",
        "datetime-pytz",
        "pandas-nat",
        "pandas-fixed-offset",
        "datetime-fixed-offset",
    ],
)
def test_complete_library_timezone_reprs_are_accepted(value: Any) -> None:
    form = fingerprint._form(value)
    assert not form.reports, form
