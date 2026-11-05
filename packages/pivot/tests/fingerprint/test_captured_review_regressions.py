"""Regression witnesses for the captured-instance review (PR #551)."""

import datetime
import functools
import json
import pathlib
import subprocess
import sys
import types
from typing import Any

import dateutil.tz
import pandas as pd
import pytest
from tests import helpers

from pivot import exceptions, fingerprint


def test_import_does_not_construct_a_probe_namedtuple() -> None:
    """A functional NamedTuple factory need not expose the probe's field on its namespace."""
    script = """
import functools
import typing
original = typing.NamedTuple
@functools.wraps(original)
def factory(name, fields, **kwargs):
    if name == '_Fields':
        return type(name, (), {})
    return original(name, fields, **kwargs)
typing.NamedTuple = factory
import pivot.fingerprint
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("value", "changed"),
    [
        (
            pd.Timestamp("2020-01-01 00:00:00.000000001"),
            pd.Timestamp("2020-01-01 00:00:00.000000002"),
        ),
        (pd.Timedelta(1, unit="ns"), pd.Timedelta(2, unit="ns")),
        (
            datetime.datetime(2020, 1, 1, tzinfo=dateutil.tz.tzutc()),
            datetime.datetime(2020, 1, 2, tzinfo=dateutil.tz.tzutc()),
        ),
        (
            datetime.datetime(2020, 1, 1, tzinfo=dateutil.tz.tzoffset("east", 3600)),
            datetime.datetime(2020, 1, 1, tzinfo=dateutil.tz.tzoffset("east", 7200)),
        ),
    ],
    ids=["timestamp", "timedelta", "dateutil-utc", "dateutil-offset"],
)
def test_complete_library_value_reprs_remain_accepted(value: Any, changed: Any) -> None:
    """These library reprs retain full values; rejecting them protects no stale cache hit."""
    for item in (value, changed):
        form = fingerprint._form(item)
        assert not form.reports, form
        assert json.loads(form.text) == [type(item).__name__, repr(item)]
    assert fingerprint._form(value).text != fingerprint._form(changed).text


@pytest.mark.parametrize("holder", ["partial", "field"])
@pytest.mark.parametrize(
    "expression",
    [
        'pd.Timestamp("2020-01-01 00:00:00.000000001")',
        'pd.Timedelta(1, unit="ns")',
        "datetime.datetime(2020, 1, 1, tzinfo=dateutil.tz.tzutc())",
    ],
)
def test_complete_library_values_run_and_cache(
    tmp_path: pathlib.Path, holder: str, expression: str
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    home = tmp_path / "home"
    initialized = helpers.run_pivot(root, home, "init")
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "text.txt").write_text("hello")
    declaration = (
        f"HELD = functools.partial(take, {expression})\n"
        if holder == "partial"
        else f"@dataclasses.dataclass\nclass Config:\n    value: object = {expression}\n"
    )
    call = "HELD(text)" if holder == "partial" else "str(Config().value) + text"
    (root / "pipeline.py").write_text(
        "import functools\nimport dataclasses\nimport datetime\nimport dateutil.tz\n"
        + "import pandas as pd\nfrom pivot import Pipeline, stage\n"
        + "def take(value, text):\n    return str(value) + text\n"
        + declaration
        + f"@stage\ndef use(text: str) -> str:\n    return {call}\n"
        + 'with Pipeline("t") as pipeline:\n    use(pipeline.input("text.txt", t=str))\n'
    )
    for expected in ("ran", "cached"):
        result = helpers.run_pivot(root, home, "repro", "--json")
        assert result.returncode == 0, result.stdout + result.stderr
        statuses = [
            json.loads(line)["status"]
            for line in result.stdout.splitlines()
            if line.startswith("{") and json.loads(line).get("type") == "stage_complete"
        ]
        assert statuses == [expected], result.stdout + result.stderr


class _Dynamic:
    value: types.DynamicClassAttribute = types.DynamicClassAttribute(lambda self: 1)


_DYNAMIC = functools.partial(str, _Dynamic())


def _uses_dynamic() -> str:
    return _DYNAMIC()


def test_a_dynamic_class_attribute_is_refused_by_member_name() -> None:
    """A library descriptor cannot silently hide executable user code in a captured class."""
    with pytest.raises(exceptions.StageDefinitionError, match=r"_Dynamic\.value") as error:
        fingerprint.get_stage_fingerprint(_uses_dynamic)
    assert "StageParams" in str(error.value)
