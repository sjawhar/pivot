import argparse
import collections
import functools
import json
import types
import typing
import urllib.parse

import numpy as np
import pytest

from pivot import fingerprint


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (collections.deque([1, 2], maxlen=3), '["deque", "deque([1, 2], maxlen=3)"]'),
        (
            urllib.parse.ParseResult("https", "example.com", "/", "", "", ""),
            '["ParseResult", ["https", "example.com", "/", "", "", ""]]',
        ),
        (np.int64(7), '["int64", "np.int64(7)"]'),
        (np.int32(7), '["int32", "np.int32(7)"]'),
        (np.uint64(7), '["uint64", "np.uint64(7)"]'),
        (np.bool_(True), '["bool", "np.True_"]'),
        (types.MappingProxyType({"a": 1}), '["mappingproxy", "mappingproxy({\'a\': 1})"]'),
        (
            collections.ChainMap({"a": 1}, {"b": 2}),
            "[\"ChainMap\", \"ChainMap({'a': 1}, {'b': 2})\"]",
        ),
        (types.SimpleNamespace(a=1), '["SimpleNamespace", "namespace(a=1)"]'),
        (argparse.Namespace(a=1), '["Namespace", "Namespace(a=1)"]'),
        (list[int], '["GenericAlias", "list[int]"]'),
        (
            vars(typing)["Optional"][str],  # Runtime legacy alias, not a type annotation.
            '["_UnionGenericAlias", "typing.Optional[str]"]',
        ),
        (str | None, '["UnionType", "str | None"]'),
    ],
    ids=[
        "scalar-deque",
        "library-namedtuple",
        "numpy-int64",
        "numpy-int32",
        "numpy-uint64",
        "numpy-bool",
        "scalar-proxy",
        "scalar-chainmap",
        "scalar-simplenamespace",
        "scalar-namespace",
        "builtin-alias",
        "typing-union",
        "builtin-union",
    ],
)
def test_sound_library_encodings_keep_mains_bytes(value: object, expected: str) -> None:
    """These full main encodings omit neither mutable state nor an executable user class."""
    assert fingerprint._form(value).text == expected
    args, keywords = fingerprint._partial_forms(functools.partial(str, value))
    assert json.loads(args.text) == ["tuple", [json.loads(expected)]]
    assert json.loads(keywords.text) == ["dict", []]
    _, keyword = fingerprint._partial_forms(functools.partial(dict, value=value))
    assert json.loads(keyword.text) == ["dict", [["value", json.loads(expected)]]]


def test_library_mapping_forms_preserve_the_holes_the_new_encoding_closes() -> None:
    """Main lost Counter's type, defaultdict's factory, and OrderedDict's order."""
    assert fingerprint._form(collections.Counter(a=1)).text != fingerprint._form({"a": 1}).text
    assert (
        fingerprint._form(collections.defaultdict(int)).text
        != fingerprint._form(collections.defaultdict(list)).text
    )
    left = collections.OrderedDict(a=1, b=2)
    right = collections.OrderedDict(b=2, a=1)
    assert fingerprint._form(left).text != fingerprint._form(right).text


def test_temporal_scalar_encodings_retain_dtype_units() -> None:
    """datetime64's NaT repr omits units; timedelta64's complete repr must retain them."""
    assert (
        fingerprint._form(np.timedelta64("NaT", "D")).text
        != fingerprint._form(np.timedelta64("NaT", "s")).text
    )
    assert (
        fingerprint._form(np.datetime64("NaT", "D")).text
        != fingerprint._form(np.datetime64("NaT", "s")).text
    )
