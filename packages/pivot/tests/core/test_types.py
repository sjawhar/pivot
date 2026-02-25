"""Tests for pivot.types — identity serialization helpers."""

from __future__ import annotations

import pytest

from pivot.types import (
    ArtifactIdentity,
    ArtifactIdentityJson,
    identity_from_json,
    identity_key,
    identity_to_json,
)

# =============================================================================
# identity_to_json / identity_from_json roundtrip
# =============================================================================


@pytest.mark.parametrize(
    ("identity", "expected_json"),
    [
        pytest.param(
            ArtifactIdentity("train", "metrics"),
            {"producer": "train", "key": "metrics"},
            id="with-key",
        ),
        pytest.param(
            ArtifactIdentity("input", None),
            {"producer": "input", "key": None},
            id="without-key",
        ),
    ],
)
def test_artifact_identity_json_roundtrip(
    identity: ArtifactIdentity,
    expected_json: ArtifactIdentityJson,
) -> None:
    payload = identity_to_json(identity)
    assert payload == expected_json
    assert identity_from_json(payload) == identity


# =============================================================================
# identity_key
# =============================================================================


@pytest.mark.parametrize(
    ("identity", "expected_key"),
    [
        pytest.param(ArtifactIdentity("train", None), "train", id="no-key"),
        pytest.param(ArtifactIdentity("train", "metrics"), "train:metrics", id="with-key"),
    ],
)
def test_identity_key(identity: ArtifactIdentity, expected_key: str) -> None:
    assert identity_key(identity) == expected_key
