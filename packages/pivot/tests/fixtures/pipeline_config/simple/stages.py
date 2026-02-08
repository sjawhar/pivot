"""Simple stage functions for testing pivot.yaml loading."""

from __future__ import annotations

import pathlib
from typing import Annotated, TypedDict

from pivot import loaders, outputs


class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("data/clean.csv", loaders.PathOnly())]


def preprocess(
    raw: Annotated[pathlib.Path, outputs.Dep("data/raw.csv", loaders.PathOnly())],
) -> PreprocessOutputs:
    """Read raw data and write clean data."""
    content = raw.read_text()
    clean_path = pathlib.Path("data/clean.csv")
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text(content.upper())
    return {"clean": clean_path}


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]


def train(
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    """Read clean data and write model."""
    content = clean.read_text()
    model_path = pathlib.Path("models/model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(f"MODEL:{len(content)}")
    return {"model": model_path}
