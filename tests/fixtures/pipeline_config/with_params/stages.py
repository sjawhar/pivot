"""Stage functions with Pydantic params for testing pivot.yaml loading."""

from __future__ import annotations

import json
import pathlib
from typing import Annotated, TypedDict

from pivot import loaders, outputs, stage_def


class TrainParams(stage_def.StageParams):
    """Parameters for training."""

    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32


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
    train: Annotated[pathlib.Path, outputs.Metric("metrics/train.json")]


def train(
    params: TrainParams,
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    """Train model with params."""
    content = clean.read_text()

    result = {
        "model": f"trained_with_lr={params.learning_rate}",
        "epochs": params.epochs,
        "batch_size": params.batch_size,
        "data_size": len(content),
    }

    model_path = pathlib.Path("models/model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(result))

    metrics_path = pathlib.Path("metrics/train.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"loss": 0.1, "accuracy": 0.95}, f)

    return {"model": model_path, "train": metrics_path}
