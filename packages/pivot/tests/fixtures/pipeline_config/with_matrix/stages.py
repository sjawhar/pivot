"""Stage functions with matrix support for testing pivot.yaml loading."""

from __future__ import annotations

import json
import pathlib
from typing import Annotated, TypedDict

from pivot import loaders, outputs, stage_def


class TrainParams(stage_def.StageParams):
    """Parameters for training."""

    learning_rate: float = 0.01
    model_type: str = "default"
    hidden_size: int = 512


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
    metrics: Annotated[pathlib.Path, outputs.Metric("metrics/train.json")]


def train(
    params: TrainParams,
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
    config: Annotated[pathlib.Path, outputs.Dep("configs/default.yaml", loaders.PathOnly())],
) -> TrainOutputs:
    """Train model with params - works for any variant."""
    clean_content = clean.read_text()
    config_content = config.read_text() if config.exists() else "{}"

    result = {
        "model_type": params.model_type,
        "learning_rate": params.learning_rate,
        "hidden_size": params.hidden_size,
        "data_size": len(clean_content),
        "config": config_content[:50],
    }

    model_path = pathlib.Path("models/model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(result))

    metrics_path = pathlib.Path("metrics/train.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"loss": 0.1 / params.learning_rate, "accuracy": 0.95}, f)

    return {"model": model_path, "metrics": metrics_path}
