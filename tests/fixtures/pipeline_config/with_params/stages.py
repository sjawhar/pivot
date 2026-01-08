"""Stage functions with Pydantic params for testing pivot.yaml loading."""

from __future__ import annotations

import json

import pydantic


class TrainParams(pydantic.BaseModel):
    """Parameters for training."""

    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32


def preprocess() -> None:
    """Read raw data and write clean data."""
    with open("data/raw.csv") as f:
        content = f.read()
    with open("data/clean.csv", "w") as f:
        f.write(content.upper())


def train(params: TrainParams) -> None:
    """Train model with params."""
    with open("data/clean.csv") as f:
        content = f.read()

    result = {
        "model": f"trained_with_lr={params.learning_rate}",
        "epochs": params.epochs,
        "batch_size": params.batch_size,
        "data_size": len(content),
    }

    with open("models/model.pkl", "w") as f:
        f.write(json.dumps(result))

    with open("metrics/train.json", "w") as f:
        json.dump({"loss": 0.1, "accuracy": 0.95}, f)
