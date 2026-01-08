"""Stage functions with matrix support for testing pivot.yaml loading."""

from __future__ import annotations

import glob
import json
import os

import pydantic


class TrainParams(pydantic.BaseModel):
    """Parameters for training."""

    learning_rate: float = 0.01
    model_type: str = "default"
    hidden_size: int = 512


def preprocess() -> None:
    """Read raw data and write clean data."""
    with open("data/raw.csv") as f:
        content = f.read()
    with open("data/clean.csv", "w") as f:
        f.write(content.upper())


def train(params: TrainParams) -> None:
    """Train model with params - works for any variant."""
    data_files = glob.glob("data/*.csv")
    total_size = 0
    for df in data_files:
        with open(df) as f:
            total_size += len(f.read())

    result = {
        "model_type": params.model_type,
        "learning_rate": params.learning_rate,
        "hidden_size": params.hidden_size,
        "data_size": total_size,
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    # Find the output file from environment or use default
    model_out = os.environ.get("PIVOT_OUT_MODEL", "models/model.pkl")
    metrics_out = os.environ.get("PIVOT_OUT_METRICS", "metrics/train.json")

    with open(model_out, "w") as f:
        f.write(json.dumps(result))

    with open(metrics_out, "w") as f:
        json.dump({"loss": 0.1 / params.learning_rate, "accuracy": 0.95}, f)
