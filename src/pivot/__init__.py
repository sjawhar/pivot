"""Pivot: High-Performance Python Pipeline Tool.

Public API for defining and running pipelines.

Example:
    >>> from pivot import stage, Out, Metric, Plot
    >>>
    >>> @stage(deps=['data.csv'], outs=['processed.parquet'])
    >>> def preprocess(input_file: str = 'data.csv'):
    ...     import pandas as pd
    ...     df = pd.read_csv(input_file)
    ...     df = df.dropna()
    ...     df.to_parquet('processed.parquet')
    >>>
    >>> @stage(deps=['processed.parquet'], outs=[Out('model.pkl'), Metric('metrics.json')])
    >>> def train(data_file: str = 'processed.parquet', lr: float = 0.01):
    ...     import pandas as pd
    ...     df = pd.read_parquet(data_file)
    ...     # Train model...
"""

from pivot import cache as cache
from pivot import dvc_compat as dvc_compat
from pivot import outputs as outputs
from pivot import parameters as parameters
from pivot import pvt as pvt
from pivot import state as state
from pivot.outputs import BaseOut as BaseOut
from pivot.outputs import IncrementalOut as IncrementalOut
from pivot.outputs import Metric as Metric
from pivot.outputs import Out as Out
from pivot.outputs import Plot as Plot
from pivot.registry import REGISTRY as REGISTRY
from pivot.registry import Variant as Variant
from pivot.registry import stage as stage

__version__ = "0.1.0-dev"
