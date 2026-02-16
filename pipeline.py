 from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fknni import FastKNNImputer


class PartialScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale_columns: Sequence[str]):
        self.scale_columns = list(scale_columns)
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        self._scale_columns_ = [c for c in self.scale_columns if c in X.columns]
        if self._scale_columns_:
            self.scaler.fit(X[self._scale_columns_])
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if getattr(self, "_scale_columns_", []):
            X[self._scale_columns_] = self.scaler.transform(X[self._scale_columns_])
        return X


class FastKNNImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors: int = 5, strategy: str = "weighted", min_data_ratio: float = 0.0):
        self.n_neighbors = n_neighbors
        self.strategy = strategy
        self.min_data_ratio = min_data_ratio

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FastKNNImputerWrapper expects a pandas DataFrame input.")
        self.feature_names_in_ = list(X.columns)
        self.imputer_ = FastKNNImputer(
            n_neighbors=self.n_neighbors,
            strategy=self.strategy,
            min_data_ratio=self.min_data_ratio,
        )
        if hasattr(self.imputer_, "fit"):
            self.imputer_.fit(X.values)
        return self

    def transform(self, X: pd.DataFrame):
        if hasattr(self.imputer_, "transform"):
            X_imp = self.imputer_.transform(X.values)
        else:
            X_imp = self.imputer_.fit_transform(X.values)
        return pd.DataFrame(X_imp, columns=self.feature_names_in_, index=X.index)


@dataclass(frozen=True)
class ElasticNetLogisticConfig:
    max_iter: int = 10_000
    solver: str = "saga"


def build_elasticnet_logistic_pipeline(
    scale_columns: Sequence[str],
    imputer_k: int = 5,
    imputer_strategy: str = "weighted",
    model_config: ElasticNetLogisticConfig = ElasticNetLogisticConfig(),
) -> Pipeline:
    model = LogisticRegression(
        penalty="elasticnet",
        solver=model_config.solver,
        max_iter=model_config.max_iter,
    )

    return Pipeline(
        steps=[
            ("scaling", PartialScaler(scale_columns=scale_columns)),
            ("imputation", FastKNNImputerWrapper(n_neighbors=imputer_k, strategy=imputer_strategy)),
            ("model", model),
        ]
    )


def default_param_grid() -> dict:
    return {
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "model__C": [0.01, 0.1, 1.0, 10.0],
    }
