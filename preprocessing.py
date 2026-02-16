from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessReport:
    n_rows_in: int
    n_rows_out: int
    n_predictors: int
    n_missing_predictors_before: int
    n_missing_predictors_after: int
    n_missing_target_after: Optional[int]


def preprocess_dataframe(
    df: pd.DataFrame,
    predictor_columns: Iterable[str],
    target_column: Optional[str] = None,
    na_tokens: Tuple[str, ...] = ("<NA>", "NA", "NaN", "nan", ""),
) -> tuple[pd.DataFrame, PreprocessReport]:
    """
    Minimal, safe preprocessing.
    - Standardises common missing tokens to np.nan in predictors
    - Does not coerce types: dynamic pipeline expects numeric/dummy-coded predictors.
    """
    df_out = df.copy()
    predictor_columns = list(predictor_columns)

    missing_before = int(df_out[predictor_columns].isna().sum().sum())
    df_out[predictor_columns] = df_out[predictor_columns].replace(list(na_tokens), np.nan)
    missing_after = int(df_out[predictor_columns].isna().sum().sum())

    missing_target_after = None
    if target_column is not None and target_column in df_out.columns:
        df_out[target_column] = df_out[target_column].replace(list(na_tokens), np.nan)
        missing_target_after = int(df_out[target_column].isna().sum())

    report = PreprocessReport(
        n_rows_in=int(df.shape[0]),
        n_rows_out=int(df_out.shape[0]),
        n_predictors=int(len(predictor_columns)),
        n_missing_predictors_before=missing_before,
        n_missing_predictors_after=missing_after,
        n_missing_target_after=missing_target_after,
    )
    return df_out, report


def split_xy(
    df: pd.DataFrame,
    predictor_columns: Iterable[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    predictor_columns = list(predictor_columns)
    X = df[predictor_columns].copy()
    y = df[target_column].astype(float).copy()
    return X, y


def assert_numeric_matrix(X: pd.DataFrame) -> None:
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise TypeError(
            "Non-numeric predictors detected. KNN imputation requires numeric columns.\n"
            f"Non-numeric columns (first 20): {non_numeric[:20]}\n"
            "Ensure predictors are dummy-coded / numeric-coded before modelling."
        )
