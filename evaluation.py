from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import clone
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from .plotting import plot_calibration_curve, plot_auc_optimism_distributions


def calibration_intercept_slope_logistic(
    y: np.ndarray,
    p_hat: np.ndarray,
    eps: float = 1e-6
) -> tuple[float, float]:
    """
    Logit-based calibration (logistic recalibration):
      logit(P(Y=1)) = a + b * logit(p_hat)
    Returns: intercept a, slope b
    """
    y = np.asarray(y, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)
    p_hat = np.clip(p_hat, eps, 1 - eps)

    eta = logit(p_hat)
    X = sm.add_constant(eta)
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return float(fit.params[0]), float(fit.params[1])


@dataclass(frozen=True)
class BootstrapConfig:
    n_bootstrap: int = 200
    cv_folds: int = 5
    random_seed: int = 0
    n_jobs: int = -1


@dataclass(frozen=True)
class EvalOutputs:
    final_model: object
    best_params_original: dict

    apparent_auc: float
    auc_corrected: float
    auc_ci_bias: Tuple[float, float]

    cal_intercept_app: float
    cal_slope_app: float
    cal_intercept_corrected: float
    cal_slope_corrected: float

    brier_app: float
    brier_median: float
    brier_ci: Tuple[float, float]

    sensitivity_median: float
    sensitivity_ci: Tuple[float, float]
    specificity_median: float
    specificity_ci: Tuple[float, float]

    bootstrap_results: pd.DataFrame
    metrics_summary: pd.DataFrame


def evaluate_dynamic_model(
    pipeline,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    config: BootstrapConfig = BootstrapConfig(),
    eps: float = 1e-6,
    save_dir: Optional[str] = None,
    plot_auc_optimism: bool = False,
) -> EvalOutputs:
    """
    Paper 2 evaluation (aligned to Paper 1 correction style):
    - Nested bootstrap resampling (B=200) with tuning inside each bootstrap
    - Discrimination:
        * Harrell optimism correction for AUC (point estimate)
        * Bias-based interval derived from optimism percentiles
    - Calibration:
        * Logit-based calibration intercept and slope
        * Strict Harrell optimism correction using per-bootstrap apparent-on-bootstrap minus test-on-original
        * LOWESS calibration curve + optimism-corrected LOWESS curve
    - Outputs:
        * Calibration curve figure
        * bootstrap results table
        * metrics summary table
    """
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    B = int(config.n_bootstrap)
    prange = np.arange(0.01, 1.0, 0.01)
    rng = np.random.RandomState(config.random_seed)

    # -----------------------------
    # STEP 1: Apparent model on original dataset (tuned)
    # -----------------------------
    gs_orig = GridSearchCV(
        pipeline,
        param_grid,
        cv=config.cv_folds,
        scoring="roc_auc",
        n_jobs=config.n_jobs,
        verbose=0,
    )
    gs_orig.fit(X, y_arr)
    final_model = gs_orig.best_estimator_
    best_params_original = gs_orig.best_params_

    p_orig = final_model.predict_proba(X)[:, 1]

    auc_apparent = float(roc_auc_score(y_arr, p_orig))
    brier_app = float(brier_score_loss(y_arr, p_orig))

    cal_int_app, cal_slope_app = calibration_intercept_slope_logistic(y_arr, p_orig, eps=eps)
    cal_apparent_lowess = lowess(y_arr, p_orig, it=0, xvals=prange)

    # -----------------------------
    # STEP 2: Bootstrap loop
    # -----------------------------
    rows = []
    cal_optimism_all = []  # for LOWESS curve correction

    for b in range(B):
        Xb, yb = resample(X, y_arr, replace=True, random_state=int(rng.randint(0, 2**31 - 1)))

        gs_b = GridSearchCV(
            clone(pipeline),
            param_grid,
            cv=config.cv_folds,
            scoring="roc_auc",
            n_jobs=config.n_jobs,
            verbose=0,
        )
        gs_b.fit(Xb, yb)
        model_b = gs_b.best_estimator_

        # predictions on bootstrap (apparent) and original (test)
        p_b_on_b = model_b.predict_proba(Xb)[:, 1]
        p_b_on_orig = model_b.predict_proba(X)[:, 1]

        # AUC optimism
        auc_app_b = float(roc_auc_score(yb, p_b_on_b))
        auc_test_b = float(roc_auc_score(y_arr, p_b_on_orig))
        optimism_auc = auc_app_b - auc_test_b

        # Sens/spec on bootstrap sample (consistent with your earlier workflow)
        yhat_b = model_b.predict(Xb)
        tn, fp, fn, tp = confusion_matrix(yb, yhat_b).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        brier_b = float(brier_score_loss(yb, p_b_on_b))

        # --- STRICT calibration optimism (apparent on bootstrap vs test on original) ---
        try:
            cal_int_app_b, cal_slope_app_b = calibration_intercept_slope_logistic(yb, p_b_on_b, eps=eps)
        except Exception:
            cal_int_app_b, cal_slope_app_b = np.nan, np.nan

        try:
            cal_int_test_b, cal_slope_test_b = calibration_intercept_slope_logistic(y_arr, p_b_on_orig, eps=eps)
        except Exception:
            cal_int_test_b, cal_slope_test_b = np.nan, np.nan

        cal_int_optim_b = cal_int_app_b - cal_int_test_b if (np.isfinite(cal_int_app_b) and np.isfinite(cal_int_test_b)) else np.nan
        cal_slope_optim_b = cal_slope_app_b - cal_slope_test_b if (np.isfinite(cal_slope_app_b) and np.isfinite(cal_slope_test_b)) else np.nan

        # LOWESS optimism for curve correction
        if len(np.unique(yb)) > 1 and len(np.unique(p_b_on_b)) > 1 and len(np.unique(p_b_on_orig)) > 1:
            cal_app_lowess_b = lowess(yb, p_b_on_b, it=0, xvals=prange)
            cal_test_lowess_b = lowess(y_arr, p_b_on_orig, it=0, xvals=prange)
            cal_optimism_all.append(cal_app_lowess_b - cal_test_lowess_b)

        rows.append(
            {
                "iteration": b + 1,
                "auc_apparent": auc_app_b,
                "auc_test": auc_test_b,
                "optimism_auc": optimism_auc,
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "brier_score": float(brier_b),

                "cal_intercept_app": float(cal_int_app_b),
                "cal_slope_app": float(cal_slope_app_b),
                "cal_intercept_test": float(cal_int_test_b),
                "cal_slope_test": float(cal_slope_test_b),
                "cal_intercept_optimism": float(cal_int_optim_b),
                "cal_slope_optimism": float(cal_slope_optim_b),
            }
        )

    bootstrap_df = pd.DataFrame(rows)

    # -----------------------------
    # STEP 3: AUC optimism correction + bias-based CI (from optimism percentiles)
    # -----------------------------
    optim_auc = bootstrap_df["optimism_auc"].dropna().values
    mean_optim_auc = float(np.mean(optim_auc))
    auc_corrected = auc_apparent - mean_optim_auc

    o_lo, o_hi = np.percentile(optim_auc, [2.5, 97.5])
    auc_ci_bias = (auc_apparent - o_hi, auc_apparent - o_lo)

    # -----------------------------
    # STEP 4: Calibration optimism correction (STRICT Harrell-style)
    # -----------------------------
    optim_int = bootstrap_df["cal_intercept_optimism"].dropna().values
    optim_slope = bootstrap_df["cal_slope_optimism"].dropna().values

    mean_optim_int = float(np.mean(optim_int)) if optim_int.size else 0.0
    mean_optim_slope = float(np.mean(optim_slope)) if optim_slope.size else 0.0

    cal_intercept_corrected = cal_int_app - mean_optim_int
    cal_slope_corrected = cal_slope_app - mean_optim_slope

    # -----------------------------
    # STEP 5: LOWESS optimism-corrected calibration curve
    # -----------------------------
    if len(cal_optimism_all):
        mean_cal_optimism = np.mean(np.array(cal_optimism_all), axis=0)
        cal_corrected_lowess = cal_apparent_lowess - mean_cal_optimism
    else:
        cal_corrected_lowess = cal_apparent_lowess

    # -----------------------------
    # STEP 6: Bootstrap summaries for sensitivity/specificity/brier
    # -----------------------------
    def _median_ci(vals: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        vals = vals[~np.isnan(vals)]
        return float(np.median(vals)), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    sens_med, sens_ci = _median_ci(bootstrap_df["sensitivity"].values)
    spec_med, spec_ci = _median_ci(bootstrap_df["specificity"].values)
    brier_med, brier_ci = _median_ci(bootstrap_df["brier_score"].values)

    # -----------------------------
    # STEP 7: Plot calibration curve
    # -----------------------------
    plot_calibration_curve(
        y_true=y_arr,
        p_orig=p_orig,
        prange=prange,
        cal_apparent_lowess=cal_apparent_lowess,
        cal_corrected_lowess=cal_corrected_lowess,
        title=f"Calibration curve: {model_name}",
        save_dir=save_dir,
        filename=f"{model_name}_calibration_curve.png",
    )

    if plot_auc_optimism:
        plot_auc_optimism_distributions(
            optim_auc,
            save_dir=save_dir,
            filename=f"{model_name}_auc_optimism.png",
        )

    # -----------------------------
    # STEP 8: Metrics summary table
    # -----------------------------
    metrics_summary = pd.DataFrame(
        [
            {"Model": model_name, "Metric": "AUC (apparent)", "Value": auc_apparent},
            {"Model": model_name, "Metric": "AUC (optimism-corrected)", "Value": auc_corrected},
            {"Model": model_name, "Metric": "AUC (bias-based interval lower)", "Value": auc_ci_bias[0]},
            {"Model": model_name, "Metric": "AUC (bias-based interval upper)", "Value": auc_ci_bias[1]},

            {"Model": model_name, "Metric": "Sensitivity (median; bootstrap)", "Value": sens_med},
            {"Model": model_name, "Metric": "Sensitivity (2.5th percentile)", "Value": sens_ci[0]},
            {"Model": model_name, "Metric": "Sensitivity (97.5th percentile)", "Value": sens_ci[1]},

            {"Model": model_name, "Metric": "Specificity (median; bootstrap)", "Value": spec_med},
            {"Model": model_name, "Metric": "Specificity (2.5th percentile)", "Value": spec_ci[0]},
            {"Model": model_name, "Metric": "Specificity (97.5th percentile)", "Value": spec_ci[1]},

            {"Model": model_name, "Metric": "Brier score (apparent)", "Value": brier_app},
            {"Model": model_name, "Metric": "Brier score (median; bootstrap)", "Value": brier_med},
            {"Model": model_name, "Metric": "Brier score (2.5th percentile)", "Value": brier_ci[0]},
            {"Model": model_name, "Metric": "Brier score (97.5th percentile)", "Value": brier_ci[1]},

            {"Model": model_name, "Metric": "Calibration intercept (logit-based; apparent)", "Value": cal_int_app},
            {"Model": model_name, "Metric": "Calibration slope (logit-based; apparent)", "Value": cal_slope_app},

            {"Model": model_name, "Metric": "Calibration intercept (logit-based; optimism-corrected)", "Value": cal_intercept_corrected},
            {"Model": model_name, "Metric": "Calibration slope (logit-based; optimism-corrected)", "Value": cal_slope_corrected},
        ]
    )

    return EvalOutputs(
        final_model=final_model,
        best_params_original=best_params_original,

        apparent_auc=auc_apparent,
        auc_corrected=auc_corrected,
        auc_ci_bias=auc_ci_bias,

        cal_intercept_app=cal_int_app,
        cal_slope_app=cal_slope_app,
        cal_intercept_corrected=cal_intercept_corrected,
        cal_slope_corrected=cal_slope_corrected,

        brier_app=brier_app,
        brier_median=brier_med,
        brier_ci=brier_ci,

        sensitivity_median=sens_med,
        sensitivity_ci=sens_ci,
        specificity_median=spec_med,
        specificity_ci=spec_ci,

        bootstrap_results=bootstrap_df,
        metrics_summary=metrics_summary,
    )
