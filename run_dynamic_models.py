import os
import joblib
import pandas as pd

from nhstt_dynamic.config import DynamicConfig
from nhstt_dynamic.preprocessing import preprocess_dataframe, split_xy, assert_numeric_matrix
from nhstt_dynamic.features_dynamic import PrecomputedDynamicNaming, get_feature_set_for_model
from nhstt_dynamic.pipeline import build_elasticnet_logistic_pipeline, default_param_grid
from nhstt_dynamic.evaluation import evaluate_dynamic_model, BootstrapConfig


def infer_scale_columns(X: pd.DataFrame) -> list[str]:
    """
    Scale numeric columns except binary {0,1} dummy indicators.
    """
    scale_cols: list[str] = []
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            continue

        vals = pd.unique(X[c].dropna())
        if len(vals) == 0:
            continue

        # Treat as dummy if values are subset of {0, 1}
        try:
            if set(map(float, vals)).issubset({0.0, 1.0}):
                continue
        except Exception:
            # If casting fails for any reason, err on the safe side and do not scale
            continue

        scale_cols.append(c)

    return scale_cols


def main():
    # Analysis-ready table matching Paper 2 sample:
    # adults, high-intensity, 5–21 sessions, etc.
    df = pd.read_parquet("data/analysis_table_dynamic.parquet")

    # --------------------------
    # Fill these with your real names
    # --------------------------
    cfg = DynamicConfig(
        baseline_predictors=[...],  # baseline predictors (numeric/dummy-coded)
        # session cols not needed here because dynamic predictors are precomputed
        phq9_session_cols={},
        gad7_session_cols={},
        wsas_session_cols={},
        # scale_columns not used in this script (we infer scaling per model)
        scale_columns=[],
        outcomes=[
            "outcome_reliable_improvement_PHQ9",
            "outcome_recovery_PHQ9",
            "outcome_reliable_recovery_PHQ9",
            "outcome_reliable_improvement_GAD7",
            "outcome_recovery_GAD7",
            "outcome_reliable_recovery_GAD7",
        ],
    )

    # Naming patterns for your precomputed dynamic predictors
    naming = PrecomputedDynamicNaming(
        phq_prefix="PHQ9",
        gad_prefix="GAD7",
        wsas_prefix="WSAS",
        latest_pattern="{prefix}_sesh_{t}",
        slope_pattern="{prefix}_linearslope_1_{t}",
        variance_pattern="{prefix}_variance_1_{t}", 
        spline_pattern="{prefix}_smoothing_spline_coeff_1_{t}_{k}",
        spline_num_coeffs=5,
        spline_starts_at_timepoint=4,
    )

    param_grid = default_param_grid()
    boot_cfg = BootstrapConfig(n_bootstrap=200, cv_folds=5, random_seed=0, n_jobs=-1)

    all_metrics = []

    # Five timepoints (baseline and sessions 2–5)
    timepoints = [1, 2, 3, 4, 5]
    aggregations = ["baseline_only", "latest", "linear_slope", "variance", "spline", "all"]

    for outcome in cfg.outcomes:
        for timepoint in timepoints:
            for aggregation in aggregations:
                # Enforce method constraints (based on availability)
                if aggregation in ("latest", "linear_slope", "all") and timepoint < 2:
                    continue
                if aggregation == "variance" and timepoint < 3:
                    continue
                if aggregation == "spline" and timepoint < 4:
                    continue

                feature_cols = get_feature_set_for_model(
                    df=df,
                    baseline_predictors=cfg.baseline_predictors,
                    naming=naming,
                    timepoint=timepoint,      # type: ignore
                    aggregation=aggregation,  # type: ignore
                )

                model_name = f"{outcome}__t{timepoint}__{aggregation}"
                out_dir = os.path.join("outputs", outcome, f"t{timepoint}", aggregation)
                os.makedirs(out_dir, exist_ok=True)

                df_clean, report = preprocess_dataframe(df, feature_cols, target_column=outcome)
                X, y = split_xy(df_clean, feature_cols, outcome)

                assert_numeric_matrix(X)

                # Infer which columns to scale for THIS model
                scale_cols_this_model = infer_scale_columns(X)

                pipeline = build_elasticnet_logistic_pipeline(
                    scale_columns=scale_cols_this_model,
                    imputer_k=5,
                    imputer_strategy="weighted",
                )

                results = evaluate_dynamic_model(
                    pipeline=pipeline,
                    param_grid=param_grid,
                    X=X,
                    y=y,
                    model_name=model_name,
                    config=boot_cfg,
                    save_dir=out_dir,
                    plot_auc_optimism=False,
                )

                # Save model + tables
                joblib.dump(results.final_model, os.path.join(out_dir, f"{model_name}_final_model.joblib"))
                results.bootstrap_results.to_csv(os.path.join(out_dir, f"{model_name}_bootstrap_results.csv"), index=False)
                results.metrics_summary.to_csv(os.path.join(out_dir, f"{model_name}_metrics_summary.csv"), index=False)

                all_metrics.append(results.metrics_summary)

                print(f"Completed: {model_name}")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv("outputs/all_models_metrics_summary.csv", index=False)
    print("All done. Summary saved to outputs/all_models_metrics_summary.csv")


if __name__ == "__main__":
    main()
