from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

Timepoint = Literal[1, 2, 3, 4, 5]
Aggregation = Literal["baseline_only", "latest", "linear_slope", "variance", "spline", "all"]

@dataclass(frozen=True)
class DynamicConfig:
    """
    Configuration for Paper 2 dynamic prediction models.

    Assumptions:
    - Single-wide modelling table with:
        - baseline predictors (dummy-coded / numeric-coded)
        - PHQ-9, GAD-7, WSAS recorded at sessions 1–5 as wide columns
    - Filtered cohort upstream to match methods, e.g,:
        - adults
        - high-intensity interventions
        - 5–21 sessions
        - required baseline measures present, etc.
    """

    # Baseline predictors used in every model (timepoint 1)
    baseline_predictors: Sequence[str]

    # Wide columns for session-by-session scores (sessions 1–5)
    phq9_session_cols: Dict[Timepoint, str]
    gad7_session_cols: Dict[Timepoint, str]
    wsas_session_cols: Dict[Timepoint, str]

    # Continuous predictors to scale (subset of baseline + engineered)
    scale_columns: Sequence[str]

    # Outcomes (six; depression and anxiety)
    outcomes: Sequence[str]

    # Timepoints for modelling (baseline and sessions 2–5)
    timepoints: Sequence[Timepoint] = (1, 2, 3, 4, 5)

    # Which aggregation sets to run at each timepoint
    aggregations: Sequence[Aggregation] = ("baseline_only", "latest", "linear_slope", "variance", "spline", "all")
