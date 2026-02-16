from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

COL_OBS = "grey"
COL_APP = "red"
COL_COR = "C0"
COL_REF = "black"


def _ensure_dir(save_dir: Optional[str]) -> Optional[Path]:
    if save_dir is None:
        return None
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _savefig(save_dir: Optional[str], filename: str) -> None:
    p = _ensure_dir(save_dir)
    if p is None:
        return
    plt.savefig(p / filename, bbox_inches="tight", dpi=300)


def plot_calibration_curve(
    y_true: np.ndarray,
    p_orig: np.ndarray,
    prange: np.ndarray,
    cal_apparent_lowess: np.ndarray,
    cal_corrected_lowess: np.ndarray,
    title: str,
    save_dir: Optional[str] = None,
    filename: str = "calibration_curve.png",
):
    plt.figure(figsize=(12, 8), dpi=300)

    plt.scatter(p_orig, y_true, s=10, alpha=0.35, c=COL_OBS, label="Observations")
    plt.plot(prange, cal_apparent_lowess, color=COL_APP, linewidth=2.5, label="Apparent")
    plt.plot([0, 1], [0, 1], color=COL_REF, linestyle="--", linewidth=1.5, alpha=0.6, label="Perfect calibration")
    plt.plot(prange, cal_corrected_lowess, color=COL_COR, linewidth=2.5, label="Optimism-corrected")

    plt.title(title, fontsize=16)
    plt.xlabel("Predicted probability", fontsize=14)
    plt.ylabel("Observed probability", fontsize=14)
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    _savefig(save_dir, filename)
    plt.show()
