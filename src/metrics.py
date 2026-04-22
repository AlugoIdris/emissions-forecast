"""
metrics.py
----------
Shared metric helpers for robust evaluation.
"""

import numpy as np


def safe_mape(y_true, y_pred, eps: float = 1.0) -> float:
    """Compute MAPE (%) with denominator floor to avoid near-zero blow-ups.

    Args:
        y_true: Ground-truth array-like.
        y_pred: Prediction array-like.
        eps: Minimum absolute denominator in target units.

    Returns:
        MAPE in percent.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    valid = ~(np.isnan(yt) | np.isnan(yp))
    if not np.any(valid):
        return float("nan")

    yt = yt[valid]
    yp = yp[valid]
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)
