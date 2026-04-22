"""
xgboost_model.py
----------------
XGBoost quantile regression model for per-facility emissions forecasting.

Trains three quantile models (Q10, Q50, Q90) per facility, returns
predictions, metrics, and SHAP values.

Authors : [Your Name]
Paper   : "Ensemble Forecasting of Industrial Facility Emissions
           Toward 2030 Targets" – Applied Energy (submitted 2026)
License : MIT
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from config import XGB_CONFIG, SHAP_CONFIG, TARGET_COL, TEST_MONTHS, MIN_TRAIN_ROWS
from src.metrics import safe_mape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_xgboost_quantile(
    df: pd.DataFrame,
    facility: str,
    feature_cols: list[str],
    metric: str = TARGET_COL,
    test_months: int = TEST_MONTHS,
) -> dict:
    """Train three XGBoost quantile regressors (Q10, Q50, Q90) for one facility.

    Uses the median model (Q50) to compute evaluation metrics. All models
    operate in the **original (unscaled) target space** — no y-scaling is
    applied.

    Args:
        df:           Full enhanced DataFrame for all facilities.
        facility:     Facility ID to filter and train on.
        feature_cols: List of feature column names to use as inputs.
        metric:       Target column name (default: ``EmissionstCO2``).
        test_months:  Number of tail months to hold out as test set.

    Returns:
        Dict with keys:
            - ``facility``   : str
            - ``mape``       : float – median model MAPE (%)
            - ``rmse``       : float – median model RMSE
            - ``models``     : dict  – {quantile: fitted XGBRegressor}
            - ``preds``      : dict  – {quantile: np.ndarray of test predictions}
            - ``X_test``     : np.ndarray
            - ``y_test``     : np.ndarray
            - ``n_features`` : int
        On failure, ``mape`` / ``rmse`` are ``np.nan`` and ``models`` is ``{}``.
    """
    df_fac = df[df["Facility"] == facility].copy()
    required_cols = list(feature_cols) + [metric]
    df_fac = df_fac.dropna(subset=required_cols)

    if len(df_fac) < MIN_TRAIN_ROWS:
        logger.warning("XGBoost: insufficient data for %s (%d rows).", facility, len(df_fac))
        return {"facility": facility, "mape": np.nan, "rmse": np.nan, "models": {}}

    X = df_fac[list(feature_cols)].values
    y = df_fac[metric].values

    split_idx = len(df_fac) - test_months
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    models, preds = {}, {}
    cfg = XGB_CONFIG

    for q in cfg.quantiles:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            random_state=cfg.random_state,
        )
        model.fit(X_train, y_train, verbose=False)
        preds[q] = model.predict(X_test)
        models[q] = model

    mape = safe_mape(y_test, preds[0.5])
    rmse = mean_squared_error(y_test, preds[0.5]) ** 0.5

    logger.info("XGBoost [%s] MAPE=%.2f%%  RMSE=%.4f", facility, mape, rmse)

    return {
        "facility":   facility,
        "mape":       mape,
        "rmse":       rmse,
        "models":     models,
        "preds":      preds,
        "X_test":     X_test,
        "y_test":     y_test,
        "n_features": X.shape[1],
    }


# ---------------------------------------------------------------------------
# 2030 Inference
# ---------------------------------------------------------------------------

def predict_xgboost_2030(
    models: dict,
    X_2030: pd.DataFrame,
) -> dict:
    """Generate Q10 / Q50 / Q90 point predictions for a 2030 feature vector.

    Args:
        models: Dict ``{quantile: fitted XGBRegressor}`` from
                ``run_xgboost_quantile``.
        X_2030: Single-row DataFrame of 2030 features (output of
                ``prepare_2030_features``).

    Returns:
        Dict ``{0.1: float, 0.5: float, 0.9: float}`` of clipped (≥0)
        predictions in original tCO₂ scale.
    """
    X_arr = X_2030.values if hasattr(X_2030, "values") else np.array(X_2030)

    return {
        q: float(np.clip(model.predict(X_arr)[0], 0, None))
        for q, model in models.items()
    }


# ---------------------------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------------------------

def shap_xgboost(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    features: list[str],
    facility: str,
    show_plot: bool = True,
) -> Optional[np.ndarray]:
    """Compute SHAP values for the XGBoost median model via TreeExplainer.

    Uses ``shap.TreeExplainer`` — exact, fast, and appropriate for
    gradient-boosted trees.

    Args:
        model:     Fitted XGBRegressor (Q50 model).
        X_test:    Test feature array (n_samples × n_features).
        features:  Feature name list matching ``X_test`` columns.
        facility:  Facility ID (used in plot title).
        show_plot: If ``True``, render a SHAP beeswarm summary plot.

    Returns:
        SHAP-like contribution array of shape ``(n_samples, n_features)``,
        or ``None`` on failure.

    Notes:
        If ``shap.TreeExplainer`` fails (common in some Windows DLL setups),
        this function falls back to XGBoost native feature contributions via
        ``pred_contribs=True``.
    """
    if hasattr(X_test, "values"):
        X_test = X_test.values
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    X_sample = X_test[:SHAP_CONFIG.xgb_max_samples]

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)

        if show_plot:
            shap.summary_plot(
                shap_vals,
                X_sample,
                feature_names=features,
                show=False,
                max_display=SHAP_CONFIG.max_display,
            )
            plt.title(f"XGBoost SHAP – {facility}")
            plt.tight_layout()
            plt.show()

        logger.info("XGBoost SHAP computed for %s  shape=%s", facility, shap_vals.shape)
        return shap_vals

    except Exception as exc:
        logger.warning("XGBoost SHAP failed for %s (%s). Trying native pred_contribs fallback.", facility, exc)

        try:
            booster = model.get_booster()
            dmat = xgb.DMatrix(X_sample, feature_names=features)
            contribs = booster.predict(dmat, pred_contribs=True)

            # XGBoost returns (n_samples, n_features + 1), with last column as bias.
            shap_vals = np.asarray(contribs)[:, :-1]

            if show_plot:
                mean_abs = np.abs(shap_vals).mean(axis=0)
                order = np.argsort(mean_abs)[::-1][:SHAP_CONFIG.max_display]
                plt.figure(figsize=(8, 4.5))
                plt.barh(
                    [features[i] for i in order][::-1],
                    mean_abs[order][::-1],
                    color="#4C78A8",
                )
                plt.title(f"XGBoost Feature Contributions (fallback) – {facility}")
                plt.xlabel("Mean |contribution|")
                plt.tight_layout()
                plt.show()

            logger.info("XGBoost fallback contributions computed for %s  shape=%s", facility, shap_vals.shape)
            return shap_vals

        except Exception as fallback_exc:
            logger.error("XGBoost contribution fallback failed for %s: %s", facility, fallback_exc)
            return None
