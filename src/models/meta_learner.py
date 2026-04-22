"""
meta_learner.py
---------------
Stacking ensemble (meta-learner) that combines out-of-fold predictions
from N-HiTS, XGBoost, and BNN into a single calibrated forecast.

Four meta-learners are trained (Ridge, ElasticNet, GBM, RandomForest)
and their outputs are blended using inverse-CV-MSE weights. Uncertainty
features from BNN MC Dropout and XGBoost quantile spread are included
as meta-features alongside the base OOF predictions.

Authors : [Your Name]
Paper   : "Ensemble Forecasting of Industrial Facility Emissions
           Toward 2030 Targets" – Applied Energy (submitted 2026)
License : MIT
"""

import logging
from typing import Optional, Union
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from config import META_CONFIG, RANDOM_SEED
from src.metrics import safe_mape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Meta-feature construction
# ---------------------------------------------------------------------------

def build_meta_features(
    meta_df: pd.DataFrame,
    bnn_res_dict: dict,
    xgb_res_dict: dict,
) -> pd.DataFrame:
    """Augment base OOF predictions with uncertainty and agreement features.

    Adds the following columns to ``meta_df``:

    **Uncertainty features**

    - ``BNNUncertainty``   – per-facility mean MC Dropout std (from BNN)
    - ``XGBUncertainty``   – per-row mean Q10/Q90 spread (from XGBoost)

    **Model agreement features**

    - ``NHITSXGBDiff``      – |NHITS_OOF − XGB_OOF|
    - ``NHITSBNNDiff``      – |NHITS_OOF − BNN_OOF|
    - ``BNNXGBDiff``        – |BNN_OOF  − XGB_OOF|
    - ``PredictionSpread``  – std across all three OOF predictions
    - ``AvgPrediction``     – mean across all three OOF predictions

    **Confidence score**

    - ``ConfidenceScore``   – 1 − normalised(spread + BNN uncertainty) / 2
      Higher = more model agreement, lower epistemic uncertainty.

    Args:
        meta_df:      DataFrame with columns ``Facility``, ``NHITS_OOF``,
                      ``XGB_OOF``, ``BNN_OOF``, ``y_true``.
        bnn_res_dict: Dict ``{facility: bnn_result_dict}`` from ``run_bnn``.
        xgb_res_dict: Dict ``{facility: xgb_result_dict}`` from
                      ``run_xgboost_quantile``.

    Returns:
        Enhanced copy of ``meta_df`` with all new columns.
    """
    meta = meta_df.copy()

    # --- BNN uncertainty (MC Dropout std) ---
    meta["BNNUncertainty"] = 0.0
    for fac, res in bnn_res_dict.items():
        if res and "predictions_std" in res:
            mask = meta["Facility"] == fac
            meta.loc[mask, "BNNUncertainty"] = res["predictions_std"][mask.sum():].mean() if mask.sum() > 0 else 0.0

    # --- XGBoost uncertainty (Q90 − Q10 spread) ---
    meta["XGBUncertainty"] = 0.0
    for idx, row in meta.iterrows():
        fac = row["Facility"]
        res = xgb_res_dict.get(fac)
        if res and "preds" in res:
            preds = res["preds"]
            if 0.1 in preds and 0.9 in preds:
                meta.loc[idx, "XGBUncertainty"] = float(
                    np.abs(preds[0.9] - preds[0.1]).mean()
                )

    # --- Pairwise model agreement ---
    meta["NHITSXGBDiff"]  = np.abs(meta["NHITS_OOF"] - meta["XGB_OOF"])
    meta["NHITSBNNDiff"]  = np.abs(meta["NHITS_OOF"] - meta["BNN_OOF"])
    meta["BNNXGBDiff"]    = np.abs(meta["BNN_OOF"]   - meta["XGB_OOF"])

    # --- Spread & average ---
    oof_cols = ["NHITS_OOF", "XGB_OOF", "BNN_OOF"]
    meta["PredictionSpread"] = meta[oof_cols].std(axis=1)
    meta["AvgPrediction"]    = meta[oof_cols].mean(axis=1)

    # --- Confidence score (0 = low confidence, 1 = high) ---
    eps = 1e-9
    spread_min = meta["PredictionSpread"].min()
    spread_max = meta["PredictionSpread"].max()
    spread_range = spread_max - spread_min
    
    if spread_range > eps:
        spread_norm = (meta["PredictionSpread"] - spread_min) / spread_range
    else:
        spread_norm = 0.0
    
    uncert_min = meta["BNNUncertainty"].min()
    uncert_max = meta["BNNUncertainty"].max()
    uncert_range = uncert_max - uncert_min
    
    if uncert_range > eps:
        uncert_norm = (meta["BNNUncertainty"] - uncert_min) / uncert_range
    else:
        uncert_norm = 0.0
    
    meta["ConfidenceScore"] = 1.0 - (spread_norm + uncert_norm) / 2.0
    
    # Fill any remaining NaN values with median/zero
    meta = meta.fillna(meta.median(numeric_only=True))
    meta = meta.fillna(0.0)

    logger.info("Meta-features built: %d rows, %d columns", len(meta), len(meta.columns))
    return meta


def select_meta_features() -> list[str]:
    """Return the canonical ordered list of meta-feature column names.

    Keeping this list in one place ensures the same feature order is used
    during training and inference — critical for correct meta-learner
    predictions.

    Returns:
        List of column name strings.
    """
    return [
        # Base OOF predictions
        "NHITS_OOF", "XGB_OOF", "BNN_OOF",
        # Uncertainty
        "BNNUncertainty", "XGBUncertainty",
        # Agreement
        "NHITSXGBDiff", "NHITSBNNDiff", "BNNXGBDiff",
        "PredictionSpread",
        # Derived
        "AvgPrediction", "ConfidenceScore",
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_meta_learner(
    meta_features_df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[dict, list[str], dict]:
    """Train four meta-learners and compute inverse-MSE ensemble weights.

    Meta-learners trained:
        1. **Ridge** (linear, robust to multicollinearity)
        2. **ElasticNet** (linear with L1+L2, implicit feature selection)
        3. **GBM** (non-linear, captures interaction effects)
        4. **RandomForest** (non-linear, bagging-based ensemble)

    All models are wrapped in a ``StandardScaler`` pipeline where
    appropriate (Ridge, ElasticNet). CV is performed with
    ``TimeSeriesSplit``-compatible ``cross_val_score`` using 5-fold CV.

    Weights are assigned proportional to ``1 / (CV_MSE + ε)`` — lower
    error → higher weight in the final ensemble blend.

    Args:
        meta_features_df: DataFrame output of ``build_meta_features``
                          containing ``y_true`` and all meta-feature columns.
        verbose:          If ``True``, log CV MSE and weights per learner.

    Returns:
        Tuple of:
            - ``meta_models``     : Dict ``{name: fitted model}``
            - ``meta_feature_cols``: List of feature column names used
            - ``weights``         : Dict ``{name: float}`` normalised weights
        Returns ``(None, None, None)`` if insufficient data.
    """
    cfg = META_CONFIG

    meta = meta_features_df.dropna(subset=["y_true"]).copy()
    if len(meta) < cfg.min_samples:
        logger.warning("Meta-learner: only %d samples, need %d. Skipping.", len(meta), cfg.min_samples)
        return None, None, None

    meta_feature_cols = select_meta_features()
    available = [f for f in meta_feature_cols if f in meta.columns]
    missing   = set(meta_feature_cols) - set(available)
    if missing:
        logger.warning("Meta-learner: missing features %s — using available only.", missing)
        meta_feature_cols = available

    X_meta = meta[meta_feature_cols].values
    y_meta = meta["y_true"].values
    
    # Check for and handle any remaining NaN values
    if np.isnan(X_meta).any():
        logger.warning("NaN values detected in meta-features. Filling with column medians.")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_meta = imputer.fit_transform(X_meta)

    if verbose:
        logger.info("Meta-learner training: %d samples × %d features", X_meta.shape[0], X_meta.shape[1])

    meta_models = {}
    cv_scores   = {}

    # 1. Ridge
    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(*cfg.ridge_alphas_log), cv=cfg.cv_folds),
    )
    ridge.fit(X_meta, y_meta)
    cv_scores["ridge"] = -cross_val_score(
        ridge, X_meta, y_meta, cv=cfg.cv_folds, scoring="neg_mean_squared_error"
    ).mean()
    meta_models["ridge"] = ridge

    # 2. ElasticNet
    elastic = make_pipeline(
        StandardScaler(),
        ElasticNetCV(
            l1_ratio=list(cfg.elastic_l1_ratios),
            cv=cfg.cv_folds,
            max_iter=cfg.elastic_max_iter,
        ),
    )
    elastic.fit(X_meta, y_meta)
    cv_scores["elastic"] = -cross_val_score(
        elastic, X_meta, y_meta, cv=cfg.cv_folds, scoring="neg_mean_squared_error"
    ).mean()
    meta_models["elastic"] = elastic

    # 3. GBM
    gbm = GradientBoostingRegressor(
        n_estimators=cfg.gbm_n_estimators,
        max_depth=cfg.gbm_max_depth,
        learning_rate=cfg.gbm_learning_rate,
        subsample=cfg.gbm_subsample,
        min_samples_leaf=cfg.gbm_min_samples_leaf,
        random_state=RANDOM_SEED,
    )
    gbm.fit(X_meta, y_meta)
    cv_scores["gbm"] = -cross_val_score(
        gbm, X_meta, y_meta, cv=cfg.cv_folds, scoring="neg_mean_squared_error"
    ).mean()
    meta_models["gbm"] = gbm

    # 4. Random Forest
    rf = RandomForestRegressor(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_meta, y_meta)
    cv_scores["rf"] = -cross_val_score(
        rf, X_meta, y_meta, cv=cfg.cv_folds, scoring="neg_mean_squared_error"
    ).mean()
    meta_models["rf"] = rf

    # Inverse-MSE weights
    mse_vals     = np.array(list(cv_scores.values()))
    inverse_mse  = 1.0 / (mse_vals + 1e-9)
    weight_arr   = inverse_mse / inverse_mse.sum()
    weights      = dict(zip(cv_scores.keys(), weight_arr))

    if verbose:
        for name, mse in cv_scores.items():
            logger.info("  %-10s CV-MSE=%.6f  Weight=%.3f", name, mse, weights[name])

    return meta_models, meta_feature_cols, weights


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_ensemble(
    meta_models: dict,
    weights: dict,
    meta_feature_cols: list[str],
    base_predictions: dict,
) -> tuple[float, float]:
    """Generate a weighted ensemble prediction from all meta-learners.

    Constructs a single meta-feature vector from ``base_predictions``,
    queries each meta-learner, and returns a weighted mean prediction
    alongside the cross-learner standard deviation as a meta-uncertainty
    estimate.

    Missing features in ``base_predictions`` default to 0.0.

    Args:
        meta_models:       Dict ``{name: fitted model}`` from
                           ``train_meta_learner``.
        weights:           Dict ``{name: float}`` normalised weights.
        meta_feature_cols: Ordered feature list from ``select_meta_features``.
        base_predictions:  Dict mapping feature names to scalar values,
                           e.g. ``{"NHITS_OOF": 52.3, "XGB_OOF": 49.1, ...}``.

    Returns:
        Tuple of:
            - ``mean_prediction`` : float – weighted ensemble point estimate
            - ``prediction_std``  : float – std across meta-learners
              (proxy for meta-level uncertainty)
    """
    meta_input = np.array([
        base_predictions.get(feat, 0.0)
        for feat in meta_feature_cols
    ]).reshape(1, -1)

    raw_preds     = []
    weighted_preds = []

    for name, model in meta_models.items():
        pred   = model.predict(meta_input)[0]
        weight = weights[name]
        raw_preds.append(pred)
        weighted_preds.append(pred * weight)

    mean_prediction = float(sum(weighted_preds))
    prediction_std  = float(np.std(raw_preds))

    return mean_prediction, prediction_std


# ---------------------------------------------------------------------------
# OOF Evaluation
# ---------------------------------------------------------------------------

def evaluate_meta_learner(
    meta_models: dict,
    weights: dict,
    meta_feature_cols: list[str],
    meta_features_df: pd.DataFrame,
    individual_oof_cols: Optional[dict] = None,
) -> dict:
    """Evaluate ensemble on OOF data and compare to individual base models.

    Args:
        meta_models:        Trained meta-learner dict.
        weights:            Normalised weight dict.
        meta_feature_cols:  Feature columns used during training.
        meta_features_df:   DataFrame with OOF features and ``y_true``.
        individual_oof_cols: Optional dict ``{model_name: oof_col_name}``
                             for per-model MAPE comparison, e.g.
                             ``{"NHITS": "NHITS_OOF", "XGB": "XGB_OOF"}``.

    Returns:
        Dict with keys ``ensemble_mape``, ``ensemble_rmse``,
        ``individual_mapes`` (dict), ``improvement`` (dict).
    """
    meta_oof = meta_features_df.dropna(subset=["y_true"]).copy()
    X_oof    = meta_oof[meta_feature_cols].values
    y_oof    = meta_oof["y_true"].values

    ensemble_preds = []
    for i in range(len(X_oof)):
        pred_dict = dict(zip(meta_feature_cols, X_oof[i]))
        mean_pred, _ = predict_ensemble(meta_models, weights, meta_feature_cols, pred_dict)
        ensemble_preds.append(mean_pred)

    ensemble_preds = np.array(ensemble_preds)
    ensemble_mape  = safe_mape(y_oof, ensemble_preds)
    ensemble_rmse  = mean_squared_error(y_oof, ensemble_preds) ** 0.5

    logger.info("Ensemble OOF  MAPE=%.2f%%  RMSE=%.4f", ensemble_mape, ensemble_rmse)

    individual_mapes = {}
    if individual_oof_cols:
        for model_name, col in individual_oof_cols.items():
            if col in meta_oof.columns:
                mape = safe_mape(y_oof, meta_oof[col].values)
                individual_mapes[model_name] = mape
                improvement = mape - ensemble_mape
                logger.info("  %-8s MAPE=%.2f%%  Improvement vs ensemble: %+.2f%%",
                            model_name, mape, improvement)

    return {
        "ensemble_mape":    ensemble_mape,
        "ensemble_rmse":    ensemble_rmse,
        "individual_mapes": individual_mapes,
        "improvement":      {k: v - ensemble_mape for k, v in individual_mapes.items()},
    }
