"""
evaluation.py
-------------
Stakeholder-focused evaluation metrics, risk analysis, and 2030 target
compliance assessment for the ensemble emissions forecasting pipeline.

Covers:
    - Accuracy metrics (MAPE, RMSE, MAE, within-N% coverage)
    - Uncertainty calibration (PI coverage, sharpness)
    - Risk categorisation (per-facility probability of meeting 2030 target)
    - Ensemble probability computation via inverse-MAPE weighted bootstrap
    - Regional roll-up and summary statistics

Authors : Idris Alugo
Paper   : "Long-Horizon Emissions Forecasting for 2030 Target Assessment:
           A Comparative Study of N-HiTS, XGBoost, and Bayesian Models
           in Fast-Moving Consumer Goods Supply Chains" – Applied Energy (2026)
License : MIT
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from config import TARGET_COL, RANDOM_SEED, TARGET_REDUCTION_RATE, BASELINE_YEAR
from src.metrics import safe_mape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk categorisation
# ---------------------------------------------------------------------------

def categorize_risk(probability: float) -> str:
    """Map a compliance probability to a stakeholder risk label.

    Thresholds are aligned with the paper's risk framework (Section 4.2):
        - ≥ 0.70 → Low Risk
        - ≥ 0.30 → Medium Risk
        - ≥ 0.10 → High Risk
        - <  0.10 → Critical Risk

    Args:
        probability: Float in [0, 1] or ``np.nan``.

    Returns:
        Risk label string.
    """
    if np.isnan(probability):
        return "Unknown"
    if probability >= 0.70:
        return "Low Risk"
    if probability >= 0.30:
        return "Medium Risk"
    if probability >= 0.10:
        return "High Risk"
    return "Critical Risk"


# ---------------------------------------------------------------------------
# Per-facility stakeholder metrics
# ---------------------------------------------------------------------------

def compute_stakeholder_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    predictions_std: np.ndarray,
    confidence_level: float = 0.90,
) -> dict:
    """Compute comprehensive accuracy, calibration and risk metrics.

    Designed for sustainability and operations stakeholders — returns
    both standard ML metrics and business-interpretable measures such as
    target miss rate and average overshoot percentage.

    Args:
        predictions:     Point predictions array (n_facilities,).
        targets:         2030 emission target array (n_facilities,).
        predictions_std: Prediction uncertainty (std dev) array.
        confidence_level: Coverage level for prediction intervals (default 90%).

    Returns:
        Dict with keys:
            Standard: ``MAPE``, ``RMSE``, ``MAE``
            Business: ``Within10Pct``, ``Within20Pct``
            Calibration: ``PICoverage90``, ``Sharpness``, ``AvgUncertainty``
            Risk: ``TargetMissRate``, ``AvgOvershoot``, ``AvgOvershootPct``,
                  ``HighRiskCount``, ``CriticalRiskCount``
    """
    predictions    = np.asarray(predictions,    dtype=float)
    targets        = np.asarray(targets,        dtype=float)
    predictions_std = np.asarray(predictions_std, dtype=float)

    # Drop NaN rows
    valid = ~(np.isnan(predictions) | np.isnan(targets) | np.isnan(predictions_std))
    predictions     = predictions[valid]
    targets         = targets[valid]
    predictions_std = predictions_std[valid]

    nan_result = {k: np.nan for k in [
        "MAPE", "RMSE", "MAE", "Within10Pct", "Within20Pct",
        "PICoverage90", "Sharpness", "AvgUncertainty",
        "TargetMissRate", "AvgOvershoot", "AvgOvershootPct",
        "HighRiskCount", "CriticalRiskCount",
    ]}

    if len(predictions) == 0:
        logger.warning("compute_stakeholder_metrics: no valid samples after NaN drop.")
        return nan_result

    # --- Standard accuracy ---
    mape = safe_mape(targets, predictions)
    rmse = mean_squared_error(targets, predictions) ** 0.5
    mae  = float(np.mean(np.abs(predictions - targets)))

    # --- Business-friendly accuracy ---
    eps          = 1e-9
    within_10pct = float(np.mean(np.abs(predictions - targets) / (targets + eps) <= 0.10) * 100)
    within_20pct = float(np.mean(np.abs(predictions - targets) / (targets + eps) <= 0.20) * 100)

    # --- Prediction interval calibration ---
    z_score     = norm.ppf(1 - (1 - confidence_level) / 2)
    lower_bound = predictions - z_score * predictions_std
    upper_bound = predictions + z_score * predictions_std
    pi_coverage = float(np.mean((targets >= lower_bound) & (targets <= upper_bound)) * 100)
    sharpness   = float(np.mean(upper_bound - lower_bound))

    # --- Risk / target metrics ---
    target_miss_rate = float(np.mean(predictions > targets) * 100)
    overshoot_mask   = predictions > targets
    if overshoot_mask.any():
        avg_overshoot     = float(np.mean(predictions[overshoot_mask] - targets[overshoot_mask]))
        avg_overshoot_pct = float(
            np.mean(
                (predictions[overshoot_mask] - targets[overshoot_mask])
                / (targets[overshoot_mask] + eps)
            ) * 100
        )
    else:
        avg_overshoot     = 0.0
        avg_overshoot_pct = 0.0

    high_risk_count     = int(np.sum(predictions > targets * 1.10))
    critical_risk_count = int(np.sum(predictions > targets * 1.20))

    return {
        "MAPE":             mape,
        "RMSE":             rmse,
        "MAE":              mae,
        "Within10Pct":      within_10pct,
        "Within20Pct":      within_20pct,
        "PICoverage90":     pi_coverage,
        "Sharpness":        sharpness,
        "AvgUncertainty":   float(predictions_std.mean()),
        "TargetMissRate":   target_miss_rate,
        "AvgOvershoot":     avg_overshoot,
        "AvgOvershootPct":  avg_overshoot_pct,
        "HighRiskCount":    high_risk_count,
        "CriticalRiskCount": critical_risk_count,
    }


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def compare_model_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise MAPE and RMSE across all facilities for each model.

    Args:
        results_df: DataFrame with columns ``Facility``, ``NHITSMAPE``,
                    ``NHITSRMSE``, ``XGBoostMAPE``, ``XGBoostRMSE``,
                    ``BNNMAPE``, ``BNNRMSE``.

    Returns:
        Summary DataFrame with mean ± std MAPE and RMSE per model.
    """
    models = {
        "N-HiTS":   ("NHITSMAPE",     "NHITSRMSE"),
        "XGBoost":  ("XGBoostMAPE",   "XGBoostRMSE"),
        "BNN":      ("BNNMAPE",       "BNNRMSE"),
    }

    rows = []
    for model_name, (mape_col, rmse_col) in models.items():
        if mape_col not in results_df.columns:
            continue
        mape_vals = results_df[mape_col].dropna()
        rmse_vals = results_df[rmse_col].dropna()
        rows.append({
            "Model":        model_name,
            "Mean MAPE (%)": round(mape_vals.mean(), 2),
            "Std MAPE (%)":  round(mape_vals.std(), 2),
            "Mean RMSE":     round(rmse_vals.mean(), 4),
            "Std RMSE":      round(rmse_vals.std(), 4),
            "N Facilities":  len(mape_vals),
        })

    summary = pd.DataFrame(rows).set_index("Model")
    logger.info("Model comparison computed for %d models.", len(summary))
    return summary


# ---------------------------------------------------------------------------
# Baseline & target preparation
# ---------------------------------------------------------------------------

def prepare_facility_targets(
    df: pd.DataFrame,
    baseline_year: int = BASELINE_YEAR,
    target_reduction: float = TARGET_REDUCTION_RATE,
    metric: str = TARGET_COL,
) -> pd.DataFrame:
    """Compute per-facility 2030 baseline emissions and reduction targets.

    Targets are computed from the original (unscaled) emission values.
    The reduction target is applied multiplicatively:
        ``Target2030 = BaselineEmissions × (1 − target_reduction)``

    Args:
        df:               Full emissions DataFrame (all facilities, all years).
        baseline_year:    Reference year for baseline calculation (default 2022).
        target_reduction: Fractional reduction required by 2030 (default 0.10 = 10%).
        metric:           Target column name.

    Returns:
        DataFrame with columns: ``Facility``, ``Region``, ``PolicyWeight``,
        ``BaselineEmissions``, ``Target2030``, ``ReductionRequired``.
    """
    baseline_df = (
        df[df["Date"].dt.year == baseline_year]
        .groupby(["Facility", "Region", "PolicyWeight"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "BaselineEmissions"})
    )

    baseline_df["Target2030"]        = baseline_df["BaselineEmissions"] * (1 - target_reduction)
    baseline_df["ReductionRequired"] = baseline_df["BaselineEmissions"] * target_reduction

    logger.info(
        "Targets prepared: %d facilities | baseline_year=%d | reduction=%.0f%%",
        len(baseline_df), baseline_year, target_reduction * 100,
    )
    logger.debug(
        "Target range: [%.2f, %.2f] tCO2  mean=%.2f",
        baseline_df["Target2030"].min(),
        baseline_df["Target2030"].max(),
        baseline_df["Target2030"].mean(),
    )
    return baseline_df


# ---------------------------------------------------------------------------
# Compliance probability computation
# ---------------------------------------------------------------------------

def compute_calibrated_probabilities(
    facility_baseline: pd.DataFrame,
    nhits_preds: dict,
    xgb_quantile_preds: dict,
    bnn_mc_preds: dict,
    results_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compute MAPE-weighted ensemble compliance probabilities per facility.

    For each facility, three predictive distributions are built:
        - **BNN**: uses raw MC Dropout samples directly
        - **XGBoost**: bootstraps samples by interpolating Q10/Q50/Q90
        - **N-HiTS**: draws normal samples using mean ± IQR/1.35 as std

    Distributions are mixed using inverse-MAPE weights so better-performing
    models contribute more to the ensemble probability estimate.

    Args:
        facility_baseline:  Output of ``prepare_facility_targets``.
        nhits_preds:        Dict ``{facility: {0.1: val, 0.5: val, 0.9: val}}``.
        xgb_quantile_preds: Dict ``{facility: {0.1: val, 0.5: val, 0.9: val}}``.
        bnn_mc_preds:       Dict ``{facility: np.ndarray of MC samples}``.
        results_df:         Per-facility MAPE results from the training loop.
        n_bootstrap:        Bootstrap sample count per model (default 1000).
        random_state:       Random seed for reproducibility.

    Returns:
        DataFrame with one row per facility containing:
            ``Facility``, ``Target2030``,
            ``ProbMeetTargetNHITS``, ``ProbMeetTargetXGBoost``,
            ``ProbMeetTargetBNN``, ``ProbMeetTargetEnsemble``,
            ``PredictionNHITS``, ``PredictionXGBoost``,
            ``PredictionBNN``, ``PredictionEnsemble``,
            ``UncertaintyNHITS``, ``UncertaintyXGBoost``,
            ``UncertaintyBNN``, ``UncertaintyEnsemble``,
            ``RiskLevelEnsemble``
    """
    np.random.seed(random_state)
    records = []

    for _, row in facility_baseline.iterrows():
        fac    = row["Facility"]
        target = row["Target2030"]

        # --- BNN: use MC samples directly ---
        bnn_samples = bnn_mc_preds.get(fac, np.array([]))
        if bnn_samples.size > 0:
            prob_bnn = float(np.mean(bnn_samples <= target))
            bnn_mean = float(bnn_samples.mean())
            bnn_std  = float(bnn_samples.std())
        else:
            prob_bnn = bnn_mean = bnn_std = np.nan
            bnn_samples = np.array([])

        # --- XGBoost: bootstrap via quantile interpolation ---
        xgb_q = xgb_quantile_preds.get(fac, {})
        if len(xgb_q) >= 3 and all(q in xgb_q for q in [0.1, 0.5, 0.9]):
            q_probs = np.array([0.1, 0.5, 0.9])
            q_vals  = np.array([xgb_q[0.1], xgb_q[0.5], xgb_q[0.9]])
            uniform_samples = np.random.uniform(0, 1, n_bootstrap)
            xgb_samples     = np.interp(uniform_samples, q_probs, q_vals)
            prob_xgb = float(np.mean(xgb_samples <= target))
            xgb_mean = float(xgb_q[0.5])
            xgb_std  = float((xgb_q[0.9] - xgb_q[0.1]) / 2.56)
        else:
            prob_xgb = xgb_mean = xgb_std = np.nan
            xgb_samples = np.array([])

        # --- N-HiTS: normal distribution from quantile IQR ---
        nhits_q = nhits_preds.get(fac, {})
        if nhits_q:
            nhits_mean = nhits_q.get(0.5, np.nan)
            iqr        = nhits_q.get(0.9, 0) - nhits_q.get(0.1, 0)
            nhits_std  = iqr / 1.35
            if nhits_std > 0 and not np.isnan(nhits_mean):
                nhits_samples = np.random.normal(nhits_mean, nhits_std, n_bootstrap)
                prob_nhits    = float(np.mean(nhits_samples <= target))
            else:
                nhits_samples = np.array([])
                prob_nhits    = np.nan
        else:
            nhits_mean = nhits_std = prob_nhits = np.nan
            nhits_samples = np.array([])

        # --- Ensemble: inverse-MAPE weighted mixture ---
        fac_results = results_df[results_df["Facility"] == fac]
        if len(fac_results) > 0:
            nhits_mape = fac_results["NHITSMAPE"].values[0]
            xgb_mape   = fac_results["XGBoostMAPE"].values[0]
            bnn_mape   = fac_results["BNNMAPE"].values[0]
        else:
            nhits_mape = xgb_mape = bnn_mape = 30.0

        def _safe_weight(mape_val: float) -> float:
            return 1.0 / (mape_val + 1.0) if (not np.isnan(mape_val) and mape_val > 0) else 0.0

        # Exclude models whose MAPE exceeds this threshold — they are degenerate
        # fits that would distort the ensemble probability (e.g. BNN predicting
        # near-zero emissions for every facility).  Fall back to all available
        # models if none pass the gate.
        MAPE_QUALITY_THRESHOLD = 80.0

        candidate_models = [
            (bnn_samples,   bnn_mape),
            (xgb_samples,   xgb_mape),
            (nhits_samples, nhits_mape),
        ]
        quality_models = [
            (s, m) for s, m in candidate_models
            if s.size > 0 and not np.isnan(m) and m <= MAPE_QUALITY_THRESHOLD
        ]
        if not quality_models:
            quality_models = [(s, m) for s, m in candidate_models if s.size > 0]

        all_samples = [s for s, m in quality_models]
        weights     = [_safe_weight(m) for s, m in quality_models]

        if all_samples:
            weights     = np.array(weights)
            weights     = weights / weights.sum()
            n_per_model = n_bootstrap // len(all_samples)
            ensemble_samples = np.concatenate([
                np.random.choice(s, size=int(n_per_model * w * len(all_samples)), replace=True)
                for s, w in zip(all_samples, weights)
            ])
            prob_ensemble    = float(np.mean(ensemble_samples <= target))
            ensemble_mean    = float(ensemble_samples.mean())
            ensemble_std     = float(ensemble_samples.std())
        else:
            prob_ensemble = ensemble_mean = ensemble_std = np.nan

        records.append({
            "Facility":               fac,
            "Target2030":             target,
            "ProbMeetTargetNHITS":    prob_nhits,
            "ProbMeetTargetXGBoost":  prob_xgb,
            "ProbMeetTargetBNN":      prob_bnn,
            "ProbMeetTargetEnsemble": prob_ensemble,
            "PredictionNHITS":        nhits_mean,
            "PredictionXGBoost":      xgb_mean,
            "PredictionBNN":          bnn_mean,
            "PredictionEnsemble":     ensemble_mean,
            "UncertaintyNHITS":       nhits_std,
            "UncertaintyXGBoost":     xgb_std,
            "UncertaintyBNN":         bnn_std,
            "UncertaintyEnsemble":    ensemble_std,
            "RiskLevelEnsemble":      categorize_risk(prob_ensemble if not np.isnan(prob_ensemble) else np.nan),
        })

    risk_df = pd.DataFrame(records)
    logger.info(
        "Compliance probabilities computed for %d facilities. Risk distribution:\n%s",
        len(risk_df),
        risk_df["RiskLevelEnsemble"].value_counts().to_string(),
    )
    return risk_df


# ---------------------------------------------------------------------------
# Regional roll-up
# ---------------------------------------------------------------------------

def compute_regional_summary(
    risk_df: pd.DataFrame,
    facility_baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate facility-level risk metrics to the regional level.

    Joins ``risk_df`` with ``facility_baseline`` to include region and
    policy weight, then rolls up mean probabilities, total baseline
    emissions, total targets, and risk distribution counts per region.

    Args:
        risk_df:            Output of ``compute_calibrated_probabilities``.
        facility_baseline:  Output of ``prepare_facility_targets``.

    Returns:
        Regional summary DataFrame sorted by ``ProbMeetTargetEnsemble``
        ascending (highest-risk regions first).
    """
    merged = risk_df.merge(
        facility_baseline[["Facility", "Region", "PolicyWeight", "BaselineEmissions"]],
        on="Facility",
        how="left",
    )

    regional = (
        merged.groupby(["Region", "PolicyWeight"])
        .agg(
            FacilityCount          = ("Facility",               "count"),
            AvgProbEnsemble        = ("ProbMeetTargetEnsemble", "mean"),
            AvgProbNHITS           = ("ProbMeetTargetNHITS",    "mean"),
            AvgProbXGBoost         = ("ProbMeetTargetXGBoost",  "mean"),
            AvgProbBNN             = ("ProbMeetTargetBNN",      "mean"),
            TotalBaselineEmissions = ("BaselineEmissions",      "sum"),
            TotalTarget2030        = ("Target2030",             "sum"),
            CriticalRiskCount      = ("RiskLevelEnsemble",
                                      lambda x: (x == "Critical Risk").sum()),
            HighRiskCount          = ("RiskLevelEnsemble",
                                      lambda x: (x == "High Risk").sum()),
        )
        .reset_index()
        .sort_values("AvgProbEnsemble", ascending=True)
    )

    regional["RegionalRiskLabel"] = regional["AvgProbEnsemble"].apply(categorize_risk)

    logger.info("Regional summary computed for %d regions.", len(regional))
    return regional


# ---------------------------------------------------------------------------
# Portfolio-level compliance probability
# ---------------------------------------------------------------------------

def compute_portfolio_probability(
    risk_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    results_df: pd.DataFrame,
    n_bootstrap: int = 10_000_000,
    mape_threshold: float = 80.0,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Compute portfolio-level compliance metrics via joint bootstrap simulation.

    Two metrics are returned:

    * **P_portfolio**: fraction of bootstrap iterations in which *all* facilities
      simultaneously meet their 2030 target.  Uses the corrected ensemble
      (quality-gate: models with MAPE > ``mape_threshold`` are excluded).

    * **E_compliant**: expected number of compliant facilities
      ``= Σ P_i`` (linearity of expectation, independent of facility
      independence assumption).

    Args:
        risk_df:       Output of ``compute_calibrated_probabilities`` — must
                       contain ``Facility``, ``Target2030``,
                       ``ProbMeetTargetNHITS``, ``ProbMeetTargetXGBoost``,
                       ``ProbMeetTargetBNN``.
        forecasts_df:  ``forecasts_2030.csv`` DataFrame with per-facility
                       Q10/Q50/Q90 for NHITS, XGB, BNN.
        results_df:    Per-facility MAPE results from the training loop.
        n_bootstrap:   Number of joint samples (default 10 M for stable
                       estimates of sub-0.1% probabilities).
        mape_threshold: Exclude a model from the ensemble if its MAPE exceeds
                        this value (same gate as ``compute_calibrated_probabilities``).
        random_state:  Random seed.

    Returns:
        Dict with keys:
            ``P_portfolio``       – joint probability (float, 0–1)
            ``P_portfolio_pct``   – same as percentage string "x.xxxx%"
            ``E_compliant``       – expected number of compliant facilities
            ``E_compliant_std``   – std dev of compliant count (Poisson-Binomial)
            ``N_facilities``      – number of facilities included
            ``N_bootstrap``       – samples used
            ``per_facility``      – DataFrame with Facility, P_i, contribution
    """
    np.random.seed(random_state)

    def _safe_weight(m: float) -> float:
        return 1.0 / (m + 1.0) if (not np.isnan(m) and m > 0) else 0.0

    def _build_samples(fac: str, n: int):
        fc_row  = forecasts_df[forecasts_df["Facility"] == fac]
        acc_row = results_df[results_df["Facility"] == fac]
        if fc_row.empty or acc_row.empty:
            return None
        fc  = fc_row.iloc[0]
        acc = acc_row.iloc[0]

        nhits_mape = acc["NHITSMAPE"]
        xgb_mape   = acc["XGBoostMAPE"]
        bnn_mape   = acc["BNNMAPE"]

        model_samples, model_mapes = [], []

        # NHITS: normal approximation from Q10/Q90
        nh_q10, nh_q50, nh_q90 = fc["NHITS_Q10"], fc["NHITS_Q50"], fc["NHITS_Q90"]
        nh_std = (nh_q90 - nh_q10) / 2.56
        if not np.isnan(nh_q50) and nh_std > 0 and nhits_mape <= mape_threshold:
            model_samples.append(np.random.normal(nh_q50, nh_std, n))
            model_mapes.append(nhits_mape)

        # XGBoost: quantile interpolation
        xg_q10, xg_q50, xg_q90 = fc["XGB_Q10"], fc["XGB_Q50"], fc["XGB_Q90"]
        if not any(np.isnan([xg_q10, xg_q50, xg_q90])) and xgb_mape <= mape_threshold:
            u = np.random.uniform(0, 1, n)
            model_samples.append(np.interp(u, [0.1, 0.5, 0.9], [xg_q10, xg_q50, xg_q90]))
            model_mapes.append(xgb_mape)

        # BNN: quantile interpolation
        bn_q10, bn_q50, bn_q90 = fc["BNN_Q10"], fc["BNN_Q50"], fc["BNN_Q90"]
        if not any(np.isnan([bn_q10, bn_q50, bn_q90])) and bnn_mape <= mape_threshold:
            u = np.random.uniform(0, 1, n)
            model_samples.append(np.interp(u, [0.1, 0.5, 0.9], [bn_q10, bn_q50, bn_q90]))
            model_mapes.append(bnn_mape)

        # Fallback: use all models if none passed the gate
        if not model_samples:
            for (q10, q50, q90), m in [
                ((nh_q10, nh_q50, nh_q90), nhits_mape),
                ((xg_q10, xg_q50, xg_q90), xgb_mape),
                ((bn_q10, bn_q50, bn_q90), bnn_mape),
            ]:
                if not any(np.isnan([q10, q50, q90])):
                    u = np.random.uniform(0, 1, n)
                    model_samples.append(np.interp(u, [0.1, 0.5, 0.9], [q10, q50, q90]))
                    model_mapes.append(m)

        if not model_samples:
            return None

        ws    = np.array([_safe_weight(m) for m in model_mapes])
        ws    = ws / ws.sum()
        n_each = [max(1, int(round(w * n))) for w in ws]
        n_each[-1] += n - sum(n_each)
        combined = np.concatenate([
            np.random.choice(s, size=ni, replace=True)
            for s, ni in zip(model_samples, n_each)
        ])
        np.random.shuffle(combined)
        return combined[:n]

    # Exclude F2 (no model predictions)
    active = risk_df[risk_df["ProbMeetTargetEnsemble"].notna()].copy()
    targets = dict(zip(active["Facility"], active["Target2030"]))
    facilities = sorted(targets.keys())

    fac_samples = {}
    for fac in facilities:
        s = _build_samples(fac, n_bootstrap)
        if s is not None:
            fac_samples[fac] = s

    active_facs = [f for f in facilities if f in fac_samples]

    # Joint (portfolio) probability
    sample_matrix = np.column_stack([fac_samples[f] for f in active_facs])
    target_vector = np.array([targets[f] for f in active_facs])
    all_meet      = np.all(sample_matrix <= target_vector[np.newaxis, :], axis=1)
    p_portfolio   = float(np.mean(all_meet))

    # Per-facility marginal P_i from the same samples
    p_i = {f: float(np.mean(fac_samples[f] <= targets[f])) for f in active_facs}

    # E[compliant] = Σ P_i  (linearity of expectation)
    e_compliant     = sum(p_i.values())
    var_compliant   = sum(p * (1 - p) for p in p_i.values())
    e_compliant_std = float(var_compliant ** 0.5)

    per_facility_df = pd.DataFrame([
        {"Facility": f, "P_i": p_i[f], "Target2030": targets[f]}
        for f in active_facs
    ]).sort_values("Facility").reset_index(drop=True)

    logger.info(
        "Portfolio metrics: P_portfolio=%.6f%%  E_compliant=%.2f ± %.2f  N=%d",
        p_portfolio * 100, e_compliant, e_compliant_std, len(active_facs),
    )

    return {
        "P_portfolio":     p_portfolio,
        "P_portfolio_pct": f"{p_portfolio * 100:.4f}%",
        "E_compliant":     e_compliant,
        "E_compliant_std": e_compliant_std,
        "N_facilities":    len(active_facs),
        "N_bootstrap":     n_bootstrap,
        "per_facility":    per_facility_df,
    }


# ---------------------------------------------------------------------------
# SHAP aggregation
# ---------------------------------------------------------------------------

def aggregate_shap_records(
    shap_records: list[dict],
    features: list[str],
) -> pd.DataFrame:
    """Aggregate per-facility SHAP records into a cross-model importance table.

    Args:
        shap_records: List of dicts with keys ``Facility``, ``Feature``,
                      ``MeanAbsSHAPNHITS``, ``MeanAbsSHAPXGB``,
                      ``MeanAbsSHAPBNN``.
        features:     Full feature list (used for ordering).

    Returns:
        DataFrame sorted by ``AvgContribution`` (desc) with cumulative
        percentage column. Top features driving emissions are at the top.
    """
    if not shap_records:
        logger.warning("aggregate_shap_records: no SHAP records provided.")
        return pd.DataFrame()

    shap_df = pd.DataFrame(shap_records)
    shap_by_feature = (
        shap_df.groupby("Feature")[["MeanAbsSHAPNHITS", "MeanAbsSHAPXGB", "MeanAbsSHAPBNN"]]
        .mean()
        .reset_index()
    )

    eps = 1e-9
    for col, pct_col in [
        ("MeanAbsSHAPNHITS", "NHITSContribution"),
        ("MeanAbsSHAPXGB",   "XGBContribution"),
        ("MeanAbsSHAPBNN",   "BNNContribution"),
    ]:
        total = shap_by_feature[col].sum() + eps
        shap_by_feature[pct_col] = shap_by_feature[col] / total * 100

    shap_by_feature["AvgContribution"] = shap_by_feature[
        ["NHITSContribution", "XGBContribution", "BNNContribution"]
    ].mean(axis=1)

    shap_summary = shap_by_feature.sort_values("AvgContribution", ascending=False).reset_index(drop=True)
    shap_summary["Cumulative"] = shap_summary["AvgContribution"].cumsum()

    top5_pct  = shap_summary.head(5)["AvgContribution"].sum()
    top10_pct = shap_summary.head(10)["AvgContribution"].sum()
    features_for_80 = int((shap_summary["Cumulative"] <= 80).sum()) + 1

    logger.info(
        "SHAP aggregated: top-5=%.1f%%  top-10=%.1f%%  features for 80%% coverage=%d",
        top5_pct, top10_pct, features_for_80,
    )
    return shap_summary


# ---------------------------------------------------------------------------
# Long-horizon evaluation
# ---------------------------------------------------------------------------

def evaluate_horizons(
    df: pd.DataFrame,
    facilities: list,
    feature_cols: list,
    horizons: list = None,
    metric: str = TARGET_COL,
) -> pd.DataFrame:
    """Evaluate all three models across multiple forecast horizons per facility.

    For each (facility, horizon) combination the models are retrained from
    scratch using data up to ``N - horizon`` months, then evaluated on the
    held-out tail.

    **Evaluation semantics per model:**

    * **XGBoost / BNN** – *oracle* evaluation: lag features in the test rows
      are derived from actual historical values (pre-computed during feature
      engineering), not from recursive predictions.  This represents an
      upper-bound on deployable multi-year accuracy and should be labelled
      accordingly in the paper.

    * **N-HiTS** – *true multi-step*: the model forecasts ``min(12, horizon)``
      steps ahead from the training cutoff in a single pass.  For horizons
      > 12 the evaluation covers only the first 12 months of the holdout.

    Args:
        df:           Full enhanced DataFrame (all facilities, all years).
        facilities:   List of facility IDs to evaluate.
        feature_cols: Feature column names used by XGBoost and BNN.
        horizons:     List of integer holdout sizes in months.
                      Defaults to ``[12, 24, 36]``.
        metric:       Target column name (default: ``EmissionstCO2``).

    Returns:
        Long-format DataFrame with columns:
            ``Facility``, ``Model``, ``Horizon_months``,
            ``MAPE``, ``RMSE``, ``N_test_steps``.
        Sorted by ``Facility``, ``Model``, ``Horizon_months``.
    """
    from src.models.xgboost_model import run_xgboost_quantile
    from src.models.bnn_model import run_bnn
    from src.models.nhits_model import run_nhits
    from config import MIN_TRAIN_ROWS

    if horizons is None:
        horizons = [12, 24, 36]

    rows = []
    for horizon in horizons:
        logger.info("evaluate_horizons: horizon=%d months", horizon)
        for fac in facilities:
            df_fac = df[df["Facility"] == fac].dropna(subset=[metric])
            if len(df_fac) - horizon < MIN_TRAIN_ROWS:
                logger.debug(
                    "evaluate_horizons: skipping %s horizon=%d (insufficient train rows).",
                    fac, horizon,
                )
                for model_name in ("N-HiTS", "XGBoost", "BNN"):
                    rows.append({
                        "Facility":       fac,
                        "Model":          model_name,
                        "Horizon_months": horizon,
                        "MAPE":           np.nan,
                        "RMSE":           np.nan,
                        "N_test_steps":   np.nan,
                    })
                continue

            # --- XGBoost ---
            xgb_res = run_xgboost_quantile(
                df, fac, feature_cols, metric=metric, test_months=horizon
            )
            rows.append({
                "Facility":       fac,
                "Model":          "XGBoost",
                "Horizon_months": horizon,
                "MAPE":           xgb_res.get("mape", np.nan),
                "RMSE":           xgb_res.get("rmse", np.nan),
                "N_test_steps":   horizon if not np.isnan(xgb_res.get("mape", np.nan)) else np.nan,
            })

            # --- BNN ---
            bnn_res = run_bnn(
                df, fac, feature_cols, metric=metric, test_months=horizon
            )
            rows.append({
                "Facility":       fac,
                "Model":          "BNN",
                "Horizon_months": horizon,
                "MAPE":           bnn_res.get("mape", np.nan),
                "RMSE":           bnn_res.get("rmse", np.nan),
                "N_test_steps":   horizon if not np.isnan(bnn_res.get("mape", np.nan)) else np.nan,
            })

            # --- N-HiTS (true multi-step; capped at 12 steps internally) ---
            nhits_res = run_nhits(
                df, fac, feature_cols=feature_cols, metric=metric, test_months=horizon
            )
            n_steps = (
                len(nhits_res["y_pred"])
                if "y_pred" in nhits_res and nhits_res["y_pred"] is not None
                else np.nan
            )
            rows.append({
                "Facility":       fac,
                "Model":          "N-HiTS",
                "Horizon_months": horizon,
                "MAPE":           nhits_res.get("mape", np.nan),
                "RMSE":           nhits_res.get("rmse", np.nan),
                "N_test_steps":   n_steps,
            })

    result = (
        pd.DataFrame(rows)
        .sort_values(["Facility", "Model", "Horizon_months"])
        .reset_index(drop=True)
    )
    logger.info(
        "evaluate_horizons complete: %d records (%d facilities × %d horizons × 3 models).",
        len(result), len(facilities), len(horizons),
    )
    return result


# ---------------------------------------------------------------------------
# Step-wise long-horizon evaluation (single-origin, per-step MAPE)
# ---------------------------------------------------------------------------

def evaluate_per_step_horizons(
    df_ext: pd.DataFrame,
    facilities: list,
    feature_cols: list,
    metric: str = TARGET_COL,
) -> tuple:
    """Step-wise horizon accuracy using a single Dec-2023 origin.

    Trains each model on Jan 2022 – Dec 2023 (the primary training window)
    and evaluates against all available future data:

    * **XGBoost** and **BNN**: ``test_months=20`` — covers Jan 2024 – Aug 2025
      (h=1 … 20) in one pass.  Per-facility, per-step absolute percentage
      errors (APEs) are extracted from ``y_test`` / ``y_pred`` arrays.

    * **N-HiTS**: capped internally at 12 steps per pass.  Two passes:

      - Dec-2023 origin → h=1 … 12 (Jan–Dec 2024)
      - Dec-2024 origin (train on 2022–2024) → h=1 … 8, re-labelled 13–20

      The second N-HiTS pass uses more training data, which is a known
      limitation; it is the standard multi-origin approach used in the
      literature for models with a fixed horizon cap.

    Args:
        df_ext:       Enhanced DataFrame covering Jan 2022 – Aug 2025
                      (output of ``create_global_features`` on the extended
                      dataset; must contain a ``Date`` column).
        facilities:   Facility IDs to evaluate.
        feature_cols: Feature columns for XGBoost / BNN.
        metric:       Target column name.

    Returns:
        Tuple ``(step_df, summary_df)`` where:

        * ``step_df``    – long-format DataFrame:
          ``Facility, Model, h, APE``
          (one row per facility × model × forecast step).

        * ``summary_df`` – aggregated DataFrame:
          ``Model, h, MAPE, Std_APE, N_facilities``
          grouped by model and step.
    """
    from src.models.xgboost_model import run_xgboost_quantile
    from src.models.bnn_model import run_bnn
    from src.models.nhits_model import run_nhits
    from config import MIN_TRAIN_ROWS

    # Number of steps ahead we want to evaluate
    H_XGB_BNN = 20   # Jan 2024 – Aug 2025 from Dec-2023 origin
    H_NHITS_1 = 12   # N-HiTS first pass  (h=1-12, 2024)
    H_NHITS_2 = 8    # N-HiTS second pass (h=13-20, 2025)

    rows = []  # Facility, Model, h, APE

    for fac in facilities:
        df_fac = df_ext[df_ext["Facility"] == fac].dropna(subset=[metric])
        n_total = len(df_fac)

        # ------------------------------------------------------------------ #
        # XGBoost — single pass, test_months = H_XGB_BNN                     #
        # ------------------------------------------------------------------ #
        if n_total - H_XGB_BNN >= MIN_TRAIN_ROWS:
            xgb_res = run_xgboost_quantile(
                df_ext, fac, feature_cols, metric=metric, test_months=H_XGB_BNN
            )
            if "y_test" in xgb_res and xgb_res["y_test"] is not None:
                y_t = np.asarray(xgb_res["y_test"])
                y_p = np.asarray(xgb_res["preds"][0.5])
                for h_idx in range(min(len(y_t), len(y_p))):
                    denom = max(abs(y_t[h_idx]), 1e-6)
                    rows.append({
                        "Facility": fac,
                        "Model":    "XGBoost",
                        "h":        h_idx + 1,
                        "APE":      abs(y_t[h_idx] - y_p[h_idx]) / denom * 100.0,
                    })
            logger.debug("evaluate_per_step_horizons: XGBoost %s done.", fac)
        else:
            logger.debug("evaluate_per_step_horizons: skip XGBoost %s (too few rows).", fac)

        # ------------------------------------------------------------------ #
        # BNN — single pass, test_months = H_XGB_BNN                         #
        # ------------------------------------------------------------------ #
        if n_total - H_XGB_BNN >= MIN_TRAIN_ROWS:
            bnn_res = run_bnn(
                df_ext, fac, feature_cols, metric=metric, test_months=H_XGB_BNN
            )
            if "y_val" in bnn_res and bnn_res["y_val"] is not None:
                y_t = np.asarray(bnn_res["y_val"])
                y_p = np.asarray(bnn_res["predictions_mean"])
                for h_idx in range(min(len(y_t), len(y_p))):
                    denom = max(abs(y_t[h_idx]), 1e-6)
                    rows.append({
                        "Facility": fac,
                        "Model":    "BNN",
                        "h":        h_idx + 1,
                        "APE":      abs(y_t[h_idx] - y_p[h_idx]) / denom * 100.0,
                    })
            logger.debug("evaluate_per_step_horizons: BNN %s done.", fac)
        else:
            logger.debug("evaluate_per_step_horizons: skip BNN %s (too few rows).", fac)

        # ------------------------------------------------------------------ #
        # N-HiTS pass 1 — Dec-2023 origin, h=1-12                            #
        # (use df filtered to end of 2024 so training data is the same)       #
        # ------------------------------------------------------------------ #
        df_ext_2024 = df_ext[pd.to_datetime(df_ext["Date"]).dt.year <= 2024]
        df_fac_2024 = df_ext_2024[df_ext_2024["Facility"] == fac].dropna(subset=[metric])
        n_2024 = len(df_fac_2024)
        if n_2024 - H_NHITS_1 >= MIN_TRAIN_ROWS:
            nh1_res = run_nhits(
                df_ext_2024, fac, feature_cols=feature_cols,
                metric=metric, test_months=H_NHITS_1
            )
            if "y_true" in nh1_res and nh1_res["y_true"] is not None:
                y_t = np.asarray(nh1_res["y_true"])
                y_p = np.asarray(nh1_res["y_pred"])
                for h_idx in range(min(len(y_t), len(y_p))):
                    denom = max(abs(y_t[h_idx]), 1e-6)
                    rows.append({
                        "Facility": fac,
                        "Model":    "N-HiTS",
                        "h":        h_idx + 1,
                        "APE":      abs(y_t[h_idx] - y_p[h_idx]) / denom * 100.0,
                    })
            logger.debug("evaluate_per_step_horizons: N-HiTS pass1 %s done.", fac)

        # ------------------------------------------------------------------ #
        # N-HiTS pass 2 — Dec-2024 origin, h=1-8 → re-labelled 13-20        #
        # ------------------------------------------------------------------ #
        if n_total - H_NHITS_2 >= MIN_TRAIN_ROWS:
            nh2_res = run_nhits(
                df_ext, fac, feature_cols=feature_cols,
                metric=metric, test_months=H_NHITS_2
            )
            if "y_true" in nh2_res and nh2_res["y_true"] is not None:
                y_t = np.asarray(nh2_res["y_true"])
                y_p = np.asarray(nh2_res["y_pred"])
                for h_idx in range(min(len(y_t), len(y_p))):
                    denom = max(abs(y_t[h_idx]), 1e-6)
                    rows.append({
                        "Facility": fac,
                        "Model":    "N-HiTS",
                        "h":        h_idx + 13,   # second-pass offset
                        "APE":      abs(y_t[h_idx] - y_p[h_idx]) / denom * 100.0,
                    })
            logger.debug("evaluate_per_step_horizons: N-HiTS pass2 %s done.", fac)

    step_df = pd.DataFrame(rows)

    if step_df.empty:
        logger.warning("evaluate_per_step_horizons: no results produced.")
        return step_df, pd.DataFrame()

    summary_df = (
        step_df.groupby(["Model", "h"])["APE"]
        .agg(MAPE="mean", Std_APE="std", N_facilities="count")
        .reset_index()
        .rename(columns={"h": "h"})
        .sort_values(["Model", "h"])
        .reset_index(drop=True)
    )

    logger.info(
        "evaluate_per_step_horizons complete: %d step-records across %d facilities.",
        len(step_df), len(facilities),
    )
    return step_df, summary_df


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss: str = "mse",
) -> dict:
    """Diebold-Mariano (1995) test with Harvey-Leybourne-Newbold (1997) correction.

    Tests H₀: equal predictive accuracy between two forecasting models.
    A negative DM statistic means model 1 is *more accurate* than model 2.

    Args:
        e1:   Forecast errors for model 1 (array of shape ``(T,)``).
              These are raw errors ``y_true - y_pred``, not absolute values.
        e2:   Forecast errors for model 2, same length as ``e1``.
        h:    Forecast horizon (number of steps ahead). Used for the HAC
              variance estimator and the HLN correction factor.
              Use ``h=1`` when pooling per-step errors from a 1-step-ahead
              evaluation; use the actual horizon otherwise.
        loss: Loss function for the loss differential.
              ``'mse'`` (default) uses squared errors;
              ``'mae'`` uses absolute errors.

    Returns:
        Dict with keys:

        * ``dm_stat``  – HLN-corrected DM test statistic (t-distributed under H₀)
        * ``p_value``  – two-sided p-value (t_{T-1} distribution)
        * ``T``        – number of observations
        * ``d_mean``   – mean loss differential
        * ``reject``   – ``True`` if H₀ rejected at 5% significance level
        * ``better``   – ``'model1'`` / ``'model2'`` / ``'neither'``
    """
    from scipy import stats

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    if len(e1) != len(e2):
        raise ValueError(
            f"e1 and e2 must have the same length ({len(e1)} vs {len(e2)})."
        )

    if loss == "mse":
        L1, L2 = e1 ** 2, e2 ** 2
    elif loss == "mae":
        L1, L2 = np.abs(e1), np.abs(e2)
    else:
        raise ValueError(f"loss must be 'mse' or 'mae', got '{loss}'.")

    d = L1 - L2          # loss differential: positive = model1 worse
    T = len(d)
    d_bar = d.mean()

    # HAC variance: sum of autocovariances up to lag h-1
    gamma0 = np.var(d, ddof=0)
    gamma_sum = gamma0
    for k in range(1, h):
        gk = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2.0 * gk

    var_d_hat = gamma_sum / T
    if var_d_hat <= 0:
        var_d_hat = gamma0 / T

    dm_raw = d_bar / np.sqrt(var_d_hat)

    # Harvey-Leybourne-Newbold small-sample correction
    hln_factor = np.sqrt(
        (T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T
    )
    dm_stat = hln_factor * dm_raw

    p_value = 2.0 * stats.t.sf(abs(dm_stat), df=T - 1)

    reject = bool(p_value < 0.05)
    if not reject:
        better = "neither"
    elif dm_stat < 0:
        better = "model1"   # model1 lower loss → more accurate
    else:
        better = "model2"

    return {
        "dm_stat":  float(dm_stat),
        "p_value":  float(p_value),
        "T":        T,
        "d_mean":   float(d_bar),
        "reject":   reject,
        "better":   better,
    }


def run_diebold_mariano_all_pairs(
    step_df: pd.DataFrame,
    h: int = 1,
    loss: str = "mse",
) -> pd.DataFrame:
    """Run pairwise DM tests across all three model pairs.

    Uses the pooled per-facility, per-step errors from
    ``evaluate_per_step_horizons``.  Each model's error is computed as
    ``y_true - y_pred``; APE values stored in ``step_df`` are converted
    back to signed errors using their absolute values (conservative
    assumption: treats APE as |error| / |y_true|, so signed error cannot
    be recovered — MSE-based DM uses APE² instead).

    Args:
        step_df: Long-format DataFrame from ``evaluate_per_step_horizons``
                 with columns ``Model, h, APE``.
        h:       Forecast horizon for HAC estimator.
        loss:    ``'mse'`` or ``'mae'``.

    Returns:
        DataFrame with one row per model pair:
        ``Model_1, Model_2, DM_stat, p_value, T, Reject_H0, Better_model``.
    """
    models = sorted(step_df["Model"].unique())
    pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]

    results = []
    for m1, m2 in pairs:
        # Align on (Facility, h) so errors are paired
        df1 = step_df[step_df["Model"] == m1][["Facility", "h", "APE"]].copy()
        df2 = step_df[step_df["Model"] == m2][["Facility", "h", "APE"]].copy()
        merged = df1.merge(df2, on=["Facility", "h"], suffixes=("_1", "_2")).dropna()

        if len(merged) < 10:
            logger.warning("DM test %s vs %s: only %d paired obs — skip.", m1, m2, len(merged))
            continue

        # Use APE as the loss value directly (non-negative, so treat as MAE-type loss)
        ape1 = merged["APE_1"].values
        ape2 = merged["APE_2"].values
        # For DM with APE, treat e_i = APE_i (loss = APE), so d = APE1 - APE2
        # This is equivalent to 'mae' loss on percentage errors
        dm = diebold_mariano_test(ape1, -ape2 + ape1, h=h, loss="mae")
        # Simpler: directly use the APE difference
        d = ape1 - ape2
        T = len(d)
        d_bar = d.mean()
        gamma0 = np.var(d, ddof=0)
        gamma_sum = gamma0
        for k in range(1, h):
            gk = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
            gamma_sum += 2.0 * gk
        var_d_hat = max(gamma_sum / T, gamma0 / T)
        dm_raw = d_bar / np.sqrt(var_d_hat)
        from scipy import stats
        hln = np.sqrt((T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T)
        dm_stat = hln * dm_raw
        p_val = 2.0 * stats.t.sf(abs(dm_stat), df=T - 1)
        reject = p_val < 0.05
        if not reject:
            better = "neither"
        elif dm_stat < 0:
            better = m1
        else:
            better = m2

        results.append({
            "Model_1":      m1,
            "Model_2":      m2,
            "DM_stat":      round(float(dm_stat), 4),
            "p_value":      round(float(p_val), 4),
            "T":            T,
            "Reject_H0":    reject,
            "Better_model": better,
        })

    return pd.DataFrame(results)
