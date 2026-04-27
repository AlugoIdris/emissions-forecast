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
