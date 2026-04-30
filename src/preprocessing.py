"""
preprocessing.py
----------------
Data loading, cleaning, and feature engineering for the industrial
emissions forecasting pipeline.

Authors : Idris Alugo
Paper   : "Long-Horizon Emissions Forecasting for 2030 Target Assessment:
           A Comparative Study of N-HiTS, XGBoost, and Bayesian Models
           in Fast-Moving Consumer Goods Supply Chains" – Applied Energy (2026)
License : MIT
"""

import logging
import warnings
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression

from config import TARGET_COL, RANDOM_SEED, OUTLIER_FACILITIES

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – override via config.py or pass as arguments
# ---------------------------------------------------------------------------

LAG_PERIODS   = [1, 3, 6, 12]
DATE_MIN_YEAR = 2016          # inclusive (removes 2015 baseline)
DATE_MAX_YEAR = 2024          # inclusive (test period = 2024, train = 2022–2023)


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw ESG data from CSV, parse dates, and sort.

    Args:
        filepath: Path to the raw CSV file (e.g., ``data/esgdata.csv``).

    Returns:
        Cleaned DataFrame with ``Date`` as ``datetime64``, sorted by
        ``Facility`` then ``Date``.
    """
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Facility", "Date"]).reset_index(drop=True)
    logger.info("Loaded %d rows from %s", len(df), filepath)
    return df


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------

def fill_missing_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Forward/back-fill ``Production`` and ``Renewable_percent`` per facility.

    Args:
        df: Raw DataFrame containing at least ``Facility``, ``Production``,
            and ``Renewable_percent`` columns.

    Returns:
        DataFrame with missing covariate values imputed.
    """
    df = df.copy()
    df[["Production", "Renewable_percent"]] = (
        df.groupby("Facility")[["Production", "Renewable_percent"]]
        .transform(lambda g: g.ffill().bfill())
    )
    remaining = df[["Production", "Renewable_percent"]].isna().sum()
    if remaining.any():
        logger.warning("Remaining NaNs after fill:\n%s", remaining)
    return df


def remove_outlier_facilities(
    df: pd.DataFrame,
    outliers: Optional[list] = None,
) -> pd.DataFrame:
    """Drop facilities identified as outliers.

    Args:
        df:       Input DataFrame with a ``Facility`` column.
        outliers: List of facility IDs to drop (case-insensitive).
                  Defaults to the module-level ``OUTLIER_FACILITIES`` list.

    Returns:
        Filtered DataFrame.
    """
    if outliers is None:
        outliers = OUTLIER_FACILITIES

    outliers_upper = {s.upper().strip() for s in outliers}
    before = len(df)
    df = df[~df["Facility"].astype(str).str.upper().isin(outliers_upper)].reset_index(drop=True)
    logger.info("Removed %d rows (outlier facilities). Remaining: %d", before - len(df), len(df))
    return df


def filter_date_range(
    df: pd.DataFrame,
    year_min: int = DATE_MIN_YEAR,
    year_max: int = DATE_MAX_YEAR,
) -> pd.DataFrame:
    """Keep only rows within [year_min, year_max].

    Args:
        df:       DataFrame with a parsed ``Date`` column.
        year_min: First year to keep (removes baseline year 2015).
        year_max: Last year to keep.

    Returns:
        Filtered DataFrame.
    """
    mask = (df["Date"].dt.year >= year_min) & (df["Date"].dt.year <= year_max)
    df = df[mask].reset_index(drop=True)
    logger.info("Date filter [%d–%d]: %d rows remain", year_min, year_max, len(df))
    return df


# ---------------------------------------------------------------------------
# 3. Basic Feature Engineering
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``Month``, ``Year``, and ``DayOfYear`` from ``Date``.

    Args:
        df: DataFrame with a parsed ``Date`` column.

    Returns:
        DataFrame with three new temporal columns.
    """
    df = df.copy()
    df["Month"]     = df["Date"].dt.month
    df["Year"]      = df["Date"].dt.year
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


def add_region_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the ``Region`` column (prefix ``Region_``).

    Normalises region labels before encoding so that values like
    ``"Region 1"`` produce clean column names (``Region_1``) with no
    embedded spaces, which is required by XGBoost's DMatrix.

    Args:
        df: DataFrame with a ``Region`` column.

    Returns:
        DataFrame with original columns plus region dummy columns.
    """
    # Strip "Region " prefix if present so "Region 1" -> "1", "R1" -> "R1"
    region_codes = df["Region"].str.replace(r"^Region\s+", "", regex=True)
    dummies = pd.get_dummies(region_codes, prefix="Region")
    df = pd.concat([df, dummies], axis=1)
    logger.info("Region dummies added: %s", list(dummies.columns))
    return df


# ---------------------------------------------------------------------------
# 4. Global Hierarchical Feature Engineering
# ---------------------------------------------------------------------------

def compute_regional_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Fit a linear trend (Emissions ~ Year) per region and add as a feature.

    Args:
        df: DataFrame containing ``Region``, ``Year``, and target column.

    Returns:
        DataFrame with a new ``RegionalYearTrend`` column (slope of the
        linear regression, mapped to each row by its region).
    """
    trends = {}
    for region in df["Region"].unique():
        region_data = df[df["Region"] == region]
        X = region_data["Year"].values.reshape(-1, 1)
        y = region_data[TARGET_COL].values
        model = LinearRegression().fit(X, y)
        trends[region] = model.coef_[0]

    df = df.copy()
    df["RegionalYearTrend"] = df["Region"].map(trends)
    return df


def create_global_features(
    df: pd.DataFrame,
    features_base: list[str],
) -> tuple[pd.DataFrame, dict, list[str]]:
    """Apply global scaling and build hierarchical features.

    Executes seven sequential steps:

    1. **Global RobustScaler** on all base features (fit on entire dataset).
    2. **Lag features** for emissions and production (lags 1, 3, 6, 12).
    3. **Facility-level** mean & std emission statistics.
    4. **Regional** mean emission and facility deviation from it.
    5. **Interaction features** (production × regional mean, energy × policy).
    6. **Rolling regional** emissions (6-month and 12-month windows).
    7. **Regional temporal trend** via ``compute_regional_trend``.

    NaN values created by lags and rolling windows are forward-filled,
    back-filled, then zero-filled.

    Args:
        df:            DataFrame after cleaning and basic feature engineering.
        features_base: List of base feature column names to scale (must not
                       include lag/hierarchy/rolling columns).

    Returns:
        Tuple of:
            - ``df_enhanced``: DataFrame with all new features.
            - ``global_scalers``: Dict with keys ``"X"`` (RobustScaler for
              features) and ``"y"`` (RobustScaler for target).
            - ``enhanced_features``: List of all feature column names to pass
              to models.
    """
    df_enhanced = df.copy()

    # ------------------------------------------------------------------
    # STEP 1: Global scaling
    # ------------------------------------------------------------------
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X_all = df_enhanced[features_base].values
    y_all = df_enhanced[TARGET_COL].values.reshape(-1, 1)

    X_scaled = scaler_X.fit_transform(X_all)
    scaler_y.fit_transform(y_all)          # fit only; y stays in original scale

    for i, col in enumerate(features_base):
        df_enhanced[col] = X_scaled[:, i]

    logger.info("NOTE: Target %s kept in ORIGINAL scale (y-scaler fitted but not applied).", TARGET_COL)

    # ------------------------------------------------------------------
    # STEP 2: Lag features
    # ------------------------------------------------------------------
    df_enhanced = df_enhanced.sort_values(["Facility", "Date"]).reset_index(drop=True)
    lag_features = []

    for lag in LAG_PERIODS:
        col_e = f"EmissionsLag{lag}"
        col_p = f"ProductionLag{lag}"
        df_enhanced[col_e] = df_enhanced.groupby("Facility")[TARGET_COL].shift(lag)
        df_enhanced[col_p] = df_enhanced.groupby("Facility")["Production"].shift(lag)
        lag_features += [col_e, col_p]

    # ------------------------------------------------------------------
    # STEP 3: Facility-level hierarchy
    # ------------------------------------------------------------------
    hierarchy_features = []

    df_enhanced["FacilityMeanEmission"] = (
        df_enhanced.groupby("Facility")[TARGET_COL].transform("mean")
    )
    df_enhanced["FacilityStdEmission"] = (
        df_enhanced.groupby("Facility")[TARGET_COL].transform("std").fillna(0)
    )
    hierarchy_features += ["FacilityMeanEmission", "FacilityStdEmission"]

    # ------------------------------------------------------------------
    # STEP 4: Regional hierarchy
    # ------------------------------------------------------------------
    df_enhanced["RegionalMeanEmission"] = (
        df_enhanced.groupby("Region")[TARGET_COL].transform("mean")
    )
    df_enhanced["FacilityDeviationFromRegion"] = (
        df_enhanced["FacilityMeanEmission"] - df_enhanced["RegionalMeanEmission"]
    )
    hierarchy_features += ["RegionalMeanEmission", "FacilityDeviationFromRegion"]

    # ------------------------------------------------------------------
    # STEP 5: Interaction features
    # ------------------------------------------------------------------
    interaction_features = []

    df_enhanced["ProductionRegionInteraction"] = (
        df_enhanced["Production"]
        / df_enhanced.groupby("Region")["Production"].transform("mean")
    )
    df_enhanced["EnergyPolicyWeightInteraction"] = (
        df_enhanced["Energy_MWh"] * df_enhanced["PolicyWeight"]
    )
    interaction_features += ["ProductionRegionInteraction", "EnergyPolicyWeightInteraction"]

    # ------------------------------------------------------------------
    # STEP 6: Rolling regional metrics
    # ------------------------------------------------------------------
    rolling_features = []

    df_enhanced = df_enhanced.sort_values("Date")
    for window, col in [(6, "RegionRollingEmissions6M"), (12, "RegionRollingEmissions12M")]:
        df_enhanced[col] = (
            df_enhanced.groupby("Region")[TARGET_COL]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        rolling_features.append(col)

    # ------------------------------------------------------------------
    # STEP 7: Regional temporal trend
    # ------------------------------------------------------------------
    df_enhanced["RegionalYearTrend"] = (
        df_enhanced.groupby("Region")["Year"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    )
    df_enhanced = compute_regional_trend(df_enhanced)
    rolling_features.append("RegionalYearTrend")

    # ------------------------------------------------------------------
    # Combine & fill NaNs
    # ------------------------------------------------------------------
    enhanced_features = (
        features_base
        + lag_features
        + hierarchy_features
        + interaction_features
        + rolling_features
    )

    for col in enhanced_features:
        if col in df_enhanced.columns:
            df_enhanced[col] = df_enhanced[col].ffill().bfill().fillna(0)

    global_scalers = {"X": scaler_X, "y": scaler_y}

    logger.info(
        "Feature engineering complete. Base: %d | Lags: %d | Hierarchy: %d | "
        "Interactions: %d | Rolling/Trend: %d | Total: %d",
        len(features_base),
        len(lag_features),
        len(hierarchy_features),
        len(interaction_features),
        len(rolling_features),
        len(enhanced_features),
    )

    return df_enhanced, global_scalers, enhanced_features


# ---------------------------------------------------------------------------
# 5. 2030 Feature Preparation (inference-time helper)
# ---------------------------------------------------------------------------

def prepare_2030_features(
    df: pd.DataFrame,
    facility: str,
    feature_cols: list[str],
    global_scalers: dict,
    features_base: list[str],
) -> pd.DataFrame:
    """Build a single-row feature vector for 2030 inference.

    Takes the last observed row for ``facility`` as a template, sets
    temporal features to represent December 2030, and re-applies the
    global X-scaler only to the base (non-derived) features.

    Rolling and lag features are **not** re-scaled because they are
    already in the scaled space (computed after global scaling).

    Args:
        df:             The enhanced DataFrame (post ``create_global_features``).
        facility:       Facility ID string.
        feature_cols:   Full list of feature columns expected by models.
        global_scalers: Dict returned by ``create_global_features``.
        features_base:  The same base feature list used during training.

    Returns:
        Single-row DataFrame with columns matching ``feature_cols``, or an
        empty DataFrame if the facility has no data.
    """
    df_fac = df[df["Facility"] == facility].copy()
    if len(df_fac) == 0:
        logger.warning("No data found for facility %s.", facility)
        return pd.DataFrame()

    last_row = df_fac.iloc[-1].copy()

    # Update temporal features to represent 2030-12-31
    last_row["Year"]      = 2030
    last_row["Month"]     = 12
    last_row["DayOfYear"] = 365

    # Re-apply scaler to base features only
    features_to_scale = [f for f in features_base if f in last_row.index]
    scaler_X = global_scalers.get("X")

    if scaler_X is not None and len(features_to_scale) == len(features_base):
        values = last_row[features_to_scale].values.reshape(1, -1)
        scaled = scaler_X.transform(values)
        for i, col in enumerate(features_to_scale):
            last_row[col] = scaled[0, i]
        logger.debug("Scaled %d base features for facility %s (2030 inference).", len(features_to_scale), facility)
    else:
        logger.warning(
            "Scaling skipped for %s: expected %d features, found %d.",
            facility, len(features_base), len(features_to_scale),
        )

    available = [f for f in feature_cols if f in last_row.index]
    return last_row[available].to_frame().T


# ---------------------------------------------------------------------------
# 6. Full Pipeline Entry-Point
# ---------------------------------------------------------------------------

def build_dataset(
    filepath: str,
    features_base_numeric: list[str],
    outliers: Optional[list] = None,
    year_min: int = DATE_MIN_YEAR,
    year_max: int = DATE_MAX_YEAR,
) -> tuple:
    """Run the complete preprocessing pipeline.

    Convenience wrapper that chains all steps:
    load → fill → remove outliers → date filter → time features →
    region dummies → global hierarchical features.

    Args:
        filepath:              Path to the raw CSV file.
        features_base_numeric: List of numeric base feature names
                               (excluding region dummies).
        outliers:              Facilities to exclude (default: module constant).
        year_min:              Earliest year to keep.
        year_max:              Latest year to keep.

    Returns:
        Tuple of:
            - ``df``               : Enhanced DataFrame ready for model training.
            - ``global_scalers``   : Scaler dict for inference.
            - ``enhanced_features``: Full feature list for models.
            - ``region_cols``      : Region dummy column names (for Prophet regressors).
    """
    df = load_data(filepath)
    df = fill_missing_covariates(df)
    df = remove_outlier_facilities(df, outliers=outliers)
    df = filter_date_range(df, year_min=year_min, year_max=year_max)
    df = add_time_features(df)
    df = add_region_dummies(df)

    region_cols = [c for c in df.columns if c.startswith("Region_")]
    features_base = features_base_numeric + region_cols

    df, global_scalers, enhanced_features = create_global_features(df, features_base)

    missing = [f for f in enhanced_features if f not in df.columns]
    if missing:
        raise ValueError(f"Feature verification failed. Missing columns: {missing}")

    logger.info("Pipeline complete. Dataset shape: %s | Features: %d", df.shape, len(enhanced_features))
    return df, global_scalers, enhanced_features, region_cols


# ---------------------------------------------------------------------------
# Convenience Wrappers for Notebook Compatibility
# ---------------------------------------------------------------------------

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load and clean raw data for the pipeline.
    
    This is a convenience wrapper that performs initial data loading,
    missing value imputation, and adds time features and region dummies.
    
    Args:
        filepath: Path to the raw CSV file.
        
    Returns:
        Cleaned DataFrame with time features and region dummies added.
    """
    df = load_data(filepath)
    df = fill_missing_covariates(df)
    df = remove_outlier_facilities(df)
    df = filter_date_range(df, year_min=DATE_MIN_YEAR, year_max=DATE_MAX_YEAR)
    df = add_time_features(df)
    df = add_region_dummies(df)
    return df


def create_global_features_with_hierarchy(
    df: pd.DataFrame,
    features_base: list[str],
) -> tuple:
    """Create global features with hierarchical structure.
    
    This is a convenience wrapper for create_global_features that matches
    the expected interface in the notebook.
    
    Args:
        df: DataFrame after initial cleaning.
        features_base: List of base feature names (including region dummies).
        
    Returns:
        Tuple of (enhanced_df, global_scalers, enhanced_features).
    """
    return create_global_features(df, features_base)


# ---------------------------------------------------------------------------
# Synthetic sample generator
# ---------------------------------------------------------------------------

def generate_synthetic_sample(
    n_facilities: int = 20,
    n_months: int = 84,
    output_path: str = "data/esgdata_sample.csv",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a fully synthetic ESG dataset for pipeline testing."""
    import os
    rng = np.random.default_rng(seed)

    regions        = ["R1", "R2", "R3", "R4"]
    policy_weights = {"R1": 0.8, "R2": 0.6, "R3": 0.4, "R4": 0.9}
    dates          = pd.date_range("2018-01-01", periods=n_months, freq="MS")

    records = []
    for i in range(1, n_facilities + 1):
        fac    = f"F{i:02d}"
        region = regions[i % len(regions)]
        base_emission   = rng.uniform(40, 120)
        base_production = rng.uniform(500, 2000)
        base_energy     = rng.uniform(200, 800)
        trend           = rng.uniform(-0.3, 0.1)

        for t, date in enumerate(dates):
            seasonal = 5 * np.sin(2 * np.pi * (date.month - 1) / 12)
            emission = max(base_emission + trend * t + seasonal + rng.normal(0, 3), 5.0)
            records.append({
                "Date":             date.strftime("%Y-%m-%d"),
                "Facility":         fac,
                "Region":           region,
                "Emissions_tCO2":   round(emission, 2),
                "Production":       round(base_production * (1 + rng.normal(0, 0.05)), 1),
                "Energy_MWh":       round(base_energy     * (1 + rng.normal(0, 0.05)), 1),
                "Waste_Kg":         round(rng.uniform(100, 500), 1),
                "Renewable_percent": round(float(np.clip(rng.normal(30, 10), 0, 100)), 1),
                "PolicyWeight":     policy_weights[region],
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Synthetic sample → %s  (%d rows)", output_path, len(df))
    return df


# ---------------------------------------------------------------------------
# Facility data-quality report
# ---------------------------------------------------------------------------

def compute_facility_quality_report(
    df: pd.DataFrame,
    metric: str = TARGET_COL,
) -> pd.DataFrame:
    """Compute per-facility data completeness and quality statistics.

    Useful for verifying that the 20 facilities retained in the pipeline
    meet minimum quality standards and for communicating data provenance
    to reviewers.

    Quality tiers are based on completeness (actual rows / expected monthly
    rows derived from each facility's date range):
        - ``Good``  : >= 90 % complete
        - ``Fair``  : >= 70 % complete
        - ``Poor``  : < 70 % complete

    Args:
        df:     DataFrame containing at least ``Facility``, ``Date``,
                and the ``metric`` column.  ``Date`` must be parseable
                by ``pd.to_datetime``.
        metric: Target column to assess for missingness (default:
                ``EmissionstCO2``).

    Returns:
        DataFrame (one row per facility) with columns:
            ``Facility``, ``N_rows``, ``Date_min``, ``Date_max``,
            ``Expected_months``, ``Completeness_pct``,
            ``Target_missing_pct``, ``N_target_valid``,
            ``Quality_tier``.
        Sorted by ``Completeness_pct`` descending.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    records = []
    for fac, grp in df.groupby("Facility"):
        grp = grp.sort_values("Date")
        date_min = grp["Date"].min()
        date_max = grp["Date"].max()

        # Expected number of monthly observations in the date range
        total_months = (
            (date_max.year - date_min.year) * 12
            + (date_max.month - date_min.month)
            + 1
        )

        n_rows    = len(grp)
        comp_pct  = round(n_rows / total_months * 100, 1) if total_months > 0 else 0.0
        n_valid   = int(grp[metric].notna().sum()) if metric in grp.columns else 0
        miss_pct  = round((1 - n_valid / n_rows) * 100, 1) if n_rows > 0 else 100.0

        if comp_pct >= 90:
            tier = "Good"
        elif comp_pct >= 70:
            tier = "Fair"
        else:
            tier = "Poor"

        records.append({
            "Facility":            fac,
            "N_rows":              n_rows,
            "Date_min":            date_min.strftime("%Y-%m"),
            "Date_max":            date_max.strftime("%Y-%m"),
            "Expected_months":     total_months,
            "Completeness_pct":    comp_pct,
            "Target_missing_pct":  miss_pct,
            "N_target_valid":      n_valid,
            "Quality_tier":        tier,
        })

    report = (
        pd.DataFrame(records)
        .sort_values("Completeness_pct", ascending=False)
        .reset_index(drop=True)
    )

    tier_counts = report["Quality_tier"].value_counts().to_dict()
    logger.info(
        "Facility quality report: %d facilities | Good=%d  Fair=%d  Poor=%d",
        len(report),
        tier_counts.get("Good", 0),
        tier_counts.get("Fair", 0),
        tier_counts.get("Poor", 0),
    )
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Emissions Forecasting — Preprocessing & Sample Generation"
    )
    subparsers = parser.add_subparsers(dest="command")

    gen = subparsers.add_parser("generate-sample",
                                help="Generate synthetic ESG dataset.")
    gen.add_argument("--n-facilities", type=int, default=20)
    gen.add_argument("--n-months",     type=int, default=84)
    gen.add_argument("--output",       type=str, default="data/esgdata_sample.csv")
    gen.add_argument("--seed",         type=int, default=42)

    args = parser.parse_args()

    if args.command == "generate-sample":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
        df = generate_synthetic_sample(
            n_facilities=args.n_facilities,
            n_months=args.n_months,
            output_path=args.output,
            seed=args.seed,
        )
        print(f"\nRows: {len(df)} | Facilities: {df['Facility'].nunique()}")
        print(f"Date range: {df['Date'].min()} → {df['Date'].max()}")
        print(f"Saved to: {args.output}")
    else:
        parser.print_help()