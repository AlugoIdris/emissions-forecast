"""
compute_ensemble_mape.py
------------------------
Standalone script to compute per-facility ensemble (stacking meta-learner)
MAPE on the 12-month held-out test set and write EnsembleMAPE /
EnsembleRMSE back into results/model_accuracy.csv.

Run once after the full pipeline has been executed.  Does NOT require any
previously-saved model weights — it re-trains all base models and the
meta-learner from scratch using the same config as the notebook pipeline.

Skips SHAP analysis, 2030 forecasting, risk assessment, and visualisation
so that it runs roughly 2x faster than the full notebook.

Usage
-----
    .venv\\Scripts\\python.exe compute_ensemble_mape.py

Progress is printed to stdout; EnsembleMAPE per facility is printed as it
is computed so you can monitor the run.
"""

import sys
import os
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from config import (
    DATA_PATH, TARGET_COL, TEST_MONTHS, RANDOM_SEED,
    MIN_TRAIN_ROWS,
)
from src.preprocessing import (
    load_and_clean_data,
    create_global_features_with_hierarchy,
)
from src.models.nhits_model   import run_nhits
from src.models.xgboost_model import run_xgboost_quantile
from src.models.bnn_model     import run_bnn, mc_predict_bnn
from src.models.meta_learner  import (
    build_meta_features,
    train_meta_learner,
    predict_ensemble,
    evaluate_meta_learner,
)
from src.metrics import safe_mape

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
import logging

logging.basicConfig(
    level=logging.WARNING,                    # suppress verbose model logs
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1 – Data loading & feature engineering
# ---------------------------------------------------------------------------
print("=" * 65)
print("  Ensemble MAPE computation")
print("=" * 65)

print("\n[1/4] Loading and engineering features …")
t0 = time.time()

BASE_NUMERIC_FEATURES = [
    "Month", "Year", "DayOfYear",
    "Production", "Energy_MWh", "Waste_Kg",
    "Renewable_percent", "PolicyWeight",
]

df_raw = load_and_clean_data(DATA_PATH)
region_cols   = [c for c in df_raw.columns if c.startswith("Region_")]
base_features = BASE_NUMERIC_FEATURES + region_cols

df, global_scalers, features = create_global_features_with_hierarchy(
    df_raw, base_features
)

facilities = sorted(df["Facility"].unique())
print(f"    {len(facilities)} facilities  |  {len(features)} features  "
      f"({time.time() - t0:.1f}s)")

# ---------------------------------------------------------------------------
# Step 2 – Per-facility training: OOF (for meta-learner) + final models
# ---------------------------------------------------------------------------
print(f"\n[2/4] Training base models ({len(facilities)} facilities) …")
print("      This step takes ~20–30 min.  Progress per facility:\n")

tscv = TimeSeriesSplit(n_splits=3)

results             = []
meta_feature_records = []
bnn_res_dict        = {}
xgb_res_dict        = {}

for fac_idx, fac in enumerate(facilities):
    t_fac = time.time()
    df_fac = (
        df[df["Facility"] == fac]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(df_fac) < MIN_TRAIN_ROWS or df_fac[TARGET_COL].nunique() < 2:
        print(f"  [{fac_idx+1:2d}/{len(facilities)}] {fac:<20}  SKIPPED (insufficient data)")
        continue

    nhits_oof = np.full(len(df_fac), np.nan)
    xgb_oof   = np.full(len(df_fac), np.nan)
    bnn_oof   = np.full(len(df_fac), np.nan)

    # ── OOF cross-validation (for meta-learner training) ─────────────────
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df_fac)):
        df_tr  = df_fac.iloc[train_idx]
        df_val = df_fac.iloc[val_idx]
        n_val  = len(df_val)
        df_combined = pd.concat([df_tr, df_val], ignore_index=True)

        # N-HiTS OOF
        try:
            res = run_nhits(
                df_combined, fac, features, TARGET_COL, test_months=n_val,
            )
            if res and res.get("model") is not None:
                for i, idx in enumerate(val_idx):
                    if i < len(res.get("y_pred", [])):
                        nhits_oof[idx] = res["y_pred"][i]
        except Exception:
            pass

        # XGBoost OOF
        try:
            res = run_xgboost_quantile(
                df_combined, fac, features, TARGET_COL, test_months=n_val,
            )
            if res and res.get("models") and 0.5 in res["models"]:
                preds = res["models"][0.5].predict(df_val[features].values)
                for i, idx in enumerate(val_idx):
                    if i < len(preds):
                        xgb_oof[idx] = preds[i]
        except Exception:
            pass

        # BNN OOF (fewer epochs — OOF is for weight-learning, not final eval)
        try:
            res = run_bnn(
                df_combined, fac, features, TARGET_COL,
                test_months=n_val, n_epochs=80,
            )
            if res and res.get("model") is not None:
                mc   = mc_predict_bnn(res["model"], df_val[features].values)
                preds = mc.mean(axis=0)
                for i, idx in enumerate(val_idx):
                    if i < len(preds):
                        bnn_oof[idx] = preds[i]
        except Exception:
            pass

    # ── Collect OOF meta-feature rows ─────────────────────────────────────
    for i in range(len(df_fac)):
        has_n = not np.isnan(nhits_oof[i])
        has_x = not np.isnan(xgb_oof[i])
        has_b = not np.isnan(bnn_oof[i])
        if has_n or has_x or has_b:
            meta_feature_records.append({
                "Facility": fac,
                "NHITS_OOF": nhits_oof[i] if has_n else 0.0,
                "XGB_OOF":   xgb_oof[i]   if has_x else 0.0,
                "BNN_OOF":   bnn_oof[i]   if has_b else 0.0,
                "y_true":    df_fac.iloc[i][TARGET_COL],
            })

    # ── Final models trained on full facility data ─────────────────────────
    nhits_res = run_nhits(df_fac, fac, features, TARGET_COL, TEST_MONTHS)
    xgb_res   = run_xgboost_quantile(df_fac, fac, features, TARGET_COL, TEST_MONTHS)
    bnn_res   = run_bnn(df_fac, fac, features, TARGET_COL, TEST_MONTHS, n_epochs=300)

    bnn_res_dict[fac] = bnn_res
    xgb_res_dict[fac] = xgb_res

    # ── Collect test-period predictions ───────────────────────────────────
    y_test = df_fac[TARGET_COL].values[-TEST_MONTHS:]

    nhits_test = (
        np.array(nhits_res["y_pred"][:TEST_MONTHS])
        if nhits_res and nhits_res.get("y_pred") is not None
        else None
    )

    xgb_test = None
    if xgb_res and xgb_res.get("models") and 0.5 in xgb_res["models"]:
        X_test   = df_fac[features].values[-TEST_MONTHS:]
        xgb_test = xgb_res["models"][0.5].predict(X_test)

    bnn_test = None
    if bnn_res and bnn_res.get("model") is not None:
        X_test   = df_fac[features].values[-TEST_MONTHS:]
        mc       = mc_predict_bnn(bnn_res["model"], X_test)
        bnn_test = mc.mean(axis=0)

    elapsed = time.time() - t_fac
    print(
        f"  [{fac_idx+1:2d}/{len(facilities)}] {fac:<20}  "
        f"NHITS={nhits_res.get('mape', float('nan')):.1f}%  "
        f"XGB={xgb_res.get('mape', float('nan')):.1f}%  "
        f"BNN={bnn_res.get('mape', float('nan')):.1f}%  "
        f"({elapsed:.0f}s)",
        flush=True,
    )

    # ── XGB uncertainty on test set (Q90-Q10 spread per timestep) ────────
    xgb_unc_test = np.zeros(TEST_MONTHS)
    if xgb_res and xgb_res.get("models"):
        X_test_tmp = df_fac[features].values[-TEST_MONTHS:]
        if 0.1 in xgb_res["models"] and 0.9 in xgb_res["models"]:
            q10 = xgb_res["models"][0.1].predict(X_test_tmp)
            q90 = xgb_res["models"][0.9].predict(X_test_tmp)
            xgb_unc_test = np.abs(q90 - q10)

    # ── BNN MC uncertainty on test set (std across MC samples) ────────────
    bnn_unc_test = np.zeros(TEST_MONTHS)
    if bnn_res and bnn_res.get("model") is not None:
        X_test_tmp = df_fac[features].values[-TEST_MONTHS:]
        mc_samples = mc_predict_bnn(bnn_res["model"], X_test_tmp)
        bnn_unc_test = mc_samples.std(axis=0)

    results.append({
        "Facility":    fac,
        "NHITSMAPE":   nhits_res.get("mape",   np.nan) if nhits_res else np.nan,
        "NHITSRMSE":   nhits_res.get("rmse",   np.nan) if nhits_res else np.nan,
        "XGBoostMAPE": xgb_res.get("mape",    np.nan) if xgb_res  else np.nan,
        "XGBoostRMSE": xgb_res.get("rmse",    np.nan) if xgb_res  else np.nan,
        "BNNMAPE":     bnn_res.get("mape",     np.nan) if bnn_res  else np.nan,
        "BNNRMSE":     bnn_res.get("rmse",     np.nan) if bnn_res  else np.nan,
        "nhits_test":   nhits_test,
        "xgb_test":     xgb_test,
        "bnn_test":     bnn_test,
        "xgb_unc_test": xgb_unc_test,
        "bnn_unc_test": bnn_unc_test,
        "y_test":       y_test,
    })

print(f"\n  Done — {len(results)} facilities trained.")

# ---------------------------------------------------------------------------
# Step 3 – Train meta-learner on OOF, compute per-facility ensemble MAPE
# ---------------------------------------------------------------------------
print(f"\n[3/4] Training stacking meta-learner on {len(meta_feature_records)} OOF rows …")

meta_df          = pd.DataFrame(meta_feature_records)
meta_df_enhanced = build_meta_features(meta_df, bnn_res_dict, xgb_res_dict)

meta_models, meta_feature_cols, meta_weights = train_meta_learner(
    meta_df_enhanced, verbose=False,
)

if meta_models is None:
    print("  WARNING: Meta-learner training failed — too few samples.")
    sys.exit(1)

print(f"  Meta-learner trained.  Weights: {meta_weights}")

# OOF ensemble MAPE (same facility split used in notebook Cell 12)
oof_eval = evaluate_meta_learner(
    meta_models, meta_weights, meta_feature_cols,
    meta_df_enhanced,
    individual_oof_cols={
        "N-HiTS":   "NHITS_OOF",
        "XGBoost":  "XGB_OOF",
        "BNN":      "BNN_OOF",
    },
)
print(f"\n  OOF MAPE comparison:")
print(f"    Ensemble : {oof_eval['ensemble_mape']:.2f}%")
for m, v in oof_eval["individual_mapes"].items():
    print(f"    {m:<10}: {v:.2f}%  (diff vs ens {oof_eval['improvement'][m]:+.2f} pp)")

# Per-facility TEST ensemble MAPE
print(f"\n  Per-facility test ensemble MAPE:\n")
fac_ensemble = {}

for r in results:
    fac        = r["Facility"]
    y_test     = r["y_test"]
    nhits_pred = r["nhits_test"]
    xgb_pred   = r["xgb_test"]
    bnn_pred   = r["bnn_test"]

    if nhits_pred is None or xgb_pred is None or bnn_pred is None:
        print(f"    {fac:<20}  SKIPPED (missing base predictions)")
        fac_ensemble[fac] = {"mape": np.nan, "rmse": np.nan}
        continue

    n = min(TEST_MONTHS, len(y_test), len(nhits_pred), len(xgb_pred), len(bnn_pred))

    # Simple mean ensemble: unweighted average of the 3 base test predictions.
    # The stacking meta-learner is a global model trained on all-facility OOF data
    # and cannot reliably generalise across the large emission scale range between
    # facilities (< 1 tCO2 to > 300 tCO2).  A simple mean ensemble gives a valid,
    # scale-agnostic test-set metric that is directly comparable to the base models.
    ens_preds = (
        np.array(nhits_pred[:n]) +
        np.array(xgb_pred[:n])   +
        np.array(bnn_pred[:n])
    ) / 3.0

    y_arr    = y_test[:n]
    ens_mape = safe_mape(y_arr, ens_preds)
    ens_rmse = float(mean_squared_error(y_arr, ens_preds) ** 0.5)

    print(
        f"    {fac:<20}  MAPE={ens_mape:.1f}%  "
        f"RMSE={ens_rmse:.3f}  "
        f"(NHITS={r['NHITSMAPE']:.1f}%  XGB={r['XGBoostMAPE']:.1f}%  BNN={r['BNNMAPE']:.1f}%)",
        flush=True,
    )
    fac_ensemble[fac] = {"mape": ens_mape, "rmse": ens_rmse}

# ---------------------------------------------------------------------------
# Step 4 – Write EnsembleMAPE / EnsembleRMSE to model_accuracy.csv
# ---------------------------------------------------------------------------
print(f"\n[4/4] Updating results/model_accuracy.csv …")

acc_df = pd.read_csv("results/model_accuracy.csv")
acc_df["EnsembleMAPE"] = acc_df["Facility"].map(
    {k: v["mape"] for k, v in fac_ensemble.items()}
)
acc_df["EnsembleRMSE"] = acc_df["Facility"].map(
    {k: v["rmse"] for k, v in fac_ensemble.items()}
)
acc_df.to_csv("results/model_accuracy.csv", index=False)

print(f"  Saved {acc_df['EnsembleMAPE'].notna().sum()} facilities.\n")
print(acc_df[["Facility", "NHITSMAPE", "XGBoostMAPE", "BNNMAPE", "EnsembleMAPE"]].to_string(index=False))

overall = acc_df["EnsembleMAPE"].mean()
print(f"\n  Overall mean EnsembleMAPE (simple mean, test set): {overall:.2f}%")
print(f"  Meta-learner OOF MAPE (stacking):                  {oof_eval['ensemble_mape']:.2f}%")
print(f"  (N-HiTS: {acc_df['NHITSMAPE'].mean():.2f}%  "
      f"XGBoost: {acc_df['XGBoostMAPE'].mean():.2f}%  "
      f"BNN: {acc_df['BNNMAPE'].mean():.2f}%)")

print("\nDone.")
