# Data

## Dataset Overview

This directory is intentionally **empty** in the public repository.  
The underlying ESG emissions dataset used in the paper contains proprietary
industrial facility data and cannot be redistributed publicly.

---

## Dataset Description

| Property | Details |
|---|---|
| **Name** | Industrial Facility ESG Emissions Dataset |
| **Format** | CSV (`esgdata.csv`) |
| **Granularity** | Monthly, per-facility |
| **Time range** | 2018 – 2024 |
| **Facilities** | 20 industrial facilities |
| **Regions** | 4 geographic regions |
| **Target variable** | `EmissionstCO2` — Scope 1 CO₂ equivalent emissions (tonnes) |
| **Size** | ~3,000 rows × 12 columns |

---

## Column Schema

| Column | Type | Description |
|---|---|---|
| `Date` | datetime | Monthly observation date (YYYY-MM-DD, first of month) |
| `Facility` | string | Facility identifier (e.g. `F01` … `F20`) |
| `Region` | string | Geographic region label |
| `EmissionstCO2` | float | Monthly Scope 1 emissions (tCO₂eq) — **target variable** |
| `Production` | float | Monthly production volume (facility-specific units) |
| `EnergyMWh` | float | Total energy consumption (MWh) |
| `WasteKg` | float | Total waste generated (kg) |
| `RenewablePercent` | float | Share of renewable energy in total consumption (0–100) |
| `PolicyWeight` | float | Regulatory exposure index (0–1 scale, higher = more regulated) |

---

## Synthetic Sample

A **fully synthetic** sample dataset (`esgdata_sample.csv`) is provided for
pipeline testing and reproduction checks. It is generated from the same
distributional properties as the real data (means, variances, temporal
structure) but contains **no real facility records**.

To generate the synthetic sample:

```bash
python src/preprocessing.py --generate-sample --n-facilities 20 --n-months 84
```

This produces `data/esgdata_sample.csv` with the same schema as above and
is sufficient to run `notebooks/01_run_pipeline.ipynb` end-to-end.

---

## Access to Real Data

Researchers wishing to access the original dataset for replication purposes
may contact the corresponding author:

**Email**: idris_olawale.alugo.dokt@pw.edu.pl 
**Subject**: *Data Access Request — Applied Energy Emissions Forecasting Paper*

Requests will be reviewed on a case-by-case basis subject to data governance
agreements with the originating industrial partners.

---

## Preprocessing Steps Applied

The following transformations are applied in `src/preprocessing.py`
before any modelling:

1. **Date parsing** — `pd.to_datetime()` with `infer_datetime_format=True`
2. **Outlier removal** — IQR-based filtering (Q1 − 3×IQR, Q3 + 3×IQR) per facility
3. **Region one-hot encoding** — `pd.get_dummies()`, prefix `Region_`
4. **Lag features** — 1-month, 3-month, 6-month, 12-month lags of `EmissionstCO2`
5. **Rolling statistics** — 3-month and 6-month rolling mean & std
6. **Hierarchy features** — facility-level mean emission embedding
7. **Global feature scaling** — `StandardScaler` fitted on training split only
   (scaler objects stored in `global_scalers` dict, never applied to the target)

No imputation is applied — rows with missing values in required columns
are dropped before each model's training split.

---

## Baseline & Target Definition

- **Baseline year**: 2022 (mean annual facility emissions)
- **2030 target**: 10% reduction from 2022 baseline
  (`Target2030 = BaselineEmissions × 0.90`)
- Configurable in `config.py` via `BASELINE_YEAR` and `TARGET_REDUCTION_RATE`

---

## Citation

If you use the synthetic sample or the preprocessing pipeline, please cite:

```bibtex
@article{alugo2026ensemble,
  title   = {Long-Horizon Emissions Forecasting for 2030 Target Assessment: A Comparative Study of N-HiTS, XGBoost, and Bayesian Models in Fast-Moving Consumer Goods Supply Chains},
  author  = {Idris Alugo},
  journal = {Applied Energy},
  year    = {2026},
  doi     = {10.5281/zenodo.19755807}
}
```
