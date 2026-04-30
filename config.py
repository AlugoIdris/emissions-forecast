"""
config.py
---------
Central configuration for the emissions forecasting pipeline.
All hyperparameters, paths, and constants live here.
Import this module in any other src/ file instead of hardcoding values.

Authors : Idris Alugo
Paper   : "Long-Horizon Emissions Forecasting for 2030 Target Assessment:
           A Comparative Study of N-HiTS, XGBoost, and Bayesian Models
           in Fast-Moving Consumer Goods Supply Chains" – Applied Energy (2026)
License : MIT
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR   = Path(__file__).resolve().parent   # project root
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "results"

RAW_DATA_PATH = DATA_DIR / "esgdata.csv"
DATA_PATH = RAW_DATA_PATH  # alias for backward compatibility


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data / Preprocessing
# ---------------------------------------------------------------------------

DATE_MIN_YEAR = 2016          # removes 2015 baseline year
DATE_MAX_YEAR = 2024          # last year of observed data (test period = 2024)

LAG_PERIODS   = [1, 3, 6, 12] # months

OUTLIER_FACILITIES = [
    # Statistically derived exclusions (IQR on mean, Z-score, CV>93%, or N<24)
    # verified against full dataset (Jan 2022 – Aug 2025, 48 facilities).
    # Names use post-rename IDs (Facility 1–48).
    #
    # High coefficient of variation (CV > 93.1% threshold = Q3 + 1.5×IQR of CV):
    "Facility 31",  # orig F1  — CV=99%  (also short coverage)
    "Facility 33",  # orig F5  — CV=228%
    "Facility 35",  # orig F8  — CV=112%
    "Facility 36",  # orig F14 — CV=176%
    "Facility 37",  # orig F16 — CV=464%
    "Facility 39",  # orig F34 — CV=120%
    "Facility 42",  # orig F38 — CV=119%
    #
    # Insufficient data (N < 24 months in 2022+ window):
    "Facility 32",  # orig F2  — N=16
    "Facility 47",  # orig F47 — no 2022+ monthly data
    #
    # IQR outlier on mean emissions (mean > Q3 + 1.5×IQR = 494 tCO2):
    "Facility 41",  # orig F37 — mean=544
    "Facility 43",  # orig F43 — mean=750
    "Facility 44",  # orig F44 — mean=674
    "Facility 45",  # orig F45 — mean=823
    "Facility 46",  # orig F46 — mean=2017 (also Z-score=5.25)
    "Facility 48",  # orig F48 — mean=717
    #
    # Data provenance exclusions (facilities transferred in corporate acquisition):
    "Facility 34",  # orig F7  — acquired; data no longer represents same entity
    "Facility 38",  # orig F20 — acquired; data no longer represents same entity
    "Facility 40",  # orig F36 — acquired; data no longer represents same entity
]

# Numeric base features (before region dummies are appended)
FEATURES_BASE_NUMERIC = [
    "Month", "Year", "DayOfYear",
    "Production", "Energy_MWh", "Waste_Kg",
    "Renewable_percent", "PolicyWeight",
]

TARGET_COL   = "Emissions_tCO2"
FACILITY_COL = "Facility"
DATE_COL     = "Date"
REGION_COL   = "Region"


# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------

TEST_MONTHS    = 12   # held-out months for final evaluation
OOF_CV_SPLITS  = 3    # TimeSeriesSplit folds for OOF meta-features
MIN_TRAIN_ROWS = 21   # skip facility if fewer rows than this
MIN_TEST_ROWS  = 3    # minimum test rows required


# ---------------------------------------------------------------------------
# Target / Risk Analysis
# ---------------------------------------------------------------------------

BASELINE_YEAR      = 2023
TARGET_YEAR        = 2030
TARGET_REDUCTION   = 0.30    # 30 % reduction from 2023 baseline by 2030
                              # Aligned with SBTi 1.5°C near-term corporate pathway:
                              # ~4.2 % linear annual reduction × 7 years (2023→2030)
                              # Reference: SBTi Corporate Manual v2.0 (2023)
TARGET_REDUCTION_RATE = TARGET_REDUCTION  # alias for backward compatibility
N_BOOTSTRAP        = 1000    # bootstrap samples for probability estimation


# ---------------------------------------------------------------------------
# XGBoost Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class XGBoostConfig:
    quantiles: list      = field(default_factory=lambda: [0.1, 0.5, 0.9])
    n_estimators: int    = 400
    learning_rate: float = 0.05
    max_depth: int       = 6
    subsample: float     = 0.9
    colsample_bytree: float = 0.9
    random_state: int    = RANDOM_SEED


XGB_CONFIG = XGBoostConfig()


# ---------------------------------------------------------------------------
# N-HiTS Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class NHiTSConfig:
    freq: str            = "M"
    max_steps: int       = 150
    learning_rate: float = 1e-3
    batch_size: int      = 8
    random_seed: int     = RANDOM_SEED
    # input_size is computed adaptively per facility (min 6, max 12)
    input_size_cap: int  = 12
    input_size_min: int  = 6


NHITS_CONFIG = NHiTSConfig()


# ---------------------------------------------------------------------------
# Bayesian Neural Network (BNN) Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class BNNConfig:
    hidden_sizes: tuple      = (128, 64, 32)
    dropout_rates: tuple     = (0.15, 0.15, 0.10)
    learning_rate: float     = 1e-3
    weight_decay: float      = 1e-4
    n_epochs: int            = 500
    patience: int            = 30
    lr_scheduler_patience: int = 15
    lr_scheduler_factor: float = 0.5
    grad_clip_norm: float    = 1.0
    mc_samples: int          = 100       # Monte Carlo dropout samples
    n_epochs_oof: int        = 100       # reduced epochs for OOF folds
    random_seed: int         = RANDOM_SEED


BNN_CONFIG = BNNConfig()


# ---------------------------------------------------------------------------
# Meta-Learner Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class MetaLearnerConfig:
    cv_folds: int            = 5
    min_samples: int         = 20        # minimum rows to train meta-learner
    # Ridge
    ridge_alphas_log: tuple  = (-3, 3, 50)   # np.logspace args
    # ElasticNet
    elastic_l1_ratios: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    elastic_max_iter: int    = 2000
    # GBM
    gbm_n_estimators: int    = 100
    gbm_max_depth: int       = 3
    gbm_learning_rate: float = 0.05
    gbm_subsample: float     = 0.8
    gbm_min_samples_leaf: int = 5
    # Random Forest
    rf_n_estimators: int     = 100
    rf_max_depth: int        = 10
    rf_min_samples_leaf: int = 3


META_CONFIG = MetaLearnerConfig()


# ---------------------------------------------------------------------------
# SHAP Settings
# ---------------------------------------------------------------------------

@dataclass
class SHAPConfig:
    xgb_max_samples: int     = 50    # TreeExplainer sample cap
    bnn_background_max: int  = 100   # KernelExplainer background size
    bnn_explain_max: int     = 20    # KernelExplainer explain set size
    bnn_nsamples: int        = 100   # KernelExplainer nsamples arg
    nhits_background_max: int = 20
    nhits_explain_max: int   = 10
    max_display: int         = 10    # features shown in summary plot


SHAP_CONFIG = SHAPConfig()
