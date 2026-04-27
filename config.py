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
    "F1", "F2",  "F5",  "F8",  "F3",  "F14", "F15", "F16", "F20",
    "F21", "F23", "F24", "F26", "F29", "F32", "F34", "F35",
    "F36", "F37", "F38", "F39", "F40", "F41", "F42", "F45",
    "F46", "F48",
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
TARGET_REDUCTION   = 0.10    # 10 % reduction from baseline
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
