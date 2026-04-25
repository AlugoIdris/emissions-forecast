"""
bnn_model.py
------------
Bayesian Neural Network (BNN) with Monte Carlo Dropout for per-facility
emissions forecasting and predictive uncertainty quantification.

Architecture: 3 hidden layers (128-64-32) with BatchNorm + MC Dropout.
Uncertainty is estimated via MC Dropout sampling at inference time.

Authors : Idris Alugo
Paper   : "Long-Horizon Emissions Forecasting for 2030 Target Assessment:
           A Comparative Study of N-HiTS, XGBoost, and Bayesian Models
           in Fast-Moving Consumer Goods Supply Chains" – Applied Energy (2026)
License : MIT
"""

import logging
import random
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from config import BNN_CONFIG, SHAP_CONFIG, TARGET_COL, TEST_MONTHS, RANDOM_SEED, MIN_TRAIN_ROWS
from src.metrics import safe_mape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------

def _set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

class BNN(nn.Module):
    """Bayesian Neural Network using MC Dropout for uncertainty estimation.

    Architecture:
        - 3 hidden layers of sizes 128 → 64 → 32
        - Each hidden layer: Linear → BatchNorm1d → ReLU → Dropout
        - Output layer: Linear (no activation — regression in original scale)
        - Xavier uniform weight initialisation

    MC Dropout is activated at inference time by calling ``model.train()``
    selectively on ``nn.Dropout`` modules, while keeping BatchNorm in eval
    mode. See ``mc_predict_bnn`` for the correct inference pattern.

    Args:
        n_inputs:      Number of input features.
        hidden_sizes:  Tuple of hidden layer widths.
        dropout_rates: Tuple of dropout probabilities (one per hidden layer).
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_sizes: tuple = BNN_CONFIG.hidden_sizes,
        dropout_rates: tuple = BNN_CONFIG.dropout_rates,
    ) -> None:
        super().__init__()

        layers = []
        prev_size = n_inputs

        for hidden_size, dropout_rate in zip(hidden_sizes, dropout_rates):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Xavier uniform initialisation for all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_bnn(
    df: pd.DataFrame,
    facility: str,
    feature_cols: list[str],
    metric: str = TARGET_COL,
    test_months: int = TEST_MONTHS,
    n_epochs: int = BNN_CONFIG.n_epochs,
    patience: int = BNN_CONFIG.patience,
) -> dict:
    """Train a BNN with early stopping for one facility.

    Trains on the **original (unscaled) target** — no y-scaling applied.
    Uses AdamW optimiser with ReduceLROnPlateau scheduler and gradient
    clipping. Best model weights (lowest validation loss) are restored
    after training.

    MC Dropout samples (``BNN_CONFIG.mc_samples``) are drawn from the
    validation set to produce a mean prediction and std uncertainty estimate.

    Args:
        df:           Full enhanced DataFrame for all facilities.
        facility:     Facility ID to filter and train on.
        feature_cols: Feature column names to use as model inputs.
        metric:       Target column (default: ``EmissionstCO2``).
        test_months:  Tail months held out as validation/test set.
        n_epochs:     Maximum training epochs.
        patience:     Early-stopping patience (epochs without improvement).

    Returns:
        Dict with keys:
            - ``facility``        : str
            - ``model``           : trained BNN (or ``None`` on failure)
            - ``mape``            : float – MC mean MAPE (%)
            - ``rmse``            : float – MC mean RMSE
            - ``predictions_mc``  : np.ndarray shape (mc_samples, n_test)
            - ``predictions_mean``: np.ndarray shape (n_test,)
            - ``predictions_std`` : np.ndarray shape (n_test,)
            - ``X_val_orig``      : np.ndarray – unscaled validation features
            - ``y_val``           : np.ndarray – validation targets
            - ``n_features``      : int
    """
    _set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = BNN_CONFIG

    df_fac = df[df["Facility"] == facility].copy()
    required_cols = list(feature_cols) + [metric]
    df_fac = df_fac.dropna(subset=required_cols)

    if len(df_fac) < MIN_TRAIN_ROWS:
        logger.warning("BNN: insufficient data for %s (%d rows).", facility, len(df_fac))
        return {"facility": facility, "mape": np.nan, "rmse": np.nan, "model": None}

    X = df_fac[list(feature_cols)].values
    y = df_fac[metric].values

    split_idx = len(df_fac) - test_months
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

    batch_size   = min(32, max(4, len(X_train) // 4))
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    model     = BNN(n_inputs=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
    )
    criterion = nn.MSELoss()

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    for epoch in range(n_epochs):
        # --- Training pass ---
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item()

        # --- Validation pass ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(X_val_t.to(device)),
                y_val_t.to(device),
            ).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch > 50:
            logger.debug("BNN early stop at epoch %d for %s.", epoch, facility)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- MC Dropout inference on validation set ---
    mc_preds = mc_predict_bnn(model, X_val, mc_samples=cfg.mc_samples)
    y_pred_mean = mc_preds.mean(axis=0)
    y_pred_std  = mc_preds.std(axis=0)

    mape = safe_mape(y_val, y_pred_mean)
    rmse = mean_squared_error(y_val, y_pred_mean) ** 0.5

    logger.info("BNN [%s] MAPE=%.2f%%  RMSE=%.4f  val_loss=%.6f", facility, mape, rmse, best_val_loss)
    logger.debug("BNN [%s] y_val range [%.2f, %.2f]  pred range [%.2f, %.2f]",
                 facility, y_val.min(), y_val.max(), y_pred_mean.min(), y_pred_mean.max())

    return {
        "facility":         facility,
        "model":            model,
        "mape":             mape,
        "rmse":             rmse,
        "predictions_mc":   mc_preds,
        "predictions_mean": y_pred_mean,
        "predictions_std":  y_pred_std,
        "X_val_orig":       X_val,
        "y_val":            y_val,
        "n_features":       X.shape[1],
    }


# ---------------------------------------------------------------------------
# MC Dropout Inference
# ---------------------------------------------------------------------------

def mc_predict_bnn(
    model: BNN,
    X: Union[np.ndarray, pd.DataFrame],
    mc_samples: int = BNN_CONFIG.mc_samples,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Generate MC Dropout predictions for uncertainty estimation.

    Sets the model to ``eval()`` mode (freezes BatchNorm statistics) but
    re-enables only the ``nn.Dropout`` modules so stochastic dropout
    remains active. Each forward pass produces a different realisation of
    the posterior predictive distribution.

    Args:
        model:      Trained BNN instance.
        X:          Input features — np.ndarray or DataFrame
                    of shape ``(n_samples, n_features)``.
        mc_samples: Number of stochastic forward passes.
        seed:       Random seed for deterministic MC sampling.

    Returns:
        Array of shape ``(mc_samples, n_samples)`` — each row is one
        posterior draw. Use ``.mean(axis=0)`` for the point estimate and
        ``.std(axis=0)`` for epistemic uncertainty.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if hasattr(X, "values"):
        X = X.values
    if X.ndim == 1:
        X = X.reshape(1, -1)

    device = next(model.parameters()).device
    X_t    = torch.tensor(X.astype(np.float32), dtype=torch.float32).to(device)

    # Freeze BatchNorm; activate Dropout
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    predictions = []
    with torch.no_grad():
        for _ in range(mc_samples):
            pred = model(X_t).cpu().numpy().flatten()
            predictions.append(pred)

    return np.array(predictions)   # shape: (mc_samples, n_samples)


# ---------------------------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------------------------

def shap_bnn(
    model: BNN,
    X_train: np.ndarray,
    X_val: np.ndarray,
    features: list[str],
    facility: str,
    show_plot: bool = True,
) -> Optional[np.ndarray]:
    """Compute SHAP values for a BNN via KernelExplainer.

    Uses a deterministic point-prediction wrapper (model in eval mode,
    dropout disabled) as the SHAP black-box function. This gives
    consistent SHAP values across runs.

    Note on N-HiTS SHAP proxy: the original notebook used
    ``np.mean(X, axis=1)`` as a linear proxy for the N-HiTS prediction
    function. That proxy is **not** used here — this function wraps the
    actual BNN forward pass.

    Args:
        model:     Trained BNN instance.
        X_train:   Training features for KernelExplainer background set.
        X_val:     Validation features to explain.
        features:  Feature name list.
        facility:  Facility ID (used in plot title).
        show_plot: If ``True``, render a SHAP beeswarm summary plot.

    Returns:
        SHAP values array of shape ``(n_explain, n_features)``, or
        ``None`` on failure.
    """
    cfg = SHAP_CONFIG

    if hasattr(X_train, "values"):
        X_train = X_train.values
    if hasattr(X_val, "values"):
        X_val = X_val.values

    X_train = np.asarray(X_train, dtype=np.float64)
    X_val   = np.asarray(X_val,   dtype=np.float64)

    background  = X_train[:cfg.bnn_background_max]
    explain_set = X_val[:cfg.bnn_explain_max]

    device = next(model.parameters()).device

    def predict_fn(X_input: np.ndarray) -> np.ndarray:
        """Deterministic BNN prediction wrapper for SHAP."""
        model.eval()   # dropout disabled — consistent predictions
        X_t = torch.tensor(X_input.astype(np.float32), dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X_t).cpu().numpy().flatten()
        return preds

    try:
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_vals = explainer.shap_values(explain_set, nsamples=cfg.bnn_nsamples)

        # KernelExplainer can return a list for multi-output — extract array
        if isinstance(shap_vals, list) and len(shap_vals) > 0:
            shap_vals = shap_vals[0]

        shap_vals = np.asarray(shap_vals, dtype=np.float64)

        if shap_vals.ndim != 2 or shap_vals.shape[1] != len(features):
            logger.error(
                "BNN SHAP shape mismatch for %s: got %s, expected (n, %d).",
                facility, shap_vals.shape, len(features),
            )
            return None

        if show_plot:
            shap.summary_plot(
                shap_vals,
                explain_set,
                feature_names=features,
                show=False,
                max_display=cfg.max_display,
            )
            plt.title(f"BNN SHAP – {facility}")
            plt.tight_layout()
            plt.show()

        logger.info("BNN SHAP computed for %s  shape=%s", facility, shap_vals.shape)
        return shap_vals

    except Exception as exc:
        logger.error("BNN SHAP failed for %s: %s", facility, exc)
        return None
