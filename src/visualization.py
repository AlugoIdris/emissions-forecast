"""
visualization.py
----------------
All publication-quality plots and stakeholder dashboards for the
ensemble emissions forecasting pipeline.

Covers:
    - MAPE distribution box plot (Figure 1)
    - Uncertainty decomposition bar chart (Figure 2)
    - SHAP contribution comparison across models (Figure 3)
    - 2030 target achievement probability distribution (Figure 4)
    - Model agreement on target achievement (Figure 5)
    - Stakeholder dashboard (6-panel summary)
    - Facility-level dot-and-whisker accuracy plot

All functions save to ``output_dir`` and optionally display inline.

Authors : Idris Alugo
Paper   : "Long-Horizon Emissions Forecasting for 2030 Target Assessment:
           A Comparative Study of N-HiTS, XGBoost, and Bayesian Models
           in Fast-Moving Consumer Goods Supply Chains" – Applied Energy (2026)
License : MIT
"""

import logging
import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

RISK_COLORS = {
    "Low Risk":      "#2ecc71",
    "Medium Risk":   "#f1c40f",
    "High Risk":     "#e67e22",
    "Critical Risk": "#e74c3c",
    "Unknown":       "#bdc3c7",
}

MODEL_COLORS = {
    "N-HiTS":   "#1f77b4",
    "XGBoost":  "#ff7f0e",
    "BNN":      "#2ca02c",
    "Ensemble": "#9467bd",
}

def _apply_style() -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.labelsize":   12,
        "axes.titlesize":   14,
        "legend.fontsize":  10,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
    })

def _save(fig: plt.Figure, path: str, show: bool) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # PNG at 300 dpi (Elsevier minimum for raster figures)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", path)
    # PDF (vector) alongside every PNG — preferred format for Elsevier submission
    pdf_path = os.path.splitext(path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info("Saved: %s", pdf_path)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — MAPE Distribution Box Plot
# ---------------------------------------------------------------------------

def plot_mape_distribution(
    results_df: pd.DataFrame,
    output_dir: str = "figures",
    show: bool = True,
) -> str:
    """Box plot of per-facility MAPE distributions for each base model.

    Args:
        results_df: DataFrame with columns ``NHITSMAPE``, ``XGBoostMAPE``,
                    ``BNNMAPE`` (and optionally ``EnsembleMAPE``).
        output_dir: Directory to save the figure.
        show:       Display inline if True.

    Returns:
        Saved file path.
    """
    _apply_style()

    model_cols = {
        "N-HiTS":   "NHITSMAPE",
        "XGBoost":  "XGBoostMAPE",
        "BNN":      "BNNMAPE",
    }
    if "EnsembleMAPE" in results_df.columns:
        model_cols["Ensemble"] = "EnsembleMAPE"

    data_to_plot = [results_df[col].dropna().values for col in model_cols.values()]
    labels       = list(model_cols.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        data_to_plot,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="red", linewidth=2),
        boxprops=dict(facecolor="lightblue", alpha=0.7),
    )

    for i, data in enumerate(data_to_plot):
        ax.plot(i + 1, np.mean(data), marker="D", markersize=8, color="green",
                label="Mean" if i == 0 else "")

    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Model Accuracy Distribution Across Facilities", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    path = os.path.join(output_dir, "figure1_mape_boxplot.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Figure 2 — Uncertainty Decomposition
# ---------------------------------------------------------------------------

def plot_uncertainty_decomposition(
    risk_df: pd.DataFrame,
    results_df: pd.DataFrame,
    selected_facilities: Optional[list] = None,
    output_dir: str = "figures",
    show: bool = True,
) -> str:
    """Stacked bar chart decomposing aleatoric vs epistemic uncertainty.

    Within-model variance (aleatoric) = weighted combination of per-model stds.
    Between-model variance (epistemic) = weighted spread across model predictions.

    Args:
        risk_df:              Output of ``compute_calibrated_probabilities``.
        results_df:           Per-facility model MAPE results.
        selected_facilities:  List of facility IDs to plot (default: top 5 by uncertainty).
        output_dir:           Directory to save the figure.
        show:                 Display inline if True.

    Returns:
        Saved file path.
    """
    _apply_style()

    if selected_facilities is None:
        selected_facilities = (
            risk_df.nlargest(5, "UncertaintyEnsemble")["Facility"].tolist()
        )

    decomp_records = []
    for fac in selected_facilities:
        fac_data = risk_df[risk_df["Facility"] == fac]
        if len(fac_data) == 0:
            continue

        pred_xgb = fac_data["PredictionXGBoost"].values[0]
        pred_bnn = fac_data["PredictionBNN"].values[0]
        pred_ens = fac_data["PredictionEnsemble"].values[0]
        unc_xgb  = fac_data["UncertaintyXGBoost"].values[0]
        unc_bnn  = fac_data["UncertaintyBNN"].values[0]
        unc_ens  = fac_data["UncertaintyEnsemble"].values[0]

        fac_results = results_df[results_df["Facility"] == fac]
        xgb_mape = fac_results["XGBoostMAPE"].values[0] if len(fac_results) > 0 else 25.0
        bnn_mape = fac_results["BNNMAPE"].values[0]     if len(fac_results) > 0 else 60.0

        w_xgb = (1.0 / (xgb_mape + 1.0)) ** 2
        w_bnn = (1.0 / (bnn_mape + 1.0)) ** 2
        total_w = w_xgb + w_bnn
        w_xgb /= total_w
        w_bnn /= total_w

        within_model = np.sqrt(w_xgb**2 * unc_xgb**2 + w_bnn**2 * unc_bnn**2)
        between_model = np.sqrt(
            w_xgb * (pred_xgb - pred_ens)**2 + w_bnn * (pred_bnn - pred_ens)**2
        )

        decomp_records.append({
            "Facility":    fac,
            "WithinModel": within_model,
            "BetweenModel": between_model,
            "Total":       unc_ens,
        })

    decomp_df = pd.DataFrame(decomp_records)
    x     = np.arange(len(decomp_df))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, decomp_df["WithinModel"],  width, label="Within-Model (Aleatoric)", color="#1f77b4", alpha=0.8)
    ax.bar(x, decomp_df["BetweenModel"], width, bottom=decomp_df["WithinModel"],
           label="Between-Model (Epistemic)", color="#ff7f0e", alpha=0.8)
    ax.plot(x, decomp_df["Total"], marker="o", color="red", linewidth=2,
            markersize=8, label="Total Uncertainty")

    ax.set_ylabel(r"Uncertainty (tCO$_2$)", fontsize=12)
    ax.set_xlabel("Facility", fontsize=12)
    ax.set_title("Ensemble Uncertainty Decomposition — Selected Facilities",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(decomp_df["Facility"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(output_dir, "figure2_uncertainty_decomposition.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Figure 3 — SHAP Contribution Comparison
# ---------------------------------------------------------------------------

def plot_shap_comparison(
    shap_summary: pd.DataFrame,
    top_n: int = 10,
    output_dir: str = "figures",
    show: bool = True,
) -> str:
    """Grouped bar chart comparing SHAP feature importance across models.

    Args:
        shap_summary: Output of ``aggregate_shap_records`` from evaluation.py.
        top_n:        Number of top features to display.
        output_dir:   Directory to save the figure.
        show:         Display inline if True.

    Returns:
        Saved file path.
    """
    _apply_style()

    # Sort by average contribution across all three models
    if "AvgContribution" not in shap_summary.columns:
        shap_summary = shap_summary.copy()
        shap_summary["AvgContribution"] = (
            shap_summary[["NHITSContribution", "XGBContribution", "BNNContribution"]].mean(axis=1)
        )
    # Top-N sorted descending so highest feature is at the top
    top = shap_summary.nlargest(top_n, "AvgContribution").sort_values("AvgContribution", ascending=True)
    y   = np.arange(len(top))
    w   = 0.18

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.55)))
    ax.barh(y - 1.5*w, top["NHITSContribution"], w, label="N-HiTS",  color=MODEL_COLORS["N-HiTS"],  alpha=0.9)
    ax.barh(y - 0.5*w, top["XGBContribution"],   w, label="XGBoost", color=MODEL_COLORS["XGBoost"], alpha=0.9)
    ax.barh(y + 0.5*w, top["BNNContribution"],    w, label="BNN",     color=MODEL_COLORS["BNN"],     alpha=0.9)
    ax.barh(y + 1.5*w, top["AvgContribution"],    w, label="Average", color="#7f7f7f",               alpha=0.9)

    ax.set_xlabel("SHAP Contribution (%)", fontsize=12)
    ax.set_title(
        f"Top {top_n} SHAP Features by Average Contribution Across N-HiTS, XGBoost, and BNN",
        fontsize=13, fontweight="bold",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(top["Feature"], fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "figure3_shap_comparison.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Figure 4 — Target Achievement Probability Distribution
# ---------------------------------------------------------------------------

def plot_probability_distribution(
    risk_df: pd.DataFrame,
    output_dir: str = "figures",
    show: bool = True,
) -> str:
    """Bar chart of 2030 compliance probability per facility, coloured by risk.

    Args:
        risk_df:    Output of ``compute_calibrated_probabilities``.
        output_dir: Directory to save the figure.
        show:       Display inline if True.

    Returns:
        Saved file path.
    """
    _apply_style()

    prob_sorted = risk_df.sort_values("ProbMeetTargetEnsemble").reset_index(drop=True)
    colors = prob_sorted["RiskLevelEnsemble"].map(RISK_COLORS).fillna("#bdc3c7")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(prob_sorted)), prob_sorted["ProbMeetTargetEnsemble"] * 100,
           color=colors)
    ax.axhline(y=50, color="black", linestyle="--", linewidth=1.5, label="50% threshold")

    ax.set_xlabel("Facility (sorted by probability)", fontsize=12)
    ax.set_ylabel("Target Achievement Probability (%)", fontsize=12)
    ax.set_title("2030 Target Achievement Probability by Facility",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(prob_sorted)))
    ax.set_xticklabels(prob_sorted["Facility"], rotation=90, fontsize=8)

    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for label, color in RISK_COLORS.items()
        if label != "Unknown"
    ]
    ax.legend(handles=[ax.get_lines()[0]] + legend_patches, loc="upper left")
    plt.tight_layout()

    path = os.path.join(output_dir, "figure4_probability_distribution.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Figure 5 — Model Agreement on Target Achievement
# ---------------------------------------------------------------------------

def plot_model_agreement(
    risk_df: pd.DataFrame,
    selected_facilities: Optional[list] = None,
    output_dir: str = "figures",
    show: bool = True,
) -> str:
    """Grouped bar chart showing per-model compliance probabilities per facility.

    Args:
        risk_df:              Output of ``compute_calibrated_probabilities``.
        selected_facilities:  Facility IDs to compare (default: 5 representative).
        output_dir:           Directory to save the figure.
        show:                 Display inline if True.

    Returns:
        Saved file path.
    """
    _apply_style()

    if selected_facilities is None:
        selected_facilities = risk_df["Facility"].head(5).tolist()

    prob_subset = risk_df[risk_df["Facility"].isin(selected_facilities)].reset_index(drop=True)
    x     = np.arange(len(selected_facilities))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, prob_subset["ProbMeetTargetNHITS"]    * 100, width,
           label="N-HiTS",  color=MODEL_COLORS["N-HiTS"])
    ax.bar(x - 0.5 * width, prob_subset["ProbMeetTargetXGBoost"]  * 100, width,
           label="XGBoost", color=MODEL_COLORS["XGBoost"])
    ax.bar(x + 0.5 * width, prob_subset["ProbMeetTargetBNN"]      * 100, width,
           label="BNN",     color=MODEL_COLORS["BNN"])
    ax.bar(x + 1.5 * width, prob_subset["ProbMeetTargetEnsemble"] * 100, width,
           label="Ensemble",color=MODEL_COLORS["Ensemble"])

    ax.axhline(y=50, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylabel("Probability of Meeting Target (%)", fontsize=12)
    ax.set_title("Model Agreement on 2030 Target Achievement",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(selected_facilities)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "figure5_model_agreement.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Stakeholder Dashboard (6-panel)
# ---------------------------------------------------------------------------

def plot_stakeholder_dashboard(
    risk_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_dir: str = "figures",
    show: bool = True,
) -> str:
    """Six-panel stakeholder summary dashboard.

    Panels:
        1. Risk level distribution (bar)
        2. Model accuracy MAPE (bar)
        3. Prediction interval coverage (bar)
        4. Target achievement probability histogram
        5. Predicted vs target emissions scatter
        6. Emissions gap vs uncertainty scatter

    Args:
        risk_df:       Output of ``compute_calibrated_probabilities``.
        comparison_df: Output of ``compare_model_metrics`` from evaluation.py.
        output_dir:    Directory to save the figure.
        show:          Display inline if True.

    Returns:
        Saved file path.
    """
    _apply_style()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ESG Emissions Forecast — Stakeholder Dashboard",
                 fontsize=16, fontweight="bold")

    # 1. Risk distribution
    ax = axes[0, 0]
    risk_counts = risk_df["RiskLevelEnsemble"].value_counts()
    bar_colors  = [RISK_COLORS.get(x, "#bdc3c7") for x in risk_counts.index]
    risk_counts.plot(kind="bar", ax=ax, color=bar_colors)
    ax.set_title("Risk Level Distribution", fontweight="bold")
    ax.set_ylabel("Number of Facilities")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)

    # 2. Model MAPE comparison
    ax = axes[0, 1]
    if "MAPE" in comparison_df.columns:
        comparison_df["MAPE"].plot(kind="bar", ax=ax, color="steelblue", legend=False)
    ax.set_title("Model Accuracy (MAPE)", fontweight="bold")
    ax.set_ylabel("MAPE (%)")
    ax.axhline(y=10, color="green", linestyle="--", label="Target 10%")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    # 3. PI coverage
    ax = axes[0, 2]
    if "PICoverage90" in comparison_df.columns:
        comparison_df["PICoverage90"].plot(kind="bar", ax=ax, color="coral", legend=False)
    ax.set_title("Prediction Interval Coverage (90%)", fontweight="bold")
    ax.set_ylabel("Coverage (%)")
    ax.axhline(y=90, color="green", linestyle="--", label="Target 90%")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    # 4. Probability histogram
    ax = axes[1, 0]
    valid_probs = risk_df["ProbMeetTargetEnsemble"].dropna()
    ax.hist(valid_probs, bins=20, color="skyblue", edgecolor="black")
    ax.axvline(x=0.5, color="red", linestyle="--", label="50% threshold")
    ax.set_title("Distribution of Target Achievement Probability", fontweight="bold")
    ax.set_xlabel("Probability of Meeting Target")
    ax.set_ylabel("Number of Facilities")
    ax.legend()

    # 5. Predicted vs target scatter
    ax = axes[1, 1]
    valid = risk_df.dropna(subset=["PredictionEnsemble", "Target2030", "UncertaintyEnsemble"])
    sc = ax.scatter(
        valid["Target2030"], valid["PredictionEnsemble"],
        c=valid["UncertaintyEnsemble"], cmap="viridis", alpha=0.6, s=100,
    )
    min_val = min(valid["Target2030"].min(), valid["PredictionEnsemble"].min())
    max_val = max(valid["Target2030"].max(), valid["PredictionEnsemble"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    ax.set_title("Predicted vs Target Emissions", fontweight="bold")
    ax.set_xlabel(r"Target 2030 (tCO$_2$)")
    ax.set_ylabel(r"Predicted 2030 (tCO$_2$)")
    plt.colorbar(sc, ax=ax, label="Uncertainty")
    ax.legend()

    # 6. Gap vs uncertainty
    ax = axes[1, 2]
    risk_df_copy = risk_df.copy()
    risk_df_copy["EmissionsGap"] = risk_df_copy["PredictionEnsemble"] - risk_df_copy["Target2030"]
    valid2 = risk_df_copy.dropna(subset=["UncertaintyEnsemble", "EmissionsGap", "RiskLevelEnsemble"])
    risk_colors_mapped = valid2["RiskLevelEnsemble"].map(RISK_COLORS).fillna("#bdc3c7")
    ax.scatter(valid2["UncertaintyEnsemble"], valid2["EmissionsGap"],
               c=risk_colors_mapped, alpha=0.6, s=100)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_title("Emissions Gap vs Uncertainty", fontweight="bold")
    ax.set_xlabel("Prediction Uncertainty (std)")
    ax.set_ylabel("Emissions Gap (Predicted − Target)")
    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for l, c in RISK_COLORS.items() if l != "Unknown"
    ]
    ax.legend(handles=legend_patches, loc="best", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "figure0_stakeholder_dashboard.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Figure 6 — Facility-level dot-and-whisker accuracy plot
# ---------------------------------------------------------------------------

def plot_facility_accuracy(
    results_df: pd.DataFrame,
    output_dir: str = "figures",
    show: bool = True,
) -> str:
    """Dot-and-whisker chart showing per-facility MAPE for each model.

    Args:
        results_df: Per-facility results DataFrame with MAPE columns.
        output_dir: Directory to save the figure.
        show:       Display inline if True.

    Returns:
        Saved file path.
    """
    _apply_style()

    model_map = {
        "N-HiTS":  "NHITSMAPE",
        "XGBoost": "XGBoostMAPE",
        "BNN":     "BNNMAPE",
    }
    facilities = results_df["Facility"].tolist()
    y_pos      = np.arange(len(facilities))

    fig, ax = plt.subplots(figsize=(10, max(6, len(facilities) * 0.4)))

    offsets = [-0.25, 0.0, 0.25]
    for (model_name, col), offset in zip(model_map.items(), offsets):
        if col not in results_df.columns:
            continue
        vals = results_df[col].values
        ax.scatter(vals, y_pos + offset,
                   label=model_name,
                   color=MODEL_COLORS.get(model_name, "gray"),
                   s=60, alpha=0.8, zorder=3)

    ax.axvline(x=10, color="green", linestyle="--", linewidth=1, label="10% MAPE target")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(facilities, fontsize=9)
    ax.set_xlabel("MAPE (%)", fontsize=12)
    ax.set_title("Per-Facility Forecast Accuracy by Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "figure6_facility_accuracy.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Figure 7 — Facility K-Means Clustering (Silhouette + PCA scatter)
# ---------------------------------------------------------------------------

def plot_facility_clusters(
    cluster_df: pd.DataFrame,
    sil_scores: dict,
    best_k: int,
    output_dir: str = "figures",
    show: bool = True,
) -> str:
    """Two-panel figure: silhouette-score elbow and PCA cluster scatter.

    Args:
        cluster_df:  DataFrame with columns ``Facility``, ``Cluster``,
                     ``PC1``, ``PC2``.
        sil_scores:  Mapping of ``{k: silhouette_score}`` for K sweep.
        best_k:      The K selected as optimal.
        output_dir:  Directory to save the figure.
        show:        Display inline if True.

    Returns:
        Saved file path.
    """
    _apply_style()

    cluster_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Facility Emissions Profile Clustering (K-Means)",
                 fontsize=14, fontweight="bold")

    # Panel 1 — Silhouette score vs K
    ks     = sorted(sil_scores.keys())
    scores = [sil_scores[k] for k in ks]
    ax1.plot(ks, scores, marker="o", linewidth=2, color="#1f77b4")
    ax1.axvline(x=best_k, color="red", linestyle="--",
                linewidth=1.5, label=f"Best K={best_k}")
    ax1.scatter([best_k], [sil_scores[best_k]], color="red", zorder=5, s=100)
    ax1.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax1.set_ylabel("Silhouette Score", fontsize=12)
    ax1.set_title("Silhouette Score vs K", fontweight="bold")
    ax1.set_xticks(ks)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2 — PCA scatter coloured by semantic label
    # Use fixed colours for semantic cluster names; fall back to palette for numeric IDs
    semantic_colors = {
        "Low Emitter":    "#2ca02c",   # green
        "Medium Emitter": "#ff7f0e",   # orange
        "High Emitter":   "#d62728",   # red
    }
    label_order = (
        ["Low Emitter", "Medium Emitter", "High Emitter"]
        if set(cluster_df["Cluster"].unique()).issubset(semantic_colors)
        else sorted(cluster_df["Cluster"].unique())
    )
    for i, label in enumerate(label_order):
        grp = cluster_df[cluster_df["Cluster"] == label]
        if grp.empty:
            continue
        color = semantic_colors.get(label, cluster_palette[i % len(cluster_palette)])
        ax2.scatter(grp["PC1"], grp["PC2"], label=label,
                    color=color, s=80, alpha=0.85, zorder=3)
        for _, row in grp.iterrows():
            ax2.annotate(
                row["Facility"],
                (row["PC1"], row["PC2"]),
                fontsize=7, alpha=0.75,
                xytext=(3, 3), textcoords="offset points",
            )

    ax2.set_xlabel("Principal Component 1", fontsize=12)
    ax2.set_ylabel("Principal Component 2", fontsize=12)
    ax2.set_title(f"Facility Clusters (K={best_k}) — PCA Projection",
                  fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "figure7_facility_clusters.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Figure 8 — Forecast Traces vs Actual (test-period, representative facilities)
# ---------------------------------------------------------------------------

def plot_forecast_traces(
    traces_df: pd.DataFrame,
    risk_df: Optional[pd.DataFrame] = None,
    results_df: Optional[pd.DataFrame] = None,
    select_facilities: Optional[list] = None,
    n_panels: int = 6,
    output_dir: str = "figures",
    show: bool = False,
) -> str:
    """2×3 grid of actual vs predicted emission traces over the 12-month test period.

    Automatically selects ``n_panels`` representative facilities spanning the
    full range of ensemble MAPE performance when ``select_facilities`` is None.

    Args:
        traces_df:          Long-format DataFrame with columns
                            [Facility, Step, Actual, NHITS, XGBoost, BNN, Ensemble].
        risk_df:            Risk assessment DataFrame (optional); used to add
                            risk label to each panel subtitle.
        results_df:         Per-facility accuracy DataFrame (optional); used to
                            annotate each panel with EnsembleMAPE.
        select_facilities:  Explicit list of facility names to display.
                            If None, auto-selects based on MAPE spread.
        n_panels:           Number of panels when auto-selecting (default 6).
        output_dir:         Directory to save the figure.
        show:               Display inline if True.

    Returns:
        Saved file path.
    """
    _apply_style()

    # ── Facility selection ────────────────────────────────────────────────
    all_facs = sorted(traces_df["Facility"].unique())

    if select_facilities is not None:
        chosen = [f for f in select_facilities if f in all_facs]
    elif results_df is not None and "EnsembleMAPE" in results_df.columns:
        # Evenly-spaced sample across sorted EnsembleMAPE range
        sorted_facs = (
            results_df[["Facility", "EnsembleMAPE"]]
            .dropna()
            .sort_values("EnsembleMAPE")["Facility"]
            .tolist()
        )
        n   = min(n_panels, len(sorted_facs))
        idx = [int(round(i * (len(sorted_facs) - 1) / (n - 1))) for i in range(n)]
        chosen = [sorted_facs[i] for i in idx]
    else:
        chosen = all_facs[:n_panels]

    n_chosen = len(chosen)
    ncols    = 3
    nrows    = int(np.ceil(n_chosen / ncols))

    # ── Build risk + MAPE lookup dicts ────────────────────────────────────
    risk_lookup = {}
    if risk_df is not None and "RiskLevelEnsemble" in risk_df.columns:
        risk_lookup = dict(zip(risk_df["Facility"], risk_df["RiskLevelEnsemble"]))

    mape_lookup = {}
    if results_df is not None and "EnsembleMAPE" in results_df.columns:
        mape_lookup = dict(zip(results_df["Facility"], results_df["EnsembleMAPE"]))

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    fig.suptitle(
        r"Test-Period Forecast Traces vs Actual Emissions (tCO$_2$)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    axes_flat = axes.flatten() if n_chosen > 1 else [axes]

    line_specs = [
        ("Actual",   "black",                  2.2,  "-",  None),
        ("NHITS",    MODEL_COLORS["N-HiTS"],   1.5,  "-",  "N-HiTS"),
        ("XGBoost",  MODEL_COLORS["XGBoost"],  1.5,  "-",  "XGBoost"),
        ("BNN",      MODEL_COLORS["BNN"],       1.5,  "-",  "BNN"),
        ("Ensemble", MODEL_COLORS["Ensemble"],  1.8,  "--", "Ensemble"),
    ]

    for idx, fac in enumerate(chosen):
        ax  = axes_flat[idx]
        sub = traces_df[traces_df["Facility"] == fac].sort_values("Step")
        steps = sub["Step"].values

        handles = []
        for col, color, lw, ls, label in line_specs:
            if col not in sub.columns:
                continue
            ln, = ax.plot(steps, sub[col].values,
                          color=color, linewidth=lw, linestyle=ls,
                          label=label or col)
            if label:
                handles.append(ln)

        # Subtitle
        risk  = risk_lookup.get(fac, "")
        mape  = mape_lookup.get(fac, None)
        parts = [fac]
        if mape is not None:
            parts.append(f"Ens.MAPE={mape:.1f}%")
        if risk:
            parts.append(risk)
        ax.set_title("  |  ".join(parts), fontsize=10, fontweight="bold")
        ax.set_xlabel("Test-Period Step (months)", fontsize=9)
        ax.set_ylabel(r"Emissions (tCO$_2$)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xticks(steps[::2] if len(steps) > 6 else steps)
        ax.grid(alpha=0.3)

    # Hide any unused axes
    for idx in range(n_chosen, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend below the grid
    if handles:
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            loc="lower center",
            ncol=len(handles),
            fontsize=10,
            frameon=True,
            bbox_to_anchor=(0.5, -0.03),
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "figure8_forecast_traces.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Figure 9 — SHAP Dependence Plots (3 features × 3 models)
# ---------------------------------------------------------------------------

def plot_shap_dependence(
    shap_raw: dict,
    X_raw: dict,
    features_of_interest: list[str] | None = None,
    shap_summary: Optional[pd.DataFrame] = None,
    output_dir: str = "figures",
    show: bool = True,
) -> str:
    """3×3 SHAP dependence grid: top-3 features (columns) × 3 models (rows).

    Each panel shows scatter of SHAP attribution (tCO₂) vs feature value,
    a dashed-red LOWESS smoother, and the Spearman rank correlation ρ.

    Args:
        shap_raw:             dict mapping model key (``"nhits"``, ``"xgb"``,
                              ``"bnn"``) to a SHAP value DataFrame.
        X_raw:                dict with the same keys — actual feature values.
        features_of_interest: Explicit list of 3 feature names.  When None,
                              the top 3 by average SHAP contribution are taken
                              from ``shap_summary`` (if provided) or computed
                              from mean absolute SHAP across all models.
        shap_summary:         Aggregated SHAP importance table (optional);
                              used to auto-select top-3 features.
        output_dir:           Directory to save the figure.
        show:                 Display inline if True.

    Returns:
        Saved file path.
    """
    from scipy.stats import spearmanr
    from statsmodels.nonparametric.smoothers_lowess import lowess

    _apply_style()

    # ── Auto-select top-3 features ──────────────────────────────────────
    # Determine which features are actually available in the raw SHAP data
    available_feats: set = set()
    for mk in ["nhits", "xgb", "bnn"]:
        df = shap_raw.get(mk)
        if df is not None and not df.empty:
            available_feats.update(df.select_dtypes(include="number").columns.tolist())

    if features_of_interest is None:
        candidates: list = []
        if shap_summary is not None and "AvgContribution" in shap_summary.columns:
            candidates = shap_summary.sort_values("AvgContribution", ascending=False)["Feature"].tolist()
        elif shap_summary is not None and "Feature" in shap_summary.columns:
            ss = shap_summary.copy()
            contrib_cols = [c for c in ss.columns if c.endswith("Contribution")]
            if contrib_cols:
                ss["_avg"] = ss[contrib_cols].mean(axis=1)
                candidates = ss.sort_values("_avg", ascending=False)["Feature"].tolist()
        if not candidates:
            # Fallback: mean |SHAP| pooled across models
            mean_abs: dict = {}
            for mk in ["nhits", "xgb", "bnn"]:
                df = shap_raw.get(mk)
                if df is not None and not df.empty:
                    for col in df.select_dtypes(include="number").columns:
                        mean_abs[col] = mean_abs.get(col, 0.0) + df[col].abs().mean()
            candidates = sorted(mean_abs, key=mean_abs.get, reverse=True)
        # Keep only features present in raw SHAP, take top 3
        features_of_interest = [f for f in candidates if f in available_feats][:3]

    features_of_interest = list(features_of_interest)[:3]

    # Per spec: N-HiTS green, XGBoost orange, BNN purple
    model_keys   = ["nhits",    "xgb",      "bnn"]
    model_labels = ["N-HiTS",   "XGBoost",  "BNN"]
    model_colors = ["#2ca02c",  "#ff7f0e",  "#9467bd"]

    fig, axes = plt.subplots(
        nrows=len(model_keys),
        ncols=len(features_of_interest),
        figsize=(5 * len(features_of_interest), 4 * len(model_keys)),
        sharex="col",
    )

    for row, (mk, ml, mc) in enumerate(zip(model_keys, model_labels, model_colors)):
        shap_df = shap_raw.get(mk)
        x_df    = X_raw.get(mk)

        if shap_df is None or x_df is None or shap_df.empty:
            for col in range(len(features_of_interest)):
                axes[row, col].set_visible(False)
            continue

        for col, feat in enumerate(features_of_interest):
            ax = axes[row, col]

            if feat not in shap_df.columns or feat not in x_df.columns:
                ax.set_visible(False)
                continue

            x_feat   = x_df[feat].values
            shap_val = shap_df[feat].values

            # Scatter
            ax.scatter(x_feat, shap_val, color=mc, alpha=0.45, s=18,
                       linewidths=0, zorder=2)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

            # LOWESS smoother (dashed red)
            try:
                sort_idx = np.argsort(x_feat)
                smoothed = lowess(shap_val[sort_idx], x_feat[sort_idx],
                                  frac=0.4, return_sorted=True)
                ax.plot(smoothed[:, 0], smoothed[:, 1],
                        color="#e74c3c", linewidth=2, linestyle="--", zorder=4)
            except Exception:
                pass

            # Spearman ρ + mean SHAP μ annotation
            try:
                rho, pval = spearmanr(x_feat, shap_val)
                sig  = "*" if pval < 0.05 else ""
                mu   = shap_val.mean()
                ax.text(0.97, 0.97,
                        fr"$\rho={rho:.2f}{sig}$" + "\n" + fr"$\mu={mu:+.3f}$",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            except Exception:
                pass

            if col == 0:
                ax.set_ylabel(fr"{ml}" + "\n" + r"SHAP (tCO$_2$)", fontsize=10)
            if row == 0:
                ax.set_title(feat.replace("_", " "), fontsize=11, fontweight="bold")
            if row == len(model_keys) - 1:
                ax.set_xlabel(feat.replace("_", " "), fontsize=10)

            ax.grid(alpha=0.25)

    fig.suptitle(
        r"SHAP Dependence — Top 3 Features by Avg Attribution (tCO$_2$)"
        "  |  Dashed red = LOWESS  |  $\\rho$ = Spearman",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    path = os.path.join(output_dir, "figure9_shap_dependence.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Figure 10 — Variance Decomposition (all facilities)
# ---------------------------------------------------------------------------

def plot_variance_decomposition(
    decomp_df: pd.DataFrame,
    output_dir: str = "figures",
    show: bool = True,
) -> str:
    """Two-panel figure: absolute variance (top) and % share (bottom) per facility.

    Args:
        decomp_df:  DataFrame with columns
                    ``Facility, within_model_var, between_model_var,
                    total_var, within_pct, between_pct``.
                    (Output of the uncertainty decomposition calculation.)
        output_dir: Directory to save the figure.
        show:       Display inline if True.

    Returns:
        Saved file path.
    """
    _apply_style()

    df = decomp_df.copy().sort_values("between_pct", ascending=False).reset_index(drop=True)
    x  = np.arange(len(df))
    w  = 0.6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={"height_ratios": [1.4, 1]})

    # ── Panel 1: absolute variance ──────────────────────────────────────────
    ax1.bar(x, df["within_model_var"],  w, label="Within-model (aleatoric)",
            color="#1f77b4", alpha=0.85)
    ax1.bar(x, df["between_model_var"], w, bottom=df["within_model_var"],
            label="Between-model (epistemic)", color="#ff7f0e", alpha=0.85)
    ax1.set_ylabel("Variance (tCO2)^2", fontsize=11)
    ax1.set_title(
        "Ensemble Uncertainty Decomposition — All Facilities\n"
        "(sorted by epistemic share, high → low)",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: percentage stacked bar ────────────────────────────────────
    ax2.bar(x, df["within_pct"],  w, label="Within-model %",  color="#1f77b4", alpha=0.85)
    ax2.bar(x, df["between_pct"], w, bottom=df["within_pct"],
            label="Between-model %", color="#ff7f0e", alpha=0.85)
    ax2.axhline(50, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Share (%)", fontsize=11)
    ax2.set_xlabel("Facility", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Facility"], rotation=45, ha="right", fontsize=9)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "figure10_variance_decomposition.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Figure 11 – Horizon crossover projection
# ---------------------------------------------------------------------------

def plot_horizon_crossover(
    horizon_summary: pd.DataFrame,
    output_dir: str = "figures",
    show: bool = False,
) -> str:
    """Plot observed per-step MAPE for XGBoost and N-HiTS with linear
    extrapolation to their projected crossover horizon.

    The chart visualises the key finding that XGBoost starts lower but
    degrades faster, while N-HiTS starts higher but has a shallower slope,
    with both trends intersecting at the projected crossover horizon h*.

    Args:
        horizon_summary: DataFrame with columns [Model, h, MAPE] —
                         mean MAPE per model per step (e.g. from
                         ``results/horizon_summary.csv``).
        output_dir:      Directory for the saved figure.
        show:            Display the figure inline if True.

    Returns:
        Path to the saved PNG file.
    """
    _apply_style()

    models = ["XGBoost", "N-HiTS"]
    colors = {m: MODEL_COLORS[m] for m in models}

    # ── Fit linear trends over observed range h = 1…20 ──────────────────
    fits = {}
    for m in models:
        sub = horizon_summary[horizon_summary["Model"] == m].sort_values("h")
        coef = np.polyfit(sub["h"], sub["MAPE"], 1)   # [slope, intercept]
        fits[m] = coef

    # ── Crossover: XGBoost(h) == N-HiTS(h) ──────────────────────────────
    sx, ix = fits["XGBoost"]
    sn, in_ = fits["N-HiTS"]
    h_cross = (in_ - ix) / (sx - sn)
    mape_cross = ix + sx * h_cross

    # ── Plot range: 1 to crossover + 15 months ──────────────────────────
    h_obs = np.arange(1, 21)
    h_ext = np.linspace(1, h_cross + 15, 300)

    fig, ax = plt.subplots(figsize=(9, 5))

    for m in models:
        sub = horizon_summary[horizon_summary["Model"] == m].sort_values("h")
        slope, intercept = fits[m]

        # Observed scatter + connecting line
        ax.plot(sub["h"], sub["MAPE"],
                color=colors[m], linewidth=1.8, zorder=3)
        ax.scatter(sub["h"], sub["MAPE"],
                   color=colors[m], s=35, zorder=4)

        # Extrapolated trend (full range, dashed)
        ax.plot(h_ext, slope * h_ext + intercept,
                color=colors[m], linewidth=1.4, linestyle="--", alpha=0.65,
                label=f"{m}  (slope {slope:+.2f} pp/month)")

    # ── Observed-data boundary marker ────────────────────────────────────
    ax.axvline(20, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(20.4, ax.get_ylim()[0] + 2, "Observed\nlimit (h=20)",
            fontsize=8.5, color="grey", va="bottom")

    # ── Crossover annotation ──────────────────────────────────────────────
    ax.axvline(h_cross, color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.scatter([h_cross], [mape_cross],
               color="#c0392b", s=90, zorder=5, marker="*")
    ax.annotate(
        f"Crossover\nh* ≈ {h_cross:.0f} months\n({h_cross/12:.1f} yrs)\nMAPE ≈ {mape_cross:.0f}%",
        xy=(h_cross, mape_cross),
        xytext=(h_cross + 4, mape_cross + 8),
        fontsize=9,
        color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    )

    # ── Region shading: XGBoost advantaged / N-HiTS advantaged ──────────
    h_fill_obs  = h_obs[h_obs <= h_cross]
    h_fill_ext  = h_ext[h_ext > h_cross]

    ax.fill_between(
        np.concatenate([h_fill_obs, h_fill_ext[:1]]),
        np.polyval(fits["XGBoost"], np.concatenate([h_fill_obs, h_fill_ext[:1]])),
        np.polyval(fits["N-HiTS"],  np.concatenate([h_fill_obs, h_fill_ext[:1]])),
        alpha=0.06, color=colors["XGBoost"],
        label="XGBoost advantage zone",
    )
    ax.fill_between(
        h_fill_ext,
        np.polyval(fits["XGBoost"], h_fill_ext),
        np.polyval(fits["N-HiTS"],  h_fill_ext),
        alpha=0.06, color=colors["N-HiTS"],
        label="N-HiTS advantage zone",
    )

    ax.set_xlabel("Forecast horizon h (months)")
    ax.set_ylabel(r"Mean MAPE (%)")
    ax.set_title(
        "Projected Crossover: XGBoost vs N-HiTS Degradation Slopes",
        fontsize=13, pad=12,
    )

    # Force y-axis to start at 0
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=1)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax.grid(axis="both", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "figure11_horizon_crossover.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Figure 12 — Risk Tier Summary (donut + horizontal bar)
# ---------------------------------------------------------------------------

def plot_risk_tier_summary(
    risk_df: pd.DataFrame,
    output_dir: str = "figures",
    show: bool = False,
) -> str:
    """Two-panel figure: donut chart of facility counts per risk tier (left)
    and a horizontal stacked bar showing tier breakdown per region (right).

    Args:
        risk_df:    Risk assessment DataFrame with columns ``RiskLevelEnsemble``
                    and ``Facility``.
        output_dir: Directory to save the figure.
        show:       Display inline if True.

    Returns:
        Saved file path.
    """
    _apply_style()

    tier_order  = ["Critical Risk", "High Risk", "Medium Risk", "Low Risk"]
    counts      = risk_df["RiskLevelEnsemble"].value_counts().reindex(tier_order, fill_value=0)
    colors_list = [RISK_COLORS[t] for t in tier_order]
    total       = counts.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("2030 Compliance Risk Tier Distribution (30 Facilities)",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: donut chart ─────────────────────────────────────────────
    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=None,
        colors=colors_list,
        autopct=lambda p: f"{p:.0f}%\n({int(round(p * total / 100))})",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax1.legend(
        wedges, [f"{t}  (n={counts[t]})" for t in tier_order],
        loc="lower center", bbox_to_anchor=(0.5, -0.18),
        ncol=2, fontsize=9, frameon=False,
    )
    ax1.set_title("Overall Risk Distribution", fontsize=12, fontweight="bold", pad=12)

    # ── Panel 2: stacked horizontal bar per cluster (if Cluster col exists)
    # Fall back to a simple sorted bar of individual facility probabilities
    if "Cluster" in risk_df.columns:
        groups   = sorted(risk_df["Cluster"].unique())
        y_pos    = np.arange(len(groups))
        left     = np.zeros(len(groups))
        for tier, color in zip(tier_order, colors_list):
            widths = [
                (risk_df[risk_df["Cluster"] == g]["RiskLevelEnsemble"] == tier).sum()
                for g in groups
            ]
            ax2.barh(y_pos, widths, left=left, color=color,
                     label=tier, height=0.55, edgecolor="white")
            left += np.array(widths)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(groups, fontsize=10)
        ax2.set_xlabel("Number of Facilities", fontsize=11)
        ax2.set_title("Risk Tier by Emission Cluster", fontsize=12, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=9, frameon=True)
        ax2.grid(axis="x", alpha=0.3)
    else:
        # Simple bar: compliance probability per facility, coloured by tier
        prob_sorted = risk_df.sort_values("ProbMeetTargetEnsemble").reset_index(drop=True)
        bar_colors  = prob_sorted["RiskLevelEnsemble"].map(RISK_COLORS).fillna("#bdc3c7")
        ax2.barh(range(len(prob_sorted)),
                 prob_sorted["ProbMeetTargetEnsemble"] * 100,
                 color=bar_colors, edgecolor="white")
        ax2.axvline(50, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax2.set_yticks(range(len(prob_sorted)))
        ax2.set_yticklabels(prob_sorted["Facility"], fontsize=7)
        ax2.set_xlabel("Ensemble Compliance Probability (%)", fontsize=11)
        ax2.set_title("Compliance Probability by Facility", fontsize=12, fontweight="bold")
        # Compact legend
        patches = [mpatches.Patch(color=RISK_COLORS[t], label=t)
                   for t in tier_order if RISK_COLORS.get(t)]
        ax2.legend(handles=patches, loc="lower right", fontsize=9, frameon=True)
        ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "figure12_risk_tier_summary.png")
    _save(fig, path, show)
    return path


# ---------------------------------------------------------------------------
# Convenience: render all figures at once
# ---------------------------------------------------------------------------

def plot_all(
    results_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    shap_summary: pd.DataFrame,
    output_dir: str = "figures",
    show: bool = False,
    shap_raw: dict | None = None,
    X_raw: dict | None = None,
    decomp_df: pd.DataFrame | None = None,
    horizon_summary: pd.DataFrame | None = None,
    traces_df: pd.DataFrame | None = None,
    cluster_df: pd.DataFrame | None = None,
) -> list[str]:
    """Render and save all publication figures plus the dashboard.

    Args:
        results_df:       Per-facility model results.
        risk_df:          Compliance probability DataFrame.
        comparison_df:    Model comparison summary.
        shap_summary:     Aggregated SHAP importance table.
        output_dir:       Directory for all saved figures.
        show:             Display each figure inline if True.
        shap_raw:         Raw SHAP value DataFrames keyed by model
                          (``"nhits"``, ``"xgb"``, ``"bnn"``).  When provided,
                          Figure 9 (dependence plots) is also generated.
        X_raw:            Corresponding feature value DataFrames (same keys).
        decomp_df:        Uncertainty variance decomposition DataFrame (output
                          of the Eq. 8 computation).  When provided, Figure 10
                          is generated.
        horizon_summary:  Per-model, per-step mean MAPE DataFrame with columns
                          [Model, h, MAPE].  When provided, Figure 11
                          (crossover projection) is generated.
        traces_df:        Long-format test-period predictions DataFrame with
                          columns [Facility, Step, Actual, NHITS, XGBoost,
                          BNN, Ensemble].  When provided, Figure 8
                          (forecast traces) is generated.
        cluster_df:       Cluster membership DataFrame with columns
                          [Facility, Cluster].  When provided, Figure 12
                          right panel shows tier breakdown by cluster.

    Returns:
        List of saved file paths.
    """
    paths = []
    os.makedirs(output_dir, exist_ok=True)

    paths.append(plot_stakeholder_dashboard(risk_df, comparison_df, output_dir, show))
    paths.append(plot_mape_distribution(results_df, output_dir, show))
    paths.append(plot_uncertainty_decomposition(risk_df, results_df, None, output_dir, show))

    if shap_summary is not None and len(shap_summary) > 0:
        paths.append(plot_shap_comparison(shap_summary, output_dir=output_dir, show=show))

    paths.append(plot_probability_distribution(risk_df, output_dir, show))
    paths.append(plot_model_agreement(risk_df, output_dir=output_dir, show=show))
    paths.append(plot_facility_accuracy(results_df, output_dir, show))

    if traces_df is not None and len(traces_df) > 0:
        paths.append(plot_forecast_traces(
            traces_df, risk_df=risk_df, results_df=results_df,
            output_dir=output_dir, show=show,
        ))

    if shap_raw is not None and X_raw is not None:
        paths.append(plot_shap_dependence(
            shap_raw, X_raw,
            shap_summary=shap_summary,
            output_dir=output_dir, show=show,
        ))

    if decomp_df is not None and len(decomp_df) > 0:
        paths.append(plot_variance_decomposition(decomp_df, output_dir=output_dir, show=show))

    if horizon_summary is not None and len(horizon_summary) > 0:
        paths.append(plot_horizon_crossover(horizon_summary, output_dir=output_dir, show=show))

    # Figure 12 — always generated (cluster_df enriches right panel if provided)
    risk_fig12 = risk_df.copy()
    if cluster_df is not None and len(cluster_df) > 0:
        risk_fig12 = risk_fig12.merge(
            cluster_df[["Facility", "Cluster"]], on="Facility", how="left"
        )
    paths.append(plot_risk_tier_summary(risk_fig12, output_dir=output_dir, show=show))

    logger.info("All figures saved to %s", output_dir)
    return paths
