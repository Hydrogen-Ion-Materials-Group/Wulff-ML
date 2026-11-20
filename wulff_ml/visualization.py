"""
Shared plotting helpers.

Every function saves figures under the appropriate figures/ subdirectory
(configured via wulff_ml/config.py) and returns the saved path so scripts stay thin.

Units
-----
All y-axes and printed labels use J/m² (SI).  The raw model targets are in
eV/Å²; conversion is applied at the plotting boundary via wulff_ml.units.to_jm2().
The eV/Å² value is shown as a secondary reference where space permits.

save_figure() — core save primitive
------------------------------------
Writes three files per call:

    <name>.png      labelled raster  (dpi=600 default)
    <name>.pdf      labelled vector  (Illustrator / LaTeX)
    <name>.svg      labelled vector  (web / Inkscape)

When export_raw=True (default) it also writes:

    <name>_raw.png  identical geometry, ALL text removed

The raw copy is produced from a deep-copy of the figure so the original
is never mutated.  "Text removed" means: figure suptitle, fig.text()
captions, per-axes title/xlabel/ylabel, tick labels (numbers only — tick
marks kept), legend, and colorbar tick labels + label.  All data artists
(curves, scatter, bars, patches, heatmaps, …) and spines are untouched.
"""
import copy
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

from wulff_ml import config
from wulff_ml.units import to_jm2, UNIT_LABEL, UNIT_LABEL_EV, UNIT_SHORT, EV_ANG2_TO_JM2


# ==============================================================================
# Internal helpers
# ==============================================================================

def _raw_path(path: str) -> str:
    """foo/bar.png  ->  foo/bar_raw.png"""
    base, ext = os.path.splitext(path)
    return f"{base}_raw{ext}"


def _strip_text(fig) -> None:
    """
    Remove every piece of text from *fig* in-place.

    Cleared  : figure suptitle, fig.text() captions, per-axes title /
               xlabel / ylabel, x-tick labels, y-tick labels, legend,
               colorbar tick labels and colorbar label.
    Kept     : all data artists, spines, tick marks.

    Implementation note
    -------------------
    We never call fig.suptitle("") before iterating fig.texts — doing so
    causes matplotlib to internally touch the same Text object we are about
    to call .remove() on, triggering "list.remove(x): x not in list".
    Instead we blank every Text object in-place (set_text + set_visible)
    rather than removing it from the figure's list.  The same defensive
    approach is used for axis legends.
    """
    # Blank all figure-level text objects (suptitle + any fig.text() calls).
    # Use set_text/set_visible instead of .remove() to avoid the
    # "x not in list" crash from inconsistent matplotlib internal state.
    for txt in list(getattr(fig, "texts", [])):
        try:
            txt.set_text("")
            txt.set_visible(False)
        except Exception:
            pass

    for ax in fig.get_axes():
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Blank tick labels (leave tick marks themselves intact)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Axes-level text objects — bar value annotations, data labels, etc.
        # These live in ax.texts and are NOT touched by set_xticklabels/ylabel.
        for txt in list(ax.texts):
            try:
                txt.set_text("")
                txt.set_visible(False)
            except Exception:
                pass

        # Legend — remove() can also fail if the legend was already detached;
        # guard with try/except and fall back to set_visible(False)
        legend = ax.get_legend()
        if legend is not None:
            try:
                legend.remove()
            except Exception:
                try:
                    legend.set_visible(False)
                except Exception:
                    pass

        # Colorbar axes (matplotlib marks them with label "<colorbar>")
        if ax.get_label() == "<colorbar>":
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")


# ==============================================================================
# Core save primitive
# ==============================================================================

def save_figure(fig_or_plt, path: str, dpi: int = 600,
                export_raw: bool = True) -> str:
    """
    Save a matplotlib figure in .png / .pdf / .svg, plus an optional
    text-free *_raw.png companion.

    Parameters
    ----------
    fig_or_plt : Figure or the plt module
    path       : destination for the primary .png
    dpi        : raster resolution (vector formats ignore this)
    export_raw : if True (default), write <name>_raw.png with all text stripped

    Returns
    -------
    str — path to the primary .png
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig = fig_or_plt.gcf() if hasattr(fig_or_plt, "gcf") else fig_or_plt
    base, _ = os.path.splitext(path)

    # Labelled outputs
    fig.savefig(path,          dpi=dpi, bbox_inches="tight")
    fig.savefig(base + ".pdf",          bbox_inches="tight")
    fig.savefig(base + ".svg",          bbox_inches="tight")
    print(f"  Saved: {path}")
    print(f"  Saved: {base}.pdf")
    print(f"  Saved: {base}.svg")

    # Text-free raw companion
    if export_raw:
        fig_raw = copy.deepcopy(fig)
        _strip_text(fig_raw)
        raw = _raw_path(path)
        fig_raw.savefig(raw, dpi=dpi, bbox_inches="tight")
        plt.close(fig_raw)
        print(f"  Saved: {raw}")

    return path


# Backwards-compatible alias
_savefig = save_figure


# ==============================================================================
# Named plot helpers
# ==============================================================================

def feature_importance_plot(model, feature_cols, out_path, top_n=20,
                             title="Top Features", export_raw=True):
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    imp.tail(top_n).plot(kind="barh", figsize=(10, 8), title=title)
    plt.tight_layout()
    save_figure(plt, out_path, export_raw=export_raw)
    plt.close()
    return imp


def parity_plot(y_true, y_pred, out_path, title="Test set parity", export_raw=True):
    """Parity plot in J/m².  eV/Å² reference shown in the subtitle."""
    y_true_jm2 = to_jm2(y_true)
    y_pred_jm2 = to_jm2(y_pred)

    r2 = r2_score(y_true, y_pred)          # R² is dimensionless — computed in native units
    lo, hi = 0, 5

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_jm2, y_pred_jm2, alpha=0.3, s=5)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"True γ ({UNIT_SHORT})")
    ax.set_ylabel(f"Predicted γ ({UNIT_SHORT})")
    ax.set_title(f"{title}  R²={r2:.3f}")
    ax.legend()
    # Conversion reference in the figure margin
    fig.text(0.99, 0.01,
             f"1 eV/Å² = {EV_ANG2_TO_JM2} J/m²",
             ha="right", va="bottom", fontsize=7, color="grey",
             transform=fig.transFigure)
    plt.tight_layout()
    save_figure(fig, out_path, export_raw=export_raw)
    plt.close(fig)


def model_comparison_plot(results_df: pd.DataFrame, out_path, export_raw=True):
    """Bar chart of MAE (J/m²) and R² across models.
    results_df must have columns Model, MAE, R2 — MAE is expected in eV/Å².
    """
    df = results_df.copy()
    df["MAE_jm2"] = to_jm2(df["MAE"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(df["Model"], df["MAE_jm2"], color="steelblue", edgecolor="black")
    axes[0].set_xlabel(f"Test MAE ({UNIT_SHORT})")
    axes[0].set_title(f"Test MAE — All Models (lower is better)")
    axes[0].axvline(df["MAE_jm2"].min(), color="red", linestyle="--", linewidth=1, label="Best MAE")
    axes[0].legend()

    df_r2 = df.sort_values("R2", ascending=True)
    axes[1].barh(df_r2["Model"], df_r2["R2"], color="darkorange", edgecolor="black")
    axes[1].set_xlabel("Test R²")
    axes[1].set_title("Test R² — All Models (higher is better)")
    axes[1].axvline(df_r2["R2"].max(), color="red", linestyle="--", linewidth=1, label="Best R²")
    axes[1].legend()

    fig.text(0.99, 0.01,
             f"1 eV/Å² = {EV_ANG2_TO_JM2} J/m²",
             ha="right", va="bottom", fontsize=7, color="grey",
             transform=fig.transFigure)
    plt.tight_layout()
    save_figure(fig, out_path, export_raw=export_raw)
    plt.close(fig)


def cv_results_plot(cv_df: pd.DataFrame, out_path, n_folds=5, export_raw=True):
    """CV results — MAE shown in J/m²."""
    import numpy as np
    from matplotlib.ticker import MultipleLocator

    df = cv_df.copy()
    df["mae_jm2"] = to_jm2(df["mae"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # MAE panel
    ax = axes[0]
    vals = df["mae_jm2"].values
    ax.bar(df["fold"], vals, color="steelblue", alpha=0.8, edgecolor="navy")
    ax.axhline(vals.mean(), color="crimson", linestyle="--", lw=1.5,
               label=f"Mean = {vals.mean():.4f}")
    ax.fill_between([0.5, n_folds + 0.5],
                    vals.mean() - vals.std(), vals.mean() + vals.std(),
                    alpha=0.15, color="crimson", label=f"±1 std = {vals.std():.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel(f"MAE ({UNIT_SHORT})")
    ax.set_title(f"5-Fold Group CV — MAE")
    ax.set_xticks(df["fold"])
    # Snap ylim to the next 0.04 boundary so MultipleLocator ticks are never clipped
    y_top = (np.ceil(vals.max() / 0.04) + 1) * 0.04
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # R² panel (dimensionless — no conversion)
    ax = axes[1]
    r2_vals = df["r2"].values
    ax.bar(df["fold"], r2_vals, color="steelblue", alpha=0.8, edgecolor="navy")
    ax.axhline(r2_vals.mean(), color="crimson", linestyle="--", lw=1.5,
               label=f"Mean = {r2_vals.mean():.4f}")
    ax.fill_between([0.5, n_folds + 0.5],
                    r2_vals.mean() - r2_vals.std(), r2_vals.mean() + r2_vals.std(),
                    alpha=0.15, color="crimson", label=f"±1 std = {r2_vals.std():.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("R²")
    ax.set_title("5-Fold Group CV — R²")
    ax.set_xticks(df["fold"])
    ax.set_ylim(bottom=0, top=1.0)                     # always show full 0–1.0 range
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("5-Fold Group Cross-Validation (mp_id grouped)", fontsize=12)
    fig.text(0.99, 0.01,
             f"1 eV/Å² = {EV_ANG2_TO_JM2} J/m²",
             ha="right", va="bottom", fontsize=7, color="grey",
             transform=fig.transFigure)
    plt.tight_layout()
    save_figure(fig, out_path, export_raw=export_raw)
    plt.close(fig)


def ablation_curve_plot(ablation_df: pd.DataFrame, out_path, x_col="frac",
                         title="Data Ablation", export_raw=True):
    """Ablation curves — MAE axis in J/m²."""
    df = ablation_df.copy()
    for col in ("train_mae", "val_mae", "test_mae"):
        df[f"{col}_jm2"] = to_jm2(df[col])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(df[x_col] * 100, df["train_mae_jm2"], "o-", color="steelblue",  label="Train MAE")
    ax.plot(df[x_col] * 100, df["val_mae_jm2"],   "s--", color="darkorange", label="Val MAE")
    ax.plot(df[x_col] * 100, df["test_mae_jm2"],  "^-", color="green",      label="Test MAE")
    ax.set_xlabel("Training data fraction (%)")
    ax.set_ylabel(f"MAE ({UNIT_SHORT})")
    ax.set_title(f"{title} — MAE vs Training Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df[x_col] * 100, df["test_r2"], "o-", color="purple")
    ax.set_xlabel("Training data fraction (%)")
    ax.set_ylabel("Test R²")
    ax.set_title(f"{title} — R² vs Training Size")
    ax.grid(True, alpha=0.3)

    fig.text(0.99, 0.01,
             f"1 eV/Å² = {EV_ANG2_TO_JM2} J/m²",
             ha="right", va="bottom", fontsize=7, color="grey",
             transform=fig.transFigure)
    plt.tight_layout()
    save_figure(fig, out_path, export_raw=export_raw)
    plt.close(fig)


def group_ablation_plot(ga_df: pd.DataFrame, groups: list, ref_mae: float, best_row,
                         rank_method: str, out_path, export_raw=True):
    """4-panel group-wise forward-selection figure — MAE axes in J/m²."""
    ga = ga_df.copy()
    ga["test_mae_jm2"]  = to_jm2(ga["test_mae"])
    ga["val_mae_jm2"]   = to_jm2(ga["val_mae"])
    ga["delta_mae_jm2"] = to_jm2(ga["delta_mae"])
    ref_jm2 = to_jm2(ref_mae)
    best_test_jm2 = to_jm2(float(best_row["test_mae"]))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    ax.plot(ga["step"], ga["test_mae_jm2"], "o-", color="crimson",   lw=1.5, ms=6, label="Test MAE")
    ax.plot(ga["step"], ga["val_mae_jm2"],  "s--", color="steelblue", lw=1.2, ms=4, alpha=0.7, label="Val MAE")
    ax.axhline(ref_jm2, color="black", linestyle="--", lw=1.2,
               label=f"Full-feature baseline ({ref_jm2:.2f} J/m²)")
    ax.axvline(best_row["step"], color="green", linestyle="--", lw=1.5, alpha=0.8,
               label=f"Best: step {int(best_row['step'])} ({int(best_row['n_features'])} feats)")
    ax.set_xticks(ga["step"].tolist())
    ax.set_xticklabels([f"[{r['group_id']}]" for _, r in ga.iterrows()], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Group added (in SHAP-ranked order)")
    ax.set_ylabel(f"MAE ({UNIT_SHORT})")
    ax.set_title(f"Group Forward Selection — MAE Curve ({rank_method})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ga["n_features"], ga["test_mae_jm2"], "o-", color="crimson", lw=1.5, ms=5, label="Test MAE")
    ax.axhline(ref_jm2, color="black", linestyle="--", lw=1.2, label="Full-feature baseline")
    ax.axvline(best_row["n_features"], color="green", linestyle="--", lw=1.5, alpha=0.8,
               label=f"Best: {int(best_row['n_features'])} features")
    for _, row in ga.iterrows():
        ax.annotate(row["group_id"], (row["n_features"], to_jm2(row["test_mae"])),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, color="gray")
    ax.set_xlabel("Cumulative number of features")
    ax.set_ylabel(f"MAE ({UNIT_SHORT})")
    ax.set_title("Cost-Accuracy Trade-off (features vs MAE)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    delta_vals = ga["delta_mae_jm2"].iloc[1:].values
    gids   = [f"[{r['group_id']}]" for _, r in ga.iloc[1:].iterrows()]
    colors = ["green" if d < 0 else "red" for d in delta_vals]
    ax.bar(range(len(delta_vals)), delta_vals, color=colors, alpha=0.75)
    ax.set_xticks(range(len(delta_vals)))
    ax.set_xticklabels(gids, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Group added")
    ax.set_ylabel(f"ΔMAE ({UNIT_SHORT}, negative = improvement)")
    ax.set_title("Per-Group Impact on Test MAE (green=helps, red=hurts)")
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1, 1]
    sorted_groups = sorted(groups, key=lambda g: g["group_shap"], reverse=True)
    glabels = [f"[{g['id']}] {g['label']}" for g in sorted_groups]
    gscores = [g["group_shap"] for g in sorted_groups]
    ax.barh(range(len(glabels)), gscores, color="steelblue", alpha=0.75)
    ax.set_yticks(range(len(glabels)))
    ax.set_yticklabels(glabels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean {rank_method} score (per feature in group)")
    ax.set_title(f"Group Importance ({rank_method})")
    ax.grid(True, alpha=0.2, axis="x")

    plt.suptitle("Group-Wise Feature Ablation", fontsize=13, y=1.01)
    fig.text(0.99, 0.01,
             f"1 eV/Å² = {EV_ANG2_TO_JM2} J/m²",
             ha="right", va="bottom", fontsize=7, color="grey",
             transform=fig.transFigure)
    plt.tight_layout()
    save_figure(fig, out_path, export_raw=export_raw)
    plt.close(fig)
