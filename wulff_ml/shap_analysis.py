"""
SHAP interpretability analysis for the trained gamma model.

  - compute_shap_values()       TreeExplainer values for X_test
  - plot_beeswarm()              full-feature global beeswarm (top N)
  - plot_filtered_beeswarm()     curated/renamed subset beeswarm (headline figure)
  - plot_all_dependence_plots()  per-feature SHAP dependence scatter, ranked by importance

Units
-----
SHAP values are in the model's native eV/Å².  Axis labels and titles display
J/m² by multiplying SHAP values by EV_ANG2_TO_JM2 before plotting.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wulff_ml import config
from wulff_ml.units import to_jm2, UNIT_SHORT, EV_ANG2_TO_JM2
from wulff_ml.visualization import save_figure
from wulff_ml.cache import cached


@cached("compute_shap_values")
def compute_shap_values(model, X_test, feature_cols):
    import shap

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    shap_values.feature_names = feature_cols
    return shap_values


def _convert_shap_to_jm2(shap_values):
    """Return a copy of a shap.Explanation with values scaled to J/m²."""
    import shap
    import copy

    sv = copy.deepcopy(shap_values)
    sv.values    = sv.values    * EV_ANG2_TO_JM2
    sv.base_values = sv.base_values * EV_ANG2_TO_JM2
    if hasattr(sv, "data"):
        pass  # feature values stay in their native units (not surface energy)
    return sv


def plot_beeswarm(shap_values, out_path, max_display=30,
                   title="SHAP Beeswarm Plot: Physics-Informed Feature Drivers"):
    import shap

    sv_jm2 = _convert_shap_to_jm2(shap_values)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"]   = 11

    plt.figure(figsize=(11, 7))
    plt.title(title, fontsize=14, pad=15)
    shap.plots.beeswarm(sv_jm2, max_display=max_display, show=False)
    plt.xlabel(f"SHAP Value — Impact on Surface Energy γ ({UNIT_SHORT})", fontsize=12)
    plt.tight_layout()
    save_figure(plt, out_path, dpi=600)
    plt.show()
    plt.close()


def plot_filtered_beeswarm(shap_values, feature_cols, out_path,
                            rename_dict: dict = None,
                            title="SHAP Beeswarm Plot: Key Physical Descriptors"):
    """Curated 11-feature headline beeswarm with publication-ready labels.
    SHAP axis is in J/m²."""
    import shap

    rename_dict = rename_dict or config.SHAP_EXPLICIT_RENAME

    selected_features = [f for f in rename_dict.keys() if f in feature_cols]
    feature_indices   = [feature_cols.index(f) for f in selected_features]
    sv_filtered = shap_values[:, feature_indices]
    sv_filtered.feature_names = [rename_dict[f] for f in selected_features]

    sv_jm2 = _convert_shap_to_jm2(sv_filtered)

    print(f"Filtered out {len(feature_cols) - len(selected_features)} descriptors; "
          f"displaying {len(selected_features)} selected core features.")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"]   = 11

    plt.figure(figsize=(10, 6.5))
    plt.title(title, fontsize=14, pad=15)
    shap.plots.beeswarm(sv_jm2, max_display=len(selected_features), show=False)
    plt.xlabel(f"SHAP Value — Impact on Surface Energy γ ({UNIT_SHORT})", fontsize=12)
    plt.tight_layout()
    save_figure(plt, out_path, dpi=600)
    plt.show()
    plt.close()


def plot_all_dependence_plots(model, shap_values, feature_cols, out_dir):
    """Per-feature SHAP dependence scatter plots, ranked by XGBoost gain.
    SHAP y-axis is in J/m²."""
    import shap

    os.makedirs(out_dir, exist_ok=True)
    importance_dict = model.get_booster().get_score(importance_type="gain")
    ranked = sorted(
        [(feat, importance_dict.get(feat, 0.0)) for feat in feature_cols],
        key=lambda x: x[1], reverse=True,
    )

    sv_jm2 = _convert_shap_to_jm2(shap_values)

    total       = len(ranked)
    results_log = []
    print(f"Generating individual SHAP dependence plots for ALL {total} features...")

    for rank, (feature, importance) in enumerate(ranked, start=1):
        if feature not in feature_cols:
            results_log.append((rank, feature, importance, "SKIPPED"))
            continue

        safe_name = (feature.replace(" ", "_").replace("/", "_")
                     .replace("\\", "_").replace("(", "").replace(")", ""))
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            shap.plots.scatter(sv_jm2[:, feature], color=sv_jm2[:, feature], ax=ax, show=False)
            ax.set_title(f"[Rank #{rank}/{total}  |  Importance: {importance:.4f}]\n"
                         f"SHAP Dependence: {feature}", fontsize=11, pad=10)
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel(f"SHAP Value for {feature} ({UNIT_SHORT})", fontsize=10)
            plt.tight_layout()
            dep_path = os.path.join(out_dir, f"shap_dep_{rank:03d}_{safe_name}.png")
            save_figure(fig, dep_path, dpi=600)
            plt.close(fig)
            results_log.append((rank, feature, importance, f"OK -> {os.path.basename(dep_path)}"))
        except Exception as e:
            plt.close(fig)
            results_log.append((rank, feature, importance, f"ERROR: {e}"))

    ok  = sum(1 for *_, s in results_log if s.startswith("OK"))
    err = sum(1 for *_, s in results_log if s.startswith("ERROR"))
    print(f"SHAP dependence plots: {ok} saved, {err} errors, out of {total}.")
    return pd.DataFrame(results_log, columns=["rank", "feature", "importance", "status"])
