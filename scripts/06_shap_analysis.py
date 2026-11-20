#!/usr/bin/env python3
"""
Step 6 — SHAP interpretability analysis.

Generates:
  1. Global beeswarm (top 30 features)          — SHAP axis in J/m²
  2. Curated/renamed beeswarm (11 headline descriptors)
  3. Per-feature SHAP dependence scatter plots  (set GENERATE_ALL=False to skip)

Outputs
-------
figures/main/
    shap_beeswarm.png / .pdf / .svg / _raw.png
    shap_beeswarm_custom_filtered.png / .pdf / .svg / _raw.png
figures/supplementary/shap_dependence/
    shap_dep_001_<feature>.png  ...  (one file per feature)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wulff_ml import config
from wulff_ml.feature_engineering import load_inputs, build_feature_matrix
from wulff_ml.cross_validation import compound_split
from wulff_ml.model_training import train_xgb, evaluate_model
from wulff_ml.shap_analysis import (
    compute_shap_values, plot_beeswarm, plot_filtered_beeswarm,
    plot_all_dependence_plots,
)
from wulff_ml.utils import banner

GENERATE_ALL_DEPENDENCE_PLOTS = True


def main():
    config.ensure_dirs()
    banner("STEP 6: SHAP INTERPRETABILITY ANALYSIS")

    targets, bulk_tr, _ = load_inputs()
    _, X, y, groups, feature_cols, _ = build_feature_matrix(targets, bulk_tr)
    splits = compound_split(X, y, groups)

    model = train_xgb(splits)
    evaluate_model(model, splits, label="gamma")

    shap_values = compute_shap_values(model, splits["X_test"], feature_cols)

    plot_beeswarm(
        shap_values,
        out_path=os.path.join(config.FIG_MAIN_DIR, "shap_beeswarm.png"),
        max_display=30,
    )
    plot_filtered_beeswarm(
        shap_values, feature_cols,
        out_path=os.path.join(config.FIG_MAIN_DIR, "shap_beeswarm_custom_filtered.png"),
    )

    if GENERATE_ALL_DEPENDENCE_PLOTS:
        plot_all_dependence_plots(
            model, shap_values, feature_cols,
            out_dir=config.FIG_SHAP_DEP_DIR,
        )


if __name__ == "__main__":
    main()
