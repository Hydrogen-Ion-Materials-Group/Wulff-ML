#!/usr/bin/env python3
"""
Step 2 — Train the primary XGBoost surface-energy (gamma) model.

Loads targets + bulk descriptors, builds the feature matrix, performs a
strict compound-wise train/val/test split, trains XGBoost, evaluates,
saves the model bundle, and predicts gamma for every (compound x hkl)
combination in the prediction pool.

Outputs
-------
models/
    model_gamma.pkl
    train_medians_gamma.pkl
    feature_cols_gamma.pkl
    cat_categories_gamma.pkl
figures/main/
    feature_importance.png / .pdf / .svg / _raw.png
    parity.png / .pdf / .svg / _raw.png
results/
    all_facets_gamma.csv          (columns: mp_id, formula_pretty, hkl,
                                   pred_gamma_eV_per_ang2, pred_gamma_J_per_m2)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wulff_ml import config
from wulff_ml.feature_engineering import load_inputs, build_feature_matrix
from wulff_ml.cross_validation import compound_split
from wulff_ml.model_training import train_xgb, evaluate_model, predict_all_facets
from wulff_ml.visualization import feature_importance_plot, parity_plot
from wulff_ml.utils import save_model_bundle, banner


def main():
    config.ensure_dirs()
    banner("STEP 2: TRAIN XGBOOST GAMMA MODEL")

    targets, bulk_tr, bulk_pr = load_inputs()
    df, X, y, groups, feature_cols, train_medians = build_feature_matrix(targets, bulk_tr)

    splits = compound_split(X, y, groups)
    model  = train_xgb(splits)
    evaluate_model(model, splits, label="gamma")

    feature_importance_plot(
        model, feature_cols,
        out_path=os.path.join(config.FIG_MAIN_DIR, "feature_importance.png"),
        title="Top 20 Features — gamma",
    )
    parity_plot(
        splits["y_test"], model.predict(splits["X_test"]),
        out_path=os.path.join(config.FIG_MAIN_DIR, "parity.png"),
        title="Test set parity — gamma",
    )

    cat_categories = {col: X[col].cat.categories for col in config.CAT_COLS}
    save_model_bundle(model, train_medians, feature_cols, cat_categories, model_name="gamma")

    banner("PREDICTING gamma FOR ALL COMPOUND x HKL COMBINATIONS")
    predict_all_facets(
        model, bulk_pr, targets, feature_cols, cat_categories, train_medians,
        out_path=os.path.join(config.RESULTS_DIR, "all_facets_gamma.csv"),
    )


if __name__ == "__main__":
    main()
