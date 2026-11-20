#!/usr/bin/env python3
"""
Step 7 — Multi-modal validation: XGBoost vs Linear / RF / LightGBM / CatBoost / ExtraTrees.

Outputs
-------
models/
    model_gamma.pkl  model_lr.pkl  model_rf.pkl  model_lgbm.pkl
    model_cb.pkl     model_et.pkl
    catboost_info/   (CatBoost training logs)
figures/validation/
    feature_importance_xgb.png / .pdf / .svg / _raw.png
    parity_xgb.png  parity_lr.png  parity_rf.png
    feature_importance_rf.png
    model_comparison.png / .pdf / .svg / _raw.png
results/
    model_comparison.csv   (MAE in J/m² + eV/Å² reference column)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wulff_ml import config
from wulff_ml.feature_engineering import load_inputs, build_feature_matrix
from wulff_ml.cross_validation import compound_split
from wulff_ml.model_training import (
    train_xgb, evaluate_model, train_linear_regression, train_random_forest,
    train_extra_trees, train_lightgbm, train_catboost, compare_models,
)
from wulff_ml.visualization import feature_importance_plot, parity_plot, model_comparison_plot
from wulff_ml.utils import save_artifact, banner


def main():
    config.ensure_dirs()
    banner("STEP 7: MULTI-MODAL VALIDATION")

    targets, bulk_tr, _ = load_inputs()
    _, X, y, groups, feature_cols, _ = build_feature_matrix(targets, bulk_tr)
    splits = compound_split(X, y, groups)

    feature_cols_num = [c for c in feature_cols if c not in config.CAT_COLS]
    splits_num = {
        "X_train": splits["X_train"][feature_cols_num],
        "X_val":   splits["X_val"][feature_cols_num],
        "X_test":  splits["X_test"][feature_cols_num],
        "y_train": splits["y_train"], "y_val": splits["y_val"], "y_test": splits["y_test"],
    }

    # ── A. XGBoost ────────────────────────────────────────────────────────
    banner("A. XGBoost", char="-")
    model_xgb = train_xgb(splits)
    evaluate_model(model_xgb, splits, label="XGBoost")
    feature_importance_plot(model_xgb, feature_cols,
                            os.path.join(config.FIG_VALID_DIR, "feature_importance_xgb.png"))
    parity_plot(splits["y_test"], model_xgb.predict(splits["X_test"]),
                os.path.join(config.FIG_VALID_DIR, "parity_xgb.png"),
                title="Test set parity — XGBoost")
    save_artifact(model_xgb, "model_gamma.pkl")

    # ── B. Linear Regression ─────────────────────────────────────────────
    banner("B. Linear Regression", char="-")
    model_lr = train_linear_regression(splits_num)
    evaluate_model(model_lr, splits_num, label="LinearRegression")
    parity_plot(splits_num["y_test"], model_lr.predict(splits_num["X_test"]),
                os.path.join(config.FIG_VALID_DIR, "parity_lr.png"),
                title="Test set parity — Linear Regression")
    save_artifact(model_lr, "model_lr.pkl")

    # ── C. Random Forest ─────────────────────────────────────────────────
    banner("C. Random Forest", char="-")
    model_rf = train_random_forest(splits_num)
    evaluate_model(model_rf, splits_num, label="RandomForest")
    feature_importance_plot(model_rf, feature_cols_num,
                            os.path.join(config.FIG_VALID_DIR, "feature_importance_rf.png"))
    save_artifact(model_rf, "model_rf.pkl")

    # ── D. LightGBM ──────────────────────────────────────────────────────
    banner("D. LightGBM", char="-")
    try:
        model_lgbm, splits_lgbm = train_lightgbm(splits)
        evaluate_model(model_lgbm, {**splits_lgbm, "y_train": splits["y_train"],
                                    "y_val": splits["y_val"], "y_test": splits["y_test"]},
                       label="LightGBM")
        save_artifact(model_lgbm, "model_lgbm.pkl")
    except ImportError:
        print("lightgbm not installed — skipping. `pip install lightgbm` to enable.")
        model_lgbm, splits_lgbm = None, None

    # ── E. CatBoost ──────────────────────────────────────────────────────
    banner("E. CatBoost", char="-")
    try:
        model_cb, splits_cb = train_catboost(splits)
        evaluate_model(model_cb, {**splits_cb, "y_train": splits["y_train"],
                                  "y_val": splits["y_val"], "y_test": splits["y_test"]},
                       label="CatBoost")
        save_artifact(model_cb, "model_cb.pkl")
    except ImportError:
        print("catboost not installed — skipping. `pip install catboost` to enable.")
        model_cb, splits_cb = None, None

    # ── F. Extra Trees ────────────────────────────────────────────────────
    banner("F. Extra Trees", char="-")
    model_et = train_extra_trees(splits_num)
    evaluate_model(model_et, splits_num, label="ExtraTrees")
    save_artifact(model_et, "model_et.pkl")

    # ── Comparison ────────────────────────────────────────────────────────
    banner("MODEL COMPARISON SUMMARY", char="-")
    registry = [
        ("Linear Regression", model_lr, splits_num["X_test"]),
        ("Random Forest",     model_rf, splits_num["X_test"]),
        ("XGBoost",           model_xgb, splits["X_test"]),
        ("Extra Trees",       model_et, splits_num["X_test"]),
    ]
    if model_lgbm is not None:
        registry.append(("LightGBM", model_lgbm, splits_lgbm["X_test"]))
    if model_cb is not None:
        registry.append(("CatBoost", model_cb, splits_cb["X_test"]))

    results_df = compare_models(registry, splits["y_test"])
    results_path = os.path.join(config.RESULTS_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    model_comparison_plot(results_df, os.path.join(config.FIG_VALID_DIR, "model_comparison.png"))


if __name__ == "__main__":
    main()
