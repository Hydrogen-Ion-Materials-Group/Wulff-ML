#!/usr/bin/env python3
"""
Step 5 — Elemental split: hold out entire elements as the test set.

Outputs
-------
figures/validation/
    feature_importance_elemental_{elements}.png / .pdf / .svg / _raw.png
    parity_elemental_{elements}.png / .pdf / .svg / _raw.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wulff_ml import config
from wulff_ml.feature_engineering import load_inputs, build_feature_matrix
from wulff_ml.cross_validation import elemental_split
from wulff_ml.model_training import train_xgb, evaluate_model
from wulff_ml.visualization import feature_importance_plot, parity_plot
from wulff_ml.utils import banner


def main():
    config.ensure_dirs()
    banner(f"STEP 5: ELEMENTAL SPLIT — holding out {config.TEST_ELEMENTS}")

    targets, bulk_tr, _ = load_inputs()
    df, X, y, groups, feature_cols, _ = build_feature_matrix(targets, bulk_tr)

    el_split = elemental_split(
        df_full=df, X_full=X, y_full=y, groups_full=groups,
        test_elements=config.TEST_ELEMENTS,
        val_fraction=config.VAL_FRACTION, random_state=config.RANDOM_STATE,
    )
    print(el_split["summary"])

    splits = {
        "X_train": el_split["X_train"], "X_val": el_split["X_val"], "X_test": el_split["X_test"],
        "y_train": el_split["y_train"], "y_val": el_split["y_val"], "y_test": el_split["y_test"],
    }

    model = train_xgb(splits)
    evaluate_model(model, splits, label=f"gamma — elemental holdout {config.TEST_ELEMENTS}")

    suffix = "_".join(config.TEST_ELEMENTS)
    feature_importance_plot(
        model, feature_cols,
        out_path=os.path.join(config.FIG_VALID_DIR, f"feature_importance_elemental_{suffix}.png"),
        title=f"Top 20 Features — gamma (holdout {config.TEST_ELEMENTS})",
    )
    parity_plot(
        splits["y_test"], model.predict(splits["X_test"]),
        out_path=os.path.join(config.FIG_VALID_DIR, f"parity_elemental_{suffix}.png"),
        title=f"Elemental holdout {config.TEST_ELEMENTS}",
    )


if __name__ == "__main__":
    main()
