#!/usr/bin/env python3
"""
Step 3 — Cross-validation & leakage diagnostics.

  ToDo 1: Compound-wise (grouped) split vs naive random split.
  ToDo 2: 5-fold GroupKFold CV.
  ToDo 5: Composition-grouped split.

Outputs
-------
figures/cv/
    cv5fold_grouped.png / .pdf / .svg / _raw.png
    composition_grouped_split.png / .pdf / .svg / _raw.png
results/
    cv_results.csv
    cross_validation_summary.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from wulff_ml import config
from wulff_ml.feature_engineering import load_inputs, build_feature_matrix
from wulff_ml.cross_validation import (
    compound_split, random_split_leakage_demo, group_kfold_cv, composition_split,
)
from wulff_ml.model_training import train_xgb
from wulff_ml.units import to_jm2, UNIT_SHORT
from wulff_ml.visualization import cv_results_plot, parity_plot
from wulff_ml.utils import banner


def main():
    config.ensure_dirs()
    banner("STEP 3: CROSS-VALIDATION & LEAKAGE DIAGNOSTICS")

    targets, bulk_tr, _ = load_inputs()
    df, X, y, groups, feature_cols, _ = build_feature_matrix(targets, bulk_tr)

    # ── Reference: compound-wise (grouped) split ──────────────────────────
    splits        = compound_split(X, y, groups)
    model_grouped = train_xgb(splits, verbose=False)
    pred_grouped  = model_grouped.predict(splits["X_test"])
    mae_grouped   = mean_absolute_error(splits["y_test"], pred_grouped)
    r2_grouped    = r2_score(splits["y_test"], pred_grouped)

    # ── ToDo 1: naive random split (leakage demo) ─────────────────────────
    banner("ToDo 1: GROUPED SPLIT vs RANDOM SPLIT", char="-")
    rnd       = random_split_leakage_demo(X, y, groups)
    model_rnd = train_xgb(
        {"X_train": rnd["X_train"], "X_val": rnd["X_val"],
         "y_train": rnd["y_train"], "y_val": rnd["y_val"]},
        verbose=False)
    pred_rnd = model_rnd.predict(rnd["X_test"])
    mae_rnd  = mean_absolute_error(rnd["y_test"], pred_rnd)
    r2_rnd   = r2_score(rnd["y_test"], pred_rnd)

    print(f"  Grouped split (correct) : MAE={to_jm2(mae_grouped):.4f} {UNIT_SHORT}  R²={r2_grouped:.4f}  overlap=0")
    print(f"  Random  split (naive)   : MAE={to_jm2(mae_rnd):.4f} {UNIT_SHORT}  R²={r2_rnd:.4f}  overlap={rnd['overlap']}")
    print("  If random MAE < grouped MAE, the gap is leakage, not generalisation.")

    # ── ToDo 2: 5-fold grouped CV ──────────────────────────────────────────
    banner("ToDo 2: 5-FOLD GROUP CROSS-VALIDATION", char="-")
    cv_df = group_kfold_cv(X, y, groups, config.PARAMS_XGB, n_folds=5)
    cv_results_plot(cv_df, os.path.join(config.FIG_CV_DIR, "cv5fold_grouped.png"))
    cv_df.to_csv(os.path.join(config.RESULTS_DIR, "cv_results.csv"), index=False)

    # ── ToDo 5: composition-grouped split ──────────────────────────────────
    banner("ToDo 5: COMPOSITION GROUPED SPLIT", char="-")
    comp_groups = df["formula_pretty"].fillna(df["mp_id"]).reset_index(drop=True)
    comp        = composition_split(X, y, comp_groups)
    model_comp  = train_xgb(
        {"X_train": comp["X_train"], "X_val": comp["X_val"],
         "y_train": comp["y_train"], "y_val": comp["y_val"]},
        verbose=False)
    pred_comp = model_comp.predict(comp["X_test"])
    mae_comp  = mean_absolute_error(comp["y_test"], pred_comp)
    r2_comp   = r2_score(comp["y_test"], pred_comp)

    print(f"  MP-ID grouped (ref)   : MAE={to_jm2(mae_grouped):.4f} {UNIT_SHORT}  R²={r2_grouped:.4f}")
    print(f"  Composition grouped   : MAE={to_jm2(mae_comp):.4f} {UNIT_SHORT}  R²={r2_comp:.4f}  overlap={comp['overlap']}")

    parity_plot(comp["y_test"], pred_comp,
                out_path=os.path.join(config.FIG_CV_DIR, "composition_grouped_split.png"),
                title="Composition-grouped split")

    summary = pd.DataFrame([
        {"strategy": "grouped_mp_id",     "mae_jm2": to_jm2(mae_grouped), "mae_ev_ang2": mae_grouped, "r2": r2_grouped,    "overlap": 0},
        {"strategy": "random_naive",       "mae_jm2": to_jm2(mae_rnd),     "mae_ev_ang2": mae_rnd,     "r2": r2_rnd,        "overlap": rnd["overlap"]},
        {"strategy": "composition_grouped","mae_jm2": to_jm2(mae_comp),    "mae_ev_ang2": mae_comp,    "r2": r2_comp,       "overlap": comp["overlap"]},
        {"strategy": "5fold_cv_mean",      "mae_jm2": to_jm2(cv_df["mae"].mean()), "mae_ev_ang2": cv_df["mae"].mean(), "r2": cv_df["r2"].mean(), "overlap": None},
    ])
    summary_path = os.path.join(config.RESULTS_DIR, "cross_validation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
