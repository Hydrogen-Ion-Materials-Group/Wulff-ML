#!/usr/bin/env python3
"""
Step 9 — Data ablation: how much training data do we actually need?

Trains the model on 10%, 20%, ..., 100% of the training compounds and records
Train/Val/Test MAE (J/m²) and R² at each fraction.

Outputs
-------
figures/ablation/
    data_ablation.png / .pdf / .svg / _raw.png
results/
    data_ablation.csv   (MAE in J/m²; eV/Å² columns included for reference)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm.auto import tqdm

from wulff_ml import config
from wulff_ml.feature_engineering import load_inputs, build_feature_matrix
from wulff_ml.cross_validation import compound_split
from wulff_ml.units import to_jm2, UNIT_SHORT
from wulff_ml.utils import clean_inf_nan, banner
from wulff_ml.visualization import ablation_curve_plot
from wulff_ml.cache import cached

DATA_FRACS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


@cached("data_ablation_one_fraction")
def _ablate_one_fraction(frac, X_train, X_val, X_test, y_train, y_val, y_test,
                          groups_train, ablation_params, random_state):
    unique_train_cpds = groups_train.unique()
    n_cpds = max(1, int(len(unique_train_cpds) * frac))
    rng = np.random.default_rng(seed=random_state)
    sampled_cpds = rng.choice(unique_train_cpds, size=n_cpds, replace=False)

    mask = groups_train.isin(sampled_cpds)
    X_tr_frac, y_tr_frac = X_train[mask], y_train[mask]

    X_tr_clean, med_frac = clean_inf_nan(X_tr_frac)
    X_vl_clean, _        = clean_inf_nan(X_val, med_frac)
    X_te_clean, _        = clean_inf_nan(X_test, med_frac)

    t0  = time.time()
    mdl = xgb.XGBRegressor(**ablation_params)
    mdl.fit(X_tr_clean, y_tr_frac, eval_set=[(X_vl_clean, y_val)], verbose=False)
    elapsed = time.time() - t0

    # Store both eV/Å² (native) and J/m² (display)
    train_mae_ev = mean_absolute_error(y_tr_frac, mdl.predict(X_tr_clean))
    val_mae_ev   = mean_absolute_error(y_val,     mdl.predict(X_vl_clean))
    test_mae_ev  = mean_absolute_error(y_test,    mdl.predict(X_te_clean))

    return {
        "frac": frac, "n_rows": len(X_tr_frac), "n_cpds": n_cpds,
        "train_mae": to_jm2(train_mae_ev), "val_mae": to_jm2(val_mae_ev), "test_mae": to_jm2(test_mae_ev),
        "train_mae_ev": train_mae_ev,       "val_mae_ev": val_mae_ev,       "test_mae_ev": test_mae_ev,
        "train_r2": r2_score(y_tr_frac, mdl.predict(X_tr_clean)),
        "test_r2":  r2_score(y_test,    mdl.predict(X_te_clean)),
        "time_s": elapsed,
    }


def main():
    config.ensure_dirs()
    banner("STEP 9: DATA ABLATION")

    targets, bulk_tr, _ = load_inputs()
    _, X, y, groups, feature_cols, _ = build_feature_matrix(targets, bulk_tr)
    splits = compound_split(X, y, groups)

    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]
    groups_train = splits["groups_train"]

    ablation_params = {**config.PARAMS_XGB, "random_state": config.RANDOM_STATE}

    print(f"Running data ablation over {len(DATA_FRACS)} fractions ...")
    print(f"  Full training set: {len(X_train)} rows ({groups_train.nunique()} compounds)")

    records = []
    for frac in tqdm(DATA_FRACS, desc="Data fraction", unit="step"):
        records.append(_ablate_one_fraction(
            frac, X_train, X_val, X_test, y_train, y_val, y_test,
            groups_train, ablation_params, config.RANDOM_STATE))

    ablation_df = pd.DataFrame(records)
    print(ablation_df[["frac","train_mae","val_mae","test_mae","test_r2"]].to_string(index=False))

    best_test_mae  = ablation_df["test_mae"].min()
    threshold      = best_test_mae * 1.01
    saturation_row = ablation_df[ablation_df["test_mae"] <= threshold].iloc[0]
    print(f"\nSaturation point (within 1% of best MAE={best_test_mae:.4f} {UNIT_SHORT}): "
          f"{saturation_row['frac']:.0%} of data "
          f"({saturation_row['n_cpds']:.0f} compounds, {saturation_row['n_rows']:.0f} rows)")

    csv_path = os.path.join(config.RESULTS_DIR, "data_ablation.csv")
    ablation_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    ablation_curve_plot(
        ablation_df,
        out_path=os.path.join(config.FIG_ABLATION_DIR, "data_ablation.png"),
        title="Data Ablation",
    )


if __name__ == "__main__":
    main()
