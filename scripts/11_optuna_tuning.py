#!/usr/bin/env python3
"""
Step 11 — Bayesian hyperparameter optimisation (Optuna) for the XGBoost gamma model.

Best hyperparameters are printed for copy-paste into wulff_ml/config.py (PARAMS_XGB).
This script does NOT overwrite config.py automatically.

Outputs
-------
results/
    optuna_best_params.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from wulff_ml import config
from wulff_ml.feature_engineering import load_inputs, build_feature_matrix
from wulff_ml.utils import clean_inf_nan, banner

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS    = 35
TUNING_SEEDS = [22, 42]


def make_objective(X, y, groups, fixed_params):
    _trial_times = []

    def objective(trial):
        t0 = time.time()
        hp = {
            "learning_rate":    trial.suggest_float("learning_rate", 0.03, 0.10, log=True),
            "max_depth":        trial.suggest_int("max_depth", 5, 9),
            "subsample":        trial.suggest_float("subsample", 0.65, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.80),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 7),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 5.0),
        }
        params = {**fixed_params, **hp}
        trial_val_maes = []

        for seed in TUNING_SEEDS:
            params["random_state"] = seed
            gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
            tv_idx, _ = next(gss_outer.split(X, y, groups))
            X_tv, y_tv, grp_tv = X.iloc[tv_idx], y.iloc[tv_idx], groups.iloc[tv_idx]

            gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
            tr_idx, vl_idx = next(gss_inner.split(X_tv, y_tv, grp_tv))
            X_tr_s, y_tr_s = X_tv.iloc[tr_idx], y_tv.iloc[tr_idx]
            X_vl_s, y_vl_s = X_tv.iloc[vl_idx], y_tv.iloc[vl_idx]

            X_tr_s, med_s = clean_inf_nan(X_tr_s)
            X_vl_s, _     = clean_inf_nan(X_vl_s, med_s)

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr_s, y_tr_s, eval_set=[(X_vl_s, y_vl_s)], verbose=False)
            trial_val_maes.append(mean_absolute_error(y_vl_s, model.predict(X_vl_s)))

        _trial_times.append(time.time() - t0)
        return float(np.mean(trial_val_maes))

    return objective, _trial_times


def main():
    config.ensure_dirs()
    banner("STEP 11: OPTUNA HYPERPARAMETER OPTIMISATION")

    targets, bulk_tr, _ = load_inputs()
    _, X, y, groups, feature_cols, _ = build_feature_matrix(targets, bulk_tr)

    objective, trial_times = make_objective(X, y, groups, config.FIXED_PARAMS_XGB)

    pbar = tqdm(total=N_TRIALS, desc="Optuna trials", unit="trial")
    best_mae_so_far = [float("inf")]

    def _pbar_callback(study, trial):
        val = trial.value if trial.value is not None else float("nan")
        if val < best_mae_so_far[0]:
            best_mae_so_far[0] = val
        pbar.set_postfix_str(f"trial_MAE={val:.5f}  best={best_mae_so_far[0]:.5f}")
        pbar.update(1)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[_pbar_callback])
    pbar.close()

    print("\nOPTIMISATION COMPLETE")
    print(f"  Best val MAE (eV/Å²) : {study.best_value:.5f}")
    print(f"  Best trial #         : {study.best_trial.number}")
    print("\nBest hyperparameters (copy into wulff_ml/config.py PARAMS_XGB):")
    for k, v in study.best_params.items():
        print(f"    '{k}': {v},")

    import json
    out_path = os.path.join(config.RESULTS_DIR, "optuna_best_params.json")
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
