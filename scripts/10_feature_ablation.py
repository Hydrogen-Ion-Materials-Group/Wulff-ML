#!/usr/bin/env python3
"""
Step 10 — Feature ablation studies.

  A. Melting-point ablation       — over-reliance on cohesion proxies?
  B. Descriptor-group ablation    — leave-one-group-out + group-only MAE
  C. Group-wise forward selection — cumulative SHAP-ranked group addition

All MAE values printed and stored in J/m² (eV/Å² reference columns included).

Outputs
-------
figures/ablation/
    melt_ablation_parity.png / .pdf / .svg / _raw.png
    descriptor_group_ablation.png / .pdf / .svg / _raw.png
    feature_group_ablation.png / .pdf / .svg / _raw.png
results/
    melting_point_ablation.csv
    descriptor_group_ablation.csv
    feature_group_ablation.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from wulff_ml import config
from wulff_ml.feature_engineering import load_inputs, build_feature_matrix
from wulff_ml.cross_validation import compound_split
from wulff_ml.model_training import train_xgb, evaluate_model
from wulff_ml.shap_analysis import compute_shap_values
from wulff_ml.feature_groups import FEATURE_ABLATION_GROUPS, resolve_groups
from wulff_ml.units import to_jm2, UNIT_SHORT
from wulff_ml.utils import clean_inf_nan, banner
from wulff_ml.visualization import group_ablation_plot, save_figure
from wulff_ml.cache import cached


@cached("feature_ablation_train_eval")
def _train_eval(feat_list, X_train, X_val, X_test, y_train, y_val, y_test, params):
    if len(feat_list) == 0:
        return float("nan"), float("nan")
    X_tr, med = clean_inf_nan(X_train[feat_list].copy())
    X_vl, _   = clean_inf_nan(X_val[feat_list].copy(), med)
    X_te, _   = clean_inf_nan(X_test[feat_list].copy(), med)
    m = xgb.XGBRegressor(**params)
    m.fit(X_tr, y_train, eval_set=[(X_vl, y_val)], verbose=False)
    p = m.predict(X_te)
    return mean_absolute_error(y_test, p), r2_score(y_test, p)


@cached("feature_ablation_fit_model")
def _fit_xgb(feat_list, X_train, X_val, y_train, y_val, params):
    X_tr, med = clean_inf_nan(X_train[feat_list].copy())
    X_vl, _   = clean_inf_nan(X_val[feat_list].copy(), med)
    m = xgb.XGBRegressor(**params)
    m.fit(X_tr, y_train, eval_set=[(X_vl, y_val)], verbose=False)
    return m, med


def melting_point_ablation(splits, feature_cols, params):
    banner("A. MELTING-POINT ABLATION", char="-")
    melt_kws  = ["melt", "MeltingT", "melting_point", "MeltTemp",
                 "BoilingT", "boil", "boiling_point", "fusion", "CohesiveEnergy"]
    melt_feats   = [f for f in feature_cols if any(kw.lower() in f.lower() for kw in melt_kws)]
    feat_no_melt = [f for f in feature_cols if f not in melt_feats]
    print(f"  Melting-point features ({len(melt_feats)}): {melt_feats}")

    if not melt_feats:
        print("  No melting-point features found; skipping.")
        return None

    model_no_melt, med_nm = _fit_xgb(feat_no_melt, splits["X_train"], splits["X_val"],
                                      splits["y_train"], splits["y_val"], params)
    X_te_nm, _ = clean_inf_nan(splits["X_test"][feat_no_melt].copy(), med_nm)

    model_full    = train_xgb(splits, verbose=False)
    pred_full     = model_full.predict(splits["X_test"])
    pred_no_melt  = model_no_melt.predict(X_te_nm)

    mae_full    = mean_absolute_error(splits["y_test"], pred_full)
    mae_no_melt = mean_absolute_error(splits["y_test"], pred_no_melt)
    print(f"  Full model    : MAE={to_jm2(mae_full):.4f} {UNIT_SHORT}")
    print(f"  No melting-pt : MAE={to_jm2(mae_no_melt):.4f} {UNIT_SHORT}  "
          f"(delta={to_jm2(mae_no_melt - mae_full):+.4f} {UNIT_SHORT})")

    y_test_jm2  = to_jm2(splits["y_test"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, pred), col in zip(
            axes, [("Full model", pred_full), ("No melting-pt", pred_no_melt)],
            ["steelblue", "darkorange"]):
        mae_  = mean_absolute_error(splits["y_test"], pred)
        r2_   = r2_score(splits["y_test"], pred)
        ax.scatter(y_test_jm2, to_jm2(pred), alpha=0.25, s=4, color=col)
        lo, hi = float(y_test_jm2.min()), float(y_test_jm2.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax.set_title(f"{name}\nMAE={to_jm2(mae_):.4f} {UNIT_SHORT}  R²={r2_:.4f}")
        ax.set_xlabel(f"True γ ({UNIT_SHORT})")
        ax.set_ylabel(f"Predicted γ ({UNIT_SHORT})")
    plt.tight_layout()
    save_figure(fig, os.path.join(config.FIG_ABLATION_DIR, "melt_ablation_parity.png"))
    plt.close(fig)

    result_df = pd.DataFrame([
        {"model": "full",            "mae_jm2": to_jm2(mae_full),    "mae_ev_ang2": mae_full},
        {"model": "no_melting_point","mae_jm2": to_jm2(mae_no_melt), "mae_ev_ang2": mae_no_melt},
    ])
    result_df.to_csv(os.path.join(config.RESULTS_DIR, "melting_point_ablation.csv"), index=False)
    return result_df


def descriptor_group_ablation(splits, feature_cols, params):
    banner("B. DESCRIPTOR-GROUP ABLATION (leave-one-group-out)", char="-")
    group_keywords = {
        "melting_point":  ["melt", "MeltingT", "melting_point", "BoilingT", "boil", "CohesiveEnergy", "fusion"],
        "d_electron":     ["NdVal", "ndval", "d_val", "d_electron", "DValence", "dvalence", "Nd_", "NdUnfilled", "nd_"],
        "volume_density": ["volume", "density", "Volume", "Density", "AtomicVol", "MolarVol", "atomic_vol"],
        "periodic_table": ["electro", "Electro", "NpUnfilled", "AtomicRad", "radius", "Radius",
                           "Period", "period_", "NpVal", "NsVal", "NfVal", "AtomicNum", "Column",
                           "Row_", "_row", "GSbandgap", "GSmagmom", "SpaceGroupNumber", "Mendeleev"],
    }
    descriptor_groups = {g: [f for f in feature_cols if any(kw.lower() in f.lower() for kw in kws)]
                         for g, kws in group_keywords.items()}
    assigned = {f for feats in descriptor_groups.values() for f in feats}
    descriptor_groups["other"] = [f for f in feature_cols if f not in assigned]

    args = (splits["X_train"], splits["X_val"], splits["X_test"],
            splits["y_train"], splits["y_val"], splits["y_test"], params)
    mae_full, r2_full = _train_eval(feature_cols, *args)

    records = []
    for gname, gfeats in descriptor_groups.items():
        if not gfeats:
            continue
        mae_only, r2_only       = _train_eval(gfeats, *args)
        feats_without           = [f for f in feature_cols if f not in gfeats]
        mae_without, r2_without = _train_eval(feats_without, *args)
        delta_mae = mae_without - mae_full
        print(f"  [{gname}] only: MAE={to_jm2(mae_only):.4f}  "
              f"without: MAE={to_jm2(mae_without):.4f}  "
              f"delta={to_jm2(delta_mae):+.4f} {UNIT_SHORT}")
        records.append({"group": gname, "n_features": len(gfeats),
                        "mae_only_jm2": to_jm2(mae_only),       "r2_only": r2_only,
                        "mae_without_jm2": to_jm2(mae_without), "r2_without": r2_without,
                        "delta_mae_jm2": to_jm2(delta_mae),
                        "mae_only_ev": mae_only, "mae_without_ev": mae_without, "delta_mae_ev": delta_mae})

    abl_df = pd.DataFrame(records).sort_values("delta_mae_jm2", ascending=False)
    abl_df.to_csv(os.path.join(config.RESULTS_DIR, "descriptor_group_ablation.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["crimson" if v > 0 else "seagreen" for v in abl_df["delta_mae_jm2"]]
    axes[0].barh(abl_df["group"], abl_df["delta_mae_jm2"], color=colors, alpha=0.8)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel(f"ΔMAE vs full model ({UNIT_SHORT}, positive = important)")
    axes[0].set_title("Leave-One-Group-Out: ΔMAE")
    axes[1].barh(abl_df["group"], abl_df["mae_only_jm2"], color="steelblue", alpha=0.8,
                 label=f"Group-only MAE")
    axes[1].axvline(to_jm2(mae_full), color="red", linestyle="--", lw=1.5,
                    label=f"Full model ({to_jm2(mae_full):.4f} {UNIT_SHORT})")
    axes[1].set_xlabel(f"Test MAE ({UNIT_SHORT})")
    axes[1].set_title("Group-Only MAE vs Full Model")
    axes[1].legend()
    plt.tight_layout()
    save_figure(fig, os.path.join(config.FIG_ABLATION_DIR, "descriptor_group_ablation.png"))
    plt.close(fig)
    return abl_df


def group_forward_selection(splits, feature_cols, params):
    banner("C. GROUP-WISE FORWARD SELECTION (SHAP-ranked)", char="-")
    model_full  = train_xgb(splits, verbose=False)
    shap_values = compute_shap_values(model_full, splits["X_test"], feature_cols)

    groups    = resolve_groups(FEATURE_ABLATION_GROUPS, feature_cols)
    mean_shap = pd.Series(np.abs(shap_values.values).mean(axis=0), index=feature_cols)
    for g in groups:
        g["group_shap"] = mean_shap.reindex(g["features"]).fillna(0).mean()
    groups.sort(key=lambda g: g["group_shap"], reverse=True)

    args = (splits["X_train"], splits["X_val"], splits["X_test"],
            splits["y_train"], splits["y_val"], splits["y_test"], params)
    ref_mae, ref_r2 = _train_eval(feature_cols, *args)
    print(f"Full-feature baseline: MAE={to_jm2(ref_mae):.4f} {UNIT_SHORT}  R²={ref_r2:.4f}")

    records, active_features, prev_mae = [], [], None
    for step, g in enumerate(groups, start=1):
        active_features = active_features + g["features"]
        val_mae, _ = _train_eval(active_features,
                                 splits["X_train"], splits["X_val"], splits["X_val"],
                                 splits["y_train"], splits["y_val"], splits["y_val"], params)
        te_mae, te_r2 = _train_eval(active_features, *args)
        delta = (te_mae - prev_mae) if prev_mae is not None else 0.0
        records.append({"step": step, "group_id": g["id"], "group_label": g["label"],
                        "n_features": len(active_features), "n_added": len(g["features"]),
                        "group_shap": g["group_shap"],
                        "val_mae": val_mae,  "test_mae": te_mae,
                        "test_r2": te_r2,    "delta_mae": delta})
        print(f"  step {step:2d} [{g['id']}] {g['label']:<22} n={len(active_features):3d} "
              f"test_mae={to_jm2(te_mae):.4f} {UNIT_SHORT}  delta={to_jm2(delta):+.4f}")
        prev_mae = te_mae

    ga_df    = pd.DataFrame(records)
    best_row = ga_df.loc[ga_df["test_mae"].idxmin()]
    ga_df.to_csv(os.path.join(config.RESULTS_DIR, "feature_group_ablation.csv"), index=False)

    group_ablation_plot(
        ga_df, groups, ref_mae, best_row, rank_method="mean |SHAP|",
        out_path=os.path.join(config.FIG_ABLATION_DIR, "feature_group_ablation.png"),
    )
    return ga_df


def main():
    config.ensure_dirs()
    banner("STEP 10: FEATURE ABLATION STUDIES")

    targets, bulk_tr, _ = load_inputs()
    _, X, y, groups, feature_cols, _ = build_feature_matrix(targets, bulk_tr)
    splits = compound_split(X, y, groups)
    params = config.PARAMS_XGB

    melting_point_ablation(splits, feature_cols, params)
    descriptor_group_ablation(splits, feature_cols, params)
    group_forward_selection(splits, feature_cols, params)


if __name__ == "__main__":
    main()
