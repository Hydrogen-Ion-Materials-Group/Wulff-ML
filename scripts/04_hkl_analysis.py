#!/usr/bin/env python3
"""
Step 4 — P(h,k,l) broken-bond surface-density analysis.

Four experiments, each saved as its own figure:

  Exp 1 — Model A (full) vs B (chem-only) vs C (chem + P(hkl))
           exp1a_r2_comparison.png
           exp1b_mae_comparison.png
           exp1c_geometry_recovery.png

  Exp 2 — SHAP importance of P(hkl) inside Model C
           exp2a_shap_model_c.png
           exp2b_model_c_parity.png

  Exp 3 — Residual test: do Model C residuals still carry hkl trends?
           exp3_residuals_<feature>.png   (one per hkl descriptor)

  Exp 4 — Predict gamma for every (compound x hkl) with Models A / B / C,
          build Wulff equilibrium shapes for each, and compare the top-3
          dominant exposed facets across the three models.
           exp4_facet_comparison_summary.png

Outputs
-------
figures/hkl/
    exp1a_r2_comparison.png / .pdf / .svg / _raw.png
    exp1b_mae_comparison.png / .pdf / .svg / _raw.png
    exp1c_geometry_recovery.png / .pdf / .svg / _raw.png
    exp2a_shap_model_c.png / .pdf / .svg / _raw.png
    exp2b_model_c_parity.png / .pdf / .svg / _raw.png
    exp3_residuals_h_norm.png / .pdf / .svg / _raw.png
    exp3_residuals_k_norm.png / .pdf / .svg / _raw.png
    exp3_residuals_l_norm.png / .pdf / .svg / _raw.png
    (+ any other hkl descriptor present in the data)
    exp4_facet_comparison_summary.png / .pdf / .svg / _raw.png
results/
    phkl_experiments.csv
    phkl_residuals.csv
    hkl_effect_wulff/
        A_pred.csv, B_pred.csv, C_pred.csv         (gamma predictions, all facets)
        A_wulff_output/, B_wulff_output/, C_wulff_output/
            <mpid>.cif
            <mpid>_<formula>_shape.png              pure Wulff shape, transparent bg
            <mpid>_<formula>_legend.png              top-5 area-fraction legend, transparent bg
            <label>_area_fractions_wulff.csv
        top3_facet_comparison.csv
models/  (config.MODEL_DIR, or <project_root>/models if unset)
    model_A_full.pkl, model_B_chem.pkl, model_C.pkl
    feats_A_full.pkl, feats_B_chem.pkl, feats_C.pkl

Note: Exp 4 downloads crystal structures from the Materials Project API
(it asks for your API key in the terminal if none is stored yet) and requires
network access + mp_api/pymatgen. Experiments 1-3 work offline.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from wulff_ml import config
from wulff_ml.api_key import get_mp_api_key
from wulff_ml.feature_engineering import load_inputs, build_feature_matrix
from wulff_ml.cross_validation import compound_split
from wulff_ml.model_training import train_xgb, evaluate_model
from wulff_ml.hkl_analysis import (
    compute_P_hkl, run_phkl_analysis, three_model_experiment,
    shap_model_c, residual_analysis,
    predict_all_facets_abc, build_wulff_shapes_abc,
    compare_top_facets_abc, plot_facet_comparison_summary,
)
from wulff_ml.units import to_jm2, UNIT_SHORT
from wulff_ml.utils import banner
from wulff_ml.visualization import save_figure

# ── Colour palette ────────────────────────────────────────────────────────────
CBLUE   = "#2471A3"
CRED    = "#C0392B"
CGREEN  = "#1A7431"
CAMBER  = "#B7770D"
CGRAY   = "#7F8C8D"


def main():
    config.ensure_dirs()
    banner("STEP 4: P(h,k,l) BROKEN-BOND SURFACE-DENSITY ANALYSIS")

    # Experiment 4 (Wulff shapes) downloads crystal structures from the
    # Materials Project. Ask for the API key now, before the long calculations;
    # pressing Enter skips Experiment 4 and everything else still runs offline.
    api_key = get_mp_api_key(required=False, allow_skip=True)
    if not api_key:
        print("No Materials Project API key: Experiment 4 (Wulff shapes) will be skipped.\n")

    # ── Data + primary model ──────────────────────────────────────────────
    targets, bulk_tr, bulk_pr = load_inputs()
    df, X, y, groups, feature_cols, _ = build_feature_matrix(targets, bulk_tr)
    splits = compound_split(X, y, groups)

    model_gamma = train_xgb(splits)
    evaluate_model(model_gamma, splits, label="gamma")

    # ── Mediation analysis on the test set ───────────────────────────────
    test_df = splits["X_test"].copy()
    test_df["surface_energy"] = splits["y_test"].values
    run_phkl_analysis(test_df, gamma_col="surface_energy")

    # ── Feature sets ─────────────────────────────────────────────────────
    HKL_FEATURES = [
        "h", "k", "l",
        "h_abs", "k_abs", "l_abs",
        "hkl_sort_0", "hkl_sort_1", "hkl_sort_2",
        "hkl_norm", "hkl_magnitude",
        "h_norm", "k_norm", "l_norm",
        "n_zeros", "max_index", "is_low_index",
        "d_hkl", "inv_d_hkl",
    ]
    HKL_FEATURES  = [f for f in HKL_FEATURES if f in feature_cols]
    chem_features = [f for f in feature_cols if f not in HKL_FEATURES]

    X_train_p = compute_P_hkl(splits["X_train"])
    X_val_p   = compute_P_hkl(splits["X_val"])
    X_test_p  = compute_P_hkl(splits["X_test"])

    # ── Experiment 1 — train Models A, B, C ─────────────────────────────
    banner("EXPERIMENT 1 — Model A vs B vs C", char="-")
    exp = three_model_experiment(
        X_train_p, X_val_p, X_test_p,
        splits["y_train"], splits["y_val"], splits["y_test"],
        feature_cols=feature_cols,
        chem_features=chem_features,
        params=config.PARAMS_XGB,
    )

    te_r2_A,  te_r2_B,  te_r2_C  = exp["r2_A"],  exp["r2_B"],  exp["r2_C"]
    te_mae_A, te_mae_B, te_mae_C = exp["mae_A"], exp["mae_B"], exp["mae_C"]
    vl_A,     vl_B,     vl_C     = exp["vl_A"],  exp["vl_B"],  exp["vl_C"]
    pred_A,   pred_C              = exp["pred_A"], exp["pred_C"]
    recov                         = exp["recovery_pct"]
    model_C_features              = exp["model_C_features"]
    model_names = ["A\n(full)", "B\n(chem only)", "C\n(chem+P)"]
    bar_colors  = [CBLUE, CGRAY, CGREEN]

    # Exp 1A — R² bar
    fig, ax = plt.subplots(figsize=(6, 5))
    r2_vals = [te_r2_A, te_r2_B, te_r2_C]
    bars = ax.bar(model_names, r2_vals, color=bar_colors, alpha=0.82, width=0.5)
    ax.axhline(te_r2_A, color=CBLUE, ls="--", lw=1.2, alpha=0.6, label="Model A baseline")
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Test R²", fontsize=12)
    ax.set_title("Exp 1: R² comparison — A vs B vs C", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(r2_vals) * 1.08)   # start at 0 — do not truncate axis
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_figure(fig, os.path.join(config.FIG_HKL_DIR, "exp1a_r2_comparison.png"))
    plt.close(fig)

    # Exp 1B — MAE bar (J/m²)
    fig, ax = plt.subplots(figsize=(6, 5))
    mae_jm2 = [to_jm2(te_mae_A), to_jm2(te_mae_B), to_jm2(te_mae_C)]
    bars = ax.bar(model_names, mae_jm2, color=bar_colors, alpha=0.82, width=0.5)
    ax.axhline(mae_jm2[0], color=CBLUE, ls="--", lw=1.2, alpha=0.6, label="Model A baseline")
    for bar, val in zip(bars, mae_jm2):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(mae_jm2) * 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel(f"Test MAE  ({UNIT_SHORT})", fontsize=12)
    ax.set_title("Exp 1: MAE comparison — A vs B vs C", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(mae_jm2) * 1.08)   # start at 0 — do not truncate axis
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_figure(fig, os.path.join(config.FIG_HKL_DIR, "exp1b_mae_comparison.png"))
    plt.close(fig)

    # Exp 1C — geometry signal recovery
    fig, ax = plt.subplots(figsize=(6, 5))
    geom_lost  = te_r2_A - te_r2_B
    geom_recov = te_r2_C - te_r2_B
    geom_miss  = geom_lost - geom_recov
    labels_ = ["Lost by\nremoving hkl\n(A→B)", "Recovered\nby P(hkl)\n(B→C)", "Still\nmissing\n(C vs A)"]
    values_ = [geom_lost, geom_recov, geom_miss]
    colors_ = [CRED, CGREEN, CAMBER]
    bars = ax.bar(labels_, values_, color=colors_, alpha=0.82, width=0.5)
    for bar, val in zip(bars, values_):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(values_) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("ΔR²", fontsize=12)
    ax.set_title(f"Exp 1: Geometry signal\nP recovers {recov:.1f}% of hkl info",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(values_) * 1.18)   # start at 0 — do not truncate axis
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    save_figure(fig, os.path.join(config.FIG_HKL_DIR, "exp1c_geometry_recovery.png"))
    plt.close(fig)

    # Save Model A / B / C + feature lists for the downstream Wulff-shape
    # prediction step (predicting gamma for every compound x hkl combination).
    MODEL_DIR = getattr(config, "MODEL_DIR", os.path.join(config.PROJECT_ROOT, "models"))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(exp["model_A"], os.path.join(MODEL_DIR, "model_A_full.pkl"))
    joblib.dump(exp["model_B"], os.path.join(MODEL_DIR, "model_B_chem.pkl"))
    joblib.dump(exp["model_C"], os.path.join(MODEL_DIR, "model_C.pkl"))
    joblib.dump(feature_cols,   os.path.join(MODEL_DIR, "feats_A_full.pkl"))
    joblib.dump(chem_features,  os.path.join(MODEL_DIR, "feats_B_chem.pkl"))
    joblib.dump(model_C_features, os.path.join(MODEL_DIR, "feats_C.pkl"))
    print(f"Saved Model A/B/C + feature lists to {MODEL_DIR}")

    # ── Experiment 2 — SHAP importance of P(h,k,l) in Model C ──────────
    banner("EXPERIMENT 2 — SHAP importance of P(hkl) in Model C", char="-")
    shap_res  = shap_model_c(exp["model_C"], X_test_p, model_C_features)
    shap_mean = shap_res["shap_mean"]
    p_rank    = shap_res["p_rank"]
    p_cols    = [c for c in shap_mean.index if c.startswith("P_hkl")]

    # Exp 2A — SHAP bar chart (top 15)
    fig, ax = plt.subplots(figsize=(9, 6))
    top15 = shap_mean.head(15)
    bar_shap_cols = [CGREEN if f in p_cols else CBLUE for f in top15.index]
    ax.barh(range(len(top15)), top15.values, color=bar_shap_cols, alpha=0.80)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15.index, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(f"Exp 2: SHAP importance in Model C  |  P(hkl) best rank = #{p_rank}",
                 fontsize=12, fontweight="bold")
    p_in_top15 = [f for f in top15.index if f in p_cols]
    if p_in_top15:
        p_idx = list(top15.index).index(p_in_top15[0])
        ax.text(top15.iloc[p_idx] + top15.max() * 0.01, p_idx,
                f"  P(h,k,l) — rank #{p_rank}",
                va="center", fontsize=9, color=CGREEN, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    save_figure(fig, os.path.join(config.FIG_HKL_DIR, "exp2a_shap_model_c.png"))
    plt.close(fig)

    # Exp 2B — Model C parity (J/m²)
    fig, ax = plt.subplots(figsize=(6, 6))
    y_true     = np.array(splits["y_test"])
    y_true_jm2 = to_jm2(y_true)
    pred_C_jm2 = to_jm2(pred_C)
    lim = [min(y_true_jm2.min(), pred_C_jm2.min()),
           max(y_true_jm2.max(), pred_C_jm2.max())]
    ax.scatter(y_true_jm2, pred_C_jm2, alpha=0.25, s=8, color=CGREEN, rasterized=True)
    ax.plot(lim, lim, "k--", lw=1.2, label="Perfect")
    ax.set_xlabel(f"True γ  ({UNIT_SHORT})", fontsize=12)
    ax.set_ylabel(f"Predicted γ  ({UNIT_SHORT})", fontsize=12)
    ax.set_title(f"Exp 2: Model C parity\nR²={te_r2_C:.4f}  MAE={to_jm2(te_mae_C):.4f} {UNIT_SHORT}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save_figure(fig, os.path.join(config.FIG_HKL_DIR, "exp2b_model_c_parity.png"))
    plt.close(fig)

    # ── Experiment 3 — Residual test ────────────────────────────────────
    banner("EXPERIMENT 3 — Residual test", char="-")
    resid_records, resid_A, resid_C = residual_analysis(
        pred_A, pred_C, splits["y_test"], splits["X_test"])

    rA_jm2 = to_jm2(resid_A)
    rC_jm2 = to_jm2(resid_C)

    for rec in resid_records:
        feat   = rec["feature"]
        x_feat = splits["X_test"][feat].values
        mask   = np.isfinite(x_feat)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x_feat[mask], rA_jm2[mask],
                   alpha=0.15, s=6, color=CGRAY,  label="Model A resid", rasterized=True)
        ax.scatter(x_feat[mask], rC_jm2[mask],
                   alpha=0.20, s=6, color=CGREEN, label="Model C resid", rasterized=True)

        for resid_jm2, col_, lbl in [(rA_jm2, CGRAY, "A"), (rC_jm2, CGREEN, "C")]:
            m_, b_, r_, _, _ = stats.linregress(x_feat[mask], resid_jm2[mask])
            xs = np.linspace(x_feat[mask].min(), x_feat[mask].max(), 100)
            ax.plot(xs, m_ * xs + b_, color=col_, lw=2, label=f"trend {lbl}: r={r_:+.3f}")

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(feat, fontsize=12)
        ax.set_ylabel(f"Residual  ({UNIT_SHORT})", fontsize=12)
        ax.set_title(f"Exp 3: Residuals vs {feat}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        safe = feat.replace(" ", "_").replace("/", "_")
        save_figure(fig, os.path.join(config.FIG_HKL_DIR, f"exp3_residuals_{safe}.png"))
        plt.close(fig)

    # ── Experiment 4 — Wulff-shape prediction + top-facet comparison ───
    banner("EXPERIMENT 4 — Wulff-shape prediction + top-facet comparison", char="-")

    cat_categories = {col: X[col].cat.categories for col in config.CAT_COLS}
    WULFF_OUT_DIR = os.path.join(config.RESULTS_DIR, "hkl_effect_wulff")

    pred_paths = predict_all_facets_abc(
        models={"A": exp["model_A"], "B": exp["model_B"], "C": exp["model_C"]},
        feats={"A": feature_cols, "B": chem_features, "C": model_C_features},
        medians={"A": exp["med_A"], "B": exp["med_B"], "C": exp["med_C"]},
        bulk_pr=bulk_pr, targets=targets,
        cat_cols=config.CAT_COLS, cat_categories=cat_categories,
        out_dir=WULFF_OUT_DIR,
    )

    if api_key:
        master_paths = build_wulff_shapes_abc(
            pred_paths, out_dir=WULFF_OUT_DIR, api_key=api_key,
        )

        facet_df = compare_top_facets_abc(master_paths, out_dir=WULFF_OUT_DIR, n=3)

        if len(facet_df):
            plot_facet_comparison_summary(
                facet_df,
                out_path=os.path.join(config.FIG_HKL_DIR, "exp4_facet_comparison_summary.png"),
                n=3,
            )
        else:
            print("No compounds present in all 3 Wulff outputs — skipping summary figure "
                  "(check your API key / network access).")
    else:
        print("Skipped Wulff shapes and facet comparison (no API key).")

    # ── Save CSVs ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame([
        {"model": "A_full",           "n_features": len(feature_cols),
         "val_mae_jm2": to_jm2(vl_A),  "test_mae_jm2": to_jm2(te_mae_A), "test_r2": te_r2_A,
         "val_mae_ev":  vl_A,           "test_mae_ev":  te_mae_A},
        {"model": "B_chem_only",       "n_features": len(chem_features),
         "val_mae_jm2": to_jm2(vl_B),  "test_mae_jm2": to_jm2(te_mae_B), "test_r2": te_r2_B,
         "val_mae_ev":  vl_B,           "test_mae_ev":  te_mae_B},
        {"model": "C_chem_plus_Phkl",  "n_features": len(model_C_features),
         "val_mae_jm2": to_jm2(vl_C),  "test_mae_jm2": to_jm2(te_mae_C), "test_r2": te_r2_C,
         "val_mae_ev":  vl_C,           "test_mae_ev":  te_mae_C},
    ])
    results_df.to_csv(os.path.join(config.RESULTS_DIR, "phkl_experiments.csv"), index=False)

    y_true = np.array(splits["y_test"])
    resid_df = pd.DataFrame({
        "y_true_jm2":  to_jm2(y_true), "y_true_ev":   y_true,
        "pred_A_jm2":  to_jm2(pred_A), "pred_C_jm2":  to_jm2(pred_C),
        "resid_A_jm2": to_jm2(resid_A),"resid_C_jm2": to_jm2(resid_C),
    })
    resid_df.to_csv(os.path.join(config.RESULTS_DIR, "phkl_residuals.csv"), index=False)
    print(f"Saved CSVs to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
