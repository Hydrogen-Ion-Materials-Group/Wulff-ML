"""
P(h,k,l) broken-bond surface-density analysis.

Demonstrates that h_norm/k_norm/l_norm correlate with surface energy
*through* the broken-bond surface density P(h,k,l) = Nb(hkl) / A(hkl):

    (h_norm, k_norm, l_norm)  ->  P(h,k,l)  ->  gamma(hkl)

Four experiments:
    Exp 1 — Model A (full) vs B (chem-only) vs C (chem + P(hkl))
    Exp 2 — SHAP importance of P(hkl) inside Model C
    Exp 3 — Residual test: do Model C residuals still carry hkl trends?
    Exp 4 — Wulff-shape prediction + top-facet comparison across A / B / C

References:
    Galanakis et al. (2002) Surf. Sci. 511, 1-12  — broken-bond rule
    Wulff (1901)  — equilibrium crystal shape from gamma(hkl) anisotropy
    Herring (1951) Phys. Rev. 82, 87
"""
import ast
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from wulff_ml import config
from wulff_ml.units import to_jm2
from wulff_ml.utils import clean_inf_nan_generic
from wulff_ml.cache import cached
from wulff_ml.api_key import get_mp_api_key

# ── Colour palette (shared with scripts/04_hkl_analysis.py) ─────────────────
CBLUE  = "#2471A3"
CRED   = "#C0392B"
CGREEN = "#1A7431"
CAMBER = "#B7770D"
CGRAY  = "#7F8C8D"


# ==============================================================================
# P(h,k,l) — direction-aware descriptor
# ==============================================================================
P_HKL_COLS = ["P_hkl_mag", "P_hkl_h", "P_hkl_k", "P_hkl_l"]


def compute_P_hkl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Direction-aware P(h,k,l) descriptor.

    A single magnitude collapses (h,k,l) down to |hkl|, so it can't
    distinguish e.g. (1,1,1) from (1,-1,-1) — they share the same |hkl|.
    This descriptor keeps the magnitude *and* the unit direction, so a
    model can still tell facets apart even when |hkl| ties:

      P_hkl_mag   sqrt(h²+k²+l²) / volume_per_atom^(2/3)   (broken-bond density)
      P_hkl_h     h / |hkl|   (unit-direction component)
      P_hkl_k     k / |hkl|
      P_hkl_l     l / |hkl|
    """
    df = df.copy()
    h = df["h"].values.astype(float)
    k = df["k"].values.astype(float)
    l = df["l"].values.astype(float)

    mag = np.sqrt(h**2 + k**2 + l**2)
    vol = df["volume_per_atom"].clip(lower=1e-3).values
    safe_mag = np.clip(mag, 1e-9, None)

    df["P_hkl_mag"] = mag / vol ** (2 / 3)
    df["P_hkl_h"] = h / safe_mag
    df["P_hkl_k"] = k / safe_mag
    df["P_hkl_l"] = l / safe_mag
    return df


# ==============================================================================
# Correlation helpers
# ==============================================================================
def _pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return 0.0, 1.0
    return stats.pearsonr(x[mask], y[mask])


def _partial_corr(x, y, z):
    """r(x, y | z) — Pearson on residuals after linearly removing z."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask.sum() < 5:
        return 0.0
    x, y, z = x[mask], y[mask], z[mask]
    xr = x - (stats.linregress(z, x).slope * z + stats.linregress(z, x).intercept)
    yr = y - (stats.linregress(z, y).slope * z + stats.linregress(z, y).intercept)
    return stats.pearsonr(xr, yr)[0]


# ==============================================================================
# Experiment 1 helpers — mediation analysis
# ==============================================================================
@cached("run_phkl_analysis")
def run_phkl_analysis(df: pd.DataFrame, gamma_col: str = "surface_energy",
                       sample_n: int = 5000, random_state: int = 42) -> dict:
    """
    P(h,k,l) mediation analysis on the test set:
      (a) compute P(h,k,l)
      (b) correlate k_norm→P_hkl and P_hkl→gamma
      (c) partial correlation of k_norm vs gamma after removing P_hkl
    Returns a dict with keys: df, df_plot, stats.
    """
    df = compute_P_hkl(df)
    df_plot = df.sample(sample_n, random_state=random_state) if len(df) > sample_n else df.copy()

    gamma = df_plot[gamma_col].values
    k_n   = df_plot["k_norm"].values
    P     = df_plot["P_hkl_mag"].values

    r_kn_P,  p_kn_P  = _pearson(k_n,  P)
    r_P_g,   p_P_g   = _pearson(P,    gamma)
    r_kn_g,  p_kn_g  = _pearson(k_n,  gamma)
    r_partial = _partial_corr(k_n, gamma, P)

    stats_dict = dict(
        r_kn_P=r_kn_P,   p_kn_P=p_kn_P,
        r_P_g=r_P_g,     p_P_g=p_P_g,
        r_kn_g=r_kn_g,   p_kn_g=p_kn_g,
        r_partial_kn_g_given_P=r_partial,
    )

    print("P(h,k,l) mediation analysis:")
    print(f"  corr(k_norm,  P_hkl)               = {r_kn_P:+.4f}  (p={p_kn_P:.2e})")
    print(f"  corr(P_hkl,   gamma)               = {r_P_g:+.4f}  (p={p_P_g:.2e})")
    print(f"  corr(k_norm,  gamma)               = {r_kn_g:+.4f}  (p={p_kn_g:.2e})")
    print(f"  partial corr(k_norm, gamma | P_hkl) = {r_partial:+.4f}")
    print("  → small |partial| means P(h,k,l) explains most of the k_norm–gamma link.")
    return dict(df=df, df_plot=df_plot, stats=stats_dict)


# ==============================================================================
# Experiment 1 — train Models A, B, C
# ==============================================================================
@cached("three_model_experiment")
def three_model_experiment(X_train, X_val, X_test,
                            y_train, y_val, y_test,
                            feature_cols, chem_features, params,
                            random_state=22):
    """
    Model A (full) vs B (chemistry-only) vs C (chemistry + P(h,k,l)),
    where P(h,k,l) is the direction-aware descriptor (magnitude + unit
    direction h/|hkl|, k/|hkl|, l/|hkl|).

    Returns a dict with fitted models, predictions, MAE, R², recovery_pct,
    val_mae per model, and the Model-C feature list.
    """
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score

    def _train_eval(feats, X_tr, X_vl, X_te):
        Xtr, med = clean_inf_nan_generic(X_tr[feats])
        Xvl, _   = clean_inf_nan_generic(X_vl[feats], med)
        Xte, _   = clean_inf_nan_generic(X_te[feats], med)
        m = xgb.XGBRegressor(**{**params, "random_state": random_state})
        m.fit(Xtr, y_train, eval_set=[(Xvl, y_val)], verbose=False)
        pred_vl = m.predict(Xvl)
        pred_te = m.predict(Xte)
        return (m, pred_te, med,
                mean_absolute_error(y_val,  pred_vl),
                mean_absolute_error(y_test, pred_te),
                r2_score(y_test, pred_te))

    model_A, pred_A, med_A, vl_A, mae_A, r2_A = _train_eval(feature_cols,  X_train, X_val, X_test)
    model_B, pred_B, med_B, vl_B, mae_B, r2_B = _train_eval(chem_features, X_train, X_val, X_test)

    p_cols = [c for c in P_HKL_COLS if c in X_train.columns]
    model_C_features = chem_features + p_cols
    model_C, pred_C, med_C, vl_C, mae_C, r2_C = _train_eval(model_C_features, X_train, X_val, X_test)

    recovery_pct = 0.0
    denom = r2_A - r2_B
    if abs(denom) > 1e-9:
        recovery_pct = (r2_C - r2_B) / denom * 100

    print(f"Model A (full)       : val_mae={vl_A:.5f}  test_mae={mae_A:.5f}  R²={r2_A:.4f}")
    print(f"Model B (chem-only)  : val_mae={vl_B:.5f}  test_mae={mae_B:.5f}  R²={r2_B:.4f}")
    print(f"Model C (chem+P_hkl) : val_mae={vl_C:.5f}  test_mae={mae_C:.5f}  R²={r2_C:.4f}")
    print(f"P(hkl) recovers {recovery_pct:.1f}% of the geometry signal lost by removing hkl")

    return dict(
        model_A=model_A, model_B=model_B, model_C=model_C,
        pred_A=pred_A,   pred_B=pred_B,   pred_C=pred_C,
        med_A=med_A,     med_B=med_B,     med_C=med_C,
        vl_A=vl_A,       vl_B=vl_B,       vl_C=vl_C,
        mae_A=mae_A,     mae_B=mae_B,     mae_C=mae_C,
        r2_A=r2_A,       r2_B=r2_B,       r2_C=r2_C,
        recovery_pct=recovery_pct,
        model_C_features=model_C_features,
        feature_cols=feature_cols,
        chem_features=chem_features,
    )


# ==============================================================================
# Experiment 2 — SHAP importance of P(h,k,l) inside Model C
# ==============================================================================
@cached("shap_model_c")
def shap_model_c(model_C, X_test, model_C_features):
    """
    Compute SHAP values for Model C and return:
      shap_mean   — pd.Series of mean |SHAP| per feature, sorted descending
      p_rank      — best rank among the P(h,k,l) feature columns (1-based)
      p_shap      — mean |SHAP| of the top-ranked P(h,k,l) feature
    """
    import shap
    Xte, _ = clean_inf_nan_generic(X_test[model_C_features])
    explainer = shap.TreeExplainer(model_C)
    shap_vals = explainer(Xte)
    shap_mean = pd.Series(
        np.abs(shap_vals.values).mean(axis=0),
        index=model_C_features,
    ).sort_values(ascending=False)

    p_cols = [c for c in P_HKL_COLS if c in shap_mean.index]
    ranked = shap_mean.index.tolist()
    p_rank = min((ranked.index(c) + 1 for c in p_cols), default=None)
    p_shap = shap_mean[p_cols].max() if p_cols else 0.0

    print(f"\nSHAP analysis — Model C:")
    print(f"  P(h,k,l) best rank        : #{p_rank} of {len(model_C_features)} features")
    print(f"  P(h,k,l) mean |SHAP|      : {p_shap:.5f}")
    print(f"  Top feature               : {shap_mean.index[0]}  ({shap_mean.iloc[0]:.5f})")
    print(f"  P(h,k,l) / top            : {p_shap / (shap_mean.iloc[0] + 1e-12) * 100:.1f}%")
    print(f"\n  Top 15 features in Model C:")
    print(f"  {'Rank':<5}  {'Feature':<42}  {'Mean |SHAP|':>12}")
    print("  " + "─" * 62)
    for rank, (feat, val) in enumerate(shap_mean.head(15).items(), 1):
        tag = "  ← P(h,k,l)" if feat in p_cols else ""
        print(f"  {rank:<5d}  {feat:<42}  {val:>12.5f}{tag}")

    return dict(shap_vals=shap_vals, shap_mean=shap_mean, p_rank=p_rank, p_shap=p_shap)


# ==============================================================================
# Experiment 3 — Residual test
# ==============================================================================
def residual_analysis(pred_A, pred_C, y_test, X_test):
    """
    Correlate Model A and Model C residuals with raw hkl descriptors.
    Returns a list of dicts with keys: feature, rA, pA, rC, pC, change.
    """
    y_true  = np.array(y_test)
    resid_A = y_true - np.array(pred_A)
    resid_C = y_true - np.array(pred_C)

    hkl_check = ["h_norm", "k_norm", "l_norm", "hkl_magnitude", "n_zeros", "max_index"]
    hkl_check = [f for f in hkl_check if f in X_test.columns]

    print(f"\nExp 3 — Residual correlations with raw hkl descriptors:")
    print(f"  {'Feature':<20}  {'r(A, feat)':>12}  {'r(C, feat)':>12}  {'Δr':>8}")
    print("  " + "─" * 58)

    records = []
    for feat in hkl_check:
        x    = X_test[feat].values
        mask = np.isfinite(x) & np.isfinite(resid_A) & np.isfinite(resid_C)
        rA, pA = stats.pearsonr(resid_A[mask], x[mask])
        rC, pC = stats.pearsonr(resid_C[mask], x[mask])
        change = rC - rA
        flag   = "  ← trend remains" if abs(rC) > 0.10 else ""
        print(f"  {feat:<20}  {rA:>+12.4f}  {rC:>+12.4f}  {change:>+8.4f}{flag}")
        records.append(dict(feature=feat, rA=rA, pA=pA, rC=rC, pC=pC, change=change))

    mean_rA = np.mean([abs(r["rA"]) for r in records])
    mean_rC = np.mean([abs(r["rC"]) for r in records])
    print(f"\n  Mean |r| — Model A residuals : {mean_rA:.4f}")
    print(f"  Mean |r| — Model C residuals : {mean_rC:.4f}")

    if mean_rC < 0.05:
        print("  → Residuals flat. P(h,k,l) fully captured the orientation signal.")
    elif mean_rC < 0.15:
        print("  → Weak hkl trends remain. P captures most but not all.")
    else:
        print("  → Strong hkl trends remain in residuals. P is incomplete.")

    return records, resid_A, resid_C


# ==============================================================================
# Experiment 4a — predict gamma for every (compound x hkl) with Models A/B/C
# ==============================================================================
def predict_all_facets_abc(models: dict, feats: dict, medians: dict,
                            bulk_pr: pd.DataFrame, targets: pd.DataFrame,
                            cat_cols: list, cat_categories: dict,
                            out_dir: str) -> dict:
    """
    Predict gamma for every (compound x hkl) combination in the prediction
    pool using Model A (full), Model B (chem-only) and Model C (chem +
    P(h,k,l)), saving one CSV per model.

    models / feats / medians are dicts keyed "A", "B", "C" — typically the
    corresponding entries returned by three_model_experiment() (model_A/
    model_B/model_C, feature_cols/chem_features/model_C_features, and
    med_A/med_B/med_C respectively).

    Returns {"A": csv_path, "B": csv_path, "C": csv_path}.
    """
    from wulff_ml.feature_engineering import expand_hkl_column

    os.makedirs(out_dir, exist_ok=True)
    all_hkl = targets["hkl"].unique().tolist()
    print(f"Unique hkl in training set : {len(all_hkl)}")

    df_hkl  = pd.DataFrame({"hkl": all_hkl})
    df_pred = bulk_pr.merge(df_hkl, how="cross")
    print(f"Total prediction rows      : {len(df_pred)} "
          f"({bulk_pr.shape[0]} compounds x {len(all_hkl)} hkl)")

    df_pred = expand_hkl_column(df_pred, "hkl")
    for col in cat_cols:
        df_pred[col] = pd.Categorical(df_pred[col].astype(str), categories=cat_categories[col])

    # Model C needs the direction-aware P(h,k,l) descriptor
    df_pred = compute_P_hkl(df_pred)

    paths = {}
    for label in ("A", "B", "C"):
        X_pred, _ = clean_inf_nan_generic(df_pred[feats[label]].copy(), fill_values=medians[label])
        preds = models[label].predict(X_pred)

        df_out = df_pred[["mp_id", "formula_pretty", "hkl"]].copy()
        df_out["pred_gamma_eV_per_ang2"] = preds
        df_out["pred_gamma_J_per_m2"] = to_jm2(preds)

        path = os.path.join(out_dir, f"{label}_pred.csv")
        df_out.to_csv(path, index=False)
        print(f"Saved: {path}  ({len(df_out)} rows)")
        paths[label] = path

    return paths


# ==============================================================================
# Experiment 4b — build Wulff shapes for Models A/B/C
# ==============================================================================
@cached("wulff_shape_for_compound_hkl_effect", ignore_kwargs=["api_key"])
def _build_one_wulff_shape_hkl(mpid: str, group_records: tuple, api_key: str = None):
    """
    Pure computation for a single compound: fetch its structure once, build
    a WulffShape from the given (hkl, gamma) pairs, and compute area
    fractions + the top-5 dominant facets. Cached per (mpid, gamma set) so
    re-running for a different model (different predicted gammas), or after
    an interruption, only re-fetches/re-computes what's missing.
    Returns a dict, or {"error": ...} on failure.
    """
    from mp_api.client import MPRester
    from pymatgen.analysis.wulff import WulffShape

    api_key = api_key or get_mp_api_key()
    try:
        with MPRester(api_key) as mpr:
            structure = mpr.get_structure_by_material_id(mpid, conventional_unit_cell=True)
        formula = structure.composition.reduced_formula

        millers, gammas = [], []
        for hkl_str, gamma in group_records:
            millers.append(tuple(ast.literal_eval(hkl_str)))
            gammas.append(float(gamma))

        wulff = WulffShape(lattice=structure.lattice, miller_list=millers, e_surf_list=gammas)

        area_fraction_rows = [
            {"mp_id": mpid, "formula_pretty": formula, "hkl": str(hkl), "area_fraction": round(float(frac), 8)}
            for hkl, frac in wulff.area_fraction_dict.items()
        ]
        top5_area_fractions = sorted(
            [(hkl, frac) for hkl, frac in wulff.area_fraction_dict.items() if frac > 0],
            key=lambda x: x[1], reverse=True,
        )[:5]

        return {
            "formula": formula,
            "cif_text": structure.to(fmt="cif"),
            "lattice_dict": structure.lattice.as_dict(),
            "miller_list": millers,
            "e_surf_list": gammas,
            "area_fraction_rows": area_fraction_rows,
            "top5_area_fractions": top5_area_fractions,
        }
    except Exception as e:
        return {"error": str(e)}


def build_wulff_shapes_abc(pred_paths: dict, out_dir: str, api_key: str = None,
                            save_dpi: int = 200) -> dict:
    """
    Build Wulff equilibrium shapes for Models A, B and C from the CSVs
    produced by predict_all_facets_abc(). For every compound, saves (per
    model, under <out_dir>/<label>_wulff_output/):
        <mpid>.cif
        <mpid>_<formula>_shape.png    pure Wulff shape, no text, transparent bg
        <mpid>_<formula>_legend.png   standalone top-5 area-fraction legend, transparent bg
    plus one master area-fraction table per model:
        <label>_area_fractions_wulff.csv

    The expensive part (structure fetch + WulffShape construction) is cached
    per-compound via _build_one_wulff_shape_hkl; only plot rendering (cheap,
    no network) always reruns.

    Returns {"A": master_csv_path, "B": master_csv_path, "C": master_csv_path}.
    """
    from pymatgen.core import Lattice
    from pymatgen.analysis.wulff import WulffShape

    api_key = api_key or get_mp_api_key()
    master_paths = {}

    for label, pred_csv in pred_paths.items():
        model_out_dir = os.path.join(out_dir, f"{label}_wulff_output")
        os.makedirs(model_out_dir, exist_ok=True)

        df = pd.read_csv(pred_csv)
        all_results = []

        for mpid, group in df.groupby("mp_id"):
            print(f"[{label}] Processing {mpid}")
            group_records = tuple(
                (row["hkl"], row["pred_gamma_eV_per_ang2"]) for _, row in group.iterrows()
            )
            result = _build_one_wulff_shape_hkl(mpid, group_records, api_key=api_key)
            if "error" in result:
                print(f"  Failed for {mpid}: {result['error']}")
                continue

            formula = result["formula"]
            safe_formula = formula.replace("/", "_")

            cif_path = os.path.join(model_out_dir, f"{mpid}.cif")
            with open(cif_path, "w") as f:
                f.write(result["cif_text"])

            # Rebuild the WulffShape from cached inputs purely to render the
            # plots (cheap — no network call, no re-fetch).
            lattice = Lattice.from_dict(result["lattice_dict"])
            wulff = WulffShape(lattice=lattice, miller_list=result["miller_list"],
                                e_surf_list=result["e_surf_list"])

            color_list, *_ = wulff._get_colors(
                color_set="PuBu", alpha=1.0, off_color="red", custom_colors={})
            color_dict = {hkl: color_list[i] for i, hkl in enumerate(wulff.hkl_list)}

            # ---- Plot 1: pure Wulff shape, no text, transparent background ----
            ax_wulff = wulff.get_plot()
            fig_shape = ax_wulff.figure
            legend = ax_wulff.get_legend()
            if legend:
                legend.remove()
            shape_path = os.path.join(model_out_dir, f"{mpid}_{safe_formula}_shape.png")
            fig_shape.savefig(shape_path, dpi=save_dpi, bbox_inches="tight", transparent=True)
            plt.close(fig_shape)

            # ---- Plot 2: standalone top-5 area-fraction legend, transparent ----
            top5 = result["top5_area_fractions"]
            if top5:
                handles = [
                    mpatches.Patch(facecolor=color_dict.get(hkl, "grey"), edgecolor="black",
                                   linewidth=0.8, label=f"{hkl}: {frac:.3f}")
                    for hkl, frac in top5
                ]
                fig_leg, ax_leg = plt.subplots(figsize=(3.3, 2.0))
                ax_leg.axis("off")
                ax_leg.legend(handles=handles, loc="center", fontsize=8, frameon=False,
                              handlelength=2.0, handleheight=1.4, labelspacing=0.4)
                legend_path = os.path.join(model_out_dir, f"{mpid}_{safe_formula}_legend.png")
                fig_leg.savefig(legend_path, dpi=save_dpi, bbox_inches="tight", transparent=True)
                plt.close(fig_leg)

            all_results.extend(result["area_fraction_rows"])

        results_df = pd.DataFrame(all_results)
        master_csv = os.path.join(model_out_dir, f"{label}_area_fractions_wulff.csv")
        results_df.to_csv(master_csv, index=False)
        print(f"[{label}] Master area fractions CSV saved -> {master_csv}")
        master_paths[label] = master_csv

    return master_paths


# ==============================================================================
# Experiment 4c — compare top-n dominant facets across Models A/B/C
# ==============================================================================
def compare_top_facets_abc(master_paths: dict, out_dir: str, n: int = 3) -> pd.DataFrame:
    """
    Compare the top-n dominant Wulff facets across Models A, B, C for every
    compound present in all three model outputs.

    Saves and returns the per-compound comparison table (top3_facet_comparison.csv):
      top{n}_A/B/C             the n most dominant exposed facets per model
      top{n}_changed_A_vs_B    did the dominant facet SET change once hkl
                                descriptors are removed (A -> B)?
      top{n}_exact_match_C_A/B does Model C's top-n set exactly match A or B?
      jaccard_A_B/A_C/B_C      pairwise Jaccard similarity of the top-n sets
      model_C_closer_to        which baseline (A or B) Model C's facet set
                                resembles more closely, by Jaccard similarity
    """
    df_A = pd.read_csv(master_paths["A"])
    df_B = pd.read_csv(master_paths["B"])
    df_C = pd.read_csv(master_paths["C"])

    def top_n_facets(df, mp_id):
        sub = df[(df["mp_id"] == mp_id) & (df["area_fraction"] > 0)]
        sub = sub.sort_values("area_fraction", ascending=False)
        return sub.head(n)["hkl"].tolist()

    def jaccard(list1, list2):
        s1, s2 = set(list1), set(list2)
        if not s1 and not s2:
            return 1.0
        return len(s1 & s2) / len(s1 | s2)

    compounds = sorted(set(df_A["mp_id"]) & set(df_B["mp_id"]) & set(df_C["mp_id"]))
    print(f"Compounds present in all 3 model outputs: {len(compounds)}")

    records = []
    for mp_id in compounds:
        formula = df_A.loc[df_A["mp_id"] == mp_id, "formula_pretty"].iloc[0]

        top_A, top_B, top_C = (top_n_facets(df_A, mp_id),
                                top_n_facets(df_B, mp_id),
                                top_n_facets(df_C, mp_id))
        set_A, set_B, set_C = set(top_A), set(top_B), set(top_C)

        jac_AB, jac_AC, jac_BC = (jaccard(top_A, top_B),
                                   jaccard(top_A, top_C),
                                   jaccard(top_B, top_C))
        if jac_AC > jac_BC:
            c_closer_to = "A"
        elif jac_BC > jac_AC:
            c_closer_to = "B"
        else:
            c_closer_to = "tie"

        records.append({
            "mp_id": mp_id, "formula_pretty": formula,
            f"top{n}_A": top_A, f"top{n}_B": top_B, f"top{n}_C": top_C,
            f"top{n}_changed_A_vs_B":  set_A != set_B,
            f"top{n}_exact_match_C_A": set_C == set_A,
            f"top{n}_exact_match_C_B": set_C == set_B,
            "jaccard_A_B": jac_AB, "jaccard_A_C": jac_AC, "jaccard_B_C": jac_BC,
            "model_C_closer_to": c_closer_to,
        })

    results_df = pd.DataFrame(records)
    results_path = os.path.join(out_dir, "top3_facet_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    n_total = len(results_df)
    if n_total:
        n_changed = int(results_df[f"top{n}_changed_A_vs_B"].sum())
        print("\n" + "=" * 70)
        print(f"SUMMARY — Top-{n} Facet Composition: Model A (full) vs Model B (bulk-only)")
        print("=" * 70)
        print(f"Total compounds analyzed             : {n_total}")
        print(f"Top-{n} facet set CHANGED   (A vs B)    : {n_changed:4d}  "
              f"({100*n_changed/n_total:5.1f}%)")
        print(f"Top-{n} facet set UNCHANGED (A vs B)    : {n_total-n_changed:4d}  "
              f"({100*(n_total-n_changed)/n_total:5.1f}%)")
        print(f"\nMean Jaccard(A,B) : {results_df['jaccard_A_B'].mean():.3f}")
        print(f"Mean Jaccard(A,C) : {results_df['jaccard_A_C'].mean():.3f}")
        print(f"Mean Jaccard(B,C) : {results_df['jaccard_B_C'].mean():.3f}")

        closer_counts = results_df["model_C_closer_to"].value_counts()
        n_C_exact_A = int(results_df[f"top{n}_exact_match_C_A"].sum())
        n_C_exact_B = int(results_df[f"top{n}_exact_match_C_B"].sum())
        print(f"\nModel C top-{n} closer to A (Jaccard) : {closer_counts.get('A', 0):4d} compounds")
        print(f"Model C top-{n} closer to B (Jaccard) : {closer_counts.get('B', 0):4d} compounds")
        print(f"Tie                                   : {closer_counts.get('tie', 0):4d} compounds")
        print(f"\nModel C top-{n} EXACTLY matches A     : {n_C_exact_A:4d}  ({100*n_C_exact_A/n_total:5.1f}%)")
        print(f"Model C top-{n} EXACTLY matches B     : {n_C_exact_B:4d}  ({100*n_C_exact_B/n_total:5.1f}%)")

    return results_df


# ==============================================================================
# Experiment 4d — summary figure for the top-facet comparison
# ==============================================================================
def plot_facet_comparison_summary(results_df: pd.DataFrame, out_path: str, n: int = 3):
    """
    3-panel summary figure for compare_top_facets_abc()'s output:
      1. Per-compound Jaccard similarity — (A,B) vs (A,C) vs (B,C)
      2. Overall count of compounds whose top-n facet SET changed (A vs B)
      3. Which baseline (A or B) Model C's top-n set resembles more closely
    """
    df = results_df.sort_values("formula_pretty").reset_index(drop=True)
    labels = df["formula_pretty"] + "\n(" + df["mp_id"] + ")"
    changed_col = f"top{n}_changed_A_vs_B"

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.55, wspace=0.3)

    # ── Panel 1 — per-compound Jaccard similarity ───────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(df))
    w = 0.26
    ax1.bar(x - w, df["jaccard_A_B"], width=w, color=CBLUE,  label="Jaccard(A, B)")
    ax1.bar(x,     df["jaccard_A_C"], width=w, color=CGREEN, label="Jaccard(A, C)")
    ax1.bar(x + w, df["jaccard_B_C"], width=w, color=CAMBER, label="Jaccard(B, C)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax1.set_ylabel(f"Jaccard similarity (top-{n} facets)", fontsize=10)
    ax1.set_ylim(0, 1.15)
    ax1.set_title(f"Top-{n} Facet Overlap Between Models — Per Compound", fontsize=12, fontweight="bold")
    ax1.axhline(1.0, color="black", lw=0.7, ls="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="upper right", ncol=3)
    ax1.grid(True, alpha=0.25, axis="y")
    for xi, changed in zip(x, df[changed_col]):
        if changed:
            ax1.text(xi, 1.05, "X", ha="center", va="center", fontsize=11, color=CRED, fontweight="bold")
    ax1.text(0.01, 0.98, "X = top-3 set changed (A vs B)", transform=ax1.transAxes,
              fontsize=8.5, color=CRED, va="top")

    # ── Panel 2 — changed vs unchanged (A vs B) ─────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    n_changed   = int(df[changed_col].sum())
    n_unchanged = len(df) - n_changed
    bars = ax2.bar(["Unchanged", "Changed"], [n_unchanged, n_changed],
                    color=[CGRAY, CRED], width=0.5, alpha=0.85)
    for bar, val in zip(bars, [n_unchanged, n_changed]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{val}\n({100*val/len(df):.0f}%)",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Number of compounds", fontsize=10)
    ax2.set_title(f"Top-{n} Facet Composition:\nModel A vs Model B", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, len(df) * 1.25)
    ax2.grid(True, alpha=0.25, axis="y")

    # ── Panel 3 — which baseline does Model C resemble? ─────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    closer_counts = df["model_C_closer_to"].value_counts().reindex(["A", "B", "tie"]).fillna(0)
    bars = ax3.bar(["Closer to A\n(full)", "Closer to B\n(bulk-only)", "Tie"],
                    closer_counts.values, color=[CGREEN, CBLUE, CGRAY], width=0.5, alpha=0.85)
    for bar, val in zip(bars, closer_counts.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{int(val)}\n({100*val/len(df):.0f}%)",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Number of compounds", fontsize=10)
    ax3.set_title("Model C — Which Baseline\nDoes P(hkl) Resemble?", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, len(df) * 1.25)
    ax3.grid(True, alpha=0.25, axis="y")

    plt.suptitle("Effect of hkl-Descriptors on Wulff-Shape Top-Facet Composition",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    from wulff_ml.visualization import save_figure
    save_figure(fig, out_path)
    plt.close(fig)
