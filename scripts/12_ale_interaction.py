#!/usr/bin/env python3
"""
Step 12 — SHAP pairwise interaction analysis.

  1. Group-level interaction matrix  (17×17 heatmap)
  2. Top-30 individual-feature pairwise interaction heatmap (Plot A)

Note: Plot B (zero-interaction features removed) lives in script 13.

Outputs
-------
figures/interaction/
    group_interaction_heatmap.png / .pdf / .svg / _raw.png
    shap_interaction_top30.png / .pdf / .svg / _raw.png   <- Plot A
results/
    group_interaction_matrix.csv
    shap_interaction_top30.csv
    shap_interaction_top20_pairs.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

from wulff_ml import config
from wulff_ml.feature_engineering import load_inputs, build_feature_matrix
from wulff_ml.cross_validation import compound_split
from wulff_ml.model_training import train_xgb
from wulff_ml.shap_analysis import compute_shap_values
from wulff_ml.feature_groups import ALE_GROUPS, resolve_groups
from wulff_ml.units import UNIT_SHORT
from wulff_ml.utils import banner
from wulff_ml.visualization import save_figure
from wulff_ml.shap_interaction_utils import (
    sanitise_dtypes, cluster_order, draw_interaction_plot,
    compute_topk_interaction_matrix, _compute_shap_interaction,
    shorten_feature_name, GROUP_PALETTE,
)


def group_interaction_matrix(shap_interaction_values, feature_cols, groups):
    abs_inter = np.abs(shap_interaction_values).mean(axis=0)
    idx       = {f: i for i, f in enumerate(feature_cols)}
    n_groups  = len(groups)
    mat       = np.zeros((n_groups, n_groups))
    for i, gi in enumerate(groups):
        idx_i = [idx[f] for f in gi["features"] if f in idx]
        for j, gj in enumerate(groups):
            idx_j = [idx[f] for f in gj["features"] if f in idx]
            if not idx_i or not idx_j:
                continue
            mat[i, j] = abs_inter[np.ix_(idx_i, idx_j)].mean()
    labels = [f"[{g['id']}] {g['label']}" for g in groups]
    return pd.DataFrame(mat, index=labels, columns=labels)


def top_feature_interaction_heatmap(model, X_test, shap_values, feature_cols,
                                     raw_groups=ALE_GROUPS, top_k=30, max_samples=500,
                                     random_state=config.RANDOM_STATE,
                                     results_dir=None, figures_dir=None):
    results_dir = results_dir or config.RESULTS_DIR
    figures_dir = figures_dir or config.FIG_INTERACTION_DIR
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Top {top_k} features (by mean |SHAP|):")
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    feat_list     = list(shap_values.feature_names)
    top_idx       = np.argsort(mean_abs_shap)[::-1][:top_k]
    for rank, idx in enumerate(top_idx, 1):
        print(f"  [{rank:02d}] {feat_list[idx]:<45}  mean|SHAP|={mean_abs_shap[idx]:.5f}")

    interaction, top_feat_names, n_sampled = compute_topk_interaction_matrix(
        model, X_test, shap_values, feature_cols,
        top_k=top_k, max_samples=max_samples, random_state=random_state)

    interaction_df  = pd.DataFrame(interaction, index=top_feat_names, columns=top_feat_names)
    interaction_csv = os.path.join(results_dir, f"shap_interaction_top{top_k}.csv")
    interaction_df.to_csv(interaction_csv)
    print(f"Saved: {interaction_csv}")

    lnk_int, ord_int = cluster_order(interaction)
    int_ordered  = interaction[np.ix_(ord_int, ord_int)]
    lab_ordered  = [top_feat_names[i] for i in ord_int]
    lab_short    = [shorten_feature_name(l) for l in lab_ordered]
    feat_to_group = {f: gid for gid, _, feats in raw_groups for f in feats}
    row_colors   = [GROUP_PALETTE.get(feat_to_group.get(lab_ordered[i], ""), "#cccccc")
                    for i in range(top_k)]

    print(f"Drawing Plot A: full top-{top_k} heatmap ...")
    fig_a = draw_interaction_plot(
        int_mat=int_ordered, lab_short_list=lab_short, row_color_list=row_colors,
        lnk=lnk_int, feat_names_list=lab_ordered, raw_groups=raw_groups,
        feat_to_group=feat_to_group,
        title_str=(f"SHAP Pairwise Interaction — Top {top_k} Features "
                   f"(sampled n={n_sampled})\n"
                   r"Surface energy $\gamma$   |   Complete-linkage clustered"
                   r"   |   Colour strip = feature group"),
        figsize=(22, 18),
    )
    out_path_a = os.path.join(figures_dir, f"shap_interaction_top{top_k}.png")
    save_figure(fig_a, out_path_a, dpi=600)
    plt.close(fig_a)

    triu_idx = np.triu_indices(top_k, k=1)
    pairs    = sorted(zip(triu_idx[0], triu_idx[1], interaction[triu_idx]),
                      key=lambda x: x[2], reverse=True)
    pairs_df = pd.DataFrame([
        {"rank": r, "feature_a": top_feat_names[i], "feature_b": top_feat_names[j], "interaction": val}
        for r, (i, j, val) in enumerate(pairs[:20], 1)
    ])
    pairs_csv = os.path.join(results_dir, "shap_interaction_top20_pairs.csv")
    pairs_df.to_csv(pairs_csv, index=False)
    print(f"\nTop 20 strongest pairwise SHAP interactions:")
    print(pairs_df.to_string(index=False))
    print(f"Saved: {pairs_csv}")

    return {"interaction_csv": interaction_csv, "pairs_csv": pairs_csv, "plot_a_path": out_path_a}


def main():
    config.ensure_dirs()
    banner("STEP 12: SHAP PAIRWISE INTERACTION ANALYSIS")

    targets, bulk_tr, _ = load_inputs()
    _, X, y, groups_s, feature_cols, _ = build_feature_matrix(targets, bulk_tr)
    splits = compound_split(X, y, groups_s)
    model  = train_xgb(splits, verbose=False)

    # ── Analysis 1: group-level interaction matrix ───────────────────────
    X_test_clean      = sanitise_dtypes(splits["X_test"])
    shap_interaction  = _compute_shap_interaction(
        model, X_test_clean, len(feature_cols), config.RANDOM_STATE, sample_size=300)

    groups   = resolve_groups(ALE_GROUPS, feature_cols)
    inter_df = group_interaction_matrix(shap_interaction, feature_cols, groups)

    csv_path = os.path.join(config.RESULTS_DIR, "group_interaction_matrix.csv")
    inter_df.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    link    = linkage(inter_df.values, method="average")
    order   = leaves_list(link)
    ordered = inter_df.iloc[order, order]

    plt.figure(figsize=(11, 9))
    sns.heatmap(ordered, cmap="magma", square=True,
                cbar_kws={"label": f"Mean |SHAP interaction| ({UNIT_SHORT})"})
    plt.title("Pairwise Feature-Group SHAP Interaction Strength")
    plt.tight_layout()
    fig_path = os.path.join(config.FIG_INTERACTION_DIR, "group_interaction_heatmap.png")
    save_figure(plt, fig_path, dpi=600)
    plt.close()

    # ── Analysis 2: top-30 individual-feature heatmap (Plot A) ──────────
    shap_values = compute_shap_values(model, splits["X_test"], feature_cols)
    result = top_feature_interaction_heatmap(model, splits["X_test"], shap_values, feature_cols)

    print("\n" + "=" * 70)
    print(" STEP 12 COMPLETE")
    print("=" * 70)
    print(f"  Group heatmap (17x17)  : {fig_path}")
    print(f"  Top-30 heatmap, Plot A : {result['plot_a_path']}")
    print("  (Plot B / filtered     : run scripts/13_pair_wise_interaction_filter.py)")
    print("=" * 70)


if __name__ == "__main__":
    main()
