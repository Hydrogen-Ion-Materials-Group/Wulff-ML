#!/usr/bin/env python3
"""
Step 13 — SHAP top-30 pairwise interaction heatmap, filtered (Plot B).

Drops zero-interaction features from the top-30 set, then draws the
resulting filtered heatmap. Reuses script 12's cached results (fast).

Output
------
figures/interaction/
    shap_interaction_top30_filtered.png / .pdf / .svg / _raw.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from wulff_ml import config
from wulff_ml.feature_engineering import load_inputs, build_feature_matrix
from wulff_ml.cross_validation import compound_split
from wulff_ml.model_training import train_xgb
from wulff_ml.shap_analysis import compute_shap_values
from wulff_ml.feature_groups import ALE_GROUPS
from wulff_ml.utils import banner
from wulff_ml.visualization import save_figure
from wulff_ml.shap_interaction_utils import (
    cluster_order, draw_interaction_plot, compute_topk_interaction_matrix,
    shorten_feature_name, GROUP_PALETTE,
)


def filtered_top_feature_interaction_heatmap(model, X_test, shap_values, feature_cols,
                                              raw_groups=ALE_GROUPS, top_k=30, max_samples=500,
                                              random_state=config.RANDOM_STATE,
                                              figures_dir=None, zero_tol=1e-3):
    figures_dir = figures_dir or config.FIG_INTERACTION_DIR
    os.makedirs(figures_dir, exist_ok=True)

    interaction, top_feat_names, n_sampled = compute_topk_interaction_matrix(
        model, X_test, shap_values, feature_cols,
        top_k=top_k, max_samples=max_samples, random_state=random_state)

    print("Identifying zero-interaction features (row sum == 0) ...")
    row_sums      = interaction.sum(axis=1)
    nonzero_mask  = row_sums > zero_tol
    n_removed     = int((~nonzero_mask).sum())
    zero_feats    = [top_feat_names[i] for i in np.where(~nonzero_mask)[0]]
    print(f"Removed {n_removed} zero-interaction feature(s):")
    for f in zero_feats:
        print(f"  ✗  {f}")

    if n_removed == 0:
        print("No zero-interaction features — Plot B identical to Plot A; skipping.")
        return None

    nonzero_idx    = np.where(nonzero_mask)[0]
    k_nz           = int(nonzero_mask.sum())
    top_feat_nz    = [top_feat_names[i] for i in nonzero_idx]
    interaction_nz = interaction[np.ix_(nonzero_idx, nonzero_idx)]

    lnk_nz, ord_nz = cluster_order(interaction_nz)
    int_ord_nz  = interaction_nz[np.ix_(ord_nz, ord_nz)]
    lab_ord_nz  = [top_feat_nz[i] for i in ord_nz]
    lab_sh_nz   = [shorten_feature_name(l) for l in lab_ord_nz]

    feat_to_group = {f: gid for gid, _, feats in raw_groups for f in feats}
    colors_nz     = [GROUP_PALETTE.get(feat_to_group.get(lab_ord_nz[i], ""), "#cccccc")
                     for i in range(k_nz)]

    scale     = k_nz / top_k
    figsize_b = (max(14, 22 * scale), max(10, 18 * scale))

    print(f"Drawing Plot B: {k_nz}-feature filtered heatmap ...")
    fig_b = draw_interaction_plot(
        int_mat=int_ord_nz, lab_short_list=lab_sh_nz, row_color_list=colors_nz,
        lnk=lnk_nz, feat_names_list=lab_ord_nz, raw_groups=raw_groups,
        feat_to_group=feat_to_group,
        title_str=(f"SHAP Pairwise Interaction — {k_nz} Active Features "
                   f"(sampled n={n_sampled}, {n_removed} zero-interaction feature(s) removed)\n"
                   r"Surface energy $\gamma$   |   Complete-linkage clustered"
                   r"   |   Colour strip = feature group"),
        figsize=figsize_b,
    )
    out_path_b = os.path.join(figures_dir, f"shap_interaction_top{top_k}_filtered.png")
    save_figure(fig_b, out_path_b, dpi=600)
    plt.close(fig_b)
    return out_path_b


def main():
    config.ensure_dirs()
    banner("STEP 13: SHAP TOP-30 INTERACTION HEATMAP (FILTERED)")

    targets, bulk_tr, _ = load_inputs()
    _, X, y, groups_s, feature_cols, _ = build_feature_matrix(targets, bulk_tr)
    splits = compound_split(X, y, groups_s)

    model       = train_xgb(splits, verbose=False)
    shap_values = compute_shap_values(model, splits["X_test"], feature_cols)

    out_path = filtered_top_feature_interaction_heatmap(
        model, splits["X_test"], shap_values, feature_cols)

    print("\n" + "=" * 70)
    print(" STEP 13 COMPLETE")
    print("=" * 70)
    if out_path:
        print(f"  Top-30 heatmap, Plot B (filtered) : {out_path}")
    else:
        print("  Nothing to draw (no zero-interaction features in the top-30 set).")
    print("=" * 70)


if __name__ == "__main__":
    main()
