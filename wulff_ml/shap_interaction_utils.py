"""
Shared building blocks for the SHAP pairwise-interaction heatmaps.

Used by both:
    scripts/12_ale_interaction.py              -> group matrix + Plot A (full top-30)
    scripts/13_pair_wise_interaction_filter.py  -> Plot B (zero-interaction features removed)

Keeping this in one place means the two scripts can't drift apart on how the
interaction tensor is computed, how features are short-labelled, or how the
colour-strip/dendrogram/heatmap layout is drawn — and it means the expensive
`_compute_shap_interaction` cache entry is shared between them (same task
name, same call signature -> a cache hit in one script is a cache hit in
the other).
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from wulff_ml import config
from wulff_ml.cache import cached

# Same 17-colour palette used for every group-coloured plot in this script.
GROUP_PALETTE = {
    "G01": "#4E79A7", "G02": "#F28E2B", "G03": "#E15759",
    "G04": "#76B7B2", "G05": "#59A14F", "G06": "#EDC948",
    "G07": "#B07AA1", "G08": "#FF9DA7", "G09": "#9C755F",
    "G10": "#BAB0AC", "G11": "#4E79A7", "G12": "#F28E2B",
    "G13": "#E15759", "G14": "#76B7B2", "G15": "#59A14F",
    "G16": "#EDC948", "G17": "#B07AA1",
}


@cached("shap_interaction_values")
def _compute_shap_interaction(model, X_test_clean, n_features, random_state, sample_size=300):
    """The expensive O(n_features^2) SHAP interaction-value computation.
    Cached per (model, data, sample_size) — the group-level analysis in
    script 12 calls this with sample_size=300, the top-30 analyses (Plot A
    in script 12, Plot B in script 13) call it with sample_size=500; each
    distinct sample_size is cached independently, and script 13 reuses
    script 12's Plot-A cache entry instead of recomputing it."""
    import shap
    sample = X_test_clean.sample(min(sample_size, len(X_test_clean)), random_state=random_state)
    explainer = shap.TreeExplainer(model)
    print(f"Computing SHAP interaction values for {len(sample)} rows x {n_features} features ...")
    return explainer.shap_interaction_values(sample)


def sanitise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert category and object columns to integer codes for SHAP interaction values."""
    df = df.copy()
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].cat.codes
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def shorten_feature_name(name: str) -> str:
    """Shorten MagpieData feature names for compact axis labels."""
    return (name.replace("MagpieData ", "")
                .replace("mean ", "\u03bc ")
                .replace("mode ", "Mo ")
                .replace("maximum ", "Max ")
                .replace("minimum ", "Min ")
                .replace("range ", "\u0394 ")
                .replace("avg_dev ", "\u03c3 "))


def compute_topk_interaction_matrix(model, X_test, shap_values, feature_cols,
                                     top_k=30, max_samples=500,
                                     random_state=config.RANDOM_STATE):
    """Steps shared by Plot A and Plot B: pick the top-K features by mean
    |SHAP|, get the (cached) SHAP interaction tensor, and slice/symmetrise
    it down to the top_k x top_k submatrix.

    Returns (interaction, top_feat_names, n_sampled):
        interaction     - (top_k, top_k) symmetric ndarray, zero diagonal
        top_feat_names  - the top_k feature names, ranked by mean |SHAP|
        n_sampled       - number of rows the interaction tensor was sampled on
    """
    shap_arr = shap_values.values
    feat_list = list(shap_values.feature_names)

    mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_k]
    top_feat_names = [feat_list[i] for i in top_idx]

    X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feat_list)
    X_test_clean = sanitise_dtypes(X_test_df)
    n_sampled = min(max_samples, len(X_test_clean))
    print(f"Computing shap_interaction_values on {n_sampled} rows x {len(feat_list)} features ...")

    shap_int_full = _compute_shap_interaction(
        model, X_test_clean, len(feat_list), random_state, sample_size=max_samples)

    shap_int_arr = np.array(shap_int_full)                            # (n, F, F)
    shap_int_top = shap_int_arr[:, np.ix_(top_idx, top_idx)[0],
                                    np.ix_(top_idx, top_idx)[1]]       # (n, top_k, top_k)
    mean_int = np.mean(np.abs(shap_int_top), axis=0)                  # (top_k, top_k)

    interaction = mean_int.copy()
    np.fill_diagonal(interaction, 0)
    interaction = (interaction + interaction.T) / 2

    return interaction, top_feat_names, n_sampled


def cluster_order(interaction: np.ndarray):
    """Complete-linkage hierarchical clustering on an interaction matrix,
    returning (linkage_matrix, leaf_order)."""
    dist = interaction.max() - interaction
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    lnk = linkage(squareform(dist, checks=False), method="complete")
    order = dendrogram(lnk, no_plot=True)["leaves"]
    return lnk, order


def draw_interaction_plot(int_mat, lab_short_list, row_color_list, lnk,
                           feat_names_list, raw_groups, feat_to_group,
                           title_str, figsize):
    """Shared colour-strip | dendrogram | heatmap layout for Plot A and Plot B.
    Pure plotting — never cached, so a cache hit on the underlying interaction
    tensor still always produces a fresh figure."""
    import matplotlib.pyplot as plt

    K = len(lab_short_list)

    fig, axes = plt.subplots(
        1, 3, figsize=figsize,
        gridspec_kw={"width_ratios": [0.02, 0.2, 1], "wspace": 0.03},
    )
    ax_color, ax_dend, ax_main = axes

    for i, c in enumerate(row_color_list):
        ax_color.add_patch(plt.Rectangle((0, i - 0.5), 1, 1, color=c))
    ax_color.set_xlim(0, 1)
    ax_color.set_ylim(-0.5, K - 0.5)
    ax_color.set_axis_off()

    dendrogram(lnk, ax=ax_dend, orientation="left",
               color_threshold=0, above_threshold_color="#888888", no_labels=True)
    ax_dend.set_axis_off()
    ax_dend.invert_yaxis()

    pos_vals = int_mat[int_mat > 0]
    vmax = np.percentile(pos_vals, 95) if len(pos_vals) else 1e-6

    im = ax_main.imshow(int_mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax_main.set_xticks(range(K))
    ax_main.set_yticks(range(K))
    ax_main.set_xticklabels(lab_short_list, rotation=55, ha="right", fontsize=8)
    ax_main.set_yticklabels(lab_short_list, fontsize=8)
    ax_main.tick_params(axis="both", length=0)

    # annotate only cells above the 80th percentile of positive values
    if len(pos_vals):
        thresh = np.percentile(pos_vals, 80)
        for i in range(K):
            for j in range(K):
                if i != j and int_mat[i, j] >= thresh:
                    fc = "white" if int_mat[i, j] >= thresh * 1.5 else "black"
                    ax_main.text(j, i, f"{int_mat[i, j]:.1e}",
                                 ha="center", va="center", fontsize=5.5, color=fc)

    plt.colorbar(im, ax=ax_main, fraction=0.025, pad=0.02,
                 label="Mean |SHAP interaction value|")

    legend_elements = [
        Patch(facecolor=GROUP_PALETTE[gid], label=f"{gid}: {gname}")
        for gid, gname, _ in raw_groups
        if gid in {feat_to_group.get(f, "") for f in feat_names_list}
    ]
    ax_main.legend(handles=legend_elements, title="Feature group",
                   loc="lower right", fontsize=7, title_fontsize=7, framealpha=0.9)

    fig.suptitle(title_str, fontsize=13)
    return fig
