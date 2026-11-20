"""
wulff_ml.plotting — thin re-export shim.

All plotting logic lives in wulff_ml/visualization.py.  This package exists so
that future per-domain plotting sub-modules (shap.py, wulff.py, ablation.py)
can be added here without breaking any existing import.

Usage (unchanged from before):
    from wulff_ml.visualization import save_figure, parity_plot, ...
"""
from wulff_ml.visualization import (          # noqa: F401
    save_figure, _savefig,
    feature_importance_plot, parity_plot,
    model_comparison_plot, cv_results_plot,
    ablation_curve_plot, group_ablation_plot,
)
