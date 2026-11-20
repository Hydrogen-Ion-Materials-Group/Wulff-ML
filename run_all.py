#!/usr/bin/env python3
"""
Run the full Wulff-ML pipeline end-to-end, in order.

Usage:
    python run_all.py                  # run everything
    python run_all.py --from 02        # resume from a given step
    python run_all.py --only 01 02     # run only specific steps
    python run_all.py --skip 01 08     # run everything except these steps
                                        # (01 and 08 download from the Materials
                                        #  Project and ask for your API key on first
                                        #  use; 08 needs predictions from 02)

Each step is a standalone script under scripts/ and can also be run directly,
e.g. `python scripts/02_train_model.py`.
"""
import argparse
import os
import runpy
import sys

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

STEPS = [
    ("01", "01_prepare_features.py", "Fetch MP bulk descriptors + Magpie features"),
    ("02", "02_train_model.py", "Train primary XGBoost gamma model"),
    ("03", "03_cross_validation.py", "Cross-validation & leakage diagnostics"),
    ("04", "04_hkl_analysis.py", "P(h,k,l) broken-bond analysis"),
    ("05", "05_element_split.py", "Elemental holdout generalisation test"),
    ("06", "06_shap_analysis.py", "SHAP interpretability analysis"),
    ("07", "07_multimodal_validation.py", "Multi-model comparison"),
    ("08", "08_wulff_prediction.py", "Wulff shape construction"),
    ("09", "09_data_ablation.py", "Data ablation"),
    ("10", "10_feature_ablation.py", "Feature / descriptor-group ablation"),
    ("11", "11_optuna_tuning.py", "Optuna hyperparameter tuning"),
    ("12", "12_ale_interaction.py", "SHAP group-interaction matrix + top-30 heatmap (Plot A)"),
    ("13", "13_pair_wise_interaction_filter.py", "SHAP top-30 heatmap, zero-interaction features removed (Plot B)")]


def main():
    parser = argparse.ArgumentParser(description="Run the Wulff-ML pipeline.")
    parser.add_argument("--from", dest="from_step", default=None,
                        help="Resume from this step number (e.g. 03)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only these step numbers")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="Skip these step numbers")
    parser.add_argument("--force-cache", action="store_true",
                        help="Ignore all cached results and recompute everything from scratch")
    parser.add_argument("--cache-status", action="store_true",
                        help="Print a summary of everything currently cached, then exit")
    args = parser.parse_args()

    if args.cache_status:
        from wulff_ml import cache
        cache.status()
        return

    if args.force_cache:
        os.environ["WULFF_CACHE_FORCE"] = "1"
        print("--force-cache set: every cached result will be recomputed.\n")

    selected = STEPS
    if args.from_step:
        idx = next(i for i, s in enumerate(STEPS) if s[0] == args.from_step)
        selected = STEPS[idx:]
    if args.only:
        selected = [s for s in STEPS if s[0] in args.only]
    if args.skip:
        selected = [s for s in selected if s[0] not in args.skip]

    print("Pipeline plan:")
    for num, fname, desc in selected:
        print(f"  [{num}] {fname:<32} {desc}")
    print()

    for num, fname, desc in selected:
        print("\n" + "#" * 78)
        print(f"# STEP {num}: {desc}")
        print("#" * 78)
        script_path = os.path.join(SCRIPTS_DIR, fname)
        try:
            runpy.run_path(script_path, run_name="__main__")
        except Exception as e:
            print(f"\n!! Step {num} ({fname}) failed: {e}")
            print("Fix the issue and resume with: python run_all.py --from "
                  f"{num}")
            sys.exit(1)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
