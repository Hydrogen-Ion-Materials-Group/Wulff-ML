"""
Predict facet surface energies (and Wulff shapes) for a few compounds,
given only their Materials Project IDs.

    from wulff_ml.prediction import predict_compounds
    gamma_df, wulff_df, out_dir = predict_compounds(["mp-1077900", "mp-21119"])

Or from the terminal:  wulff-predict mp-1077900 mp-21119

What happens:
  1. the trained model is loaded from models/ (it is trained once on the
     bundled DFT data if it does not exist yet)
  2. bulk descriptors of the requested compounds are downloaded from the
     Materials Project and Magpie features are added
  3. gamma(hkl) is predicted for every Miller index of the training set
  4. the Wulff shape and facet area fractions are built (optional)
"""
import os
import re
import time
from typing import List, Tuple

import pandas as pd

from wulff_ml import config
from wulff_ml.api_key import get_mp_api_key
from wulff_ml.units import to_jm2, UNIT_SHORT

_MPID_RE = re.compile(r"^(mp|mvc)-\d+$")


# ==============================================================================
# Input handling
# ==============================================================================
def parse_mpids(raw) -> Tuple[List[str], List[str]]:
    """Turn free-form input into (valid_ids, invalid_tokens).

    Accepts a string or a list of strings; separators may be spaces, commas,
    semicolons or new lines. ``mp-149``, ``MP-149`` and plain ``149`` all
    become ``mp-149``. Duplicates are removed, order is kept.
    """
    if isinstance(raw, str):
        raw = [raw]
    tokens = []
    for item in raw:
        tokens += [t for t in re.split(r"[\s,;]+", str(item).strip("\"' ")) if t]

    valid, invalid = [], []
    for tok in tokens:
        tok = tok.strip("\"'").lower()
        if tok.isdigit():
            tok = f"mp-{tok}"
        if _MPID_RE.match(tok):
            if tok not in valid:
                valid.append(tok)
        else:
            invalid.append(tok)
    return valid, invalid


# ==============================================================================
# Model
# ==============================================================================
def train_main_model():
    """Train the primary gamma model on the bundled DFT data and save it to
    models/ (same procedure as scripts/02_train_model.py, without the plots)."""
    from wulff_ml.feature_engineering import build_feature_matrix
    from wulff_ml.cross_validation import compound_split
    from wulff_ml.model_training import train_xgb, evaluate_model
    from wulff_ml.utils import save_model_bundle

    targets = pd.read_csv(config.SURFACE_ENERGY_TARGETS_CSV)
    bulk_tr = pd.read_csv(config.BULK_TRAIN_MAGPIE)
    df, X, y, groups, feature_cols, train_medians = build_feature_matrix(targets, bulk_tr)

    splits = compound_split(X, y, groups)
    model = train_xgb(splits)
    evaluate_model(model, splits, label="gamma")

    cat_categories = {col: X[col].cat.categories for col in config.CAT_COLS}
    save_model_bundle(model, train_medians, feature_cols, cat_categories, model_name="gamma")
    return model, train_medians, feature_cols, cat_categories


def ensure_model():
    """Load the saved gamma model, training it first if necessary."""
    from wulff_ml.utils import load_model_bundle

    try:
        model, train_medians, feature_cols, cat_categories = load_model_bundle("gamma")
        if cat_categories is None:
            raise FileNotFoundError("cat_categories_gamma.pkl")
        return model, train_medians, feature_cols, cat_categories
    except FileNotFoundError:
        print("No trained model found in models/. Training it now on the bundled "
              "DFT data (one-time step) ...")
    except Exception as e:  # e.g. pickle written by another xgboost version
        print(f"The saved model could not be loaded ({e}). Re-training it now ...")
    return train_main_model()


# ==============================================================================
# Prediction
# ==============================================================================
def predict_gamma_for_ids(mpids, api_key=None, model_bundle=None) -> pd.DataFrame:
    """Predict gamma for every facet of the given compounds.

    Returns a DataFrame with columns mp_id, formula_pretty, hkl,
    pred_gamma_eV_per_ang2, pred_gamma_J_per_m2 (one row per compound x facet).
    IDs that the Materials Project does not know are dropped with a message.
    """
    from wulff_ml.feature_engineering import (
        fetch_bulk_descriptors_for_ids, magpie_features_from_dataframe,
        build_prediction_matrix, hkl_features,
    )

    model, train_medians, feature_cols, cat_categories = model_bundle or ensure_model()

    print(f"\nDownloading bulk descriptors for {len(mpids)} compound(s) from the Materials Project ...")
    bulk = fetch_bulk_descriptors_for_ids(mpids, api_key=api_key)
    found = set(bulk["mp_id"]) if len(bulk) else set()
    unknown = [m for m in mpids if m not in found]
    if unknown:
        print(f"  Not found in the Materials Project (skipped): {', '.join(unknown)}")
    if not found:
        raise ValueError("None of the requested IDs was found in the Materials Project.")

    bulk = magpie_features_from_dataframe(bulk)

    # Any descriptor the model expects but the download lacks becomes NaN and is
    # median-filled like during training.
    hkl_cols = set(hkl_features("(1, 0, 0)"))   # names of the Miller-index features
    missing = [c for c in feature_cols if c not in bulk.columns and c not in hkl_cols]
    if missing:
        print(f"  Note: {len(missing)} descriptor(s) unavailable, filled with training medians: "
              f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
        for c in missing:
            bulk[c] = float("nan")

    all_hkl = pd.read_csv(config.SURFACE_ENERGY_TARGETS_CSV, usecols=["hkl"])["hkl"].unique().tolist()
    df_pred, X_pred = build_prediction_matrix(
        bulk, all_hkl, feature_cols, cat_categories, train_medians)

    df_pred["pred_gamma_eV_per_ang2"] = model.predict(X_pred)
    df_pred["pred_gamma_J_per_m2"] = to_jm2(df_pred["pred_gamma_eV_per_ang2"])
    return df_pred[["mp_id", "formula_pretty", "hkl",
                    "pred_gamma_eV_per_ang2", "pred_gamma_J_per_m2"]].reset_index(drop=True)


def predict_compounds(mpids, make_wulff: bool = True, out_dir: str = None,
                      api_key: str = None, verbose: bool = True):
    """Full workflow for a handful of compounds.

    Returns (gamma_df, wulff_df, out_dir). ``wulff_df`` holds the facet area
    fractions (empty if ``make_wulff`` is False or no shape could be built).
    """
    from wulff_ml.wulff_construction import build_wulff_shapes

    config.ensure_dirs()
    api_key = api_key or get_mp_api_key()

    gamma_df = predict_gamma_for_ids(mpids, api_key=api_key)

    out_dir = out_dir or os.path.join(config.RESULTS_DIR, "predict",
                                      time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    gamma_csv = os.path.join(out_dir, "gamma_predictions.csv")
    gamma_df.to_csv(gamma_csv, index=False)

    wulff_df = pd.DataFrame()
    if make_wulff:
        print("\nBuilding Wulff shapes ...")
        wulff_df = build_wulff_shapes(gamma_csv, api_key=api_key,
                                      output_dir=os.path.join(out_dir, "wulff"))

    if verbose:
        print_summary(gamma_df, wulff_df)
        print(f"\nAll results were saved to:\n  {os.path.abspath(out_dir)}")
    return gamma_df, wulff_df, out_dir


# ==============================================================================
# Terminal summary
# ==============================================================================
def print_summary(gamma_df: pd.DataFrame, wulff_df: pd.DataFrame = None, n: int = 5):
    for mpid, grp in gamma_df.groupby("mp_id", sort=False):
        formula = grp["formula_pretty"].iloc[0]
        print("\n" + "=" * 70)
        print(f" {formula}  ({mpid})   |   {len(grp)} facets predicted")
        print("=" * 70)

        low = grp.nsmallest(n, "pred_gamma_J_per_m2")[["hkl", "pred_gamma_J_per_m2"]]
        print(f" Lowest-energy facets ({UNIT_SHORT}):")
        for _, r in low.iterrows():
            print(f"   {r['hkl']:<14} {r['pred_gamma_J_per_m2']:.3f}")

        if wulff_df is not None and len(wulff_df):
            w = wulff_df[wulff_df["mp_id"] == mpid]
            w = w[w["area_fraction"] > 0].nlargest(n, "area_fraction")
            if len(w):
                print(" Largest facets of the Wulff shape (area fraction):")
                for _, r in w.iterrows():
                    print(f"   {r['hkl']:<14} {100 * r['area_fraction']:.1f} %")
