"""
Feature engineering.

Three responsibilities, mirroring notebook 01 and the start of every
training notebook (02-07):

  1. fetch_bulk_descriptors_from_mp() / docs_to_dataframe()
       Pull bulk material descriptors from the Materials Project API.
  2. add_magpie_features()
       Featurize a bulk-descriptor CSV with ~130 Magpie composition
       statistics via matminer.
  3. parse_hkl() / hkl_features() / build_feature_matrix()
       Turn (mp_id, hkl) target rows + bulk descriptors into the final
       X / y / groups feature matrix used by every model.
"""
import ast
import subprocess
import sys

import numpy as np
import pandas as pd

from wulff_ml import config
from wulff_ml.utils import clean_inf_nan
from wulff_ml.cache import cached
from wulff_ml.api_key import get_mp_api_key


# ==============================================================================
# 1. Materials Project bulk descriptor fetch
# ==============================================================================
def safe_vrh(val):
    """Extract the Voigt-Reuss-Hill average from an elastic-property object,
    or pass through a plain numeric value."""
    if val is None:
        return None
    if hasattr(val, "vrh"):
        return val.vrh
    if isinstance(val, (int, float)):
        return val
    return None


MP_SUMMARY_FIELDS = [
    "material_id", "formula_pretty", "nsites", "nelements", "composition",
    "volume", "density", "density_atomic", "symmetry",
    "formation_energy_per_atom", "energy_per_atom", "energy_above_hull",
    "is_stable", "band_gap", "is_metal",
    "bulk_modulus", "shear_modulus", "universal_anisotropy", "homogeneous_poisson",
    "total_magnetization",
    "weighted_surface_energy", "weighted_surface_energy_EV_PER_ANG2",
]


def docs_to_dataframe(docs) -> pd.DataFrame:
    """Flatten a list of MPRester SummaryDoc objects into a feature DataFrame."""
    rows = []
    for doc in docs:
        sym = doc.symmetry
        comp = doc.composition
        elements = list(comp.elements)
        total_amt = sum(comp[el] for el in elements)

        def wmean(attr):
            vals = [getattr(el, attr, None) for el in elements]
            if any(v is None for v in vals):
                return None
            return sum(comp[el] * v for el, v in zip(elements, vals)) / total_amt

        def mean_d_electrons():
            d_counts = [sum(n for _, l, n in el.full_electronic_structure if l == 2)
                        for el in elements]
            return sum(comp[el] * d for el, d in zip(elements, d_counts)) / total_amt

        row = {
            "mp_id": str(doc.material_id),
            "formula_pretty": doc.formula_pretty,
            "nsites": doc.nsites,
            "nelements": doc.nelements,
            "volume": doc.volume,
            "volume_per_atom": doc.volume / doc.nsites if doc.nsites else None,
            "density": doc.density,
            "density_atomic": doc.density_atomic,
            "space_group_number": sym.number if sym else None,
            "crystal_system": str(sym.crystal_system) if sym else None,
            "point_group": sym.point_group if sym else None,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "energy_per_atom": doc.energy_per_atom,
            "energy_above_hull": doc.energy_above_hull,
            "is_stable": int(doc.is_stable) if doc.is_stable is not None else None,
            "band_gap": doc.band_gap,
            "is_metal": int(doc.is_metal) if doc.is_metal is not None else None,
            "bulk_modulus_vrh": safe_vrh(doc.bulk_modulus),
            "shear_modulus_vrh": safe_vrh(doc.shear_modulus),
            "universal_anisotropy": doc.universal_anisotropy,
            "homogeneous_poisson": doc.homogeneous_poisson,
            "total_magnetization": doc.total_magnetization,
            "weighted_surface_energy": doc.weighted_surface_energy,
            "weighted_surface_energy_EV_PER_ANG2": doc.weighted_surface_energy_EV_PER_ANG2,
            "mean_electronegativity": wmean("X"),
            "mean_atomic_radius": wmean("atomic_radius"),
            "mean_melting_point": wmean("melting_point"),
            "max_electronegativity_diff": max(el.X for el in elements) - min(el.X for el in elements),
            "mean_d_electrons": mean_d_electrons(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


@cached("fetch_bulk_descriptors_from_mp_core", ignore_kwargs=["api_key"])
def _fetch_bulk_descriptors_core(csv_path: str, id_col: str, api_key: str = None) -> pd.DataFrame:
    """Pure computation: pulls descriptors from the Materials Project API.
    No file I/O — see fetch_bulk_descriptors_from_mp() below, which always
    (re-)writes out_path even on a cache hit, so you never need to re-hit
    the network just to regenerate a deleted/renamed output file."""
    from mp_api.client import MPRester

    api_key = api_key or get_mp_api_key()
    df_in = pd.read_csv(csv_path)
    mpids = list(df_in[id_col].astype(str).unique())
    print(f"\nFetching {len(mpids)} mp_ids from {csv_path} ...")

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(material_ids=mpids, fields=MP_SUMMARY_FIELDS)

    print(f"  -> Received {len(docs)} documents from MP")
    df_out = docs_to_dataframe(docs)

    returned = set(df_out["mp_id"].tolist())
    missing = [m for m in mpids if m not in returned]
    if missing:
        print(f"  WARNING: {len(missing)} mp_ids not returned: {missing[:10]} ...")

    return df_out


def fetch_bulk_descriptors_from_mp(csv_path: str, id_col: str, out_path: str,
                                    api_key: str = None) -> pd.DataFrame:
    """Read mp_ids from csv_path, pull descriptors from Materials Project,
    save to out_path and return the resulting DataFrame. The network call
    is cached; re-running with the same mp_id list never re-hits the API."""
    api_key = api_key or get_mp_api_key()
    df_out = _fetch_bulk_descriptors_core(csv_path, id_col, api_key=api_key)
    df_out.to_csv(out_path, index=False)
    print(f"  -> Saved {len(df_out)} rows to {out_path}")
    return df_out


@cached("fetch_bulk_descriptors_for_ids", ignore_kwargs=["api_key"])
def _fetch_descriptors_for_ids_core(mpids: tuple, api_key: str = None) -> pd.DataFrame:
    from mp_api.client import MPRester

    api_key = api_key or get_mp_api_key()
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(material_ids=list(mpids),
                                            fields=MP_SUMMARY_FIELDS)
    return docs_to_dataframe(docs)


def fetch_bulk_descriptors_for_ids(mpids, api_key: str = None) -> pd.DataFrame:
    """Download the bulk descriptors of the given Materials Project IDs
    (e.g. ["mp-1077900", "mp-21119"]). IDs unknown to the Materials Project
    are simply absent from the returned DataFrame. Results are cached."""
    return _fetch_descriptors_for_ids_core(tuple(mpids), api_key=api_key)


# ==============================================================================
# 2. Magpie composition features (matminer)
# ==============================================================================
@cached("add_magpie_features_core")
def _add_magpie_features_core(csv_in: str, label: str = "") -> pd.DataFrame:
    """Pure computation: Magpie featurization. No file I/O — see
    add_magpie_features() below, which always (re-)writes csv_out even on
    a cache hit."""
    try:
        import matminer  # noqa: F401
    except ImportError:
        print("Installing matminer...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matminer", "-q"])

    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty

    ep = ElementProperty.from_preset("magpie")
    print(f"Magpie features to add: {len(ep.feature_labels())}")

    df = pd.read_csv(csv_in)
    print(f"\n{label}: loaded {df.shape[0]} rows, {df.shape[1]} columns")

    df["composition"] = df["formula_pretty"].apply(Composition)
    df = ep.featurize_dataframe(df, col_id="composition", ignore_errors=True)
    df = df.drop(columns=["composition"])
    return df


def add_magpie_features(csv_in: str, csv_out: str, label: str = "") -> pd.DataFrame:
    """Featurize a bulk-descriptor CSV with Magpie elemental-property
    statistics (mean/std/min/max/range of atomic number, electronegativity,
    covalent radius, valence electrons, etc.). The (slow) featurization
    itself is cached; the CSV write always runs."""
    df = _add_magpie_features_core(csv_in, label)
    df.to_csv(csv_out, index=False)
    print(f"{label}: saved {df.shape[0]} rows, {df.shape[1]} columns -> {csv_out}")
    return df


def magpie_features_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Same Magpie featurization as add_magpie_features(), but for an
    in-memory DataFrame that has a ``formula_pretty`` column."""
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty

    ep = ElementProperty.from_preset("magpie")
    out = df.copy()
    out["composition"] = out["formula_pretty"].apply(Composition)
    out = ep.featurize_dataframe(out, col_id="composition", ignore_errors=True, pbar=False)
    return out.drop(columns=["composition"])


# ==============================================================================
# 3. hkl (Miller index) feature extraction
# ==============================================================================
def parse_hkl(hkl_str):
    """Parse a Miller-index string like '(1, 1, 0)' or '[1,1,0]' into a list."""
    try:
        return ast.literal_eval(hkl_str)
    except Exception:
        clean = hkl_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        return [int(x) for x in clean.split(",")]


def hkl_features(hkl_str) -> dict:
    """Symmetry-aware physical features derived from a single (h, k, l)."""
    h, k, l = parse_hkl(hkl_str)
    h_abs, k_abs, l_abs = abs(h), abs(k), abs(l)
    sorted_hkl = sorted([h_abs, k_abs, l_abs])
    norm = h ** 2 + k ** 2 + l ** 2

    magnitude = float(norm) ** 0.5
    safe_mag = magnitude if magnitude > 0 else 1.0
    h_norm, k_norm, l_norm = h / safe_mag, k / safe_mag, l / safe_mag

    return {
        "h": h, "k": k, "l": l,
        "h_abs": h_abs, "k_abs": k_abs, "l_abs": l_abs,
        "hkl_sort_0": sorted_hkl[0], "hkl_sort_1": sorted_hkl[1], "hkl_sort_2": sorted_hkl[2],
        "hkl_norm": norm,
        "n_zeros": [h, k, l].count(0),
        "max_index": max(h_abs, k_abs, l_abs),
        "is_low_index": int(norm <= 3),
        "hkl_magnitude": magnitude,
        "h_norm": h_norm, "k_norm": k_norm, "l_norm": l_norm,
    }


def expand_hkl_column(df: pd.DataFrame, hkl_col: str = "hkl") -> pd.DataFrame:
    """Append the hkl_features() columns to a DataFrame containing an hkl column."""
    hkl_df = pd.DataFrame(df[hkl_col].apply(hkl_features).tolist())
    return pd.concat([df.reset_index(drop=True), hkl_df], axis=1)


# ==============================================================================
# Full feature-matrix builder (targets + bulk descriptors -> X, y, groups)
# ==============================================================================
@cached("load_inputs")
def load_inputs(target_csv: str = None, bulk_train_csv: str = None,
                 bulk_predict_csv: str = None):
    target_csv = target_csv or config.SURFACE_ENERGY_TARGETS_CSV
    bulk_train_csv = bulk_train_csv or config.BULK_TRAIN_MAGPIE
    bulk_predict_csv = bulk_predict_csv or config.BULK_PREDICT_MAGPIE

    targets = pd.read_csv(target_csv)
    bulk_tr = pd.read_csv(bulk_train_csv)
    bulk_pr = pd.read_csv(bulk_predict_csv)

    print(f"Targets loaded      : {targets.shape}")
    print(f"Bulk train loaded   : {bulk_tr.shape}")
    print(f"Bulk predict loaded : {bulk_pr.shape}")
    return targets, bulk_tr, bulk_pr


@cached("build_feature_matrix")
def build_feature_matrix(targets: pd.DataFrame, bulk_tr: pd.DataFrame,
                          target_col: str = config.TARGET_GAMMA,
                          cat_cols=None, drop_cols=None):
    """
    Reproduce the standard preprocessing used in every training notebook:
      1. expand hkl -> physical features
      2. merge with bulk descriptors on mp_id
      3. set categorical dtypes
      4. split into X / y / groups
      5. drop rows with a missing target (can't train/score on a NaN label)
      6. clean inf/nan (median fill)

    Returns: df (merged), X, y, groups, FEATURE_COLS, train_medians
    """
    cat_cols = cat_cols or config.CAT_COLS
    drop_cols = drop_cols or config.DROP_COLS

    print("Extracting hkl features...")
    targets = expand_hkl_column(targets, "hkl")
    print(f"Targets after hkl expansion: {targets.shape}")

    df = targets.merge(bulk_tr, on="mp_id", how="left")
    print(f"Shape after structural property merge: {df.shape}")

    for col in cat_cols:
        df[col] = df[col].astype(str).astype("category")

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    groups = df["mp_id"].copy()

    # A row with a NaN target can't be trained or scored on, and once it's
    # mixed into any split/CV fold downstream it surfaces as a generic
    # "Input contains NaN" failure far from its actual cause. Drop it here,
    # once, for every script that calls build_feature_matrix.
    valid = y.notna()
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"  WARNING: dropping {n_dropped} rows with missing "
              f"'{target_col}' (NaN target)")
        df = df.loc[valid].reset_index(drop=True)
        X = X.loc[valid].reset_index(drop=True)
        y = y.loc[valid].reset_index(drop=True)
        groups = groups.loc[valid].reset_index(drop=True)

    X, train_medians = clean_inf_nan(X, label="Training Features")
    print(f"Features used ({len(feature_cols)})")

    return df, X, y, groups, feature_cols, train_medians


def build_prediction_matrix(bulk_pr: pd.DataFrame, all_hkl: list, feature_cols: list,
                             cat_categories: dict, train_medians, cat_cols=None):
    """Build the cross-product (compound x hkl) feature matrix used to predict
    gamma for every facet of every compound in the prediction pool."""
    cat_cols = cat_cols or config.CAT_COLS

    df_hkl = pd.DataFrame({"hkl": all_hkl})
    df_pred = bulk_pr.merge(df_hkl, how="cross")
    df_pred = expand_hkl_column(df_pred, "hkl")

    for col in cat_cols:
        df_pred[col] = pd.Categorical(df_pred[col].astype(str), categories=cat_categories[col])

    X_pred = df_pred[feature_cols].copy()
    X_pred, _ = clean_inf_nan(X_pred, medians=train_medians, label="Prediction features")
    return df_pred, X_pred
