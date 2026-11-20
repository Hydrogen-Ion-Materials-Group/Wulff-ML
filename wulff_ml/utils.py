"""
Generic helpers shared across the whole pipeline:
  - NaN / Inf cleaning for feature matrices
  - joblib save / load wrappers with consistent naming
  - small printing / logging utilities
"""
import os
import joblib
import numpy as np
import pandas as pd

from wulff_ml import config


# ==============================================================================
# Cleaning
# ==============================================================================
def clean_inf_nan(X_in: pd.DataFrame, medians: pd.Series = None, label: str = ""):
    """
    Replace inf with NaN, then fill NaN with column medians (computed from
    training data and reused at inference time so there is no leakage).

    Categorical columns are left untouched. Returns (X_clean, medians).
    """
    X_out = X_in.copy()
    cat_features = X_out.select_dtypes(include=["category"]).columns.tolist()
    num_features = [c for c in X_out.columns if c not in cat_features]

    X_out[num_features] = X_out[num_features].apply(pd.to_numeric, errors="coerce")
    X_out[num_features] = X_out[num_features].replace([np.inf, -np.inf], np.nan)

    if medians is None:
        medians = X_out[num_features].median(numeric_only=True)

    X_out[num_features] = X_out[num_features].fillna(medians)

    still_nan = X_out[num_features].columns[X_out[num_features].isna().any()].tolist()
    if still_nan:
        X_out[still_nan] = X_out[still_nan].fillna(0)

    X_out[num_features] = X_out[num_features].clip(lower=-1e15, upper=1e15)
    return X_out, medians


def clean_inf_nan_generic(df: pd.DataFrame, fill_values: dict = None):
    """
    Variant used by the hkl/P-analysis code: numeric columns -> median,
    categorical/object columns -> mode. Returns (df_clean, fill_values).
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    if fill_values is None:
        fill_values = {}
        for col in df.columns:
            if col in num_cols:
                fill_values[col] = df[col].median()
            else:
                mode_vals = df[col].mode()
                fill_values[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else None

    for col in df.columns:
        if fill_values.get(col) is not None:
            df[col] = df[col].fillna(fill_values[col])

    return df, fill_values


# ==============================================================================
# Model / artifact IO
# ==============================================================================
def save_artifact(obj, filename: str, directory: str = None):
    directory = directory or config.MODEL_DIR
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    joblib.dump(obj, path)
    print(f"  Saved: {path}")
    return path


def load_artifact(filename: str, directory: str = None):
    directory = directory or config.MODEL_DIR
    path = os.path.join(directory, filename)
    return joblib.load(path)


def save_model_bundle(model, train_medians, feature_cols, cat_categories=None,
                       model_name="gamma"):
    """Persist a trained model plus the artifacts needed to reproduce its
    preprocessing at inference time (median fill values, feature order,
    categorical category sets)."""
    save_artifact(model,         f"model_{model_name}.pkl")
    save_artifact(train_medians, f"train_medians_{model_name}.pkl")
    save_artifact(feature_cols,  f"feature_cols_{model_name}.pkl")
    if cat_categories is not None:
        save_artifact(cat_categories, f"cat_categories_{model_name}.pkl")


def load_model_bundle(model_name="gamma"):
    model          = load_artifact(f"model_{model_name}.pkl")
    train_medians  = load_artifact(f"train_medians_{model_name}.pkl")
    feature_cols   = load_artifact(f"feature_cols_{model_name}.pkl")
    try:
        cat_categories = load_artifact(f"cat_categories_{model_name}.pkl")
    except FileNotFoundError:
        cat_categories = None
    return model, train_medians, feature_cols, cat_categories


# ==============================================================================
# Printing helpers
# ==============================================================================
def banner(title: str, width: int = 70, char: str = "="):
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_params(name: str, params: dict):
    print(f"{name}:")
    for k, v in params.items():
        print(f"  {k:<28}: {v}")
