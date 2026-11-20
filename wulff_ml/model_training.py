"""
Model training & evaluation for the surface-energy (gamma) regression task.

Supports the primary XGBoost model plus the alternative models used in the
multi-modal validation (Linear Regression, Random Forest, LightGBM, CatBoost,
Extra Trees), all through a common evaluate_model() interface.

Units
-----
Model targets (y) are stored internally in eV/Å².  evaluate_model() prints
and returns metrics in J/m² (the SI convention used in all figures/tables).
R² and rank correlations are dimensionless and are unaffected by the unit.
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr, kendalltau

from wulff_ml import config
from wulff_ml.units import to_jm2, UNIT_SHORT
from wulff_ml.cache import cached


# ==============================================================================
# Evaluation
# ==============================================================================
def evaluate_model(model, splits: dict, label: str = "") -> pd.DataFrame:
    """Print and return Train/Val/Test MAE (J/m²), R², Spearman, Kendall for a
    fitted model.  `splits` must contain X_train/X_val/X_test and
    y_train/y_val/y_test (keys produced by cross_validation.compound_split).

    MAE is converted from native eV/Å² to J/m² for display and storage.
    """
    print(f"\nPerformance Evaluation Summary [{label}]  (MAE in {UNIT_SHORT}):")
    rows = []
    for name in ["train", "val", "test"]:
        X_s, y_s = splits[f"X_{name}"], splits[f"y_{name}"]
        pred     = model.predict(X_s)
        mae_ev   = mean_absolute_error(y_s, pred)
        mae_jm2  = to_jm2(mae_ev)
        r2       = r2_score(y_s, pred)
        spearman = spearmanr(y_s, pred).correlation
        kendall  = kendalltau(y_s, pred).correlation
        rows.append({"set": name,
                     "mae_jm2": mae_jm2, "mae_ev_ang2": mae_ev,
                     "r2": r2, "spearman": spearman, "kendall": kendall})
        print(f"    {name.capitalize():5s} Set -> "
              f"MAE = {mae_jm2:.4f} {UNIT_SHORT}  "
              f"({mae_ev:.5f} eV/Å²)  |  "
              f"R² = {r2:.4f}  |  Spearman = {spearman:.4f}  |  Kendall = {kendall:.4f}")
    return pd.DataFrame(rows)


# ==============================================================================
# XGBoost (primary model)
# ==============================================================================
@cached("train_xgb", ignore_kwargs=["verbose"])
def train_xgb(splits: dict, params: dict = None, verbose=150):
    import xgboost as xgb

    params = params or config.PARAMS_XGB
    model  = xgb.XGBRegressor(**params)
    model.fit(splits["X_train"], splits["y_train"],
              eval_set=[(splits["X_val"], splits["y_val"])], verbose=verbose)
    return model


@cached("predict_all_facets")
def _predict_all_facets_core(model, bulk_pr, targets, feature_cols, cat_categories,
                              train_medians):
    """Pure computation: cross-product (compound x hkl) -> predicted gamma.
    No file I/O here, so this is safe to cache — see predict_all_facets()
    below, which always (re-)writes out_path even on a cache hit."""
    from wulff_ml.feature_engineering import build_prediction_matrix

    all_hkl = targets["hkl"].unique().tolist()
    print(f"Unique hkl in training set : {len(all_hkl)}")

    df_pred, X_pred = build_prediction_matrix(
        bulk_pr, all_hkl, feature_cols, cat_categories, train_medians)
    df_pred["pred_gamma_eV_per_ang2"] = model.predict(X_pred)
    # Add J/m² column alongside the native eV/Å² column for convenience
    df_pred["pred_gamma_J_per_m2"] = to_jm2(df_pred["pred_gamma_eV_per_ang2"])
    return df_pred


def predict_all_facets(model, bulk_pr, targets, feature_cols, cat_categories,
                        train_medians, out_path=None):
    """Predict gamma for every (compound x hkl) combination in the prediction
    pool.  Returns the prediction DataFrame (columns include both eV/Å² and
    J/m²).  The CSV write always runs so out_path exists even on a cache hit.
    """
    df_pred = _predict_all_facets_core(model, bulk_pr, targets, feature_cols,
                                        cat_categories, train_medians)
    if out_path:
        df_pred[["mp_id", "formula_pretty", "hkl",
                 "pred_gamma_eV_per_ang2", "pred_gamma_J_per_m2"]].to_csv(
            out_path, index=False)
        print(f"Saved: {out_path}")
    return df_pred


# ==============================================================================
# Alternative models (multi-modal validation)
# ==============================================================================
@cached("train_linear_regression")
def train_linear_regression(splits_numeric: dict):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression(n_jobs=-1)
    model.fit(splits_numeric["X_train"], splits_numeric["y_train"])
    return model


@cached("train_random_forest")
def train_random_forest(splits_numeric: dict, params: dict = None):
    from sklearn.ensemble import RandomForestRegressor

    params = params or config.PARAMS_RF
    model  = RandomForestRegressor(**params)
    model.fit(splits_numeric["X_train"], splits_numeric["y_train"])
    return model


@cached("train_extra_trees")
def train_extra_trees(splits_numeric: dict, params: dict = None):
    from sklearn.ensemble import ExtraTreesRegressor

    params = params or config.PARAMS_ET
    model  = ExtraTreesRegressor(**params)
    model.fit(splits_numeric["X_train"], splits_numeric["y_train"])
    return model


@cached("train_lightgbm")
def train_lightgbm(splits: dict, cat_cols=None, params: dict = None):
    import lightgbm as lgb

    cat_cols = cat_cols or config.CAT_COLS
    params   = params   or config.PARAMS_LGBM

    X_train, X_val, X_test = (splits["X_train"].copy(), splits["X_val"].copy(),
                               splits["X_test"].copy())
    for col in cat_cols:
        for X in (X_train, X_val, X_test):
            X[col] = X[col].astype(str).astype("category")

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, splits["y_train"],
        eval_set=[(X_val, splits["y_val"])],
        callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False),
                   lgb.log_evaluation(period=150)],
    )
    return model, {"X_train": X_train, "X_val": X_val, "X_test": X_test}


@cached("train_catboost")
def train_catboost(splits: dict, cat_cols=None, params: dict = None):
    from catboost import CatBoostRegressor

    cat_cols = cat_cols or config.CAT_COLS
    # PARAMS_CATBOOST already includes train_dir=config.CATBOOST_TRAIN_DIR
    params   = params   or config.PARAMS_CATBOOST

    X_train, X_val, X_test = (splits["X_train"].copy(), splits["X_val"].copy(),
                               splits["X_test"].copy())
    for X in (X_train, X_val, X_test):
        for col in cat_cols:
            X[col] = X[col].astype(str)

    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    model   = CatBoostRegressor(**params)
    model.fit(X_train, splits["y_train"], cat_features=cat_idx,
              eval_set=(X_val, splits["y_val"]))
    return model, {"X_train": X_train, "X_val": X_val, "X_test": X_test}


def compare_models(model_registry: list, y_test) -> pd.DataFrame:
    """
    model_registry: list of (name, fitted_model, X_test) tuples.
    Returns a DataFrame ranked by Test MAE (ascending = best first).
    MAE stored in both eV/Å² (mae_ev_ang2) and J/m² (MAE / mae_jm2).
    """
    from wulff_ml.units import to_jm2

    results = []
    print("=" * 90)
    print(f" MODEL COMPARISON SUMMARY — Test Set  (MAE in {UNIT_SHORT})")
    print("=" * 90)
    for name, model, X_te in model_registry:
        pred     = model.predict(X_te)
        mae_ev   = mean_absolute_error(y_test, pred)
        mae_jm2  = to_jm2(mae_ev)
        r2       = r2_score(y_test, pred)
        spearman = spearmanr(y_test, pred).correlation
        kendall  = kendalltau(y_test, pred).correlation
        results.append({"Model": name,
                         "MAE": mae_jm2,           # primary column, J/m²
                         "mae_ev_ang2": mae_ev,
                         "R2": r2,
                         "Spearman": spearman, "Kendall": kendall})
        print(f"  {name:<20} {mae_jm2:>10.4f} {UNIT_SHORT}  "
              f"({mae_ev:.5f} eV/Å²)  R²={r2:>7.4f}  "
              f"Spearman={spearman:>8.4f}  Kendall={kendall:>8.4f}")
    return pd.DataFrame(results).sort_values("MAE")
