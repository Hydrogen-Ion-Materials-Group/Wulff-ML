"""
Train/val/test splitting strategies.

  - compound_split()          strict compound-wise (GroupShuffleSplit by mp_id)
  - elemental_split()         hold out entire elements from training
  - composition_split()       hold out entire chemical formulas (stricter than mp_id)
  - random_split_leakage_demo() naive random split, for comparison against compound_split
  - group_kfold_cv()          5-fold grouped CV to check evaluation stability
"""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split

from wulff_ml import config
from wulff_ml.utils import clean_inf_nan
from wulff_ml.cache import cached


# ==============================================================================
# Standard compound-wise split (used by every training script)
# ==============================================================================
@cached("compound_split")
def compound_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                    test_fraction: float = config.TEST_FRACTION,
                    val_fraction: float = config.VAL_FRACTION,
                    random_state: int = config.RANDOM_STATE) -> dict:
    """Strict compound-wise (mp_id) train/val/test split with zero leakage."""
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    train_val_idx, test_idx = next(gss_outer.split(X, y, groups))

    X_trainval, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
    y_trainval, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
    groups_trainval = groups.iloc[train_val_idx]
    groups_test = groups.iloc[test_idx]

    gss_inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    train_idx, val_idx = next(gss_inner.split(X_trainval, y_trainval, groups_trainval))

    X_train, med = clean_inf_nan(X_trainval.iloc[train_idx].copy())
    X_val, _ = clean_inf_nan(X_trainval.iloc[val_idx].copy(), med)
    X_test_clean, _ = clean_inf_nan(X_test.copy(), med)

    result = dict(
        X_train=X_train, X_val=X_val, X_test=X_test_clean,
        y_train=y_trainval.iloc[train_idx], y_val=y_trainval.iloc[val_idx], y_test=y_test,
        groups_train=groups_trainval.iloc[train_idx], groups_val=groups_trainval.iloc[val_idx],
        groups_test=groups_test,
        train_val_idx=train_val_idx, test_idx=test_idx,
        train_medians=med,
    )
    n_train = result["groups_train"].nunique()
    n_val = result["groups_val"].nunique()
    n_test = result["groups_test"].nunique()
    print("Split (compound-wise):")
    print(f"  Train : {len(result['X_train']):>6} rows | {n_train} compounds")
    print(f"  Val   : {len(result['X_val']):>6} rows | {n_val} compounds")
    print(f"  Test  : {len(result['X_test']):>6} rows | {n_test} compounds")
    return result


# ==============================================================================
# Elemental split — hold out entire elements
# ==============================================================================
def extract_elements(formula: str) -> set:
    """Parse a chemical formula string into the set of element symbols."""
    return set(re.findall(r"[A-Z][a-z]{0,2}", formula))


@cached("elemental_split")
def elemental_split(df_full: pd.DataFrame, X_full: pd.DataFrame, y_full: pd.Series,
                     groups_full: pd.Series, test_elements: list,
                     formula_col: str = "formula_pretty",
                     val_fraction: float = config.VAL_FRACTION,
                     random_state: int = config.RANDOM_STATE) -> dict:
    """
    Split so that ALL compounds containing any element in `test_elements`
    are held out as the test set — the strictest extrapolation test.
    """
    test_set = set(test_elements)

    row_has_test_elem = df_full[formula_col].apply(
        lambda f: bool(extract_elements(str(f)) & test_set)
    )
    test_mp_ids = set(df_full.loc[row_has_test_elem, "mp_id"].unique())
    train_mp_ids = set(df_full["mp_id"].unique()) - test_mp_ids

    test_mask = groups_full.isin(test_mp_ids)
    train_mask = groups_full.isin(train_mp_ids)

    X_test_el, y_test_el, grp_test = X_full[test_mask], y_full[test_mask], groups_full[test_mask]
    X_trainval, y_trainval, grp_tv = X_full[train_mask], y_full[train_mask], groups_full[train_mask]

    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    tr_idx, vl_idx = next(gss.split(X_trainval, y_trainval, grp_tv))

    X_train_el, X_val_el = X_trainval.iloc[tr_idx], X_trainval.iloc[vl_idx]
    y_train_el, y_val_el = y_trainval.iloc[tr_idx], y_trainval.iloc[vl_idx]
    grp_train, grp_val = grp_tv.iloc[tr_idx], grp_tv.iloc[vl_idx]

    summary = "\n".join([
        f"Elemental Split - test elements: {sorted(test_set)}",
        f"  Test  : {len(X_test_el):>7} rows | {grp_test.nunique():>5} compounds "
        f"(contains {sorted(test_set)})",
        f"  Train : {len(X_train_el):>7} rows | {grp_train.nunique():>5} compounds",
        f"  Val   : {len(X_val_el):>7} rows | {grp_val.nunique():>5} compounds",
        f"  Total : {len(X_full):>7} rows | {len(test_mp_ids) + len(train_mp_ids):>5} compounds",
        f"  Test element coverage: "
        f"{len(test_mp_ids) / (len(test_mp_ids) + len(train_mp_ids)) * 100:.1f}% of compounds",
    ])

    return dict(
        X_train=X_train_el, X_val=X_val_el, X_test=X_test_el,
        y_train=y_train_el, y_val=y_val_el, y_test=y_test_el,
        groups_train=grp_train, groups_val=grp_val, groups_test=grp_test,
        test_mp_ids=test_mp_ids, train_mp_ids=train_mp_ids, summary=summary,
    )


# ==============================================================================
# Composition split — hold out entire chemical formulas
# ==============================================================================
@cached("composition_split")
def composition_split(X: pd.DataFrame, y: pd.Series, comp_groups: pd.Series,
                       test_fraction: float = config.TEST_FRACTION,
                       val_fraction: float = config.VAL_FRACTION,
                       random_state: int = config.RANDOM_STATE) -> dict:
    """Group split by chemical formula instead of mp_id — a stricter test
    since polymorphs / different terminations share a formula."""
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    tv_idx, te_idx = next(gss_outer.split(X, y, comp_groups))

    train_comps = set(comp_groups.iloc[tv_idx])
    test_comps = set(comp_groups.iloc[te_idx])
    overlap = len(train_comps & test_comps)

    X_tv, y_tv, grp_tv = X.iloc[tv_idx], y.iloc[tv_idx], comp_groups.iloc[tv_idx]
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    tr_idx, vl_idx = next(gss_inner.split(X_tv, y_tv, grp_tv))

    X_train, med = clean_inf_nan(X_tv.iloc[tr_idx].copy())
    X_val, _ = clean_inf_nan(X_tv.iloc[vl_idx].copy(), med)
    X_test, _ = clean_inf_nan(X.iloc[te_idx].copy(), med)

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_tv.iloc[tr_idx], y_val=y_tv.iloc[vl_idx], y_test=y.iloc[te_idx],
        train_comps=train_comps, test_comps=test_comps, overlap=overlap,
        train_medians=med,
    )


# ==============================================================================
# Leakage demo — random split vs. compound-wise split
# ==============================================================================
@cached("random_split_leakage_demo")
def random_split_leakage_demo(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                               test_fraction: float = config.TEST_FRACTION,
                               val_fraction: float = config.VAL_FRACTION,
                               random_state: int = config.RANDOM_STATE) -> dict:
    """Naive random row-wise split — same compound can appear in train AND
    test, exposing data leakage. Used only to demonstrate why compound_split
    is necessary."""
    idx = np.arange(len(X))
    rnd_tr_idx, rnd_te_idx = train_test_split(idx, test_size=test_fraction, random_state=random_state)
    rnd_tv_idx, rnd_vl_idx = train_test_split(rnd_tr_idx, test_size=val_fraction, random_state=random_state)

    X_train, med = clean_inf_nan(X.iloc[rnd_tv_idx].copy())
    X_val, _ = clean_inf_nan(X.iloc[rnd_vl_idx].copy(), med)
    X_test, _ = clean_inf_nan(X.iloc[rnd_te_idx].copy(), med)

    overlap = len(set(groups.iloc[rnd_tv_idx]) & set(groups.iloc[rnd_te_idx]))

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y.iloc[rnd_tv_idx], y_val=y.iloc[rnd_vl_idx], y_test=y.iloc[rnd_te_idx],
        groups_train=groups.iloc[rnd_tv_idx], groups_test=groups.iloc[rnd_te_idx],
        overlap=overlap, train_medians=med,
    )


# ==============================================================================
# 5-fold grouped cross-validation
# ==============================================================================
@cached("group_kfold_cv")
def group_kfold_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict,
                    n_folds: int = 5, random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """5-fold GroupKFold CV (grouped by mp_id). Trains one XGBoost model per
    fold and reports mean +/- std MAE / R2, to check that a single train/test
    split isn't a lucky/unlucky draw."""
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score

    gkf = GroupKFold(n_splits=n_folds)
    records = []

    for fold, (cv_tv_idx, cv_te_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_cv_tv, y_cv_tv, grp_cv_tv = X.iloc[cv_tv_idx], y.iloc[cv_tv_idx], groups.iloc[cv_tv_idx]

        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
        inner_tr, inner_vl = next(gss_inner.split(X_cv_tv, y_cv_tv, grp_cv_tv))

        X_cv_tr, med_cv = clean_inf_nan(X_cv_tv.iloc[inner_tr].copy())
        X_cv_vl, _ = clean_inf_nan(X_cv_tv.iloc[inner_vl].copy(), med_cv)
        X_cv_te, _ = clean_inf_nan(X.iloc[cv_te_idx].copy(), med_cv)

        y_cv_tr, y_cv_vl, y_cv_te = y_cv_tv.iloc[inner_tr], y_cv_tv.iloc[inner_vl], y.iloc[cv_te_idx]

        model_cv = xgb.XGBRegressor(**params)
        model_cv.fit(X_cv_tr, y_cv_tr, eval_set=[(X_cv_vl, y_cv_vl)], verbose=False)

        pred_cv = model_cv.predict(X_cv_te)
        fold_mae = mean_absolute_error(y_cv_te, pred_cv)
        fold_r2 = r2_score(y_cv_te, pred_cv)
        n_comp = groups.iloc[cv_te_idx].nunique()

        records.append({"fold": fold, "mae": fold_mae, "r2": fold_r2,
                         "n_test_compounds": n_comp, "n_test_rows": len(cv_te_idx)})
        print(f"  Fold {fold}: MAE={fold_mae:.5f} | R2={fold_r2:.4f} "
              f"| {n_comp} test compounds ({len(cv_te_idx)} rows)")

    cv_df = pd.DataFrame(records)
    print(f"\nMean MAE: {cv_df['mae'].mean():.5f} +/- {cv_df['mae'].std():.5f}")
    print(f"Mean R2 : {cv_df['r2'].mean():.4f} +/- {cv_df['r2'].std():.4f}")
    return cv_df
