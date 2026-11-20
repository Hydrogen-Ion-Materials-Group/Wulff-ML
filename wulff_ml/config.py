"""
Central configuration for the Wulff-ML pipeline.

Every path, filename and hyperparameter set that used to be hardcoded at the
top of each notebook now lives here, so changing one constant changes it
everywhere (scripts/, wulff_ml/, run_all.py all import from this module).
"""
import os

# ==============================================================================
# Paths
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_EXTERNAL_DIR  = os.path.join(PROJECT_ROOT, "data", "external")

MODEL_DIR          = os.path.join(PROJECT_ROOT, "models")
CATBOOST_TRAIN_DIR = os.path.join(MODEL_DIR, "catboost_info")   # CatBoost logs live here
RESULTS_DIR        = os.path.join(PROJECT_ROOT, "results")

# ── Figure output directories (one per figure category) ──────────────────────
FIGURES_DIR        = os.path.join(PROJECT_ROOT, "figures")
FIG_MAIN_DIR       = os.path.join(FIGURES_DIR, "main")          # paper-ready figures
FIG_SI_DIR         = os.path.join(FIGURES_DIR, "supplementary") # supplementary / SI
FIG_VALID_DIR      = os.path.join(FIGURES_DIR, "validation")    # multi-modal model comparison
FIG_CV_DIR         = os.path.join(FIGURES_DIR, "cv")            # cross-validation
FIG_HKL_DIR        = os.path.join(FIGURES_DIR, "hkl")           # P(h,k,l) / broken-bond
FIG_ABLATION_DIR   = os.path.join(FIGURES_DIR, "ablation")      # data & feature ablation
FIG_INTERACTION_DIR= os.path.join(FIGURES_DIR, "interaction")   # SHAP pairwise interaction
FIG_SHAP_DEP_DIR   = os.path.join(FIG_SI_DIR,  "shap_dependence")  # per-feature dep plots

ALL_DIRS = [
    DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_EXTERNAL_DIR,
    MODEL_DIR, CATBOOST_TRAIN_DIR, RESULTS_DIR,
    FIG_MAIN_DIR, FIG_SI_DIR, FIG_VALID_DIR, FIG_CV_DIR,
    FIG_HKL_DIR, FIG_ABLATION_DIR, FIG_INTERACTION_DIR, FIG_SHAP_DEP_DIR,
]


def check_installation():
    """The code reads data/raw/ and writes models/, results/ and figures/ next to
    the repository, so the package must be installed in editable mode
    (``pip install -e .``) or run from the cloned repository."""
    if not os.path.isdir(DATA_RAW_DIR):
        raise RuntimeError(
            f"Input data not found: {DATA_RAW_DIR}\n"
            "wulff-ml must be installed from the cloned repository in editable mode:\n"
            "    cd wulff-ml\n"
            "    pip install -e ."
        )


def ensure_dirs():
    """Create every project directory if it doesn't already exist."""
    check_installation()
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)


# ==============================================================================
# Raw input filenames — place these CSVs under data/raw/
#
# There are only 6 source files. `training_data.csv` does double duty:
#   (a) scripts/01 reads its `mp_id` column to know which compounds to pull
#       bulk descriptors for from the Materials Project
#   (b) scripts/02+ read it as the DFT surface-energy TARGET table
#       (mp_id, hkl, gamma_eV_per_ang2, area_fraction, ...)
# `prediction_mpids.csv` plays the equivalent role for the prediction pool
# (compounds with no DFT surface energy yet — only used to fetch descriptors).
# ==============================================================================
TRAINING_DATA_CSV    = os.path.join(DATA_RAW_DIR, "training_data.csv")
PREDICTION_MPIDS_CSV = os.path.join(DATA_RAW_DIR, "prediction_mpids.csv")

# Step 0 outputs / Step 1 inputs — bulk descriptors
BULK_TRAIN_RAW   = os.path.join(DATA_RAW_DIR, "03_DFT_surff_bulk_descriptors_2575.csv")
BULK_PREDICT_RAW = os.path.join(DATA_RAW_DIR, "04_wulff_to_pred_bulk_data.csv")

# Magpie-augmented bulk descriptors (output of feature engineering step)
BULK_TRAIN_MAGPIE   = os.path.join(DATA_RAW_DIR, "05_DFT_surff_bulk_descriptors_2575_magpie.csv")
BULK_PREDICT_MAGPIE = os.path.join(DATA_RAW_DIR, "06_wulff_to_pred_bulk_descriptor_magpie.csv")

# DFT surface-energy target table: mp_id, hkl, gamma_eV_per_ang2, area_fraction, ...
# (this is the SAME physical file as TRAINING_DATA_CSV — kept as a separate
# name only so call sites read clearly; do not point it at a different file)
SURFACE_ENERGY_TARGETS_CSV = TRAINING_DATA_CSV

# Materials Project API key: not stored here. Use wulff_ml.api_key.get_mp_api_key(),
# which reads the MP_API_KEY environment variable or the key saved by
# `wulff-key`, and asks for it in the terminal if neither exists.

# ==============================================================================
# Modelling constants
# ==============================================================================
TARGET_GAMMA = "gamma_eV_per_ang2"
RANDOM_STATE = 22
VAL_FRACTION = 0.20
TEST_FRACTION = 0.20

CAT_COLS = ["crystal_system", "point_group"]

DROP_COLS = [
    "mp_id", "formula_pretty", "hkl", TARGET_GAMMA,
    "area_fraction", "weighted_surface_energy_EV_PER_ANG2",
    "weighted_surface_energy",
]

# ==============================================================================
# XGBoost hyperparameters (primary model — gamma surface energy regressor)
# ==============================================================================
FIXED_PARAMS_XGB = {
    "n_estimators":          10000,
    "tree_method":           "hist",
    "early_stopping_rounds": 120,
    "n_jobs":                -1,
    "enable_categorical":    True,
    "objective":             "reg:absoluteerror",
    "eval_metric":           "mae",
}

PARAMS_XGB = {
    **FIXED_PARAMS_XGB,
    "random_state":      RANDOM_STATE,
    "learning_rate":     0.0358371388,
    "max_depth":         5,
    "min_child_weight":  6,
    "subsample":         0.85,
    "colsample_bytree":  0.60,
    "colsample_bylevel": 1.0,
    "colsample_bynode":  0.80,
    "reg_alpha":         0.10,
    "reg_lambda":         2.05,
    "gamma":             0.10,
    "max_cat_to_onehot": 4,
    "max_cat_threshold": 64,
}

# ==============================================================================
# LightGBM hyperparameters (multi-modal validation)
# ==============================================================================
FIXED_PARAMS_LGBM = {
    "n_estimators":  5000,
    "learning_rate": 0.0358371388,
    "n_jobs":        -1,
    "verbose":       -1,
    "objective":     "mae",
    "metric":        "mae",
}

PARAMS_LGBM = {
    **FIXED_PARAMS_LGBM,
    "random_state":       RANDOM_STATE,
    "max_depth":          5,
    "num_leaves":         31,
    "min_child_samples":  20,
    "subsample":          0.85,
    "colsample_bytree":   0.60,
    "subsample_freq":     1,
    "reg_alpha":          0.10,
    "reg_lambda":         2.05,
    "min_split_gain":     0.10,
    "cat_smooth":         10,
}

# ==============================================================================
# CatBoost hyperparameters (multi-modal validation)
# Training artefacts (logs, snapshots) go into models/catboost_info/ so they
# never pollute the project root.
# ==============================================================================
PARAMS_CATBOOST = {
    "iterations":            5000,
    "learning_rate":         0.0358371388,
    "depth":                 5,
    "l2_leaf_reg":           2.05,
    "random_seed":           RANDOM_STATE,
    "loss_function":         "MAE",
    "eval_metric":           "MAE",
    "early_stopping_rounds": 120,
    "verbose":               False,
    "train_dir":             CATBOOST_TRAIN_DIR,
}

# ==============================================================================
# Random Forest / Extra Trees hyperparameters (multi-modal validation)
# ==============================================================================
PARAMS_RF = {
    "n_estimators":      500,
    "max_depth":         None,
    "min_samples_leaf":  4,
    "min_samples_split": 8,
    "max_features":      0.60,
    "n_jobs":            -1,
    "random_state":      RANDOM_STATE,
}

PARAMS_ET = {
    "n_estimators":      500,
    "max_depth":         None,
    "min_samples_leaf":  4,
    "min_samples_split": 8,
    "max_features":      0.60,
    "n_jobs":            -1,
    "random_state":      RANDOM_STATE,
}

# ==============================================================================
# Elemental split (scripts/05_element_split.py)
# ==============================================================================
TEST_ELEMENTS = ["Cu", "Ni"]

# ==============================================================================
# SHAP — explicit feature rename map for the "headline" beeswarm figure
# ==============================================================================
SHAP_EXPLICIT_RENAME = {
    "mean_melting_point": "Melting Point",
    "volume_per_atom": "Volume per atom",
    "MagpieData mean NdUnfilled": "No of unfilled d electron",
    "MagpieData maximum NdValence": "No of valence d electron",
    "MagpieData avg_dev MendeleevNumber": "Periodicity",
    "k_norm": "hkl normalised",
    "homogeneous_poisson": "Poisson ratio",
    "formation_energy_per_atom": "formation energy per atom",
    "total_magnetization": "Total Magnatisation",
    "mean_atomic_radius": "Atomic Radius",
    "MagpieData maximum Electronegativity": "Electronegativity",
}

# ==============================================================================
# Wulff construction (scripts/08_wulff_prediction.py)
# ==============================================================================
WULFF_OUTPUT_SUBDIR = "wulff_output"
