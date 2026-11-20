#!/usr/bin/env python3
"""
Step 1 — Prepare bulk descriptors and Magpie composition features.

Pulls bulk descriptors from the Materials Project for both the training
compound pool and the prediction pool, then augments both with ~130 Magpie
elemental-property statistics via matminer.

Inputs  (data/raw/):
    training_data.csv        mp_id, hkl, gamma_eV_per_ang2, ...
                              (mp_id column is used here to know which
                              compounds to query MP for; the full file is
                              re-read later, by build_feature_matrix(), as
                              the surface-energy TARGET table)
    prediction_mpids.csv     mp_id, formula   (compounds to predict)

Outputs (data/raw/):
    03_DFT_surff_bulk_descriptors_2575.csv
    04_wulff_to_pred_bulk_data.csv
    05_DFT_surff_bulk_descriptors_2575_magpie.csv
    06_wulff_to_pred_bulk_descriptor_magpie.csv

If your data/raw/ already has all 6 CSVs (the usual case — see the README),
this script has nothing to do and you can skip straight to
scripts/02_train_model.py.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wulff_ml import config
from wulff_ml.api_key import get_mp_api_key
from wulff_ml.feature_engineering import fetch_bulk_descriptors_from_mp, add_magpie_features


def main():
    config.ensure_dirs()
    api_key = get_mp_api_key()   # asks once in the terminal if no key is stored yet

    print("Step 1a — Fetching bulk descriptors from Materials Project")
    fetch_bulk_descriptors_from_mp(
        csv_path=config.TRAINING_DATA_CSV, id_col="mp_id",
        out_path=config.BULK_TRAIN_RAW, api_key=api_key,
    )
    fetch_bulk_descriptors_from_mp(
        csv_path=config.PREDICTION_MPIDS_CSV, id_col="mp_id",
        out_path=config.BULK_PREDICT_RAW, api_key=api_key,
    )

    print("\nStep 1b — Adding Magpie composition features")
    add_magpie_features(config.BULK_TRAIN_RAW, config.BULK_TRAIN_MAGPIE, label="Training set")
    add_magpie_features(config.BULK_PREDICT_RAW, config.BULK_PREDICT_MAGPIE, label="Prediction set")

    print("\nAll done.")
    print(f"  Training descriptors   : {config.BULK_TRAIN_MAGPIE}")
    print(f"  Prediction descriptors : {config.BULK_PREDICT_MAGPIE}")


if __name__ == "__main__":
    main()
