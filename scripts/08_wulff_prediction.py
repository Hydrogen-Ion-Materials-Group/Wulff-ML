#!/usr/bin/env python3
"""
Step 8 — Build Wulff equilibrium crystal shapes from predicted facet surface
energies (output of scripts/02_train_model.py).

Outputs
-------
results/wulff_output/
    <mpid>.cif
    <mpid>_<formula>_wulff.png / .pdf / .svg / _raw.png
    area_fractions_wulff.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wulff_ml import config
from wulff_ml.api_key import get_mp_api_key
from wulff_ml.wulff_construction import build_wulff_shapes
from wulff_ml.utils import banner


def main():
    config.ensure_dirs()
    banner("STEP 8: WULFF SHAPE CONSTRUCTION")

    predictions_csv = os.path.join(config.RESULTS_DIR, "all_facets_gamma.csv")
    if not os.path.exists(predictions_csv):
        raise FileNotFoundError(
            f"{predictions_csv} not found — run scripts/02_train_model.py first."
        )

    # Asks for the Materials Project API key (once) if none is stored yet.
    api_key = get_mp_api_key()
    build_wulff_shapes(predictions_csv, api_key=api_key)


if __name__ == "__main__":
    main()
