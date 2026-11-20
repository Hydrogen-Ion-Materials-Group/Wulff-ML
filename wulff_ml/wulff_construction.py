"""
Wulff shape construction from predicted facet surface energies.

Downloads each compound's conventional structure from the Materials
Project, builds a WulffShape from the predicted gamma(hkl) values, saves
the rendered crystal shape, and aggregates equilibrium facet area fractions
into a single master CSV.
"""
import ast
import os

import pandas as pd
import matplotlib.pyplot as plt

from wulff_ml import config
from wulff_ml.visualization import save_figure
from wulff_ml.cache import cached
from wulff_ml.api_key import get_mp_api_key


@cached("wulff_shape_for_compound", ignore_kwargs=["api_key"])
def _build_one_wulff_shape(mpid: str, group_records: tuple, api_key: str = None):
    """Pure computation for a single compound: fetch structure, build the
    WulffShape, compute area fractions. Cached per-compound (mpid + its
    predicted facet rows), so an interrupted run resumes without re-fetching
    or re-computing compounds already done. Returns a dict with the
    area-fraction rows, the structure CIF text, and Wulff summary stats —
    or {"error": ...} on failure. (Plot rendering happens outside the cache,
    in build_wulff_shapes(), since matplotlib figures aren't cleanly
    cache-able and are cheap to redraw from the cached shape data.)
    """
    from mp_api.client import MPRester
    from pymatgen.analysis.wulff import WulffShape

    api_key = api_key or get_mp_api_key()
    try:
        with MPRester(api_key) as mpr:
            structure = mpr.get_structure_by_material_id(mpid, conventional_unit_cell=True)
        formula = structure.composition.reduced_formula

        millers, gammas = [], []
        for hkl_str, gamma in group_records:
            millers.append(tuple(ast.literal_eval(hkl_str)))
            gammas.append(float(gamma))

        wulff = WulffShape(lattice=structure.lattice, miller_list=millers, e_surf_list=gammas)

        area_fraction_rows = [
            {"mp_id": mpid, "formula_pretty": formula, "hkl": str(hkl), "area_fraction": round(float(frac), 8)}
            for hkl, frac in wulff.area_fraction_dict.items()
        ]
        return {
            "formula": formula,
            "cif_text": structure.to(fmt="cif"),
            "lattice_dict": structure.lattice.as_dict(),
            "weighted_surface_energy": wulff.weighted_surface_energy,
            "shape_factor": wulff.shape_factor,
            "anisotropy": wulff.anisotropy,
            "miller_list": millers,
            "e_surf_list": gammas,
            "area_fraction_rows": area_fraction_rows,
        }
    except Exception as e:
        return {"error": str(e)}


def build_wulff_shapes(predictions_csv: str, api_key: str = None,
                        output_dir: str = None) -> pd.DataFrame:
    """
    predictions_csv: output of model_training.predict_all_facets()
                      (columns: mp_id, formula_pretty, hkl, pred_gamma_eV_per_ang2)
    Returns a DataFrame of facet area fractions for every compound that
    succeeded, and saves it to <output_dir>/area_fractions_wulff.csv.

    The expensive part (MP structure fetch + WulffShape construction) is
    cached per-compound: re-running after an interruption, or after adding
    new compounds to predictions_csv, only does work for compounds that
    haven't been processed yet.
    """
    from pymatgen.analysis.wulff import WulffShape
    from pymatgen.core import Lattice

    api_key = api_key or get_mp_api_key()
    output_dir = output_dir or os.path.join(config.RESULTS_DIR, config.WULFF_OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(predictions_csv)
    all_results = []

    for mpid, group in df.groupby("mp_id"):
        print(f"\nProcessing {mpid}")
        group_records = tuple(
            (row["hkl"], row["pred_gamma_eV_per_ang2"]) for _, row in group.iterrows()
        )
        result = _build_one_wulff_shape(mpid, group_records, api_key=api_key)

        if "error" in result:
            print(f"  Failed for {mpid}: {result['error']}")
            continue

        formula = result["formula"]
        cif_path = os.path.join(output_dir, f"{mpid}.cif")
        with open(cif_path, "w") as f:
            f.write(result["cif_text"])

        print(f"  Weighted surface energy : {result['weighted_surface_energy']:.6f}")
        print(f"  Shape factor            : {result['shape_factor']:.6f}")
        print(f"  Anisotropy              : {result['anisotropy']:.6f}")

        # Rebuild the WulffShape from cached inputs purely to render the plot
        # (cheap — no network call, no re-fetch).
        lattice = Lattice.from_dict(result["lattice_dict"])
        wulff = WulffShape(lattice=lattice, miller_list=result["miller_list"],
                            e_surf_list=result["e_surf_list"])
        ax = wulff.get_plot()
        fig = ax.figure
        fig.text(0.5, 0.2, f"{formula}  |  {mpid}", ha="center", va="bottom",
                  fontsize=22, fontweight="bold", color="#1a1a1a",
                  transform=fig.transFigure)
        safe_formula = formula.replace("/", "_")
        save_figure(fig, os.path.join(output_dir, f"{mpid}_{safe_formula}_wulff.png"), dpi=600)
        plt.close(fig)

        all_results.extend(result["area_fraction_rows"])

    results_df = pd.DataFrame(all_results)
    master_csv = os.path.join(output_dir, "area_fractions_wulff.csv")
    results_df.to_csv(master_csv, index=False)
    print(f"\nMaster area fractions CSV saved -> {master_csv}")
    return results_df
