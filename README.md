<div align="center">

# Wulff-ML

**Slab-free machine-learning prediction of intermetallic surface energies and equilibrium crystal morphologies**

A Python toolkit for predicting facet-resolved surface energies from bulk materials descriptors and constructing the corresponding Wulff shapes.

</div>

---

## Overview

**Wulff-ML** is an open-source Python package for predicting the surface energies, γ(hkl), of intermetallic crystals using machine learning and for constructing their equilibrium Wulff shapes.

The model is an XGBoost regressor trained on density functional theory (DFT) surface energies from the [SurFF database](https://doi.org/10.1038/s43588-025-00839-0). Its inputs combine bulk-material descriptors from the [Materials Project](https://materialsproject.org/) with geometric features derived from the Miller indices. The predicted facet energies are used to construct Wulff shapes and the area fraction of each exposed facet. Surface energies are reported in J/m².

The repository also contains the validation, ablation, interpretability (SHAP), and Wulff analyses that accompany the manuscript.

---

## Installation

Clone the repository and install it in editable mode (required, because the package reads its data from, and writes results to, the cloned repository):

```bash
git clone https://github.com/bhavukRohilla/WulffML.git
cd WulffML
pip install -e .
```

Using a separate environment is recommended. A Conda environment is provided:

```bash
conda env create -f environment.yml
conda activate wulff_ml
pip install -e .
```

Wulff-ML was developed and tested with Python 3.13.8. The installation also provides the commands `wulff-predict` and `wulff-key`.

> **Important:** Use XGBoost 3.2.0, which is pinned in `setup.py` and `requirements.txt`. Other versions have not been validated; for example, XGBoost 3.4.1 produced a model that ignored the Miller-index features. See the [documentation](docs/DOCUMENTATION.md#xgboost-version) for details.

---

## Materials Project API key

Downloading data from the Materials Project requires a free API key. No setup is needed in advance: the first time a key is required, the program asks for it in the terminal. Get your key from the [Materials Project API page](https://next-gen.materialsproject.org/api) and paste it when prompted. It is checked, saved outside the repository (`~/.wulff_ml/mp_api_key`), and requested only once.

```bash
wulff-key             # Enter or replace the saved key
wulff-key --show      # Show the key in masked form
wulff-key --forget    # Delete the saved key
```

Alternatively, set the `MP_API_KEY` environment variable. See the [documentation](docs/DOCUMENTATION.md#materials-project-api-key) for details.

---

## Usage

### Predict compounds

All you need are the Materials Project IDs of the compounds (for example, `mp-149`):

```bash
wulff-predict                          # Enter the IDs interactively
wulff-predict mp-1077900 mp-21119      # Provide IDs directly
wulff-predict --file ids.txt           # Read IDs from a text file
wulff-predict mp-149 --no-wulff        # Surface energies only, no Wulff shapes
wulff-predict mp-149 --out my_results  # Choose an output directory
```

The first run trains the model once on the bundled DFT data, which usually takes up to a few minutes; later runs start immediately. Results are saved in `results/predict/<date_time>/` and include the predicted surface energy of every facet (CSV) and the Wulff shapes (PNG, PDF, SVG). Surface energies are predicted for the 35 Miller indices of the training set (maximum index 2).

You can also run predictions directly in the provided [Jupyter notebook](https://github.com/bhavukRohilla/WulffML/blob/main/wulffML_prediction.ipynb).

### Run the complete test analysis

```bash
python run_all.py                  # Run all analysis steps in order
python run_all.py --only 02 06     # Run only selected steps
```

The 13 analysis steps are listed in the [documentation](docs/DOCUMENTATION.md#analysis-pipeline).

---

## Documentation

The [full documentation](docs/DOCUMENTATION.md) covers:

- [Installation details](docs/DOCUMENTATION.md#installation-details), including tested software versions
- [API key management](docs/DOCUMENTATION.md#materials-project-api-key)
- [Prediction options and outputs](docs/DOCUMENTATION.md#predicting-compounds), including the Python interface
- [The analysis pipeline](docs/DOCUMENTATION.md#running-the-complete-analysis)
- [Data files and dataset summary](docs/DOCUMENTATION.md#data)
- [Method](docs/DOCUMENTATION.md#method): features, model, evaluation, units, and caching
- [Repository structure](docs/DOCUMENTATION.md#repository-structure)
- [Notes and limitations](docs/DOCUMENTATION.md#notes-and-limitations)

---

## Data

The DFT surface energies are taken from the [SurFF database](https://doi.org/10.1038/s43588-025-00839-0) ([figshare](https://doi.org/10.6084/m9.figshare.26395756), [Zenodo](https://doi.org/10.5281/zenodo.15480651)), and the bulk descriptors from the [Materials Project](https://materialsproject.org/). All input files are included in `data/raw/`. If you reuse the data, please cite the sources listed in the [References](#references) and follow their terms of use.

---

## License

The Python code is licensed under the [MIT License](LICENSE).

---

## Citation

The manuscript describing this work is in preparation. The citation will be added here after publication.
The data is stored on Zenodo - 10.5281/zenodo.24072781

If you use the surface-energy data, please also cite the SurFF paper and dataset listed below.

---

## Contact
- **Name:** Soshi Iimura
- **Email:** IIMURA.Soshi@nims.go.jps
- **Affiliation:** NIMS (Japan)
  
- **Name:** Bhavuk Rohilla
- **Email:** ROHILLA.Bhavuk@nims.go.jp
- **Affiliation:** NIMS (Japan)

---

## References

1. Yin, J., Chen, H., Qiu, J., Li, W., He, P., Li, J., Karimi, I. A., Lan, X., Wang, T. & Wang, X. SurFF: a foundation model for surface exposure and morphology across intermetallic crystals. *Nature Computational Science* **5**, 782–792 (2025). https://doi.org/10.1038/s43588-025-00839-0
2. Yin, J. *et al.* SurFF_coredatafiles. *figshare* (2025). https://doi.org/10.6084/m9.figshare.26395756
3. Yin, J. *et al.* SurFF: 1.0. *Zenodo* (2025). https://doi.org/10.5281/zenodo.15480651
4. Jain, A. *et al.* Commentary: The Materials Project: a materials genome approach to accelerating materials innovation. *APL Materials* **1**, 011002 (2013). https://doi.org/10.1063/1.4812323
5. Ong, S. P. *et al.* Python Materials Genomics (pymatgen): a robust, open-source Python library for materials analysis. *Computational Materials Science* **68**, 314–319 (2013). https://doi.org/10.1016/j.commatsci.2012.10.028
6. Ward, L. *et al.* Matminer: an open source toolkit for materials data mining. *Computational Materials Science* **152**, 60–69 (2018). https://doi.org/10.1016/j.commatsci.2018.05.018
7. Ward, L., Agrawal, A., Choudhary, A. & Wolverton, C. A general-purpose machine learning framework for predicting properties of inorganic materials. *npj Computational Materials* **2**, 16028 (2016). https://doi.org/10.1038/npjcompumats.2016.28
8. Chen, T. & Guestrin, C. XGBoost: a scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794 (ACM, 2016). https://doi.org/10.1145/2939672.2939785
9. Lundberg, S. M. *et al.* From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence* **2**, 56–67 (2020). https://doi.org/10.1038/s42256-019-0138-9
