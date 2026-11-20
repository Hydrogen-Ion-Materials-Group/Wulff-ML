from setuptools import setup, find_packages

setup(
    name="wulff_ml",
    version="0.1.0",
    description="ML pipeline for predicting DFT surface energies and constructing Wulff shapes",
    packages=find_packages(include=["wulff_ml", "wulff_ml.*"]),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "wulff-predict=wulff_ml.cli:predict_main",
            "wulff-key=wulff_ml.api_key:main",
        ],
    },
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "xgboost==3.2.0",   # other versions are not supported (see README)
        "lightgbm>=4.0",
        "catboost>=1.2",
        "shap>=0.44",
        "matplotlib>=3.7",
        "seaborn>=0.13",
        "joblib>=1.3",
        "tqdm>=4.65",
        "optuna>=3.4",
        "matminer>=0.9",
        "pymatgen>=2024.2",
        "mp-api>=0.41",
    ],
)
