# Crime Data Dimensionality Reduction using PCA–Neural Regression

**Domain:** Public Safety / Social Data  
**Dataset:** UCI Communities and Crime  
**Task:** Predict `ViolentCrimesPerPop` with an MLP regression model, using PCA to reduce the number of input features.

## 1. Project overview

This project investigates whether Principal Component Analysis (PCA) can reduce the dimensionality of community-level socioeconomic and law-enforcement features while retaining useful information for violent-crime regression.

The pipeline:
1. Downloads and inspects the UCI Communities and Crime dataset.
2. Separates the target and removes identifier/non-predictive columns.
3. Splits the data into train and test sets **before fitting preprocessing or PCA**.
4. Imputes missing values and standardizes features using training data only.
5. Fits PCA on training data and keeps enough components to meet a configurable explained-variance threshold.
6. Trains an MLP regressor using stochastic gradient descent with mini-batches.
7. Compares the PCA + MLP model with a mean-prediction baseline and a full-feature MLP.
8. Reports MAE, RMSE, R², explained variance, retained component count, and parameter reduction, and saves plots.

> **Important:** This dataset contains observational community-level data. Predictions are not causal conclusions and should not be used to profile individuals or make operational policing decisions.

## 2. Dataset

- Official source: [UCI Machine Learning Repository — Communities and Crime](https://archive.ics.uci.edu/dataset/183/communities+and+crime)
- Expected dataset size: approximately 1,994 community records.
- Target: `ViolentCrimesPerPop`.
- The raw data includes identifier fields and missing values represented by `?`.

The script downloads the official UCI ZIP at runtime. If the repository changes its download URL, download the dataset from the official page and place `communities.data` and `communities.names` in `data/raw/`.

## 3. Project structure

```text
crime_pca_neural_regression/
├── data/
│   └── raw/                  # downloaded data (created automatically)
├── outputs/                  # CSV reports and plots (created automatically)
├── src/
│   └── train.py              # end-to-end pipeline
├── requirements.txt
├── run.py
└── README.md
```

## 4. Setup and run

Python 3.10+ recommended.

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python run.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

If you already downloaded the dataset, place `communities.data` in `data/raw/`; the pipeline will use it instead of downloading.

## 5. Methodology

- **Split:** 80% training / 20% testing, fixed random seed.
- **Missing values:** convert `?` to missing values and median-impute based on training data.
- **Scaling:** standardize features using training-set statistics.
- **PCA:** fit only on the training set; retain the smallest number of components reaching the configured variance threshold (default 95%).
- **Neural network:** one hidden layer MLP, ReLU activation, SGD optimizer, mini-batch updates, adaptive learning rate.
- **Baseline:** `DummyRegressor(strategy="mean")`.
- **Comparison:** PCA + MLP versus full-feature MLP.
- **Metrics:** MAE, RMSE, R². PCA explained-variance ratio and dimensionality reduction are also reported.

The MLP uses a validation fraction from the training split for early stopping. The final test set remains untouched until evaluation.

## 6. Output files

After a successful run, check `outputs/`:

- `metrics.csv` — baseline and model metrics.
- `pca_summary.json` — component count, cumulative explained variance, and dimensionality reduction.
- `explained_variance.png` — cumulative PCA explained variance.
- `actual_vs_predicted.png` — held-out test predictions for the PCA model.
- `training_curve.png` — training loss curve for the PCA model.
- `feature_missingness.csv` — missing-value count by feature before imputation.

## 7. How to interpret results

- Lower **MAE/RMSE** is better.
- Higher **R²** is better; a negative test R² means the model performed worse than predicting the test-set mean.
- **Explained variance** describes feature variance retained by PCA; it is not the same as target prediction accuracy.
- **Parameter reduction** compares the number of input features before and after PCA. It is an input-dimensionality reduction estimate, not a guarantee of faster inference or better generalization.
- If PCA + MLP underperforms the full-feature MLP, report that honestly. PCA preserves high-variance directions, which are not necessarily the most predictive directions for the target.

## 8. Suggested report sections

1. Abstract and problem statement
2. Dataset description and ethical limitations
3. Data cleaning and leakage prevention
4. PCA method and explained-variance plot
5. Neural network architecture and mini-batch SGD settings
6. Baseline and full-feature comparison
7. Metrics and plots
8. Limitations and conclusion

## 9. Reproducibility

The random seed is fixed in `src/train.py`. Results may vary slightly across Python/scikit-learn versions. Cite the UCI dataset in your submission and include the metrics generated by your own run; do not report made-up results.
