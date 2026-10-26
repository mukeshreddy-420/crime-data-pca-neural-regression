from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"
DATA_URL = "https://archive.ics.uci.edu/static/public/183/communities+and+crime.zip"
SEED = 42
TEST_SIZE = 0.20
PCA_VARIANCE = 0.95
TARGET = "ViolentCrimesPerPop"

def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset() -> Path:
    """Download and extract the official UCI ZIP if the raw data is absent."""
    data_path = RAW_DIR / "communities.data"
    if data_path.exists():
        print(f"Using existing dataset: {data_path}")
        return data_path

    print("Downloading Communities and Crime from the UCI repository...")
    try:
        with urlopen(DATA_URL, timeout=45) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.namelist()
            candidate = next(
                (name for name in members if name.lower().endswith("communities.data")),
                None,
            )
            if candidate is None:
                raise FileNotFoundError(
                    "communities.data was not found in the downloaded ZIP. "
                    "Download it manually from the official UCI dataset page."
                )
            data_path.write_bytes(archive.read(candidate))
            names_candidate = next(
                (name for name in members if name.lower().endswith("communities.names")),
                None,
            )
            if names_candidate:
                (RAW_DIR / "communities.names").write_bytes(archive.read(names_candidate))
    except Exception as exc:
        raise RuntimeError(
            f"Dataset download failed: {exc}\n"
            "Download the dataset from https://archive.ics.uci.edu/dataset/183/communities+and+crime "
            "and put communities.data in data/raw/."
        ) from exc
    return data_path

def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw CSV-like data using the attribute names documented by UCI."""
    names_path = RAW_DIR / "communities.names"
    if not names_path.exists():
        # Attribute order from the UCI dataset documentation.
        # The last attribute is the violent-crime target.
        raise FileNotFoundError(
            "communities.names was not available after download. Please download both "
            "communities.data and communities.names from the official UCI page and place "
            "them in data/raw/. The names file is required to safely identify the target."
        )

    text = names_path.read_text(encoding="latin-1", errors="replace")
    attributes = []
    in_attributes = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("@attribute"):
            parts = stripped.split()
            if len(parts) >= 2:
                attributes.append(parts[1].strip("'\""))
        elif stripped.lower().startswith("attribute information"):
            in_attributes = True
        elif in_attributes:
            # The original .names file describes attributes in prose, not ARFF.
            continue

    # UCI .names file often uses a numbered prose list rather than @attribute lines.
    # Use the canonical column order if no ARFF declarations are present.
    if len(attributes) != 128:
        attributes = [
            "state", "county", "community", "communityname", "fold",
            "population", "householdsize", "racepctblack", "racePctWhite",
            "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29",
            "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome",
            "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst",
            "pctWRetire", "medFamInc", "perCapInc", " whitePerCap",
        ]
        # Parse attribute names from the documented numbered list when possible.
        import re
        parsed = []
        for line in text.splitlines():
            match = re.match(r"\s*\d+\.\s*([A-Za-z][A-Za-z0-9_]*)(?:\s|$)", line)
            if match:
                parsed.append(match.group(1))
        if len(parsed) >= 100:
            attributes = parsed
        else:
            # Fail safely rather than silently assigning incorrect feature names.
            raise ValueError(
                "Could not parse 128 column names from communities.names. "
                "Check the UCI names file and its format."
            )

    frame = pd.read_csv(
        path, header=None, names=attributes, na_values=["?", " ?", ""],
        skipinitialspace=True, low_memory=False
    )
    if TARGET not in frame.columns:
        # Case-insensitive matching handles small naming differences.
        matches = [c for c in frame.columns if c.strip().lower() == TARGET.lower()]
        if matches:
            frame = frame.rename(columns={matches[0]: TARGET})
        else:
            raise ValueError(
                f"Target '{TARGET}' was not found. Last columns are: {list(frame.columns[-5:])}"
            )

    # Exclude the target rows where the target itself is missing.
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.loc[frame[TARGET].notna()].copy()

    # Identifier/administrative fields are not predictive inputs.
    non_features = [c for c in ["state", "county", "community", "communityname", "fold"] if c in frame.columns]
    y = frame[TARGET].astype(float)
    X = frame.drop(columns=[TARGET] + non_features, errors="ignore")

    # Keep numeric columns only; convert unexpected strings to NaN for imputation.
    X = X.apply(pd.to_numeric, errors="coerce")
    # Remove columns that are entirely missing (fit/transform stability).
    X = X.dropna(axis=1, how="all")
    return X, y

def make_mlp(seed: int = SEED) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="sgd",
        learning_rate="adaptive",
        learning_rate_init=0.01,
        batch_size=32,                 # explicit mini-batch size
        max_iter=500,
        early_stopping=True,           # validation split comes only from training data
        validation_fraction=0.15,
        n_iter_no_change=25,
        tol=1e-4,
        random_state=seed,
    )

def score_model(name: str, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return {
        "model": name,
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, pred)),
    }

def main() -> None:
    ensure_dirs()
    path = download_dataset()
    X, y = load_data(path)

    print(f"Loaded {len(X):,} rows and {X.shape[1]} numeric candidate features.")
    missing = X.isna().sum().sort_values(ascending=False)
    missing.rename("missing_values").to_csv(OUT_DIR / "feature_missingness.csv", header=True)

    # Split first. No imputer, scaler, or PCA sees the test data during fitting.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # Fit preprocessing only on training data.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # PCA selection is based exclusively on training-set features.
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    pca_model = make_mlp()
    pca_model.fit(X_train_pca, y_train)

    full_model = make_mlp()
    full_model.fit(X_train_scaled, y_train)

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_scores = score_model("Mean baseline", baseline, np.zeros((len(y_test), 1)), y_test)

    rows = [
        baseline_scores,
        score_model("Full-feature MLP (mini-batch SGD)", full_model, X_test_scaled, y_test),
        score_model("PCA + MLP (mini-batch SGD)", pca_model, X_test_pca, y_test),
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)

    reduction = 100.0 * (1.0 - X_train_pca.shape[1] / X_train_scaled.shape[1])
    summary = {
        "dataset_rows_used": int(len(X)),
        "original_numeric_features": int(X_train_scaled.shape[1]),
        "pca_components_retained": int(X_train_pca.shape[1]),
        "requested_cumulative_explained_variance": PCA_VARIANCE,
        "actual_cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "input_dimension_reduction_percent": float(reduction),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target": TARGET,
        "random_seed": SEED,
        "mlp_hidden_layers": [64],
        "optimizer": "SGD",
        "batch_size": 32,
        "max_iter": 500,
    }
    (OUT_DIR / "pca_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot cumulative explained variance.
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cumulative) + 1), cumulative, marker=".", linewidth=1)
    plt.axhline(PCA_VARIANCE, linestyle="--", label=f"Target variance = {PCA_VARIANCE:.0%}")
    plt.axvline(len(cumulative), linestyle=":", label=f"Retained components = {len(cumulative)}")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Cumulative Explained Variance (training data)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "explained_variance.png", dpi=160)
    plt.close()

    # Plot actual vs predicted for the PCA model.
    predictions = pca_model.predict(X_test_pca)
    low = float(min(np.min(y_test), np.min(predictions)))
    high = float(max(np.max(y_test), np.max(predictions)))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual ViolentCrimesPerPop")
    plt.ylabel("Predicted ViolentCrimesPerPop")
    plt.title("PCA + MLP: Actual vs Predicted (held-out test set)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    # Training loss curve for the PCA model.
    plt.figure(figsize=(8, 5))
    plt.plot(pca_model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title("PCA + MLP Training Loss (mini-batch SGD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curve.png", dpi=160)
    plt.close()

    print("\nEvaluation metrics (held-out test set):")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nPCA summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nReports and plots saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
