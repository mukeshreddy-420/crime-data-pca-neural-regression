# Crime Data Dimensionality Reduction using PCA–Neural Regression

## Overview
This project predicts violent crime rates using the UCI Communities and Crime dataset. It uses Principal Component Analysis (PCA) for dimensionality reduction and a Multilayer Perceptron (MLP) for regression.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- PCA
- MLP Neural Network
- Mini-Batch Stochastic Gradient Descent (SGD)

## Dataset
UCI Communities and Crime Dataset:  
https://archive.ics.uci.edu/dataset/183/communities+and+crime

**Target variable:** `ViolentCrimesPerPop`

## Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean Baseline | 0.1788 | 0.2212 | -0.0217 |
| Full-feature MLP | 0.0979 | 0.1438 | 0.5681 |
| PCA + MLP | 0.1043 | 0.1475 | 0.5459 |

PCA reduced the input features from 122 to 43 while retaining approximately 95% of the variance.

## How to Run

```bash
git clone https://github.com/YOUR-USERNAME/crime-data-pca-neural-regression.git
cd crime-data-pca-neural-regression
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## Conclusion
The full-feature MLP achieved slightly better predictive accuracy, while PCA reduced the number of input dimensions by approximately 64.75%.

## Author
Mukesh Redy  
GITAM University — CSE AIML
