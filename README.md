# Machine Learning Student Project: Regression

This is a machine learning project for intermediate students. It covers one thing, **regression** (predicting a number), using scikit-learn, and the code is heavily commented to explain each step.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [The Seven Models](#the-seven-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Module Guide](#module-guide)
- [Exercise](#exercise)
- [Troubleshooting](#troubleshooting)

---

## Overview

The project trains seven regression models on the same data, evaluates them all the same way, and compares them in a table and in plots.

**Dataset**: scikit-learn's Diabetes dataset. It has 442 patients, 10 baseline measurements (age, sex, BMI, blood pressure and six blood serum values), and the target is a number measuring disease progression one year later.

**The workflow:**

```
1. Load and validate the data
2. Split into 80% training / 20% test
3. For each model:
     a. 5-fold cross-validation on the training set
     b. Fit on the training set
     c. Predict the test set → MAE, MSE, RMSE, R²
4. Rank the models by test RMSE and save the comparison table
5. Plot: predictions, model comparison, residuals, feature importance
```

### Learning Objectives

1. How the regression workflow works from start to finish
2. How linear, tree-based and instance-based models differ
3. How to read MAE, MSE, RMSE and R², and why cross-validation matters
4. When feature scaling is required (Ridge, Lasso, KNN) and when it is not (trees)
5. Bagging (Random Forest) compared with boosting (Gradient Boosting)

---

## Project Structure

```
regression-workflow/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── main.py                          # Runs the full regression workflow
│
├── config/
│   └── app_config.py                # Settings: random_state, test_size, cv_folds, output_dir
│
├── utils/
│   ├── data_validator.py            # Checks for NaN, inf, wrong shapes
│   ├── plot_manager.py              # Saves and/or shows plots
│   └── logger.py                    # Logging setup
│
├── data_loaders/
│   └── dataset_manager.py           # Loads the Diabetes dataset
│
├── workflows/
│   └── regression_workflow.py       # The 7 models, metrics, table and plots
│
├── examples/
│   └── run_regression.py            # Step-by-step demo with interpretation
│
├── exercises/
│   └── exercise_01_regression.py    # Practice challenge (TODOs + questions)
│
├── outputs/
│   └── regression/                  # Generated table (CSV) and plots (PNG)
│
└── visuals/                         # Optional: your own diagrams
```

---

## Installation

You need Python 3.10 or newer and pip.

```bash
cd regression-workflow

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

Run every command **from the project folder** (`regression-workflow/`).

### Option 1: Run the full workflow

```bash
python main.py --save-plots --no-show
```

This trains all seven models, prints the comparison table, and saves everything to `outputs/regression/`.

| Flag | Meaning |
|------|---------|
| `--save-plots` | Save plots as PNG files |
| `--no-show` | Do not open plot windows |
| `--output-dir DIR` | Save to a different folder (default `outputs/regression`) |

### Option 2: Run the step-by-step example

```bash
python examples/run_regression.py
```

This does the same work but explains what each result means. It always saves to `outputs/regression/`.

### Option 3: Run individual modules

Each module can be run on its own:

```bash
python -m config.app_config
python -m utils.data_validator
python -m data_loaders.dataset_manager
python -m workflows.regression_workflow
```

---

## The Seven Models

| # | Model | Family | Scaled? | What it does |
|---|-------|--------|---------|--------------|
| 1 | **Linear Regression** | Linear | No | Fits a straight-line (hyperplane) relationship between the features and the target. Simple and easy to interpret. |
| 2 | **Ridge** | Linear + L2 penalty | Yes | Linear regression that shrinks the coefficients toward zero to reduce overfitting. Useful when features are correlated. |
| 3 | **Lasso** | Linear + L1 penalty | Yes | Like Ridge, but it can set some coefficients exactly to zero, so it also performs feature selection. |
| 4 | **Decision Tree** | Tree | No | Splits the data with yes/no questions ("is BMI > 0.02?") and predicts the average of each leaf. Easy to interpret but can overfit. |
| 5 | **Random Forest** | Tree ensemble (bagging) | No | Many trees, each trained on a random sample of the data, with their predictions averaged. Robust. |
| 6 | **Gradient Boosting** ⭐ new | Tree ensemble (boosting) | No | Trees are built **one after another**, and each new tree corrects the errors of the previous ones. Often the most accurate model on tabular data. |
| 7 | **K-Nearest Neighbors** ⭐ new | Instance-based | Yes | Learns no equation. To predict a patient, it finds the 10 most similar training patients and averages their values. |

### Why these two new models?

- **Gradient Boosting** pairs naturally with Random Forest. Both combine many trees, but bagging trains them independently while boosting trains them in sequence. This is one of the most important ideas in practical machine learning.
- **K-Nearest Neighbors** is a different kind of model from everything else here. It is distance-based, so it shows clearly why feature scaling matters (try removing the scaler in the exercise).

### Why are some models scaled?

`StandardScaler` is placed **inside** a pipeline for Ridge, Lasso and KNN:

- Ridge and Lasso penalise coefficient size, which only makes sense if the features are on the same scale.
- KNN measures distance, so a feature with a large range would otherwise dominate.
- Trees split on one feature at a time, so scale does not affect them.

Because the scaler is inside the pipeline, it is fitted only on the training data, and no information leaks from the test set.

---

## Evaluation Metrics

| Metric | Meaning | Better |
|--------|---------|--------|
| **MAE** | Mean absolute error: the average size of an error, in target units | Lower |
| **MSE** | Mean squared error: squares each error, so big misses count much more (units²) | Lower |
| **RMSE** | √MSE: back in target units, but still punishes big misses. Always ≥ MAE | Lower |
| **R²** | Share of the target's variance that the model explains. 1 = perfect, 0 = no better than predicting the mean | Higher |
| **CV R² (mean ± std)** | Average R² over 5 cross-validation folds of the training set. More reliable than a single test split | Higher |

---

## Results

Output of `python main.py` (random_state = 42):

| Model | MAE | RMSE | R² | CV R² (mean) |
|-------|-----|------|----|--------------|
| Gradient Boosting | 42.98 | 53.28 | 0.464 | 0.407 |
| Lasso | 42.81 | 53.71 | 0.456 | 0.451 |
| Ridge | 42.81 | 53.78 | 0.454 | 0.451 |
| Linear Regression | 42.79 | 53.85 | 0.453 | 0.449 |
| Random Forest | 44.05 | 54.33 | 0.443 | 0.391 |
| K-Nearest Neighbors | 44.25 | 55.58 | 0.417 | 0.384 |
| Decision Tree | 48.10 | 59.60 | 0.329 | 0.274 |

**Things to notice:**

- Gradient Boosting has the lowest RMSE on this test split, but the linear models have a **higher cross-validated R²**. On a small dataset (442 rows) a single test split can be lucky, so the top-four models are effectively tied. The simple linear models are a very strong baseline here.
- The single Decision Tree is clearly worst. In the actual-vs-predicted plot its predictions form horizontal bands, one for each leaf.
- An R² of about 0.45 means the 10 baseline measurements explain roughly half of the variation in disease progression. The rest depends on things not in the data.

### Generated files (`outputs/regression/`)

| File | Shows |
|------|-------|
| `regression_model_comparison.csv` | The full comparison table (open in Excel / Sheets) |
| `regression_model_comparison.png` | RMSE and R² bar charts for all models |
| `regression_all_models_actual_vs_predicted.png` | Actual vs predicted, one panel per model |
| `regression_actual_vs_predicted.png` | Actual vs predicted for Linear Regression |
| `regression_residuals_best_model.png` | Residuals of the best model (scatter + histogram) |
| `regression_feature_importance.png` | Feature importance: Random Forest vs Gradient Boosting |

---

## Module Guide

### `config/app_config.py`

A frozen dataclass holding all the settings. Invalid values (for example `test_size=1.5`) raise an error immediately.

```python
from config import AppConfig
config = AppConfig(random_state=42, test_size=0.2, cv_folds=5, save_plots=True, show_plots=False)
```

### `utils/`

- `DataValidator.validate_features(X)` / `validate_target(y)` catch NaN, inf, empty data and wrong shapes before training.
- `PlotManager(config).finish("name.png")` saves and/or shows the current figure, then closes it.
- `configure_logging()` sets up timestamped log messages.

### `data_loaders/dataset_manager.py`

```python
from data_loaders import DatasetManager
X, y = DatasetManager().load_regression_data()   # (442, 10), (442,)
```

### `workflows/regression_workflow.py`

```python
from workflows import RegressionWorkflow

workflow = RegressionWorkflow(config)
results, best_model = workflow.run(X, y)   # table sorted by RMSE, best fitted pipeline
best_model.predict(X.head())
```

**To add your own model**, add one line to the `self.models` dictionary in `__init__`. If the model is scale-sensitive, also add its class to `scaled_models`. It will then be trained, evaluated, plotted and included in the table automatically.

---

## Exercise

**File**: `exercises/exercise_01_regression.py` (about 30–45 minutes)

The scenario is predicting diabetes progression for a health analytics company. You build the workflow yourself by filling in the TODOs, then tune the hyperparameters (Ridge alpha, KNN `n_neighbors`, Gradient Boosting `learning_rate`) and answer the questions at the bottom of the file.

```bash
python exercises/exercise_01_regression.py
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'config'`**
Run commands from the project folder, or use `python -m workflows.regression_workflow` instead of running the file by its path.

**`ModuleNotFoundError: No module named 'sklearn'`**
Activate your virtual environment and run `pip install -r requirements.txt`.

**No plot files appear**
`main.py` only saves plots when you pass `--save-plots`. The example script always saves them.

**Plot windows block the program**
Add `--no-show`.

**Different numbers on each run**
Every model with randomness uses `random_state` from `AppConfig`. Keep it fixed (default `42`) to get identical results.
