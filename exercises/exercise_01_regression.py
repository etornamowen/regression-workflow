"""
Exercise 1: Regression Challenge
=================================

Difficulty: Intermediate
Time: 30-45 minutes

SCENARIO:
You work for a health analytics company. Your task is to build a
regression model to predict diabetes disease progression based on
patient baseline measurements.

OBJECTIVES:
1. Load and explore the diabetes dataset
2. Compare different regression algorithms
3. Evaluate using MAE, RMSE and R² metrics
4. Understand which features are most important
5. Improve model performance

LEARNING GOALS:
- Practice complete regression workflow
- Understand regression metrics
- Compare linear vs non-linear models
- Compare bagging (Random Forest) vs boosting (Gradient Boosting)
- See why distance-based models (KNN) need feature scaling
- Interpret model predictions

Run with:
    python exercises/exercise_01_regression.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import AppConfig
from data_loaders import DatasetManager
from utils import configure_logging


def exercise_1():
    """
    TODO: Complete this function to solve the regression challenge.

    Steps to complete:
    1. Load the diabetes dataset
    2. Explore the data (shape, ranges, distributions)
    3. Split into train/test (80/20)
    4. Train multiple regression models:
       - Linear Regression
       - Ridge Regression (try different alpha values)
       - Lasso Regression
       - Decision Tree
       - Random Forest
       - Gradient Boosting
       - K-Nearest Neighbors (try different n_neighbors)
    5. Evaluate with multiple metrics (MAE, MSE, RMSE, R²)
    6. Analyze feature importance (for tree-based models)
    7. Try to improve the best model

    Hints:
    - Scale features for Ridge, Lasso and KNN:
      make_pipeline(StandardScaler(), KNeighborsRegressor())
    - RMSE = sqrt(MSE)
    - Lower MAE/RMSE is better; higher R² is better (max = 1.0)
    - Use feature_importances_ for Random Forest / Gradient Boosting
    - For Gradient Boosting, a smaller learning_rate usually needs
      more n_estimators
    """

    # Step 1: Setup
    configure_logging()
    dm = DatasetManager()

    # TODO: Load the regression dataset
    # X, y = ...

    print("="*70)
    print("EXERCISE 1: REGRESSION CHALLENGE")
    print("="*70)
    print()

    # TODO: Explore the data
    # print(f"Samples: ...")
    # print(f"Features: ...")
    # print(f"Feature names: ...")
    # print(f"Target range: {y.min():.1f} to {y.max():.1f}")
    # print(f"Target mean: {y.mean():.1f}")

    # Step 2: Split the data
    # TODO: Split with test_size=0.2, random_state=42
    # X_train, X_test, y_train, y_test = ...

    # Step 3: Train multiple models
    # TODO: Create and train different regression models
    # Compare their performance using MAE, RMSE and R²

    # Example structure:
    # models = {
    #     "Linear Regression": LinearRegression(),
    #     "Ridge (alpha=1.0)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    #     "KNN (k=10)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
    #     # ... add more models
    # }

    # for name, model in models.items():
    #     # Train
    #     # Predict
    #     # Calculate metrics
    #     # Print results

    # Step 4: Feature importance
    # TODO: Train a Random Forest and a Gradient Boosting model and
    # compare their feature_importances_. Do they agree?

    # Step 5: Hyperparameter tuning
    # TODO: Try different alpha values for Ridge/Lasso
    # Try different max_depth for Decision Tree
    # Try different n_neighbors for KNN (1, 5, 10, 20, 50)
    # Try different learning_rate / n_estimators for Gradient Boosting

    # Step 6: Visualize predictions
    # TODO: Create a scatter plot of actual vs predicted values
    # Points should be close to y=x line for good predictions

    print("\n" + "="*70)
    print("QUESTIONS TO ANSWER:")
    print("="*70)
    print("1. Which model performed best? Linear or non-linear?")
    print("2. What are the MAE and RMSE in interpretable units? Why is RMSE larger?")
    print("3. What does the R² score tell you?")
    print("4. Which features are most important for prediction?")
    print("5. What happens to KNN if you remove the StandardScaler?")
    print("6. How does Gradient Boosting differ from Random Forest?")
    print("="*70)


# QUESTIONS (write your answers here as comments):

# Q1: Which model performed best? Why?
# Answer:

# Q2: Interpret MAE and RMSE in context:
# Answer:

# Q3: Interpret R² score:
# Answer:

# Q4: Most important features:
# Answer:

# Q5: KNN with and without scaling:
# Answer:

# Q6: Bagging (Random Forest) vs boosting (Gradient Boosting):
# Answer:


if __name__ == "__main__":
    exercise_1()
