"""
Dataset Manager Module
======================

Purpose:
    Loads and validates the dataset used by the regression workflow.

Key Concepts:
    - Scikit-learn datasets: Built-in datasets for learning and testing
    - Data frames: Pandas DataFrame (like Excel tables) for features
    - Series: Pandas Series (single column) for targets
    - Data validation: Always check data before returning it

Dataset Used:
    Diabetes (Regression): Predict disease progression one year after
    baseline from 10 patient measurements.

Why Use a Built-in Dataset?
    - No download needed
    - Well-documented
    - Standard benchmark
    - Perfect for learning

    Later, you'll use this same pattern for your own datasets!
"""

import logging

import pandas as pd
from sklearn.datasets import load_diabetes

from utils.data_validator import DataValidator


class DatasetManager:
    """
    Loads and summarizes the regression dataset.

    Methods:
        - load_regression_data(): Continuous target prediction (Diabetes)
    """

    def __init__(self) -> None:
        """Initialize the dataset manager with logging."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_regression_data(
        self,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load the Diabetes dataset for regression.

        Dataset Details:
            - Samples: 442 diabetes patients
            - Features: 10 baseline variables
            - Target: Continuous (quantitative measure of disease progression)
            - Source: Efron et al. (2004)

        Features Include:
            - age, sex, bmi (body mass index)
            - bp (average blood pressure)
            - s1-s6: six blood serum measurements

        Note on Preprocessing:
            The features in this dataset are already mean-centered and scaled
            by scikit-learn. We still apply StandardScaler inside the model
            pipelines for scale-sensitive models (Ridge, Lasso, KNN), because
            that is what you must do with real, unscaled data.

        Returns:
            X (DataFrame): Features (442 rows × 10 columns)
            y (Series): Target values (442 continuous values)

        Example:
            >>> dm = DatasetManager()
            >>> X, y = dm.load_regression_data()
            >>> print(f"Shape: {X.shape}")
            Shape: (442, 10)
            >>> print(f"Target range: {y.min():.1f} to {y.max():.1f}")
            Target range: 25.0 to 346.0

        Note:
            This is a regression task because the target is continuous,
            not discrete classes. We're predicting a number, not a category.

        Raises:
            ValueError: If data fails validation checks
        """
        # Load with both X and y in one call
        X, y = load_diabetes(return_X_y=True, as_frame=True)

        # Validate data
        DataValidator.validate_features(X)
        DataValidator.validate_target(y)

        # Log information
        self.logger.info(
            "Regression dataset loaded: %s observations, %s features",
            X.shape[0],
            X.shape[1],
        )

        return X, y


# Example usage (for testing this module independently)
if __name__ == "__main__":
    from utils.logger import configure_logging

    configure_logging()

    print("=== Dataset Manager Example ===\n")
    dm = DatasetManager()

    print("Loading Regression Dataset (Diabetes):")
    X_reg, y_reg = dm.load_regression_data()
    print(f"   Features shape: {X_reg.shape}")
    print(f"   Target shape: {y_reg.shape}")
    print(f"   Target range: {y_reg.min():.1f} to {y_reg.max():.1f}")
    print(f"   Feature names: {list(X_reg.columns)}\n")

    print("Dataset loaded successfully!")
