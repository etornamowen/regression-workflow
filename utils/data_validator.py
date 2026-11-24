"""
Data Validation Module
=======================

Purpose:
    Validates machine learning data before training to catch errors early.
    "Garbage in, garbage out" - if your data is bad, your model will be bad.

Key Concepts:
    - Static methods: Functions that don't need instance state (like utility functions)
    - Data quality checks: Missing values, infinite values, empty data, etc.
    - Defensive programming: Check assumptions before they cause problems

Common Data Issues:
    1. Missing values (NaN): Can break many ML algorithms
    2. Infinite values (inf): Result from division by zero or overflow
    3. Wrong dimensions: ML algorithms expect 2D features (rows × columns)
    4. Empty data: Can't train on nothing!
    5. Mismatched lengths: Features and targets must have same number of rows

Why Validate?
    Finding a data problem before training saves hours of debugging later.
    Clear error messages help you fix issues quickly.
"""

import numpy as np
import pandas as pd


class DataValidator:
    """
    Reusable validation for tabular machine learning data.
    
    This class contains static methods for validating features (X) and targets (y).
    Static methods are used because validation doesn't need any instance state.
    
    Usage Pattern:
        1. Load your data
        2. Call DataValidator.validate_features(X)
        3. Call DataValidator.validate_target(y)
        4. If no errors, proceed with training
    """
    
    @staticmethod
    def validate_features(X: pd.DataFrame | np.ndarray) -> None:
        """
        Validate feature matrix (X) for machine learning.
        
        Args:
            X: Feature matrix, either pandas DataFrame or numpy array
               Should be 2D with shape (n_samples, n_features)
        
        Raises:
            TypeError: If X is not a DataFrame or ndarray
            ValueError: If X has wrong dimensions, is empty, or contains invalid values
        
        Example:
            >>> import pandas as pd
            >>> X = pd.DataFrame({'age': [25, 30, 35], 'income': [50000, 60000, 70000]})
            >>> DataValidator.validate_features(X)  # No error = valid data
        """
        
        # Check 1: Must be the right type
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                f"Features must be a DataFrame or NumPy array, got {type(X)}. "
                "Convert your data to one of these types first."
            )
        
        # Check 2: Must be 2D (rows and columns)
        # Even if you have 1 feature, it should be shape (n, 1) not (n,)
        if X.ndim != 2:
            raise ValueError(
                f"Features must be two-dimensional (rows × columns), got {X.ndim}D. "
                f"Shape: {X.shape}. Hint: Use X.reshape(-1, 1) for a single feature."
            )
        
        # Check 3: Must have data
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(
                f"Features cannot be empty. Got shape {X.shape}. "
                "Check your data loading code."
            )
        
        # Check 4: No missing values (NaN)
        # Different check methods for pandas vs numpy
        has_missing = (
            pd.isna(X).any().any() 
            if isinstance(X, pd.DataFrame) 
            else np.isnan(X.astype(float)).any()
        )
        
        if has_missing:
            if isinstance(X, pd.DataFrame):
                missing_cols = X.columns[X.isna().any()].tolist()
                raise ValueError(
                    f"Features contain missing values in columns: {missing_cols}. "
                    "Options: (1) Remove rows with missing data, "
                    "(2) Fill with mean/median, (3) Use imputation."
                )
            else:
                raise ValueError(
                    "Features contain missing values (NaN). "
                    "Handle missing data before training."
                )
        
        # Check 5: No infinite values
        # Convert to float array to check for inf values
        try:
            X_array = np.asarray(X, dtype=float)
            if not np.isfinite(X_array).all():
                raise ValueError(
                    "Features contain infinite (inf) or non-finite values. "
                    "This usually happens from division by zero or data errors. "
                    "Check your data preprocessing steps."
                )
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not convert features to numeric format: {e}. "
                "Make sure all features are numeric."
            ) from e
    
    @staticmethod
    def validate_target(y: pd.Series | np.ndarray) -> None:
        """
        Validate target variable (y) for machine learning.
        
        Args:
            y: Target variable, either pandas Series or numpy array
               Should be 1D with shape (n_samples,)
        
        Raises:
            ValueError: If y is empty, contains missing values, or too small
        
        Example:
            >>> import pandas as pd
            >>> y = pd.Series([0, 1, 1, 0, 1])
            >>> DataValidator.validate_target(y)  # No error = valid data
        """
        
        # Check 1: Must have data
        if len(y) == 0:
            raise ValueError(
                "Target cannot be empty. Check your data loading code."
            )
        
        # Check 2: No missing values
        has_missing = pd.isna(y).any() if isinstance(y, pd.Series) else np.isnan(y).any()
        
        if has_missing:
            raise ValueError(
                "Target contains missing values (NaN). "
                "You cannot train a model without knowing the true labels. "
                "Options: (1) Remove rows with missing targets, "
                "(2) Use a different dataset."
            )
        
        # Check 3: Must have at least 2 observations
        # (Need at least 2 points to split into train/test)
        if len(y) < 2:
            raise ValueError(
                f"At least two observations are required, got {len(y)}. "
                "Cannot split data with only one sample."
            )


# Example usage (for testing this module independently)
if __name__ == "__main__":
    print("=== Data Validation Examples ===\n")
    
    # Example 1: Valid data
    print("1. Testing VALID data:")
    X_valid = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })
    y_valid = pd.Series([0, 1, 0, 1, 1])
    
    try:
        DataValidator.validate_features(X_valid)
        DataValidator.validate_target(y_valid)
        print("   ✓ Data is valid!\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    # Example 2: Missing values in features
    print("2. Testing data with MISSING VALUES:")
    X_missing = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })
    
    try:
        DataValidator.validate_features(X_missing)
    except ValueError as e:
        print(f"   ✓ Caught error: {e}\n")
    
    # Example 3: Wrong dimensions
    print("3. Testing data with WRONG DIMENSIONS:")
    X_1d = np.array([1, 2, 3, 4, 5])  # 1D instead of 2D
    
    try:
        DataValidator.validate_features(X_1d)
    except ValueError as e:
        print(f"   ✓ Caught error: {e}\n")
    
    # Example 4: Empty data
    print("4. Testing EMPTY data:")
    X_empty = pd.DataFrame()
    
    try:
        DataValidator.validate_features(X_empty)
    except ValueError as e:
        print(f"   ✓ Caught error: {e}\n")
    
    # Example 5: Target with missing values
    print("5. Testing target with MISSING VALUES:")
    y_missing = pd.Series([0, 1, np.nan, 1, 0])
    
    try:
        DataValidator.validate_target(y_missing)
    except ValueError as e:
        print(f"   ✓ Caught error: {e}\n")
    
    print("All validation tests completed!")
