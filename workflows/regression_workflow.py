"""
Regression Workflow Module
===========================

Purpose:
    Complete regression pipeline for predicting continuous values.
    Regression = predicting numbers (prices, temperatures, disease progression, etc.)

Key Differences from Classification:
    - Target is continuous (not discrete classes)
    - Different metrics (MAE, RMSE, R², not accuracy)
    - Different models (though some overlap)
    - Visualization shows predictions vs actual values

Common Regression Tasks:
    - House price prediction
    - Stock market forecasting
    - Disease progression estimation
    - Temperature prediction
    - Sales forecasting

Models Compared (7):
    Linear models:   Linear Regression, Ridge, Lasso
    Tree models:     Decision Tree, Random Forest (bagging), Gradient Boosting (boosting)
    Instance-based:  K-Nearest Neighbors

Evaluation Metrics Explained:
    - MAE:  Mean Absolute Error. Average size of the errors, in the target's
            units. Easy to explain: "on average we are off by MAE units".
    - MSE:  Mean Squared Error. Squares each error first, so large errors
            are punished much more than small ones. Units are squared.
    - RMSE: Square root of MSE. Back in the target's units, but still
            sensitive to large errors. RMSE >= MAE always.
      Example: RMSE=5000 for house prices means typical error of about $5,000

    - R²: How well model explains the data
      R²=1.0: Perfect predictions
      R²=0.5: Explains 50% of variance
      R²=0.0: No better than predicting the mean
      R²<0.0: Worse than predicting the mean!

    - CV R²: Average R² over k-fold cross-validation on the training set.
      A single test split can be lucky or unlucky; cross-validation gives a
      more reliable estimate of how the model generalizes.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config.app_config import AppConfig
from utils.data_validator import DataValidator
from utils.plot_manager import PlotManager


class RegressionWorkflow:
    """
    Complete regression workflow: train, evaluate, and compare regression models.

    This class implements a regression pipeline that:
    - Validates data quality
    - Compares multiple regression algorithms
    - Evaluates with appropriate metrics (MAE, MSE, RMSE, R², CV R²)
    - Saves a comparison table
    - Visualizes predictions, residuals, and model comparison

    Attributes:
        config (AppConfig): Application configuration
        logger (Logger): For tracking progress
        plots (PlotManager): For handling visualizations
        models (dict): Collection of regression algorithms to try
        scaled_models (tuple): Model types that need feature scaling
    """

    # Models whose results depend on feature scale.
    # - Ridge/Lasso: the penalty treats all coefficients equally, so a feature
    #   measured in large units would be penalized differently than a small one.
    # - KNN: uses distances; a feature with a large range would dominate.
    # Tree models split on one feature at a time, so scaling does not matter.
    scaled_models = (Ridge, Lasso, KNeighborsRegressor)

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the regression workflow.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.plots = PlotManager(config)

        # Collection of regression models to compare
        self.models: dict[str, Any] = {
            # Basic linear regression: Simple, interpretable, fast
            # Assumption: Linear relationship between features and target
            "Linear Regression": LinearRegression(),

            # Ridge: Linear regression + L2 regularization
            # Shrinks coefficients to prevent overfitting
            # Good when features are correlated
            "Ridge": Ridge(alpha=1.0),  # alpha controls strength of regularization

            # Lasso: Linear regression + L1 regularization
            # Can set coefficients to exactly zero (feature selection!)
            # Good for high-dimensional data with many irrelevant features
            "Lasso": Lasso(
                alpha=0.1,      # Smaller alpha = less regularization
                max_iter=10000  # May need many iterations to converge
            ),

            # Decision Tree: Non-linear, interpretable, prone to overfitting
            # Splits data based on feature thresholds
            "Decision Tree": DecisionTreeRegressor(
                max_depth=3,  # Limit depth to prevent overfitting
                random_state=config.random_state,
            ),

            # Random Forest: Ensemble of decision trees (BAGGING)
            # Trees are trained independently on random samples, then averaged
            # Robust, handles non-linearity, rarely overfits
            "Random Forest": RandomForestRegressor(
                random_state=config.random_state,
            ),

            # ----- NEW MODEL 1 -----
            # Gradient Boosting: Ensemble of decision trees (BOOSTING)
            # Trees are trained one after another; each new tree tries to
            # correct the errors (residuals) left by the trees before it.
            # Compare with Random Forest to see bagging vs boosting.
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200,    # Number of trees added in sequence
                learning_rate=0.05,  # How much each tree contributes (smaller = more careful)
                max_depth=2,         # Shallow trees ("weak learners")
                subsample=0.8,       # Each tree sees 80% of rows (adds randomness, reduces overfitting)
                random_state=config.random_state,
            ),

            # ----- NEW MODEL 2 -----
            # K-Nearest Neighbors: Instance-based ("lazy") learning
            # No equation is learned. To predict, find the K most similar
            # training patients and average their target values.
            # Needs feature scaling because it is distance-based.
            "K-Nearest Neighbors": KNeighborsRegressor(
                n_neighbors=10,      # Average over the 10 closest patients
                weights="distance",  # Closer neighbors count more
            ),
        }

    def build_pipeline(self, estimator: Any) -> Pipeline:
        """
        Wrap a model in a pipeline, adding StandardScaler when needed.

        Putting the scaler inside the pipeline means it is fitted only on the
        training data (no information leaks from the test set), including
        inside each cross-validation fold.

        Args:
            estimator: An unfitted scikit-learn regressor

        Returns:
            A scikit-learn Pipeline ready to fit
        """
        model = clone(estimator)  # Fresh, unfitted copy
        if isinstance(model, self.scaled_models):
            return make_pipeline(StandardScaler(), model)
        return make_pipeline(model)

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, Pipeline]:
        """
        Run complete regression workflow: train, evaluate, and compare models.

        Workflow Steps:
            1. Validate data
            2. Split into train/test
            3. For each model:
               a. Cross-validate on the training set (CV R²)
               b. Train on the full training set
               c. Predict on test set
               d. Calculate MAE, MSE, RMSE, R²
            4. Compare results (table saved as CSV)
            5. Create visualizations

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - continuous values

        Returns:
            results_df: DataFrame comparing all models (sorted by RMSE)
            best_model: Trained pipeline of the best model (lowest test RMSE)

        Example Output:
            | Model              | MAE   | RMSE  | R²    |
            |--------------------|-------|-------|-------|
            | Gradient Boosting  | 42.10 | 52.80 | 0.474 |
            | Linear Regression  | 42.79 | 53.85 | 0.453 |
        """
        # Validate input data
        DataValidator.validate_features(X)
        DataValidator.validate_target(y)

        # Ensure matching lengths
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same number of rows. "
                f"Got X: {len(X)}, y: {len(y)}"
            )

        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            # Note: No stratify for regression (that's only for classification)
        )

        self.logger.info(
            f"Data split: {len(X_train)} training, {len(X_test)} testing samples"
        )

        results = []
        fitted_models: dict[str, Pipeline] = {}
        predictions: dict[str, np.ndarray] = {}

        # Train and evaluate each model
        for name, estimator in self.models.items():
            pipeline = self.build_pipeline(estimator)

            # Cross-validation on the TRAINING set only.
            # The test set stays untouched until the final evaluation.
            cv_r2 = cross_val_score(
                pipeline, X_train, y_train, cv=self.config.cv_folds, scoring="r2"
            )

            # Train the model
            pipeline.fit(X_train, y_train)

            # Make predictions on test set
            y_pred = pipeline.predict(X_test)

            # Calculate evaluation metrics
            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred))

            results.append({
                "Model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R²": r2,
                "CV R² (mean)": float(cv_r2.mean()),
                "CV R² (std)": float(cv_r2.std()),
            })

            # Keep trained model and predictions for plots
            fitted_models[name] = pipeline
            predictions[name] = y_pred

            self.logger.info(
                f"{name:20s} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | "
                f"R²: {r2:6.3f} | CV R²: {cv_r2.mean():.3f}"
            )

        # Create results DataFrame sorted by RMSE (lower is better)
        results_df = (
            pd.DataFrame(results)
            .sort_values("RMSE")
            .reset_index(drop=True)
        )
        best_name = results_df.iloc[0]["Model"]
        self.logger.info(f"Best model by test RMSE: {best_name}")

        # Save the comparison table so it can be opened in Excel/Sheets
        self._save_table(results_df)

        # Visualizations
        # 1) Original plot: Linear Regression (most interpretable for teaching)
        self._plot_predictions(
            fitted_models["Linear Regression"],
            X_test,
            y_test,
        )
        # 2) Bar charts comparing all models
        self._plot_model_comparison(results_df)
        # 3) Actual vs predicted for every model, side by side
        self._plot_all_predictions(predictions, y_test, results_df)
        # 4) Residuals of the best model
        self._plot_residuals(best_name, predictions[best_name], y_test)
        # 5) Which features the tree ensembles rely on
        self._plot_feature_importance(fitted_models, list(X.columns))

        return results_df, fitted_models[best_name]

    def _save_table(self, results_df: pd.DataFrame) -> Path:
        """Save the model comparison table as CSV in the output directory."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / "regression_model_comparison.csv"
        results_df.round(4).to_csv(path, index=False)
        self.logger.info(f"Comparison table saved to: {path}")
        return path

    def _plot_predictions(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """
        Create actual vs predicted scatter plot for Linear Regression.

        This visualization shows how well the model predicts:
        - Points on red line = perfect predictions
        - Points above line = overprediction (predicted too high)
        - Points below line = underprediction (predicted too low)
        - Spread from line = prediction error

        Interpretation:
            - Tight cluster around line = good predictions
            - Scattered points = poor predictions
            - Systematic deviation = model bias
        """
        predictions = model.predict(X_test)

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(
            y_test,
            predictions,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
            label='Predictions'
        )

        # Add perfect prediction line (y=x)
        min_val = min(float(y_test.min()), float(predictions.min()))
        max_val = max(float(y_test.max()), float(predictions.max()))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--',
            linewidth=2,
            label='Perfect Prediction (y=x)'
        )

        # Add metrics to plot
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(
            0.05, 0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=props
        )

        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title("Linear Regression: Actual vs Predicted", fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self.plots.finish("regression_actual_vs_predicted.png")

    def _plot_model_comparison(self, results_df: pd.DataFrame) -> None:
        """
        Horizontal bar charts of RMSE (lower is better) and R² (higher is better).

        Seeing the metrics side by side makes it easy to spot that a model
        with low RMSE also has high R² (they measure related things).
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        names = results_df["Model"]
        best = names.iloc[0]
        colors = ["tab:green" if n == best else "tab:blue" for n in names]

        axes[0].barh(names, results_df["RMSE"], color=colors)
        axes[0].set_title("Test RMSE (lower is better)", fontweight="bold")
        axes[0].set_xlabel("RMSE")
        axes[0].invert_yaxis()  # Best model at the top

        axes[1].barh(names, results_df["R²"], color=colors)
        axes[1].set_title("Test R² (higher is better)", fontweight="bold")
        axes[1].set_xlabel("R²")

        for ax, col in zip(axes, ["RMSE", "R²"]):
            for i, value in enumerate(results_df[col]):
                ax.text(value, i, f" {value:.2f}" if col == "RMSE" else f" {value:.3f}",
                        va="center", fontsize=9)
            ax.grid(True, axis="x", alpha=0.3)

        fig.suptitle("Regression Model Comparison (best in green)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        self.plots.finish("regression_model_comparison.png")

    def _plot_all_predictions(
        self,
        predictions: dict[str, np.ndarray],
        y_test: pd.Series,
        results_df: pd.DataFrame,
    ) -> None:
        """
        Grid of actual-vs-predicted plots, one per model, in ranking order.

        Notice how the single Decision Tree only predicts a few distinct
        values (one per leaf), producing horizontal "bands" of points.
        """
        names = list(results_df["Model"])
        n_cols = 4
        n_rows = int(np.ceil(len(names) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        lo, hi = float(y_test.min()), float(y_test.max())
        for ax, name in zip(axes, names):
            y_pred = predictions[name]
            r2 = results_df.loc[results_df["Model"] == name, "R²"].iloc[0]
            ax.scatter(y_test, y_pred, s=18, alpha=0.6, edgecolors="black", linewidths=0.3)
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
            ax.set_title(f"{name}\nR² = {r2:.3f}", fontsize=10)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.3)

        # Hide unused panels
        for ax in axes[len(names):]:
            ax.axis("off")

        fig.suptitle("Actual vs Predicted: All Models", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self.plots.finish("regression_all_models_actual_vs_predicted.png")

    def _plot_residuals(
        self,
        name: str,
        y_pred: np.ndarray,
        y_test: pd.Series,
    ) -> None:
        """
        Residual plot (residual = actual - predicted) for the best model.

        What to look for:
            - Points randomly scattered around 0  → good
            - A curve or funnel shape            → model is missing a pattern
            - Histogram roughly bell-shaped      → errors are well-behaved
        """
        residuals = np.asarray(y_test) - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors="black", linewidths=0.4)
        axes[0].axhline(0, color="red", linestyle="--", linewidth=2)
        axes[0].set_xlabel("Predicted Values")
        axes[0].set_ylabel("Residual (Actual - Predicted)")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(residuals, bins=20, edgecolor="black", alpha=0.8)
        axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Residuals")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Residual Analysis: {name} (best model)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        self.plots.finish("regression_residuals_best_model.png")

    def _plot_feature_importance(
        self,
        fitted_models: dict[str, Pipeline],
        feature_names: list[str],
    ) -> None:
        """
        Compare feature importances of Random Forest and Gradient Boosting.

        Both models report how much each feature reduced prediction error
        across all their splits. If both agree on the top features, we can
        be more confident those features really matter.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, name in zip(axes, ["Random Forest", "Gradient Boosting"]):
            model = fitted_models[name][-1]  # Last step of the pipeline
            importances = pd.Series(model.feature_importances_, index=feature_names)
            importances.sort_values().plot.barh(ax=ax, color="tab:blue")
            ax.set_title(name, fontweight="bold")
            ax.set_xlabel("Importance")
            ax.grid(True, axis="x", alpha=0.3)

        fig.suptitle("Feature Importance: Bagging vs Boosting",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        self.plots.finish("regression_feature_importance.png")


# Example usage (for testing this module independently)
if __name__ == "__main__":
    from utils.logger import configure_logging
    from data_loaders.dataset_manager import DatasetManager

    # Set up
    configure_logging()
    config = AppConfig(
        output_dir=Path("outputs/regression"), save_plots=True, show_plots=False
    )

    # Load regression data
    print("=== Regression Workflow Demo ===\n")
    dm = DatasetManager()
    X, y = dm.load_regression_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Target range: {y.min():.1f} to {y.max():.1f}\n")

    # Run workflow
    print("Training and comparing models...")
    workflow = RegressionWorkflow(config)
    results, best_model = workflow.run(X, y)

    # Display results
    print("\n" + "="*60)
    print("REGRESSION MODEL COMPARISON")
    print("="*60)
    print(results.round(3).to_string(index=False))
    print()

    # Interpretation
    print("Interpretation:")
    print(f"  Best model: {results.iloc[0]['Model']}")
    print(f"  MAE:  {results.iloc[0]['MAE']:.3f} (average absolute error)")
    print(f"  RMSE: {results.iloc[0]['RMSE']:.3f} (typical error, punishes big misses)")
    print(f"  R²:   {results.iloc[0]['R²']:.3f} (proportion of variance explained)")

    print("\n✓ Regression workflow completed successfully!")
