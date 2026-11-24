"""
Main Orchestrator - Run the Regression Workflow
================================================

This script runs the complete regression workflow:
1. Load and validate the Diabetes dataset
2. Train 7 regression models (Linear, Ridge, Lasso, Decision Tree,
   Random Forest, Gradient Boosting, K-Nearest Neighbors)
3. Evaluate with MAE, MSE, RMSE, R² and cross-validated R²
4. Save a comparison table and plots

Usage:
    python main.py

    # Save plots without displaying them:
    python main.py --save-plots --no-show

    # Custom output directory:
    python main.py --output-dir my_results --save-plots
"""

import argparse
import logging
from pathlib import Path

from config import AppConfig
from data_loaders import DatasetManager
from utils import configure_logging
from workflows import RegressionWorkflow


class MachineLearningApplication:
    """
    Orchestrates the regression workflow.

    Keeps setup and error handling in one place so the workflow
    module can focus on the machine learning.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the application.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.datasets = DatasetManager()
        self.regression = RegressionWorkflow(config)

    def run_all(self) -> None:
        """Run the regression workflow and report the results."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        print("="*70)
        print("REGRESSION WORKFLOW ORCHESTRATOR")
        print("="*70)

        try:
            self._run_regression()
            print("\n✓ Regression workflow completed successfully!")
        except Exception as e:
            self.logger.exception(f"Regression workflow failed: {e}")
            print(f"\n✗ Regression workflow failed: {e}")
            raise SystemExit(1)

        print("\n" + "="*70)
        print("REGRESSION WORKFLOW COMPLETED")
        print("="*70)
        print(f"Results saved to: {self.config.output_dir}")
        if not self.config.save_plots:
            print("(Plots were not saved. Add --save-plots to save them.)")

    def _run_regression(self) -> None:
        """Run regression workflow."""
        X, y = self.datasets.load_regression_data()
        results, model = self.regression.run(X, y)

        print("\nModel Comparison (sorted by test RMSE):")
        print(results.round(3).to_string(index=False))

        print(f"\nBest Model: {results.iloc[0]['Model']}")
        print(f"  MAE:  {results.iloc[0]['MAE']:.3f}")
        print(f"  RMSE: {results.iloc[0]['RMSE']:.3f}")
        print(f"  R²:   {results.iloc[0]['R²']:.3f}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the regression workflow demonstration."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/regression"),
        help="Directory for generated artifacts (default: outputs/regression/)",
    )

    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save generated figures to output directory",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures interactively",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    configure_logging()
    args = parse_arguments()

    config = AppConfig(
        random_state=42,
        test_size=0.20,
        cv_folds=5,
        output_dir=args.output_dir,
        save_plots=args.save_plots,
        show_plots=not args.no_show,
    )

    app = MachineLearningApplication(config)
    app.run_all()


if __name__ == "__main__":
    main()
