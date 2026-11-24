"""
Standalone Regression Example
==============================

This script demonstrates a complete regression workflow.
Run this to see regression in action!

Usage:
    python examples/run_regression.py

What This Script Does:
    1. Loads diabetes dataset
    2. Compares 7 regression algorithms
    3. Evaluates with MAE, MSE, RMSE, R² and cross-validated R²
    4. Saves a comparison table (CSV) and plots to outputs/regression/

Learning Objectives:
    - Understand the regression workflow
    - Compare linear, tree-based, and instance-based algorithms
    - Interpret MAE, RMSE and R²
    - Evaluate prediction quality with residuals
"""

import sys
from pathlib import Path

# Add parent directory to path so the project packages can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig
from data_loaders import DatasetManager
from utils import configure_logging
from workflows import RegressionWorkflow


def main():
    """Run complete regression workflow."""
    configure_logging()

    config = AppConfig(
        random_state=42,
        test_size=0.20,
        cv_folds=5,
        output_dir=PROJECT_ROOT / "outputs" / "regression",
        save_plots=True,
        show_plots=False,
    )

    print("="*70)
    print("REGRESSION WORKFLOW DEMO")
    print("="*70)
    print()

    # Load data
    print("Step 1: Loading Data...")
    dm = DatasetManager()
    X, y = dm.load_regression_data()
    print(f"  ✓ Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"  ✓ Target range: {y.min():.1f} to {y.max():.1f}")
    print()

    # Run workflow
    print("Step 2: Training and Comparing Models...")
    workflow = RegressionWorkflow(config)
    results, best_model = workflow.run(X, y)
    print()

    # Display results
    print("="*70)
    print("MODEL COMPARISON (sorted by test RMSE)")
    print("="*70)
    print(results.round(3).to_string(index=False))
    print()

    best = results.iloc[0]
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"Best Model: {best['Model']}")
    print()
    print(f"MAE: {best['MAE']:.3f}")
    print(f"  → On average, predictions are off by {best['MAE']:.1f} units")
    print()
    print(f"RMSE: {best['RMSE']:.3f}")
    print(f"  → Typical error of {best['RMSE']:.1f} units (big misses count more)")
    print(f"  → Lower is better")
    print()
    print(f"R² Score: {best['R²']:.3f}")
    print(f"  → Model explains {best['R²']*100:.1f}% of variance in the data")
    print(f"  → 1.0 = perfect, 0.0 = no better than mean")
    print()
    print(f"CV R²: {best['CV R² (mean)']:.3f} ± {best['CV R² (std)']:.3f}")
    print(f"  → Cross-validated estimate; more reliable than one test split")
    print()
    print(f"Table and plots saved to: {config.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
