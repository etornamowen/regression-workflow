"""
Plot Management Module
======================

Purpose:
    Centralizes figure display and saving logic throughout the application.
    Handles the difference between interactive use (showing plots) and 
    automated use (saving plots to files).

Key Concepts:
    - Matplotlib backends: Different ways to render plots (screen, file, etc.)
    - Context manager pattern: Ensures resources are properly cleaned up
    - DRY principle: Don't Repeat Yourself - centralize common logic

Why This Matters:
    - In development: You want to see plots interactively
    - In production: You want to save plots to files
    - In testing: You don't want plots at all
    
    This class handles all three cases consistently.
"""

import logging
from pathlib import Path

import matplotlib
# Use non-interactive backend for automated environments (servers, CI/CD)
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config.app_config import AppConfig


class PlotManager:
    """
    Centralizes figure display and optional saving.
    
    This class decides whether to:
    1. Save plots to disk (for reports, papers, documentation)
    2. Display plots interactively (for exploration, debugging)
    3. Just close them (for automated testing)
    
    Usage Pattern:
        1. Create figures with matplotlib (plt.figure(), plt.plot(), etc.)
        2. Call plot_manager.finish("filename.png")
        3. PlotManager handles the rest
    
    Attributes:
        config (AppConfig): Application configuration
        logger (Logger): Logger for this class
    """
    
    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the plot manager.
        
        Args:
            config: Application configuration containing:
                   - save_plots: Whether to save figures to disk
                   - show_plots: Whether to display figures interactively
                   - output_dir: Directory where plots are saved
        
        Side Effects:
            If save_plots is True, creates the output directory
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory if we're saving plots
        if self.config.save_plots:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory created: {self.config.output_dir}")
    
    def finish(self, filename: str) -> None:
        """
        Save, display, or close the current matplotlib figure.
        
        This method is called after you've finished creating a plot.
        It handles saving and/or displaying based on the configuration.
        
        Args:
            filename: Name of the file to save (e.g., "confusion_matrix.png")
                     Only used if save_plots is True
        
        Workflow:
            1. If save_plots: Save figure to output_dir/filename
            2. If show_plots: Display figure interactively (blocks execution)
            3. Always close the figure to free memory
        
        Example:
            >>> plt.plot([1, 2, 3], [1, 4, 9])
            >>> plt.title("My Plot")
            >>> plot_manager.finish("my_plot.png")
        
        Technical Details:
            - dpi=150: Dots per inch (higher = better quality, larger file)
            - bbox_inches="tight": Remove extra whitespace around the plot
            - plt.close(): Free memory (important in loops creating many plots)
        """
        
        # Step 1: Save to file if requested
        if self.config.save_plots:
            # Construct full path: output_dir / filename
            output_path = self.config.output_dir / filename
            
            # Save with high quality and tight bounding box
            plt.savefig(
                output_path,
                dpi=150,              # Good quality for papers/reports
                bbox_inches="tight"   # Remove unnecessary whitespace
            )
            
            self.logger.info(f"Plot saved to: {output_path}")
        
        # Step 2: Display interactively if requested
        if self.config.show_plots:
            # This will block execution until the plot window is closed
            plt.show()
        
        # Step 3: Always close the figure to free memory
        # Important: Prevents memory leaks when creating many plots
        plt.close()


# Example usage (for testing this module independently)
if __name__ == "__main__":
    from utils.logger import configure_logging
    
    # Configure logging to see what PlotManager is doing
    configure_logging()
    
    print("=== Plot Manager Examples ===\n")
    
    # Example 1: Save plots only (no display)
    print("1. Creating config to SAVE plots (not display):")
    config_save = AppConfig(
        output_dir=Path("test_plots"),
        save_plots=True,
        show_plots=False
    )
    
    pm_save = PlotManager(config_save)
    
    # Create a simple plot
    plt.figure(figsize=(8, 6))
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'bo-', linewidth=2, markersize=8)
    plt.title("Example Plot: y = x²")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    
    # Save it
    pm_save.finish("example_plot_1.png")
    print("   ✓ Plot saved to test_plots/example_plot_1.png\n")
    
    # Example 2: Display only (no save) - commented out to avoid blocking
    # Uncomment to test interactively
    """
    print("2. Creating config to DISPLAY plots (not save):")
    config_display = AppConfig(
        save_plots=False,
        show_plots=True
    )
    
    pm_display = PlotManager(config_display)
    
    plt.figure(figsize=(8, 6))
    plt.scatter([1, 2, 3, 4, 5], [2, 4, 5, 4, 5], c='red', s=100, alpha=0.6)
    plt.title("Example Scatter Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    pm_display.finish("ignored_filename.png")  # filename ignored when not saving
    print("   ✓ Plot displayed (and now closed)\n")
    """
    
    # Example 3: Neither save nor display (for testing)
    print("2. Creating config to NEITHER save nor display:")
    config_silent = AppConfig(
        save_plots=False,
        show_plots=False
    )
    
    pm_silent = PlotManager(config_silent)
    
    plt.figure(figsize=(8, 6))
    plt.bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
    plt.title("Example Bar Chart")
    
    pm_silent.finish("ignored.png")
    print("   ✓ Plot created and closed (not saved or displayed)\n")
    
    print("All PlotManager tests completed!")
    print(f"\nCheck the 'test_plots' directory for saved plots.")
