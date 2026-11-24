"""
Logging Configuration Module
=============================

Purpose:
    Sets up consistent logging across the entire application.
    Logging is essential for debugging and understanding what your code is doing.

Key Concepts:
    - Logging levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
    - Format string: Defines how log messages appear
    - Logger: Object that actually writes log messages

Why Logging Instead of Print?
    - Can control verbosity (show only errors, or everything)
    - Includes timestamps automatically
    - Can write to files or console
    - Can turn off in production
    - Easier to filter and search

Example Output:
    2024-11-08 23:30:15 | INFO | DatasetManager | Regression dataset loaded: 442 observations, 10 features
"""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure application-wide logging with a consistent format.
    
    Args:
        level: The logging level (default: logging.INFO)
               - logging.DEBUG: Show everything (very verbose)
               - logging.INFO: Show general information (recommended)
               - logging.WARNING: Show only warnings and errors
               - logging.ERROR: Show only errors
    
    Format Explanation:
        %(asctime)s     - Current timestamp (when the log was written)
        %(levelname)s   - Log level (INFO, WARNING, ERROR, etc.)
        %(name)s        - Name of the logger (usually the class name)
        %(message)s     - The actual log message
    
    Example:
        >>> configure_logging()
        >>> logger = logging.getLogger("MyClass")
        >>> logger.info("This is an info message")
        2024-11-08 23:30:15 | INFO | MyClass | This is an info message
    """
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True  # Override any existing configuration
    )
    
    # Log that logging has been configured
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")


# Example usage (for testing this module independently)
if __name__ == "__main__":
    # Configure logging
    configure_logging()
    
    # Create a logger for this module
    logger = logging.getLogger("LoggerDemo")
    
    # Demonstrate different log levels
    logger.debug("This is a DEBUG message (very detailed, for developers)")
    logger.info("This is an INFO message (general information)")
    logger.warning("This is a WARNING message (something unexpected happened)")
    logger.error("This is an ERROR message (something went wrong)")
    logger.critical("This is a CRITICAL message (very serious error)")
    
    print("\n--- Testing with Different Levels ---")
    
    # Reconfigure to show only warnings and above
    configure_logging(level=logging.WARNING)
    logger = logging.getLogger("WarningDemo")
    
    logger.info("You won't see this (INFO is below WARNING)")
    logger.warning("But you will see this WARNING")
    logger.error("And this ERROR")
