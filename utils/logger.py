"""
Logging utilities for the Multi-Document RAG System.

Provides structured logging with consistent formatting across the application.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Creates a logger with consistent formatting including timestamps,
    log levels, and module names for easy debugging and monitoring.
    
    Args:
        name: The name of the logger (typically __name__).
        level: Optional logging level. Defaults to INFO if not specified.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Set log level
        logger.setLevel(level or logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level or logging.INFO)
        
        # Create formatter with structured output
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger
