"""
Centralized logging configuration for AutoDataLab.
Provides consistent logging across all modules with file and console output.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# Import settings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings


class LoggerConfig:
    """Configuration for application logging."""
    
    # Logging paths
    LOGS_DIR = Path(__file__).parent.parent / "logs"
    LOG_FILE = LOGS_DIR / "application.log"
    ERROR_LOG_FILE = LOGS_DIR / "errors.log"
    
    # Ensure logs directory exists
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Logging format
    DETAILED_FORMAT = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    SIMPLE_FORMAT = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Log levels
    LOG_LEVEL = getattr(logging, settings.LOG_LEVEL, logging.INFO)


def setup_logger(name: str, log_level: Optional[int] = None) -> logging.Logger:
    """
    Setup and configure logger with file and console handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (defaults to settings.LOG_LEVEL)
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Application started")
    """
    log_level = log_level or LoggerConfig.LOG_LEVEL
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(log_level)
    
    # File handler - All logs
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LoggerConfig.LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(LoggerConfig.DETAILED_FORMAT)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file handler: {e}")
    
    # File handler - Errors only
    try:
        error_file_handler = logging.handlers.RotatingFileHandler(
            LoggerConfig.ERROR_LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(LoggerConfig.DETAILED_FORMAT)
        logger.addHandler(error_file_handler)
    except Exception as e:
        print(f"Warning: Could not setup error file handler: {e}")
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(LoggerConfig.SIMPLE_FORMAT)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create logger for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.error("An error occurred")
    """
    return setup_logger(name)


def log_function_call(logger: logging.Logger, func_name: str, args: dict):
    """
    Log function call with arguments.
    
    Args:
        logger: Logger instance
        func_name: Function name
        args: Function arguments as dictionary
    """
    args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
    logger.debug(f"Calling {func_name}({args_str})")


def log_function_result(logger: logging.Logger, func_name: str, result_type: str):
    """
    Log function result.
    
    Args:
        logger: Logger instance
        func_name: Function name
        result_type: Type of result returned
    """
    logger.debug(f"{func_name} completed, returned {result_type}")


# Create application-level logger
app_logger = setup_logger("AutoDataLab")


if __name__ == "__main__":
    # Test logging configuration
    logger = get_logger(__name__)
    logger.info("Testing info level")
    logger.warning("Testing warning level")
    logger.error("Testing error level")
    logger.critical("Testing critical level")
    print(f"\nLog files created:")
    print(f"  - {LoggerConfig.LOG_FILE}")
    print(f"  - {LoggerConfig.ERROR_LOG_FILE}")
