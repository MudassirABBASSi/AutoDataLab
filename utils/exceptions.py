"""
Custom exception classes for AutoDataLab.
Provides structured error handling across all modules.
"""

from typing import Optional


class AutoDataLabException(Exception):
    """Base exception class for AutoDataLab."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            error_code: Optional error code for categorization
        """
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        return f"[{self.error_code}] {self.message}"


class DataValidationError(AutoDataLabException):
    """Raised when data validation fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "DATA_VALIDATION_ERROR")


class DataLoadingError(AutoDataLabException):
    """Raised when data loading fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "DATA_LOADING_ERROR")


class DataCleaningError(AutoDataLabException):
    """Raised when data cleaning fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "DATA_CLEANING_ERROR")


class FeatureEngineeringError(AutoDataLabException):
    """Raised when feature engineering fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "FEATURE_ENGINEERING_ERROR")


class FeatureSelectionError(AutoDataLabException):
    """Raised when feature selection fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "FEATURE_SELECTION_ERROR")


class ModelingError(AutoDataLabException):
    """Raised when model training/inference fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "MODELING_ERROR")


class EvaluationError(AutoDataLabException):
    """Raised when model evaluation fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "EVALUATION_ERROR")


class PipelineError(AutoDataLabException):
    """Raised when pipeline execution fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "PIPELINE_ERROR")


class ConfigurationError(AutoDataLabException):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str):
        super().__init__(message, "CONFIGURATION_ERROR")


class CachingError(AutoDataLabException):
    """Raised when caching operation fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "CACHING_ERROR")


class VisualizationError(AutoDataLabException):
    """Raised when visualization fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "VISUALIZATION_ERROR")


class ReportingError(AutoDataLabException):
    """Raised when report generation fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "REPORTING_ERROR")


# Utility function to handle exceptions
def handle_exception(exception: Exception, logger, show_traceback: bool = False) -> str:
    """
    Handle exception and return user-friendly message.
    
    Args:
        exception: Exception instance
        logger: Logger instance
        show_traceback: Whether to log full traceback
        
    Returns:
        str: User-friendly error message
        
    Example:
        >>> try:
        ...     do_something()
        ... except Exception as e:
        ...     user_msg = handle_exception(e, logger)
        ...     st.error(user_msg)
    """
    error_msg = str(exception)
    
    if show_traceback:
        logger.exception(error_msg)
    else:
        logger.error(error_msg)
    
    # Return appropriate user message
    if isinstance(exception, AutoDataLabException):
        return error_msg
    else:
        return f"An unexpected error occurred: {error_msg}"
