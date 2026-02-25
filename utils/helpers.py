"""
Helper utilities for AutoDataLab.
Provides common utility functions used across modules.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def validate_dataframe(df: Any) -> bool:
    """
    Validate if input is a valid pandas DataFrame.
    
    Args:
        df: Object to validate
        
    Returns:
        bool: True if valid DataFrame, False otherwise
    """
    return isinstance(df, pd.DataFrame) and not df.empty


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of numerical columns from DataFrame.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: Column names with numerical data types
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of categorical columns from DataFrame.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: Column names with categorical data types
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of datetime columns from DataFrame.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: Column names with datetime data types
    """
    return df.select_dtypes(include=['datetime64']).columns.tolist()


def format_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Format comprehensive DataFrame information.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        dict: Formatted DataFrame information
    """
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'numeric_columns': len(get_numeric_columns(df)),
        'categorical_columns': len(get_categorical_columns(df)),
        'datetime_columns': len(get_datetime_columns(df))
    }


def remove_special_characters(text: str) -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Cleaned text
    """
    import re
    return re.sub(r'[^a-zA-Z0-9_]', '', text)


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with specified decimal places.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        str: Formatted number
    """
    return f"{value:.{decimals}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, avoiding zero division.
    
    Args:
        numerator: Dividend
        denominator: Divisor
        default: Default value if division by zero
        
    Returns:
        float: Division result or default value
    """
    return numerator / denominator if denominator != 0 else default


def get_column_summary(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Get comprehensive summary for a single column.
    
    Args:
        df: pandas DataFrame
        column: Column name
        
    Returns:
        dict: Column summary information
    """
    try:
        series = df[column]
        summary = {
            'dtype': str(series.dtype),
            'non_null': series.notna().sum(),
            'null': series.isnull().sum(),
            'unique': series.nunique(),
            'null_percentage': round((series.isnull().sum() / len(series)) * 100, 2)
        }
        
        if pd.api.types.is_numeric_dtype(series):
            summary.update({
                'mean': float(series.mean()) if series.notna().sum() > 0 else None,
                'median': float(series.median()) if series.notna().sum() > 0 else None,
                'std': float(series.std()) if series.notna().sum() > 0 else None,
                'min': float(series.min()) if series.notna().sum() > 0 else None,
                'max': float(series.max()) if series.notna().sum() > 0 else None,
            })
        
        return summary
    
    except Exception as e:
        logger.error(f"Error getting column summary: {e}")
        return {}


def convert_bytes_to_mb(bytes_value: int) -> float:
    """
    Convert bytes to megabytes.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        float: Size in megabytes
    """
    return bytes_value / (1024 * 1024)
