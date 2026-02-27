"""
Data validation module for AutoDataLab.
Provides comprehensive data validation and quality checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from utils.logger import get_logger
from utils.exceptions import DataValidationError

logger = get_logger(__name__)


class DataValidator:
    """Comprehensive data validation suite."""
    
    @staticmethod
    def validate_dataframe(df: Any) -> Tuple[bool, str]:
        """
        Validate if input is a valid non-empty DataFrame.
        
        Args:
            df: Object to validate
            
        Returns:
            tuple: (is_valid, message)
            
        Raises:
            DataValidationError: If validation fails
        """
        if df is None:
            raise DataValidationError("DataFrame is None")
        
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"Expected pandas DataFrame, got {type(df).__name__}"
            )
        
        if df.empty:
            raise DataValidationError("DataFrame is empty")
        
        logger.debug(f"DataFrame validation passed: shape {df.shape}")
        return True, "DataFrame valid"
    
    @staticmethod
    def validate_file_path(file_path: str) -> Tuple[bool, str]:
        """
        Validate file exists and is readable.
        
        Args:
            file_path: Path to file
            
        Returns:
            tuple: (is_valid, message)
            
        Raises:
            DataValidationError: If file is invalid
        """
        path = Path(file_path)
        
        if not path.exists():
            raise DataValidationError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise DataValidationError(f"Path is not a file: {file_path}")
        
        if not path.stat().st_size > 0:
            raise DataValidationError(f"File is empty: {file_path}")
        
        logger.debug(f"File validation passed: {file_path}")
        return True, "File valid"
    
    @staticmethod
    def validate_file_format(file_path: str, allowed_formats: List[str]) -> Tuple[bool, str]:
        """
        Validate file format/extension.
        
        Args:
            file_path: Path to file
            allowed_formats: List of allowed file extensions (e.g., ['csv', 'xlsx'])
            
        Returns:
            tuple: (is_valid, message)
            
        Raises:
            DataValidationError: If format is invalid
        """
        path = Path(file_path)
        file_extension = path.suffix.lower().lstrip('.')
        
        if file_extension not in allowed_formats:
            raise DataValidationError(
                f"Invalid file format: .{file_extension}. "
                f"Allowed formats: {', '.join(allowed_formats)}"
            )
        
        logger.debug(f"File format validation passed: .{file_extension}")
        return True, "File format valid"
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
        """
        Validate that DataFrame contains required columns.
        
        Args:
            df: pandas DataFrame
            required_columns: List of required column names
            
        Returns:
            tuple: (is_valid, message)
            
        Raises:
            DataValidationError: If required columns missing
        """
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise DataValidationError(
                f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        logger.debug(f"Column validation passed for {len(required_columns)} columns")
        return True, "Columns valid"
    
    @staticmethod
    def validate_target_column(df: pd.DataFrame, target_column: str) -> Tuple[bool, str]:
        """
        Validate target column exists and has valid values.
        
        Args:
            df: pandas DataFrame
            target_column: Name of target column
            
        Returns:
            tuple: (is_valid, message)
            
        Raises:
            DataValidationError: If target column invalid
        """
        if target_column not in df.columns:
            raise DataValidationError(f"Target column not found: {target_column}")
        
        if df[target_column].isna().all():
            raise DataValidationError(f"Target column is all NaN: {target_column}")
        
        logger.debug(f"Target column validation passed: {target_column}")
        return True, "Target column valid"
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> Dict[str, float]:
        """
        Check for missing values in DataFrame.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: Column names with missing value percentages
        """
        missing_pct = (df.isna().sum() / len(df) * 100).to_dict()
        missing_cols = {col: pct for col, pct in missing_pct.items() if pct > 0}
        
        if missing_cols:
            logger.warning(f"Missing values found: {missing_cols}")
        
        return missing_cols
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for duplicate rows in DataFrame.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: Duplicate statistics
        """
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df) * 100) if len(df) > 0 else 0
        
        result = {
            "duplicate_rows": int(duplicate_count),
            "duplicate_percentage": round(duplicate_pct, 2)
        }
        
        if duplicate_count > 0:
            logger.warning(f"Duplicates found: {duplicate_count} rows ({duplicate_pct:.2f}%)")
        
        return result
    
    @staticmethod
    def check_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        Check data types of all columns.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: Column names with their data types
        """
        dtypes = df.dtypes.astype(str).to_dict()
        logger.debug(f"Data types: {dtypes}")
        return dtypes
    
    @staticmethod
    def check_numeric_columns(df: pd.DataFrame) -> List[str]:
        """
        Identify numeric columns.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            list: Names of numeric columns
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.debug(f"Numeric columns: {numeric_cols}")
        return numeric_cols
    
    @staticmethod
    def check_categorical_columns(df: pd.DataFrame) -> List[str]:
        """
        Identify categorical columns.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            list: Names of categorical columns
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.debug(f"Categorical columns: {categorical_cols}")
        return categorical_cols
    
    @staticmethod
    def check_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Detect outliers using IQR method.
        
        Args:
            df: pandas DataFrame
            columns: Columns to check (if None, checks all numeric)
            
        Returns:
            dict: Outlier statistics by column
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = (outlier_count / len(df) * 100) if len(df) > 0 else 0
            
            outliers[col] = {
                "outlier_count": int(outlier_count),
                "outlier_percentage": round(outlier_pct, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2)
            }
        
        logger.debug(f"Outlier detection completed for {len(outliers)} columns")
        return outliers
    
    @staticmethod
    def get_comprehensive_report(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: pandas DataFrame
            target_column: Optional target column name
            
        Returns:
            dict: Comprehensive quality report
        """
        logger.info("Generating comprehensive data quality report")
        
        report = {
            "shape": df.shape,
            "shape_description": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "columns": list(df.columns),
            "dtypes": DataValidator.check_data_types(df),
            "missing_values": DataValidator.check_missing_values(df),
            "duplicates": DataValidator.check_duplicates(df),
            "numeric_columns": DataValidator.check_numeric_columns(df),
            "categorical_columns": DataValidator.check_categorical_columns(df),
            "outliers": DataValidator.check_outliers(df),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        if target_column:
            try:
                DataValidator.validate_target_column(df, target_column)
                report["target_column"] = target_column
                report["target_dtype"] = str(df[target_column].dtype)
                report["target_unique_values"] = int(df[target_column].nunique())
            except DataValidationError as e:
                logger.error(f"Target column validation error: {e}")
                report["target_column_error"] = str(e)
        
        return report


# Convenience functions
def validate_dataframe(df: Any) -> bool:
    """Quick DataFrame validation."""
    try:
        DataValidator.validate_dataframe(df)
        return True
    except DataValidationError:
        return False


def get_missing_values(df: pd.DataFrame) -> Dict[str, float]:
    """Get missing values summary."""
    return DataValidator.check_missing_values(df)


def get_data_quality_report(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive data quality report."""
    return DataValidator.get_comprehensive_report(df, target_column)


if __name__ == "__main__":
    # Example usage
    logger.info("Testing data validator")
    
    # Create sample DataFrame
    df_sample = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['a', 'b', 'c', 'd', 'e']
    })
    
    try:
        DataValidator.validate_dataframe(df_sample)
        print("✓ DataFrame validation passed")
        
        report = DataValidator.get_comprehensive_report(df_sample)
        print("\nData Quality Report:")
        for key, value in report.items():
            print(f"  {key}: {value}")
    
    except DataValidationError as e:
        print(f"✗ Validation error: {e}")
