"""
DataProfiler module for generating statistical profiles of datasets.
Provides comprehensive data profiling without visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from utils.logger import get_logger
from utils.exceptions import DataValidationError

logger = get_logger(__name__)


class DataProfiler:
    """
    Generate comprehensive statistical profiles of pandas DataFrames.
    
    Attributes:
        df (pd.DataFrame): The dataframe to profile
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataProfiler with a DataFrame.
        
        Args:
            df: pandas DataFrame to profile
            
        Raises:
            DataValidationError: If input is not a pandas DataFrame or empty
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(f"Expected pandas DataFrame, got {type(df)}")
        
        if df.empty:
            raise DataValidationError("Cannot profile an empty DataFrame")
        
        self.df = df
        logger.info(f"DataProfiler initialized with shape {df.shape}")
    
    def get_statistical_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistical summary of numerical columns.
        
        Returns:
            dict: Statistical summary including count, mean, std, min, max, quartiles
        """
        try:
            summary = self.df.describe().to_dict()
            
            # Convert numpy types to Python native types for JSON serialization
            summary_clean = {}
            for col, stats in summary.items():
                summary_clean[col] = {
                    key: float(value) if isinstance(value, (np.integer, np.floating)) else value
                    for key, value in stats.items()
                }
            
            logger.info("Statistical summary generated successfully")
            return summary_clean
        
        except Exception as e:
            logger.error(f"Error generating statistical summary: {e}")
            raise
    
    def get_missing_value_percentage(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate missing value count and percentage for each column.
        
        Returns:
            dict: Column names with their missing value count and percentage
        """
        try:
            missing_info = {}
            total_rows = len(self.df)
            
            for column in self.df.columns:
                missing_count = int(self.df[column].isnull().sum())
                percentage = (missing_count / total_rows) * 100
                missing_info[column] = {
                    'missing_count': missing_count,
                    'missing_percentage': round(percentage, 2)
                }
            
            logger.info("Missing value percentage calculated")
            return missing_info
        
        except Exception as e:
            logger.error(f"Error calculating missing values: {e}")
            raise
    
    def get_duplicate_count(self) -> Dict[str, Any]:
        """
        Get duplicate row information.
        
        Returns:
            dict: Total duplicate rows, fully duplicated rows, and duplicates by column
        """
        try:
            total_rows = len(self.df)
            fully_duplicated = self.df.duplicated().sum()
            
            # Duplicates per column
            duplicates_by_column = {}
            for column in self.df.columns:
                duplicates_by_column[column] = int(self.df[column].duplicated().sum())
            
            result = {
                'total_rows': total_rows,
                'fully_duplicated_rows': int(fully_duplicated),
                'duplicate_percentage': round((fully_duplicated / total_rows) * 100, 2),
                'duplicates_by_column': duplicates_by_column
            }
            
            logger.info(f"Duplicate count: {fully_duplicated} rows")
            return result
        
        except Exception as e:
            logger.error(f"Error calculating duplicates: {e}")
            raise
    
    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get correlation matrix for numerical columns.
        
        Returns:
            dict: Correlation matrix between all numerical columns
        """
        try:
            # Select only numerical columns
            numerical_df = self.df.select_dtypes(include=[np.number])
            
            if numerical_df.empty:
                logger.warning("No numerical columns found for correlation analysis")
                return {}
            
            correlation = numerical_df.corr()
            
            # Convert to nested dictionary and handle NaN values
            corr_dict = {}
            for col in correlation.columns:
                corr_dict[col] = {
                    key: float(value) if not np.isnan(value) else None
                    for key, value in correlation[col].to_dict().items()
                }
            
            logger.info(f"Correlation matrix generated for {len(corr_dict)} numerical columns")
            return corr_dict
        
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            raise
    
    def generate_profile(self) -> Dict[str, Any]:
        """
        Generate complete data profile combining all analyses.
        
        Returns:
            dict: Comprehensive profile containing all analyses
        """
        try:
            profile = {
                'shape': {
                    'rows': len(self.df),
                    'columns': len(self.df.columns)
                },
                'column_names': list(self.df.columns),
                'data_types': {
                    col: str(dtype) 
                    for col, dtype in zip(self.df.columns, self.df.dtypes)
                },
                'statistical_summary': self.get_statistical_summary(),
                'missing_values': self.get_missing_value_percentage(),
                'duplicates': self.get_duplicate_count(),
                'correlation_matrix': self.get_correlation_matrix()
            }
            
            logger.info("Complete profile generated successfully")
            return profile
        
        except Exception as e:
            logger.error(f"Error generating complete profile: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    import os
    
    try:
        # Create sample data
        sample_data = pd.DataFrame({
            'Age': [25, 30, 35, 40, 25, None],
            'Salary': [50000, 60000, 55000, 70000, 50000, 65000],
            'Experience': [1, 5, 10, 15, 1, 8],
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Alice', 'Eve']
        })
        
        # Create profiler and generate profile
        profiler = DataProfiler(sample_data)
        
        print("=== Statistical Summary ===")
        print(profiler.get_statistical_summary())
        
        print("\n=== Missing Values ===")
        print(profiler.get_missing_value_percentage())
        
        print("\n=== Duplicates ===")
        print(profiler.get_duplicate_count())
        
        print("\n=== Correlation Matrix ===")
        print(profiler.get_correlation_matrix())
        
        print("\n=== Complete Profile ===")
        profile = profiler.generate_profile()
        print(f"Profile keys: {list(profile.keys())}")
    
    except Exception as e:
        print(f"Error: {e}")
