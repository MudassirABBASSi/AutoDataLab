"""
DataCleaner module for handling data quality issues.
Provides methods to handle missing values, duplicates, and outliers.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Literal
from utils.logger import get_logger
from utils.exceptions import DataValidationError, DataCleaningError

logger = get_logger(__name__)


class DataCleaner:
    """
    Clean pandas DataFrames by handling missing values, duplicates, and outliers.
    Does not modify the original DataFrame.
    
    Attributes:
        df (pd.DataFrame): The original dataframe (never modified)
    """
    
    MISSING_STRATEGIES = {'drop', 'mean', 'median', 'mode'}
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataCleaner with a DataFrame.
        
        Args:
            df: pandas DataFrame to clean
            
        Raises:
            DataValidationError: If input is not a pandas DataFrame or empty
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(f"Expected pandas DataFrame, got {type(df)}")
        
        if df.empty:
            raise DataValidationError("Cannot clean an empty DataFrame")
        
        self.df = df
        logger.info(f"DataCleaner initialized with shape {df.shape}")
    
    def handle_missing_values(
        self,
        strategy: Literal['drop', 'mean', 'median', 'mode'] = 'drop',
        subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values using specified strategy.
        
        Args:
            strategy: Method to handle missing values
                     - 'drop': Remove rows with missing values
                     - 'mean': Fill with mean (numerical columns only)
                     - 'median': Fill with median (numerical columns only)
                     - 'mode': Fill with mode (most frequent value)
            subset: List of columns to apply strategy. If None, applies to all.
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled
            
        Raises:
            DataValidationError: If strategy is invalid or subset columns don't exist
            DataCleaningError: If cleaning fails
        """
        try:
            if strategy not in self.MISSING_STRATEGIES:
                raise DataValidationError(
                    f"Invalid strategy '{strategy}'. "
                    f"Choose from: {', '.join(self.MISSING_STRATEGIES)}"
                )
            
            # Create a copy to avoid modifying original
            df_cleaned = self.df.copy()
            
            # If subset specified, validate columns exist
            if subset:
                missing_cols = set(subset) - set(df_cleaned.columns)
                if missing_cols:
                    raise DataValidationError(
                        f"Columns not found in DataFrame: {missing_cols}"
                    )
                columns_to_process = subset
            else:
                columns_to_process = df_cleaned.columns.tolist()
            
            if strategy == 'drop':
                df_cleaned = df_cleaned.dropna(subset=columns_to_process)
                logger.info(f"Dropped {len(self.df) - len(df_cleaned)} rows with missing values")
            
            elif strategy == 'mean':
                numerical_cols = df_cleaned[columns_to_process].select_dtypes(
                    include=[np.number]
                ).columns
                
                for col in numerical_cols:
                    if df_cleaned[col].isnull().sum() > 0:
                        mean_val = df_cleaned[col].mean()
                        df_cleaned[col].fillna(mean_val, inplace=True)
                        logger.info(f"Filled {col} with mean value: {mean_val:.2f}")
            
            elif strategy == 'median':
                numerical_cols = df_cleaned[columns_to_process].select_dtypes(
                    include=[np.number]
                ).columns
                
                for col in numerical_cols:
                    if df_cleaned[col].isnull().sum() > 0:
                        median_val = df_cleaned[col].median()
                        df_cleaned[col].fillna(median_val, inplace=True)
                        logger.info(f"Filled {col} with median value: {median_val:.2f}")
            
            elif strategy == 'mode':
                for col in columns_to_process:
                    if df_cleaned[col].isnull().sum() > 0:
                        mode_val = df_cleaned[col].mode()
                        if len(mode_val) > 0:
                            mode_val = mode_val[0]
                            df_cleaned[col].fillna(mode_val, inplace=True)
                            logger.info(f"Filled {col} with mode value: {mode_val}")
            
            return df_cleaned
        
        except DataValidationError:
            logger.error("Validation error handling missing values", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise DataCleaningError(f"Failed to handle missing values: {e}")
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.
        
        Args:
            subset: List of column names to consider for identifying duplicates.
                   If None, considers all columns.
        
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
            
        Raises:
            DataValidationError: If subset columns don't exist
            DataCleaningError: If duplicate removal fails
        """
        try:
            df_cleaned = self.df.copy()
            
            # Validate subset columns if provided
            if subset:
                missing_cols = set(subset) - set(df_cleaned.columns)
                if missing_cols:
                    raise DataValidationError(
                        f"Columns not found in DataFrame: {missing_cols}"
                    )
            
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates(subset=subset)
            removed_rows = initial_rows - len(df_cleaned)
            
            logger.info(f"Removed {removed_rows} duplicate rows")
            return df_cleaned
        
        except DataValidationError:
            logger.error("Validation error removing duplicates", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            raise DataCleaningError(f"Failed to remove duplicates: {e}")
    
    def remove_outliers_iqr(
        self,
        columns: Optional[List[str]] = None,
        multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers using Interquartile Range (IQR) method.
        
        Outliers are defined as values outside: [Q1 - multiplier*IQR, Q3 + multiplier*IQR]
        
        Args:
            columns: List of numerical columns to check for outliers.
                    If None, checks all numerical columns.
            multiplier: IQR multiplier for outlier threshold (default 1.5)
        
        Returns:
            pd.DataFrame: DataFrame with outliers removed
            
        Raises:
            DataValidationError: If specified columns don't exist or aren't numerical
            DataCleaningError: If outlier removal fails
        """
        try:
            df_cleaned = self.df.copy()
            
            # Get numerical columns
            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numerical_cols:
                logger.warning("No numerical columns found for outlier detection")
                return df_cleaned
            
            # If columns specified, validate and filter
            if columns:
                missing_cols = set(columns) - set(numerical_cols)
                if missing_cols:
                    raise DataValidationError(
                        f"Specified columns not found or not numerical: {missing_cols}"
                    )
                columns_to_check = columns
            else:
                columns_to_check = numerical_cols
            
            initial_rows = len(df_cleaned)
            
            # Calculate IQR boundaries for each column
            for col in columns_to_check:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                # Remove outliers
                df_cleaned = df_cleaned[
                    (df_cleaned[col] >= lower_bound) & 
                    (df_cleaned[col] <= upper_bound)
                ]
                
                logger.info(
                    f"Outlier bounds for '{col}': [{lower_bound:.2f}, {upper_bound:.2f}]"
                )
            
            removed_rows = initial_rows - len(df_cleaned)
            logger.info(f"Removed {removed_rows} rows containing outliers")
            return df_cleaned
        
        except DataValidationError:
            logger.error("Validation error removing outliers", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            raise DataCleaningError(f"Failed to remove outliers: {e}")
    
    def clean_pipeline(
        self,
        missing_strategy: Literal['drop', 'mean', 'median', 'mode'] = 'drop',
        remove_duplicates_flag: bool = True,
        remove_outliers_flag: bool = True,
        outlier_multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Execute a complete cleaning pipeline in sequence.
        
        Order: Missing values -> Duplicates -> Outliers
        
        Args:
            missing_strategy: Strategy for missing values
            remove_duplicates_flag: Whether to remove duplicates
            remove_outliers_flag: Whether to remove outliers
            outlier_multiplier: IQR multiplier for outlier detection
        
        Returns:
            pd.DataFrame: Fully cleaned DataFrame
        """
        try:
            logger.info("Executing cleaning pipeline...")
            
            # Step 1: Handle missing values
            df_cleaned = self.handle_missing_values(strategy=missing_strategy)
            
            # Step 2: Remove duplicates
            if remove_duplicates_flag:
                cleaner = DataCleaner(df_cleaned)
                df_cleaned = cleaner.remove_duplicates()
            
            # Step 3: Remove outliers
            if remove_outliers_flag:
                cleaner = DataCleaner(df_cleaned)
                df_cleaned = cleaner.remove_outliers_iqr(multiplier=outlier_multiplier)
            
            logger.info(
                f"Pipeline complete. Rows reduced from {len(self.df)} to {len(df_cleaned)}"
            )
            return df_cleaned
        
        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {e}")
            raise DataCleaningError(f"Failed in cleaning pipeline: {e}")


if __name__ == "__main__":
    # Example usage
    try:
        sample_data = pd.DataFrame({
            'Age': [25, 30, 35, 40, 25, 150, None],
            'Salary': [50000, 60000, 55000, 70000, 50000, 65000, 58000],
            'Experience': [1, 5, 10, 15, 1, 8, 3],
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Alice', 'Eve', 'Frank']
        })
        
        print("Original Data:")
        print(sample_data)
        print(f"\nShape: {sample_data.shape}")
        
        cleaner = DataCleaner(sample_data)
        
        print("\n=== Handle Missing Values (drop) ===")
        cleaned_missing = cleaner.handle_missing_values(strategy='drop')
        print(cleaned_missing)
        
        print("\n=== Remove Duplicates ===")
        cleaned_duplicates = cleaner.remove_duplicates()
        print(cleaned_duplicates)
        
        print("\n=== Remove Outliers (IQR) ===")
        cleaned_outliers = cleaner.remove_outliers_iqr()
        print(cleaned_outliers)
        
        print("\n=== Complete Pipeline ===")
        cleaned = cleaner.clean_pipeline(missing_strategy='drop')
        print(cleaned)
        print(f"Final shape: {cleaned.shape}")
    
    except Exception as e:
        print(f"Error: {e}")
