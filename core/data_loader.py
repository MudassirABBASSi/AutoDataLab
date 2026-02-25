"""
DataLoader module for loading and validating data files.
Supports CSV and Excel formats.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Union
from utils.logger import get_logger
from utils.exceptions import DataLoadingError, DataValidationError

logger = get_logger(__name__)


class DataLoader:
    """
    Load and validate CSV and Excel files.
    
    Attributes:
        df (pd.DataFrame): The loaded dataframe
        file_path (Path): Path to the loaded file
    """
    
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
    
    def __init__(self):
        """Initialize DataLoader."""
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[Path] = None
    
    def _validate_file_type(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that the file has a supported extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Path: Validated Path object
            
        Raises:
            DataValidationError: If file extension is not supported
        """
        path = Path(file_path)
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise DataValidationError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        return path
    
    def _check_file_exists(self, file_path: Path) -> None:
        """
        Check if file exists.
        
        Args:
            file_path: Path to the file
            
        Raises:
            DataLoadingError: If file does not exist
        """
        if not file_path.exists():
            raise DataLoadingError(f"File not found: {file_path}")
    
    def load(self, file_path: Union[str, Path, any]) -> pd.DataFrame:
        """
        Load CSV or Excel file into a pandas DataFrame.
        
        Args:
            file_path: Path to CSV or Excel file, or file-like object (e.g., Streamlit UploadedFile)
            
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            DataValidationError: If file type is not supported
            DataLoadingError: If file does not exist or cannot be read
        """
        try:
            # Check if it's a file-like object (Streamlit UploadedFile, BytesIO, etc.)
            if hasattr(file_path, 'name') and hasattr(file_path, 'read'):
                # It's a file-like object
                file_name = file_path.name
                file_ext = Path(file_name).suffix.lower()
                
                if file_ext not in self.SUPPORTED_EXTENSIONS:
                    raise DataValidationError(
                        f"Unsupported file type '{file_ext}'. "
                        f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
                    )
                
                # Load based on file extension
                if file_ext == '.csv':
                    self.df = pd.read_csv(file_path)
                    logger.info(f"Successfully loaded CSV file: {file_name}")
                
                elif file_ext in {'.xlsx', '.xls'}:
                    self.df = pd.read_excel(file_path)
                    logger.info(f"Successfully loaded Excel file: {file_name}")
                
                self.file_path = Path(file_name)
                return self.df
            
            else:
                # It's a file path (string or Path object)
                path = self._validate_file_type(file_path)
                
                # Check file exists
                self._check_file_exists(path)
                
                # Load based on file extension
                if path.suffix.lower() == '.csv':
                    self.df = pd.read_csv(path)
                    logger.info(f"Successfully loaded CSV file: {path}")
                
                elif path.suffix.lower() in {'.xlsx', '.xls'}:
                    self.df = pd.read_excel(path)
                    logger.info(f"Successfully loaded Excel file: {path}")
                
                self.file_path = path
                return self.df
        
        except (DataLoadingError, DataValidationError):
            logger.error("Data loading or validation error", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading file: {e}")
            raise DataLoadingError(f"Failed to load file: {str(e)}")
    
    def get_shape(self) -> tuple:
        """
        Get the shape (rows, columns) of the dataframe.
        
        Returns:
            tuple: (rows, columns)
            
        Raises:
            DataValidationError: If no data is loaded
        """
        if self.df is None:
            raise DataValidationError("No data loaded. Call load() first.")
        
        return self.df.shape
    
    def get_missing_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary of missing values in the dataframe.
        
        Returns:
            dict: Column names with missing count and percentage
            
        Raises:
            DataValidationError: If no data is loaded
        """
        if self.df is None:
            raise DataValidationError("No data loaded. Call load() first.")
        
        missing_summary = {}
        
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            
            missing_summary[column] = {
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_percentage, 2)
            }
        
        return missing_summary
    
    def get_data_types(self) -> Dict[str, str]:
        """
        Get data types of all columns.
        
        Returns:
            dict: Column names with their data types
            
        Raises:
            DataValidationError: If no data is loaded
        """
        if self.df is None:
            raise DataValidationError("No data loaded. Call load() first.")
        
        return {column: str(dtype) for column, dtype in zip(
            self.df.columns, self.df.dtypes
        )}
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the loaded dataframe.
        
        Returns:
            pd.DataFrame: The loaded dataframe
            
        Raises:
            ValueError: If no data is loaded
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() first.")
        
        return self.df


# Example usage and testing
if __name__ == "__main__":
    loader = DataLoader()
    
    # Test with a sample CSV
    try:
        # Create a sample CSV for testing
        import os
        sample_data = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie', 'David'],
            'Age': [25, 30, None, 35],
            'Salary': [50000, 60000, 55000, None]
        })
        
        sample_file = 'sample_data.csv'
        sample_data.to_csv(sample_file, index=False)
        
        # Load the file
        df = loader.load(sample_file)
        print(f"Shape: {loader.get_shape()}")
        print(f"\nData Types:\n{loader.get_data_types()}")
        print(f"\nMissing Values:\n{loader.get_missing_summary()}")
        
        # Clean up
        os.remove(sample_file)
    
    except Exception as e:
        print(f"Error: {e}")
