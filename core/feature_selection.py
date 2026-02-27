"""
FeatureSelector module for feature selection techniques.
Provides methods to identify and select important features without model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from utils.logger import get_logger
from utils.exceptions import DataValidationError, FeatureSelectionError

logger = get_logger(__name__)


class FeatureSelector:
    """
    Select important features from pandas DataFrames.
    Uses multiple feature selection techniques.
    Does not include model training.
    
    Attributes:
        df (pd.DataFrame): The original dataframe
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (optional)
        selected_features (list): Currently selected feature names
    """
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """
        Initialize FeatureSelector with a DataFrame.
        
        Args:
            df: pandas DataFrame containing features
            target_column: Name of target column for supervised methods.
                          If None, only unsupervised methods available.
        
        Raises:
            DataValidationError: If input is not a pandas DataFrame, empty, or target invalid
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(f"Expected pandas DataFrame, got {type(df)}")
        
        if df.empty:
            raise DataValidationError("Cannot select features from empty DataFrame")
        
        self.df = df
        self.selected_features: List[str] = list(df.columns)
        self.y: Optional[pd.Series] = None
        self.target_column = target_column
        
        # If target column specified, separate X and y
        if target_column:
            if target_column not in df.columns:
                raise DataValidationError(
                    f"Target column '{target_column}' not found in DataFrame"
                )
            self.y = df[target_column]
            self.X = df.drop(columns=[target_column])
        else:
            self.X = df
        
        logger.info(
            f"FeatureSelector initialized with {len(self.X.columns)} features"
            f"{f' and target: {target_column}' if target_column else ''}"
        )
    
    def variance_threshold(self, threshold: float = 0.0) -> List[str]:
        """
        Select features with variance above threshold.
        
        Removes features with variance <= threshold.
        Useful for removing constant or near-constant features.
        
        Args:
            threshold: Variance threshold (default 0.0)
        
        Returns:
            list: Names of selected features
        
        Raises:
            DataValidationError: If threshold is negative
            FeatureSelectionError: If selection fails
        """
        try:
            if threshold < 0:
                raise DataValidationError("Threshold must be non-negative")
            
            # Get numerical columns only
            X_numerical = self.X.select_dtypes(include=[np.number])
            
            if X_numerical.empty:
                logger.warning("No numerical features found for variance threshold")
                return self.selected_features
            
            # Calculate variance for numerical features
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X_numerical)
            
            # Get selected numerical column names
            selected_numerical = X_numerical.columns[selector.get_support()].tolist()
            
            # Combine with non-numerical features
            non_numerical = self.X.select_dtypes(exclude=[np.number]).columns.tolist()
            selected = selected_numerical + non_numerical
            
            removed_count = len(self.X.columns) - len(selected)
            logger.info(
                f"Variance threshold: Removed {removed_count} features "
                f"with variance <= {threshold}"
            )
            
            self.selected_features = selected
            return selected
        
        except DataValidationError:
            logger.error("Validation error in variance threshold", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error in variance threshold: {e}")
            raise FeatureSelectionError(f"Failed variance threshold: {e}")
    
    def correlation_threshold(self, threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features.
        
        For each pair of features with correlation >= threshold,
        removes the feature that appears second.
        
        Args:
            threshold: Correlation threshold (0 to 1). Default 0.95
        
        Returns:
            list: Names of selected features
            
        Raises:
            ValueError: If threshold is not between 0 and 1
        """
        try:
            if not (0 <= threshold <= 1):
                raise ValueError("Threshold must be between 0 and 1")
            
            # Get numerical columns only
            X_numerical = self.X.select_dtypes(include=[np.number])
            
            if X_numerical.empty:
                logger.warning("No numerical features found for correlation threshold")
                return self.selected_features
            
            # Calculate correlation matrix
            corr_matrix = X_numerical.corr().abs()
            
            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with correlation greater than threshold
            to_drop = [
                column for column in upper.columns
                if any(upper[column] > threshold)
            ]
            
            selected_numerical = [
                col for col in X_numerical.columns if col not in to_drop
            ]
            
            # Combine with non-numerical features
            non_numerical = self.X.select_dtypes(exclude=[np.number]).columns.tolist()
            selected = selected_numerical + non_numerical
            
            logger.info(
                f"Correlation threshold: Removed {len(to_drop)} highly correlated features"
            )
            
            self.selected_features = selected
            return selected
        
        except Exception as e:
            logger.error(f"Error in correlation threshold: {e}")
            raise
    
    def select_k_best(
        self,
        k: int = 10,
        score_func: str = 'f_classif'
    ) -> Dict[str, Any]:
        """
        Select top K features using statistical score function.
        
        Args:
            k: Number of top features to select
            score_func: Scoring function
                       - 'f_classif': ANOVA F-value (classification)
                       - 'f_regression': F-value (regression)
                       - 'mutual_info_classif': Mutual information (classification)
                       - 'mutual_info_regression': Mutual information (regression)
        
        Returns:
            dict: Selected features and their scores
            
        Raises:
            ValueError: If target variable not provided or invalid score_func
        """
        try:
            if self.y is None:
                raise ValueError(
                    "Target variable required for SelectKBest. "
                    "Provide target_column in constructor."
                )
            
            # Map score function names to actual functions
            score_functions = {
                'f_classif': f_classif,
                'f_regression': f_regression,
                'mutual_info_classif': mutual_info_classif,
                'mutual_info_regression': mutual_info_regression
            }
            
            if score_func not in score_functions:
                raise ValueError(
                    f"Invalid score_func '{score_func}'. "
                    f"Choose from: {', '.join(score_functions.keys())}"
                )
            
            # Get numerical columns
            X_numerical = self.X.select_dtypes(include=[np.number])
            
            if X_numerical.empty:
                raise ValueError("No numerical features found for SelectKBest")
            
            # Remove rows with NaN values for SelectKBest (sklearn requirement)
            # Find rows without NaN in both X and y
            valid_idx = X_numerical.dropna().index
            X_clean = X_numerical.loc[valid_idx]
            y_clean = self.y.loc[valid_idx]
            
            # Ensure k is reasonable
            k = min(k, len(X_numerical.columns))
            
            # Apply SelectKBest
            selector = SelectKBest(
                score_func=score_functions[score_func],
                k=k
            )
            selector.fit(X_clean, y_clean)
            
            # Get selected features and scores
            selected_mask = selector.get_support()
            scores = selector.scores_
            
            feature_scores = {}
            for feature, score, selected in zip(
                X_numerical.columns, scores, selected_mask
            ):
                if selected:
                    feature_scores[feature] = float(score)
            
            # Sort by score descending
            sorted_features = sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            selected = [feat for feat, _ in sorted_features]
            
            # Add non-numerical features
            non_numerical = self.X.select_dtypes(exclude=[np.number]).columns.tolist()
            selected_all = selected + non_numerical
            
            logger.info(
                f"SelectKBest ({score_func}): Selected {len(selected)} features"
            )
            
            self.selected_features = selected_all
            
            return {
                'selected_features': selected_all,
                'feature_scores': dict(sorted_features),
                'method': score_func
            }
        
        except Exception as e:
            logger.error(f"Error in SelectKBest: {e}")
            raise
    
    def tree_based_importance(
        self,
        estimator_type: str = 'classifier',
        threshold_percentile: float = 50
    ) -> Dict[str, Any]:
        """
        Select features based on tree-based feature importance.
        
        Uses Random Forest to calculate feature importance.
        Does not train a model - only calculates importance scores.
        
        Args:
            estimator_type: 'classifier' or 'regressor'
            threshold_percentile: Keep features above this percentile (0-100)
        
        Returns:
            dict: Selected features and their importance scores
            
        Raises:
            ValueError: If target variable not provided or invalid estimator_type
        """
        try:
            if self.y is None:
                raise ValueError(
                    "Target variable required for tree-based importance. "
                    "Provide target_column in constructor."
                )
            
            if estimator_type not in ['classifier', 'regressor']:
                raise ValueError(
                    "estimator_type must be 'classifier' or 'regressor'"
                )
            
            # Get numerical columns
            X_numerical = self.X.select_dtypes(include=[np.number])
            
            if X_numerical.empty:
                raise ValueError("No numerical features found for tree-based importance")
            
            # Remove rows with NaN values (tree methods require clean data)
            # Find rows without NaN in both X and y
            valid_idx = X_numerical.dropna().index
            X_clean = X_numerical.loc[valid_idx]
            y_clean = self.y.loc[valid_idx]
            
            # Choose estimator
            if estimator_type == 'classifier':
                estimator = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Fit estimator to get importance
            estimator.fit(X_clean, y_clean)
            
            # Get feature importances
            importances = estimator.feature_importances_
            
            # Create feature importance dictionary
            feature_importance = {}
            for feature, importance in zip(X_numerical.columns, importances):
                feature_importance[feature] = float(importance)
            
            # Calculate threshold based on percentile
            importance_values = list(feature_importance.values())
            threshold = np.percentile(importance_values, threshold_percentile)
            
            # Select features above threshold
            selected = [
                feat for feat, imp in feature_importance.items()
                if imp >= threshold
            ]
            
            # Sort by importance descending
            selected = sorted(
                selected,
                key=lambda x: feature_importance[x],
                reverse=True
            )
            
            # Add non-numerical features
            non_numerical = self.X.select_dtypes(exclude=[np.number]).columns.tolist()
            selected_all = selected + non_numerical
            
            logger.info(
                f"Tree-based importance ({estimator_type}): "
                f"Selected {len(selected)} features above {threshold_percentile}th percentile"
            )
            
            self.selected_features = selected_all
            
            return {
                'selected_features': selected_all,
                'feature_importances': feature_importance,
                'threshold': float(threshold),
                'estimator_type': estimator_type
            }
        
        except Exception as e:
            logger.error(f"Error in tree-based importance: {e}")
            raise
    
    def get_selected_features(self) -> List[str]:
        """
        Get the currently selected features.
        
        Returns:
            list: Names of selected features
        """
        return self.selected_features
    
    def compare_methods(self) -> Dict[str, Any]:
        """
        Compare results from multiple feature selection methods.
        Returns selected features from each method.
        
        Returns:
            dict: Results from each method
        """
        try:
            results = {}
            
            logger.info("Comparing feature selection methods...")
            
            # Variance threshold
            try:
                results['variance_threshold'] = self.variance_threshold(threshold=0.01)
            except Exception as e:
                logger.warning(f"Variance threshold failed: {e}")
                results['variance_threshold'] = []
            
            # Correlation threshold
            try:
                results['correlation_threshold'] = self.correlation_threshold(threshold=0.95)
            except Exception as e:
                logger.warning(f"Correlation threshold failed: {e}")
                results['correlation_threshold'] = []
            
            # SelectKBest
            if self.y is not None:
                try:
                    k_best = self.select_k_best(k=10, score_func='f_classif')
                    results['select_k_best'] = k_best['selected_features']
                except Exception as e:
                    logger.warning(f"SelectKBest failed: {e}")
                    results['select_k_best'] = []
                
                # Tree-based importance
                try:
                    tree_import = self.tree_based_importance(estimator_type='classifier')
                    results['tree_based_importance'] = tree_import['selected_features']
                except Exception as e:
                    logger.warning(f"Tree-based importance failed: {e}")
                    results['tree_based_importance'] = []
            
            logger.info("Feature selection comparison complete")
            return results
        
        except Exception as e:
            logger.error(f"Error in compare methods: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    try:
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'Feature1': np.random.randn(100),
            'Feature2': np.random.randn(100),
            'Feature3': np.random.randn(100) * 0.01,  # Low variance
            'Feature4': np.random.randn(100),
            'Feature5': np.random.randn(100),  # Highly correlated with Feature1
            'Target': np.random.randint(0, 2, 100)
        })
        
        # Make Feature5 highly correlated with Feature1
        sample_data['Feature5'] = sample_data['Feature1'] + np.random.randn(100) * 0.1
        
        print("Original Data Shape:", sample_data.shape)
        print("Features:", list(sample_data.columns))
        print()
        
        # Initialize selector
        selector = FeatureSelector(sample_data, target_column='Target')
        
        print("=== Variance Threshold ===")
        var_features = selector.variance_threshold(threshold=0.01)
        print(f"Selected: {var_features}\n")
        
        print("=== Correlation Threshold ===")
        corr_features = selector.correlation_threshold(threshold=0.95)
        print(f"Selected: {corr_features}\n")
        
        print("=== Select K-Best (f_classif) ===")
        kbest_result = selector.select_k_best(k=3, score_func='f_classif')
        print(f"Selected: {kbest_result['selected_features']}")
        print(f"Scores: {kbest_result['feature_scores']}\n")
        
        print("=== Tree-Based Importance ===")
        tree_result = selector.tree_based_importance(
            estimator_type='classifier',
            threshold_percentile=50
        )
        print(f"Selected: {tree_result['selected_features']}")
        print(f"Importances: {tree_result['feature_importances']}\n")
        
        print("=== Compare All Methods ===")
        comparison = selector.compare_methods()
        for method, features in comparison.items():
            print(f"{method}: {features}")
    
    except Exception as e:
        print(f"Error: {e}")
