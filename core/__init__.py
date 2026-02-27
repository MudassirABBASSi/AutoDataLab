"""Core module for data processing."""

from .data_loader import DataLoader
from .data_validator import DataValidator, validate_dataframe, get_data_quality_report
from .profiler import DataProfiler
from .cleaning import DataCleaner
from .feature_engineering import (
    FeatureEngineer, detect_column_types, column_type_summary,
    apply_log_transform, apply_polynomial, apply_binning, apply_interactions,
    apply_frequency_encoding, apply_target_encoding, apply_transformations,
)
from .feature_selection import FeatureSelector
from .eda import EDAVisualizer, bivariate_analysis, univariate_analysis, multivariate_analysis
from .models import SupervisedModels, UnsupervisedModels, SemiSupervisedModels
from .evaluation import ModelMetrics, MetricsVisualizer
from .pipeline import ModelTrainer

__all__ = [
    'DataLoader', 'DataValidator', 'validate_dataframe', 'get_data_quality_report',
    'DataProfiler', 'DataCleaner',
    'FeatureEngineer', 'detect_column_types', 'column_type_summary',
    'apply_log_transform', 'apply_polynomial', 'apply_binning', 'apply_interactions',
    'apply_frequency_encoding', 'apply_target_encoding', 'apply_transformations',
    'FeatureSelector',
    'EDAVisualizer', 'bivariate_analysis', 'univariate_analysis', 'multivariate_analysis',
    'SupervisedModels', 'UnsupervisedModels', 'SemiSupervisedModels',
    'ModelMetrics', 'MetricsVisualizer', 'ModelTrainer',
]
