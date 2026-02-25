"""
FeatureEngineer module for feature transformation and engineering.
Provides methods for encoding, scaling, and date feature extraction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Literal
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from utils.logger import get_logger
from utils.exceptions import DataValidationError, FeatureEngineeringError

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Perform feature engineering tasks on pandas DataFrames.
    Includes encoding, scaling, and feature extraction.
    Does not modify the original DataFrame.
    
    Attributes:
        df (pd.DataFrame): The original dataframe (never modified)
        encoders (dict): Stores fitted encoders for later use
        scalers (dict): Stores fitted scalers for later use
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize FeatureEngineer with a DataFrame.
        
        Args:
            df: pandas DataFrame to engineer
            
        Raises:
            DataValidationError: If input is not a pandas DataFrame or empty
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(f"Expected pandas DataFrame, got {type(df)}")
        
        if df.empty:
            raise DataValidationError("Cannot engineer an empty DataFrame")
        
        self.df = df
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        logger.info(f"FeatureEngineer initialized with shape {df.shape}")
    
    def one_hot_encode(
        self,
        columns: List[str],
        drop: Literal['first', 'if_binary', None] = 'first'
    ) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns.
        
        Args:
            columns: List of column names to encode
            drop: Strategy for handling collinearity
                 - 'first': Drop first category
                 - 'if_binary': Drop if binary feature
                 - None: Keep all categories
        
        Returns:
            pd.DataFrame: DataFrame with one-hot encoded columns
            
        Raises:
            DataValidationError: If columns don't exist
            FeatureEngineeringError: If encoding fails
        """
        try:
            df_encoded = self.df.copy()
            
            # Validate columns exist
            missing_cols = set(columns) - set(df_encoded.columns)
            if missing_cols:
                raise DataValidationError(f"Columns not found: {missing_cols}")
            
            # Perform one-hot encoding for each column
            for col in columns:
                encoder = OneHotEncoder(
                    sparse_output=False,
                    drop=drop,
                    handle_unknown='ignore'
                )
                
                encoded = encoder.fit_transform(df_encoded[[col]])
                
                # Get feature names
                feature_names = encoder.get_feature_names_out([col])
                
                # Create encoded dataframe
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=feature_names,
                    index=df_encoded.index
                )
                
                # Drop original column and concatenate encoded columns
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                
                logger.info(f"One-hot encoded column '{col}' into {len(feature_names)} features")
            
            return df_encoded
        
        except DataValidationError:
            logger.error("Validation error in one-hot encoding", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error in one-hot encoding: {e}")
            raise FeatureEngineeringError(f"Failed one-hot encoding: {e}")
    
    def label_encode(self, columns: List[str]) -> pd.DataFrame:
        """
        Apply label encoding to categorical columns.
        
        Args:
            columns: List of column names to encode
        
        Returns:
            pd.DataFrame: DataFrame with label-encoded columns
            
        Raises:
            ValueError: If columns don't exist
        """
        try:
            df_encoded = self.df.copy()
            
            # Validate columns exist
            missing_cols = set(columns) - set(df_encoded.columns)
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
            
            # Perform label encoding for each column
            for col in columns:
                encoder = LabelEncoder()
                
                # Handle missing values
                non_null_mask = df_encoded[col].notna()
                
                if non_null_mask.sum() > 0:
                    # Fit and transform non-null values
                    df_encoded.loc[non_null_mask, col] = encoder.fit_transform(
                        df_encoded.loc[non_null_mask, col].astype(str)
                    )
                    
                    # Store encoder for potential future use
                    self.encoders[col] = encoder
                    
                    logger.info(f"Label encoded column '{col}' with {len(encoder.classes_)} classes")
            
            return df_encoded
        
        except Exception as e:
            logger.error(f"Error in label encoding: {e}")
            raise
    
    def standard_scale(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply standard scaling (z-score normalization) to numerical columns.
        
        Formula: (x - mean) / std
        
        Args:
            columns: List of columns to scale. If None, scales all numerical columns.
        
        Returns:
            pd.DataFrame: DataFrame with scaled columns
            
        Raises:
            ValueError: If specified columns don't exist or aren't numerical
        """
        try:
            df_scaled = self.df.copy()
            
            # Get numerical columns
            numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numerical_cols:
                logger.warning("No numerical columns found for scaling")
                return df_scaled
            
            # If columns specified, validate and filter
            if columns:
                missing_cols = set(columns) - set(numerical_cols)
                if missing_cols:
                    raise ValueError(
                        f"Specified columns not found or not numerical: {missing_cols}"
                    )
                columns_to_scale = columns
            else:
                columns_to_scale = numerical_cols
            
            # Scale each column
            scaler = StandardScaler()
            df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
            
            # Store scaler
            self.scalers['standard'] = scaler
            
            logger.info(f"Applied standard scaling to {len(columns_to_scale)} columns")
            return df_scaled
        
        except Exception as e:
            logger.error(f"Error in standard scaling: {e}")
            raise
    
    def minmax_scale(
        self,
        columns: Optional[List[str]] = None,
        feature_range: tuple = (0, 1)
    ) -> pd.DataFrame:
        """
        Apply MinMax scaling (normalization) to numerical columns.
        
        Formula: (x - min) / (max - min) * (range_max - range_min) + range_min
        
        Args:
            columns: List of columns to scale. If None, scales all numerical columns.
            feature_range: Tuple of (min, max) for scaling range
        
        Returns:
            pd.DataFrame: DataFrame with scaled columns
            
        Raises:
            ValueError: If specified columns don't exist or aren't numerical
        """
        try:
            df_scaled = self.df.copy()
            
            # Get numerical columns
            numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numerical_cols:
                logger.warning("No numerical columns found for scaling")
                return df_scaled
            
            # If columns specified, validate and filter
            if columns:
                missing_cols = set(columns) - set(numerical_cols)
                if missing_cols:
                    raise ValueError(
                        f"Specified columns not found or not numerical: {missing_cols}"
                    )
                columns_to_scale = columns
            else:
                columns_to_scale = numerical_cols
            
            # Scale each column
            scaler = MinMaxScaler(feature_range=feature_range)
            df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
            
            # Store scaler
            self.scalers['minmax'] = scaler
            
            logger.info(
                f"Applied MinMax scaling to {len(columns_to_scale)} columns "
                f"with range {feature_range}"
            )
            return df_scaled
        
        except Exception as e:
            logger.error(f"Error in MinMax scaling: {e}")
            raise
    
    def extract_date_features(self, date_column: str) -> pd.DataFrame:
        """
        Extract date features from a datetime column.
        
        Creates: year, month, day, dayofweek, quarter, dayofyear
        
        Args:
            date_column: Name of the column containing datetime data
        
        Returns:
            pd.DataFrame: DataFrame with extracted date features (original column removed)
            
        Raises:
            ValueError: If column doesn't exist
            TypeError: If column cannot be converted to datetime
        """
        try:
            df_features = self.df.copy()
            
            # Validate column exists
            if date_column not in df_features.columns:
                raise ValueError(f"Column '{date_column}' not found")
            
            # Convert to datetime
            try:
                date_series = pd.to_datetime(df_features[date_column])
            except Exception as e:
                raise TypeError(
                    f"Cannot convert column '{date_column}' to datetime: {e}"
                )
            
            # Extract date features
            prefix = date_column.replace('_date', '').replace('_time', '')
            
            df_features[f'{prefix}_year'] = date_series.dt.year
            df_features[f'{prefix}_month'] = date_series.dt.month
            df_features[f'{prefix}_day'] = date_series.dt.day
            df_features[f'{prefix}_dayofweek'] = date_series.dt.dayofweek
            df_features[f'{prefix}_quarter'] = date_series.dt.quarter
            df_features[f'{prefix}_dayofyear'] = date_series.dt.dayofyear
            
            # Drop original date column
            df_features = df_features.drop(columns=[date_column])
            
            logger.info(
                f"Extracted date features from '{date_column}': "
                f"year, month, day, dayofweek, quarter, dayofyear"
            )
            return df_features
        
        except Exception as e:
            logger.error(f"Error extracting date features: {e}")
            raise
    
    def transform_features(
        self,
        one_hot_cols: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        scale_strategy: Optional[Literal['standard', 'minmax']] = None,
        date_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply multiple feature transformations in sequence.
        
        Order: Date extraction -> Label encoding -> One-hot encoding -> Scaling
        
        Args:
            one_hot_cols: Columns for one-hot encoding
            label_cols: Columns for label encoding
            scale_strategy: 'standard' or 'minmax'. If None, no scaling applied.
            date_cols: Date columns to extract features from
        
        Returns:
            pd.DataFrame: Fully transformed DataFrame
        """
        try:
            df_transformed = self.df.copy()
            
            logger.info("Starting feature transformation pipeline...")
            
            # Step 1: Extract date features
            if date_cols:
                for col in date_cols:
                    engineer = FeatureEngineer(df_transformed)
                    df_transformed = engineer.extract_date_features(col)
            
            # Step 2: Label encoding
            if label_cols:
                engineer = FeatureEngineer(df_transformed)
                df_transformed = engineer.label_encode(label_cols)
            
            # Step 3: One-hot encoding
            if one_hot_cols:
                engineer = FeatureEngineer(df_transformed)
                df_transformed = engineer.one_hot_encode(one_hot_cols)
            
            # Step 4: Scaling
            if scale_strategy == 'standard':
                engineer = FeatureEngineer(df_transformed)
                df_transformed = engineer.standard_scale()
            elif scale_strategy == 'minmax':
                engineer = FeatureEngineer(df_transformed)
                df_transformed = engineer.minmax_scale()
            
            logger.info(
                f"Feature transformation complete. "
                f"Shape: {self.df.shape} -> {df_transformed.shape}"
            )
            return df_transformed
        
        except Exception as e:
            logger.error(f"Error in feature transformation pipeline: {e}")
            raise


# ==================== COLUMN TYPE DETECTION ====================

_ID_NAME_PATTERNS = ("id", "uuid", "code", "key", "index", "no", "num", "ref")
_HIGH_UNIQUENESS_THRESHOLD = 0.95
_LOW_UNIQUE_NUMERIC_MAX = 20
_LOW_UNIQUE_NUMERIC_RATIO = 0.05
_HIGH_CARDINALITY_THRESHOLD = 50

# Max numeric columns allowed for polynomial expansion before a warning is issued
_POLY_WARN_THRESHOLD = 8


# ==================== MODULAR SMART TRANSFORMATIONS ====================

def apply_log_transform(
    df: pd.DataFrame,
    columns: List[str],
    new_column: bool = True,
) -> tuple:
    """
    Apply log1p transformation to numeric columns.

    Args:
        df: Input DataFrame (never modified).
        columns: Numeric columns to transform.
        new_column: If True, add '<col>_log1p' instead of replacing in-place.

    Returns:
        (transformed_df, log_messages): DataFrame and list of status strings.
    """
    result = df.copy()
    messages: List[str] = []

    for col in columns:
        if col not in result.columns:
            messages.append(f"⚠ Skipped '{col}': column not found.")
            continue
        if not pd.api.types.is_numeric_dtype(result[col]):
            messages.append(f"⚠ Skipped '{col}': not numeric.")
            continue

        series = result[col]
        n_invalid = int((series <= 0).sum())
        safe = series.clip(lower=0)  # treat negatives as 0 before log1p

        if n_invalid > 0:
            messages.append(
                f"⚠ '{col}': {n_invalid} non-positive value(s) clipped to 0 before log1p."
            )

        transformed = np.log1p(safe)
        if new_column:
            result[f"{col}_log1p"] = transformed
            messages.append(f"✓ Log1p column created: '{col}_log1p'.")
        else:
            result[col] = transformed
            messages.append(f"✓ Log1p applied in-place to '{col}'.")

    logger.info(f"apply_log_transform: {messages}")
    return result, messages


def apply_polynomial(
    df: pd.DataFrame,
    columns: List[str],
    degree: int = 2,
    interaction_only: bool = False,
    drop_original: bool = False,
) -> tuple:
    """
    Generate polynomial / interaction features for selected numeric columns.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to expand.
        degree: Polynomial degree (2 or 3).
        interaction_only: Only create cross-product terms (no powers).
        drop_original: Remove original selected columns from result.

    Returns:
        (transformed_df, log_messages)
    """
    from sklearn.preprocessing import PolynomialFeatures

    result = df.copy()
    messages: List[str] = []

    valid = [c for c in columns if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if not valid:
        return result, ["⚠ No valid numeric columns selected for polynomial features."]

    if len(valid) > _POLY_WARN_THRESHOLD:
        messages.append(
            f"⚠ {len(valid)} columns selected — polynomial expansion may generate many features. "
            f"Consider selecting fewer columns."
        )

    X = result[valid].fillna(0).values
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(valid)

    # Only keep new features (exclude originals already present)
    existing = set(valid)
    new_features = {name: X_poly[:, i] for i, name in enumerate(feature_names) if name not in existing}

    for name, values in new_features.items():
        result[name] = values

    if drop_original:
        result.drop(columns=valid, inplace=True)
        messages.append(f"✓ Original columns removed: {valid}.")

    n_new = len(new_features)
    messages.append(
        f"✓ Polynomial features (degree={degree}, interaction_only={interaction_only}): "
        f"{n_new} new feature(s) created from {valid}."
    )
    logger.info(messages[-1])
    return result, messages


def apply_binning(
    df: pd.DataFrame,
    column: str,
    n_bins: int = 5,
    strategy: str = "equal_width",
    label_bins: bool = True,
) -> tuple:
    """
    Bin a numeric column into categorical intervals.

    Args:
        df: Input DataFrame.
        column: Numeric column to bin.
        n_bins: Number of bins.
        strategy: 'equal_width' (pd.cut) or 'quantile' (pd.qcut).
        label_bins: If True, label bins with readable range strings.

    Returns:
        (transformed_df, log_messages)
    """
    result = df.copy()
    messages: List[str] = []
    new_col = f"{column}_binned"

    if column not in result.columns:
        return result, [f"⚠ Column '{column}' not found."]
    if not pd.api.types.is_numeric_dtype(result[column]):
        return result, [f"⚠ Column '{column}' is not numeric — cannot bin."]

    series = result[column].dropna()
    if series.nunique() < n_bins:
        n_bins = max(2, series.nunique())
        messages.append(f"⚠ '{column}': fewer unique values than bins; reduced to {n_bins} bins.")

    try:
        if strategy == "quantile":
            binned = pd.qcut(result[column], q=n_bins, duplicates="drop", labels=None if label_bins else False)
        else:
            binned = pd.cut(result[column], bins=n_bins, labels=None if label_bins else False)

        result[new_col] = binned.astype(str)
        messages.append(
            f"✓ Binned '{column}' → '{new_col}' ({strategy}, {n_bins} bins)."
        )
    except Exception as exc:
        messages.append(f"⚠ Binning failed for '{column}': {exc}")

    logger.info(f"apply_binning: {messages}")
    return result, messages


def apply_interactions(
    df: pd.DataFrame,
    columns: List[str],
    operation: str = "multiply",
) -> tuple:
    """
    Create pairwise interaction terms between numeric columns.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to cross.
        operation: 'multiply' (col1 * col2) or 'divide' (col1 / col2).

    Returns:
        (transformed_df, log_messages)
    """
    result = df.copy()
    messages: List[str] = []

    valid = [c for c in columns if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if len(valid) < 2:
        return result, ["⚠ Need at least 2 valid numeric columns for interaction features."]

    created: List[str] = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            c1, c2 = valid[i], valid[j]
            if operation == "multiply":
                new_col = f"{c1}_x_{c2}"
                result[new_col] = result[c1] * result[c2]
            else:  # divide
                new_col = f"{c1}_div_{c2}"
                denominator = result[c2].replace(0, np.nan)
                result[new_col] = result[c1] / denominator
            created.append(new_col)

    messages.append(
        f"✓ Interaction features ({operation}): {len(created)} new column(s) — {created}."
    )
    logger.info(messages[-1])
    return result, messages


def apply_frequency_encoding(
    df: pd.DataFrame,
    columns: List[str],
    new_column: bool = True,
) -> tuple:
    """
    Replace each category with its frequency (count) in the dataset.

    Args:
        df: Input DataFrame.
        columns: Categorical columns to encode.
        new_column: If True, add '<col>_freq' column; else replace in-place.

    Returns:
        (transformed_df, log_messages)
    """
    result = df.copy()
    messages: List[str] = []

    for col in columns:
        if col not in result.columns:
            messages.append(f"⚠ Skipped '{col}': column not found.")
            continue
        freq_map = result[col].value_counts().to_dict()
        encoded = result[col].map(freq_map)
        if new_column:
            result[f"{col}_freq"] = encoded
            messages.append(f"✓ Frequency encoding created: '{col}_freq'.")
        else:
            result[col] = encoded
            messages.append(f"✓ Frequency encoding applied in-place to '{col}'.")

    logger.info(f"apply_frequency_encoding: {messages}")
    return result, messages


def apply_target_encoding(
    df: pd.DataFrame,
    columns: List[str],
    target: str,
    smoothing: float = 1.0,
    new_column: bool = True,
) -> tuple:
    """
    Replace each category with smoothed mean of the target variable.

    Formula: (count * category_mean + smoothing * global_mean) / (count + smoothing)

    Args:
        df: Input DataFrame.
        columns: Categorical columns to encode.
        target: Numeric target column name.
        smoothing: Smoothing factor (higher = closer to global mean).
        new_column: If True, add '<col>_target_enc'; else replace in-place.

    Returns:
        (transformed_df, log_messages)
    """
    result = df.copy()
    messages: List[str] = []

    if target not in result.columns:
        return result, [f"⚠ Target column '{target}' not found."]
    if not pd.api.types.is_numeric_dtype(result[target]):
        return result, [f"⚠ Target column '{target}' must be numeric."]

    global_mean = result[target].mean()

    for col in columns:
        if col not in result.columns:
            messages.append(f"⚠ Skipped '{col}': column not found.")
            continue

        stats = result.groupby(col)[target].agg(["count", "mean"])
        smoothed = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )
        enc_map = smoothed.to_dict()
        encoded = result[col].map(enc_map)

        if new_column:
            result[f"{col}_target_enc"] = encoded
            messages.append(f"✓ Target encoding created: '{col}_target_enc' (target='{target}').")
        else:
            result[col] = encoded
            messages.append(f"✓ Target encoding applied in-place to '{col}'.")

    logger.info(f"apply_target_encoding: {messages}")
    return result, messages


def apply_transformations(df: pd.DataFrame, config: dict) -> tuple:
    """
    Apply a batch of smart transformations defined by a config dictionary.

    Supported keys in config:
        log        : dict  — {columns, new_column}
        polynomial : dict  — {columns, degree, interaction_only, drop_original}
        binning    : list  — [{column, n_bins, strategy, label_bins}, ...]
        interactions: dict — {columns, operation}
        frequency  : dict  — {columns, new_column}
        target_enc : dict  — {columns, target, smoothing, new_column}
        label      : list  — column names for label encoding
        onehot     : list  — column names for one-hot encoding
        scale      : str   — 'standard' | 'minmax' | None
        date       : list  — column names for date feature extraction

    Args:
        df: Original DataFrame (never modified).
        config: Transformation configuration dictionary.

    Returns:
        (transformed_df, all_messages): Transformed DataFrame and collected log list.
    """
    result = df.copy()
    all_messages: List[str] = []

    try:
        if config.get("log"):
            result, msgs = apply_log_transform(result, **config["log"])
            all_messages.extend(msgs)
    except Exception as exc:
        all_messages.append(f"⚠ Log transform error: {exc}")

    try:
        if config.get("polynomial"):
            result, msgs = apply_polynomial(result, **config["polynomial"])
            all_messages.extend(msgs)
    except Exception as exc:
        all_messages.append(f"⚠ Polynomial error: {exc}")

    try:
        if config.get("binning"):
            for bin_cfg in config["binning"]:
                result, msgs = apply_binning(result, **bin_cfg)
                all_messages.extend(msgs)
    except Exception as exc:
        all_messages.append(f"⚠ Binning error: {exc}")

    try:
        if config.get("interactions"):
            result, msgs = apply_interactions(result, **config["interactions"])
            all_messages.extend(msgs)
    except Exception as exc:
        all_messages.append(f"⚠ Interactions error: {exc}")

    try:
        if config.get("frequency"):
            result, msgs = apply_frequency_encoding(result, **config["frequency"])
            all_messages.extend(msgs)
    except Exception as exc:
        all_messages.append(f"⚠ Frequency encoding error: {exc}")

    try:
        if config.get("target_enc"):
            result, msgs = apply_target_encoding(result, **config["target_enc"])
            all_messages.extend(msgs)
    except Exception as exc:
        all_messages.append(f"⚠ Target encoding error: {exc}")

    try:
        if config.get("label"):
            eng = FeatureEngineer(result)
            result = eng.label_encode(config["label"])
            all_messages.append(f"✓ Label encoded: {config['label']}.")
    except Exception as exc:
        all_messages.append(f"⚠ Label encoding error: {exc}")

    try:
        if config.get("onehot"):
            eng = FeatureEngineer(result)
            result = eng.one_hot_encode(config["onehot"])
            all_messages.append(f"✓ One-hot encoded: {config['onehot']}.")
    except Exception as exc:
        all_messages.append(f"⚠ One-hot encoding error: {exc}")

    try:
        if config.get("scale"):
            eng = FeatureEngineer(result)
            if config["scale"] == "standard":
                result = eng.standard_scale()
                all_messages.append("✓ Standard scaling applied.")
            elif config["scale"] == "minmax":
                result = eng.minmax_scale()
                all_messages.append("✓ MinMax scaling applied.")
    except Exception as exc:
        all_messages.append(f"⚠ Scaling error: {exc}")

    try:
        if config.get("date"):
            for col in config["date"]:
                eng = FeatureEngineer(result)
                result = eng.extract_date_features(col)
                all_messages.append(f"✓ Date features extracted from '{col}'.")
    except Exception as exc:
        all_messages.append(f"⚠ Date extraction error: {exc}")

    logger.info(f"apply_transformations complete: {len(all_messages)} messages, shape {result.shape}")
    return result, all_messages


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically detect column types in a DataFrame.

    Returns a dictionary with keys:
        - 'numeric'     : continuous numeric columns
        - 'categorical' : string/object columns and low-cardinality numerics
        - 'binary'      : columns with exactly 2 unique non-null values
        - 'datetime'    : datetime-typed or safely datetime-parseable columns
        - 'id_columns'  : high-uniqueness columns with ID-like names or near-row-count cardinality

    Args:
        df: pandas DataFrame to inspect.

    Returns:
        Dict mapping type labels to lists of column names.
    """
    result: Dict[str, List[str]] = {
        "numeric": [],
        "categorical": [],
        "binary": [],
        "datetime": [],
        "id_columns": [],
    }

    if df.empty:
        return result

    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        n_non_null = series.count()

        # Skip entirely empty columns
        if n_non_null == 0:
            continue

        n_unique = series.nunique(dropna=True)
        uniqueness_ratio = n_unique / n_rows if n_rows > 0 else 0
        col_lower = col.lower()

        # --- Datetime detection ---
        if pd.api.types.is_datetime64_any_dtype(series):
            result["datetime"].append(col)
            continue
        if series.dtype == object:
            try:
                converted = pd.to_datetime(series.dropna().head(100), infer_datetime_format=True, errors="raise")
                if len(converted) > 0:
                    result["datetime"].append(col)
                    continue
            except Exception:
                pass

        # --- ID column detection ---
        is_id_by_name = any(pat in col_lower for pat in _ID_NAME_PATTERNS)
        is_id_by_ratio = uniqueness_ratio >= _HIGH_UNIQUENESS_THRESHOLD and n_unique >= max(10, n_rows * 0.5)
        if is_id_by_name and is_id_by_ratio:
            result["id_columns"].append(col)
            continue
        # Pure ID even without name pattern if nearly all rows unique and not obviously numeric predictor
        if is_id_by_ratio and (series.dtype == object or (
            pd.api.types.is_integer_dtype(series) and n_unique == n_rows
        )):
            result["id_columns"].append(col)
            continue

        # --- Binary detection (before numeric / categorical split) ---
        if n_unique == 2:
            result["binary"].append(col)
            continue

        # --- Numeric detection ---
        if pd.api.types.is_numeric_dtype(series):
            # Low-cardinality numeric -> treat as categorical
            if n_unique <= _LOW_UNIQUE_NUMERIC_MAX and uniqueness_ratio <= _LOW_UNIQUE_NUMERIC_RATIO:
                result["categorical"].append(col)
            else:
                result["numeric"].append(col)
            continue

        # --- Categorical (object / category dtype) ---
        result["categorical"].append(col)

    return result


def column_type_summary(df: pd.DataFrame) -> Dict[str, object]:
    """
    Build a human-readable summary of column types.

    Returns a dict with:
        - 'types'              : output of detect_column_types()
        - 'total_columns'      : int
        - 'counts_per_type'    : dict of type -> count
        - 'high_cardinality'   : list of categorical columns with >50 unique values
    """
    types = detect_column_types(df)
    counts = {k: len(v) for k, v in types.items()}
    high_card = [
        col for col in types["categorical"]
        if df[col].nunique(dropna=True) > _HIGH_CARDINALITY_THRESHOLD
    ]
    return {
        "types": types,
        "total_columns": len(df.columns),
        "counts_per_type": counts,
        "high_cardinality": high_card,
    }


# Example usage and testing
if __name__ == "__main__":
    try:
        # Create sample data
        sample_data = pd.DataFrame({
            'Age': [25, 30, 35, 40, 25],
            'Salary': [50000, 60000, 55000, 70000, 50000],
            'City': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
            'Gender': ['M', 'F', 'M', 'M', 'F'],
            'JoinDate': ['2020-01-15', '2019-05-10', '2021-03-20', '2018-11-05', '2020-07-30']
        })
        
        print("Original Data:")
        print(sample_data)
        print(f"\nShape: {sample_data.shape}\n")
        
        engineer = FeatureEngineer(sample_data)
        
        # Test individual methods
        print("=== Label Encoding ===")
        df_label = engineer.label_encode(['Gender'])
        print(df_label)
        
        print("\n=== One-Hot Encoding ===")
        df_onehot = engineer.one_hot_encode(['City'])
        print(df_onehot)
        
        print("\n=== Standard Scaling ===")
        df_standard = engineer.standard_scale(['Age', 'Salary'])
        print(df_standard)
        
        print("\n=== MinMax Scaling ===")
        df_minmax = engineer.minmax_scale(['Age', 'Salary'])
        print(df_minmax)
        
        print("\n=== Date Feature Extraction ===")
        df_dates = engineer.extract_date_features('JoinDate')
        print(df_dates)
        
        print("\n=== Complete Pipeline ===")
        df_transformed = engineer.transform_features(
            label_cols=['Gender'],
            one_hot_cols=['City'],
            scale_strategy='standard',
            date_cols=['JoinDate']
        )
        print(df_transformed)
        print(f"Final shape: {df_transformed.shape}")
    
    except Exception as e:
        print(f"Error: {e}")
