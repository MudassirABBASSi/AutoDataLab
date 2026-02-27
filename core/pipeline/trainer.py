"""
Model Training Pipeline module.
Handles train-test split, model training, and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from utils.logger import get_logger
from utils.exceptions import ModelingError, DataValidationError

logger = get_logger(__name__)


class ModelTrainer:
    """
    Complete model training and evaluation pipeline.
    Handles data preprocessing, training, and result storage.
    """

    def __init__(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize trainer with data.

        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for testing (0-1)
            random_state: Random seed for reproducibility
        """
        self.df = df.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoder = None
        self.model = None
        self.task_type = None

        logger.info(f"ModelTrainer initialized with {len(df)} rows, target={target_col}")

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target.
        Handle missing values, encode categorical variables.

        Returns:
            (X, y)
        """
        # Separate target and features
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found")

        y = self.df[self.target_col].copy()
        X = self.df.drop(columns=[self.target_col]).copy()

        # Drop rows with missing target
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]

        # Handle missing values in X: fill numeric with median, categorical with mode
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown", inplace=True)

        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Encode target if categorical (for classification)
        if not pd.api.types.is_numeric_dtype(y):
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y), index=y.index)

        self.X = X
        self.y = y

        logger.info(f"Data prepared: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train/test sets.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Call prepare_data() first")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        logger.info(f"Data split: train {len(self.X_train)}, test {len(self.X_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_features(self, fit_on_train: bool = True) -> None:
        """
        Scale numeric features using StandardScaler.

        Args:
            fit_on_train: If True, fit scaler on training data only
        """
        if self.X_train is None:
            raise RuntimeError("Call split_data() first")

        self.scaler = StandardScaler()

        if fit_on_train:
            self.scaler.fit(self.X_train)
            self.X_train = pd.DataFrame(
                self.scaler.transform(self.X_train),
                columns=self.X_train.columns,
                index=self.X_train.index
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=self.X_test.columns,
                index=self.X_test.index
            )
        else:
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(self.X_train),
                columns=self.X_train.columns,
                index=self.X_train.index
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=self.X_test.columns,
                index=self.X_test.index
            )

        logger.info("Features scaled")

    def train_model(self, model: Any, scale: bool = True) -> Any:
        """
        Train model on training data.

        Args:
            model: Untrained model instance
            scale: Whether to scale features

        Returns:
            Trained model
        """
        if self.X_train is None:
            raise RuntimeError("Call split_data() first")

        if scale:
            self.scale_features()

        self.model = model
        self.model.fit(self.X_train, self.y_train)

        logger.info(f"Model trained: {model.__class__.__name__}")
        return self.model

    def get_predictions(self, use_test: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions on train or test set.

        Args:
            use_test: If True, predict on test set; else on train set

        Returns:
            (predictions, probabilities or None)
        """
        if self.model is None:
            raise RuntimeError("Call train_model() first")

        X = self.X_test if use_test else self.X_train
        y_pred = self.model.predict(X)

        # Get probabilities if available (classification)
        y_pred_proba = None
        if hasattr(self.model, "predict_proba"):
            y_pred_proba = self.model.predict_proba(X)

        return y_pred, y_pred_proba

    def get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Return prepared train/test sets."""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to pickle file.

        Args:
            filepath: Path to save file
        """
        if self.model is None:
            raise RuntimeError("No trained model to save")

        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> Any:
        """
        Load model from pickle file.

        Args:
            filepath: Path to model file

        Returns:
            Loaded model
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)

        logger.info(f"Model loaded from {filepath}")
        return self.model

    def get_feature_names(self) -> list:
        """Get list of feature names after preprocessing."""
        if self.X is None:
            return []
        return list(self.X.columns)
