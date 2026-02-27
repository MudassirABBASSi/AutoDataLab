"""
Semi-Supervised Learning Models module.
Provides algorithms for learning from partially labeled data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from utils.logger import get_logger
from utils.exceptions import ModelingError, DataValidationError

from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

logger = get_logger(__name__)


class SemiSupervisedModels:
    """
    Wrapper for semi-supervised learning algorithms.
    Works with partially labeled datasets combining supervised and unsupervised learning.
    """

    SEMI_SUPERVISED_MODELS = {
        "Label Propagation": LabelPropagation(n_neighbors=7, max_iter=1000),
        "Label Spreading": LabelSpreading(kernel="rbf", alpha=0.8, max_iter=30),
        "Self-training (Logistic)": SelfTrainingClassifier(
            estimator=LogisticRegression(max_iter=1000),
            threshold=0.75,
            max_iter=10
        ),
        "Self-training (Random Forest)": SelfTrainingClassifier(
            estimator=RandomForestClassifier(n_estimators=100, random_state=42),
            threshold=0.75,
            max_iter=10
        ),
    }

    def __init__(self):
        """Initialize semi-supervised models wrapper."""
        logger.info("SemiSupervisedModels initialized with %d algorithms", len(self.SEMI_SUPERVISED_MODELS))

    @staticmethod
    def get_hyperparameters(model_name: str) -> Dict[str, Any]:
        """
        Get configurable hyperparameters for semi-supervised models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict with parameter names and configurations
        """
        params = {
            "Label Propagation": {"n_neighbors": (3, 20, 7), "max_iter": (100, 10000, 1000)},
            "Label Spreading": {"alpha": (0.0, 1.0, 0.8), "gamma": (0.0, 10.0, 20.0)},
            "Self-training (Logistic)": {"threshold": (0.3, 0.99, 0.75), "max_iter": (5, 50, 10)},
            "Self-training (Random Forest)": {"threshold": (0.3, 0.99, 0.75), "max_iter": (5, 50, 10)},
        }
        
        return params.get(model_name, {})

    @staticmethod
    def get_model(model_name: str, hyperparameters: Optional[Dict] = None) -> Any:
        """
        Get a semi-supervised model instance with optional hyperparameter overrides.
        
        Args:
            model_name: Name of the model
            hyperparameters: Dict of hyperparameters to override
            
        Returns:
            Model instance
        """
        hp = hyperparameters or {}
        
        if model_name == "Label Propagation":
            return LabelPropagation(
                n_neighbors=hp.get("n_neighbors", 7),
                max_iter=hp.get("max_iter", 1000)
            )
        elif model_name == "Label Spreading":
            return LabelSpreading(
                kernel="rbf",
                alpha=hp.get("alpha", 0.8),
                gamma=hp.get("gamma", 20.0),
                max_iter=hp.get("max_iter", 30)
            )
        elif model_name == "Self-training (Logistic)":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return SelfTrainingClassifier(
                    estimator=LogisticRegression(max_iter=1000, random_state=42),
                    threshold=hp.get("threshold", 0.75),
                    max_iter=hp.get("max_iter", 10)
                )
        elif model_name == "Self-training (Random Forest)":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return SelfTrainingClassifier(
                    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
                    threshold=hp.get("threshold", 0.75),
                    max_iter=hp.get("max_iter", 10)
                )
        else:
            raise ValueError(f"Unknown semi-supervised model: {model_name}")

    @staticmethod
    def get_available_models() -> list:
        """
        Get list of available semi-supervised models.
        
        Returns:
            Sorted list of model names
        """
        return sorted(list(SemiSupervisedModels.SEMI_SUPERVISED_MODELS.keys()))

    @staticmethod
    def validate_unlabeled_data(y: np.ndarray) -> bool:
        """
        Validate that data contains both labeled and unlabeled samples.
        
        Args:
            y: Target array with -1 for unlabeled samples
            
        Returns:
            True if data has unlabeled samples
        """
        unlabeled_count = np.sum(y == -1)
        labeled_count = np.sum(y != -1)
        is_valid = unlabeled_count > 0 and labeled_count > 0
        
        if is_valid:
            logger.info(f"Semi-supervised data valid: {labeled_count} labeled, {unlabeled_count} unlabeled")
        else:
            logger.warning(f"Semi-supervised data invalid: {labeled_count} labeled, {unlabeled_count} unlabeled")
        
        return is_valid
