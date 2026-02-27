"""
Supervised Learning Models module.
Provides wrapper classes for regression and classification models with 23+ algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
from utils.logger import get_logger
from utils.exceptions import ModelingError, DataValidationError

# Regression imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Classification imports
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# XGBoost & LightGBM
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = get_logger(__name__)


class SupervisedModels:
    """
    Wrapper for supervised learning models.
    Supports 12 regression and 11 classification algorithms.
    """

    REGRESSION_MODELS = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.1, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0),
        "SVR": SVR(kernel="rbf", C=1.0, gamma="scale"),
        "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }

    CLASSIFICATION_MODELS = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "Ridge Classifier": RidgeClassifier(alpha=1.0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=0),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True, C=1.0, random_state=42),
        "Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }

    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        REGRESSION_MODELS["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
        CLASSIFICATION_MODELS["LightGBM"] = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)

    def __init__(self):
        """Initialize the models wrapper."""
        logger.info("SupervisedModels initialized with %d regression and %d classification models", 
                   len(self.REGRESSION_MODELS), len(self.CLASSIFICATION_MODELS))

    @staticmethod
    def get_hyperparameters(model_name: str, task_type: str) -> Dict[str, Any]:
        """
        Get default and configurable hyperparameters for a model.
        
        Args:
            model_name: Name of the model
            task_type: "classification" or "regression"
            
        Returns:
            Dict with parameter names and default values
        """
        params = {
            # Regression
            "Linear Regression": {},
            "Ridge": {"alpha": (0.001, 10.0, 1.0)},
            "Lasso": {"alpha": (0.001, 10.0, 0.1)},
            "ElasticNet": {"alpha": (0.001, 10.0, 0.1), "l1_ratio": (0.0, 1.0, 0.5)},
            "Decision Tree": {"max_depth": (1, 30, 10), "min_samples_split": (2, 20, 2)},
            "Random Forest": {"n_estimators": (10, 500, 100), "max_depth": (1, 30, 15)},
            "Gradient Boosting": {"n_estimators": (10, 500, 100), "learning_rate": (0.001, 1.0, 0.1)},
            "XGBoost": {"n_estimators": (10, 500, 100), "max_depth": (1, 10, 5), "learning_rate": (0.001, 1.0, 0.1)},
            "SVR": {"C": (0.1, 100.0, 1.0), "gamma": ("scale", "auto")},
            "KNN": {"n_neighbors": (1, 30, 5)},
            "MLP": {"hidden_layer_sizes": [(100,), (100, 50), (100, 50, 25)]},
            "LightGBM": {"n_estimators": (10, 500, 100), "max_depth": (1, 10, 5)},
            # Classification
            "Logistic Regression": {"C": (0.001, 100.0, 1.0)},
            "Ridge Classifier": {"alpha": (0.001, 10.0, 1.0)},
        }
        
        return params.get(model_name, {})
    
    @staticmethod
    def detect_task_type(y: pd.Series) -> str:
        """
        Detect whether target is classification or regression.

        Args:
            y: Target variable (pandas Series)

        Returns:
            str: "classification" or "regression"
        """
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"

        # Check number of unique values
        n_unique = y.nunique()
        
        # If very few unique values, likely classification
        if n_unique <= 20 or (n_unique / len(y)) < 0.05:
            # Check if values are integers only
            if all(val == int(val) for val in y.dropna()):
                return "classification"

        return "regression"

    @staticmethod
    def get_model(model_name: str, task_type: str, hyperparameters: Optional[Dict] = None) -> Any:
        """
        Get an untrained model instance with optional hyperparameter overrides.

        Args:
            model_name: Name of the model
            task_type: "classification" or "regression"
            hyperparameters: Dict of hyperparameters to override defaults

        Returns:
            Model instance
        """
        hp = hyperparameters or {}
        
        if task_type == "regression":
            if model_name == "Linear Regression":
                return LinearRegression()
            elif model_name == "Ridge":
                return Ridge(alpha=hp.get("alpha", 1.0), random_state=42)
            elif model_name == "Lasso":
                return Lasso(alpha=hp.get("alpha", 0.1), random_state=42)
            elif model_name == "ElasticNet":
                return ElasticNet(alpha=hp.get("alpha", 0.1), l1_ratio=hp.get("l1_ratio", 0.5), random_state=42)
            elif model_name == "Decision Tree":
                return DecisionTreeRegressor(max_depth=hp.get("max_depth", 10), random_state=42)
            elif model_name == "Random Forest":
                return RandomForestRegressor(
                    n_estimators=hp.get("n_estimators", 100),
                    max_depth=hp.get("max_depth", 15),
                    random_state=42, n_jobs=-1
                )
            elif model_name == "Gradient Boosting":
                return GradientBoostingRegressor(
                    n_estimators=hp.get("n_estimators", 100),
                    learning_rate=hp.get("learning_rate", 0.1),
                    max_depth=hp.get("max_depth", 5),
                    random_state=42
                )
            elif model_name == "XGBoost":
                return xgb.XGBRegressor(
                    n_estimators=hp.get("n_estimators", 100),
                    max_depth=hp.get("max_depth", 5),
                    learning_rate=hp.get("learning_rate", 0.1),
                    random_state=42, verbosity=0
                )
            elif model_name == "SVR":
                return SVR(C=hp.get("C", 1.0), gamma=hp.get("gamma", "scale"))
            elif model_name == "KNN":
                return KNeighborsRegressor(n_neighbors=hp.get("n_neighbors", 5), n_jobs=-1)
            elif model_name == "MLP":
                return MLPRegressor(hidden_layer_sizes=hp.get("hidden_layer_sizes", (100, 50)), max_iter=500, random_state=42)
            elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
                return lgb.LGBMRegressor(
                    n_estimators=hp.get("n_estimators", 100),
                    max_depth=hp.get("max_depth", 5),
                    random_state=42, verbose=-1
                )
            else:
                raise ValueError(f"Unknown regression model: {model_name}")
        
        elif task_type == "classification":
            if model_name == "Logistic Regression":
                return LogisticRegression(max_iter=1000, C=hp.get("C", 1.0), random_state=42, n_jobs=-1)
            elif model_name == "Ridge Classifier":
                return RidgeClassifier(alpha=hp.get("alpha", 1.0), random_state=42)
            elif model_name == "Decision Tree":
                return DecisionTreeClassifier(max_depth=hp.get("max_depth", 10), random_state=42)
            elif model_name == "Random Forest":
                return RandomForestClassifier(
                    n_estimators=hp.get("n_estimators", 100),
                    max_depth=hp.get("max_depth", 15),
                    random_state=42, n_jobs=-1
                )
            elif model_name == "Gradient Boosting":
                return GradientBoostingClassifier(
                    n_estimators=hp.get("n_estimators", 100),
                    learning_rate=hp.get("learning_rate", 0.1),
                    max_depth=hp.get("max_depth", 5),
                    random_state=42
                )
            elif model_name == "XGBoost":
                return xgb.XGBClassifier(
                    n_estimators=hp.get("n_estimators", 100),
                    max_depth=hp.get("max_depth", 5),
                    learning_rate=hp.get("learning_rate", 0.1),
                    random_state=42, verbosity=0
                )
            elif model_name == "KNN":
                return KNeighborsClassifier(n_neighbors=hp.get("n_neighbors", 5), n_jobs=-1)
            elif model_name == "SVM":
                return SVC(kernel="rbf", probability=True, C=hp.get("C", 1.0), random_state=42)
            elif model_name == "Naive Bayes":
                return GaussianNB()
            elif model_name == "MLP":
                return MLPClassifier(hidden_layer_sizes=hp.get("hidden_layer_sizes", (100, 50)), max_iter=500, random_state=42)
            elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
                return lgb.LGBMClassifier(
                    n_estimators=hp.get("n_estimators", 100),
                    max_depth=hp.get("max_depth", 5),
                    random_state=42, verbose=-1
                )
            else:
                raise ValueError(f"Unknown classification model: {model_name}")
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    @staticmethod
    def get_available_models(task_type: str) -> list:
        """
        Get list of available models for a task type.

        Args:
            task_type: "classification" or "regression"

        Returns:
            List of model names sorted alphabetically
        """
        if task_type == "regression":
            return sorted(list(SupervisedModels.REGRESSION_MODELS.keys()))
        elif task_type == "classification":
            return sorted(list(SupervisedModels.CLASSIFICATION_MODELS.keys()))
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    @staticmethod
    def get_feature_importance(model: Any, feature_names: list, X_test: Optional[np.ndarray] = None, 
                               y_test: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Extract feature importance from trained model.
        Falls back to permutation importance if native importance unavailable.

        Args:
            model: Trained model
            feature_names: List of feature column names
            X_test: Test features for permutation importance
            y_test: Test labels for permutation importance

        Returns:
            DataFrame with feature importance; empty if not supported
        """
        importance_dict = {}

        # Try native feature importance first
        if hasattr(model, "feature_importances_"):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, "coef_"):
            # For linear models, use absolute value of coefficients
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = np.abs(coef[0])
            else:
                coef = np.abs(coef)
            importance_dict = dict(zip(feature_names, coef))
        
        # Fall back to permutation importance if available
        elif X_test is not None and y_test is not None:
            try:
                from sklearn.inspection import permutation_importance
                result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                importance_dict = dict(zip(feature_names, result.importances_mean))
            except Exception as e:
                logger.warning(f"Permutation importance failed: {e}")
                return pd.DataFrame()

        if not importance_dict:
            return pd.DataFrame()

        df = pd.DataFrame(
            list(importance_dict.items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False)

        return df
