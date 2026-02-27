"""
Model Evaluation Metrics module.
Computes and reports comprehensive metrics for all paradigms: 
supervised (classification/regression), unsupervised (clustering), and semi-supervised.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from utils.logger import get_logger
from utils.exceptions import EvaluationError, DataValidationError

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, roc_curve, silhouette_score, davies_bouldin_score,
    calinski_harabasz_score
)

logger = get_logger(__name__)


class ModelMetrics:
    """
    Comprehensive model evaluation metrics.
    Handles both classification and regression metrics.
    """

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Compute classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for binary classification ROC-AUC)

        Returns:
            Dictionary with metrics
        """
        metrics = {}

        # Basic metrics
        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        
        # Handle binary vs multiclass
        n_classes = len(np.unique(y_true))
        average_method = "binary" if n_classes == 2 else "weighted"

        metrics["Precision"] = precision_score(y_true, y_pred, average=average_method, zero_division=0)
        metrics["Recall"] = recall_score(y_true, y_pred, average=average_method, zero_division=0)
        metrics["F1 Score"] = f1_score(y_true, y_pred, average=average_method, zero_division=0)

        # ROC-AUC for binary classification
        if n_classes == 2 and y_pred_proba is not None:
            try:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics["ROC-AUC"] = None

        # Confusion matrix
        metrics["Confusion Matrix"] = confusion_matrix(y_true, y_pred)

        logger.info(f"Classification metrics computed: {list(metrics.keys())}")
        return metrics

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with metrics
        """
        metrics = {}

        metrics["MAE"] = mean_absolute_error(y_true, y_pred)
        metrics["MSE"] = mean_squared_error(y_true, y_pred)
        metrics["RMSE"] = np.sqrt(metrics["MSE"])
        metrics["R²"] = r2_score(y_true, y_pred)
        metrics["Adj. R²"] = 1 - (1 - metrics["R²"]) * (len(y_true) - 1) / (len(y_true) - 2) if len(y_true) > 2 else metrics["R²"]

        logger.info(f"Regression metrics computed: {list(metrics.keys())}")
        return metrics

    @staticmethod
    def get_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve data for binary classification.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class

        Returns:
            (fpr, tpr, thresholds)
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            return fpr, tpr, thresholds
        except Exception as e:
            logger.error(f"Error computing ROC curve: {e}")
            return None, None, None

    @staticmethod
    def summary_report(metrics: Dict[str, Any], task_type: str) -> str:
        """
        Generate a text summary of metrics.

        Args:
            metrics: Dictionary of computed metrics
            task_type: "classification", "regression", "clustering", or "anomaly"

        Returns:
            Formatted summary string
        """
        lines = []

        if task_type == "classification":
            lines.append("=== Classification Metrics ===")
            for key in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                if key in metrics:
                    lines.append(f"{key}: {metrics[key]:.4f}")
            if metrics.get("ROC-AUC"):
                lines.append(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")

        elif task_type == "regression":
            lines.append("=== Regression Metrics ===")
            for key in ["MAE", "RMSE", "R²"]:
                if key in metrics:
                    lines.append(f"{key}: {metrics[key]:.4f}")

        elif task_type == "clustering":
            lines.append("=== Clustering Metrics ===")
            for key in ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]:
                if key in metrics:
                    lines.append(f"{key}: {metrics[key]:.4f}")

        elif task_type == "anomaly":
            lines.append("=== Anomaly Detection ===")
            for key in ["Anomalies Detected", "Anomaly %"]:
                if key in metrics:
                    lines.append(f"{key}: {metrics[key]}")

        logger.info("Summary report generated")
        return "\n".join(lines)

    @staticmethod
    def clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute clustering evaluation metrics.

        Args:
            X: Feature matrix
            labels: Cluster labels

        Returns:
            Dictionary with clustering metrics
        """
        metrics = {}
        
        # Only compute if we have more than 1 cluster
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            logger.warning("Clustering needs at least 2 clusters")
            return metrics

        try:
            metrics["Silhouette"] = silhouette_score(X, labels)
        except Exception as e:
            logger.warning(f"Silhouette score failed: {e}")

        try:
            metrics["Davies-Bouldin"] = davies_bouldin_score(X, labels)
        except Exception as e:
            logger.warning(f"Davies-Bouldin score failed: {e}")

        try:
            metrics["Calinski-Harabasz"] = calinski_harabasz_score(X, labels)
        except Exception as e:
            logger.warning(f"Calinski-Harabasz score failed: {e}")

        logger.info(f"Clustering metrics computed: {list(metrics.keys())}")
        return metrics

    @staticmethod
    def anomaly_metrics(X: np.ndarray, anomaly_labels: np.ndarray) -> Dict[str, Any]:
        """
        Compute anomaly detection metrics.

        Args:
            X: Feature matrix
            anomaly_labels: -1 for anomalies, 1 for normal

        Returns:
            Dictionary with anomaly detection metrics
        """
        metrics = {}

        n_anomalies = np.sum(anomaly_labels == -1)
        n_normal = np.sum(anomaly_labels == 1)
        total = len(anomaly_labels)

        metrics["Anomalies Detected"] = n_anomalies
        metrics["Normal Samples"] = n_normal
        metrics["Anomaly %"] = f"{(n_anomalies / total * 100):.2f}%"

        logger.info(f"Anomaly detection metrics: {n_anomalies} anomalies found")
        return metrics

    @staticmethod
    def cross_validation_score(model: Any, X: np.ndarray, y: np.ndarray, 
                              cv: int = 5, task_type: str = "classification") -> Dict[str, float]:
        """
        Compute cross-validation scores.

        Args:
            model: Sklearn model
            X: Features
            y: Target
            cv: Number of CV folds
            task_type: "classification" or "regression"

        Returns:
            Dictionary with CV metrics
        """
        from sklearn.model_selection import cross_val_score
        
        metrics = {}
        
        if task_type == "classification":
            scoring = ["accuracy", "f1_weighted"]
        else:
            scoring = ["r2", "neg_mean_absolute_error"]
        
        for score in scoring:
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=score)
                metrics[f"{score}_mean"] = cv_scores.mean()
                metrics[f"{score}_std"] = cv_scores.std()
            except Exception as e:
                logger.warning(f"CV score {score} failed: {e}")
        
        logger.info(f"Cross-validation completed: {list(metrics.keys())}")
        return metrics
