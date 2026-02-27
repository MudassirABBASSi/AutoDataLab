"""
Metrics Visualization module.
Provides visualization for model evaluation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, Tuple
from utils.logger import get_logger
from utils.exceptions import VisualizationError

logger = get_logger(__name__)


class MetricsVisualizer:
    """Visualize model evaluation metrics."""

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, labels: list = None) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.

        Args:
            cm: Confusion matrix array
            labels: Class labels (optional)

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        if labels is None:
            labels = [f"Class {i}" for i in range(cm.shape[0])]

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=labels, yticklabels=labels, ax=ax
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        logger.info("Confusion matrix plot created")
        return fig

    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float = None) -> plt.Figure:
        """
        Plot ROC curve.

        Args:
            fpr: False positive rate
            tpr: True positive rate
            auc: AUC score (optional, for display)

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc:.3f})" if auc else "ROC Curve")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        logger.info("ROC curve plot created")
        return fig

    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
        """
        Plot top N feature importances.

        Args:
            importance_df: DataFrame with 'Feature' and 'Importance' columns
            top_n: Number of top features to display

        Returns:
            matplotlib Figure
        """
        if importance_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Feature importance not available for this model",
                   ha="center", va="center", fontsize=12)
            return fig

        df_top = importance_df.head(top_n)
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

        ax.barh(df_top["Feature"], df_top["Importance"], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importance")
        ax.invert_yaxis()

        logger.info(f"Feature importance plot created (top {top_n})")
        return fig

    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """
        Plot residuals for regression.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            matplotlib Figure
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residuals vs predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, color="steelblue")
        axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Predicted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Residuals")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of Residuals")
        axes[1].grid(True, alpha=0.3, axis="y")

        logger.info("Residuals plot created")
        return fig

    @staticmethod
    def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """
        Plot predicted vs actual values (for regression).

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(y_true, y_pred, alpha=0.5, color="steelblue")

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Predicted vs Actual Values")
        ax.legend()
        ax.grid(True, alpha=0.3)

        logger.info("Prediction vs actual plot created")
        return fig
