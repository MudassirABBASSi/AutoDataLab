"""
Unsupervised Learning Models module.
Provides clustering, dimensionality reduction, and anomaly detection algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from utils.logger import get_logger
from utils.exceptions import ModelingError, DataValidationError

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Dimensionality Reduction
try:
    from sklearn.decomposition import KernelPCA
    KERNEL_PCA_AVAILABLE = True
except ImportError:
    KERNEL_PCA_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logger = get_logger(__name__)


class UnsupervisedModels:
    """
    Wrapper for unsupervised learning algorithms.
    Supports clustering (5 algorithms), dimensionality reduction (4+), and anomaly detection (3).
    """

    CLUSTERING_MODELS = {
        "KMeans": KMeans(n_clusters=3, n_init=10, random_state=42),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "Agglomerative": AgglomerativeClustering(n_clusters=3, linkage="ward"),
        "Spectral": SpectralClustering(n_clusters=3, affinity="nearest_neighbors", random_state=42),
    }

    DIMENSIONALITY_REDUCTION = {
        "PCA": PCA(n_components=2, random_state=42),
        "t-SNE": TSNE(n_components=2, random_state=42, max_iter=1000),
    }

    ANOMALY_DETECTION = {
        "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
        "One-Class SVM": OneClassSVM(kernel="rbf", nu=0.1),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1),
    }

    # Add conditional imports
    if KERNEL_PCA_AVAILABLE:
        DIMENSIONALITY_REDUCTION["Kernel PCA"] = KernelPCA(n_components=2, kernel="rbf", random_state=42)
    
    if UMAP_AVAILABLE:
        DIMENSIONALITY_REDUCTION["UMAP"] = umap.UMAP(n_components=2, random_state=42)

    def __init__(self):
        """Initialize the unsupervised models wrapper."""
        logger.info("UnsupervisedModels initialized with clustering, DR, and anomaly detection")

    @staticmethod
    def get_hyperparameters(model_name: str, model_type: str) -> Dict[str, Any]:
        """
        Get configurable hyperparameters for unsupervised models.
        
        Args:
            model_name: Name of the model
            model_type: "clustering", "reduction", or "anomaly"
            
        Returns:
            Dict with parameter names and configurations
        """
        params = {
            # Clustering
            "KMeans": {"n_clusters": (2, 10, 3), "n_init": (5, 20, 10)},
            "DBSCAN": {"eps": (0.1, 2.0, 0.5), "min_samples": (2, 20, 5)},
            "Agglomerative": {"n_clusters": (2, 10, 3), "linkage": ("ward", "complete", "average")},
            "Spectral": {"n_clusters": (2, 10, 3)},
            # Dimensionality Reduction
            "PCA": {"n_components": (2, 50, 2)},
            "Kernel PCA": {"n_components": (2, 50, 2), "kernel": ("rbf", "poly", "linear")},
            "t-SNE": {"n_components": (2, 3, 2), "perplexity": (5, 50, 30)},
            "UMAP": {"n_components": (2, 50, 2), "n_neighbors": (5, 50, 15)},
            # Anomaly Detection
            "Isolation Forest": {"contamination": (0.01, 0.5, 0.1), "n_estimators": (50, 500, 100)},
            "One-Class SVM": {"nu": (0.01, 0.5, 0.1)},
            "Local Outlier Factor": {"n_neighbors": (5, 50, 20), "contamination": (0.01, 0.5, 0.1)},
        }
        
        return params.get(model_name, {})

    @staticmethod
    def get_model(model_name: str, model_type: str, hyperparameters: Optional[Dict] = None) -> Any:
        """
        Get an unsupervised model instance with optional hyperparameter overrides.
        
        Args:
            model_name: Name of the model
            model_type: "clustering", "reduction", or "anomaly"
            hyperparameters: Dict of hyperparameters to override
            
        Returns:
            Model instance
        """
        hp = hyperparameters or {}
        
        if model_type == "clustering":
            if model_name == "KMeans":
                return KMeans(
                    n_clusters=hp.get("n_clusters", 3),
                    n_init=hp.get("n_init", 10),
                    random_state=42
                )
            elif model_name == "DBSCAN":
                return DBSCAN(
                    eps=hp.get("eps", 0.5),
                    min_samples=hp.get("min_samples", 5)
                )
            elif model_name == "Agglomerative":
                return AgglomerativeClustering(
                    n_clusters=hp.get("n_clusters", 3),
                    linkage=hp.get("linkage", "ward")
                )
            elif model_name == "Spectral":
                return SpectralClustering(
                    n_clusters=hp.get("n_clusters", 3),
                    affinity="nearest_neighbors",
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown clustering model: {model_name}")
        
        elif model_type == "reduction":
            if model_name == "PCA":
                return PCA(n_components=hp.get("n_components", 2), random_state=42)
            elif model_name == "Kernel PCA" and KERNEL_PCA_AVAILABLE:
                return KernelPCA(
                    n_components=hp.get("n_components", 2),
                    kernel=hp.get("kernel", "rbf"),
                    random_state=42
                )
            elif model_name == "t-SNE":
                return TSNE(
                    n_components=hp.get("n_components", 2),
                    perplexity=hp.get("perplexity", 30),
                    random_state=42,
                    max_iter=1000
                )
            elif model_name == "UMAP" and UMAP_AVAILABLE:
                return umap.UMAP(
                    n_components=hp.get("n_components", 2),
                    n_neighbors=hp.get("n_neighbors", 15),
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown dimensionality reduction model: {model_name}")
        
        elif model_type == "anomaly":
            if model_name == "Isolation Forest":
                return IsolationForest(
                    contamination=hp.get("contamination", 0.1),
                    n_estimators=hp.get("n_estimators", 100),
                    random_state=42
                )
            elif model_name == "One-Class SVM":
                return OneClassSVM(kernel="rbf", nu=hp.get("nu", 0.1))
            elif model_name == "Local Outlier Factor":
                return LocalOutlierFactor(
                    n_neighbors=hp.get("n_neighbors", 20),
                    contamination=hp.get("contamination", 0.1)
                )
            else:
                raise ValueError(f"Unknown anomaly detection model: {model_name}")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_available_models(model_type: str) -> list:
        """
        Get list of available models for a given type.
        
        Args:
            model_type: "clustering", "reduction", or "anomaly"
            
        Returns:
            Sorted list of model names
        """
        if model_type == "clustering":
            return sorted(list(UnsupervisedModels.CLUSTERING_MODELS.keys()))
        elif model_type == "reduction":
            return sorted(list(UnsupervisedModels.DIMENSIONALITY_REDUCTION.keys()))
        elif model_type == "anomaly":
            return sorted(list(UnsupervisedModels.ANOMALY_DETECTION.keys()))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
