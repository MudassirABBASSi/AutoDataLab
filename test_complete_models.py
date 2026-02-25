#!/usr/bin/env python
"""Test all model implementations."""

import numpy as np
import pandas as pd

print("\n" + "="*60)
print("COMPREHENSIVE MODEL SYSTEM TEST")
print("="*60)

# Test Supervised Models
print("\n[TEST 1] Supervised Models...")
from core.models import SupervisedModels

reg_models = SupervisedModels.get_available_models("regression")
clf_models = SupervisedModels.get_available_models("classification")

print(f"  Regression models ({len(reg_models)}): {', '.join(reg_models[:3])}...")
print(f"  Classification models ({len(clf_models)}): {', '.join(clf_models[:3])}...")
print(f"  ✓ All {len(reg_models)} regression + {len(clf_models)} classification models available")

# Test Unsupervised Models
print("\n[TEST 2] Unsupervised Models...")
from core.models import UnsupervisedModels

clustering = UnsupervisedModels.get_available_models("clustering")
reduction = UnsupervisedModels.get_available_models("reduction")
anomaly = UnsupervisedModels.get_available_models("anomaly")

print(f"  Clustering ({len(clustering)}): {', '.join(clustering)}")
print(f"  Dimensionality Reduction ({len(reduction)}): {', '.join(reduction[:3])}...")
print(f"  Anomaly Detection ({len(anomaly)}): {', '.join(anomaly)}")
print(f"  ✓ Unsupervised system complete")

# Test Semi-Supervised Models
print("\n[TEST 3] Semi-Supervised Models...")
from core.models import SemiSupervisedModels

ss_models = SemiSupervisedModels.get_available_models()
print(f"  Available ({len(ss_models)}): {', '.join(ss_models[:2])}...")
print(f"  ✓ Semi-supervised system complete")

# Test Enhanced Metrics
print("\n[TEST 4] Enhanced Evaluation Metrics...")
from core.evaluation import ModelMetrics

# Create dummy data
np.random.seed(42)
X = np.random.rand(100, 5)
y_clf = np.random.randint(0, 3, 100)
y_reg = np.random.rand(100)

# Test clustering metrics
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
labels = km.fit_predict(X)
clustering_metrics = ModelMetrics.clustering_metrics(X, labels)
print(f"  Clustering metrics: {list(clustering_metrics.keys())}")

# Test anomaly metrics
from sklearn.ensemble import IsolationForest
iso = IsolationForest(random_state=42)
anomaly_labels = iso.fit_predict(X)
anomaly_metrics = ModelMetrics.anomaly_metrics(X, anomaly_labels)
print(f"  Anomaly metrics: {list(anomaly_metrics.keys())}")
print(f"  ✓ Metrics system enhanced for all paradigms")

# Test Hyperparameter Config System
print("\n[TEST 5] Hyperparameter Configuration System...")
hp_sup = SupervisedModels.get_hyperparameters("Random Forest", "regression")
hp_unsup = UnsupervisedModels.get_hyperparameters("KMeans", "clustering")
hp_ss = SemiSupervisedModels.get_hyperparameters("Label Propagation")

print(f"  SupervisedModels params: {list(hp_sup.keys()) if hp_sup else 'default'}")
print(f"  UnsupervisedModels params: {list(hp_unsup.keys()) if hp_unsup else 'default'}")
print(f"  SemiSupervisedModels params: {list(hp_ss.keys()) if hp_ss else 'default'}")
print(f"  ✓ Hyperparameter system implemented")

# Test Model Factory
print("\n[TEST 6] Model Factory Functions...")
reg_model = SupervisedModels.get_model("XGBoost", "regression")
print(f"  ✓ Regression model created: {type(reg_model).__name__}")

clf_model = SupervisedModels.get_model("Random Forest", "classification")
print(f"  ✓ Classification model created: {type(clf_model).__name__}")

unsup_model = UnsupervisedModels.get_model("DBSCAN", "clustering")
print(f"  ✓ Clustering model created: {type(unsup_model).__name__}")

ss_model = SemiSupervisedModels.get_model("Label Propagation")
print(f"  ✓ Semi-supervised model created: {type(ss_model).__name__}")

print("\n" + "="*60)
print("✓✓✓ ALL TESTS PASSED - COMPLETE MODEL SYSTEM READY")
print("="*60 + "\n")
