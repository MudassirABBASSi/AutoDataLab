#!/usr/bin/env python
"""Direct test of model implementations (bypasses EDA/UMAP issues)."""

import numpy as np
import pandas as pd
import sys

print("\n" + "="*60)
print("COMPREHENSIVE MODEL SYSTEM TEST")
print("="*60)

try:
    # Test Supervised Models
    print("\n[TEST 1] Supervised Models...")
    from core.models.supervised import SupervisedModels

    reg_models = SupervisedModels.get_available_models("regression")
    clf_models = SupervisedModels.get_available_models("classification")

    print(f"  ✓ Regression models ({len(reg_models)}): {reg_models[:3]}")
    print(f"  ✓ Classification models ({len(clf_models)}): {clf_models[:3]}")
    print(f"  SUCCESS: {len(reg_models)} regression + {len(clf_models)} classification")

    # Test Unsupervised Models
    print("\n[TEST 2] Unsupervised Models...")
    from core.models.unsupervised import UnsupervisedModels

    clustering = UnsupervisedModels.get_available_models("clustering")
    reduction = UnsupervisedModels.get_available_models("reduction")
    anomaly = UnsupervisedModels.get_available_models("anomaly")

    print(f"  ✓ Clustering ({len(clustering)}): {clustering}")
    print(f"  ✓ Dimensionality Reduction ({len(reduction)}): {reduction[:3]}")
    print(f"  ✓ Anomaly Detection ({len(anomaly)}): {anomaly}")
    print(f"  SUCCESS: {len(clustering)} + {len(reduction)} + {len(anomaly)} models")

    # Test Semi-Supervised Models
    print("\n[TEST 3] Semi-Supervised Models...")
    from core.models.semi_supervised import SemiSupervisedModels

    ss_models = SemiSupervisedModels.get_available_models()
    print(f"  ✓ Semi-supervised ({len(ss_models)}): {ss_models}")
    print(f"  SUCCESS: {len(ss_models)} semi-supervised algorithms")

    # Test Model Factory
    print("\n[TEST 4] Model Factory Functions...")
    reg_model = SupervisedModels.get_model("XGBoost", "regression")
    print(f"  ✓ Regression model: {type(reg_model).__name__}")
    
    clf_model = SupervisedModels.get_model("Random Forest", "classification")
    print(f"  ✓ Classification model: {type(clf_model).__name__}")
    
    unsup_model = UnsupervisedModels.get_model("KMeans", "clustering")
    print(f"  ✓ Clustering model: {type(unsup_model).__name__}")
    
    unsup_dr = UnsupervisedModels.get_model("PCA", "reduction")
    print(f"  ✓ DR model: {type(unsup_dr).__name__}")
    
    anomaly_model = UnsupervisedModels.get_model("Isolation Forest", "anomaly")
    print(f"  ✓ Anomaly model: {type(anomaly_model).__name__}")
    
    ss_model = SemiSupervisedModels.get_model("Label Propagation")
    print(f"  ✓ Semi-supervised model: {type(ss_model).__name__}")
    print(f"  SUCCESS: All model factory functions working")

    # Test Enhanced Metrics
    print("\n[TEST 5] Enhanced Evaluation Metrics...")
    from core.evaluation.metrics import ModelMetrics
    
    # Create dummy data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y_clf = np.random.randint(0, 3, 100)
    
    # Test clustering metrics
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, random_state=42)
    labels = km.fit_predict(X)
    clustering_metrics = ModelMetrics.clustering_metrics(X, labels)
    print(f"  ✓ Clustering metrics: {list(clustering_metrics.keys())}")
    
    # Test anomaly metrics
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(random_state=42)
    anomaly_labels = iso.fit_predict(X)
    anomaly_metrics = ModelMetrics.anomaly_metrics(X, anomaly_labels)
    print(f"  ✓ Anomaly metrics: {list(anomaly_metrics.keys())}")
    print(f"  SUCCESS: Metrics system enhanced")

    # Test Hyperparameter Config System
    print("\n[TEST 6] Hyperparameter Configuration System...")
    hp_sup = SupervisedModels.get_hyperparameters("Random Forest", "regression")
    hp_unsup = UnsupervisedModels.get_hyperparameters("KMeans", "clustering")
    hp_ss = SemiSupervisedModels.get_hyperparameters("Label Propagation")
    
    print(f"  ✓ Supervised hyperparams: {list(hp_sup.keys())}")
    print(f"  ✓ Unsupervised hyperparams: {list(hp_unsup.keys())}")
    print(f"  ✓ Semi-supervised hyperparams: {list(hp_ss.keys())}")
    print(f"  SUCCESS: Hyperparameter system complete")

    print("\n" + "="*60)
    print("✓✓✓ ALL TESTS PASSED - COMPLETE MODEL SYSTEM READY")
    print("="*60)
    print("\nSystem Inventory:")
    print(f"  - Supervised: {len(reg_models)} regression + {len(clf_models)} classification")
    print(f"  - Unsupervised: {len(clustering)} clustering + {len(reduction)} DR + {len(anomaly)} anomaly")
    print(f"  - Semi-supervised: {len(ss_models)} algorithms")
    print(f"  - Total: {len(reg_models) + len(clf_models) + len(clustering) + len(reduction) + len(anomaly) + len(ss_models)} models/algorithms")
    print("="*60 + "\n")

except Exception as e:
    print(f"\n✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
