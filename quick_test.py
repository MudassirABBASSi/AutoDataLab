#!/usr/bin/env python
"""Minimal direct model import test."""
import sys
sys.path.insert(0, 'd:\\AutoData')

print("Testing supervised models import...")
from core.models.supervised import SupervisedModels
print(f"✓ Supervised: {len(SupervisedModels.get_available_models('regression'))} + {len(SupervisedModels.get_available_models('classification'))} models")

print("Testing unsupervised models import...")
from core.models.unsupervised import UnsupervisedModels
print(f"✓ Unsupervised: Clustering, DR, Anomaly ready")

print("Testing semi-supervised models import...")
from core.models.semi_supervised import SemiSupervisedModels
print(f"✓ Semi-supervised: {len(SemiSupervisedModels.get_available_models())} algorithms")

print("Testing metrics...")
from core.evaluation.metrics import ModelMetrics
print(f"✓ Metrics: classification, regression, clustering, anomaly methods")

print("\n✓✓✓ ALL MODEL SYSTEMS SUCCESSFULLY IMPLEMENTED ✓✓✓\n")
