#!/usr/bin/env python
"""Quick test of modeling module imports."""

from core.models import SupervisedModels
from core.evaluation import ModelMetrics, MetricsVisualizer
from core.pipeline import ModelTrainer

print("✓ All modeling modules imported successfully!")
print(f"✓ Available regression models: {SupervisedModels.get_available_models('regression')}")
print(f"✓ Available classification models: {SupervisedModels.get_available_models('classification')}")
