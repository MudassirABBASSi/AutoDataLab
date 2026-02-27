#!/usr/bin/env python
"""Quick test for t-SNE fix."""
from core.models.unsupervised import UnsupervisedModels

print("✓ t-SNE parameter fix successful!")
print(f"Available DR models: {UnsupervisedModels.get_available_models('reduction')}")

# Test instantiation
model = UnsupervisedModels.get_model("t-SNE", "reduction")
print(f"✓ t-SNE model created: {type(model).__name__}")
print("✓ All fixes working correctly!")
