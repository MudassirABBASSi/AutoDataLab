#!/usr/bin/env python
"""Test semi-supervised models import and fixes."""
import sys
sys.path.insert(0, 'd:\\AutoData')

try:
    print("Testing semi-supervised models import...")
    from core.models.semi_supervised import SemiSupervisedModels
    
    print("✓ SemiSupervisedModels imported successfully!")
    
    # Test getting available models
    models = SemiSupervisedModels.get_available_models()
    print(f"✓ Available semi-supervised models: {models}")
    
    # Test model instantiation
    for model_name in models:
        model = SemiSupervisedModels.get_model(model_name)
        print(f"✓ {model_name}: {type(model).__name__}")
    
    print("\n✓✓✓ All semi-supervised models working correctly!")
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
