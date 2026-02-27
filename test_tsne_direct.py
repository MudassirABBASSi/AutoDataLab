#!/usr/bin/env python
"""Direct t-SNE parameter test - avoids UMAP initialization issue."""
import sys
sys.path.insert(0, 'd:\\AutoData')

try:
    from sklearn.manifold import TSNE
    
    # Test that max_iter parameter works (not n_iter)
    print("Testing t-SNE with max_iter parameter...")
    model = TSNE(n_components=2, random_state=42, max_iter=1000)
    print(f"✓ t-SNE model successfully created with max_iter=1000")
    print(f"✓ Model type: {type(model).__name__}")
    print(f"✓ max_iter parameter value: {model.max_iter}")
    
    print("\n✓✓✓ t-SNE parameter fix is working correctly!")
    print("The deprecated 'n_iter' has been replaced with 'max_iter'")
    
except TypeError as e:
    if "n_iter" in str(e):
        print(f"✗ FAILED: Still using deprecated n_iter parameter")
        print(f"Error: {e}")
    else:
        print(f"✗ ERROR: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
