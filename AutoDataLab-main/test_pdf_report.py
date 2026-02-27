"""
Test PDF Report Generation
Quick test to verify PDF report functionality works correctly.
"""

import pandas as pd
import numpy as np
from reporting import generate_eda_report

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(18, 80, 100),
    'salary': np.random.randint(30000, 150000, 100),
    'experience': np.random.randint(0, 20, 100),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
    'department': np.random.choice(['IT', 'Sales', 'HR', 'Finance'], 100),
    'performance': np.random.uniform(0, 100, 100)
})

# Add some missing values
df.loc[0:5, 'salary'] = np.nan
df.loc[10:15, 'experience'] = np.nan

print("Testing PDF Report Generation...")
print(f"Dataset shape: {df.shape}")

try:
    # Generate report without model
    report_path = generate_eda_report(
        df=df,
        output_dir="reports",
        filename="test_report.pdf"
    )
    print(f"✓ PDF report generated successfully: {report_path}")
    
    # Test with sample metrics (simulated)
    model_metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1': 0.85
    }
    
    feature_importance = pd.DataFrame({
        'feature': ['age', 'salary', 'experience', 'performance'],
        'importance': [0.35, 0.28, 0.22, 0.15]
    }).sort_values('importance', ascending=False)
    
    report_path_with_model = generate_eda_report(
        df=df,
        model="RandomForest",
        model_metrics=model_metrics,
        feature_importance=feature_importance,
        output_dir="reports",
        filename="test_report_with_model.pdf"
    )
    print(f"✓ PDF report with model metrics generated: {report_path_with_model}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
