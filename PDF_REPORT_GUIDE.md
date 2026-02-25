# PDF Report Generator - Usage Guide

## Overview
The PDF Report Generator creates comprehensive, professional PDF reports containing:
- Dataset summary and statistics
- Missing values analysis with visualizations
- Distribution analysis for numeric features
- Correlation matrix and top correlations
- Feature importance (when available)
- Model performance metrics and visualizations

## Installation

The required dependencies are already added to `requirements.txt`:
```bash
pip install reportlab Pillow
```

Or install from the requirements file:
```bash
pip install -r requirements.txt
```

## Usage in Streamlit App

### Step 1: Load and Process Data
1. Go to **Upload** tab and load your dataset
2. Clean and prepare your data in the **Cleaning** tab
3. Optionally engineer features in **Feature Engineering** tab

### Step 2: Train Model (Optional)
1. Navigate to **Modeling** tab
2. Select target column and model type
3. Click **Train Model**
4. Model metrics and feature importance will be automatically stored for the report

### Step 3: Generate PDF Report
1. Go to **Export** tab
2. Review the current data preview
3. Click **Generate PDF Report** button
4. Wait for the report to be generated (may take 10-30 seconds)
5. Click **📥 Download PDF Report** to download

### What's Included in the Report

#### With Model (after training):
- ✅ Cover page with dataset info
- ✅ Dataset summary and data types
- ✅ Missing values analysis with bar plot
- ✅ Distribution analysis (histograms & boxplots)
- ✅ Correlation matrix heatmap
- ✅ Top 5 correlated feature pairs
- ✅ Feature importance bar plot (top 10 features)
- ✅ Model performance metrics
- ✅ Classification: Confusion matrix, ROC curve
- ✅ Regression: Prediction vs Actual scatter plot

#### Without Model (EDA only):
- ✅ Cover page with dataset info
- ✅ Dataset summary and data types
- ✅ Missing values analysis with bar plot
- ✅ Distribution analysis (histograms & boxplots)
- ✅ Correlation matrix heatmap
- ✅ Top 5 correlated feature pairs

## Programmatic Usage

### Basic Usage (EDA Report Only)
```python
from reporting import generate_eda_report
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Generate report
report_path = generate_eda_report(
    df=df,
    output_dir="reports",
    filename="my_report.pdf"
)
print(f"Report saved to: {report_path}")
```

### Advanced Usage (With Model Metrics)
```python
from reporting import generate_eda_report
import pandas as pd
import numpy as np

# Your trained model and data
df = pd.read_csv("your_data.csv")
model = your_trained_model  # Any scikit-learn model

# Prepare metrics for classification
model_metrics = {
    'accuracy': 0.85,
    'precision': 0.82,
    'recall': 0.88,
    'f1': 0.85,
    'confusion_matrix': confusion_matrix,  # numpy array
    'roc_curve': (fpr, tpr, thresholds)  # tuple of arrays
}

# Feature importance DataFrame
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance_values
}).sort_values('importance', ascending=False)

# Generate comprehensive report
report_path = generate_eda_report(
    df=df,
    model=model,
    model_metrics=model_metrics,
    feature_importance=feature_importance,
    output_dir="reports",
    filename="full_report.pdf"
)
```

### Regression Example
```python
# Regression metrics
model_metrics = {
    'mae': 150.5,
    'rmse': 200.3,
    'r2': 0.85,
    'y_true': y_test,  # actual values (numpy array)
    'y_pred': y_pred   # predicted values (numpy array)
}

report_path = generate_eda_report(
    df=df,
    model=model,
    model_metrics=model_metrics,
    feature_importance=feature_importance
)
```

## Function Signature

```python
def generate_eda_report(
    df: pd.DataFrame,
    model: Any = None,
    model_metrics: Optional[Dict[str, float]] = None,
    feature_importance: Optional[pd.DataFrame] = None,
    output_dir: str = "reports",
    filename: str = None
) -> str:
    """
    Generate comprehensive EDA PDF report with model performance.
    
    Args:
        df: Input DataFrame to analyze
        model: Trained model object (optional)
        model_metrics: Dictionary of model evaluation metrics (optional)
        feature_importance: DataFrame with columns ['feature', 'importance'] (optional)
        output_dir: Directory to save the report (default: "reports")
        filename: Custom filename (default: AutoDataLab_Report_YYYYMMDD_HHMMSS.pdf)
        
    Returns:
        str: Path to the generated PDF report
    """
```

## Model Metrics Format

### Classification Metrics
```python
{
    'accuracy': float,
    'precision': float,
    'recall': float,
    'f1': float,
    'roc_auc': float,  # optional
    'confusion_matrix': np.ndarray,  # 2D array
    'roc_curve': (fpr, tpr, thresholds)  # optional tuple
}
```

### Regression Metrics
```python
{
    'mae': float,
    'rmse': float,
    'r2': float,
    'y_true': np.ndarray,  # actual values for scatter plot
    'y_pred': np.ndarray   # predicted values for scatter plot
}
```

## Customization

### Change Output Directory
```python
report_path = generate_eda_report(
    df=df,
    output_dir="custom/path/to/reports"
)
```

### Custom Filename
```python
report_path = generate_eda_report(
    df=df,
    filename="Q4_2026_Analysis.pdf"
)
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'reportlab'
**Solution:**
```bash
pip install reportlab Pillow
```

### Issue: Large datasets causing slow report generation
**Solution:**
- The report automatically limits distributions to top 5 numeric columns
- Correlation matrices handle large column counts efficiently
- For very large datasets (>1M rows), consider sampling:
```python
df_sample = df.sample(n=100000, random_state=42)
report_path = generate_eda_report(df=df_sample)
```

### Issue: Report generation fails with specific data
**Solution:**
- Ensure DataFrame has no all-NaN columns
- Check that model_metrics dictionary has correct structure
- Verify feature_importance DataFrame has correct column names

## Performance Tips

1. **Pre-process data**: Clean and prepare data before generating report
2. **Sample large datasets**: Use df.sample() for datasets >1M rows
3. **Limit features**: For datasets with >50 columns, select relevant features first
4. **Use appropriate dtypes**: Ensure columns have correct data types

## Examples

See `test_pdf_report.py` for complete working examples.

## Support

For issues or questions:
1. Check the error message in the Streamlit app
2. Review the logs in the console
3. Verify all dependencies are installed
4. Ensure data is properly formatted

## Version Information

- reportlab>=4.0.0
- Pillow>=10.0.0
- pandas>=2.0.0
- matplotlib>=3.8.0
- seaborn>=0.13.0
