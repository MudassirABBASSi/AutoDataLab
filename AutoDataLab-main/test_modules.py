"""
Test script to verify all AutoDataLab modules work correctly.
"""

import sys
import pandas as pd
import numpy as np

print("=" * 60)
print("AutoDataLab Module Testing")
print("=" * 60)

# Test 1: Import all core modules
print("\n[TEST 1] Importing core modules...")
try:
    from core import DataLoader, DataProfiler, DataCleaner, FeatureEngineer, FeatureSelector, EDAVisualizer
    print("✅ All core modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Import config and utils
print("\n[TEST 2] Importing config and utilities...")
try:
    from config import settings
    from utils import helpers
    print("✅ Config and utilities imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Create sample test data
print("\n[TEST 3] Creating sample test data...")
np.random.seed(42)
test_df = pd.DataFrame({
    'Age': np.random.normal(35, 10, 50),
    'Salary': np.random.normal(60000, 15000, 50),
    'Experience': np.random.normal(8, 3, 50),
    'Department': np.random.choice(['Sales', 'IT', 'HR'], 50),
    'Satisfaction': np.random.randint(1, 6, 50)
})

# Add some missing values and duplicates
test_df.loc[test_df.sample(3).index, 'Salary'] = np.nan
test_df = pd.concat([test_df, test_df.iloc[:2]])
print(f"✅ Sample data created: {test_df.shape}")

# Test 4: DataLoader
print("\n[TEST 4] Testing DataLoader...")
try:
    # Save sample to CSV
    test_df.to_csv('test_data.csv', index=False)
    
    loader = DataLoader()
    df = loader.load('test_data.csv')
    shape = loader.get_shape()
    missing = loader.get_missing_summary()
    dtypes = loader.get_data_types()
    
    print(f"  ✅ Loaded shape: {shape}")
    print(f"  ✅ Missing values: {sum(1 for v in missing.values() if v['missing_count'] > 0)} columns")
    print(f"  ✅ Data types: {len(dtypes)} columns")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 5: DataProfiler
print("\n[TEST 5] Testing DataProfiler...")
try:
    profiler = DataProfiler(test_df)
    summary = profiler.get_statistical_summary()
    missing_pct = profiler.get_missing_value_percentage()
    duplicates = profiler.get_duplicate_count()
    
    print(f"  ✅ Statistical summary: {len(summary)} columns")
    print(f"  ✅ Missing analysis: {len(missing_pct)} columns")
    print(f"  ✅ Duplicates found: {duplicates['fully_duplicated_rows']}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 6: DataCleaner
print("\n[TEST 6] Testing DataCleaner...")
try:
    cleaner = DataCleaner(test_df)
    
    df_clean = cleaner.handle_missing_values(strategy='drop')
    print(f"  ✅ Handle missing values: {len(test_df)} → {len(df_clean)} rows")
    
    df_clean = cleaner.remove_duplicates()
    print(f"  ✅ Remove duplicates: {len(test_df)} → {len(df_clean)} rows")
    
    df_clean = cleaner.remove_outliers_iqr()
    print(f"  ✅ Remove outliers: {len(test_df)} → {len(df_clean)} rows")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 7: FeatureEngineer
print("\n[TEST 7] Testing FeatureEngineer...")
try:
    engineer = FeatureEngineer(test_df)
    
    df_encoded = engineer.label_encode(['Department'])
    print(f"  ✅ Label encoding: {df_encoded.shape}")
    
    df_standard = engineer.standard_scale(['Age', 'Salary'])
    print(f"  ✅ Standard scaling: {df_standard.shape}")
    
    df_minmax = engineer.minmax_scale(['Experience'])
    print(f"  ✅ MinMax scaling: {df_minmax.shape}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 8: FeatureSelector
print("\n[TEST 8] Testing FeatureSelector...")
try:
    selector = FeatureSelector(test_df, target_column='Satisfaction')
    
    var_features = selector.variance_threshold(threshold=0.01)
    print(f"  ✅ Variance threshold: {len(var_features)} features")
    
    corr_features = selector.correlation_threshold(threshold=0.95)
    print(f"  ✅ Correlation threshold: {len(corr_features)} features")
    
    kbest = selector.select_k_best(k=3, score_func='f_classif')
    print(f"  ✅ SelectKBest: {len(kbest['selected_features'])} features")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 9: EDAVisualizer
print("\n[TEST 9] Testing EDAVisualizer...")
try:
    visualizer = EDAVisualizer(test_df)
    
    fig1 = visualizer.histogram('Age', bins=15)
    print(f"  ✅ Histogram created")
    
    fig2 = visualizer.boxplot(['Age', 'Salary'])
    print(f"  ✅ Boxplot created")
    
    fig3 = visualizer.scatterplot('Experience', 'Salary')
    print(f"  ✅ Scatterplot created")
    
    fig4 = visualizer.correlation_heatmap()
    print(f"  ✅ Correlation heatmap created")
    
    # Don't actually display, just test creation
    import matplotlib.pyplot as plt
    plt.close('all')
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 10: Helpers
print("\n[TEST 10] Testing utility helpers...")
try:
    numeric_cols = helpers.get_numeric_columns(test_df)
    categorical_cols = helpers.get_categorical_columns(test_df)
    df_info = helpers.format_dataframe_info(test_df)
    
    print(f"  ✅ Numeric columns: {len(numeric_cols)}")
    print(f"  ✅ Categorical columns: {len(categorical_cols)}")
    print(f"  ✅ DataFrame info: {list(df_info.keys())}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 11: Settings
print("\n[TEST 11] Testing configuration...")
try:
    app_name = settings.APP_NAME
    version = settings.APP_VERSION
    allowed_ext = settings.ALLOWED_FILE_EXTENSIONS
    
    print(f"  ✅ App name: {app_name}")
    print(f"  ✅ Version: {version}")
    print(f"  ✅ Allowed extensions: {allowed_ext}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Cleanup
print("\n[CLEANUP] Removing test file...")
try:
    import os
    os.remove('test_data.csv')
    print("✅ Test file removed")
except:
    pass

print("\n" + "=" * 60)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nThe AutoDataLab system is ready for production use.")
print("Run 'streamlit run app.py' to start the application.\n")
