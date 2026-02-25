# AutoDataLab 📊

**Automated Data Analysis and Feature Engineering Laboratory**

A comprehensive Streamlit-based application for end-to-end data science workflows. AutoDataLab provides intuitive interfaces for data loading, profiling, cleaning, visualization, feature engineering, and feature selection.

## Features ✨

- **Data Loading** 📁: Load CSV and Excel files with validation
- **Data Profiling** 📈: Generate statistical summaries, correlation analysis, missing value detection
- **Data Cleaning** 🧹: Handle missing values, remove duplicates, detect outliers using IQR
- **EDA Visualization** 🎨: Create histograms, boxplots, scatterplots, heatmaps, and distribution grids
- **Feature Engineering** 🔧: One-hot encoding, label encoding, standard scaling, MinMax scaling, date feature extraction
- **Feature Selection** ✨: Variance thresholding, correlation analysis, SelectKBest, tree-based importance
- **Complete Pipeline** 📊: Automated end-to-end data processing workflow

## Architecture 🏗️

```
AutoDataLab/
├── app.py                      # Main Streamlit application (UI only)
├── core/                        # Business logic modules
│   ├── data_loader.py          # Load and validate data
│   ├── profiler.py             # Statistical profiling
│   ├── cleaning.py             # Data cleaning operations
│   ├── eda.py                  # Visualization module
│   ├── feature_engineering.py  # Feature transformation
│   └── feature_selection.py    # Feature selection techniques
├── config/
│   └── settings.py             # Configuration management
├── utils/
│   └── helpers.py              # Utility functions
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

## Installation 🚀

### 1. Clone or create the project directory
```bash
cd AutoData
```

### 2. Create and activate virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage 🎯

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Workflow Example

1. **Load Data**: Upload your CSV/Excel file or use sample data
2. **Profile**: Analyze data structure, statistics, and quality metrics
3. **Clean**: Handle missing values, remove duplicates, detect outliers
4. **Visualize**: Create charts to understand patterns and distributions
5. **Engineer**: Transform features with encoding and scaling
6. **Select**: Identify the most important features
7. **Export**: Download processed data

## Core Modules 📚

### DataLoader
Load and validate data files (CSV, Excel) with error handling.

```python
from core import DataLoader

loader = DataLoader()
df = loader.load('data.csv')
print(loader.get_shape())
print(loader.get_missing_summary())
```

### DataProfiler
Generate comprehensive statistical profiles of datasets.

```python
from core import DataProfiler

profiler = DataProfiler(df)
profile = profiler.generate_profile()
print(profiler.get_statistical_summary())
```

### DataCleaner
Handle missing values, duplicates, and outliers.

```python
from core import DataCleaner

cleaner = DataCleaner(df)
df_clean = cleaner.handle_missing_values(strategy='drop')
df_clean = cleaner.remove_duplicates()
df_clean = cleaner.remove_outliers_iqr()
```

### FeatureEngineer
Transform features using encoding and scaling.

```python
from core import FeatureEngineer

engineer = FeatureEngineer(df)
df_transformed = engineer.one_hot_encode(['city', 'category'])
df_transformed = engineer.standard_scale(['age', 'salary'])
df_transformed = engineer.extract_date_features('date_column')
```

### FeatureSelector
Select important features using multiple techniques.

```python
from core import FeatureSelector

selector = FeatureSelector(df, target_column='target')
selected = selector.variance_threshold(0.01)
selected = selector.correlation_threshold(0.95)
result = selector.select_k_best(k=10, score_func='f_classif')
```

### EDAVisualizer
Create exploratory data analysis visualizations.

```python
from core import EDAVisualizer

visualizer = EDAVisualizer(df)
fig = visualizer.histogram('age')
fig = visualizer.boxplot(['age', 'salary'])
fig = visualizer.scatterplot('experience', 'salary')
fig = visualizer.correlation_heatmap()
```

## Configuration ⚙️

Edit `config/settings.py` to customize:

- Default thresholds for feature selection
- Scaling methods and parameters
- File upload limits
- Logging levels
- Visualization defaults

## Data Processing Pipeline 🔄

The complete pipeline performs these steps in sequence:

1. **Missing Values** → Drop or impute using mean/median/mode
2. **Duplicates** → Remove duplicate rows
3. **Outliers** → Remove using Interquartile Range (IQR)
4. **Encoding** → Convert categorical to numerical
5. **Scaling** → Normalize/standardize numerical features
6. **Selection** → Identify and remove low-value features

## Best Practices 💡

- Always profile your data first to understand its structure
- Handle missing values before outlier detection
- Scale numerical features before machine learning models
- Use feature selection to improve model performance
- Export processed data for use in other tools

## Troubleshooting 🔧

### App won't start
```bash
# Reinstall streamlit
pip install --upgrade streamlit
```

### Import errors
```bash
# Verify all dependencies are installed
pip install -r requirements.txt
```

### Memory issues with large files
- Work with subsets of data
- Increase system memory or use data sampling
- Process in batches

## Contributing 🤝

To add new features or modules:

1. Implement logic in `core/` directory
2. Export in `core/__init__.py`
3. Add UI in appropriate section of `app.py`
4. Update documentation

## Dependencies 📦

- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning tools
- **matplotlib** - Visualization library
- **seaborn** - Statistical visualization
- **streamlit** - Web app framework
- **openpyxl** - Excel file handling

## License 📄

This project is provided as-is for educational and commercial use.

## Support 📞

For issues or questions, please check:
- Module docstrings in `core/` files
- Configuration options in `config/settings.py`
- Examples in module `if __name__ == "__main__"` sections

---

**Version**: 1.0.0  
**Last Updated**: 2026-02-18  
**Built with**: Streamlit • Pandas • Scikit-learn • Matplotlib • Seaborn
