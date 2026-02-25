# AutoDataLab - Production-Ready Architecture

## Overview
AutoDataLab has been refactored into a modular, production-ready architecture with clear separation of concerns, centralized logging, custom exception handling, and performance optimization.

## Architecture

### Directory Structure
```
autodatalab/
├── app.py                      # Streamlit UI (UI logic only)
├── config/
│   ├── __init__.py
│   └── settings.py             # Centralized configuration
├── core/                       # Business logic
│   ├── __init__.py
│   ├── data_loader.py          # Data loading & validation
│   ├── data_validator.py       # Data quality checks
│   ├── profiler.py             # Statistical profiling
│   ├── cleaning.py             # Data cleaning operations
│   ├── eda.py                  # Exploratory data analysis
│   ├── feature_engineering.py  # Feature transformation
│   ├── feature_selection.py    # Feature importance & selection
│   ├── models/                 # ML model wrappers
│   │   ├── supervised.py       # 23 supervised models
│   │   ├── unsupervised.py     # 13 unsupervised models
│   │   └── semi_supervised.py  # 4 semi-supervised models
│   ├── evaluation/
│   │   ├── metrics.py          # Model evaluation metrics
│   │   └── visualization.py    # Metrics visualization
│   └── pipeline/
│       └── trainer.py          # Training pipeline
├── visualization/              # Plot utilities
│   ├── __init__.py
│   ├── plots.py                # Centralized plotting functions
│   └── themes.py               # Color themes
├── reporting/                  # Report generation
│   ├── __init__.py
│   └── report_generator.py     # HTML/JSON/TXT reports
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── logger.py               # Centralized logging
│   ├── exceptions.py           # Custom exceptions
│   ├── caching.py              # Caching layer
│   └── helpers.py              # Helper functions
└── logs/                       # Application logs
    ├── application.log         # All logs
    └── errors.log              # Error logs only
```

## Key Features

### 1. Centralized Logging
- **Location**: `utils/logger.py`
- **Features**:
  - Rotating file handlers (10MB max, 5 backups)
  - Separate error log file
  - Structured log format with timestamps
  - Different log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

**Usage**:
```python
from utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
logger.error("An error occurred")
```

### 2. Custom Exception Handling
- **Location**: `utils/exceptions.py`
- **Exception Classes**:
  - `DataValidationError` - Invalid data/parameters
  - `DataLoadingError` - File loading failures
  - `DataCleaningError` - Data cleaning failures
  - `FeatureEngineeringError` - Feature transformation failures
  - `FeatureSelectionError` - Feature selection failures
  - `ModelingError` - Model training/prediction failures
  - `EvaluationError` - Metrics computation failures
  - `PipelineError` - Pipeline execution failures
  - `CachingError` - Cache operation failures
  - `VisualizationError` - Plot generation failures
  - `ReportingError` - Report generation failures

**Usage**:
```python
from utils.exceptions import DataValidationError, handle_exception

try:
    validate_data(df)
except DataValidationError as e:
    logger.error(f"Validation failed: {e}")
    raise
```

### 3. Caching Layer
- **Location**: `utils/caching.py`
- **Features**:
  - Disk-based caching with pickle
  - Streamlit session caching
  - DataFrame content hashing
  - Cache invalidation on data changes
  - Cache statistics tracking

**Usage**:
```python
from utils.caching import cache_to_disk, streamlit_cache

@cache_to_disk
def expensive_computation(data):
    return process(data)

@streamlit_cache(ttl=3600)
def load_preprocessed_data(df):
    return df.dropna()
```

### 4. Configuration Management
- **Location**: `config/settings.py`
- **Configurable Parameters**:
  - Random state
  - Test/train split ratios
  - Feature engineering strategies
  - Visualization settings
  - File upload limits
  - Logging configuration
  - Cache settings

### 5. Data Validation
- **Location**: `core/data_validator.py`
- **Features**:
  - DataFrame validation
  - File format validation
  - Column existence checks
  - Missing value detection
  - Duplicate detection
  - Outlier detection (IQR method)
  - Comprehensive quality reports

**Usage**:
```python
from core import DataValidator

validator = DataValidator()
validator.validate_dataframe(df)
quality_report = validator.get_comprehensive_report(df)
```

### 6. Visualization Module
- **Location**: `visualization/`
- **Features**:
  - Centralized plot generation
  - Consistent styling
  - Multiple plot types (histogram, scatter, box, heatmap, etc.)
  - Theme support (default, dark, ocean, forest, sunset)
  - Plot export functionality

**Usage**:
```python
from visualization import PlotGenerator

plot_gen = PlotGenerator()
fig, ax = plot_gen.histogram(df['column'], title="Distribution")
plot_gen.save_plot(fig, "histogram.png")
```

### 7. Report Generation
- **Location**: `reporting/report_generator.py`
- **Features**:
  - Multiple formats: HTML, JSON, Text
  - Professional HTML reports with styling
  - DataFrame summaries
  - Statistics sections
  - Model results sections
  - Export functionality

**Usage**:
```python
from reporting import ReportGenerator

report = ReportGenerator("Data Analysis Report")
report.add_dataframe_section("Input Data", df)
report.add_statistics_section("Statistics", df)
report.save_report("analysis_report", format="html")
```

## Machine Learning Models

### Supervised Learning (23 Models)
**Regression (12)**:
- Linear Regression, Ridge, Lasso, ElasticNet
- Decision Tree, Random Forest, Gradient Boosting
- XGBoost, LightGBM
- SVR, KNN, MLP

**Classification (11)**:
- Logistic Regression, Ridge Classifier
- Decision Tree, Random Forest, Gradient Boosting
- XGBoost, LightGBM
- KNN, SVM, Naive Bayes, MLP

### Unsupervised Learning (13 Models)
**Clustering (4)**:
- KMeans, DBSCAN, Agglomerative, Spectral

**Dimensionality Reduction (4)**:
- PCA, Kernel PCA, t-SNE, UMAP

**Anomaly Detection (3)**:
- Isolation Forest, One-Class SVM, Local Outlier Factor

### Semi-Supervised Learning (4 Models)
- Label Propagation
- Label Spreading
- Self-Training (Logistic Regression)
- Self-Training (Random Forest)

## Performance Optimizations

1. **Memory Efficiency**:
   - Categorical dtype optimization
   - Avoid unnecessary DataFrame copies
   - Vectorized pandas operations

2. **Computation Optimization**:
   - Caching expensive operations
   - Reuse computed matrices
   - Lazy evaluation where possible

3. **UI Optimization**:
   - Streamlit caching for data/results
   - Progress indicators for long operations
   - Background processing where applicable

## Error Handling

### Application-Level
- Global error handler in `app.py`
- User-friendly error messages
- No raw stack traces in UI
- Detailed logging for debugging

### Module-Level
- Custom exceptions with error codes
- Try-except blocks around operations
- Proper error propagation
- Context-aware error messages

## Logging Strategy

### Log Levels
- **DEBUG**: Development/troubleshooting
- **INFO**: Normal operations, milestones
- **WARNING**: Non-critical issues
- **ERROR**: Failures (operation continues)
- **CRITICAL**: Severe failures (operation halts)

### Log Files
- `logs/application.log`: All logs (INFO and above)
- `logs/errors.log`: Errors only (ERROR and above)

### Log Rotation
- 10MB file size limit
- 5 backup files retained
- Automatic rotation

## Configuration Settings

### Key Settings (`config/settings.py`)
```python
# Application
APP_NAME = "AutoDataLab"
APP_VERSION = "1.0.0"

# Data Processing
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# Caching
CACHE_ENABLED = True
CACHE_TTL = 3600
MAX_CACHE_SIZE_MB = 500

# Logging
LOG_LEVEL = 'INFO'

# Performance
OPTIMIZE_MEMORY = True
ENABLE_PROFILING = False
```

## Usage Examples

### Basic Workflow
```python
# 1. Load data
from core import DataLoader
loader = DataLoader()
df = loader.load("data.csv")

# 2. Validate data
from core import DataValidator
validator = DataValidator()
validator.validate_dataframe(df)

# 3. Clean data
from core import DataCleaner
cleaner = DataCleaner(df)
df_clean = cleaner.clean_pipeline(missing_strategy='mean')

# 4. Feature engineering
from core import FeatureEngineer
engineer = FeatureEngineer(df_clean)
df_fe = engineer.scale_features(['age', 'salary'])

# 5. Train model
from core import SupervisedModels, ModelTrainer
model = SupervisedModels.get_model("Random Forest", task_type="classification")
trainer = ModelTrainer(model, df_fe, target_column="target")
results = trainer.train()

# 6. Generate report
from reporting import ReportGenerator
report = ReportGenerator("ML Report")
report.add_model_results_section("Results", "Random Forest", results['metrics'])
report.save_report("ml_report", format="html")
```

## Testing

Run import test:
```bash
python -c "from core import *; from utils import *; from visualization import *; from reporting import *; print('✓ All imports successful')"
```

Run application:
```bash
streamlit run app.py
```

## Best Practices

1. **Always use centralized logger**
   ```python
   from utils.logger import get_logger
   logger = get_logger(__name__)
   ```

2. **Use custom exceptions**
   ```python
   from utils.exceptions import DataValidationError
   raise DataValidationError("Invalid input")
   ```

3. **Cache expensive operations**
   ```python
   from utils.caching import cache_to_disk
   @cache_to_disk
   def expensive_function():
       pass
   ```

4. **Validate inputs**
   ```python
   from core import DataValidator
   DataValidator.validate_dataframe(df)
   ```

5. **Don't expose raw errors in UI**
   ```python
   try:
       operation()
   except Exception as e:
       user_msg = handle_exception(e, logger)
       st.error(user_msg)
   ```

## Deployment

### Development
```bash
streamlit run app.py
```

### Production
Configure `config/settings.py`:
- Set `LOG_LEVEL = 'WARNING'`
- Set `ENABLE_PROFILING = False`
- Set appropriate `CACHE_TTL`

### Docker (Future)
Ready for containerization with clean architecture.

### Cloud Deployment (Future)
Architecture supports deployment to:
- Streamlit Cloud
- AWS/GCP/Azure
- Heroku

## Maintenance

### Adding New Features
1. Add business logic to appropriate `core/` module
2. Use centralized logger and exceptions
3. Update `core/__init__.py` exports
4. Add UI in `app.py` (UI logic only)
5. Update documentation

### Monitoring
- Check `logs/application.log` for operations
- Check `logs/errors.log` for failures
- Monitor cache size: `get_cache_stats()`

### Troubleshooting
1. Check error logs
2. Enable DEBUG logging
3. Clear cache if needed: `clear_disk_cache()`
4. Verify configuration settings

## Summary

AutoDataLab is now a production-ready application with:
- ✓ Modular architecture
- ✓ Centralized logging
- ✓ Custom exception handling
- ✓ Performance optimization
- ✓ Caching layer
- ✓ Configuration management
- ✓ Clean separation of concerns
- ✓ Comprehensive ML capabilities (40 models)
- ✓ Professional reporting
- ✓ Data validation
- ✓ Error handling
- ✓ Scalable design

Ready for production deployment and future enhancements.
