"""
Configuration settings for AutoDataLab.
Centralized configuration management.
"""

# Application Settings
APP_NAME = "AutoDataLab"
APP_DESCRIPTION = "Automated Data Analysis and Feature Engineering Laboratory"
APP_VERSION = "1.0.0"

# Data Processing Settings
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# Feature Engineering Settings
ONE_HOT_DROP = 'first'
SCALING_STRATEGY = 'standard'  # 'standard' or 'minmax'

# Feature Selection Settings
VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95
K_BEST_FEATURES = 10
FEATURE_IMPORTANCE_PERCENTILE = 50

# Data Cleaning Settings
MISSING_VALUE_STRATEGY = 'drop'  # 'drop', 'mean', 'median', 'mode'
OUTLIER_IQR_MULTIPLIER = 1.5

# Visualization Settings
DEFAULT_FIGSIZE = (12, 6)
PLOT_COLOR_PALETTE = 'husl'
HISTOGRAM_BINS = 30

# File Upload Settings
ALLOWED_FILE_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE_MB = 100

# Logging Settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Caching Settings
CACHE_ENABLED = True
CACHE_TTL = 3600  # Time to live in seconds (1 hour)
MAX_CACHE_SIZE_MB = 500  # Maximum cache size in MB

# Performance Settings
OPTIMIZE_MEMORY = True
ENABLE_PROFILING = False  # Set to True for performance debugging
