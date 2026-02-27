"""
AutoDataLab with Authentication - Example Integration
This shows how to add authentication to your existing app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Import core modules including authentication
from core import (
    DataLoader, DataValidator, validate_dataframe,
    DataProfiler, DataCleaner,
    FeatureEngineer,
    FeatureSelector,
    EDAVisualizer,
    bivariate_analysis,
    univariate_analysis,
    multivariate_analysis,
    detect_column_types,
    column_type_summary,
    apply_log_transform,
    apply_polynomial,
    apply_binning,
    apply_interactions,
    apply_frequency_encoding,
    apply_target_encoding,
    SupervisedModels,
    UnsupervisedModels,
    SemiSupervisedModels,
    ModelMetrics,
    MetricsVisualizer,
    ModelTrainer,
    # Authentication imports
    init_session_state,
    is_authenticated,
    get_current_user,
    login_page,
    show_user_profile,
    user_management_page,
)

# Import configuration and utilities
from config import settings
from utils import helpers
from utils.logger import get_logger, setup_logger
from utils.exceptions import AutoDataLabException, handle_exception
from visualization import PlotGenerator, apply_theme
from reporting import ReportGenerator

# Setup centralized logging
logger = setup_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="AutoDataLab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize authentication session state
init_session_state()

# Professional Color Scheme
COLORS = {
    "primary": "#1F3A8A",
    "secondary": "#334155",
    "background": "#F8FAFC",
    "accent": "#0F766E",
    "text": "#0F172A",
    "border": "#E2E8F0",
    "success": "#10B981",
    "error": "#EF4444"
}

WORKFLOW_STEPS = [
    "Upload",
    "Cleaning",
    "EDA",
    "Feature Engineering",
    "Feature Selection",
    "Modeling",
    "Export"
]

# ============= AUTHENTICATION CHECK =============
# If not authenticated, show login page and stop
if not is_authenticated():
    login_page()
    st.stop()  # This prevents the rest of the app from loading
# ================================================

# If we reach here, user is authenticated
# Show user profile in sidebar
show_user_profile()

# Rest of your styling and app code...
st.markdown(f"""
    <style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    html {{
        scroll-behavior: smooth;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    
    html, body {{
        background-color: {COLORS['background']};
        color: {COLORS['text']};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }}
    
    .main {{
        padding: 2rem 3rem;
        background-color: {COLORS['background']};
    }}
    
    h1 {{
        color: {COLORS['primary']};
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    </style>
""", unsafe_allow_html=True)

# Get current user info
user = get_current_user()

# ============= SIDEBAR NAVIGATION =============
with st.sidebar:
    st.title("🔬 AutoDataLab")
    st.caption("Automated Data Science Platform")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Data Analysis", "👥 User Management"] if user['role'] == 'admin' else ["🏠 Data Analysis"],
        label_visibility="collapsed"
    )

# ============= MAIN APP LOGIC =============
if page == "🏠 Data Analysis":
    # Your existing app.py content goes here
    st.title("🔬 AutoDataLab - Data Analysis")
    st.markdown(f"Welcome, **{user['full_name']}**!")
    
    # Initialize session state for data
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    
    # Your existing workflow steps
    workflow_step = st.sidebar.selectbox(
        "Workflow Step",
        WORKFLOW_STEPS,
        index=0
    )
    
    # Example: Upload step
    if workflow_step == "Upload":
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel"
        )
        
        if uploaded_file:
            try:
                loader = DataLoader()
                df = loader.load_file(uploaded_file)
                st.session_state.df = df
                
                st.success(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
                
                with st.expander("🔍 Preview Data", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                with st.expander("📊 Dataset Info"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows", df.shape[0])
                    col2.metric("Columns", df.shape[1])
                    col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Add your other workflow steps here...
    elif workflow_step == "Cleaning":
        st.header("🧹 Data Cleaning")
        if st.session_state.df is not None:
            st.info("Data cleaning features go here...")
            # Add your cleaning logic
        else:
            st.warning("Please upload data first")
    
    elif workflow_step == "EDA":
        st.header("📊 Exploratory Data Analysis")
        if st.session_state.df is not None:
            st.info("EDA features go here...")
            # Add your EDA logic
        else:
            st.warning("Please upload data first")
    
    # ... continue with other steps

elif page == "👥 User Management":
    # Only admins can access this page
    if user['role'] == 'admin':
        user_management_page()
    else:
        st.error("Access denied. Admin privileges required.")

# ============= FOOTER =============
st.markdown("---")
st.caption("AutoDataLab v1.0 | Automated Data Science Platform")
