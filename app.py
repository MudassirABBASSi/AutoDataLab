"""
AutoDataLab - End-to-End Automated Data Science Platform
Professional Streamlit Application

Clean, enterprise-grade interface for data analysis, cleaning, feature engineering, and visualization.
UI logic only - all business logic in core modules.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Import core modules
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

# Apply custom CSS for professional styling
st.markdown(f"""
    <style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    /* Enable smooth scrolling */
    html {{
        scroll-behavior: smooth;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    
    html, body {{
        background-color: {COLORS['background']};
        color: {COLORS['text']};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
    }}
    
    .main {{
        padding: 2rem 3rem;
        background-color: {COLORS['background']};
    }}
    
    [data-testid="stAppViewContainer"] {{
        background-color: {COLORS['background']};
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {COLORS['primary']}05;
        border-right: 1px solid {COLORS['border']};
    }}
    
    [data-testid="stSidebarNav"] {{
        background-color: transparent;
    }}
    
    .stContainer {{
        background-color: white;
        border-radius: 8px;
        border: 1px solid {COLORS['border']};
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}
    
    .card {{
        background-color: white;
        border-radius: 8px;
        border: 1px solid {COLORS['border']};
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: box-shadow 0.2s ease;
    }}
    
    .card:hover {{
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }}
    
    h1 {{
        color: {COLORS['primary']};
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    h2 {{
        color: {COLORS['primary']};
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid {COLORS['accent']};
        padding-bottom: 0.5rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    h3 {{
        color: {COLORS['secondary']};
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        cursor: help;
    }}
    
    /* Tooltip styling */
    h3[title]:hover {{
        color: {COLORS['primary']};
        transition: color 0.2s ease;
    }}
    
    h3[title]::after {{
        content: '';
        display: inline-block;
        width: 0;
        height: 0;
    }}
    
    h4, h5, h6 {{
        color: {COLORS['text']};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    p, span, li {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        line-height: 1.6;
    }}
    
    .stRadio > div {{
        gap: 1rem;
    }}
    
    .stCheckbox > label {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    .stMetric {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid {COLORS['border']};
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}
    
    .stButton > button {{
        background-color: {COLORS['primary']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.75rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.25s ease;
        box-shadow: 0 2px 4px rgba(31, 58, 138, 0.2);
        min-height: 44px;
        cursor: pointer;
        touch-action: manipulation;
        -webkit-tap-highlight-color: transparent;
    }}
    
    .stButton > button:hover {{
        background-color: {COLORS['secondary']};
        box-shadow: 0 6px 14px rgba(31, 58, 138, 0.25);
        transform: translateY(-1px);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(31, 58, 138, 0.15);
    }}
    
    .stSelectbox, .stMultiSelect, .stSlider, .stTextInput, .stTextArea {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background-color: white;
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: thin;
    }}
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
        height: 4px;
    }}
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {{
        background: {COLORS['background']};
        border-radius: 4px;
    }}
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {{
        background: {COLORS['primary']};
        border-radius: 4px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        background-color: transparent;
        color: {COLORS['text']};
        border: none;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {COLORS['primary']}08;
        color: {COLORS['primary']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']} !important;
        color: white !important;
    }}
    
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 2rem;
    }}

    .workflow-progress {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        background-color: white;
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin-bottom: 0.5rem;
        overflow-x: auto;
    }}

    .workflow-step {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        white-space: nowrap;
    }}

    .workflow-step .step-index {{
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background-color: {COLORS['border']};
        color: {COLORS['secondary']};
        font-weight: 700;
        font-size: 0.85rem;
    }}

    .workflow-step .step-label {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {COLORS['secondary']};
    }}

    .workflow-step.completed .step-index {{
        background-color: {COLORS['accent']};
        color: white;
    }}

    .workflow-step.completed .step-label {{
        color: {COLORS['text']};
    }}

    .workflow-step.active .step-index {{
        background-color: {COLORS['primary']};
        color: white;
        box-shadow: 0 0 0 3px {COLORS['primary']}22;
    }}

    .workflow-step.active .step-label {{
        color: {COLORS['primary']};
    }}

    .workflow-connector {{
        height: 2px;
        width: 24px;
        background-color: {COLORS['border']};
        border-radius: 999px;
    }}

    .workflow-connector.completed {{
        background-color: {COLORS['accent']};
    }}

    .workflow-selector [role="radiogroup"] {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem 0.75rem;
        margin-bottom: 1.5rem;
    }}

    .workflow-selector [data-baseweb="radio"] {{
        background-color: white;
        border: 1px solid {COLORS['border']};
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }}

    .workflow-selector input:checked + div {{
        color: {COLORS['primary']};
    }}
    
    .stFileUploader {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    .success-message {{
        background-color: {COLORS['success']}12;
        border-left: 4px solid {COLORS['success']};
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    .error-message {{
        background-color: {COLORS['error']}12;
        border-left: 4px solid {COLORS['error']};
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    .info-box {{
        background-color: {COLORS['accent']}12;
        border-left: 4px solid {COLORS['accent']};
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    .section-divider {{
        margin: 2rem 0;
        border-top: 1px solid {COLORS['border']};
    }}
    
    table {{
        border-collapse: collapse;
        width: 100%;
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        overflow: hidden;
        background-color: white;
        display: block;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }}
    
    tbody {{
        display: table;
        width: 100%;
    }}
    
    thead {{
        background-color: {COLORS['primary']};
        color: white;
    }}
    
    tbody tr {{
        border-bottom: 1px solid {COLORS['border']};
    }}
    
    tbody tr:nth-child(even) {{
        background-color: {COLORS['background']};
    }}
    
    tbody tr:hover {{
        background-color: {COLORS['primary']}08;
    }}
    
    td, th {{
        padding: 0.875rem;
        text-align: left;
        border-right: 1px solid {COLORS['border']};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    th {{
        font-weight: 600;
    }}
    
    /* Dataframe container and styling */
    .dataframe-container {{
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        margin: 1rem 0;
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}
    
    .dataframe {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
        min-width: 600px;
    }}
    
    .dataframe thead th {{
        position: sticky;
        top: 0;
        background-color: {COLORS['primary']};
        color: white;
        z-index: 10;
        font-weight: 600;
        padding: 0.875rem;
    }}
    
    .dataframe tbody tr:nth-child(even) {{
        background-color: {COLORS['background']};
    }}
    
    .dataframe tbody tr:hover {{
        background-color: {COLORS['primary']}08;
    }}
    
    /* Streamlit's native dataframe */
    div[data-testid="stDataFrame"] {{
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }}
    
    div[data-testid="stDataFrame"] > div {{
        max-width: 100%;
    }}
    
    /* ==================== RESPONSIVE MEDIA QUERIES ==================== */
    
    /* Large Desktops (1920px and above) */
    @media (min-width: 1920px) {{
        .main {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        
        h1 {{
            font-size: 2.75rem;
        }}
        
        h2 {{
            font-size: 2rem;
        }}
        
        .stMetric {{
            padding: 2rem;
        }}
    }}
    
    /* Standard Desktops (1024px - 1919px) */
    @media (max-width: 1919px) and (min-width: 1024px) {{
        .main {{
            padding: 2rem;
        }}
        
        h1 {{
            font-size: 2.5rem;
        }}
        
        h2 {{
            font-size: 1.8rem;
        }}
    }}
    
    /* Tablets (768px - 1023px) */
    @media (max-width: 1023px) and (min-width: 768px) {{
        .main {{
            padding: 1.5rem 1rem;
        }}
        
        [data-testid="stSidebar"] {{
            min-width: 250px !important;
        }}
        
        h1 {{
            font-size: 2rem;
        }}
        
        h2 {{
            font-size: 1.5rem;
        }}
        
        h3 {{
            font-size: 1.1rem;
        }}
        
        .stMetric {{
            padding: 1rem;
        }}
        
        .stButton > button {{
            padding: 0.6rem 1.25rem;
            font-size: 0.9rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
        }}
        
        .card {{
            padding: 1rem;
        }}
        
        table td, table th {{
            padding: 0.6rem;
            font-size: 0.9rem;
        }}
    }}
    
    /* Mobile Devices (600px - 767px) */
    @media (max-width: 767px) and (min-width: 600px) {{
        .main {{
            padding: 1rem 0.75rem;
        }}
        
        [data-testid="stSidebar"] {{
            min-width: 200px !important;
        }}
        
        h1 {{
            font-size: 1.75rem;
        }}
        
        h2 {{
            font-size: 1.35rem;
        }}
        
        h3 {{
            font-size: 1rem;
        }}
        
        .stMetric {{
            padding: 0.75rem;
        }}
        
        .stMetric label {{
            font-size: 0.85rem;
        }}
        
        .stMetric [data-testid="stMetricValue"] {{
            font-size: 1.25rem;
        }}
        
        .stButton > button {{
            padding: 0.5rem 1rem;
            font-size: 0.85rem;
            width: 100%;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.25rem;
            padding: 0.35rem;
            overflow-x: auto;
            flex-wrap: nowrap;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 0.5rem 0.75rem;
            font-size: 0.85rem;
            white-space: nowrap;
        }}
        
        .card {{
            padding: 0.75rem;
        }}
        
        .section-divider {{
            margin: 1rem 0;
        }}
        
        table td, table th {{
            padding: 0.5rem;
            font-size: 0.85rem;
        }}
        
        .success-message, .error-message, .info-box {{
            padding: 0.75rem;
            font-size: 0.9rem;
        }}
    }}
    
    /* Small Mobile Devices (below 600px) */
    @media (max-width: 599px) {{
        .main {{
            padding: 0.75rem 0.5rem;
        }}
        
        [data-testid="stSidebar"] {{
            min-width: 100% !important;
        }}
        
        [data-testid="stSidebar"][aria-expanded="true"] {{
            width: 100% !important;
        }}
        
        h1 {{
            font-size: 1.5rem;
            text-align: center;
        }}
        
        h2 {{
            font-size: 1.25rem;
        }}
        
        h3 {{
            font-size: 1rem;
        }}
        
        p {{
            font-size: 0.9rem;
        }}
        
        .stMetric {{
            padding: 0.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .stMetric label {{
            font-size: 0.8rem;
        }}
        
        .stMetric [data-testid="stMetricValue"] {{
            font-size: 1.1rem;
        }}
        
        .stButton > button {{
            padding: 0.5rem;
            font-size: 0.8rem;
            width: 100%;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.15rem;
            padding: 0.25rem;
            overflow-x: auto;
            flex-wrap: nowrap;
            -webkit-overflow-scrolling: touch;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 0.4rem 0.6rem;
            font-size: 0.75rem;
            white-space: nowrap;
            flex-shrink: 0;
        }}
        
        .stTabs [data-baseweb="tab-panel"] {{
            padding-top: 1rem;
        }}
        
        .card {{
            padding: 0.5rem;
            margin-bottom: 0.75rem;
        }}
        
        .section-divider {{
            margin: 0.75rem 0;
        }}
        
        table {{
            font-size: 0.75rem;
            display: block;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }}
        
        table td, table th {{
            padding: 0.4rem;
            font-size: 0.75rem;
            white-space: nowrap;
        }}
        
        .dataframe {{
            font-size: 0.75rem;
        }}
        
        .success-message, .error-message, .info-box {{
            padding: 0.5rem;
            font-size: 0.85rem;
        }}
        
        .stSelectbox, .stMultiSelect, .stSlider {{
            font-size: 0.85rem;
        }}
        
        /* Stack columns on mobile */
        [data-testid="column"] {{
            width: 100% !important;
            margin-bottom: 0.5rem;
        }}
    }}
    
    /* Landscape Mobile (small height) */
    @media (max-height: 600px) and (orientation: landscape) {{
        .main {{
            padding: 0.5rem;
        }}
        
        h1 {{
            font-size: 1.35rem;
        }}
        
        h2 {{
            font-size: 1.15rem;
        }}
        
        h3 {{
            font-size: 0.95rem;
        }}
        
        .stMetric {{
            padding: 0.4rem;
        }}
        
        .section-divider {{
            margin: 0.5rem 0;
        }}
    }}
    
    /* Print styles */
    @media print {{
        [data-testid="stSidebar"] {{
            display: none !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            display: none !important;
        }}
        
        .stButton {{
            display: none !important;
        }}
        
        .main {{
            padding: 0;
        }}
        
        * {{
            background-color: white !important;
            color: black !important;
        }}
    }}
    </style>
""", unsafe_allow_html=True)


# Session state initialization
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'workflow_step' not in st.session_state:
        st.session_state.workflow_step = WORKFLOW_STEPS[0]


initialize_session_state()


# ==================== HEADER ====================
def render_header():
    """Render professional dashboard header."""
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        st.markdown(f"<h1 style='text-align: left; font-size: 2rem; margin-bottom: 0;'>AutoDataLab</h1>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(
            "<p style='text-align: center; color: #334155; font-size: 0.95rem; margin-top: 0.5rem;'>Professional Data Science Analytics Platform</p>",
            unsafe_allow_html=True
        )
    
    with col3:
        if st.session_state.current_df is not None:
            st.markdown(
                f"<p style='text-align: right; color: #10B981; font-weight: 600; margin-top: 0.5rem;'>● Dataset Active</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<p style='text-align: right; color: #94A3B8; margin-top: 0.5rem;'>○ No Dataset</p>",
                unsafe_allow_html=True
            )
    
    st.markdown("<div style='margin: 1rem 0; border-top: 2px solid #E2E8F0;'></div>", unsafe_allow_html=True)


def render_workflow_progress(active_step_index: int):
    """Render a horizontal workflow progress indicator."""
    step_items = []
    total_steps = len(WORKFLOW_STEPS)

    for index, step in enumerate(WORKFLOW_STEPS):
        if index < active_step_index:
            state_class = "completed"
        elif index == active_step_index:
            state_class = "active"
        else:
            state_class = "upcoming"

        step_items.append(
            f"<div class='workflow-step {state_class}'>"
            f"<span class='step-index'>{index + 1}</span>"
            f"<span class='step-label'>{step}</span>"
            "</div>"
        )

        if index < total_steps - 1:
            connector_class = "workflow-connector completed" if index < active_step_index else "workflow-connector"
            step_items.append(f"<div class='{connector_class}'></div>")

    st.markdown(
        f"<div class='workflow-progress'>{''.join(step_items)}</div>",
        unsafe_allow_html=True
    )


# ==================== SIDEBAR ====================
def render_sidebar():
    """Render professional sidebar for uploads and info."""
    with st.sidebar:
        st.markdown("<h3 style='margin-top: 0;'>Data Upload</h3>", unsafe_allow_html=True)
        
        # File uploader section
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            try:
                loader = DataLoader()
                df = loader.load(uploaded_file)
                st.session_state.original_df = df
                st.session_state.current_df = df.copy()
                st.session_state.processing_history = []
                st.success("File loaded successfully")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Sample data button
        if st.button("Load Sample Data", use_container_width=True):
            np.random.seed(42)
            sample_df = pd.DataFrame({
                'Age': np.random.normal(35, 10, 100),
                'Salary': np.random.normal(60000, 15000, 100),
                'Experience': np.random.normal(8, 3, 100),
                'Department': np.random.choice(['Sales', 'IT', 'HR', 'Marketing'], 100),
                'Performance': np.random.randint(1, 6, 100)
            })
            sample_df.loc[sample_df.sample(5).index, 'Salary'] = np.nan
            sample_df = pd.concat([sample_df, sample_df.iloc[:3]])
            
            st.session_state.original_df = sample_df
            st.session_state.current_df = sample_df.copy()
            st.session_state.processing_history = []
            st.success("Sample data loaded")
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Dataset info
        if st.session_state.current_df is not None:
            st.markdown("<h3>Dataset Overview</h3>", unsafe_allow_html=True)
            
            info = helpers.format_dataframe_info(st.session_state.current_df)
            
            st.metric("Rows", info['rows'])
            st.metric("Columns", info['columns'])
            st.metric("Memory (MB)", f"{info['memory_usage_mb']:.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Numeric", info['numeric_columns'])
            with col2:
                st.metric("Categorical", info['categorical_columns'])

            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("<h3>Dataset Health Summary</h3>", unsafe_allow_html=True)

            df = st.session_state.current_df
            total_cells = df.shape[0] * df.shape[1]
            missing_pct = (df.isna().sum().sum() / total_cells * 100) if total_cells else 0.0
            duplicate_rows = int(df.duplicated().sum())

            numeric_df = df.select_dtypes(include=[np.number])
            outlier_cols = 0
            if not numeric_df.empty:
                q1 = numeric_df.quantile(0.25)
                q3 = numeric_df.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_mask = (numeric_df.lt(lower) | numeric_df.gt(upper))
                outlier_cols = int(outlier_mask.any(axis=0).sum())

            categorical_df = df.select_dtypes(exclude=[np.number])
            high_cardinality = int((categorical_df.nunique(dropna=True) > 50).sum()) if not categorical_df.empty else 0

            health_col1, health_col2 = st.columns(2)
            with health_col1:
                st.metric("Missing Values %", f"{missing_pct:.2f}%")
                st.metric("Duplicate Rows", f"{duplicate_rows:,}")
            with health_col2:
                st.metric("Outlier Columns", outlier_cols)
                st.metric("High Cardinality Cats", high_cardinality)
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Download processed data
            csv = st.session_state.current_df.to_csv(index=False)
            st.download_button(
                label="Download Data",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv",
                use_container_width=True
            )


# ==================== OVERVIEW DASHBOARD ====================
def render_overview():
    """Render dashboard overview with key metrics."""
    if st.session_state.current_df is None:
        st.markdown("""
        <div class='info-box'>
        <h3 style='margin-top: 0;'>Welcome to AutoDataLab</h3>
        <p>Upload a dataset using the sidebar to get started with automated data analysis, cleaning, and feature engineering.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **Data Operations**
            - CSV/Excel file loading
            - Statistical profiling
            - Data quality analysis
            """)
        with col2:
            st.markdown("""
            **Processing**
            - Missing value handling
            - Outlier removal
            - Duplicate detection
            """)
        with col3:
            st.markdown("""
            **Feature Engineering**
            - Encoding & scaling
            - Feature selection
            - Automated pipelines
            """)
        return
    
    # Display key metrics
    profiler = DataProfiler(st.session_state.current_df)
    profile = profiler.generate_profile()
    
    st.markdown("<h2>Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Rows", f"{profile['shape']['rows']:,}")
    with col2:
        st.metric("Total Columns", profile['shape']['columns'])
    with col3:
        try:
            missing_values = profile.get('missing_values', {})
            if missing_values:
                total_missing_pct = sum([v['missing_percentage'] for v in missing_values.values()]) / len(missing_values)
            else:
                total_missing_pct = 0
        except (TypeError, ZeroDivisionError, KeyError):
            total_missing_pct = 0
        st.metric("Missing %", f"{total_missing_pct:.1f}%")
    with col4:
        st.metric("Duplicates", profile['duplicates']['fully_duplicated_rows'])
    with col5:
        memory_mb = st.session_state.current_df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory (MB)", f"{memory_mb:.2f}")
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Data Types Distribution</h3>", unsafe_allow_html=True)
        dt_df = pd.DataFrame(list(profile['data_types'].items()), columns=['Column', 'Type'])
        st.dataframe(dt_df, use_container_width=True, height=300)
    
    with col2:
        st.markdown("<h3>Missing Values</h3>", unsafe_allow_html=True)
        missing_df = pd.DataFrame([
            {'Column': col, 'Missing %': info['missing_percentage']}
            for col, info in profile['missing_values'].items()
            if info['missing_percentage'] > 0
        ])
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True, height=300)
        else:
            st.markdown("<div class='info-box'>No missing values detected</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Statistical Summary
    st.markdown("<h3>Statistical Summary</h3>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(profile['statistical_summary']), use_container_width=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
    st.dataframe(st.session_state.current_df.head(10), use_container_width=True)


# ==================== DATA LOADING ====================
def render_data_loading():
    """Render data loading page."""
    if st.session_state.current_df is None:
        st.markdown("""
        <div class='info-box'>
        <h3 style='margin-top: 0;'>No Data Loaded</h3>
        <p>Upload a file using the sidebar or load sample data to begin.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<h2>Current Dataset</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{st.session_state.current_df.shape[0]:,}")
        with col2:
            st.metric("Columns", st.session_state.current_df.shape[1])
        with col3:
            numeric_cols = len(helpers.get_numeric_columns(st.session_state.current_df))
            st.metric("Numeric Cols", numeric_cols)
        with col4:
            categorical_cols = len(helpers.get_categorical_columns(st.session_state.current_df))
            st.metric("Categorical Cols", categorical_cols)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<h3>Data Preview (First 20 Rows)</h3>", unsafe_allow_html=True)
            st.dataframe(st.session_state.current_df.head(20), use_container_width=True)
        
        with col2:
            st.markdown("<h3>Column Types</h3>", unsafe_allow_html=True)
            dtypes_df = pd.DataFrame({
                'Column': st.session_state.current_df.columns,
                'Type': st.session_state.current_df.dtypes.astype(str)
            })
            st.dataframe(dtypes_df, use_container_width=True, height=400)


# ==================== DATA PROFILING ====================
def render_profiling():
    """Render data profiling page."""
    st.markdown("<h2>Data Profiling</h2>", unsafe_allow_html=True)
    
    if st.session_state.current_df is None:
        st.markdown(
            "<div class='error-message'>No data loaded. Please upload data first.</div>",
            unsafe_allow_html=True
        )
        return
    
    try:
        profiler = DataProfiler(st.session_state.current_df)
        profile = profiler.generate_profile()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", profile['shape']['rows'])
        with col2:
            st.metric("Columns", profile['shape']['columns'])
        with col3:
            st.metric("Total Cells", profile['shape']['rows'] * profile['shape']['columns'])
        with col4:
            st.metric("Memory (MB)", f"{st.session_state.current_df.memory_usage(deep=True).sum() / 1024**2:.2f}")
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Data types
        st.markdown("<h3>Data Types</h3>", unsafe_allow_html=True)
        dt_df = pd.DataFrame(list(profile['data_types'].items()), columns=['Column', 'Type'])
        st.dataframe(dt_df, use_container_width=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Statistical summary
        st.markdown("<h3>Statistical Summary</h3>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(profile['statistical_summary']), use_container_width=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Missing values
        st.markdown("<h3>Missing Values</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            missing_df = pd.DataFrame([
                {
                    'Column': col,
                    'Missing Count': info['missing_count'],
                    'Missing %': info['missing_percentage']
                }
                for col, info in profile['missing_values'].items()
            ])
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            missing_data = missing_df[missing_df['Missing %'] > 0]
            if not missing_data.empty:
                st.bar_chart(missing_data.set_index('Column')['Missing %'])
            else:
                st.markdown("<div class='info-box'>No missing values found</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Duplicates
        st.markdown("<h3>Duplicate Analysis</h3>", unsafe_allow_html=True)
        dup_info = profile['duplicates']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Duplicates", dup_info['fully_duplicated_rows'])
        with col2:
            st.metric("Duplicate %", f"{dup_info['duplicate_percentage']:.2f}%")
        with col3:
            st.metric("Unique Rows", dup_info['total_rows'] - dup_info['fully_duplicated_rows'])
    
    except Exception as e:
        st.markdown(
            f"<div class='error-message'>Error profiling data: {str(e)}</div>",
            unsafe_allow_html=True
        )


# ==================== DATA CLEANING ====================
def render_cleaning():
    """Render data cleaning page."""
    st.markdown("<h2>Data Cleaning</h2>", unsafe_allow_html=True)
    
    if st.session_state.current_df is None:
        st.markdown(
            "<div class='error-message'>No data loaded. Please upload data first.</div>",
            unsafe_allow_html=True
        )
        return
    
    try:
        cleaner = DataCleaner(st.session_state.current_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                "<h3 title='Choose a strategy to handle missing data: drop rows, fill with mean/median, or use most frequent value (mode)'>Handle Missing Values <span style='color: #000000; font-weight: bold; font-size: 1.1em;'>?</span></h3>",
                unsafe_allow_html=True
            )
            strategy = st.selectbox(
                "Strategy",
                ['drop', 'mean', 'median', 'mode'],
                label_visibility="collapsed"
            )
            if st.button("Apply Missing Value Handling", key='missing'):
                df_clean = cleaner.handle_missing_values(strategy=strategy)
                st.session_state.current_df = df_clean
                st.session_state.processing_history.append(f"Missing values: {strategy}")
                st.markdown(
                    f"<div class='success-message'>Applied {strategy} strategy</div>",
                    unsafe_allow_html=True
                )
        
        with col2:
            st.markdown(
                "<h3 title='Identifies and removes duplicate rows, keeping only the first occurrence of each unique row'>Remove Duplicates <span style='color: #000000; font-weight: bold; font-size: 1.1em;'>?</span></h3>",
                unsafe_allow_html=True
            )
            if st.button("Remove Duplicate Rows", key='dup'):
                df_clean = cleaner.remove_duplicates()
                st.session_state.current_df = df_clean
                st.session_state.processing_history.append("Removed duplicates")
                st.markdown(
                    "<div class='success-message'>Duplicates removed</div>",
                    unsafe_allow_html=True
                )
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                "<h3 title='Removes outliers using Interquartile Range method. Values outside [Q1 - IQR×multiplier, Q3 + IQR×multiplier] are removed. Default: 1.5'>Remove Outliers (IQR) <span style='color: #000000; font-weight: bold; font-size: 1.1em;'>?</span></h3>",
                unsafe_allow_html=True
            )
            multiplier = st.slider("IQR Multiplier", 0.5, 3.0, 1.5, 0.1)
            if st.button("Remove Outliers", key='outlier'):
                df_clean = cleaner.remove_outliers_iqr(multiplier=multiplier)
                st.session_state.current_df = df_clean
                st.session_state.processing_history.append(f"Outliers removed (IQR={multiplier})")
                st.markdown(
                    "<div class='success-message'>Outliers removed</div>",
                    unsafe_allow_html=True
                )
        
        with col2:
            st.markdown(
                "<h3 title='Automated cleaning workflow: 1) Drops missing values 2) Removes duplicate rows 3) Removes outliers using IQR method (1.5 multiplier). One-click operation with default settings.'>Complete Pipeline <span style='color: #000000; font-weight: bold; font-size: 1.1em;'>?</span></h3>",
                unsafe_allow_html=True
            )
            if st.button("Run Full Cleaning Pipeline", key='pipeline'):
                df_clean = cleaner.clean_pipeline()
                st.session_state.current_df = df_clean
                st.session_state.processing_history.append("Complete cleaning pipeline")
                st.markdown(
                    "<div class='success-message'>Cleaning pipeline completed</div>",
                    unsafe_allow_html=True
                )
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        st.markdown("<h3>Preview</h3>", unsafe_allow_html=True)
        st.dataframe(st.session_state.current_df.head(10), use_container_width=True)
    
    except Exception as e:
        st.markdown(
            f"<div class='error-message'>Error during cleaning: {str(e)}</div>",
            unsafe_allow_html=True
        )


# ==================== EDA VISUALIZATION ====================
def render_eda():
    """
    Render EDA visualization page.
    
    ARCHITECTURE:
    - Business Logic: EDAVisualizer (core/eda.py) creates matplotlib Figure objects
    - UI Layer: Streamlit (app.py) displays figures using st.pyplot(fig)
    - No plotting code in UI layer, only user interactions
    """
    st.markdown("<h2>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.current_df is None:
        st.markdown(
            "<div class='error-message'>No data loaded. Please upload data first.</div>",
            unsafe_allow_html=True
        )
        return
    
    try:
        # Initialize visualizer (Business Logic Layer)
        visualizer = EDAVisualizer(st.session_state.current_df)
        numeric_cols = helpers.get_numeric_columns(st.session_state.current_df)
        categorical_cols = helpers.get_categorical_columns(st.session_state.current_df)
        
        # Create tabs for different visualization types
        tabs = st.tabs([
            "Univariate", "Bivariate Analysis", "Multivariate", "Advanced"
        ])
        
        # ==================== UNIVARIATE ANALYSIS ====================
        with tabs[0]:
            st.markdown("<h3>Univariate Analysis</h3>", unsafe_allow_html=True)

            df_uni = st.session_state.current_df
            analyzer = univariate_analysis(df_uni)

            # ---- Column selector ----
            all_cols = list(df_uni.columns)
            uni_col = st.selectbox(
                "Select a column to analyze",
                all_cols,
                key="uni_col_select",
            )

            col_type = analyzer.detect_type(uni_col)
            type_color = {
                "numeric": "#1F3A8A",
                "categorical": "#0F766E",
                "datetime": "#F59E0B",
                "binary": "#8B5CF6",
            }.get(col_type, "#334155")
            st.markdown(
                f"<span style='background:{type_color};color:#fff;padding:2px 10px;"
                f"border-radius:99px;font-size:0.78rem;font-weight:700'>"
                f"{col_type.upper()}</span>",
                unsafe_allow_html=True,
            )

            # ---- Validation messages ----
            validations = analyzer.validate(uni_col)
            for v in validations:
                if v.level == "error":
                    st.error(v.message)
                elif v.level == "warning":
                    st.warning(v.message)
                else:
                    st.info(v.message)

            has_error = any(v.level == "error" for v in validations)
            if has_error:
                st.stop()

            # ===== NUMERIC branch =====
            if col_type == "numeric":
                stats_dict = analyzer.compute_numeric_stats(uni_col)

                # Metrics row 1
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean", f"{stats_dict['mean']:.4g}")
                m2.metric("Median", f"{stats_dict['median']:.4g}")
                m3.metric("Std Dev", f"{stats_dict['std']:.4g}")
                m4.metric("Outliers (IQR)", f"{stats_dict['outlier_count']:,}")

                # Metrics row 2
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Skewness", f"{stats_dict['skewness']:.3f}")
                m6.metric("Kurtosis", f"{stats_dict['kurtosis']:.3f}")
                m7.metric("Missing %", f"{stats_dict['missing_pct']:.1f}%")
                m8.metric("N", f"{stats_dict['n']:,}")

                st.caption(
                    f"Distribution: {stats_dict['symmetry']} — {stats_dict['tails']}"
                )

                # Normality tests
                if stats_dict.get("shapiro_p") is not None:
                    normal_flag = " (normal)" if stats_dict["shapiro_p"] > 0.05 else " (not normal)"
                    st.caption(
                        f"Shapiro-Wilk p={stats_dict['shapiro_p']:.4f}{normal_flag}"
                    )
                if stats_dict.get("dagostino_p") is not None:
                    normal_flag2 = " (normal)" if stats_dict["dagostino_p"] > 0.05 else " (not normal)"
                    st.caption(
                        f"D'Agostino p={stats_dict['dagostino_p']:.4f}{normal_flag2}"
                    )

                st.divider()

                # Plot controls
                num_plot_options = [
                    "Histogram", "KDE", "Boxplot", "Violin",
                    "Rug + KDE", "ECDF", "Q-Q Plot", "Log Comparison",
                ]
                num_plot_kind_map = {
                    "Histogram": "histogram",
                    "KDE": "kde",
                    "Boxplot": "boxplot",
                    "Violin": "violin",
                    "Rug + KDE": "rug",
                    "ECDF": "ecdf",
                    "Q-Q Plot": "qq",
                    "Log Comparison": "log_comparison",
                }

                pc1, pc2 = st.columns([2, 2])
                with pc1:
                    num_plot_label = st.selectbox("Plot type", num_plot_options, key="uni_num_plot")
                num_plot_kind = num_plot_kind_map[num_plot_label]

                with st.expander("Advanced Options", expanded=False):
                    adv1, adv2, adv3 = st.columns(3)
                    with adv1:
                        bins = st.slider("Bins (histogram)", 5, 200, 30, key="uni_bins")
                        kde_overlay = st.checkbox("KDE overlay (histogram)", value=False, key="uni_kde_ov")
                        normal_overlay = st.checkbox("Normal curve overlay", value=False, key="uni_norm_ov")
                    with adv2:
                        show_mean = st.checkbox("Show mean line", value=True, key="uni_mean")
                        show_median = st.checkbox("Show median line", value=True, key="uni_med")
                        highlight_outliers = st.checkbox("Highlight outliers", value=False, key="uni_out")
                    with adv3:
                        winsorize = st.checkbox("Winsorize (clip 5%–95%)", value=False, key="uni_winz")
                        log_scale = st.checkbox("Log1p scale", value=False, key="uni_log")
                        fill_kde = st.checkbox("Fill KDE", value=True, key="uni_fill2")

                if st.button("Generate Plot", key="uni_num_btn"):
                    try:
                        fig = analyzer.plot_numeric(
                            uni_col,
                            plot_kind=num_plot_kind,
                            bins=bins,
                            kde_overlay=kde_overlay,
                            fill_kde=fill_kde,
                            log_scale=log_scale,
                            show_mean=show_mean,
                            show_median=show_median,
                            highlight_outliers=highlight_outliers,
                            normal_overlay=normal_overlay,
                            winsorize=winsorize,
                        )
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Plot error: {e}")

            # ===== CATEGORICAL / BINARY branch =====
            elif col_type in ("categorical", "binary"):
                cat_stats = analyzer.compute_categorical_stats(uni_col)

                cm1, cm2, cm3, cm4 = st.columns(4)
                cm1.metric("Unique Values", f"{cat_stats['unique']:,}")
                cm2.metric("Mode", str(cat_stats['mode'])[:20])
                cm3.metric("Entropy", f"{cat_stats['entropy']:.3f}")
                cm4.metric("Missing %", f"{cat_stats['missing_pct']:.1f}%")

                st.caption(
                    f"Cardinality: {cat_stats['cardinality']} — "
                    f"{cat_stats['rare_count']} rare categories (<5%)"
                )

                st.divider()

                cat_plot_options = ["Bar Chart", "Horizontal Bar", "Pie", "Donut", "Pareto", "Treemap"]
                cat_plot_kind_map = {
                    "Bar Chart": "bar",
                    "Horizontal Bar": "hbar",
                    "Pie": "pie",
                    "Donut": "donut",
                    "Pareto": "pareto",
                    "Treemap": "treemap",
                }

                cat_plot_label = st.selectbox("Plot type", cat_plot_options, key="uni_cat_plot")
                cat_plot_kind = cat_plot_kind_map[cat_plot_label]

                with st.expander("Advanced Options", expanded=False):
                    cadv1, cadv2 = st.columns(2)
                    with cadv1:
                        top_n_cat = st.number_input(
                            "Show top N categories (0 = all)",
                            min_value=0, max_value=100, value=0, key="uni_topn2"
                        )
                        group_rare = st.checkbox("Group remaining into 'Others'", value=False, key="uni_grp")
                    with cadv2:
                        sort_by_freq = st.checkbox("Sort by frequency", value=True, key="uni_sort")

                cat_sub_tab = st.radio(
                    "View",
                    ["Plot", "Frequency Table", "Rare Categories"],
                    horizontal=True,
                    key="uni_cat_view",
                )

                if cat_sub_tab == "Plot":
                    if st.button("Generate Plot", key="uni_cat_btn"):
                        try:
                            fig = analyzer.plot_categorical(
                                uni_col,
                                plot_kind=cat_plot_kind,
                                top_n=int(top_n_cat) if top_n_cat > 0 else None,
                                group_rare=group_rare,
                                sort_by_freq=sort_by_freq,
                            )
                            st.pyplot(fig)
                        except ImportError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Plot error: {e}")

                elif cat_sub_tab == "Frequency Table":
                    vc = cat_stats["value_counts"]
                    ft = vc.rename_axis("Category").reset_index(name="Count")
                    ft["Percentage"] = (ft["Count"] / cat_stats["n"] * 100).round(2)
                    st.dataframe(ft, use_container_width=True)

                else:  # Rare Categories
                    vc = cat_stats["value_counts"]
                    rare_mask = (vc / cat_stats["n"]) < 0.05
                    rare_df = vc[rare_mask].rename_axis("Category").reset_index(name="Count")
                    if rare_df.empty:
                        st.info("No rare categories below 5% frequency.")
                    else:
                        rare_df["Percentage"] = (rare_df["Count"] / cat_stats["n"] * 100).round(2)
                        st.dataframe(rare_df, use_container_width=True)

            # ===== DATETIME branch =====
            elif col_type == "datetime":
                dt_stats = analyzer.compute_datetime_stats(uni_col)

                dm1, dm2, dm3, dm4 = st.columns(4)
                dm1.metric("Earliest", str(dt_stats["min"])[:10])
                dm2.metric("Latest", str(dt_stats["max"])[:10])
                dm3.metric("Span (days)", f"{dt_stats['span_days']:,}")
                dm4.metric("Inferred Freq", str(dt_stats["inferred_freq"] or "unknown"))
                st.caption(f"Missing: {dt_stats['missing_pct']:.1f}%")

                st.divider()

                dt_plot_label = st.selectbox(
                    "Plot type",
                    ["Time Series", "Monthly Trend", "Yearly Trend", "Day of Week", "Seasonal Decomposition"],
                    key="uni_dt_plot",
                )
                dt_kind_map = {
                    "Time Series": "timeseries",
                    "Monthly Trend": "monthly",
                    "Yearly Trend": "yearly",
                    "Day of Week": "dayofweek",
                    "Seasonal Decomposition": "seasonal",
                }

                if st.button("Generate Plot", key="uni_dt_btn"):
                    try:
                        fig = analyzer.plot_datetime(uni_col, plot_kind=dt_kind_map[dt_plot_label])
                        st.pyplot(fig)
                    except ImportError as e:
                        st.error(str(e))
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Plot error: {e}")
        
        # ==================== BIVARIATE ANALYSIS ====================
        with tabs[1]:
            df = st.session_state.current_df
            max_categories = 100
            auto_sample = len(df) > 50000

            col1_sel, col2_sel = st.columns(2)
            with col1_sel:
                col1 = st.selectbox("Column 1", df.columns, key='bivar_col1')
            with col2_sel:
                default_index = 1 if len(df.columns) > 1 else 0
                col2 = st.selectbox("Column 2", df.columns, index=default_index, key='bivar_col2')

            with st.expander("Advanced Options", expanded=False):
                missing_strategy = st.selectbox(
                    "Missing values",
                    ["drop", "treat_as_category"],
                    help="Drop rows with missing values or treat missing as a category for categorical columns.",
                    key='bivar_missing'
                )

                palette = st.selectbox(
                    "Color palette",
                    ["husl", "Set2", "viridis", "rocket", "mako", "crest"],
                    key='bivar_palette'
                )

                fig_width = st.slider("Figure width", 6, 16, 10, key='bivar_fig_w')
                fig_height = st.slider("Figure height", 4, 10, 6, key='bivar_fig_h')

                show_ci = st.checkbox("Show confidence intervals", value=True, key='bivar_ci')
                show_annotations = st.checkbox("Show statistical annotations", value=True, key='bivar_annot')

                if auto_sample:
                    sample_enabled = st.checkbox("Enable sampling for large datasets", value=True, key='bivar_sample')
                    sample_size = st.slider(
                        "Sample size",
                        10000,
                        min(len(df), 150000),
                        50000,
                        step=5000,
                        key='bivar_sample_size'
                    )
                else:
                    sample_enabled = False
                    sample_size = len(df)

            analyzer = bivariate_analysis(
                df,
                max_rows=sample_size,
                sample=sample_enabled,
                random_state=42
            )

            messages = analyzer.validate_selection(col1, col2, max_categories=max_categories)
            has_error = False
            for msg in messages:
                if msg.level == "error":
                    st.error(msg.message)
                    has_error = True
                elif msg.level == "warning":
                    st.warning(msg.message)

            if has_error:
                st.markdown("<div class='info-box'>Resolve the selection issues to continue.</div>", unsafe_allow_html=True)
            else:
                pair_type = analyzer.detect_pair_type(col1, col2)
                figsize = (fig_width, fig_height)

                if pair_type == "numeric_numeric":
                    hue_col = None
                    if categorical_cols:
                        use_hue = st.checkbox("Color by category", value=False, key='bivar_num_hue')
                        if use_hue:
                            hue_col = st.selectbox("Hue column", categorical_cols, key='bivar_num_hue_col')

                    plot_kind = st.selectbox(
                        "Plot type",
                        ["Scatter", "Hexbin", "Jointplot", "2D KDE"],
                        key='bivar_num_plot'
                    )
                    show_regression = st.checkbox("Show regression line", value=False, key='bivar_reg')
                    show_lowess = st.checkbox("Show LOWESS", value=False, key='bivar_lowess')
                    show_outliers = st.checkbox("Highlight outliers (IQR)", value=False, key='bivar_outliers')
                    show_kde_contour = st.checkbox("Show KDE contours", value=False, key='bivar_kde_contour')

                    stats = analyzer.compute_numeric_numeric_stats(col1, col2, missing_strategy=missing_strategy)
                    annotation_text = None
                    if "error" not in stats:
                        annotation_text = (
                            f"Pearson r: {stats['pearson_r']:.3f}\n"
                            f"Spearman r: {stats['spearman_r']:.3f}\n"
                            f"Strength: {stats['strength']}"
                        )

                    if st.button("Generate Plot", key='bivar_num_plot_btn'):
                        kind_map = {
                            "Scatter": "scatter",
                            "Hexbin": "hexbin",
                            "Jointplot": "joint",
                            "2D KDE": "scatter"
                        }
                        fig = analyzer.plot_numeric_numeric(
                            col1,
                            col2,
                            plot_kind=kind_map[plot_kind],
                            hue=hue_col,
                            regression=show_regression,
                            lowess=show_lowess,
                            hexbin=(plot_kind == "Hexbin"),
                            kde_contour=(plot_kind == "2D KDE") or show_kde_contour,
                            highlight_outliers=show_outliers,
                            palette=palette,
                            figsize=figsize,
                            missing_strategy=missing_strategy,
                            annotation_text=annotation_text,
                            show_annotations=show_annotations
                        )
                        st.pyplot(fig)

                    if "error" in stats:
                        st.warning(stats["error"])
                    else:
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("Pearson r", f"{stats['pearson_r']:.3f}")
                            st.metric("Pearson p", f"{stats['pearson_p']:.3e}")
                        with stat_col2:
                            st.metric("Spearman r", f"{stats['spearman_r']:.3f}")
                            st.metric("Spearman p", f"{stats['spearman_p']:.3e}")
                        with stat_col3:
                            st.metric("Strength", stats['strength'].title())
                            st.metric("Rows", f"{stats['n']:,}")

                elif pair_type == "categorical_numeric":
                    type1 = analyzer.detect_column_type(df[col1])
                    if type1 == "categorical":
                        cat_col, num_col = col1, col2
                    else:
                        cat_col, num_col = col2, col1

                    sort_by_mean = st.checkbox("Sort categories by mean", value=True, key='bivar_cat_sort')
                    top_n = st.number_input("Top N categories (0 = all)", 0, max_categories, 0, key='bivar_cat_topn')
                    plot_kind = st.selectbox(
                        "Plot type",
                        ["Boxplot", "Violin", "Swarm", "Strip", "Mean + CI", "Ridgeline", "Grouped Histogram"],
                        key='bivar_cat_plot'
                    )

                    stats = analyzer.compute_cat_num_stats(cat_col, num_col, missing_strategy=missing_strategy)
                    annotation_text = None
                    if "error" not in stats:
                        annotation_text = (
                            f"{stats['recommended_test']} p: {stats['anova_p']:.3e}" if stats['recommended_test'] == "ANOVA"
                            else f"{stats['recommended_test']} p: {stats['kruskal_p']:.3e}"
                        )

                    if st.button("Generate Plot", key='bivar_cat_plot_btn'):
                        kind_map = {
                            "Boxplot": "box",
                            "Violin": "violin",
                            "Swarm": "swarm",
                            "Strip": "strip",
                            "Mean + CI": "mean_ci",
                            "Ridgeline": "ridgeline",
                            "Grouped Histogram": "hist"
                        }
                        selected_kind = kind_map[plot_kind]

                        if selected_kind == "swarm" and "n" in stats and stats["n"] > 5000:
                            st.warning("Swarm plot is heavy for large datasets. Falling back to strip plot.")
                            selected_kind = "strip"

                        fig = analyzer.plot_cat_num(
                            cat_col,
                            num_col,
                            plot_kind=selected_kind,
                            top_n=top_n if top_n > 0 else None,
                            sort_by_mean=sort_by_mean,
                            show_ci=show_ci,
                            palette=palette,
                            figsize=figsize,
                            missing_strategy=missing_strategy,
                            annotation_text=annotation_text,
                            show_annotations=show_annotations
                        )
                        st.pyplot(fig)

                    if "error" in stats:
                        st.warning(stats["error"])
                    else:
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("ANOVA p", f"{stats['anova_p']:.3e}")
                            st.metric("Kruskal p", f"{stats['kruskal_p']:.3e}")
                        with stat_col2:
                            st.metric("Eta Squared", f"{stats['eta_sq']:.3f}")
                            st.metric("Recommended", stats['recommended_test'])
                        with stat_col3:
                            st.metric("Groups", f"{stats['groups']}")
                            st.metric("Rows", f"{stats['n']:,}")

                else:
                    plot_kind = st.selectbox(
                        "Plot type",
                        ["Heatmap", "Stacked Bar", "Mosaic", "Grouped Bar", "Association Plot"],
                        key='bivar_catcat_plot'
                    )
                    normalize = st.selectbox(
                        "Normalize",
                        ["none", "row", "column", "overall"],
                        key='bivar_catcat_norm'
                    )

                    stats = analyzer.compute_cat_cat_stats(col1, col2, missing_strategy=missing_strategy)
                    annotation_text = None
                    if "error" not in stats:
                        annotation_text = (
                            f"Chi2 p: {stats['p_value']:.3e}\n"
                            f"Cramer's V: {stats['cramers_v']:.3f}"
                        )

                    if st.button("Generate Plot", key='bivar_catcat_plot_btn'):
                        kind_map = {
                            "Heatmap": "heatmap",
                            "Stacked Bar": "stacked",
                            "Mosaic": "mosaic",
                            "Grouped Bar": "grouped",
                            "Association Plot": "association"
                        }
                        fig = analyzer.plot_cat_cat(
                            col1,
                            col2,
                            plot_kind=kind_map[plot_kind],
                            normalize=normalize,
                            palette=palette,
                            figsize=figsize,
                            missing_strategy=missing_strategy,
                            annotation_text=annotation_text,
                            show_annotations=show_annotations
                        )
                        st.pyplot(fig)

                    if "error" in stats:
                        st.warning(stats["error"])
                    else:
                        stat_col1, stat_col2 = st.columns(2)
                        with stat_col1:
                            st.metric("Chi-square", f"{stats['chi2']:.2f}")
                            st.metric("p-value", f"{stats['p_value']:.3e}")
                        with stat_col2:
                            st.metric("Cramer's V", f"{stats['cramers_v']:.3f}")
                            st.metric("Strength", stats['strength'].title())
        
        # ==================== MULTIVARIATE ANALYSIS ====================
        with tabs[2]:
            st.markdown("<h3>Multivariate Analysis</h3>", unsafe_allow_html=True)

            df_mv = st.session_state.current_df
            mv = multivariate_analysis(df_mv)
            if mv.sampled:
                st.info(f"Large dataset auto-sampled to {len(mv.df):,} rows for performance.")

            mv_section = st.radio(
                "Analysis section",
                ["Correlation", "Pairwise", "Dimensionality Reduction",
                 "Clustering", "Feature Interactions", "Target Analysis", "Advanced Stats"],
                horizontal=True,
                key="mv_section",
            )

            # ---- helpers ----
            all_num = numeric_cols
            all_cat = categorical_cols

            # ==================================================================
            if mv_section == "Correlation":
                st.markdown("#### Correlation Matrix")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    corr_method = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="mv_corr_meth")
                with mc2:
                    mask_upper = st.checkbox("Mask upper triangle", value=True, key="mv_corr_mask")
                with mc3:
                    annotate = st.checkbox("Annotate values", value=True, key="mv_corr_ann")

                with st.expander("Advanced Options", expanded=False):
                    adv1, adv2 = st.columns(2)
                    with adv1:
                        threshold = st.slider("Hide correlations below", 0.0, 1.0, 0.0, 0.05, key="mv_corr_thr")
                        corr_cols = st.multiselect("Restrict to columns (all = leave empty)",
                                                   all_num, key="mv_corr_cols")
                    with adv2:
                        show_vif = st.checkbox("Show VIF (multicollinearity)", value=False, key="mv_vif")
                        show_strong = st.checkbox("Show strong correlation pairs (>0.7)", value=True, key="mv_strong")

                if st.button("Compute & Plot", key="mv_corr_btn"):
                    sel_cols = corr_cols if corr_cols else None
                    try:
                        fig = mv.plot_correlation_heatmap(
                            method=corr_method, columns=sel_cols,
                            mask_upper=mask_upper, threshold=threshold, annotate=annotate,
                        )
                        st.pyplot(fig)
                        if show_strong:
                            strong_df = mv.strong_correlations(mv.compute_correlation(method=corr_method, columns=sel_cols))
                            if strong_df.empty:
                                st.info("No pairs with |correlation| >= 0.7")
                            else:
                                st.markdown("**Strong correlations (|r| ≥ 0.7)**")
                                st.dataframe(strong_df, use_container_width=True)
                        if show_vif:
                            with st.spinner("Computing VIF…"):
                                try:
                                    vif_df = mv.compute_vif(columns=sel_cols)
                                    high_vif = vif_df[vif_df["VIF"] > 10]
                                    st.markdown("**VIF Report**  (VIF > 10 = high multicollinearity)")
                                    st.dataframe(vif_df, use_container_width=True)
                                    if not high_vif.empty:
                                        st.warning(f"{len(high_vif)} feature(s) with VIF > 10: {', '.join(high_vif['Feature'].tolist())}")
                                except Exception as e:
                                    st.error(f"VIF error: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")

            # ==================================================================
            elif mv_section == "Pairwise":
                st.markdown("#### Pairplot  *(max 5 columns)*")
                if len(all_num) < 2:
                    st.warning("Need at least 2 numeric columns.")
                else:
                    pair_cols = st.multiselect(
                        "Select columns (max 5)", all_num,
                        key="mv_pair_cols"
                    )
                    pc1, pc2, pc3 = st.columns(3)
                    with pc1:
                        diag_kind = st.selectbox("Diagonal kind", ["kde", "hist"], key="mv_pair_diag")
                    with pc2:
                        use_hue = st.checkbox("Color by category", value=False, key="mv_pair_hue_chk")
                    with pc3:
                        hue_col_pair = st.selectbox("Hue column", all_cat, key="mv_pair_hue") if (use_hue and all_cat) else None

                    if len(pair_cols) > 5:
                        st.warning("Only the first 5 selected columns will be used.")

                    if st.button("Generate Pairplot", key="mv_pair_btn"):
                        if len(pair_cols) < 2:
                            st.warning("Select at least 2 columns.")
                        else:
                            with st.spinner("Generating pairplot… (may take a moment for large data)"):
                                try:
                                    fig = mv.plot_pairplot(columns=pair_cols[:5], hue=hue_col_pair, diag_kind=diag_kind)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error: {e}")

            # ==================================================================
            elif mv_section == "Dimensionality Reduction":
                st.markdown("#### Dimensionality Reduction")
                if len(all_num) < 2:
                    st.warning("Need at least 2 numeric columns.")
                else:
                    dr_algo = st.radio("Algorithm", ["PCA", "t-SNE", "UMAP (if installed)"], horizontal=True, key="mv_dr_algo")
                    dr_cols = st.multiselect("Features (all numeric if empty)", all_num, key="mv_dr_cols")
                    sel_dr = dr_cols if dr_cols else None
                    use_hue_dr = st.checkbox("Color by column", value=False, key="mv_dr_hue_chk")
                    hue_dr = st.selectbox("Color column", list(df_mv.columns), key="mv_dr_hue") if use_hue_dr else None

                    if dr_algo == "PCA":
                        proj = st.radio("Projection", ["2D", "3D", "Scree / Variance"], horizontal=True, key="mv_pca_proj")
                        if st.button("Run PCA", key="mv_pca_btn"):
                            with st.spinner("Running PCA…"):
                                try:
                                    if proj == "2D":
                                        fig = mv.plot_pca_2d(columns=sel_dr, hue=hue_dr)
                                    elif proj == "3D":
                                        if len(sel_dr or all_num) < 3:
                                            st.warning("Need at least 3 features for 3D PCA.")
                                            st.stop()
                                        fig = mv.plot_pca_3d(columns=sel_dr, hue=hue_dr)
                                    else:
                                        fig = mv.plot_scree(columns=sel_dr)
                                        result = mv.compute_pca(columns=sel_dr)
                                        ev = result["explained_variance_ratio"]
                                        cumv = result["cumulative_variance"]
                                        scree_tbl = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(ev))],
                                                                   "Explained %": (ev * 100).round(2),
                                                                   "Cumulative %": (cumv * 100).round(2)})
                                        st.pyplot(fig)
                                        st.dataframe(scree_tbl, use_container_width=True)
                                        st.stop()
                                    st.pyplot(fig)
                                    # Loadings table
                                    with st.expander("PC Loadings", expanded=False):
                                        r = mv.compute_pca(columns=sel_dr, n_components=2)
                                        st.dataframe(r["loadings"].round(3), use_container_width=True)
                                except Exception as e:
                                    st.error(f"PCA error: {e}")

                    elif dr_algo == "t-SNE":
                        ts1, ts2, ts3 = st.columns(3)
                        with ts1:
                            perplexity = st.slider("Perplexity", 5, 100, 30, key="mv_tsne_perp")
                        with ts2:
                            lr = st.slider("Learning rate", 10, 1000, 200, key="mv_tsne_lr")
                        with ts3:
                            n_iter = st.select_slider("Iterations", [250, 500, 1000, 2000], value=1000, key="mv_tsne_iter")
                        if st.button("Run t-SNE", key="mv_tsne_btn"):
                            with st.spinner("Running t-SNE… this may take a while."):
                                try:
                                    fig = mv.plot_tsne(columns=sel_dr, hue=hue_dr,
                                                       perplexity=perplexity, learning_rate=lr, n_iter=n_iter)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"t-SNE error: {e}")

                    else:  # UMAP
                        um1, um2 = st.columns(2)
                        with um1:
                            n_neighbors = st.slider("n_neighbors", 5, 100, 15, key="mv_umap_nn")
                        with um2:
                            min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, 0.05, key="mv_umap_md")
                        if st.button("Run UMAP", key="mv_umap_btn"):
                            with st.spinner("Running UMAP…"):
                                try:
                                    fig = mv.plot_umap(columns=sel_dr, hue=hue_dr,
                                                       n_neighbors=n_neighbors, min_dist=min_dist)
                                    st.pyplot(fig)
                                except ImportError as e:
                                    st.error(str(e))
                                except Exception as e:
                                    st.error(f"UMAP error: {e}")

            # ==================================================================
            elif mv_section == "Clustering":
                st.markdown("#### Clustering Analysis")
                if len(all_num) < 2:
                    st.warning("Need at least 2 numeric columns.")
                else:
                    cl_cols = st.multiselect("Features (all numeric if empty)", all_num, key="mv_cl_cols")
                    sel_cl = cl_cols if cl_cols else None

                    algo = st.radio("Algorithm", ["kmeans", "hierarchical", "dbscan"], horizontal=True, key="mv_cl_algo")

                    cl1, cl2, cl3 = st.columns(3)
                    n_clusters = 3
                    eps = 0.5
                    min_samples = 5
                    with cl1:
                        if algo in ("kmeans", "hierarchical"):
                            n_clusters = st.number_input("n_clusters", 2, 20, 3, key="mv_cl_k")
                        if algo == "kmeans":
                            show_elbow = st.checkbox("Show elbow plot", value=True, key="mv_cl_elbow")
                    with cl2:
                        if algo == "dbscan":
                            eps = st.number_input("eps", 0.01, 10.0, 0.5, key="mv_cl_eps")
                            min_samples = st.number_input("min_samples", 2, 50, 5, key="mv_cl_ms")
                    with cl3:
                        scatter_dim = st.radio("Scatter", ["2D", "3D"], horizontal=True, key="mv_cl_dim")

                    if algo == "kmeans" and show_elbow:
                        if st.button("Show Elbow", key="mv_elbow_btn"):
                            try:
                                fig = mv.plot_elbow(columns=sel_cl)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Elbow error: {e}")

                    if st.button("Fit Clusters", key="mv_cl_btn"):
                        with st.spinner("Fitting clusters…"):
                            try:
                                result = mv.fit_clusters(
                                    algorithm=algo, columns=sel_cl,
                                    n_clusters=int(n_clusters), eps=float(eps), min_samples=int(min_samples),
                                )
                                sm1, sm2, sm3 = st.columns(3)
                                sm1.metric("Clusters found", result["n_found"])
                                sm2.metric("Silhouette score",
                                           f"{result['silhouette']:.3f}" if result["silhouette"] is not None else "N/A")
                                sm3.metric("Noise points (DBSCAN)",
                                           int((result["labels"] == -1).sum()) if algo == "dbscan" else "—")

                                if scatter_dim == "2D" or len(result["columns"]) < 3:
                                    fig = mv.plot_cluster_scatter(result)
                                else:
                                    fig = mv.plot_cluster_scatter_3d(result)
                                st.pyplot(fig)

                                with st.expander("Cluster Sizes", expanded=False):
                                    st.dataframe(result["cluster_sizes"], use_container_width=True)
                                if result["centroids"] is not None:
                                    with st.expander("Cluster Centroids (KMeans)", expanded=False):
                                        st.dataframe(result["centroids"].round(3), use_container_width=True)
                            except Exception as e:
                                st.error(f"Clustering error: {e}")

            # ==================================================================
            elif mv_section == "Feature Interactions":
                st.markdown("#### Feature Interactions")
                fi_type = st.selectbox(
                    "Interaction type",
                    ["3D Scatter / Bubble", "Parallel Coordinates", "Radar Chart", "Grouped Heatmap"],
                    key="mv_fi_type",
                )

                if fi_type == "3D Scatter / Bubble":
                    if len(all_num) < 3:
                        st.warning("Need at least 3 numeric columns.")
                    else:
                        ax1, ax2, ax3 = st.columns(3)
                        with ax1:
                            x_col = st.selectbox("X axis", all_num, key="mv_3d_x")
                        with ax2:
                            y_col = st.selectbox("Y axis", all_num, index=min(1, len(all_num)-1), key="mv_3d_y")
                        with ax3:
                            z_col = st.selectbox("Z axis", all_num, index=min(2, len(all_num)-1), key="mv_3d_z")
                        bx1, bx2 = st.columns(2)
                        with bx1:
                            use_bubble = st.checkbox("Bubble size by 4th column", value=False, key="mv_bubble_chk")
                            size_col = st.selectbox("Size column", all_num, key="mv_bubble_col") if use_bubble else None
                        with bx2:
                            use_color_3d = st.checkbox("Color by category", value=False, key="mv_3d_col_chk")
                            color_3d = st.selectbox("Color column", list(df_mv.columns), key="mv_3d_col") if use_color_3d else None
                        if st.button("Generate 3D Scatter", key="mv_3d_btn"):
                            try:
                                fig = mv.plot_3d_scatter(x_col, y_col, z_col, size_col=size_col, color_col=color_3d)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error: {e}")

                elif fi_type == "Parallel Coordinates":
                    pc_cols = st.multiselect("Features (all numeric if empty)", all_num, key="mv_pc_cols")
                    pc_hue_chk = st.checkbox("Color by category", value=False, key="mv_pc_hue_chk")
                    pc_hue = st.selectbox("Color column", all_cat, key="mv_pc_hue") if (pc_hue_chk and all_cat) else None
                    if st.button("Generate", key="mv_pc_btn"):
                        try:
                            fig = mv.plot_parallel_coordinates(columns=pc_cols or None, hue=pc_hue)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error: {e}")

                elif fi_type == "Radar Chart":
                    rad_cols = st.multiselect("Features (min 3, all numeric if empty)", all_num, key="mv_rad_cols")
                    rad_hue_chk = st.checkbox("Compare groups", value=False, key="mv_rad_hue_chk")
                    rad_hue = st.selectbox("Group column", list(df_mv.columns), key="mv_rad_hue") if rad_hue_chk else None
                    if st.button("Generate", key="mv_rad_btn"):
                        try:
                            fig = mv.plot_radar(columns=rad_cols or None, hue=rad_hue)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error: {e}")

                else:  # Grouped Heatmap
                    if not all_cat:
                        st.warning("Need at least one categorical column for grouping.")
                    else:
                        grp_col = st.selectbox("Group by", all_cat, key="mv_gh_grp")
                        grp_num = st.multiselect("Numeric features (all if empty)", all_num, key="mv_gh_cols")
                        grp_agg = st.selectbox("Aggregation", ["mean", "median", "std", "min", "max"], key="mv_gh_agg")
                        if st.button("Generate", key="mv_gh_btn"):
                            try:
                                fig = mv.plot_grouped_heatmap(group_col=grp_col, columns=grp_num or None, agg=grp_agg)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error: {e}")

            # ==================================================================
            elif mv_section == "Target Analysis":
                st.markdown("#### Target-Based Analysis")
                all_cols_mv = list(df_mv.columns)
                tgt_col = st.selectbox("Target column", all_cols_mv, key="mv_tgt_col")
                tgt_feat_cols = st.multiselect("Feature columns (all numeric if empty)", all_num, key="mv_tgt_feat")
                sel_tgt = tgt_feat_cols if tgt_feat_cols else None

                tgt_view = st.radio(
                    "View",
                    ["Correlation with Target", "Feature Importance (RF)", "Grouped Statistics"],
                    horizontal=True, key="mv_tgt_view",
                )

                if tgt_view == "Correlation with Target":
                    corr_meth = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="mv_tgt_corr_meth")
                    if st.button("Compute", key="mv_tgt_corr_btn"):
                        try:
                            fig, corr_df = mv.plot_target_correlation_bar(
                                target=tgt_col, columns=sel_tgt, method=corr_meth
                            )
                            st.pyplot(fig)
                            st.dataframe(corr_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")

                elif tgt_view == "Feature Importance (RF)":
                    task_choice = st.radio("Task type", ["auto", "classification", "regression"], horizontal=True, key="mv_rf_task")
                    if st.button("Compute RF Importance", key="mv_rf_btn"):
                        with st.spinner("Fitting Random Forest…"):
                            try:
                                fig = mv.plot_feature_importance(target=tgt_col, columns=sel_tgt, task=task_choice)
                                st.pyplot(fig)
                                imp_df = mv.compute_feature_importance(target=tgt_col, columns=sel_tgt, task=task_choice)
                                with st.expander("Importance Table", expanded=False):
                                    st.dataframe(imp_df, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error: {e}")

                else:  # Grouped Statistics
                    if st.button("Compute Grouped Stats", key="mv_grp_stats_btn"):
                        try:
                            grp_df = mv.compute_grouped_stats(target=tgt_col, columns=sel_tgt)
                            st.dataframe(grp_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")

            # ==================================================================
            else:  # Advanced Stats
                st.markdown("#### Advanced Statistics")
                adv_view = st.radio(
                    "Report",
                    ["Covariance Matrix", "Partial Correlation", "Multicollinearity Report"],
                    horizontal=True, key="mv_adv_view",
                )
                adv_cols = st.multiselect("Features (all numeric if empty)", all_num, key="mv_adv_cols")
                sel_adv = adv_cols if adv_cols else None

                if adv_view == "Covariance Matrix":
                    if st.button("Compute", key="mv_cov_btn"):
                        try:
                            fig, cov_df = mv.plot_covariance_heatmap(columns=sel_adv)
                            st.pyplot(fig)
                            st.dataframe(cov_df.round(4), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")

                elif adv_view == "Partial Correlation":
                    if st.button("Compute", key="mv_pcorr_btn"):
                        try:
                            fig = mv.plot_partial_correlation_heatmap(columns=sel_adv)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error: {e}")

                else:  # Multicollinearity Report
                    if st.button("Run Report", key="mv_mcoll_btn"):
                        with st.spinner("Running multicollinearity checks…"):
                            try:
                                report = mv.multicollinearity_report(columns=sel_adv)
                                cond = report["condition_number"]
                                cond_flag = "high" if cond > 30 else ("moderate" if cond > 10 else "low")
                                st.metric("Condition Number", f"{cond:.1f}", delta=cond_flag,
                                          delta_color="inverse" if cond_flag != "low" else "off")
                                st.markdown("**VIF Table** (VIF > 10 = high multicollinearity)")
                                vif_styled = report["vif"].copy()
                                st.dataframe(vif_styled, use_container_width=True)
                                if not report["high_vif"].empty:
                                    st.warning("High VIF features: " + ", ".join(report["high_vif"]["Feature"].tolist()))
                                st.markdown("**Strong Correlations (|r| ≥ 0.7)**")
                                if report["strong_correlations"].empty:
                                    st.info("No strongly correlated pairs found.")
                                else:
                                    st.dataframe(report["strong_correlations"], use_container_width=True)
                            except Exception as e:
                                st.error(f"Multicollinearity report error: {e}")
        
        # ==================== ADVANCED ====================
        with tabs[3]:
            st.markdown("<h3>Advanced Visualizations</h3>", unsafe_allow_html=True)
            
            viz_type = st.selectbox(
                "Visualization",
                ["Distribution Grid", "Crosstab Heatmap", "Grouped Count Plot"],
                key='adv_type'
            )
            
            if viz_type == "Distribution Grid":
                if numeric_cols:
                    cols_to_plot = st.multiselect(
                        "Select columns",
                        numeric_cols,
                        key='adv_dist_cols'
                    )
                    bins = st.slider("Bins per histogram", 10, 50, 20, key='adv_dist_bins')
                    
                    if st.button("Generate Grid", key='adv_dist_btn') and cols_to_plot:
                        fig = visualizer.distribution_grid(columns=cols_to_plot, bins=bins)
                        st.pyplot(fig)
            
            elif viz_type == "Crosstab Heatmap":
                if len(categorical_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        cat1 = st.selectbox("Category 1 (rows)", categorical_cols, key='adv_cross_cat1')
                    with col2:
                        cat2 = st.selectbox("Category 2 (columns)", categorical_cols, key='adv_cross_cat2', index=min(1, len(categorical_cols)-1))
                    
                    normalize = st.selectbox(
                        "Normalize by",
                        ["None", "Rows (index)", "Columns", "All"],
                        key='adv_cross_norm'
                    )
                    norm_map = {"None": None, "Rows (index)": "index", "Columns": "columns", "All": "all"}
                    
                    if st.button("Generate Heatmap", key='adv_cross_btn'):
                        fig = visualizer.plot_crosstab_heatmap(cat1, cat2, normalize=norm_map[normalize])
                        st.pyplot(fig)
                else:
                    st.warning("Need at least 2 categorical columns")
            
            elif viz_type == "Grouped Count Plot":
                if len(categorical_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        cat1 = st.selectbox("Primary category (x-axis)", categorical_cols, key='adv_count_cat1')
                    with col2:
                        cat2 = st.selectbox("Secondary category (hue)", categorical_cols, key='adv_count_cat2', index=min(1, len(categorical_cols)-1))
                    
                    if st.button("Generate Plot", key='adv_count_btn'):
                        fig = visualizer.plot_grouped_countplot(cat1, cat2)
                        st.pyplot(fig)
                else:
                    st.warning("Need at least 2 categorical columns")
    
    except Exception as e:
        user_message = handle_exception(e, logger)
        st.markdown(
            f"<div class='error-message'>Error generating visualization: {user_message}</div>",
            unsafe_allow_html=True
        )

# ==================== FEATURE ENGINEERING ====================
def render_feature_engineering():
    """Render feature engineering page."""
    st.markdown("<h2>Feature Engineering</h2>", unsafe_allow_html=True)

    if st.session_state.current_df is None:
        st.markdown(
            "<div class='error-message'>No data loaded. Please upload data first.</div>",
            unsafe_allow_html=True
        )
        return

    try:
        df_fe = st.session_state.current_df

        # ---- Performance warning ----
        if len(df_fe) > 50_000:
            st.warning(
                f"⚠ Large dataset ({len(df_fe):,} rows) — some transformations may be slow. "
                "Consider sampling before applying polynomial or interaction features."
            )

        # ---- Column Type Summary ----
        summary = column_type_summary(df_fe)
        types = summary["types"]
        counts = summary["counts_per_type"]
        high_card = summary["high_cardinality"]

        with st.expander("Column Type Summary", expanded=True):
            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("Total Columns", summary["total_columns"])
            sm2.metric("Numeric", counts["numeric"])
            sm3.metric("Categorical", counts["categorical"])
            sm4.metric("Binary", counts["binary"])
            sm5.metric("Datetime", counts["datetime"])

            type_rows = [
                ("Numeric",     types["numeric"]),
                ("Categorical", types["categorical"]),
                ("Binary",      types["binary"]),
                ("Datetime",    types["datetime"]),
                ("ID Columns",  types["id_columns"]),
            ]
            for lbl, cols in type_rows:
                if cols:
                    st.markdown(f"**{lbl}** ({len(cols)}): " + " · ".join(f"`{c}`" for c in cols))
                else:
                    st.markdown(f"**{lbl}**: —")

            if high_card:
                st.warning(
                    "High cardinality (>50 unique values): "
                    + ", ".join(f"`{c}`" for c in high_card)
                )

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        numeric_cols    = types["numeric"]
        categorical_cols = types["categorical"] + types["binary"]
        all_cols        = list(df_fe.columns)

        # ======================= TRANSFORMATION TABS =======================
        tabs = st.tabs([
            "Encoding",
            "Scaling",
            "Log Transform",
            "Polynomial",
            "Binning",
            "Interactions",
            "Freq / Target Encode",
            "Date Features",
        ])

        # -------- TAB 0: Encoding --------
        with tabs[0]:
            st.markdown("### Encoding")
            enc_c1, enc_c2 = st.columns(2)

            with enc_c1:
                st.markdown(
                    "**Label Encoding** — converts categories to integers (best for ordinal data / tree models)."
                )
                if categorical_cols:
                    le_cols = st.multiselect("Columns to label-encode", categorical_cols, key="fe_le_cols")
                    if st.button("Apply Label Encoding", key="fe_le_btn"):
                        if le_cols:
                            eng = FeatureEngineer(st.session_state.current_df)
                            st.session_state.current_df = eng.label_encode(le_cols)
                            msg = f"Label encoded: {le_cols}"
                            st.session_state.processing_history.append(msg)
                            st.success(msg)
                        else:
                            st.warning("Select at least one column.")
                else:
                    st.info("No categorical columns detected.")

            with enc_c2:
                st.markdown(
                    "**One-Hot Encoding** — binary column per category (best for nominal data; increases column count)."
                )
                if categorical_cols:
                    ohe_cols = st.multiselect("Columns to one-hot encode", categorical_cols, key="fe_ohe_cols")
                    if st.button("Apply One-Hot Encoding", key="fe_ohe_btn"):
                        if ohe_cols:
                            eng = FeatureEngineer(st.session_state.current_df)
                            st.session_state.current_df = eng.one_hot_encode(ohe_cols)
                            msg = f"One-hot encoded: {ohe_cols}"
                            st.session_state.processing_history.append(msg)
                            st.success(msg)
                        else:
                            st.warning("Select at least one column.")
                else:
                    st.info("No categorical columns detected.")

        # -------- TAB 1: Scaling --------
        with tabs[1]:
            st.markdown("### Scaling")
            sc_c1, sc_c2 = st.columns(2)

            with sc_c1:
                st.markdown("**Standard Scaling** — mean=0, std=1. Best for SVM / KNN / neural nets.")
                if numeric_cols and st.button("Apply Standard Scaling", key="fe_std_btn"):
                    eng = FeatureEngineer(st.session_state.current_df)
                    st.session_state.current_df = eng.standard_scale(numeric_cols)
                    st.session_state.processing_history.append("Standard scaling applied")
                    st.success("Standard scaling applied.")
                elif not numeric_cols:
                    st.info("No numeric columns detected.")

            with sc_c2:
                st.markdown("**MinMax Scaling** — rescales to [0, 1]. Preserves distribution shape.")
                if numeric_cols and st.button("Apply MinMax Scaling", key="fe_mm_btn"):
                    eng = FeatureEngineer(st.session_state.current_df)
                    st.session_state.current_df = eng.minmax_scale(numeric_cols)
                    st.session_state.processing_history.append("MinMax scaling applied")
                    st.success("MinMax scaling applied.")
                elif not numeric_cols:
                    st.info("No numeric columns detected.")

        # -------- TAB 2: Log Transform --------
        with tabs[2]:
            st.markdown("### Log Transform (log1p)")
            st.caption(
                "Applies `log(1 + x)` to reduce right-skewed distributions. "
                "Non-positive values are clipped to 0 before transformation."
            )
            if numeric_cols:
                log_cols  = st.multiselect("Select numeric columns", numeric_cols, key="fe_log_cols")
                log_new   = st.toggle("Add as new columns (col_log1p)", value=True, key="fe_log_new")
                if st.button("Apply Log Transform", key="fe_log_btn"):
                    if log_cols:
                        result, msgs = apply_log_transform(
                            st.session_state.current_df, log_cols, new_column=log_new
                        )
                        st.session_state.current_df = result
                        for m in msgs:
                            st.session_state.processing_history.append(m)
                            st.info(m)
                    else:
                        st.warning("Select at least one column.")
            else:
                st.info("No numeric columns detected.")

        # -------- TAB 3: Polynomial --------
        with tabs[3]:
            st.markdown("### Polynomial Features")
            st.caption("Generates degree-n powers and cross-product terms. Can grow column count rapidly.")
            if numeric_cols:
                poly_cols   = st.multiselect("Select numeric columns (max 8 recommended)", numeric_cols, key="fe_poly_cols")
                poly_deg    = st.slider("Degree", 2, 3, 2, key="fe_poly_deg")
                poly_int    = st.toggle("Interaction terms only (no powers)", value=False, key="fe_poly_int")
                poly_drop   = st.toggle("Drop original columns", value=False, key="fe_poly_drop")

                if poly_cols:
                    from sklearn.preprocessing import PolynomialFeatures as _PF
                    _pf = _PF(degree=poly_deg, interaction_only=poly_int, include_bias=False)
                    _pf.fit([[0] * len(poly_cols)])
                    n_out = len(_pf.get_feature_names_out(poly_cols)) - len(poly_cols)
                    st.caption(f"Estimated new features: **{n_out}** (total output columns: **{len(df_fe.columns) + n_out}**)")
                    if n_out > 200:
                        st.warning("⚠ Over 200 new features — this may slow down downstream models.")

                if st.button("Apply Polynomial Features", key="fe_poly_btn"):
                    if poly_cols:
                        result, msgs = apply_polynomial(
                            st.session_state.current_df, poly_cols,
                            degree=poly_deg, interaction_only=poly_int, drop_original=poly_drop,
                        )
                        st.session_state.current_df = result
                        for m in msgs:
                            st.session_state.processing_history.append(m)
                            st.info(m)
                    else:
                        st.warning("Select at least one column.")
            else:
                st.info("No numeric columns detected.")

        # -------- TAB 4: Binning --------
        with tabs[4]:
            st.markdown("### Binning (Discretisation)")
            st.caption("Convert a continuous column into categorical intervals.")
            if numeric_cols:
                bin_col      = st.selectbox("Select column to bin", numeric_cols, key="fe_bin_col")
                bin_n        = st.slider("Number of bins", 2, 20, 5, key="fe_bin_n")
                bin_strategy = st.radio("Strategy", ["equal_width", "quantile"], horizontal=True, key="fe_bin_strat")
                if st.button("Apply Binning", key="fe_bin_btn"):
                    result, msgs = apply_binning(
                        st.session_state.current_df, bin_col,
                        n_bins=bin_n, strategy=bin_strategy,
                    )
                    st.session_state.current_df = result
                    for m in msgs:
                        st.session_state.processing_history.append(m)
                        st.info(m)
            else:
                st.info("No numeric columns detected.")

        # -------- TAB 5: Interactions --------
        with tabs[5]:
            st.markdown("### Pairwise Interaction Features")
            st.caption("Creates new features from pairs of numeric columns (multiply or divide).")
            if numeric_cols:
                ia_cols = st.multiselect("Select numeric columns (min 2)", numeric_cols, key="fe_ia_cols")
                ia_op   = st.radio("Operation", ["multiply", "divide"], horizontal=True, key="fe_ia_op")
                if ia_cols:
                    n_pairs = len(ia_cols) * (len(ia_cols) - 1) // 2
                    st.caption(f"Will create **{n_pairs}** new column(s).")
                if st.button("Apply Interaction Features", key="fe_ia_btn"):
                    if len(ia_cols) >= 2:
                        result, msgs = apply_interactions(
                            st.session_state.current_df, ia_cols, operation=ia_op
                        )
                        st.session_state.current_df = result
                        for m in msgs:
                            st.session_state.processing_history.append(m)
                            st.info(m)
                    else:
                        st.warning("Select at least 2 columns.")
            else:
                st.info("No numeric columns detected.")

        # -------- TAB 6: Frequency / Target Encoding --------
        with tabs[6]:
            st.markdown("### Frequency & Target Encoding")
            fe_c1, fe_c2 = st.columns(2)

            with fe_c1:
                st.markdown("**Frequency Encoding** — replace each category with its row count.")
                if categorical_cols:
                    freq_cols = st.multiselect("Columns to frequency-encode", categorical_cols, key="fe_freq_cols")
                    freq_new  = st.toggle("Add as new columns (col_freq)", value=True, key="fe_freq_new")
                    if st.button("Apply Frequency Encoding", key="fe_freq_btn"):
                        if freq_cols:
                            result, msgs = apply_frequency_encoding(
                                st.session_state.current_df, freq_cols, new_column=freq_new
                            )
                            st.session_state.current_df = result
                            for m in msgs:
                                st.session_state.processing_history.append(m)
                                st.info(m)
                        else:
                            st.warning("Select at least one column.")
                else:
                    st.info("No categorical columns detected.")

            with fe_c2:
                st.markdown("**Target Encoding** — replace category with smoothed mean of a numeric target.")
                if categorical_cols and numeric_cols:
                    tenc_cols   = st.multiselect("Categorical columns to encode", categorical_cols, key="fe_tenc_cols")
                    tenc_target = st.selectbox("Target column (numeric)", numeric_cols, key="fe_tenc_target")
                    tenc_smooth = st.slider("Smoothing factor", 0.1, 50.0, 1.0, step=0.5, key="fe_tenc_smooth")
                    tenc_new    = st.toggle("Add as new columns (col_target_enc)", value=True, key="fe_tenc_new")
                    if st.button("Apply Target Encoding", key="fe_tenc_btn"):
                        if tenc_cols:
                            result, msgs = apply_target_encoding(
                                st.session_state.current_df, tenc_cols, tenc_target,
                                smoothing=tenc_smooth, new_column=tenc_new,
                            )
                            st.session_state.current_df = result
                            for m in msgs:
                                st.session_state.processing_history.append(m)
                                st.info(m)
                        else:
                            st.warning("Select at least one categorical column.")
                else:
                    st.info("Requires both categorical and numeric columns.")

        # -------- TAB 7: Date Features --------
        with tabs[7]:
            st.markdown("### Date Feature Extraction")
            st.caption("Extract year, month, day-of-week, etc. from datetime columns.")
            date_cols = types["datetime"]
            if date_cols:
                dt_col = st.selectbox("Select datetime column", date_cols, key="fe_dt_col")
                if st.button("Extract Date Features", key="fe_dt_btn"):
                    eng = FeatureEngineer(st.session_state.current_df)
                    st.session_state.current_df = eng.extract_date_features(dt_col)
                    msg = f"Date features extracted from '{dt_col}'"
                    st.session_state.processing_history.append(msg)
                    st.success(msg)
            else:
                st.info("No datetime columns detected.")

        # ======================= BOTTOM PANEL =======================
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        bot_left, bot_right = st.columns([3, 1])

        with bot_right:
            st.markdown("### Reset")
            st.caption("Restore dataset to its state at upload.")
            if st.button("Reset to Original", key="fe_reset_btn", type="secondary"):
                if st.session_state.get("original_df") is not None:
                    st.session_state.current_df = st.session_state.original_df.copy()
                    st.session_state.processing_history.append("Dataset reset to original.")
                    st.success("Dataset reset to original upload.")
                else:
                    st.warning("Original dataset not available.")

        with bot_left:
            if st.session_state.processing_history:
                with st.expander("Transformation Log", expanded=False):
                    for i, entry in enumerate(reversed(st.session_state.processing_history), 1):
                        st.markdown(f"{i}. {entry}")
            else:
                st.info("No transformations applied yet.")

        st.markdown("---")
        st.markdown("### Dataset Preview (first 10 rows)")
        st.dataframe(st.session_state.current_df.head(10), use_container_width=True)
        new_shape = st.session_state.current_df.shape
        st.caption(f"Current shape: **{new_shape[0]:,} rows × {new_shape[1]} columns**")

    except Exception as e:
        st.markdown(
            f"<div class='error-message'>Error during feature engineering: {str(e)}</div>",
            unsafe_allow_html=True
        )


# ==================== FEATURE SELECTION ====================
def render_feature_selection():
    """Render feature selection page."""
    st.markdown("<h2>Feature Selection</h2>", unsafe_allow_html=True)
    
    if st.session_state.current_df is None:
        st.markdown(
            "<div class='error-message'>No data loaded. Please upload data first.</div>",
            unsafe_allow_html=True
        )
        return
    
    try:
        numeric_cols = helpers.get_numeric_columns(st.session_state.current_df)
        
        if not numeric_cols:
            st.markdown(
                "<div class='error-message'>No numeric columns for feature selection</div>",
                unsafe_allow_html=True
            )
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                "<h3 title='Removes features with low variance (near-constant values). Features with variance below threshold are dropped. Helps remove uninformative features.'>Variance Threshold <span style='color: #000000; font-weight: bold; font-size: 1.1em;'>?</span></h3>",
                unsafe_allow_html=True
            )
            threshold = st.slider("Threshold", 0.0, 1.0, 0.01, 0.01, key='var')
            if st.button("Apply", key='var_btn'):
                selector = FeatureSelector(st.session_state.current_df)
                selected = selector.variance_threshold(threshold=threshold)
                st.write(f"Selected: {selected}")
        
        with col2:
            st.markdown(
                "<h3 title='Removes highly correlated features (correlation > threshold). Keeps one feature from each correlated pair. Reduces multicollinearity and redundancy.'>Correlation Threshold <span style='color: #000000; font-weight: bold; font-size: 1.1em;'>?</span></h3>",
                unsafe_allow_html=True
            )
            corr_threshold = st.slider("Threshold", 0.0, 1.0, 0.95, 0.05, key='corr')
            if st.button("Apply", key='corr_btn'):
                selector = FeatureSelector(st.session_state.current_df)
                selected = selector.correlation_threshold(threshold=corr_threshold)
                st.write(f"Selected: {selected}")
    
    except Exception as e:
        st.markdown(
            f"<div class='error-message'>Error during feature selection: {str(e)}</div>",
            unsafe_allow_html=True
        )


# ==================== MODELING ====================
def render_modeling():
    """Render modeling page with model training, evaluation, and deployment."""
    st.markdown("<h2>Modeling</h2>", unsafe_allow_html=True)

    if st.session_state.current_df is None:
        st.markdown(
            "<div class='error-message'>No data loaded. Please upload data first.</div>",
            unsafe_allow_html=True
        )
        return

    try:
        df = st.session_state.current_df

        # === STEP 1: TARGET SELECTION ===
        st.markdown("### Step 1: Select Target Column")
        target_col = st.selectbox("Target Column", df.columns, key="mod_target")

        if not target_col:
            st.warning("Select a target column to continue")
            return

        # === STEP 2: TASK DETECTION ===
        task_type = SupervisedModels.detect_task_type(df[target_col])
        st.info(f"Detected Task: **{task_type.upper()}**")

        # === STEP 3: MODEL SELECTION ===
        st.markdown("### Step 2: Select Model")
        available_models = SupervisedModels.get_available_models(task_type)
        selected_model = st.selectbox("Model", available_models, key="mod_model")

        # === STEP 4: TRAIN-TEST SPLIT ===
        st.markdown("### Step 3: Data Split")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5, key="mod_split") / 100
        random_state = st.number_input("Random Seed", 0, 1000, 42, key="mod_seed")

        # === STEP 5: SCALING ===
        scale_features = st.checkbox("Scale Features (StandardScaler)", value=True, key="mod_scale")

        # === TRAIN BUTTON ===
        if st.button("Train Model", type="primary", key="mod_train_btn"):
            with st.spinner("Training model..."):
                # Prepare data
                trainer = ModelTrainer(df, target_col, test_size=test_size, random_state=random_state)
                trainer.prepare_data()
                trainer.split_data()

                # Get model instance
                model = SupervisedModels.get_model(selected_model, task_type)

                # Train model
                trainer.train_model(model, scale=scale_features)

                # Get predictions
                y_pred_train, y_pred_proba_train = trainer.get_predictions(use_test=False)
                y_pred_test, y_pred_proba_test = trainer.get_predictions(use_test=True)

                # Get train/test data
                X_train, X_test, y_train, y_test = trainer.get_train_test_data()

                # Store in session
                st.session_state.trained_model = trainer.model
                st.session_state.trainer = trainer
                st.session_state.task_type = task_type
                st.session_state.model_name = selected_model

            st.success("Model trained successfully!")

        # === DISPLAY RESULTS (if model is trained) ===
        if "trained_model" in st.session_state:
            trainer = st.session_state.trainer
            model = st.session_state.trained_model
            task_type = st.session_state.task_type

            X_train, X_test, y_train, y_test = trainer.get_train_test_data()
            y_pred_train, y_pred_proba_train = trainer.get_predictions(use_test=False)
            y_pred_test, y_pred_proba_test = trainer.get_predictions(use_test=True)

            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("### Results")

            # Compute metrics
            if task_type == "classification":
                metrics_train = ModelMetrics.classification_metrics(y_train.values, y_pred_train, y_pred_proba_train)
                metrics_test = ModelMetrics.classification_metrics(y_test.values, y_pred_test, y_pred_proba_test)
                
                # Store for PDF report generation
                st.session_state.last_model = model
                st.session_state.last_model_metrics = metrics_test
                st.session_state.last_task_type = task_type

                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Train Accuracy", f"{metrics_test['Accuracy']:.4f}")
                col2.metric("Test Precision", f"{metrics_test['Precision']:.4f}")
                col3.metric("Test Recall", f"{metrics_test['Recall']:.4f}")
                col4.metric("Test F1", f"{metrics_test['F1 Score']:.4f}")
                if metrics_test.get("ROC-AUC"):
                    col5.metric("ROC-AUC", f"{metrics_test['ROC-AUC']:.4f}")

                # Visualizations
                viz_tabs = st.tabs(["Confusion Matrix", "ROC Curve", "Metrics Summary"])

                with viz_tabs[0]:
                    if len(np.unique(y_test)) <= 10:
                        fig = MetricsVisualizer.plot_confusion_matrix(metrics_test["Confusion Matrix"])
                        st.pyplot(fig)
                    else:
                        st.info("Too many classes for confusion matrix display")

                with viz_tabs[1]:
                    if len(np.unique(y_test)) == 2 and metrics_test.get("ROC-AUC"):
                        from core.evaluation.metrics import ModelMetrics as MM
                        fpr, tpr, _ = MM.get_roc_curve(y_test.values, y_pred_proba_test[:, 1])
                        if fpr is not None:
                            fig = MetricsVisualizer.plot_roc_curve(fpr, tpr, metrics_test["ROC-AUC"])
                            st.pyplot(fig)

                with viz_tabs[2]:
                    st.markdown("**Test Metrics:**")
                    for key, val in metrics_test.items():
                        if key not in ["Confusion Matrix"]:
                            st.markdown(f"- **{key}**: {val:.4f}" if isinstance(val, (int, float)) else f"- **{key}**: {val}")

            else:  # regression
                metrics_train = ModelMetrics.regression_metrics(y_train.values, y_pred_train)
                metrics_test = ModelMetrics.regression_metrics(y_test.values, y_pred_test)
                
                # Store for PDF report generation
                st.session_state.last_model = model
                # Add y_true and y_pred for scatter plot in PDF
                metrics_test['y_true'] = y_test.values
                metrics_test['y_pred'] = y_pred_test
                st.session_state.last_model_metrics = metrics_test
                st.session_state.last_task_type = task_type

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Test MAE", f"{metrics_test['MAE']:.4f}")
                col2.metric("Test RMSE", f"{metrics_test['RMSE']:.4f}")
                col3.metric("Test R²", f"{metrics_test['R²']:.4f}")
                col4.metric("Adj. R²", f"{metrics_test['Adj. R²']:.4f}")

                # Visualizations
                viz_tabs = st.tabs(["Predictions vs Actual", "Residuals", "Metrics Summary"])

                with viz_tabs[0]:
                    fig = MetricsVisualizer.plot_prediction_vs_actual(y_test.values, y_pred_test)
                    st.pyplot(fig)

                with viz_tabs[1]:
                    fig = MetricsVisualizer.plot_residuals(y_test.values, y_pred_test)
                    st.pyplot(fig)

                with viz_tabs[2]:
                    st.markdown("**Test Metrics:**")
                    for key, val in metrics_test.items():
                        if key not in ['y_true', 'y_pred']:
                            st.markdown(f"- **{key}**: {val:.4f}")

            # === FEATURE IMPORTANCE ===
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("### Feature Importance")
            feature_names = trainer.get_feature_names()
            importance_df = SupervisedModels.get_feature_importance(model, feature_names)
            
            # Store for PDF report generation
            st.session_state.last_feature_importance = importance_df if not importance_df.empty else None

            if not importance_df.empty:
                top_n = st.slider("Top N Features", 5, 30, 10, key="mod_top_n")
                fig = MetricsVisualizer.plot_feature_importance(importance_df, top_n=top_n)
                st.pyplot(fig)
            else:
                st.info("Feature importance not available for this model")

            # === MODEL DOWNLOAD ===
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("### Export Model")

            import io
            model_bytes = io.BytesIO()
            import pickle
            pickle.dump(model, model_bytes)
            model_bytes.seek(0)

            st.download_button(
                label="Download Model (.pkl)",
                data=model_bytes.getvalue(),
                file_name=f"{st.session_state.model_name.replace(' ', '_')}_{random_state}.pkl",
                mime="application/octet-stream",
                key="mod_download"
            )

    except Exception as e:
        st.markdown(
            f"<div class='error-message'>Error during modeling: {str(e)}</div>",
            unsafe_allow_html=True
        )
        logger.error(f"Modeling error: {e}", exc_info=True)


# ==================== PIPELINE ====================
def render_pipeline():
    """Render automated pipeline page."""
    st.markdown("<h2>Automated Pipeline</h2>", unsafe_allow_html=True)
    
    if st.session_state.current_df is None:
        st.markdown(
            "<div class='error-message'>No data loaded. Please upload data first.</div>",
            unsafe_allow_html=True
        )
        return
    
    st.markdown(
        "<div class='info-box'>Run a complete data processing workflow in sequence</div>",
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3>Options</h3>", unsafe_allow_html=True)
        clean_missing = st.checkbox("Handle missing values", value=True)
        clean_dup = st.checkbox("Remove duplicates", value=True)
        clean_outliers = st.checkbox("Remove outliers", value=True)
        scale_features = st.checkbox("Scale features", value=True)
    
    with col2:
        st.markdown("<h3>Scaling</h3>", unsafe_allow_html=True)
        scaling_type = st.radio("Type", ['standard', 'minmax'], label_visibility="collapsed")
    
    if st.button("Run Pipeline", use_container_width=True):
        try:
            with st.spinner("Processing..."):
                df_processed = st.session_state.current_df.copy()
                
                if clean_missing or clean_dup or clean_outliers:
                    cleaner = DataCleaner(df_processed)
                    if clean_missing:
                        df_processed = cleaner.handle_missing_values(strategy='drop')
                    if clean_dup:
                        cleaner = DataCleaner(df_processed)
                        df_processed = cleaner.remove_duplicates()
                    if clean_outliers:
                        cleaner = DataCleaner(df_processed)
                        df_processed = cleaner.remove_outliers_iqr()
                
                if scale_features:
                    engineer = FeatureEngineer(df_processed)
                    if scaling_type == 'standard':
                        df_processed = engineer.standard_scale()
                    else:
                        df_processed = engineer.minmax_scale()
                
                st.session_state.current_df = df_processed
                st.session_state.processing_history.append("Complete pipeline executed")
                st.markdown(
                    "<div class='success-message'>Pipeline completed successfully</div>",
                    unsafe_allow_html=True
                )
        
        except Exception as e:
            st.markdown(
                f"<div class='error-message'>Pipeline error: {str(e)}</div>",
                unsafe_allow_html=True
            )
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("<h3>Result Preview</h3>", unsafe_allow_html=True)
    st.dataframe(st.session_state.current_df.head(10), use_container_width=True)
    
    csv = st.session_state.current_df.to_csv(index=False)
    st.download_button(
        label="Download Processed Data",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # PDF Report Generation
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h3>Generate PDF Report</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>Generate a comprehensive PDF report with EDA, visualizations, and model performance.</div>",
        unsafe_allow_html=True
    )
    
    # Check if model results are available
    has_model = hasattr(st.session_state, 'last_model') and st.session_state.last_model is not None
    has_metrics = hasattr(st.session_state, 'last_model_metrics') and st.session_state.last_model_metrics is not None
    has_importance = hasattr(st.session_state, 'last_feature_importance') and st.session_state.last_feature_importance is not None
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if has_model:
            st.success("✓ Model data detected - will include in report")
        else:
            st.info("ℹ️ No model data - will generate EDA report only")
    
    with col2:
        if st.button("Generate PDF Report", use_container_width=True, type="primary"):
            try:
                with st.spinner("Generating PDF report... This may take a moment."):
                    from reporting import generate_eda_report
                    
                    # Prepare model metrics if available
                    model_metrics = st.session_state.last_model_metrics if has_metrics else None
                    feature_importance = st.session_state.last_feature_importance if has_importance else None
                    model = st.session_state.last_model if has_model else None
                    
                    # Generate report
                    report_path = generate_eda_report(
                        df=st.session_state.current_df,
                        model=model,
                        model_metrics=model_metrics,
                        feature_importance=feature_importance,
                        output_dir="reports"
                    )
                    
                    # Offer download
                    with open(report_path, "rb") as f:
                        pdf_data = f.read()
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_data,
                            file_name="AutoDataLab_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    st.success(f"✓ Report generated successfully!")
                    
            except ImportError as e:
                st.error("❌ PDF generation library not installed. Please install reportlab: `pip install reportlab`")
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
                logger.error(f"PDF report generation error: {e}", exc_info=True)


# ==================== MAIN APPLICATION ====================
def main():
    """Main application entry point."""
    render_header()
    render_sidebar()

    active_step_index = WORKFLOW_STEPS.index(st.session_state.workflow_step)
    render_workflow_progress(active_step_index)

    st.markdown("<div class='workflow-selector'>", unsafe_allow_html=True)
    selected_step = st.radio(
        "Workflow",
        WORKFLOW_STEPS,
        index=active_step_index,
        horizontal=True,
        key="workflow_step",
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_step == "Upload":
        render_overview()
    elif selected_step == "Cleaning":
        render_cleaning()
    elif selected_step == "EDA":
        render_eda()
    elif selected_step == "Feature Engineering":
        render_feature_engineering()
    elif selected_step == "Feature Selection":
        render_feature_selection()
    elif selected_step == "Modeling":
        render_modeling()
    elif selected_step == "Export":
        render_pipeline()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        user_message = handle_exception(e, logger)
        st.error(user_message)
