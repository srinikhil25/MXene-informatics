# -*- coding: utf-8 -*-
"""
Materials Informatics — Entry Point
====================================
Slim entry point for the multipage Streamlit app.
Page config, theme CSS, and session state initialization.

Run:  streamlit run app.py
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Materials Informatics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark theme styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #1e1e2e;
        border: 1px solid #3b3b5c;
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #a0a0c0 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e0e0ff !important;
    }
    .finding-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-left: 4px solid #06b6d4;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    .tech-card {
        border-radius: 12px;
        padding: 16px;
        min-height: 140px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "project" not in st.session_state:
    st.session_state["project"] = None

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
overview = st.Page("pages/01_Overview.py", title="Overview", icon="🏠", default=True)

# Technique pages — always registered, show/hide handled inside each page
xrd_page = st.Page("pages/02_XRD.py", title="XRD Analysis", icon="📊")
xps_page = st.Page("pages/03_XPS.py", title="XPS Analysis", icon="📈")
uv_drs_page = st.Page("pages/04_UV_DRS.py", title="UV-DRS", icon="🌈")
microscopy_page = st.Page("pages/05_Microscopy.py", title="Microscopy", icon="🔬")
eds_page = st.Page("pages/06_EDS.py", title="EDS Analysis", icon="⚡")
transport_page = st.Page("pages/07_Transport.py", title="Transport", icon="🌡️")

all_pages = {
    "Main": [overview],
    "Analysis": [xrd_page, xps_page, uv_drs_page, microscopy_page, eds_page, transport_page],
}

pg = st.navigation(all_pages)
pg.run()
