import streamlit as st
import os
import sys

# Ensure project root is in sys.path so components can import from 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_client import APIClient
from components.predictor import render_predictor
from components.analytics import render_analytics_tab

# 1. Page Configuration for a Premium, Professional Interface
st.set_page_config(
    page_title="Phishing URL Detector | Elite Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Richer Aesthetics
st.markdown("""
<style>
    /* Main Layout Styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Metric Cards Styling */
    [data-testid="stMetric"] {
        background-color: #262730;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #3e3f4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Button Hover Effects */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(66, 133, 244, 0.3);
    }

    /* Sidebar Status Indicator */
    .status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Initialize Global API Client
client = APIClient()
API_URL = client.base_url

# 4. Sidebar Navigation & System Status Integration
with st.sidebar:
    # App Branding
    st.title("🛡️ Elite Detection")
    st.markdown("---")
    
    # System Status (Live Feed from Health Endpoint)
    health = client.get_health()
    is_online = health["status"] == "healthy"
    status_display = "🟢 ONLINE" if is_online else "🔴 OFFLINE"
    
    st.markdown(f"**Backend Status:** {status_display}")
    if is_online:
        st.caption(f"App Version: `{health.get('version', '1.0.0')}`")
        st.caption(f"System Uptime: `{health.get('uptime_seconds', 0)}s`")
    
    st.markdown("---")
    
    # Navigation Menu
    st.markdown("### 🗺️ Navigation")
    menu = st.radio(
        label="Select Workspace",
        options=["🔍 Real-time Analysis", "📊 Batch Processing", "📈 Performance Logs"],
        index=0
    )
    
    st.markdown("---")
    st.info("💡 **Pro-Tip:** Check the TGIS trust scores in the sidebar results for malicious cluster alerts.")

# 5. Main Content Routing Logic
if menu == "🔍 Real-time Analysis":
    st.title("🛡️ Phishing URL Detector")
    st.markdown("Combine Machine Learning, WHOIS Metadata, and Trust Graph Intelligence for deep link analysis.")
    
    tab1, tab2 = st.tabs(["🔍 URL Scanner", "📊 Threat Analytics"])
    
    with tab1:
        render_predictor(client)
    
    with tab2:
        render_analytics_tab(API_URL)

elif menu == "📊 Batch Processing":
    st.subheader("📊 Batch Analysis Mode")
    st.info("Bulk analysis features (Phase 8) are coming soon. Access requires administrative privileges.")
    # Placeholder for file uploader
    st.file_uploader("Upload URL list (.csv)", type=["csv"], disabled=True)

elif menu == "📈 Performance Logs":
    st.subheader("📈 System Metrics & Logs")
    st.markdown("Real-time telemetry from the detection pipeline.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Hits", "1,245", delta="+12%")
    with col2:
        st.metric("Avg Latency", "245ms", delta="-15ms", delta_color="normal")
    
    st.warning("Telemetry data in development.")
