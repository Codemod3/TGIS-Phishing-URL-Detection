import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests

def render_analytics_tab(api_url: str):
    """
    Fetches historical data from the API and renders a beautiful analytical dashboard.
    """
    st.subheader("📊 Threat Intelligence Analytics")
    st.markdown("Forensic overview of recent phishing scans and system performance.")

    try:
        # 1. Fetch History from API
        response = requests.get(f"{api_url}/api/v1/history", timeout=5)
        
        if response.status_code != 200:
            st.error(f"Failed to fetch history (HTTP {response.status_code}).")
            return

        data = response.json()
        results = data.get("results", [])
        
        if not results:
            st.info("🕒 No historical data available yet. Start scanning URLs to see analytics!")
            return

        # 2. Process Data for Visualization
        df = pd.DataFrame(results)
        # Convert timestamps for better sorting/display
        df['created_at'] = pd.to_datetime(df['created_at'])

        # 3. KPI Metrics Row
        total_scans = len(df)
        phishing_blocked = len(df[df['prediction_label'] == 'phishing'])
        safe_urls = total_scans - phishing_blocked

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Scans", total_scans, help="Total URLs analyzed in current history window.")
        with col2:
            st.metric("Phishing Blocked", phishing_blocked, delta=f"{phishing_blocked/total_scans*100:.1f}%", delta_color="inverse")
        with col3:
            st.metric("Safe URLs", safe_urls)

        st.markdown("---")

        # 4. Distribution Visuals
        left_col, right_col = st.columns([1, 2])

        with left_col:
            st.markdown("### 🏹 Threat Ratio")
            # Donut chart for Phishing vs Safe
            fig_ratio = px.pie(
                df, 
                names='prediction_label', 
                hole=0.6,
                color='prediction_label',
                color_discrete_map={'phishing': '#FF4B4B', 'safe': '#00C781'},
                category_orders={'prediction_label': ['phishing', 'safe']}
            )
            fig_ratio.update_traces(textposition='inside', textinfo='percent+label')
            fig_ratio.update_layout(
                showlegend=False, 
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

        with right_col:
            st.markdown("### 📐 Risk Score Distribution")
            # Histogram of Risk Scores
            fig_dist = px.histogram(
                df, 
                x="risk_score", 
                nbins=20,
                color_discrete_sequence=['#4285F4'],
                labels={'risk_score': 'Analytical Risk Score'}
            )
            fig_dist.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Risk Score (0.0 - 1.0)",
                yaxis_title="Scan Count"
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # 5. Interactive Forensic Table
        st.markdown("### 📜 Recent Analytical History")
        
        # Select and rename columns for display
        display_df = df[['created_at', 'url', 'prediction_label', 'confidence', 'risk_score', 'tgis_trust_score']].copy()
        display_df.columns = ['Timestamp', 'URL', 'Verdict', 'Confidence', 'Risk Score', 'TGIS Trust']
        
        # Styling the dataframe
        st.dataframe(
            display_df,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn("Timestamp", format="D MMM YYYY, HH:mm"),
                "URL": st.column_config.LinkColumn("Analyzed URL"),
                "Verdict": st.column_config.TextColumn("Verdict"),
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                "Risk Score": st.column_config.NumberColumn("Risk", format="%.2f"),
                "TGIS Trust": st.column_config.NumberColumn("Trust", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Analytical Engine Error: {str(e)}")
        st.info("Make sure the Phishing Detection API is running at http://localhost:8000")
