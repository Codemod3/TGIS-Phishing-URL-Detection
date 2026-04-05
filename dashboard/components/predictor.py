import streamlit as st
import pandas as pd
import time
from typing import Dict, Any
from components.explainer import render_deep_dive

def render_predictor(client):
    """
    Main prediction UI component for the Phishing Detection Dashboard.
    Provides interactive URL analysis and detailed result visualization.
    """
    st.subheader("🔍 Elite URL Analysis")
    st.markdown("Execute the full detection pipeline across 60 features and the global trust graph.")
    
    # URL Input Field
    url_input = st.text_input(
        "Enter Target URL", 
        placeholder="https://example-phishing-site.net/login.php",
        help="Paste the full URL including http/https"
    )
    
    # Analyze Button
    if st.button("🚀 Run Deep Analysis", type="primary", use_container_width=True):
        if not url_input:
            st.warning("⚠️ Please provide a URL for analysis.")
            return

        with st.spinner("Initiating Phishing URL Detection Pipeline..."):
            # Call backend API
            result = client.predict_url(url_input)
            
            if "error" in result:
                st.error(f"Analysis Failed: {result['error']}")
                return
            
            # 1. Main Verdict Header
            st.divider()
            
            verdict = result["prediction"].upper()
            color = "#ff4b4b" if verdict == "PHISHING" else "#00c853"
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                    <div style='background-color: {color}22; padding: 20px; border-radius: 10px; border: 1px solid {color}'>
                        <h4 style='color: {color}; margin: 0;'>VERDICT: {verdict}</h3>
                        <p style='margin: 0; opacity: 0.8;'>Confidence: {result['confidence']*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.metric("Risk Score", f"{result['risk_score']:.4f}")
                st.caption(f"Analysis Time: {result['processing_time_ms']}ms")
            
            with col2:
                st.markdown("### 📝 Analysis Summary")
                reason = result.get("explanation", {}).get("reason", "URL analyzed using ensemble logic.")
                st.info(reason)
                
                # Dynamic Top Features
                st.markdown("**Core Risk Indicators:**")
                for feat in result.get("top_features", []):
                    importance_pct = int(feat['importance'] * 100)
                    st.write(f"- `{feat['name']}`: **{feat['value']}** (Weight: {importance_pct}%)")

            # 2. Model Decision Breakdown
            st.divider()
            st.subheader("📊 Model Consensus Breakdown")
            scores = result["model_scores"]
            
            # Map raw scores to risk probabilities (TGIS: high trust = low phishing)
            chart_data = pd.DataFrame({
                "Model": ["Random Forest", "XGBoost", "TGIS Graph", "Ensemble Summary"],
                "Phishing Probability": [
                    scores["random_forest"], 
                    scores["xgboost"], 
                    1 - scores["tgis"], 
                    scores["ensemble"]
                ]
            })
            st.bar_chart(chart_data, x="Model", y="Phishing Probability", color="#4285f4")
            
            # 3. Component Details (Graph vs external)
            st.divider()
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("### 🕸️ TGIS Graph Intelligence")
                ga = result["graph_analysis"]
                st.write(f"**Trust Score:** `{ga['trust_score']:.4f}`")
                st.write(f"**Cluster Risk:** `{ga['cluster_risk'].upper()}`")
                st.write(f"**Suspicious Neighbors:** `{ga['suspicious_neighbors']}`")
                
            with c2:
                st.markdown("### 🔌 External Verification")
                api = result["api_checks"]
                sb_color = "🔴" if api["safe_browsing"]["is_flagged"] else "🟢"
                st.write(f"{sb_color} **Safe Browsing:** {'FLAGGED' if api['safe_browsing']['is_flagged'] else 'CLEAN'}")
                st.write(f"🏢 **Registrar:** `{api['whois']['registrar'] or 'N/A'}`")

            # 4. Deep Dive Analysis (Phase 7 Explainer)
            render_deep_dive(result.get("features", {}))
