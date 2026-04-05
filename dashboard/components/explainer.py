import streamlit as st
import math

def _get_feature_metadata():
    """Returns a dictionary of descriptions and risk-coloring logic for each feature."""
    return {
        # --- URL Structural Features ---
        'url_length': {'desc': 'Total character count of the URL.', 'is_bad': lambda v: v > 75, 'is_good': lambda v: v < 40},
        'domain_length': {'desc': 'Character count of the domain.', 'is_bad': lambda v: v > 30},
        'num_dots': {'desc': 'Number of dots in the domain/path.', 'is_bad': lambda v: v > 3, 'is_good': lambda v: v <= 2},
        'num_hyphens': {'desc': 'Commonly used in typosquatting.', 'is_bad': lambda v: v > 1},
        'num_underscores': {'desc': 'Often used to hide redirects.', 'is_bad': lambda v: v > 0},
        'num_slashes': {'desc': 'Deep file paths often host phishing kits.', 'is_bad': lambda v: v > 4},
        'num_special_chars': {'desc': 'Excessive symbols suggest obfuscation.', 'is_bad': lambda v: v > 5},
        'has_ip_address': {'desc': 'Direct IP URLs are almost always malicious.', 'is_bad': lambda v: v == 1, 'is_good': lambda v: v == 0},
        'subdomain_count': {'desc': 'Multi-level domains often simulate brands.', 'is_bad': lambda v: v > 2, 'is_good': lambda v: v == 0},
        
        # --- Domain Metadata Features ---
        'domain_age_days': {'desc': 'Days since registration.', 'is_bad': lambda v: v < 30 or v == -1, 'is_good': lambda v: v > 365},
        'domain_registration_length': {'desc': 'Duration of current lease.', 'is_bad': lambda v: v < 365, 'is_good': lambda v: v > 730},
        'is_registered': {'desc': 'Whether WHOIS info exists.', 'is_bad': lambda v: v == 0, 'is_good': lambda v: v == 1},
        'ssl_certificate_valid': {'desc': 'Valid HTTPS certificate.', 'is_bad': lambda v: v == 0, 'is_good': lambda v: v == 1},
        'alexa_rank': {'desc': 'Global traffic popularity.', 'is_bad': lambda v: v == -1 or v > 1000000, 'is_good': lambda v: v > 0 and v < 50000},
        'domain_entropy': {'desc': 'Randomness of the domain string.', 'is_bad': lambda v: v > 4.5, 'is_good': lambda v: v < 3.5},
        'tld_suspicious': {'desc': 'TLDs like .xyz, .top, .app.', 'is_bad': lambda v: v == 1, 'is_good': lambda v: v == 0},
        
        # --- Content Fingerprints ---
        'has_login_form': {'desc': 'Presence of input fields on the site.', 'is_bad': lambda v: v == 1},
        'num_external_links': {'desc': 'Links pointing to other domains.', 'is_bad': lambda v: v > 10},
        'external_internal_ratio': {'desc': 'High ratio often hide phishing targets.', 'is_bad': lambda v: v > 0.8},
        'has_iframe': {'desc': 'Iframes frequently hide cross-site overlays.', 'is_bad': lambda v: v == 1, 'is_good': lambda v: v == 0},
        'num_redirects': {'desc': 'Total bounces in the hop-chain.', 'is_bad': lambda v: v > 2, 'is_good': lambda v: v <= 1},
        'favicon_matches_domain': {'desc': 'Inconsistent favicons suggest brand spoofing.', 'is_bad': lambda v: v == 0, 'is_good': lambda v: v == 1},
        'html_title_brand_mismatch': {'desc': 'Title tag does not match brand heuristics.', 'is_bad': lambda v: v == 1, 'is_good': lambda v: v == 0},
        
        # --- Trust Graph (TGIS) ---
        'domain_cluster_size': {'desc': 'Number of associated entities.', 'is_good': lambda v: v > 5},
        'suspicious_neighbor_count': {'desc': 'Directly linked known-bad entities.', 'is_bad': lambda v: v > 0, 'is_good': lambda v: v == 0},
        'registrar_trust_score': {'desc': 'Historical reputation of the registrar.', 'is_bad': lambda v: v < 0.4, 'is_good': lambda v: v > 0.8},
        'graph_centrality_score': {'desc': 'Influence level in the local domain cluster.', 'is_bad': lambda v: v > 0.7}
    }

def render_deep_dive(features_dict: dict):
    """
    Renders a human-readable analysis of the extracted features.
    Translates raw ML vectors into actionable security insights for analysts.
    """
    from src.core.schema import URL_FEATURES, DOMAIN_FEATURES, CONTENT_FEATURES, GRAPH_FEATURES
    
    with st.expander("🔍 Detailed Flag Analysis", expanded=False):
        st.markdown("### 🛡️ Security Indicator Breakdown")
        st.info("The following flags are based on raw feature extraction. These signals directly influence the ML ensemble's risk score.")
        
        # 1. High-Level Summary Alerts
        age = features_dict.get('domain_age_days', -1)
        if age == -1 or (isinstance(age, (int, float)) and age < 30):
            st.warning("🚩 **Domain Age**: This domain is brand new (or WHOIS is masked). High risk for ephemeral infrastructure.")
            
        if features_dict.get('tld_suspicious', 0) == 1:
            st.error("🚩 **TLD Reputation**: The extension used is frequently associated with malicious activity.")
            
        if features_dict.get('ssl_certificate_valid', 0) == 0:
            st.error("🚩 **Enscription Alert**: No valid SSL certificate detected. Unsafe for sensitive data.")
            
        # 2. Detailed Forensic Categories
        st.divider()
        st.subheader("🧬 Full Forensic Feature Vector")
        st.caption("Categorized analysis with conditional safety highlights (Red = Suspicious, Green = Safe)")
        
        meta = _get_feature_metadata()
        
        categories = [
            ("🔗 URL Structural Integrity", URL_FEATURES),
            ("🏢 Domain & WHOIS Intelligence", DOMAIN_FEATURES),
            ("📄 Website Content Analysis", CONTENT_FEATURES),
            ("🕸️ TGIS Graph & Neighborhood Trust", GRAPH_FEATURES)
        ]
        
        for cat_name, cat_keys in categories:
            st.write(f"#### {cat_name}")
            for key in cat_keys:
                if key in features_dict:
                    val = features_dict[key]
                    info = meta.get(key, {'desc': 'No description available.'})
                    
                    # Determine color
                    color = "#e0e0e0" # Default Grey
                    if info.get('is_bad') and info['is_bad'](val):
                        color = "#ff4b4b" # Red
                    elif info.get('is_good') and info['is_good'](val):
                        color = "#00c853" # Green
                        
                    # Format value for display
                    disp_val = f"{val:.4f}" if isinstance(val, (float)) else str(val)
                    
                    st.markdown(f"""
                        <div style="padding: 5px 0;">
                            <span style="color: {color}; font-weight: bold; font-family: monospace;">{key}: {disp_val}</span><br/>
                            <small style="color: #9e9e9e;">{info['desc']}</small>
                        </div>
                    """, unsafe_allow_html=True)
            st.write("") # Spacer
