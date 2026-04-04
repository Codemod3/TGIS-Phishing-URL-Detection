"""
Master Feature Schema for the Elite Phishing URL Detection System.
This file is the Single Source of Truth for feature column alignment.
"""

# Categorized Feature Groups (Total: 60)
URL_FEATURES = [
    'url_length', 'domain_length', 'path_length', 'num_dots', 'num_hyphens', 
    'num_underscores', 'num_slashes', 'num_question_marks', 'num_equals', 
    'num_at_symbols', 'num_ampersands', 'num_special_chars', 'has_ip_address', 
    'has_port', 'subdomain_count'
]

DOMAIN_FEATURES = [
    'domain_age_days', 'domain_expiry_days', 'domain_registration_length', 'is_registered', 
    'registrar_reputation_score', 'dns_record_count', 'has_mx_record', 'has_spf_record', 
    'num_nameservers', 'ssl_certificate_valid', 'ssl_certificate_age_days', 
    'ssl_issuer_trusted', 'alexa_rank', 'google_indexed', 'page_rank_score', 
    'domain_in_brand_list', 'tld_suspicious', 'shortest_word_length', 
    'longest_word_length', 'domain_entropy'
]

CONTENT_FEATURES = [
    'has_login_form', 'num_external_links', 'num_internal_links', 'external_internal_ratio', 
    'has_iframe', 'num_redirects', 'favicon_matches_domain', 'has_popup', 
    'uses_javascript_obfuscation', 'html_title_brand_mismatch', 'num_images', 
    'num_forms', 'form_has_password_field', 'uses_https', 'has_mixed_content'
]

GRAPH_FEATURES = [
    'domain_cluster_size', 'suspicious_neighbor_count', 'cluster_phishing_ratio', 
    'registrar_trust_score', 'ip_shared_domains_count', 'nameserver_trust_score', 
    'ssl_issuer_trust_score', 'graph_centrality_score', 'community_detection_label', 
    'anomaly_score_in_cluster'
]

# The Master Ordered List (Guaranteed Stable)
FEATURE_ORDER = URL_FEATURES + DOMAIN_FEATURES + CONTENT_FEATURES + GRAPH_FEATURES

# Metadata Columns (Excluded from ML training)
METADATA_COLUMNS = ['url', 'label']
