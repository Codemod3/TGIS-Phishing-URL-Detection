import networkx as nx
from pprint import pprint
import time
import os
import sys

# Add the project root to sys.path to allow imports from 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.builder import GraphBuilder
from src.graph.trust_propagation import calculate_trust_score
from src.features.graph_features import GraphFeatureExtractor
from src.core.logger import log

def run_tgis_demo():
    log.info("🚀 Starting TGIS (Trust Graph Intelligence System) Demonstration...")
    
    builder = GraphBuilder()
    extractor = GraphFeatureExtractor()
    
    # --- 1. BUILD SAFE CLUSTER ---
    log.info("Building Safe Cluster (google.com infrastructure)...")
    builder.add_node('DOMAIN', 'google.com', domain='google.com', label='safe')
    builder.add_node('URL', 'https://google.com/search', label='safe')
    builder.add_node('IP', '8.8.8.8', ip_address='8.8.8.8', label='safe')
    builder.add_node('REGISTRAR', 'markmonitor', name='MarkMonitor Inc.', reputation_score=1.0)
    
    builder.add_edge('https://google.com/search', 'google.com', 'URL_TO_DOMAIN')
    builder.add_edge('google.com', '8.8.8.8', 'DOMAIN_TO_IP')
    builder.add_edge('google.com', 'markmonitor', 'DOMAIN_TO_REGISTRAR')

    # --- 2. BUILD MALICIOUS CLUSTER (Phish Farm) ---
    log.info("Building Malicious Cluster (Infrastructure Sharing Phish-Farm)...")
    
    # Shared malicious infrastructure
    builder.add_node('IP', '66.66.66.66', ip_address='66.66.66.66', label='phishing')
    builder.add_node('REGISTRAR', 'shady-reg', name='Bulletproof Registrar', reputation_score=0.1)
    
    # 3 different phishing domains sharing the same bad intellectual property
    phish_scenarios = [
        ('login-secure-bank.tk', 'http://login-secure-bank.tk/verify'),
        ('update-account-verify.cf', 'http://update-account-verify.cf/login'),
        ('verify-identity-safe.ga', 'http://verify-identity-safe.ga/secure')
    ]
    
    for domain_id, url_id in phish_scenarios:
        builder.add_node('DOMAIN', domain_id, domain=domain_id, label='phishing')
        builder.add_node('URL', url_id, label='phishing')
        
        builder.add_edge(url_id, domain_id, 'URL_TO_DOMAIN')
        builder.add_edge(domain_id, '66.66.66.66', 'DOMAIN_TO_IP')
        builder.add_edge(domain_id, 'shady-reg', 'DOMAIN_TO_REGISTRAR')

    # --- 3. RUN TRUST PROPAGATION ---
    log.info("Running Trust Propagation Algorithm...")
    start_time = time.time()
    trust_scores = calculate_trust_score(builder.graph, damping_factor=0.3, iterations=10)
    propagation_time = time.time() - start_time
    
    log.info(f"Trust Propagation completed in {propagation_time:.4f}s")
    
    # --- 4. ANALYZE RESULTS ---
    print("\n" + "="*50)
    print("💎 TGIS TRUST SCORES (PROPAGATED)")
    print("="*50)
    
    # Show how shared infrastructure influences trust
    entities_to_show = [
        'google.com', 
        '8.8.8.8', 
        'markmonitor',
        '66.66.66.66', 
        'shady-reg',
        'login-secure-bank.tk'
    ]
    
    for entity in entities_to_show:
        score = trust_scores.get(entity, 'N/A')
        label = builder.graph.nodes[entity].get('label', 'unknown')
        node_type = builder.graph.nodes[entity].get('type', 'Unknown')
        print(f"[{node_type:10}] {entity:25} | Score: {score:.4f} | Label: {label}")

    # --- 5. EXTRACT GRAPH FEATURES ---
    target_url = "http://login-secure-bank.tk/verify"
    log.info(f"Extracting 10 Graph Features for: {target_url}")
    
    graph_features = extractor.extract(target_url, builder.graph)
    
    print("\n" + "="*50)
    print(f"📊 GRAPH FEATURES FOR: {target_url}")
    print("="*50)
    pprint(graph_features)
    print("="*50)
    print(f"✅ Total Features Extracted: {len(graph_features)}")

if __name__ == "__main__":
    run_tgis_demo()
