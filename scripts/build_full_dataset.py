import pandas as pd
import numpy as np
import os
import requests
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from typing import Dict, Any, Optional

# Add the project root to sys.path to allow imports from 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core Phishing Detection Components
from src.features.pipeline import FeaturePipeline
from src.features.graph_features import GraphFeatureExtractor
from src.graph.builder import GraphBuilder
from src.graph.trust_propagation import calculate_trust_score
from src.graph.analyzer import detect_communities
import networkx as nx
from src.core.logger import log
from src.core.schema import FEATURE_ORDER, METADATA_COLUMNS

# --- Configuration ---
SAMPLE_SIZE_PER_CLASS = 100
MAX_WORKERS = 3  # Reduced from 10 to prevent WHOIS/DNS rate limiting
RAW_PHISHING_PATH = "data/raw/phishing_urls.csv"
RAW_LEGIT_PATH = "data/raw/legit_urls.csv"
PROCESSED_TRAIN_PATH = "data/processed/train.parquet"
PROCESSED_TEST_PATH = "data/processed/test.parquet"
GRAPH_SAVE_PATH = "data/graphs/domain_graph.gpickle"

# --- Global Component Initialization ---
pipeline = FeaturePipeline()
graph_builder = GraphBuilder()
graph_lock = threading.Lock()

def is_alive(url: str) -> bool:
    """Fast pre-flight check to see if a URL is actually reachable."""
    try:
        # Fast 3-second ping (HEAD request is lightweight)
        # We allow redirects to ensure we find the final destination
        response = requests.head(url, timeout=3, allow_redirects=True)
        return response.status_code < 400
    except Exception:
        return False

def process_url(url: str, label_name: str) -> Optional[Dict[str, Any]]:
    """
    Worker function to extract 50 base features and update the TGIS graph.
    Designed for concurrent execution via ThreadPoolExecutor with rate-limit logic.
    """
    try:
        # 0. Pre-flight Check: Skip dead sites to save time and prevent "empty" samples
        if not is_alive(url):
            log.info(f"⏩ Skipping dead URL: {url}")
            return None

        # Polite backoff: Introduce a delay to avoid bombarding WHOIS/DNS servers
        time.sleep(1.0)
        # 1. Map labels to binary target (Phishing = 1, Safe = 0)
        # Handles various common label formats
        label = 1 if str(label_name).lower() in ['phishing', '1', 'bad', 'malicious'] else 0
        
        # 2. Extract Base features (Structural, Domain, Content)
        # This may include network calls (WHOIS, DNS, etc.)
        features = pipeline.extract_all(url)
        features['url'] = url
        features['label'] = label
        
        # 3. Synchronized Trust Graph Update
        # We use a Lock because NetworkX DiGraph is not thread-safe for mutations
        domain = urlparse(url).netloc.split(':')[0]
        if not domain:
            domain = url.split('/')[0] # Fallback for malformed URLs
            
        with graph_lock:
            # Register URL node with basic metadata
            graph_builder.add_node('URL', url, url=url, label='phishing' if label == 1 else 'safe')
            # Register Domain node
            graph_builder.add_node('DOMAIN', domain, domain=domain, label='phishing' if label == 1 else 'safe')
            # Establish Directed Relationship
            graph_builder.add_edge(url, domain, 'URL_TO_DOMAIN')
            
        return features
        
    except Exception as e:
        # Graceful error handling: log and return None so the batch continues
        log.error(f"Critical failure processing URL {url}: {str(e)}")
        return None

def main():
    """
    High-performance dataset construction entry point.
    Orchestrates bulk feature extraction, graph modeling, and data splitting.
    """
    log.info("--- 🚀 Elite Phishing System: Dataset Construction Started ---")
    
    # 1. Validate and Load Raw Data
    if not os.path.exists(RAW_PHISHING_PATH) or not os.path.exists(RAW_LEGIT_PATH):
        log.error(f"Missing raw data. Expected: {RAW_PHISHING_PATH} and {RAW_LEGIT_PATH}")
        return

    log.info("Loading raw CSV datasets...")
    df_phish = pd.read_csv(RAW_PHISHING_PATH)
    df_legit = pd.read_csv(RAW_LEGIT_PATH)
    
    # Sampling to ensure balance and speed for this specific run
    log.info(f"Sampling {SAMPLE_SIZE_PER_CLASS} records from each class...")
    df_phish_sampled = df_phish.sample(n=min(SAMPLE_SIZE_PER_CLASS, len(df_phish)), random_state=42)
    df_legit_sampled = df_legit.sample(n=min(SAMPLE_SIZE_PER_CLASS, len(df_legit)), random_state=42)
    
    # Combine into working set
    df_master = pd.concat([df_phish_sampled, df_legit_sampled]).reset_index(drop=True)
    
    # Prepare URL/Label pairs for the worker pool
    # Assumes common headers 'url' and 'type' or 'label'
    url_col = 'url' if 'url' in df_master.columns else df_master.columns[0]
    label_col = 'type' if 'type' in df_master.columns else 'label' if 'label' in df_master.columns else df_master.columns[1]
    
    pairs = list(zip(df_master[url_col], df_master[label_col]))
    log.info(f"Targeting {len(pairs)} URLs using {MAX_WORKERS} concurrent threads.")

    # 2. Batch Base Feature Extraction (Parallel)
    results_base = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map pairs to the threaded worker
        future_to_url = {executor.submit(process_url, url, label): url for url, label in pairs}
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                if data:
                    results_base.append(data)
            except Exception as e:
                log.error(f"Thread failure for {url}: {e}")

    # Build intermediate DataFrame
    df_features_base = pd.DataFrame(results_base)
    log.success(f"Base extraction complete. Records: {len(df_features_base)}")

    # 3. Global Graph Pre-calculation
    # To avoid O(N^2) scaling, we run heavy algorithms once and store results as node attributes
    if len(df_features_base) > 0:
        log.info("--- 🧠 Performing Global Graph Intelligence Pass ---")
        graph = graph_builder.graph
        
        # Run algorithms exactly once for the entire graph
        trust_scores = calculate_trust_score(graph)
        communities = detect_communities(graph)
        
        log.info("Calculating Betweenness Centrality (Global)...")
        centrality = nx.betweenness_centrality(graph, weight='weight', normalized=True)
        
        # Inject results back into the NetworkX graph as node attributes
        log.info("Injecting intelligence attributes into TGIS nodes...")
        for node in graph.nodes:
            graph.nodes[node]['trust_score'] = trust_scores.get(node, 0.5)
            graph.nodes[node]['community_id'] = communities.get(node, -1)
            graph.nodes[node]['centrality'] = centrality.get(node, 0.0)

        # 4. Post-Process Graph Features (TGIS Enrichment)
        log.info("Enriching dataset with Graph Features (O(1) Lookups)...")
        graph_extractor = GraphFeatureExtractor()
        full_results = []
        
        # This loop now runs in linear time as it just reads pre-calculated attributes
        for _, row in df_features_base.iterrows():
            graph_feats = graph_extractor.extract(row['url'], graph)
            # Combine base 50 features with 10 graph features
            merged_row = {**row, **graph_feats}
            full_results.append(merged_row)
            
            df_final = pd.DataFrame(full_results)
    else:
        log.error("No features extracted. Aborting split.")
        return

    # 4. Standardize Schema and Stratified Split
    log.info(f"Standardizing feature schema ({len(FEATURE_ORDER)} columns)...")
    # Reorder columns strictly. Any missing features (e.g. failed lookups) are filled with 0.
    # StandardScaler will handle these 0 values normally during training.
    df_final = df_final.reindex(columns=FEATURE_ORDER + METADATA_COLUMNS, fill_value=0)
    
    log.info("Splitting dataset (80/20 train/test)...")
    
    # We keep 'url' for metadata and reporting, though it won't be used as a feature.
    # ModelTrainer already handles dropping 'url' before training.
    df_final = df_final.copy() 
    
    train_set, test_set = train_test_split(
        df_final, 
        test_size=0.2, 
        stratify=df_final['label'], 
        random_state=42
    )

    log.info("Writing Parquet files to disk...")
    os.makedirs("data/processed", exist_ok=True)
    train_set.to_parquet(PROCESSED_TRAIN_PATH, index=False)
    test_set.to_parquet(PROCESSED_TEST_PATH, index=False)
    
    # Save the synchronized Trust Graph
    log.info("Saving TGIS graph state...")
    graph_builder.save_graph(GRAPH_SAVE_PATH)
    
    log.success(f"--- ✅ Processing Complete ---")
    log.info(f"Training Samples: {len(train_set)}")
    log.info(f"Testing Samples:  {len(test_set)}")
    log.info(f"Graph Nodes:      {graph_builder.get_node_count()}")

if __name__ == "__main__":
    main()
