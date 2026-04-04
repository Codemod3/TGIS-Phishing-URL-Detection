import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# Add the project root to sys.path to allow imports from 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.pipeline import FeaturePipeline
from src.graph.builder import GraphBuilder
from src.features.graph_features import GraphFeatureExtractor
from src.core.logger import log

def build_mini_dataset():
    raw_path = "data/raw/dataset.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(raw_path):
        log.error(f"Raw dataset not found at {raw_path}")
        return

    log.info(f"Loading raw dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    
    # 1. Stratified Sampling (20 Phishing, 20 Safe/Legitimate)
    phish_pool = df_raw[df_raw['Type'].str.lower() == 'phishing']
    safe_pool = df_raw[df_raw['Type'].str.lower().isin(['safe', 'legitimate'])]
    
    if len(phish_pool) >= 20:
        phish_samples = phish_pool.sample(n=20, random_state=42)
    else:
        phish_samples = phish_pool
        log.warning(f"Only {len(phish_pool)} phishing URLs found.")

    if len(safe_pool) >= 20:
        safe_samples = safe_pool.sample(n=20, random_state=42)
    else:
        log.warning(f"Only {len(safe_pool)} safe URLs found in dataset. Augmenting with common domains...")
        # Augment with common safe URLs to ensure the test pipeline can run
        safe_urls = [
            "https://www.google.com", "https://www.wikipedia.org", "https://www.microsoft.com", 
            "https://www.apple.com", "https://www.amazon.com", "https://www.github.com",
            "https://www.linkedin.com", "https://www.reddit.com", "https://www.bbc.com",
            "https://www.nytimes.com", "https://www.cnn.com", "https://www.facebook.com",
            "https://www.instagram.com", "https://www.twitter.com", "https://www.netflix.com",
            "https://www.spotify.com", "https://www.dropbox.com", "https://www.slack.com",
            "https://www.zoom.us", "https://www.paypal.com"
        ]
        safe_samples = pd.DataFrame({'url': safe_urls[:20], 'Type': ['Safe'] * 20})
    
    df_sampled = pd.concat([phish_samples, safe_samples]).reset_index(drop=True)
    log.info(f"Sampled {len(df_sampled)} URLs for feature extraction ({len(phish_samples)} phish, {len(safe_samples)} safe).")

    # 2. Initialize Extractors
    pipeline = FeaturePipeline()
    graph_builder = GraphBuilder()
    graph_extractor = GraphFeatureExtractor()
    
    processed_records = []
    
    log.info("Starting Batch Feature Extraction (60 features per URL)...")
    
    for idx, row in df_sampled.iterrows():
        url = row['url']
        label_str = row['Type']
        label = 1 if label_str == 'Phishing' else 0
        
        log.info(f"[{idx+1}/{len(df_sampled)}] Processing: {url} ({label_str})")
        
        try:
            # A. Extract 50 structural/content features
            features_50 = pipeline.extract_all(url)
            
            # B. Populate Graph (to provide meaningful TGIS features)
            # We add the node and its known label to the builder
            graph_builder.add_node('URL', url, label='phishing' if label == 1 else 'safe')
            # Extract domain to link correctly
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.split(':')[0]
            graph_builder.add_node('DOMAIN', domain, domain=domain, label='phishing' if label == 1 else 'safe')
            graph_builder.add_edge(url, domain, 'URL_TO_DOMAIN')
            
            # C. Extract 10 graph features
            features_10 = graph_extractor.extract(url, graph_builder.graph)
            
            # D. Merge and add label
            combined_features = {**features_50, **features_10}
            combined_features['label'] = label
            combined_features['url'] = url # Keep for reference
            
            processed_records.append(combined_features)
            
        except Exception as e:
            log.error(f"Failed to process {url}: {e}")
            continue

    # 3. Create Final DataFrame
    df_processed = pd.DataFrame(processed_records)
    
    if df_processed.empty:
        log.error("No URLs were successfully processed. Terminating.")
        return
        
    log.success(f"Successfully processed {len(df_processed)} URLs.")

    # 4. Stratified Split (70 Train / 30 Test)
    train_df, test_df = train_test_split(
        df_processed, 
        test_size=0.3, 
        random_state=42, 
        stratify=df_processed['label']
    )
    
    # 5. Save as Parquet
    train_path = os.path.join(processed_dir, "train.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    log.success(f"Saved {len(train_df)} train samples to {train_path}")
    log.success(f"Saved {len(test_df)} test samples to {test_path}")

if __name__ == "__main__":
    build_mini_dataset()
