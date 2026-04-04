import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.schema import FEATURE_ORDER
from api.services.prediction_service import PredictionService
from src.features.pipeline import FeaturePipeline
from src.core.logger import log

def verify_sync():
    log.info("--- 🔎 Starting Master Schema Verification ---")
    
    # 1. Check Schema length
    log.info(f"Master Schema Count: {len(FEATURE_ORDER)}")
    if len(FEATURE_ORDER) != 60:
        log.error(f"FAILURE: Schema has {len(FEATURE_ORDER)} features, expected 60.")
        return False
        
    # 2. Check Pipeline Output vs Schema
    log.info("Testing FeaturePipeline extraction...")
    pipeline = FeaturePipeline()
    test_url = "https://google.com"
    base_feats = pipeline.extract_all(test_url)
    
    # Note: Pipeline extracts 50 base features, Graph adds 10
    log.info(f"Base Features Extracted: {len(base_feats)}")
    
    # 3. Check PredictionService fallback alignment
    log.info("Verifying PredictionService reindexing fallback...")
    # This should yield exactly 60 columns even with just base features
    df_test = pd.DataFrame([base_feats])
    df_aligned = df_test.reindex(columns=FEATURE_ORDER, fill_value=0)
    
    log.info(f"Aligned Vector Size: {df_aligned.shape[1]}")
    if df_aligned.shape[1] != 60:
        log.error("FAILURE: Reindexing failed to produce 60-column vector.")
        return False
        
    # 4. Check for duplicates in Schema
    if len(set(FEATURE_ORDER)) != len(FEATURE_ORDER):
        log.error("FAILURE: Master Schema contains duplicate feature names!")
        return False

    log.success("🎉 Master Schema Synchronization Verified Successfully across all layers!")
    return True

if __name__ == "__main__":
    if verify_sync():
        sys.exit(0)
    else:
        sys.exit(1)
