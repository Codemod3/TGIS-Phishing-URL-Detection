import time
import os
import sys
from pprint import pprint

# Add the project root to sys.path to allow imports from 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.pipeline import FeaturePipeline
from src.core.logger import log

def test_pipeline_run():
    # 1. Initialize the FeaturePipeline
    pipeline = FeaturePipeline()
    
    # 2. Define test URLs
    test_urls = [
        "http://signin.ebay.com.authentication.net",
        "http://amaz0n.co"   
    ]
    
    print("\n" + "="*80)
    print("🛡️  Elite Phishing URL Detection System - Phase 2 Pipeline Test")
    print("="*80 + "\n")

    for url in test_urls:
        log.info(f"🚀 Starting individual test run for: {url}")
        
        # 3. Track execution time
        start_time = time.time()
        
        try:
            # 4. Run through the pipeline
            full_feature_vector = pipeline.extract_all(url)
            end_time = time.time()
            
            # 5. Pretty-print the results
            print(f"\n--- Results for: {url} ---")
            print(f"⏱️  Execution Time: {end_time - start_time:.4f} seconds")
            print(f"📊 Total Features Captured: {len(full_feature_vector)}")
            print("\nFull 50-Feature Vector:")
            pprint(full_feature_vector, sort_dicts=True, indent=4)
            print("\n" + "-"*80)
            
        except Exception as e:
            log.error(f"❌ Critical failure during pipeline execution for {url}: {e}")

if __name__ == "__main__":
    test_pipeline_run()
