from src.features.pipeline import FeaturePipeline
from src.core.logger import log

def verify_pipeline():
    pipeline = FeaturePipeline()
    
    test_urls = [
        "signin.ebay.com.authentication.net",
        "amaz0n.co"
    ]
    
    log.info("Starting Full Feature Engineering Pipeline verification...")
    
    for url in test_urls:
        log.info(f"--- 🌀 Executing Full Pipeline: {url} ---")
        all_features = pipeline.extract_all(url)
        
        # Check total count
        log.success(f"Total Features Extracted: {len(all_features)}")
        
        # Print first 5 and last 5 to verify merge
        feature_names = list(all_features.keys())
        log.info("Sample Feature Values:")
        for name in feature_names[:5]:
            log.info(f"  {name}: {all_features[name]}")
        log.info("  ...")
        for name in feature_names[-5:]:
            log.info(f"  {name}: {all_features[name]}")

if __name__ == "__main__":
    verify_pipeline()
