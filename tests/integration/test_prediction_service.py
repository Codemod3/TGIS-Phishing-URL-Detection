import sys
import os
from api.services.prediction_service import PredictionService
from src.core.logger import log

def test_prediction_service():
    # Ensure PYTHONPATH is set so imports work
    log.info("Starting PredictionService integration test...")
    
    # Initialize Service
    try:
        service = PredictionService()
        
        # Test URL
        test_url = "https://google.com" # Should be safe
        log.info(f"Testing with known safe URL: {test_url}")
        
        response = service.predict_single_url(test_url)
        
        print("\n" + "="*50)
        print("PASSED: PredictionResponse received")
        print(f"URL: {response.url}")
        print(f"Verdict: {response.prediction}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Risk Score: {response.risk_score:.2f}")
        print(f"Time: {response.processing_time_ms}ms")
        print("="*50 + "\n")
        
        # Verify JSON serialization
        json_data = response.json()
        log.success("JSON serialization successful.")
        
    except Exception as e:
        log.error(f"Test failed spectacularly: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_prediction_service()
