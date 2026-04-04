from fastapi.testclient import TestClient
from api.main import app
from src.core.logger import log

client = TestClient(app)

def test_api_integration():
    """
    Integration tests for the core FastAPI endpoints.
    Verifies routing, schema validation, and service integration.
    """
    log.info("Starting FastAPI Integration Tests...")
    
    # 1. Verify Root Endpoint
    log.info("Testing [GET /]...")
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"
    log.success("Root endpoint reachable.")

    # 2. Verify Health Check
    log.info("Testing [GET /health]...")
    response = client.get("/health")
    assert response.status_code == 200
    health_data = response.json()
    assert health_data["status"] == "healthy"
    assert "uptime_seconds" in health_data
    log.success("Health check functional.")

    # 3. Verify Prediction Endpoint
    log.info("Testing [POST /api/v1/predict]...")
    payload = {
        "url": "https://google.com",
        "include_explanation": True,
        "fetch_content": False
    }
    response = client.post("/api/v1/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["url"] == payload["url"]
    assert data["prediction"] in ["safe", "phishing"]
    assert "confidence" in data
    assert "model_scores" in data
    
    log.success(f"Prediction endpoint functional. Verdict: {data['prediction']}")

    print("\n" + "="*50)
    print("✨ ALL API VERIFICATION TESTS PASSED")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_api_integration()
