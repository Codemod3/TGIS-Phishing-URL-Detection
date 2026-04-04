
import json
from api.schemas.request import PredictionRequest, BatchPredictionRequest
from api.schemas.response import PredictionResponse, BatchPredictionResponse, HealthResponse

def test_prediction_request():
    payload = {
        "url": "https://example.com",
        "include_explanation": True,
        "fetch_content": False
    }
    req = PredictionRequest(**payload)
    print("PredictionRequest validated.")

def test_batch_prediction_request():
    payload = {
        "urls": ["https://example1.com", "https://example2.com"],
        "max_urls": 100
    }
    req = BatchPredictionRequest(**payload)
    print("BatchPredictionRequest validated.")

def test_prediction_response():
    # Constructing a complex response manually to test all nested models
    payload = {
        "url": "https://example.com",
        "prediction": "phishing",
        "confidence": 0.87,
        "risk_score": 0.87,
        "processing_time_ms": 234,
        "model_scores": {
            "random_forest": 0.92,
            "xgboost": 0.85,
            "tgis": 0.15,
            "ensemble": 0.87
        },
        "api_checks": {
            "safe_browsing": {
                "is_flagged": True,
                "threat_types": ["SOCIAL_ENGINEERING"]
            },
            "whois": {
                "domain_age_days": 15,
                "registrar": "Namecheap"
            }
        },
        "graph_analysis": {
            "trust_score": 0.15,
            "cluster_risk": "high",
            "suspicious_neighbors": 12
        },
        "top_features": [
            {"name": "domain_age_days", "value": 15, "importance": 0.23},
            {"name": "tld_suspicious", "value": 1, "importance": 0.18}
        ],
        "explanation": {
            "shap_values": {"domain_age": 0.5, "tld": 0.3},
            "reason": "Domain is very new."
        }
    }
    res = PredictionResponse(**payload)
    print("PredictionResponse validated.")

def test_batch_prediction_response():
    payload = {
        "total_urls": 2,
        "processed": 2,
        "failed": 0,
        "phishing_count": 1,
        "safe_count": 1,
        "processing_time_ms": 500,
        "results": [
            {
                "url": "https://example1.com",
                "prediction": "phishing",
                "confidence": 0.87,
                "risk_score": 0.87
            }
        ]
    }
    res = BatchPredictionResponse(**payload)
    print("BatchPredictionResponse validated.")

def test_health_response():
    payload = {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": 12345,
        "components": {
            "models": {
                "random_forest": "loaded",
                "xgboost": "loaded",
                "graph": "loaded"
            },
            "external_apis": {
                "safe_browsing": "reachable",
                "whois": "reachable"
            },
            "cache": "connected"
        },
        "metrics": {
            "total_predictions": 1234,
            "avg_response_time_ms": 245,
            "cache_hit_rate": 0.76
        }
    }
    res = HealthResponse(**payload)
    print("HealthResponse validated.")

if __name__ == "__main__":
    test_prediction_request()
    test_batch_prediction_request()
    test_prediction_response()
    test_batch_prediction_response()
    test_health_response()
    print("All schemas verified successfully!")
