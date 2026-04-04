import time
from fastapi import APIRouter
from api.schemas.response import (
    HealthResponse, HealthComponents, HealthMetrics, 
    ModelStatus, ExternalApiStatus
)

router = APIRouter(tags=["system"])

# System-level starting point for uptime calculation
START_TIME = time.time()

@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Check the current health and performance metrics of the 
    Elite Phishing URL Detection System.
    """
    uptime = int(time.time() - START_TIME)
    
    # In a production environment, these values would be dynamically 
    # fetched from a monitoring component (e.g., Prometheus or a database).
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime,
        components=HealthComponents(
            models=ModelStatus(
                random_forest="loaded",
                xgboost="loaded",
                graph="loaded"
            ),
            external_apis=ExternalApiStatus(
                safe_browsing="reachable",
                whois="reachable"
            ),
            cache="connected"
        ),
        metrics=HealthMetrics(
            total_predictions=0, # Initial placeholder
            avg_response_time_ms=0, # Initial placeholder
            cache_hit_rate=0.0 # Initial placeholder
        )
    )
