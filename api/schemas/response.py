from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any

# Prediction Response Components
class ModelScores(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    random_forest: float = Field(..., description="Random Forest model prediction score")
    xgboost: float = Field(..., description="XGBoost model prediction score")
    tgis: float = Field(..., description="TGIS graph-based score")
    ensemble: float = Field(..., description="Final ensemble prediction score")

class SafeBrowsingCheck(BaseModel):
    is_flagged: bool = Field(..., description="Whether Google Safe Browsing flagged the URL")
    threat_types: List[str] = Field(..., description="Types of threats found (if any)")

class WhoisCheck(BaseModel):
    domain_age_days: Optional[int] = Field(None, description="Age of the domain in days")
    registrar: Optional[str] = Field(None, description="Domain registrar name")

class ApiChecks(BaseModel):
    safe_browsing: SafeBrowsingCheck
    whois: WhoisCheck

class GraphAnalysis(BaseModel):
    trust_score: float = Field(..., description="Calculated trust score from the domain graph")
    cluster_risk: str = Field(..., description="Risk level of the domain cluster")
    suspicious_neighbors: int = Field(..., description="Number of malicious URLs in the same cluster")

class TopFeature(BaseModel):
    name: str = Field(..., description="Name of the feature")
    value: Any = Field(..., description="Value of the feature")
    importance: float = Field(..., description="Importance weight of this feature in the prediction")

class Explanation(BaseModel):
    shap_values: Dict[str, float] = Field(..., description="SHAP contribution values for top features")
    reason: str = Field(..., description="Human-readable explanation of why this prediction was made")

class PredictionResponse(BaseModel):
    """Detailed prediction response for a single URL."""
    model_config = ConfigDict(protected_namespaces=())
    url: str = Field(..., description="The URL analyzed")
    prediction: str = Field(..., description="Final prediction: 'phishing' or 'safe'")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    risk_score: float = Field(..., description="Calculated risk score (0-1)")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    model_scores: ModelScores
    api_checks: ApiChecks
    graph_analysis: GraphAnalysis
    top_features: List[TopFeature]
    explanation: Optional[Explanation] = None

# Batch Prediction Components
class BatchResult(BaseModel):
    url: str = Field(..., description="The URL analyzed")
    prediction: str = Field(..., description="Final prediction: 'phishing' or 'safe'")
    confidence: float = Field(..., description="Confidence score")
    risk_score: float = Field(..., description="Risk score")

class BatchPredictionResponse(BaseModel):
    """Summary response for bulk URL prediction."""
    total_urls: int = Field(..., description="Total number of URLs in request")
    processed: int = Field(..., description="Number of URLs successfully processed")
    failed: int = Field(..., description="Number of failures")
    phishing_count: int = Field(..., description="Count of URLs predicted as phishing")
    safe_count: int = Field(..., description="Count of URLs predicted as safe")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    results: List[BatchResult]

# Health Check Components
class ModelStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    random_forest: str
    xgboost: str
    graph: str

class ExternalApiStatus(BaseModel):
    safe_browsing: str
    whois: str

class HealthComponents(BaseModel):
    models: ModelStatus
    external_apis: ExternalApiStatus
    cache: str

class HealthMetrics(BaseModel):
    total_predictions: int
    avg_response_time_ms: int
    cache_hit_rate: float

class HealthResponse(BaseModel):
    """System health status response."""
    status: str = Field(..., description="Overall system status (e.g., 'healthy')")
    version: str = Field(..., description="System version")
    uptime_seconds: int = Field(..., description="System uptime in seconds")
    components: HealthComponents
    metrics: HealthMetrics
