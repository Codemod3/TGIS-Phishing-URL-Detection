from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    """Request model for single URL prediction."""
    url: str = Field(..., description="The URL to analyze", example="https://example.com")
    include_explanation: bool = Field(default=True, description="Whether to include SHAP-based explanations")
    fetch_content: bool = Field(default=False, description="Whether to fetch and analyze page content")

class BatchPredictionRequest(BaseModel):
    """Request model for bulk URL analysis."""
    urls: List[str] = Field(..., description="List of URLs to analyze")
    max_urls: int = Field(default=100, description="Maximum number of URLs per request", le=100)
