from fastapi import APIRouter, Depends, HTTPException
from api.schemas.request import PredictionRequest
from api.schemas.response import PredictionResponse
from api.services.prediction_service import PredictionService
from api.dependencies import get_prediction_service
from api.database import get_db
from sqlalchemy.orm import Session
from src.core.logger import log

router = APIRouter(prefix="/api/v1", tags=["prediction"])

@router.post("/predict", response_model=PredictionResponse)
def predict_url(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
    db: Session = Depends(get_db)
):
    """
    Analyze a URL and predict its phishing potential using structural, 
    domain, and TGIS graph features.
    """
    log.info(f"API Request: POST /api/v1/predict for {request.url}")
    
    try:
        # Forward to the PredictionService business logic
        return service.predict_single_url(request.url, db=db)
    except Exception as e:
        log.error(f"Prediction Pipeline Error: {str(e)}")
        # Formalizing internal error as HTTP 500
        raise HTTPException(
            status_code=500,
            detail=f"Phishing detection pipeline failed: {str(e)}"
        )
