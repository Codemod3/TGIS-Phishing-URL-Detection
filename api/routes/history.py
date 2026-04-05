from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.database import get_db
from api.models import Prediction
from api.schemas.response import HistoryResponse
from src.core.logger import log

router = APIRouter(prefix="/api/v1", tags=["history"])

@router.get("/history", response_model=HistoryResponse)
def get_history(db: Session = Depends(get_db)):
    """
    Retrieve the last 100 analysis results from the database.
    Provides forensic history for the analytical dashboard.
    """
    log.info("API Request: GET /api/v1/history")
    
    try:
        # Query predictions, ordering by timestamp descending
        predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(100).all()
        
        return HistoryResponse(
            total=len(predictions),
            results=predictions
        )
    except Exception as e:
        log.error(f"Failed to fetch analysis history: {e}")
        # Returning an empty result set on error instead of hard failure 
        # to ensure dashboard stability
        return HistoryResponse(total=0, results=[])
