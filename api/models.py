import uuid
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Prediction(Base):
    """
    ORM Model for tracking Phishing URL detections.
    Stores core verdict metrics and metadata.
    """
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(Text, nullable=False, index=True)
    prediction_label = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    tgis_trust_score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # One-to-one relationship with the raw feature vector
    # uselist=False ensures its single-object mapping
    feature_vector = relationship("FeatureVector", back_populates="prediction", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Prediction(url='{self.url[:30]}...', label='{self.prediction_label}')>"

class FeatureVector(Base):
    """
    ORM Model for the full 60-feature vector.
    Separated from the main prediction table to optimize query performance
    when forensic details are not required.
    """
    __tablename__ = "feature_vectors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id"), unique=True, nullable=False)
    
    # Using JSONB for high-performance querying in PostgreSQL
    features = Column(JSONB, nullable=False)
    
    # Back association to the parent prediction
    prediction = relationship("Prediction", back_populates="feature_vector")

    def __repr__(self):
        return f"<FeatureVector(prediction_id='{self.prediction_id}')>"
