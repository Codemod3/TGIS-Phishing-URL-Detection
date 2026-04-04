from xgboost import XGBClassifier
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src.core.logger import log

# Exact parameters from ARCHITECTURE.md
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1.5,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss'
}

class PhishingXGBoost:
    """
    XGBoost Classifier for Phishing URL Detection.
    Optimized for high-dimensional feature spaces and class imbalance.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or XGB_PARAMS
        self.model = XGBClassifier(**self.params)
        log.info("Initialized PhishingXGBoost with histogram-based tree method.")

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Train the XGBoost model. 
        Supports early stopping if eval_set (X_val, y_val) is provided.
        """
        log.info(f"Training XGBoost on {X.shape[0]} samples...")
        
        if eval_set:
            log.info("Early stopping enabled for XGBoost training.")
            self.model.fit(
                X, y, 
                eval_set=[eval_set], 
                verbose=False
            )
        else:
            self.model.fit(X, y)
        
        log.success("XGBoost Training Complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels (0 = Safe, 1 = Phishing)."""
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return a sorted dictionary of feature importances."""
        importances = self.model.feature_importances_
        # Feature names should be provided by the model if configured
        return dict(enumerate(importances))
