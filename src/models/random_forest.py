from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Dict, Any, Optional
from src.core.logger import log

# Exact parameters from ARCHITECTURE.md
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'oob_score': True
}

class PhishingRandomForest:
    """
    Random Forest Classifier for Phishing URL Detection.
    Strictly follows the architectural configuration for consistency and reproducibility.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or RF_PARAMS
        self.model = RandomForestClassifier(**self.params)
        log.info(f"Initialized PhishingRandomForest with {self.params['n_estimators']} estimators.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the Random Forest model."""
        log.info(f"Training Random Forest on {X.shape[0]} samples...")
        self.model.fit(X, y)
        if self.params.get('oob_score'):
            log.success(f"RF Training Complete. OOB Score: {self.model.oob_score_:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels (0 = Safe, 1 = Phishing)."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """Return a sorted dictionary of feature importances."""
        importances = self.model.feature_importances_
        return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))
