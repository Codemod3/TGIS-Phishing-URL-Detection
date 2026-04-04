import joblib
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.core.logger import log
from src.core.schema import FEATURE_ORDER

class EnsemblePredictor:
    """
    Combines predictions from Random Forest, XGBoost, and TGIS Trust Scores.
    Uses a weighted consensus mechanism as defined in the architecture.
    """

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        self.rf_path = os.path.join(model_dir, "random_forest.pkl")
        self.xgb_path = os.path.join(model_dir, "xgboost.pkl")
        
        log.info("Loading Ensemble Models...")
        
        if not os.path.exists(self.rf_path) or not os.path.exists(self.xgb_path):
            log.error(f"Models not found in {model_dir}. Ensure training is completed first.")
            self.rf_model = None
            self.xgb_model = None
        else:
            self.rf_model = joblib.load(self.rf_path)
            self.xgb_model = joblib.load(self.xgb_path)
            log.success("Ensemble Models Loaded Successfully.")

    def predict(self, features: Any, tgis_score: float = 0.5) -> Dict[str, Any]:
        """
        Produce a phishing verdict by aggregating clinical ML and graph-based trust scores.
        Automatically aligns incoming features with the order expected by the trained models.
        
        Args:
            features (Union[Dict, pd.DataFrame, np.ndarray]): The features to analyze.
            tgis_score (float): The trust score from Trust Propagation.
            
        Returns:
            Dict[str, Any]: Detailed prediction results.
        """
        if self.rf_model is None or self.xgb_model is None:
            return {'error': 'Models not loaded'}

        # 1. Feature Alignment Logic
        # Convert to DataFrame if needed and align columns strictly with training order
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            # If already an array, we assume it's pre-aligned (e.g. by PredictionService)
            df = None
            X = features

        if df is not None:
            # Extract the expected feature names from the trained model
            # This ensures O(1) alignment even if the dictionary keys were shuffled
            if hasattr(self.rf_model, 'feature_names_in_'):
                expected_cols = list(self.rf_model.feature_names_in_)
                actual_cols = list(df.columns)
                
                # DEBUG: Print out the exact mismatch to your terminal for troubleshooting
                missing_from_live = set(expected_cols) - set(actual_cols)
                extra_in_live = set(actual_cols) - set(expected_cols)
                
                if missing_from_live:
                    log.warning(f"🚨 FEATURE MISMATCH: Live API forgot to extract: {missing_from_live}")
                if extra_in_live:
                    log.warning(f"🚨 FEATURE MISMATCH: Live API extracted extra features: {extra_in_live}")
                
                # BRUTE FORCE ALIGNMENT: Force the live data to match the expected columns
                # This perfectly sorts them and fills any missing columns with 0
                df = df.reindex(columns=expected_cols, fill_value=0)
                X = df.values
            else:
                log.warning("Model lacks feature_names_in_. Using master schema for alignment.")
                df = df.reindex(columns=FEATURE_ORDER, fill_value=0)
                X = df.values

        # Ensure features are in the right shape (1, N)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # 2. Get Probabilities (Class 1 = Phishing)
        rf_proba = self.rf_model.predict_proba(X)[0, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[0, 1]
        
        # 3. Dynamic Weighted Calculation
        # Identify if we are in a 'cold-start' scenario for TGIS (isolated domain)
        # Find the index of domain_cluster_size in our standardized schema
        cluster_idx = FEATURE_ORDER.index('domain_cluster_size')
        cluster_size = X[0, cluster_idx]
        is_cold_start = (tgis_score == 0.5) or (cluster_size <= 1)
        
        if is_cold_start:
            log.debug("TGIS Cold-Start detected. Using 50/50 ML weighted split.")
            final_score = (0.5 * rf_proba) + (0.5 * xgb_proba)
            tgis_weight = 0.0
        else:
            # Use architectural consensus: 40% RF, 40% XGB, 20% TGIS
            final_score = (0.4 * rf_proba) + (0.4 * xgb_proba) + (0.2 * (1 - tgis_score))
            tgis_weight = 0.2
            
        # 4. Apply 0.5 Threshold
        prediction = 'phishing' if final_score > 0.5 else 'safe'
        
        result = {
            'prediction': prediction,
            'confidence': float(final_score if final_score > 0.5 else 1 - final_score),
            'rf_score': float(rf_proba),
            'xgb_score': float(xgb_proba),
            'tgis_score': float(tgis_score),
            'tgis_weight': tgis_weight,
            'final_score': float(final_score),
            'is_cold_start': is_cold_start
        }
        
        log.debug(f"Ensemble Verdict: {prediction} (score={final_score:.4f}, weight_tgis={tgis_weight})")
        return result
