import time
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

# Feature Pipeline & Graph Components
from src.features.pipeline import FeaturePipeline
from src.features.graph_features import GraphFeatureExtractor
# ... (rest of imports remain same)
from src.graph.builder import GraphBuilder
from src.graph.trust_propagation import calculate_trust_score

# Machine Learning Components
from src.models.ensemble import EnsemblePredictor
from src.data.preprocessor import DataPreprocessor

# External API Verification
from src.external.safe_browsing import SafeBrowsingClient
from src.external.whois_client import WHOISClient
from src.core.schema import FEATURE_ORDER

# API Schemas
from api.schemas.response import (
    PredictionResponse, ModelScores, ApiChecks, 
    SafeBrowsingCheck, WhoisCheck, GraphAnalysis, TopFeature, Explanation
)

from src.core.logger import log

class PredictionService:
    """
    Core business logic for the Phishing URL Detection System.
    Orchestrates feature extraction, graph analysis, and ensemble model inference.
    """

    def __init__(self, model_dir: str = "data/models", graph_path: str = "data/graphs/domain_graph.gpickle"):
        log.info("Initializing Prediction Service...")
        
        # 1. Feature Engineering Components
        self.feature_pipeline = FeaturePipeline()
        self.graph_extractor = GraphFeatureExtractor()
        
        # 2. Trust Graph (TGIS)
        self.graph_builder = GraphBuilder()
        self.graph_builder.load_graph(graph_path)
        
        # 3. Machine Learning Components
        self.ensemble_predictor = EnsemblePredictor(model_dir=model_dir)
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load(model_dir)
        
        # 4. External Verification Clients
        self.safe_browsing = SafeBrowsingClient()
        self.whois_client = WHOISClient()

    def predict_single_url(self, url: str) -> PredictionResponse:
        """
        Perform a full prediction cycle on a single URL.
        """
        start_time = time.time()
        log.info(f"--- 🛡️ Starting Prediction Request: {url} ---")
        
        try:
            # 1. Base Feature Extraction (Structural, Domain, Content)
            base_features = self.feature_pipeline.extract_all(url)
            
            # 2. Graph Feature Extraction (Trust Graph Metadata)
            graph_features = self.graph_extractor.extract(url, self.graph_builder.graph)
            
            # 3. Concatenate and Align Features (60 total)
            combined_dict = {**base_features, **graph_features}
            
            # 4. Vector Preprocessing & Model Inference
            # Ensure strict feature alignment using the model's internal schema
            # This prevents the "Feature names mismatch" 500 error in Preprocessor
            df_features = pd.DataFrame([combined_dict])
            
            if hasattr(self.ensemble_predictor.rf_model, 'feature_names_in_'):
                expected_cols = list(self.ensemble_predictor.rf_model.feature_names_in_)
                df_features = df_features.reindex(columns=expected_cols, fill_value=0)
            else:
                # Fallback to the master architectural order if model names aren't available
                df_features = df_features.reindex(columns=FEATURE_ORDER, fill_value=0)
            
            # Impute and Scale
            processed_vector = self.preprocessor.transform(df_features)
            
            # Get TGIS Trust Score
            parsed = urlparse(url)
            domain = parsed.netloc.split(':')[0]
            trust_scores = calculate_trust_score(self.graph_builder.graph)
            tgis_trust = trust_scores.get(domain, trust_scores.get(url, 0.5))
            
            # Execute Ensemble Prediction
            inference_results = self.ensemble_predictor.predict(processed_vector, tgis_score=tgis_trust)
            
            if 'error' in inference_results:
                raise RuntimeError(f"Model inference failed: {inference_results['error']}")

            # --- 💡 SANITIZATION LAYER: Conver NaNs to -1.0 before mapping to Pydantic ---
            # We keep the originals for internal logic, but sanitize for the response
            safe_base = self._sanitize_features(base_features)
            safe_graph = self._sanitize_features(graph_features)
            safe_results = self._sanitize_features(inference_results)
            safe_tgis_trust = -1.0 if math.isnan(tgis_trust) else tgis_trust

            # 5. External API Verification (Metadata Enrichment)
            sb_check = self.safe_browsing.check_url(url)
            whois_raw = self.whois_client.lookup(domain) or {}
            
            # 6. Calculate Metrics & Logic Summary
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # 7. Map Results to PredictionResponse Schema
            response = PredictionResponse(
                url=url,
                prediction=safe_results['prediction'],
                confidence=round(safe_results['confidence'], 4),
                risk_score=round(safe_results['final_score'], 4),
                processing_time_ms=processing_time_ms,
                model_scores=ModelScores(
                    random_forest=round(safe_results['rf_score'], 4),
                    xgboost=round(safe_results['xgb_score'], 4),
                    tgis=round(safe_results['tgis_score'], 4),
                    ensemble=round(safe_results['final_score'], 4)
                ),
                api_checks=ApiChecks(
                    safe_browsing=SafeBrowsingCheck(
                        is_flagged=sb_check.get('is_threat', False),
                        threat_types=sb_check.get('threat_types', [])
                    ),
                    whois=WhoisCheck(
                        domain_age_days=int(safe_base.get('domain_age_days', -1)),
                        registrar=whois_raw.get('registrar')
                    )
                ),
                graph_analysis=GraphAnalysis(
                    trust_score=safe_tgis_trust,
                    cluster_risk=self._classify_cluster_risk(safe_tgis_trust),
                    suspicious_neighbors=int(safe_graph.get('suspicious_neighbor_count', 0))
                ),
                top_features=self._identify_top_contributors(safe_base, safe_results),
                explanation=Explanation(
                    shap_values={}, 
                    reason=self._generate_logic_summary(safe_results, safe_base, safe_tgis_trust)
                )
            )
            
            log.success(f"--- ✅ Prediction Completed: {safe_results['prediction'].upper()} ---")
            return response

        except Exception as e:
            log.error(f"Critical error in PredictionService: {str(e)}")
            raise RuntimeError(f"The phishing detection pipeline encountered an error: {str(e)}") from e

    def _sanitize_features(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Iterates through features and converts NaNs to -1.0 for JSON compatibility."""
        sanitized = {}
        for k, v in features_dict.items():
            if isinstance(v, float) and math.isnan(v):
                sanitized[k] = -1.0
            else:
                sanitized[k] = v
        return sanitized

    def _classify_cluster_risk(self, trust_score: float) -> str:
        """Maps continuous trust score to a qualitative risk string."""
        if trust_score < 0: return "unknown" # Sanitized NaN value
        if trust_score < 0.35: return "high"
        if trust_score < 0.70: return "medium"
        return "low"

    def _identify_top_contributors(self, features: Dict[str, Any], results: Dict[str, Any]) -> List[TopFeature]:
        """Simplified feature importance mapping."""
        return [
            TopFeature(name="domain_age_days", value=features.get('domain_age_days', -1), importance=0.45),
            TopFeature(name="tgis_trust_score", value=results.get('tgis_score', -1.0), importance=0.35),
            TopFeature(name="tld_suspicious", value=features.get('tld_suspicious', 0), importance=0.20)
        ]

    def _generate_logic_summary(self, results: Dict[str, Any], features: Dict[str, Any], trust: float) -> str:
        verdict = results['prediction']
        confidence = results['confidence'] * 100
        
        if verdict == 'phishing':
            summary = f"Detected as Phishing with {confidence:.1f}% confidence. "
            if trust >= 0 and trust < 0.4:
                summary += "The domain source is linked to a low-trust neighborhood in our global graph. "
            age = features.get('domain_age_days', -1)
            if age >= 0 and age < 90:
                summary += f"The domain is relatively new ({int(age)} days old), a common trait of phishing campaigns."
        else:
            summary = f"Verified as Safe with {confidence:.1f}% confidence. "
            if trust > 0.8:
                summary += "The domain is well-established and trusted within the TGIS network."
        
        return summary
