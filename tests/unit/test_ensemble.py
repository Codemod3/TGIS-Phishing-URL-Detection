import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import os
from src.models.ensemble import EnsemblePredictor

class TestEnsemblePredictor(unittest.TestCase):

    @patch('joblib.load')
    @patch('os.path.exists')
    def setUp(self, mock_exists, mock_load):
        # Set up mocks for models
        mock_exists.return_value = True
        
        self.mock_rf = MagicMock()
        self.mock_xgb = MagicMock()
        
        # Mapping for mock_load to return different models
        mock_load.side_effect = [self.mock_rf, self.mock_xgb]
        
        self.ensemble = EnsemblePredictor(model_dir="dummy_dir")
        self.dummy_features = np.zeros((1, 60))

    def test_phishing_ensemble(self):
        # High ML proba, Low Trust (0.0)
        self.mock_rf.predict_proba.return_value = np.array([[0.0, 1.0]])
        self.mock_xgb.predict_proba.return_value = np.array([[0.0, 1.0]])
        
        result = self.ensemble.predict(self.dummy_features, tgis_score=0.0)
        
        # 0.4*1.0 + 0.4*1.0 + 0.2*(1-0) = 0.4 + 0.4 + 0.2 = 1.0
        self.assertEqual(result['prediction'], 'phishing')
        self.assertEqual(result['final_score'], 1.0)

    def test_safe_ensemble(self):
        # Low ML proba, High Trust (1.0)
        self.mock_rf.predict_proba.return_value = np.array([[1.0, 0.0]])
        self.mock_xgb.predict_proba.return_value = np.array([[1.0, 0.0]])
        
        result = self.ensemble.predict(self.dummy_features, tgis_score=1.0)
        
        # 0.4*0.0 + 0.4*0.0 + 0.2*(1-1) = 0.0
        self.assertEqual(result['prediction'], 'safe')
        self.assertEqual(result['final_score'], 0.0)

    def test_borderline_safe_recovery(self):
        # Borderline ML (0.6 each), High Trust (1.0)
        self.mock_rf.predict_proba.return_value = np.array([[0.4, 0.6]])
        self.mock_xgb.predict_proba.return_value = np.array([[0.4, 0.6]])
        
        result = self.ensemble.predict(self.dummy_features, tgis_score=1.0)
        
        # 0.4*0.6 + 0.4*0.6 + 0.2*(1-1) = 0.24 + 0.24 = 0.48
        # Should be 'safe' because trust score is high (threshold=0.5)
        self.assertEqual(result['prediction'], 'safe')
        self.assertLess(result['final_score'], 0.5)

    def test_borderline_phishing_override(self):
        # Borderline ML (0.4 each), Low Trust (0.0)
        self.mock_rf.predict_proba.return_value = np.array([[0.6, 0.4]])
        self.mock_xgb.predict_proba.return_value = np.array([[0.6, 0.4]])
        
        result = self.ensemble.predict(self.dummy_features, tgis_score=0.0)
        
        # 0.4*0.4 + 0.4*0.4 + 0.2*(1-0) = 0.16 + 0.16 + 0.2 = 0.52
        # Should be 'phishing' because trust score is low
        self.assertEqual(result['prediction'], 'phishing')
        self.assertGreater(result['final_score'], 0.5)

if __name__ == "__main__":
    unittest.main()
