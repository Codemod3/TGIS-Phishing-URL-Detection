import pandas as pd
import numpy as np
import joblib
import os
import sys
from pprint import pprint

# Add the project root to sys.path to allow imports from 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.trainer import ModelTrainer
from src.models.ensemble import EnsemblePredictor
from src.data.preprocessor import DataPreprocessor
from src.core.logger import log
from sklearn.metrics import classification_report

def test_ml_pipeline():
    data_processed_dir = "data/processed"
    models_dir = "data/models"
    
    # 1. Initialize and Run ModelTrainer
    log.info("Step 1: Training Models on Processed Data...")
    trainer = ModelTrainer(data_dir=data_processed_dir, model_dir=models_dir)
    trainer.train_all()
    
    # 2. Initialize EnsemblePredictor
    log.info("Step 2: Initializing Ensemble Predictor...")
    ensemble = EnsemblePredictor(model_dir=models_dir)
    
    # 3. Load Test Data for a Single Sample
    log.info("Step 3: Loading Test Dataset and Sampling a Row...")
    test_path = os.path.join(data_processed_dir, "test.parquet")
    if not os.path.exists(test_path):
        log.error(f"Test data not found at {test_path}. Ensure scripts/build_mini_dataset.py was run.")
        return
        
    df_test = pd.read_parquet(test_path)
    if df_test.empty:
        log.error("Test dataset is empty.")
        return

    # Take first sample
    sample_row = df_test.iloc[0]
    
    # Handle missing 'url' column gracefully
    url = sample_row.get('url', "Unknown/Placeholder URL")
    true_label = sample_row['label']
    
    # Prepare features for prediction (drop metadata)
    # The ModelTrainer already preprocessed the data before saving it for tests, 
    # but in a real-world scenario, we'd need to scale it.
    X_sample_raw = df_test.drop(['label', 'url'], axis=1, errors='ignore').iloc[0:1]
    
    # Load the fitted preprocessor from the models directory
    preprocessor = DataPreprocessor()
    preprocessor.load(models_dir)
    X_sample_processed = preprocessor.transform(X_sample_raw)

    # 4. Perform Ensemble Prediction
    log.info(f"Step 4: Running Inference on Sample URL: {url}")
    # Mock a TGIS trust score for this test
    # (High trust = 1.0, Suspicious = 0.0)
    mock_tgis_score = 0.5 if true_label == 1 else 0.9
    
    prediction_result = ensemble.predict(X_sample_processed, tgis_score=mock_tgis_score)
    
    # 5. Reporting
    print("\n" + "="*60)
    print("🎯 ENSEMBLE PREDICTION RESULT")
    print("="*60)
    print(f"URL: {url}")
    print(f"Actual Label: {'Phishing' if true_label == 1 else 'Safe'}")
    pprint(prediction_result)
    print("="*60)
    
    # 6. Global Model Metrics (from Trainer)
    # We re-run metrics here briefly for pprint-ing as requested
    log.info("Step 5: Printing Model Evaluation Metrics...")
    
    # Load test set for evaluation
    X_test_all = df_test.drop(['label', 'url'], axis=1, errors='ignore')
    y_test_all = df_test['label']
    X_test_processed = preprocessor.transform(X_test_all)
    
    rf_preds = ensemble.rf_model.predict(X_test_processed)
    xgb_preds = ensemble.xgb_model.predict(X_test_processed)
    
    print("\n[Random Forest Metrics]")
    print(classification_report(y_test_all, rf_preds))
    
    print("\n[XGBoost Metrics]")
    print(classification_report(y_test_all, xgb_preds))

if __name__ == "__main__":
    test_ml_pipeline()
