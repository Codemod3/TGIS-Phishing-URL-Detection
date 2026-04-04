import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.models.random_forest import PhishingRandomForest
from src.models.xgboost_model import PhishingXGBoost
from src.core.logger import log
from src.core.schema import FEATURE_ORDER

class ModelTrainer:
    """
    Orchestrates the complete ML training pipeline.
    Loads data, preprocesses, balances, trains models, and evaluates results.
    """

    def __init__(self, data_dir: str = "data/processed", model_dir: str = "data/models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.splitter = DataSplitter()
        
        self.rf_model = PhishingRandomForest()
        self.xgb_model = PhishingXGBoost()

    def train_all(self):
        """Execute the full training pipeline."""
        log.info("--- 🚀 Initializing Model Training Pipeline ---")
        
        # 1. Load Data
        log.info("Loading training and testing datasets...")
        train_path = os.path.join(self.data_dir, "train.parquet")
        test_path = os.path.join(self.data_dir, "test.parquet")
        
        df_train_raw = self.loader.load_data(train_path)
        df_test_raw = self.loader.load_data(test_path)
        
        if df_train_raw is None or df_test_raw is None:
            log.error("Data loading failed. Terminating training pipeline.")
            return

        # 2. Extract Features and Labels
        log.info(f"Standardizing training schema ({len(FEATURE_ORDER)} columns)...")
        X_train_raw = df_train_raw.drop(['label', 'url'], axis=1, errors='ignore')
        X_train_raw = X_train_raw.reindex(columns=FEATURE_ORDER, fill_value=0)
        y_train_raw = df_train_raw['label']
        
        X_test_raw = df_test_raw.drop(['label', 'url'], axis=1, errors='ignore')
        X_test_raw = X_test_raw.reindex(columns=FEATURE_ORDER, fill_value=0)
        y_test_raw = df_test_raw['label']

        # 3. Preprocessing (Fit on Train, Transform Test)
        log.info("Preprocessing and Normalizing Feature Vectors...")
        X_train_processed = self.preprocessor.fit_transform(X_train_raw)
        X_test_processed = self.preprocessor.transform(X_test_raw)
        
        # Save Preprocessor artifacts
        self.preprocessor.save(self.model_dir)

        # 4. Split for Validation and Apply SMOTE
        # Split train_raw further to get a validation set for XGBoost early stopping
        X_train_sub, X_val, X_test_ignore, y_train_sub, y_val, y_test_ignore = self.splitter.train_val_test_split(
            pd.DataFrame(X_train_processed), y_train_raw, test_size=0.1, val_size=0.1
        )
        # Note: splitter.train_val_test_split returns 70/15/15 by default, 
        # but here we just need a simple split. We'll use apply_smote on X_train_sub.
        
        X_train_balanced, y_train_balanced = self.splitter.apply_smote(X_train_sub, y_train_sub)

        # 5. Train Random Forest
        log.info("Training Random Forest...")
        self.rf_model.fit(X_train_balanced, y_train_balanced)
        
        # 6. Train XGBoost (with early stopping on validation)
        log.info("Training XGBoost...")
        self.xgb_model.fit(X_train_balanced, y_train_balanced, eval_set=(X_val, y_val))

        # 7. Evaluate on Test Set
        self._evaluate_and_log(X_test_processed, y_test_raw)

        # 8. Persistence
        log.info(f"Saving model artifacts to {self.model_dir}...")
        joblib.dump(self.rf_model.model, os.path.join(self.model_dir, "random_forest.pkl"))
        joblib.dump(self.xgb_model.model, os.path.join(self.model_dir, "xgboost.pkl"))
        
        log.success("🎉 Model Training Pipeline Completed Successfully.")

    def _evaluate_and_log(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate models and print key metrics."""
        log.info("\n" + "="*50 + "\n📈 FINAL TEST SET EVALUATION\n" + "="*50)
        
        models = [
            ("Random Forest", self.rf_model),
            ("XGBoost", self.xgb_model)
        ]
        
        for name, model_wrapper in models:
            preds = model_wrapper.predict(X_test)
            proba = model_wrapper.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, proba)
            
            log.info(f"[{name}] Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            log.debug(f"\n{classification_report(y_test, preds)}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all()
