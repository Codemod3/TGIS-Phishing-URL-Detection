import unittest
import numpy as np
import os
import shutil
import pandas as pd
from src.models.random_forest import PhishingRandomForest
from src.models.xgboost_model import PhishingXGBoost
from src.models.trainer import ModelTrainer

class TestModels(unittest.TestCase):

    def setUp(self):
        # Create dummy data for testing
        self.X = np.random.rand(50, 60)
        self.y = np.random.randint(0, 2, 50)
        
        self.test_dir = "tests/tmp_models"
        self.data_dir = os.path.join(self.test_dir, "data")
        self.model_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save dummy parquets for trainer test
        df = pd.DataFrame(self.X, columns=[f"f{i}" for i in range(60)])
        df['label'] = self.y
        df['url'] = [f"https://site{i}.com" for i in range(50)]
        
        df.iloc[:35].to_parquet(os.path.join(self.data_dir, "train.parquet"), index=False)
        df.iloc[35:].to_parquet(os.path.join(self.data_dir, "test.parquet"), index=False)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_random_forest_fit_predict(self):
        rf = PhishingRandomForest()
        rf.fit(self.X, self.y)
        preds = rf.predict(self.X)
        self.assertEqual(len(preds), 50)
        self.assertIn(0, preds)
        self.assertIn(1, preds)

    def test_xgboost_fit_predict(self):
        # Using a subset for val set
        xgb = PhishingXGBoost()
        eval_set = (self.X[:10], self.y[:10])
        xgb.fit(self.X[10:], self.y[10:], eval_set=eval_set)
        preds = xgb.predict(self.X)
        self.assertEqual(len(preds), 50)

    def test_trainer_pipeline(self):
        trainer = ModelTrainer(data_dir=self.data_dir, model_dir=self.model_dir)
        trainer.train_all()
        
        # Verify persistence
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "random_forest.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "xgboost.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "scaler.joblib")))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "imputer.joblib")))

if __name__ == "__main__":
    unittest.main()
