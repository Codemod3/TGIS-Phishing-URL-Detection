import unittest
import pandas as pd
import numpy as np
import os
import shutil
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter

class TestDataHandling(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/tmp_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.csv_path = os.path.join(self.test_dir, "test_features.csv")
        
        # Create a mock dataset (100 samples, 4 features + label)
        # Class Balance: 10 phishing (1), 90 safe (0) - Imbalanced
        data = {
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'f3': np.random.randn(100),
            'f4': np.random.randn(100),
            'label': [1]*10 + [0]*90
        }
        # Introduce some missing values
        data['f1'][0] = np.nan
        data['f2'][1:5] = np.nan
        
        self.df = pd.DataFrame(data)
        self.df.to_csv(self.csv_path, index=False)
        
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.splitter = DataSplitter()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_loader(self):
        df_loaded = self.loader.load_data(self.csv_path)
        self.assertIsNotNone(df_loaded)
        self.assertEqual(len(df_loaded), 100)
        self.assertIn('label', df_loaded.columns)

    def test_preprocessor(self):
        X = self.df.drop('label', axis=1)
        X_processed = self.preprocessor.fit_transform(X)
        
        # Verify no NaNs
        self.assertFalse(np.isnan(X_processed).any())
        
        # Verify scaling (mean ~0, std ~1)
        self.assertAlmostEqual(np.mean(X_processed), 0, places=1)
        self.assertAlmostEqual(np.std(X_processed), 1, places=1)
        
        # Test saving and loading
        save_dir = os.path.join(self.test_dir, "scaler")
        self.preprocessor.save(save_dir)
        
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load(save_dir)
        self.assertTrue(new_preprocessor.is_fitted)

    def test_splitter_and_smote(self):
        X = self.df.drop('label', axis=1)
        y = self.df['label']
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.train_val_test_split(X, y)
        
        # Proportions check (flexible due to rounding/stratification)
        self.assertIn(len(X_train), [69, 70, 71])
        self.assertIn(len(X_val), [14, 15, 16])
        self.assertIn(len(X_test), [14, 15, 16])
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), 100)
        
        # Preprocess before SMOTE (required due to NaNs)
        X_train_processed = self.preprocessor.fit_transform(pd.DataFrame(X_train))
        
        # Check initial imbalance in y_train
        y_train_phish = sum(y_train)
        self.assertLess(y_train_phish, len(y_train) / 2)
        
        # Apply SMOTE
        X_train_sm, y_train_sm = self.splitter.apply_smote(X_train_processed, y_train)
        
        # Verify balancing
        counts = np.bincount(y_train_sm)
        self.assertEqual(counts[0], counts[1]) # Should be balanced
        self.assertGreater(len(y_train_sm), 70) # More samples after oversampling

if __name__ == "__main__":
    unittest.main()
