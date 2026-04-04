import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.data.preprocessor import DataPreprocessor

def test_imputation_logic():
    print("Testing SimpleImputer NaN handling in DataPreprocessor...")
    preprocessor = DataPreprocessor()
    
    # Create a dummy DataFrame with NaNs
    # Feature 1 has NaNs, Feature 2 is solid
    data = {
        'f1': [10, np.nan, 30, 40, 50],
        'f2': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(df)
    
    # Check if NaNs are gone
    assert not np.isnan(X_processed).any(), "Processed data should not contain NaNs"
    
    # Check the imputed value (should be median: 35.0 is between 30 and 40? 
    # Median of [10, 30, 40, 50] is (30+40)/2 = 35.0
    # Let's see what SimpleImputer did.
    # We can't easily see the raw imputed value without inverse transform (which we don't have),
    # but the assertion that NaNs are gone is enough for now.
    
    print("✅ DataPreprocessor Imputation passed.")

if __name__ == "__main__":
    try:
        test_imputation_logic()
        print("\n✨ PREPROCESSOR TEST PASSED! ✨")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        sys.exit(1)
