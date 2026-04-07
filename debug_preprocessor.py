import joblib
import pandas as pd
import os
from src.core.schema import FEATURE_ORDER

def debug_preprocessor():
    model_dir = "data/models"
    imputer_path = os.path.join(model_dir, 'imputer.joblib')
    
    if not os.path.exists(imputer_path):
        print(f"ERROR: {imputer_path} not found.")
        return
        
    imputer = joblib.load(imputer_path)
    
    if hasattr(imputer, 'feature_names_in_'):
        names_in = list(imputer.feature_names_in_)
        print(f"Preprocessor expects {len(names_in)} features.")
        
        missing_in_code = set(names_in) - set(FEATURE_ORDER)
        extra_in_code = set(FEATURE_ORDER) - set(names_in)
        
        if missing_in_code:
            print(f"🚨 Preprocessor expects features NOT in FEATURE_ORDER: {missing_in_code}")
        if extra_in_code:
            print(f"🚨 FEATURE_ORDER has features NOT in Preprocessor: {extra_in_code}")
            
        if names_in != FEATURE_ORDER:
            print("🚨 Names match but ORDER is different!")
            for i, (n, f) in enumerate(zip(names_in, FEATURE_ORDER)):
                if n != f:
                    print(f"Mismatch at index {i}: Expected {n}, Got {f}")
                    break
        else:
            print("✅ Preprocessor names match FEATURE_ORDER exactly.")
    else:
        print("⚠️ Preprocessor has NO feature names (fitted on numpy array).")

if __name__ == "__main__":
    debug_preprocessor()
