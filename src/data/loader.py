import pandas as pd
import os
from typing import Optional
from src.core.logger import log

class DataLoader:
    """
    Handles loading of feature datasets from various formats.
    Supports CSV and Parquet.
    """
    
    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load data from a given path.
        
        Args:
            file_path (str): Path to the dataset.
            
        Returns:
            Optional[pd.DataFrame]: The loaded DataFrame or None if error.
        """
        if not os.path.exists(file_path):
            log.error(f"File not found: {file_path}")
            return None
            
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(file_path)
                log.info(f"Loaded CSV dataset: {file_path} ({len(df)} rows)")
            elif ext == '.parquet':
                df = pd.read_parquet(file_path)
                log.info(f"Loaded Parquet dataset: {file_path} ({len(df)} rows)")
            else:
                log.error(f"Unsupported file format: {ext}")
                return None
                
            if 'label' not in df.columns:
                log.warning(f"Dataset {file_path} is missing 'label' column.")
                
            return df
            
        except Exception as e:
            log.error(f"Failed to load data from {file_path}: {e}")
            return None
