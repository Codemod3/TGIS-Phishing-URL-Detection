import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional
from src.core.logger import log

class DataSplitter:
    """
    Handles dataset splitting (Train/Val/Test) and class balancing (SMOTE).
    """

    def train_val_test_split(self, X: pd.DataFrame, y: pd.Series, 
                             test_size: float = 0.15, val_size: float = 0.15, 
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into Train, Validation, and Test sets (70/15/15).
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Binary labels.
            test_size (float): Portion of data for test set.
            val_size (float): Portion of training data for validation set.
            random_state (int): Seed for reproducibility.
            
        Returns:
            Tuple[X_train, X_val, X_test, y_train, y_val, y_test]
        """
        log.info(f"Splitting data (70/15/15) for {X.shape[0]} samples...")
        
        # 1. Split out the test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 2. Split out the validation set from the remaining training data
        # Adjust val_size relative to the remaining data
        val_size_adj = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adj, random_state=random_state, stratify=y_train_val
        )
        
        log.success(f"Split completed: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def apply_smote(self, X_train: np.ndarray, y_train: np.ndarray, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance the minority class (phishing URLs) in the training set only.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            random_state (int): Seed for reproducibility.
            
        Returns:
            Tuple[X_train_reshaped, y_train_reshaped]
        """
        log.info("Applying SMOTE to balance training classes...")
        
        # Check initial distribution
        unique, counts = np.unique(y_train, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        log.debug(f"Initial class distribution: {counts_dict}")
        
        # 🛡️ Safety check: SMOTE requires k_neighbors + 1 samples. 
        # Default is 5 neighbors, so we need at least 6 samples.
        min_class_size = min(counts)
        if min_class_size < 2:
            log.warning(f"Minority class size ({min_class_size}) is too small for SMOTE. Skipping balancing.")
            return X_train, y_train
            
        # Dynamically adjust k_neighbors if we have 2-5 samples
        k_neighbors = min(5, min_class_size - 1)
        if k_neighbors < 5:
            log.info(f"Small minority class detected ({min_class_size}). Adjusting SMOTE k_neighbors to {k_neighbors}.")
            
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        
        # Check final distribution
        unique_sm, counts_sm = np.unique(y_train_sm, return_counts=True)
        log.success(f"SMOTE applied. Balanced distribution: {dict(zip(unique_sm, counts_sm))}")
        
        return X_train_sm, y_train_sm
