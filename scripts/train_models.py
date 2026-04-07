import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.trainer import ModelTrainer
from src.core.logger import log

def main():
    log.info("Starting System-wide Model Re-training...")
    try:
        # Initializing trainer with default paths
        trainer = ModelTrainer(
            data_dir="data/processed", 
            model_dir="data/models"
        )
        
        # This will fit and save the imputer, scaler, and models
        # ensuring they all use the CURRENT environment's scikit-learn version.
        trainer.train_all()
        
        log.success("Re-training complete. Model artifacts are now synchronized with this environment.")
    except Exception as e:
        log.error(f"Re-training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
