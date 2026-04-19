"""
Training Script for Chicago Crime Analysis Model

This script runs the complete training pipeline and saves the trained model.
Usage: python train_model.py
"""

import os
import sys
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import preprocess_pipeline
from src.model import full_training_pipeline


def main():
    """Run the complete training pipeline."""
    print("=" * 60)
    print("Chicago Crime Analysis - Model Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    df_cleaned, df_final, encoders = preprocess_pipeline(nrows=50000)
    
    # Step 2: Train models
    print("\n[Step 2] Training models...")
    results = full_training_pipeline(
        df_final,
        model_save_path='models/xgboost_model.pkl'
    )
    
    # Step 3: Save encoders for later use
    print("\n[Step 3] Saving encoders...")
    os.makedirs('models', exist_ok=True)
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("Encoders saved to models/encoders.pkl")
    
    # Step 4: Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel saved to: models/xgboost_model.pkl")
    print(f"Encoders saved to: models/encoders.pkl")
    print(f"\nDataset Statistics:")
    print(f"  - Total records processed: {len(df_final)}")
    print(f"  - Features: {df_final.shape[1] - 1}")
    print(f"  - Target distribution:")
    print(f"    Arrests (1): {(df_final['Arrest'] == 1).sum()}")
    print(f"    No Arrests (0): {(df_final['Arrest'] == 0).sum()}")
    
    print("\nNext steps:")
    print("  1. Run SQL pipeline: python sql_integration.py")
    print("  2. Test predictions: python test_prediction.py")
    print("  3. Explore notebook: jupyter notebook notebooks/crime_analysis.ipynb")


if __name__ == "__main__":
    main()
