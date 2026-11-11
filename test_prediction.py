"""
Test Prediction Script for Chicago Crime Analysis

This script allows you to test the trained model with custom inputs.
Usage: python test_prediction.py
"""

import os
import sys
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import preprocess_pipeline
from src.model import load_model, predict_arrest_likelihood


def main():
    """Test the model with sample predictions."""
    print("=" * 60)
    print("Chicago Crime Analysis - Test Predictions")
    print("=" * 60)
    
    model_path = 'models/xgboost_model.pkl'
    encoder_path = 'models/encoders.pkl'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train_model.py first.")
        sys.exit(1)
        
    print(f"Loading trained model from '{model_path}'...")
    model = load_model(model_path)
    
    # Load encoders if available; otherwise perform minimal preprocessing
    if os.path.exists(encoder_path):
        print(f"Loading pre-computed encoders from '{encoder_path}'...")
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
        df_cleaned = None
    else:
        print(f"Encoders not found at '{encoder_path}'. Extracting encoders via preprocessing...")
        df_cleaned, _, encoders = preprocess_pipeline(nrows=10000)
    
    # Test scenarios
    print("\n" + "=" * 60)
    print("Running Test Scenarios")
    print("=" * 60)
    
    # Scenario 1: NARCOTICS at night in high-crime district
    print("\n[Test 1] NARCOTICS crime at night in District 11")
    predict_arrest_likelihood(
        district=11,
        hour=23,
        primary_type='NARCOTICS',
        domestic=0,
        encoders=encoders,
        model=model,
        df_cleaned=df_cleaned
    )
    
    # Scenario 2: THEFT during afternoon
    print("\n[Test 2] THEFT crime in afternoon in District 6")
    predict_arrest_likelihood(
        district=6,
        hour=14,
        primary_type='THEFT',
        domestic=0,
        encoders=encoders,
        model=model,
        df_cleaned=df_cleaned
    )
    
    # Scenario 3: Domestic BATTERY
    print("\n[Test 3] Domestic BATTERY in evening in District 8")
    predict_arrest_likelihood(
        district=8,
        hour=20,
        primary_type='BATTERY',
        domestic=1,
        encoders=encoders,
        model=model,
        df_cleaned=df_cleaned
    )
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print("\nEnter your own test case (or press 'q' to quit):")
    
    while True:
        try:
            print("\n--- New Test Case ---")
            district = input("District (e.g., 11): ").strip()
            if district.lower() == 'q':
                break
            
            hour = input("Hour (0-23): ").strip()
            if hour.lower() == 'q':
                break
            
            crime_type = input("Crime Type (e.g., THEFT, BATTERY, NARCOTICS): ").strip()
            if crime_type.lower() == 'q':
                break
            
            domestic = input("Domestic? (0=No, 1=Yes): ").strip()
            if domestic.lower() == 'q':
                break
            
            predict_arrest_likelihood(
                district=int(district),
                hour=int(hour),
                primary_type=crime_type.upper(),
                domestic=int(domestic),
                encoders=encoders,
                model=model,
                df_cleaned=df_cleaned
            )
            
        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for testing!")


if __name__ == "__main__":
    main()
