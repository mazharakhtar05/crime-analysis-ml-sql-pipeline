"""
SQL Integration Script for Chicago Crime Analysis

This script runs the SQL pipeline to migrate data, run analytics,
and store model predictions.
Usage: python sql_integration.py
"""

import os
import sys
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import preprocess_pipeline
from src.model import load_model
from src.sql_pipeline import complete_sql_pipeline


def main():
    """Run the complete SQL pipeline."""
    print("=" * 60)
    print("Chicago Crime Analysis - SQL Integration Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading and preprocessing data...")
    df_cleaned, df_final, encoders = preprocess_pipeline(nrows=50000)
    
    # Step 2: Load trained model
    print("\n[Step 2] Loading trained model...")
    if not os.path.exists('models/xgboost_model.pkl'):
        print("Error: Model not found. Please run train_model.py first.")
        sys.exit(1)
    
    model = load_model('models/xgboost_model.pkl')
    
    # Step 3: Run SQL pipeline
    print("\n[Step 3] Running SQL pipeline...")
    results = complete_sql_pipeline(
        df_cleaned=df_cleaned,
        df_final=df_final,
        model=model
    )
    
    # Step 4: Summary
    print("\n" + "=" * 60)
    print("SQL Pipeline Complete!")
    print("=" * 60)
    
    if results['success']:
        print("\n[SUCCESS] Data migrated to database successfully")
        print("[SUCCESS] Analytics queries executed")
        print("[SUCCESS] Predictions stored in database")
        
        # Display analytics summary
        analytics = results['analytics']
        print("\n--- Analytics Summary ---")
        if 'top_districts' in analytics and not analytics['top_districts'].empty:
            print(f"\nTop District by Volume:")
            print(f"  District {analytics['top_districts'].iloc[0]['District']}: "
                  f"{analytics['top_districts'].iloc[0]['Incident_Count']} incidents")
        
        if 'district_volume_ranking' in analytics and not analytics['district_volume_ranking'].empty:
            print(f"\nRanked Districts (Window Analytics):")
            top_ranked = analytics['district_volume_ranking'].iloc[0]
            print(f"  Rank 1 District: {top_ranked['District']} "
                  f"({top_ranked['Incident_Count']} incidents, Arrest Rate: {top_ranked['Arrest_Rate']}%)")
        
        print(f"\nPredictions saved: {results['predictions_saved']}")
    else:
        print("\n[ERROR] Pipeline encountered errors. Check logs above.")


if __name__ == "__main__":
    main()
