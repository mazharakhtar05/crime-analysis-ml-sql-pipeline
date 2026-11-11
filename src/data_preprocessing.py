"""
Data Preprocessing Module for Chicago Crime Analysis

This module handles data loading, cleaning, feature engineering,
and preparation of ML-ready datasets from the Chicago Crimes dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import kagglehub
from typing import Tuple, Dict, Optional


def load_data(nrows: Optional[int] = 50000, file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load Chicago Crimes dataset from a local path or Kaggle.
    
    Args:
        nrows: Number of rows to load (default: 50000)
        file_path: Optional path to a local CSV file
        
    Returns:
        DataFrame with raw crime data
    """
    # 1. Check if an explicit file path is provided and exists
    if file_path and os.path.exists(file_path):
        print(f"Loading dataset from specified file: {file_path}")
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"Loaded {len(df)} rows")
        return df
    
    # 2. Check local fallback directories
    local_candidates = [
        os.path.join('data', 'raw', 'Crimes_-_2001_to_Present.csv'),
        os.path.join('data', 'sample', 'crime_sample.csv'),
        os.path.join('data', 'Crimes_-_2001_to_Present.csv'),
    ]
    for candidate in local_candidates:
        if os.path.exists(candidate):
            print(f"Loading dataset from local cache: {candidate}")
            df = pd.read_csv(candidate, nrows=nrows)
            print(f"Loaded {len(df)} rows from local cache")
            return df
    
    # 3. Fallback to Kaggle download
    print("No local dataset found. Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("utkarshx27/crimes-2001-to-present")
    print(f"Path to dataset files: {path}")
    
    csv_file = os.path.join(path, 'Crimes_-_2001_to_Present.csv')
    df = pd.read_csv(csv_file, nrows=nrows)
    print(f"Loaded {len(df)} rows")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and converting data types.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Convert Date to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p')
    
    # Extract temporal features
    df['Hour'] = df['Date'].dt.hour
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    print("Date conversion and feature extraction complete.")
    
    # Show missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Drop rows with missing critical values
    df = df.dropna(subset=['Latitude', 'Longitude', 'District'])
    print(f"\nRemaining rows after dropping nulls: {len(df)}")
    
    # Feature selection - keep only relevant columns
    columns_to_keep = [
        'Date', 'Primary Type', 'Location Description',
        'Arrest', 'Domestic', 'District',
        'Latitude', 'Longitude', 'Hour', 'Month'
    ]
    
    df_cleaned = df[columns_to_keep].copy()
    print(f"Dataset Shape after feature selection: {df_cleaned.shape}")
    
    return df_cleaned


def engineer_features(df_cleaned: pd.DataFrame, 
                      encoders: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Create advanced features for ML model.
    Supports both training mode (encoders=None, fits encoders) and
    inference mode (encoders provided, transforms using existing parameters).
    
    Args:
        df_cleaned: Cleaned DataFrame
        encoders: Optional pre-fitted encoders dictionary
        
    Returns:
        Tuple of (DataFrame with engineered features, encoders dict)
    """
    # Create 'is_weekend' feature
    df_cleaned['is_weekend'] = df_cleaned['Date'].dt.dayofweek.apply(
        lambda x: 1 if x >= 5 else 0
    )
    
    # Define time of day categorization
    def get_time_of_day(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 24:
            return 'Evening'
        else:
            return 'Night'
    
    df_cleaned['time_of_day'] = df_cleaned['Hour'].apply(get_time_of_day)
    print("Temporal features created.")
    
    is_fit_mode = encoders is None
    
    if is_fit_mode:
        le_type = LabelEncoder()
        le_loc = LabelEncoder()
        le_tod = LabelEncoder()
        
        df_cleaned['Primary Type Target'] = le_type.fit_transform(df_cleaned['Primary Type'])
        df_cleaned['Location Description Target'] = le_loc.fit_transform(
            df_cleaned['Location Description'].astype(str)
        )
        df_cleaned['time_of_day_encoded'] = le_tod.fit_transform(df_cleaned['time_of_day'])
        
        district_counts = df_cleaned['District'].value_counts(normalize=True).to_dict()
        df_cleaned['area_risk'] = df_cleaned['District'].map(district_counts).fillna(0.0)
        
        median_lat = float(df_cleaned['Latitude'].median()) if 'Latitude' in df_cleaned and not df_cleaned['Latitude'].empty else 41.8781
        median_lon = float(df_cleaned['Longitude'].median()) if 'Longitude' in df_cleaned and not df_cleaned['Longitude'].empty else -87.6298
        
        encoders = {
            'le_type': le_type,
            'le_loc': le_loc,
            'le_tod': le_tod,
            'district_counts': district_counts,
            'median_latitude': median_lat,
            'median_longitude': median_lon,
            'feature_names': [
                'Domestic', 'District', 'Latitude', 'Longitude',
                'Hour', 'Month', 'is_weekend', 'Primary Type Target',
                'Location Description Target', 'time_of_day_encoded', 'area_risk'
            ]
        }
    else:
        # Transform using pre-fitted encoders to prevent test leakage
        le_type = encoders['le_type']
        le_loc = encoders['le_loc']
        le_tod = encoders['le_tod']
        district_counts = encoders['district_counts']
        
        type_classes = set(le_type.classes_)
        df_cleaned['Primary Type Target'] = df_cleaned['Primary Type'].apply(
            lambda x: le_type.transform([x])[0] if x in type_classes else (
                le_type.transform(['THEFT'])[0] if 'THEFT' in type_classes else 0
            )
        )
        
        loc_classes = set(le_loc.classes_)
        df_cleaned['Location Description Target'] = df_cleaned['Location Description'].astype(str).apply(
            lambda x: le_loc.transform([x])[0] if x in loc_classes else 0
        )
        
        tod_classes = set(le_tod.classes_)
        df_cleaned['time_of_day_encoded'] = df_cleaned['time_of_day'].apply(
            lambda x: le_tod.transform([x])[0] if x in tod_classes else 0
        )
        
        if isinstance(district_counts, dict):
            mean_risk = sum(district_counts.values()) / max(len(district_counts), 1)
            df_cleaned['area_risk'] = df_cleaned['District'].map(district_counts).fillna(mean_risk)
        else:
            df_cleaned['area_risk'] = df_cleaned['District'].map(district_counts).fillna(district_counts.mean())
    
    print("Feature engineering and encoding complete.")
    return df_cleaned, encoders


def prepare_ml_dataset(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare final ML-ready dataset with numeric features only.
    
    Args:
        df_cleaned: DataFrame with engineered features
        
    Returns:
        DataFrame ready for ML modeling
    """
    # Convert booleans to 0/1
    if 'Arrest' in df_cleaned.columns:
        df_cleaned['Arrest'] = df_cleaned['Arrest'].astype(int)
    if 'Domestic' in df_cleaned.columns:
        df_cleaned['Domestic'] = df_cleaned['Domestic'].astype(int)
    
    # Select final ML features
    ml_features = [
        'Domestic', 'District', 'Latitude', 'Longitude',
        'Hour', 'Month', 'is_weekend', 'Primary Type Target',
        'Location Description Target', 'time_of_day_encoded', 'area_risk'
    ]
    if 'Arrest' in df_cleaned.columns:
        ml_features = ['Arrest'] + ml_features
    
    df_final = df_cleaned[ml_features].copy()
    print(f"Final ML Dataset Shape: {df_final.shape}")
    
    return df_final


def preprocess_pipeline(nrows: int = 50000, 
                        file_path: Optional[str] = None,
                        encoders: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete preprocessing pipeline from raw data to ML-ready dataset.
    
    Args:
        nrows: Number of rows to load
        file_path: Optional path to a local CSV file
        encoders: Optional pre-fitted encoders dictionary
        
    Returns:
        Tuple of (df_cleaned, df_final, encoders)
    """
    # Load data
    df = load_data(nrows=nrows, file_path=file_path)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Engineer features
    df_cleaned, encoders = engineer_features(df_cleaned, encoders=encoders)
    
    # Prepare ML dataset
    df_final = prepare_ml_dataset(df_cleaned)
    
    return df_cleaned, df_final, encoders
