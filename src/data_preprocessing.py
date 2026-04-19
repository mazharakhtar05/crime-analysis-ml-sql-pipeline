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


def load_data(nrows: Optional[int] = 50000) -> pd.DataFrame:
    """
    Load Chicago Crimes dataset from Kaggle.
    
    Args:
        nrows: Number of rows to load (default: 50000)
        
    Returns:
        DataFrame with raw crime data
    """
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("utkarshx27/crimes-2001-to-present")
    print(f"Path to dataset files: {path}")
    
    file_path = os.path.join(path, 'Crimes_-_2001_to_Present.csv')
    df = pd.read_csv(file_path, nrows=nrows)
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


def engineer_features(df_cleaned: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Create advanced features for ML model.
    
    Args:
        df_cleaned: Cleaned DataFrame
        
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
    
    # Categorical Encoding
    le_type = LabelEncoder()
    le_loc = LabelEncoder()
    le_tod = LabelEncoder()
    
    df_cleaned['Primary Type Target'] = le_type.fit_transform(df_cleaned['Primary Type'])
    df_cleaned['Location Description Target'] = le_loc.fit_transform(
        df_cleaned['Location Description'].astype(str)
    )
    df_cleaned['time_of_day_encoded'] = le_tod.fit_transform(df_cleaned['time_of_day'])
    
    print("Encoding complete.")
    
    # Calculate area risk (crime frequency per district)
    district_counts = df_cleaned['District'].value_counts(normalize=True)
    df_cleaned['area_risk'] = df_cleaned['District'].map(district_counts)
    
    print("Area risk mapping complete.")
    
    # Store encoders for later use
    encoders = {
        'le_type': le_type,
        'le_loc': le_loc,
        'le_tod': le_tod,
        'district_counts': district_counts
    }
    
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
    df_cleaned['Arrest'] = df_cleaned['Arrest'].astype(int)
    df_cleaned['Domestic'] = df_cleaned['Domestic'].astype(int)
    
    # Select final ML features
    ml_features = [
        'Arrest', 'Domestic', 'District', 'Latitude', 'Longitude',
        'Hour', 'Month', 'is_weekend', 'Primary Type Target',
        'Location Description Target', 'time_of_day_encoded', 'area_risk'
    ]
    
    df_final = df_cleaned[ml_features].copy()
    print(f"Final ML Dataset Shape: {df_final.shape}")
    
    return df_final


def preprocess_pipeline(nrows: int = 50000) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete preprocessing pipeline from raw data to ML-ready dataset.
    
    Args:
        nrows: Number of rows to load
        
    Returns:
        Tuple of (df_cleaned, df_final, encoders)
    """
    # Load data
    df = load_data(nrows=nrows)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Engineer features
    df_cleaned, encoders = engineer_features(df_cleaned)
    
    # Prepare ML dataset
    df_final = prepare_ml_dataset(df_cleaned)
    
    return df_cleaned, df_final, encoders
