"""
Machine Learning Model Module for Chicago Crime Analysis

This module handles model training, optimization, evaluation,
and prediction for arrest likelihood classification.
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict, Optional


def split_data(df_final: pd.DataFrame, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and testing sets.
    
    Args:
        df_final: ML-ready DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df_final.drop('Arrest', axis=1)
    y = df_final['Arrest']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Testing set size: {X_test.shape[0]} rows")
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, 
                random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to handle class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        
    Returns:
        Tuple of (X_train_balanced, y_train_balanced)
    """
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original training shape: {y_train.value_counts().to_dict()}")
    print(f"Balanced training shape: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    
    return X_train_balanced, y_train_balanced


def train_baseline_models(X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Train baseline models (Logistic Regression and Random Forest).
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        
    Returns:
        Dictionary with trained models and predictions
    """
    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    log_preds = log_model.predict(X_test)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    print("Baseline models trained successfully.")
    
    return {
        'log_model': log_model,
        'rf_model': rf_model,
        'log_preds': log_preds,
        'rf_preds': rf_preds
    }


def evaluate_model(y_test: pd.Series, y_pred: pd.Series, 
                   model_name: str = "Model") -> None:
    """
    Evaluate model performance and print metrics.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display
    """
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


def optimize_models(X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Train optimized models with SMOTE and GridSearchCV.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        
    Returns:
        Dictionary with best models and their predictions
    """
    # Apply SMOTE
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    # Define parameter grids
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    xgb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 6]
    }
    
    # Tune Random Forest
    print("\nTuning Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        rf_params, cv=3, scoring='f1', n_jobs=-1
    )
    rf_grid.fit(X_train_balanced, y_train_balanced)
    
    # Tune XGBoost
    print("\nTuning XGBoost...")
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=42), 
        xgb_params, cv=3, scoring='f1', n_jobs=-1
    )
    xgb_grid.fit(X_train_balanced, y_train_balanced)
    
    print("GridSearch complete.")
    print(f"Best RF Params: {rf_grid.best_params_}")
    print(f"Best XGB Params: {xgb_grid.best_params_}")
    
    # Get best models
    best_rf = rf_grid.best_estimator_
    best_xgb = xgb_grid.best_estimator_
    
    # Generate predictions
    rf_final_preds = best_rf.predict(X_test)
    xgb_final_preds = best_xgb.predict(X_test)
    
    return {
        'best_rf': best_rf,
        'best_xgb': best_xgb,
        'rf_preds': rf_final_preds,
        'xgb_preds': xgb_final_preds,
        'rf_best_params': rf_grid.best_params_,
        'xgb_best_params': xgb_grid.best_params_
    }


def plot_feature_importance(model, X: pd.DataFrame, 
                           title: str = 'Feature Importances') -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        X: Feature DataFrame
        title: Plot title
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()


def save_model(model, filepath: str) -> None:
    """
    Save trained model to disk using pickle.
    
    Args:
        model: Trained model object
        filepath: Path to save the model
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {filepath}")
    return model


def predict_arrest_likelihood(district: int, hour: int, primary_type: str,
                             domestic: int = 0, encoders: Dict = None,
                             model=None, df_cleaned: pd.DataFrame = None) -> Dict:
    """
    Predict arrest likelihood for a given crime incident.
    
    Args:
        district: Police district number
        hour: Hour of day (0-23)
        primary_type: Type of crime (e.g., 'THEFT', 'BATTERY')
        domestic: Whether crime is domestic (0 or 1)
        encoders: Dictionary with fitted label encoders
        model: Trained model for prediction
        df_cleaned: Cleaned dataframe for reference values
        
    Returns:
        Dictionary with prediction results
    """
    if encoders is None or model is None or df_cleaned is None:
        raise ValueError("encoders, model, and df_cleaned must be provided")
    
    # Map district to area_risk score
    district_counts = encoders['district_counts']
    risk_score = district_counts.get(district, district_counts.mean())
    
    # Encode the primary type
    le_type = encoders['le_type']
    try:
        type_encoded = le_type.transform([primary_type])[0]
    except ValueError:
        # Default to most common type if unknown
        type_encoded = le_type.transform(['THEFT'])[0]
    
    # Create input data with median/mode values for missing features
    input_data = pd.DataFrame([{
        'Domestic': domestic,
        'District': district,
        'Latitude': df_cleaned['Latitude'].median(),
        'Longitude': df_cleaned['Longitude'].median(),
        'Hour': hour,
        'Month': 9,  # Default to September
        'is_weekend': 0,
        'Primary Type Target': type_encoded,
        'Location Description Target': 0,
        'time_of_day_encoded': 0,
        'area_risk': risk_score
    }])
    
    # Ensure column order matches training
    # Note: Adjust columns based on your actual model training
    input_data = input_data[model.feature_names_in_]
    
    # Generate prediction
    prob = model.predict_proba(input_data)[0][1]
    prediction = "YES" if prob > 0.5 else "NO"
    
    result = {
        'crime_type': primary_type,
        'district': district,
        'hour': hour,
        'arrest_predicted': prediction,
        'probability': prob * 100
    }
    
    print(f"--- Prediction Result ---")
    print(f"Crime Type: {primary_type}")
    print(f"District: {district} | Hour: {hour}")
    print(f"Arrest Predicted: {prediction}")
    print(f"Probability of Arrest: {prob*100:.2f}%")
    print("-------------------------")
    
    return result


def full_training_pipeline(df_final: pd.DataFrame, 
                          model_save_path: str = 'models/xgboost_model.pkl') -> Dict:
    """
    Complete training pipeline with optimization and model saving.
    
    Args:
        df_final: ML-ready DataFrame
        model_save_path: Path to save the best model
        
    Returns:
        Dictionary with trained models and metadata
    """
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_final)
    
    # Train baseline models
    print("\n=== Training Baseline Models ===")
    baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)
    
    # Evaluate baseline models
    evaluate_model(y_test, baseline_results['log_preds'], "Logistic Regression")
    evaluate_model(y_test, baseline_results['rf_preds'], "Random Forest")
    
    # Optimize models with SMOTE and GridSearch
    print("\n=== Training Optimized Models ===")
    optimized_results = optimize_models(X_train, y_train, X_test, y_test)
    
    # Evaluate optimized models
    evaluate_model(y_test, optimized_results['rf_preds'], "Optimized Random Forest")
    evaluate_model(y_test, optimized_results['xgb_preds'], "Optimized XGBoost")
    
    # Plot feature importance for XGBoost
    X = df_final.drop('Arrest', axis=1)
    plot_feature_importance(optimized_results['best_xgb'], X, 
                           'Feature Importances (XGBoost Model)')
    
    # Save best model (XGBoost)
    save_model(optimized_results['best_xgb'], model_save_path)
    
    return {
        'baseline': baseline_results,
        'optimized': optimized_results,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
