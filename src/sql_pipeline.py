"""
SQL Pipeline Module for Chicago Crime Analysis

This module handles database connections, data migration,
SQL analytics, and model inference pipeline.
"""

import os
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import Optional, Dict
from dotenv import load_dotenv


def get_db_credentials() -> Dict[str, Optional[str]]:
    """
    Get database credentials from environment variables.
    
    Returns:
        Dictionary with database credentials or None values if unset.
    """
    # Load .env file if it exists
    load_dotenv()
    
    credentials = {
        'DB_HOST': os.getenv('DB_HOST'),
        'DB_USER': os.getenv('DB_USER'),
        'DB_PASS': os.getenv('DB_PASS'),
        'DB_NAME': os.getenv('DB_NAME'),
        'DB_PORT': os.getenv('DB_PORT', '3306'),
        'DATABASE_URL': os.getenv('DATABASE_URL')
    }
    
    return credentials


def create_db_engine(credentials: Optional[Dict[str, Optional[str]]] = None, 
                     sqlite_path: str = 'data/crime_analysis.db'):
    """
    Create SQLAlchemy engine for database connection with automatic SQLite fallback.
    
    Args:
        credentials: Database credentials dict (uses env vars if None)
        sqlite_path: File path for local SQLite fallback database
        
    Returns:
        SQLAlchemy engine object
    """
    if credentials is None:
        credentials = get_db_credentials()
    
    # 1. Check if direct DATABASE_URL is set
    db_url = credentials.get('DATABASE_URL')
    if db_url:
        print("Connecting to database using configured DATABASE_URL.")
        return create_engine(db_url)
    
    host = credentials.get('DB_HOST')
    user = credentials.get('DB_USER')
    password = credentials.get('DB_PASS')
    name = credentials.get('DB_NAME')
    port = credentials.get('DB_PORT', '3306')
    
    # 2. If MySQL credentials are provided, connect to MySQL
    if host and user and password and name:
        connection_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}"
        print(f"Connecting to MySQL database on {host} (database: {name})...")
        engine = create_engine(connection_url)
        print("MySQL database engine initialized successfully.")
        return engine
    
    # 3. Fallback to local SQLite database
    os.makedirs(os.path.dirname(sqlite_path) or '.', exist_ok=True)
    sqlite_url = f"sqlite:///{sqlite_path}"
    print(f"No MySQL credentials configured. Using local SQLite database at: {sqlite_url}")
    engine = create_engine(sqlite_url)
    print("SQLite database engine initialized successfully.")
    return engine


def migrate_to_sql(df: pd.DataFrame, table_name: str, 
                   engine=None, chunksize: int = 1000) -> bool:
    """
    Migrate DataFrame to MySQL database with optimized batch insertion.
    
    Args:
        df: DataFrame to migrate
        table_name: Target table name
        engine: SQLAlchemy engine (creates new one if None)
        chunksize: Number of rows per batch
        
    Returns:
        True if successful, False otherwise
    """
    if engine is None:
        engine = create_db_engine()
    
    start_time = time.time()
    
    try:
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',
            index=False,
            chunksize=chunksize,
            method='multi'
        )
        
        duration = time.time() - start_time
        print(f"Successfully migrated {len(df)} rows to '{table_name}'")
        print(f"Time taken: {duration:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"Migration failed: {e}")
        return False


def run_sql_query(query: str, engine=None) -> pd.DataFrame:
    """
    Execute SQL query and return results as DataFrame.
    
    Args:
        query: SQL query string
        engine: SQLAlchemy engine (creates new one if None)
        
    Returns:
        DataFrame with query results
    """
    if engine is None:
        engine = create_db_engine()
    
    try:
        df_result = pd.read_sql(query, engine)
        return df_result
    except Exception as e:
        print(f"Query failed: {e}")
        return pd.DataFrame()


def get_analytics_queries() -> Dict[str, str]:
    """
    Returns a dictionary of standardized analytics SQL queries.
    All queries are compatible with MySQL 8+ and SQLite 3.25+.
    
    Returns:
        Dictionary mapping query names to SQL statements
    """
    return {
        'top_districts': """
            SELECT District, COUNT(*) as Incident_Count
            FROM chicago_crimes_optimized
            GROUP BY District
            ORDER BY Incident_Count DESC
            LIMIT 5;
        """,
        'district_volume_ranking': """
            SELECT
                District,
                COUNT(*) as Incident_Count,
                DENSE_RANK() OVER (ORDER BY COUNT(*) DESC) as Volume_Rank,
                ROUND(AVG(Arrest) * 100, 2) as Arrest_Rate
            FROM chicago_crimes_optimized
            GROUP BY District
            ORDER BY Volume_Rank ASC
            LIMIT 10;
        """,
        'arrest_rates': """
            SELECT `Primary Type`,
                   COUNT(*) as Incident_Count,
                   ROUND(AVG(Arrest) * 100, 2) as Arrest_Percentage
            FROM chicago_crimes_optimized
            GROUP BY `Primary Type`
            HAVING COUNT(*) > 100
            ORDER BY Arrest_Percentage DESC
            LIMIT 10;
        """,
        'peak_hours': """
            SELECT
                Hour,
                COUNT(*) as Crime_Count,
                DENSE_RANK() OVER (ORDER BY COUNT(*) DESC) as Peak_Rank,
                CASE
                    WHEN Hour BETWEEN 0 AND 5 THEN 'Late Night'
                    WHEN Hour BETWEEN 6 AND 11 THEN 'Morning'
                    WHEN Hour BETWEEN 12 AND 17 THEN 'Afternoon'
                    ELSE 'Evening'
                END as Time_Period
            FROM chicago_crimes_optimized
            GROUP BY Hour
            ORDER BY Crime_Count DESC
            LIMIT 5;
        """,
        'gap_areas': """
            SELECT
                District,
                COUNT(*) as Incident_Count,
                ROUND(AVG(Arrest) * 100, 2) as Arrest_Rate,
                ROUND(AVG(AVG(Arrest) * 100) OVER (), 2) as Citywide_Avg_Arrest_Rate
            FROM chicago_crimes_optimized
            GROUP BY District
            HAVING COUNT(*) > 1000
            ORDER BY Arrest_Rate ASC
            LIMIT 5;
        """
    }


def run_analytics(engine=None, queries: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive SQL analytics queries.
    
    Args:
        engine: SQLAlchemy engine (creates default if None)
        queries: Optional custom dictionary of queries
        
    Returns:
        Dictionary with analytics results DataFrames
    """
    if engine is None:
        engine = create_db_engine()
    
    query_dict = queries or get_analytics_queries()
    results = {}
    
    titles = {
        'top_districts': "Top 5 High-Volume Districts",
        'district_volume_ranking': "District Crime Volume & Window Rankings",
        'arrest_rates': "Top 10 Arrest Rates by Crime Type",
        'peak_hours': "Peak Crime Hours (Ranked)",
        'gap_areas': "Gap Areas (High Volume, Low Arrest Rates vs Citywide Avg)"
    }
    
    for key, query in query_dict.items():
        title = titles.get(key, key.replace('_', ' ').title())
        print(f"\n--- {title} ---")
        df_result = run_sql_query(query, engine)
        results[key] = df_result
        if not df_result.empty:
            print(df_result)
        else:
            print("No records returned.")
    
    return results


def generate_and_store_predictions(model, df_final: pd.DataFrame,
                                   table_name: str = 'predictions',
                                   engine=None) -> bool:
    """
    Generate predictions using trained model and store in database.
    
    Args:
        model: Trained model for prediction
        df_final: DataFrame with features for prediction
        table_name: Target table name for predictions
        engine: SQLAlchemy engine
        
    Returns:
        True if successful, False otherwise
    """
    if engine is None:
        engine = create_db_engine()
    
    try:
        # Prepare features (exclude target variable)
        X_infer = df_final.drop('Arrest', axis=1)
        
        # Generate predictions
        print("Generating model predictions...")
        preds = model.predict(X_infer)
        probs = model.predict_proba(X_infer)[:, 1]
        
        # Create prediction DataFrame
        df_predictions = pd.DataFrame({
            'District': df_final['District'],
            'Hour': df_final['Hour'],
            'prediction': preds,
            'probability': np.round(probs, 4)
        })
        
        # Save predictions to MySQL
        print(f"Writing predictions to new MySQL table: '{table_name}'...")
        df_predictions.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',
            index=False,
            chunksize=1000,
            method='multi'
        )
        
        print("Success! Predictions are now stored in the database.")
        print(f"Sample predictions:")
        print(df_predictions.head())
        
        return True
        
    except Exception as e:
        print(f"Failed to save predictions: {e}")
        return False


def complete_sql_pipeline(df_cleaned: pd.DataFrame, df_final: pd.DataFrame,
                         model, credentials: Optional[Dict[str, str]] = None) -> Dict:
    """
    Complete SQL pipeline: migrate data, run analytics, generate predictions.
    
    Args:
        df_cleaned: Cleaned DataFrame with original columns
        df_final: ML-ready DataFrame
        model: Trained model for predictions
        credentials: Database credentials (optional)
        
    Returns:
        Dictionary with pipeline results
    """
    # Create engine
    engine = create_db_engine(credentials)
    
    # Step 1: Migrate cleaned data to SQL
    print("\n=== Step 1: Data Migration ===")
    migrate_success = migrate_to_sql(df_cleaned, 'chicago_crimes_optimized', engine)
    
    if not migrate_success:
        print("Data migration failed. Aborting pipeline.")
        return {'success': False}
    
    # Step 2: Run analytics
    print("\n=== Step 2: SQL Analytics ===")
    analytics_results = run_analytics(engine)
    
    # Step 3: Generate and store predictions
    print("\n=== Step 3: Generate Predictions ===")
    prediction_success = generate_and_store_predictions(model, df_final, engine=engine)
    
    return {
        'success': migrate_success and prediction_success,
        'analytics': analytics_results,
        'predictions_saved': prediction_success
    }
