# Final Project Report: Chicago Crime Analysis

## 1. Executive Summary

This project successfully built a predictive system achieving 86% accuracy in determining arrest probability. The pipeline integrates raw data ingestion, feature engineering, and cloud-based storage.

## 2. Data Engineering Findings

- **Volume**: 50,000 raw records processed; 48,989 records cleaned and migrated.
- **Migration Efficiency**: Implemented chunk-based insertion (chunksize=1000) reducing migration time to ~41 seconds for ~50k rows.
- **Database Structure**: Cleaned data and model predictions are stored in separate relational tables (`chicago_crimes_optimized` and `predictions`).

## 3. Machine Learning Performance

- **Algorithm**: XGBoost Classifier
- **Pre-processing**: SMOTE was critical for addressing the minority class (Arrests).
- **Metrics**:
  - Accuracy: 86%
  - Arrest Recall: 67%
  - Arrest F1-Score: 0.70
- **Feature Importance**: Location (Area Risk) and Temporal factors (Hour) were the top predictors.

## 4. Strategic Insights

- **Hotspots**: Districts 11, 8, and 6 show the highest crime density.
- **Gap Analysis**: Identified 'Gap Areas' with high volume but low arrest rates, providing actionable targets for law enforcement resource reallocation.

## 5. Conclusion

The project demonstrates a scalable workflow for public safety data, moving from descriptive analysis to proactive predictive modeling.
