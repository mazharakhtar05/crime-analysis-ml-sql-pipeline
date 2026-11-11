"""
Unit tests for Machine Learning Models and Prediction Inference.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Bootstrap missing third-party dependencies if not installed
for mod in [
    'pandas', 'numpy', 'matplotlib', 'matplotlib.pyplot',
    'sklearn', 'sklearn.model_selection', 'sklearn.linear_model',
    'sklearn.ensemble', 'sklearn.metrics', 'sklearn.preprocessing',
    'xgboost', 'imblearn', 'imblearn.over_sampling'
]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.model as model_mod


class TestModelModule(unittest.TestCase):
    """Test suite for evaluation metrics, model persistence, and prediction inference."""

    def test_evaluate_model_metrics_calculation(self):
        """Verify evaluate_model returns complete metrics dictionary with ROC-AUC and Confusion Matrix."""
        y_test = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 0]
        y_prob = [0.1, 0.9, 0.2, 0.4]

        with patch('src.model.accuracy_score', return_value=0.75), \
             patch('src.model.balanced_accuracy_score', return_value=0.75), \
             patch('src.model.precision_score', return_value=1.0), \
             patch('src.model.recall_score', return_value=0.5), \
             patch('src.model.f1_score', return_value=0.67), \
             patch('src.model.roc_auc_score', return_value=0.85), \
             patch('src.model.confusion_matrix', return_value=MagicMock(tolist=lambda: [[2, 0], [1, 1]])), \
             patch('src.model.classification_report', return_value="mock report"):

            metrics = model_mod.evaluate_model(y_test, y_pred, y_prob=y_prob, model_name="TestModel")
            
            self.assertEqual(metrics['model_name'], "TestModel")
            self.assertEqual(metrics['accuracy'], 0.75)
            self.assertEqual(metrics['balanced_accuracy'], 0.75)
            self.assertEqual(metrics['precision'], 1.0)
            self.assertEqual(metrics['recall'], 0.5)
            self.assertEqual(metrics['f1_score'], 0.67)
            self.assertEqual(metrics['roc_auc'], 0.85)
            self.assertEqual(metrics['confusion_matrix'], [[2, 0], [1, 1]])

    def test_save_and_load_model_serialization(self):
        """Verify model roundtrip serialization via pickle."""
        test_model_obj = {'name': 'MockXGBoostModel', 'weights': [0.1, 0.5, 0.9]}
        temp_path = 'tests/temp_model.pkl'

        try:
            model_mod.save_model(test_model_obj, temp_path)
            self.assertTrue(os.path.exists(temp_path))

            loaded_model = model_mod.load_model(temp_path)
            self.assertEqual(loaded_model['name'], 'MockXGBoostModel')
            self.assertEqual(loaded_model['weights'], [0.1, 0.5, 0.9])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_predict_arrest_likelihood_with_metadata_fallback(self):
        """Verify inference succeeds using encoders metadata without requiring df_cleaned."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = [
            'Domestic', 'District', 'Latitude', 'Longitude',
            'Hour', 'Month', 'is_weekend', 'Primary Type Target',
            'Location Description Target', 'time_of_day_encoded', 'area_risk'
        ]
        # Predict 80% probability of arrest
        mock_model.predict_proba.return_value = [[0.2, 0.8]]

        mock_le = MagicMock()
        mock_le.classes_ = ['THEFT', 'BATTERY', 'NARCOTICS']
        mock_le.transform.return_value = [2]

        encoders = {
            'le_type': mock_le,
            'district_counts': {11: 0.15, 8: 0.10},
            'median_latitude': 41.8781,
            'median_longitude': -87.6298
        }

        # Mock pd.DataFrame inside predict_arrest_likelihood
        mock_df = MagicMock()
        mock_df.__getitem__.return_value = mock_df
        with patch('src.model.pd.DataFrame', return_value=mock_df):
            result = model_mod.predict_arrest_likelihood(
                district=11,
                hour=23,
                primary_type='NARCOTICS',
                domestic=0,
                encoders=encoders,
                model=mock_model,
                df_cleaned=None  # Verifies decoupled inference
            )

            self.assertEqual(result['crime_type'], 'NARCOTICS')
            self.assertEqual(result['district'], 11)
            self.assertEqual(result['hour'], 23)
            self.assertEqual(result['arrest_predicted'], 'YES')
            self.assertAlmostEqual(result['probability'], 80.0)


if __name__ == '__main__':
    unittest.main()
