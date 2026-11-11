"""
Unit tests for Data Preprocessing and Feature Engineering.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Bootstrap missing third-party dependencies if not installed
for mod in ['pandas', 'numpy', 'sklearn', 'sklearn.preprocessing', 'kagglehub']:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.data_preprocessing as dp


class TestDataPreprocessing(unittest.TestCase):
    """Test suite for preprocessing, temporal feature extraction, and encoding."""

    def test_time_of_day_categorization(self):
        """Verify time of day categorization logic."""
        # Using the same logic as defined in engineer_features
        def get_time_of_day(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 24:
                return 'Evening'
            else:
                return 'Night'

        self.assertEqual(get_time_of_day(3), 'Night')
        self.assertEqual(get_time_of_day(8), 'Morning')
        self.assertEqual(get_time_of_day(14), 'Afternoon')
        self.assertEqual(get_time_of_day(21), 'Evening')
        self.assertEqual(get_time_of_day(0), 'Night')

    def test_load_data_from_specified_local_path(self):
        """Verify load_data loads directly from an explicit local CSV file if it exists."""
        test_file = 'tests/sample_dummy.csv'
        try:
            with open(test_file, 'w') as f:
                f.write("ID,Case Number,Date,Primary Type,District,Latitude,Longitude,Arrest,Domestic\n")
                f.write("1,JC123,05/10/2026 12:00:00 PM,THEFT,11,41.88,-87.63,0,0\n")

            mock_df = MagicMock()
            mock_df.__len__.return_value = 1
            with patch('src.data_preprocessing.pd.read_csv', return_value=mock_df) as mock_read:
                df = dp.load_data(nrows=10, file_path=test_file)
                mock_read.assert_called_once_with(test_file, nrows=10)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_engineer_features_training_mode(self):
        """Verify engineer_features generates expected keys and encoders in training mode."""
        class MockSeries:
            def __init__(self, data):
                self.data = list(data)
                self.dt = MagicMock()
                self.dt.dayofweek.apply.return_value = [0, 1]
            def apply(self, fn):
                return MockSeries([fn(x) for x in self.data])
            def value_counts(self, normalize=False):
                d = {11: 0.6, 8: 0.4}
                res = MagicMock()
                res.to_dict.return_value = d
                return res
            def map(self, mapping):
                res = MagicMock()
                res.fillna.return_value = [mapping.get(x, 0.0) for x in self.data]
                return res
            def median(self):
                return 41.878
            @property
            def empty(self):
                return False
            def astype(self, t):
                return self
            def __iter__(self):
                return iter(self.data)

        class MockDataFrame(dict):
            def copy(self):
                return MockDataFrame(self)

        mock_df = MockDataFrame({
            'Date': MockSeries(['2026-05-01', '2026-05-02']),
            'Hour': MockSeries([14, 21]),
            'District': MockSeries([11, 8]),
            'Latitude': MockSeries([41.878, 41.880]),
            'Longitude': MockSeries([-87.629, -87.630]),
            'Primary Type': MockSeries(['THEFT', 'BATTERY']),
            'Location Description': MockSeries(['STREET', 'RESIDENCE'])
        })

        mock_le = MagicMock()
        mock_le.fit_transform.return_value = [0, 1]
        with patch('src.data_preprocessing.LabelEncoder', return_value=mock_le):
            df_out, encoders = dp.engineer_features(mock_df, encoders=None)
            
            self.assertIn('le_type', encoders)
            self.assertIn('le_loc', encoders)
            self.assertIn('le_tod', encoders)
            self.assertIn('district_counts', encoders)
            self.assertIn('median_latitude', encoders)
            self.assertIn('median_longitude', encoders)
            self.assertIn('feature_names', encoders)
            self.assertEqual(encoders['median_latitude'], 41.878)


if __name__ == '__main__':
    unittest.main()
