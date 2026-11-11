"""
Unit tests for the SQL Pipeline and Analytical Queries.
"""

import os
import sys
import sqlite3
import unittest
from unittest.mock import patch, MagicMock

# Gracefully bootstrap missing third-party packages for test runners
for mod in ['pandas', 'numpy', 'sqlalchemy', 'dotenv', 'pymysql']:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sql_pipeline import get_db_credentials, get_analytics_queries


class TestSQLPipeline(unittest.TestCase):
    """Test suite for SQL pipeline components and analytical queries."""

    def setUp(self):
        """Set up in-memory SQLite database populated with test crime data."""
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        
        # Create table matching chicago_crimes_optimized schema
        self.cursor.execute("""
            CREATE TABLE chicago_crimes_optimized (
                District INTEGER,
                `Primary Type` TEXT,
                Arrest INTEGER,
                Domestic INTEGER,
                Hour INTEGER,
                Month INTEGER,
                Latitude REAL,
                Longitude REAL
            );
        """)
        
        # Insert test sample records across multiple districts and crime types
        test_data = [
            # District 11 (High volume, low arrest)
            (11, 'THEFT', 0, 0, 14, 5, 41.878, -87.629),
            (11, 'THEFT', 0, 0, 14, 5, 41.878, -87.629),
            (11, 'BATTERY', 1, 1, 20, 6, 41.879, -87.630),
            (11, 'NARCOTICS', 1, 0, 23, 7, 41.880, -87.631),
            # District 6 (Moderate volume)
            (6, 'THEFT', 0, 0, 10, 4, 41.850, -87.600),
            (6, 'BATTERY', 1, 0, 15, 4, 41.851, -87.601),
            # District 8 (Arrest heavy)
            (8, 'NARCOTICS', 1, 0, 22, 8, 41.860, -87.610),
            (8, 'NARCOTICS', 1, 0, 23, 8, 41.861, -87.611),
            (8, 'BATTERY', 1, 1, 19, 8, 41.862, -87.612),
        ]
        
        self.cursor.executemany("""
            INSERT INTO chicago_crimes_optimized 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, test_data)
        self.conn.commit()

    def tearDown(self):
        """Close test database connection."""
        self.conn.close()

    def test_get_db_credentials_no_hardcoded_passwords(self):
        """Verify that default credentials contain no hardcoded plaintext passwords."""
        with patch.dict(os.environ, {}, clear=True):
            creds = get_db_credentials()
            self.assertIsInstance(creds, dict)
            self.assertIn('DB_HOST', creds)
            self.assertIn('DB_USER', creds)
            self.assertIn('DB_PASS', creds)
            # Must not contain the exposed legacy credential
            self.assertNotEqual(creds.get('DB_PASS'), 'PDU5W6uflR')
            self.assertIsNone(creds.get('DB_PASS'))

    def test_analytics_queries_structure(self):
        """Verify that all standard analytics queries are defined and non-empty."""
        queries = get_analytics_queries()
        expected_keys = [
            'top_districts', 
            'district_volume_ranking', 
            'arrest_rates', 
            'peak_hours', 
            'gap_areas'
        ]
        for key in expected_keys:
            self.assertIn(key, queries)
            self.assertIsInstance(queries[key], str)
            self.assertTrue(len(queries[key].strip()) > 0)

    def test_top_districts_query_execution(self):
        """Verify top_districts query execution and volume ranking."""
        query = get_analytics_queries()['top_districts']
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        self.assertTrue(len(rows) > 0)
        # Top district should be District 11 (4 incidents)
        self.assertEqual(rows[0][0], 11)
        self.assertEqual(rows[0][1], 4)

    def test_window_ranking_query_execution(self):
        """Verify that the DENSE_RANK() window function query executes properly."""
        query = get_analytics_queries()['district_volume_ranking']
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        self.assertTrue(len(rows) >= 3)
        # Verify columns: District, Incident_Count, Volume_Rank, Arrest_Rate
        top_row = rows[0]
        self.assertEqual(top_row[0], 11)      # District 11
        self.assertEqual(top_row[1], 4)       # 4 incidents
        self.assertEqual(top_row[2], 1)       # Rank 1

    def test_peak_hours_query_execution(self):
        """Verify peak hours query with time period categorization and DENSE_RANK."""
        query = get_analytics_queries()['peak_hours']
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        self.assertTrue(len(rows) > 0)
        # Check that result has Hour, Crime_Count, Peak_Rank, Time_Period
        first_row = rows[0]
        self.assertEqual(len(first_row), 4)
        self.assertIn(first_row[3], ['Late Night', 'Morning', 'Afternoon', 'Evening'])


if __name__ == '__main__':
    unittest.main()
