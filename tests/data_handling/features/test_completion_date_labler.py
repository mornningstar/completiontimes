import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from src.data_handling.features.completion_date_labler import CompletionDateLabler

MOCK_NOW = pd.Timestamp('2025-02-01 00:00:00').tz_localize(None)

class TestCompletionDateLabler(unittest.TestCase):
    def setUp(self):
        # Instantiate the class, mocking the logger to suppress output
        with patch('src.data_handling.features.completion_date_labler.logging.getLogger', return_value=MagicMock()):
            self.labler = CompletionDateLabler()

        # Manually set the mock 'now' time for the instance
        self.labler.now = MOCK_NOW

        # Set stability constants for easy reference in tests
        self.MIN_COMMITS = CompletionDateLabler.STABILITY_MIN_COMMITS
        self.MIN_DAYS = CompletionDateLabler.STABILITY_MIN_DAYS
        self.IDLE_DAYS = CompletionDateLabler.STABILITY_IDLE_DAYS
        self.PERCENTAGE = CompletionDateLabler.STABILITY_PERCENTAGE_CHANGE

    def test_add_days_until_completion_calculation(self):
        data = {
            'path': ['f1', 'f1', 'f2'],
            'date': ['2024-12-01', '2024-12-10', '2025-01-20'],
            'completion_date': ['2024-12-30', '2024-12-15', '2025-01-20'],
        }
        df = pd.DataFrame(data)

        result_df = self.labler.add_days_until_completion(df)
        expected_days = [29, 5, 0]

        self.assertTrue("days_until_completion" in result_df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(result_df["days_until_completion"]))
        np.testing.assert_array_equal(result_df["days_until_completion"].values, expected_days)

    def _create_stable_df(self):
        path = "src/stable_file.py"

        data = {
            'path': [path] * 5,
            'date': [
                '2024-11-01',
                '2024-11-02',
                '2024-12-15',
                '2024-12-20',
                '2025-01-01',
            ],
            'line_change': [
                100,
                100,
                3,
                2,
                1,
            ],
            'size': [100] * 5,
            'commit_interval_days': [0, 1, 43, 5, 12],
        }

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        return df, pd.Timestamp('2025-01-01 00:00:00')

    def test_label_stable_line_change_success(self):
        df, expected_date = self._create_stable_df()

        result_df, _, _, _ = self.labler.label(df)

        self.assertEqual(result_df["completion_reason"].iloc[-1], "stable_line_change")
        self.assertTrue(result_df["completion_date"].iloc[-1] == expected_date)

    def test_label_raises_error_on_missing_commit_interval_days(self):
        data = {
            'path': ['f1'],
            'date': ['2024-12-20'],
            'line_change': [5],
            'size': [100]
        }
        df = pd.DataFrame(data)

        with self.assertRaisesRegex(ValueError, "'commit_interval_days' must be calculated before "
                                                "calling this labler"):
            self.labler.label(df)

    def test_label_deleted(self):
        data = {
            'path': ['f2', 'f2'],
            'date': ['2024-10-01', '2024-10-15'],
            'line_change': [50, -50],
            'size': [50, 0],
            'commit_interval_days': [0, 14],
        }

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        result_df, _, _, _ = self.labler.label(df)

        expected_date = pd.Timestamp('2024-10-15 00:00:00')
        self.assertEqual(result_df["completion_reason"].iloc[-1], "deleted")
        self.assertEqual(result_df["completion_date"].iloc[-1], expected_date)

    def test_label_idle_timeout(self):
        data = {
            'path': ['f3', 'f3', 'f3'],
            'date': ['2024-10-01', '2024-11-10', '2024-12-01'],
            'line_change': [100, 100, 100],
            'size': [100, 100, 100],
            'commit_interval_days': [0, 9, 22],
        }
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        result_df, _, _, _ = self.labler.label(df)

        expected_date = pd.Timestamp('2024-12-01 00:00:00')
        self.assertEqual(result_df["completion_reason"].iloc[-1], "idle_timeout")
        self.assertEqual(result_df["completion_date"].iloc[-1], expected_date)