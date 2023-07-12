"""Unit Test for tabular/utils.py"""
import unittest
from parameterized import parameterized
import pandas as pd
import numpy as np
from projects.psa_scanner import utils

_COL_NAMES = ['PID', 'Time1', 'Time2', 'psa1', 'psa2']


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data = pd.DataFrame(columns=_COL_NAMES)
        self.data.PID = ['20345', '75281', '55482']
        self.data.Time1 = [pd.Timestamp('2014-01-24 13:03:12.000000'),
                           pd.Timestamp('2014-01-27 11:57:18.000000'),
                           pd.Timestamp('2014-01-23 10:07:47.660000')]
        self.data.Time2 = [pd.Timestamp('2014-01-26 23:41:21.870000'),
                           pd.Timestamp('2014-04-27 15:38:22.540000'),
                           pd.Timestamp('2014-01-23 18:50:41.420000')]
        self.data.psa1 = [2, 1, 7]
        self.data.psa2 = [3, np.nan, 15]

    @parameterized.expand([
        ('hour', 'h', np.array([59.0, 2164.0, 9.0])),
        ('day', 'd', np.array([2.0, 90.0, 0.0])),
        ('month', 'm', np.array([0, 3.0, 0.0]))
    ])
    def test_calc_interval(self, name, t_unit, expected):
        """Test for time interval calc."""
        utils.calc_interval(self.data, 'diff', 'Time1', 'Time2', t_unit)
        calc = self.data['diff'].to_numpy()
        np.testing.assert_array_equal(calc, expected, f'{calc} does not equal to {expected}')

    @parameterized.expand([
        ('h', np.array([1/59.0, np.nan, 8.0/9.0])),
        ('d', np.array([1/2.0, np.nan, np.inf])),
        ('m', np.array([np.inf, np.nan, np.inf]))
    ])
    def test_calc_sequence_gradient(self, t_unit, expected):
        """Test for delta and gradient calc."""
        utils.calc_gradient(self.data, ('Time1', 'psa1'), ('Time2', 'psa2'),
                            'psa1', unit=t_unit)
        calc = self.data['psa1_gradient'].to_numpy()
        np.testing.assert_array_equal(calc, expected, f'{calc} is not equal to {expected}')

    @parameterized.expand([
        ('max', np.max, [np.nan, 14, 15, 12]),
    ])
    def test_extract_sequence_statistics(self, t_unit, expected):
        """Test for delta and gradient calc."""


if __name__ == '__main__':
    unittest.main()
