"""Unit Test for tabular_processor.py"""
import unittest
from parameterized import parameterized
import numpy as np
import pandas as pd
import datetime

import codebase_settings as cbs
from preprocessor.tabular import tabular_processor

_TEST_FILE = cbs.CODEBASE_PATH / 'preprocessor' / 'tabular' / 'test_data' / 'test01.xlsx'


class TestTabularProcessor(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.processor = tabular_processor.TabularProcessor(_TEST_FILE)

    @parameterized.expand([
        ({'primary_gleason_score': int}, np.int64),
        ({'secondary_gleason_score': float}, np.float64),
        ({'total_gleason_score': str}, str)
    ])
    def test_convert_data_type(self, conv_dict, expected):
        self.processor.convert_data_types(conv_dict)
        col = list(conv_dict.keys())[0]
        element = self.processor.data.loc[1, col]
        self.assertIsInstance(element, expected,
                              f'{element} has unexpected type: {type(element)}')

    @parameterized.expand([
        ({'primary_stage_n': {'n0': 0, 'n1': 1}}, [0, 0, 1, 0])
    ])
    def test_convert_category_to_numerical(self, conv_dict, expected):
        self.processor.convert_category_to_numerical(conv_dict)
        col = list(conv_dict.keys())[0]
        elements = self.processor.data[col]
        self.assertListEqual(elements.to_list(), expected)

    def test_sequence_builder(self):
        self.processor.sequence_builder(
            old_base_names=('psadate', 'psa'), reference=('rtstartdate', 'pretreatmentpsa'),
            new_base_names=('date_after_rt_t', 'psa_t'))
        time2 = self.processor.data['date_after_rt_t1'].to_list()
        expected = pd.to_datetime(
            pd.Series([pd.NaT, '10/13/2021', '11/03/2022', '8/24/2021'])).to_list()
        self.assertListEqual(time2, expected, f'{time2} not equal to {expected}')


if __name__ == '__main__':
    unittest.main()
