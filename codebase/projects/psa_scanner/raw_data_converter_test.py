import unittest
from parameterized import parameterized
import numpy as np
import pandas as pd


class TestRawDataConvertor(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.raw_data