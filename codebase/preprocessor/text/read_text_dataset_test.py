"""Unit Test for read_text_dataset.py"""
import unittest
# from parameterized import parameterized
import tensorflow as tf
import codebase_settings as cbs
from preprocessor.text import read_text_dataset

_TEST_PATH = cbs.CODEBASE_PATH / 'preprocessor' / 'text' / 'test_data'
_PATTERN = 'test*.csv'
_BATCH_SIZE = 5


class TestTextFromCSVProcessor(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.processor = read_text_dataset.TextFromCSVProcessor(
            data_dir=_TEST_PATH, pattern=_PATTERN, batch_size=_BATCH_SIZE,
            shuffle=False, phase='train', col_names=read_text_dataset._COL_NAMES,
            col_defaults=read_text_dataset._COL_DEFAULTS,
            label_col=read_text_dataset._COL_NAMES[-1])

    def test_get_dataset(self):
        dataset = self.processor.get_dataset()
        for note, label in dataset.take(1):
            self.assertEqual(tf.equal(note[4], '"just a test"'), True)
            # self.assertEqual(label[4].numpy().decode("utf-8"), 'no')
            self.assertEqual(label[1].numpy(), 1)
            self.assertEqual(label[4].numpy(), 0)


if __name__ == '__main__':
    unittest.main()
