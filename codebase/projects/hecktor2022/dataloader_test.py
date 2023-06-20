"""Unit Test for dataloader.py"""
import unittest
from typing import List, Optional
from parameterized import parameterized
import torch

import codebase.codebase_settings as cbs
from trash_can import dataloader
import codebase.terminology as term

_TEST_PATH = cbs.CODEBASE_PATH / 'projects' / 'hecktor2022' / 'test_data'
_EXPECTED_SLICE = torch.Tensor([[1, 2, 0, 0], [2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
_EXPECTED_INPUT_SHAPE = (1, 2, 3, 4, 4)  # (B, C, D, W, H)
_EXPECTED_LABEL_SHAPE = (1, 1, 3, 4, 4)  # (B, C, D, W, H)


class TestHECKTORDataset(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    @parameterized.expand([
        ([], _EXPECTED_SLICE),
        (['h_flip'], None),
    ])
    def test_get_loader(self, transform_types: List[str],
                        expected: Optional[torch.Tensor]) -> None:
        train_loader = dataloader.get_loader(
            _TEST_PATH, term.Phase.TRAIN, transform_types=transform_types)
        train_features, train_labels = next(iter(train_loader))
        self.assertEqual(train_features.shape, _EXPECTED_INPUT_SHAPE,
                         f'{train_features.shape} is not expected')
        self.assertEqual(train_labels.shape, _EXPECTED_LABEL_SHAPE,
                         f'{train_labels.shape} is not expected')
        train_slice = train_features[0, 0, 1, :, :]
        label_slice = train_labels[0, 0, 1, :, :]
        if expected is not None:
            self.assertEqual(train_slice.equal(expected),
                             True, f'{train_slice}')
        else:
            mask = (train_slice > 0).int()
            self.assertEqual(mask.equal(label_slice),
                             True, f'{mask} is not {label_slice}')


if __name__ == '__main__':
    unittest.main()
