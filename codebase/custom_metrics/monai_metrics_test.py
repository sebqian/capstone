"""Unit Test for monai_metrics.py"""
import unittest
from typing import Tuple
import torch
import numpy as np

from codebase.custom_metrics import monai_metrics

_METRICS_CONFIG = {'name': 'dice', 'include_background': False, 'reduction': 'none',
                   'get_not_nans': True, 'ignore_empty': False, 'num_classes': 3}


def _create_2d_spatial_fake_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates 2D spatial fake data.
        Pairs of grnds and preds can be combined into a one hot form.
    """
    grnd = torch.zeros(4, 4)  # [H, W]
    pred = torch.zeros(4, 4)

    # grnd= [0,0,0,0] pred= [0,0,0,0]
    #       [0,1,1,0]       [0,1,1,0]
    #       [0,1,1,0]       [1,1,0,0]
    #       [0,0,0,0]       [0,0,0,0]
    grnd[1, 1] = grnd[1, 2] = grnd[2, 1] = grnd[2, 2] = 1
    pred[1, 1] = pred[1, 2] = pred[2, 0] = pred[2, 1] = 1

    grnd1 = torch.zeros(4, 4)  # [H, W]
    pred1 = torch.zeros(4, 4)

    # grnd= [1,1,1,1] pred= [0,0,0,1]
    #       [0,0,0,0]       [0,0,0,1]
    #       [0,0,0,0]       [0,0,0,1]
    #       [0,0,0,0]       [0,0,0,1]
    grnd1[0, :] = 1
    pred1[:, 3] = 1

    grnd2 = torch.zeros(4, 4)  # [H, W]
    pred2 = torch.zeros(4, 4)

    # grnd= [0,0,0,0] pred= [1,1,1,0]
    #       [1,0,0,1]       [1,0,0,0]
    #       [1,0,0,1]       [0,0,1,0]
    #       [1,1,1,1]       [1,1,1,0]
    grnd2[3, :] = 1
    grnd2[1:3, 0] = grnd2[1:3, 3] = 1
    pred2[0, 0:3] = pred2[3, 0:3] = 1
    pred2[1, 0] = pred2[2, 2] = 1

    # Shape: [CHW]
    return torch.stack([pred, pred1, pred2]), torch.stack([grnd, grnd1, grnd2])


def create_2d_fake_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates 2D fake data. """
    grnd, pred = _create_2d_spatial_fake_data()
    grnd = torch.stack([grnd, grnd], dim=0)  # shape: [BCHW]
    pred = torch.stack([pred, pred], dim=0)
    return pred, grnd


class TestMonaiMetrics(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.features, self.targets = create_2d_fake_data()

    def test_segmentation_metric(self) -> None:
        metrics = monai_metrics.get_segmentation_metrics(config=_METRICS_CONFIG)
        prediction = metrics(self.features, self.targets).numpy()
        expected = np.array([[0.125, 0.25], [0.125, 0.25]]) * 2.0
        # self.assertEqual(torch.all(prediction.eq(expected)), True,
        #                  msg=f'Prediction {prediction} is not expected')
        np.testing.assert_array_almost_equal(
            prediction, expected, decimal=4,
            err_msg=f'Prediction {prediction} is not expected')


if __name__ == '__main__':
    unittest.main()
