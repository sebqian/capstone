"""Unit Test for monai_metrics.py"""
import unittest
from typing import Tuple
import torch
import numpy as np

from codebase.custom_metrics import monai_metrics

_METRICS_CONFIG = {'name': 'generalized_dice', 'include_background': False, 'reduction': 'mean',
                   'get_not_nans': True, 'num_classes': 3}


def _create_2d_spatial_fake_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates 2D spatial fake data. """
    grnd = torch.zeros(4, 4)  # [H, W]
    pred = torch.zeros(4, 4)

    # grnd= [0,0,0,0] pred= [0,0,0,0]
    #       [0,1,1,0]       [0,0.85,0.85,0]
    #       [0,1,1,0]       [0.85,0.85,0,0]
    #       [0,0,0,0]       [0,0,0,0]
    grnd[1, 1] = grnd[1, 2] = grnd[2, 1] = grnd[2, 2] = 1
    pred[1, 1] = pred[1, 2] = pred[2, 0] = pred[2, 1] = 0.85
    return pred, grnd


def create_2d_fake_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates 2D fake data. """
    grnd, pred = _create_2d_spatial_fake_data()
    grnd_2d = [grnd, grnd, grnd]  # 3 classes
    grnd_2d = torch.stack(grnd_2d, dim=0)
    pred_2d = [pred, pred, grnd]
    pred_2d = torch.stack(pred_2d, dim=0)
    assert pred_2d.shape == (3, 4, 4), f'Bad shape: {pred_2d.shape}'
    return pred, grnd


def create_3d_fake_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates 3D fake data. """
    pred_2d, grnd_2d = create_2d_fake_data()
    grnd_3d = [grnd_2d, grnd_2d]  # D==2 slices
    grnd_3d = torch.stack(grnd_3d, dim=-1)
    grnd_3d = [grnd_3d, grnd_3d, grnd_3d]  # C=2 classes
    grnd_3d = torch.stack(grnd_3d, dim=0)
    pred_3d = [pred_2d, pred_2d]
    pred_3d = torch.stack(pred_3d, dim=-1)
    pred_3d = [pred_3d, pred_3d, pred_3d]
    pred_3d = torch.stack(pred_3d, dim=0)
    assert pred_3d.shape == (3, 4, 4, 2), f'Bad shape: {pred_3d.shape}'
    return pred_3d, grnd_3d


class TestSegmentationMetrics(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.features, self.targets = create_3d_fake_data()

    def test_segmentation_metric(self) -> None:
        metrics = monai_metrics.get_segmentation_metrics(config=_METRICS_CONFIG)
        prediction = metrics(self.features[None], self.targets[None])
        prediction = prediction.mean().item()  # type:ignore
        expected = 0.6892
        self.assertAlmostEqual(prediction, expected, delta=0.0001,
                               msg=f'Prediction {prediction} is not expected')
        # np.testing.assert_array_almost_equal(
        #     prediction, expected, decimal=4,
        #     err_msg=f'Prediction {prediction} is not expected')


if __name__ == '__main__':
    unittest.main()
