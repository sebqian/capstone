"""Unit Test for model_module_test.py"""
import unittest
import torch
from etils import epath

from lightning_module import seg_model_module
from codebase.projects.hecktor2022 import read_config

_CONFIG_FILE = epath.Path('/workspace/codebase/projects/hecktor2022/unittest_config.yml')


# Define the unit test class
class SegmentationModelModuleTestCase(unittest.TestCase):

    # @patch("monai_models.get_model", new=MagicMock(return_value=torch.nn.Identity()))
    def setUp(self):
        config = read_config.read_experiment_config(_CONFIG_FILE)
        config['metric']['num_classes'] = 2
        self.module = seg_model_module.SegmentationModelModule(
            config, optimizer_class=torch.optim.AdamW)
        self.module.net = torch.nn.Identity()

        self.test_tensor = torch.tensor(  # shape = [1, 2, 3, 3]
            [
                [
                    [[1, 2, 3], [4, 5, 6], [16, 17, 18]],
                    [[10, 11, 12], [13, 14, 15], [7, 8, 9]]
                ]
            ], dtype=torch.float
        )
        self.test_label = torch.tensor(  # shape = [1, 2, 3, 3]
            [
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
                ]
            ]
        )
        self.batch = {'input': self.test_tensor, 'label': self.test_label}

    def test_model(self):
        prediction = self.module.net(self.test_tensor)
        self.assertEqual(torch.equal(prediction, self.test_tensor), True)

    def test_logits_to_onehot(self):
        one_hot_tensor = self.module.logits_to_onehot(self.test_tensor)
        expected = torch.tensor(  # shape = [1, 2, 3, 3]
            [
                [
                    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [0, 0, 0]]
                ]
            ], dtype=torch.float
        )
        self.assertEqual(torch.all(one_hot_tensor.eq(expected)), True,
                         f'One hot tensor {one_hot_tensor} is not expected.')
        self.assertIsInstance(one_hot_tensor, torch.LongTensor)

    def test_training_step(self):
        batch = {'input': self.test_tensor, 'label': self.test_label}
        loss = self.module.training_step(batch, 0)
        self.assertAlmostEqual(loss.item(), 0.2, places=3, msg=f'Unexpected loss: {loss}')

    # def test_one_hot(self):
    #     """Test forward function."""
    #     original_1 = torch.sum(torch.eq(torch.round(y), 1))
    #     one_hot_1 = torch.sum(torch.gt(one_hot_y[:, 1, ...], 0.55))
    #     original_2 = torch.sum(torch.eq(torch.round(y), 2))
    #     one_hot_2 = torch.sum(torch.gt(one_hot_y[:, 2, ...], 0.55))
    #     deviation = 1 - one_hot_2 / original_2
    #     assert torch.abs(deviation) > 0.05, \
    #         f'Original count {(original_1, original_2)} vs one hot count {(one_hot_1, one_hot_2)}'
