"""Unit Test for read_config.py"""
import unittest

import codebase.codebase_settings as cbs
from codebase.preprocessor import read_config

_CONFIG_FILE = cbs.CODEBASE_PATH / 'projects' / 'hecktor2022' / 'unittest_config.yml'


class TestReadConfig(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.gt_dict = {
            'model': {
                'spatial_dim': 3,
                'input_channel': 2,
                'output_channel': 3,
                'dropout': 0.3,
                'architecture': {
                    'use_conv_final': True,
                    'blocks_down': [1, 2, 2, 4, 4]}
            },
            'loss': {
                'name': 'generalized_dice_focal_loss',
                'include_background': False,
                'sigmoid': False,
                'softmax': True,
                'to_onehot_y': False
            },
            'metric': {
                'name': 'dice',
                'include_background': False,
                'num_classes': 3,
                'reduction': 'none',
            }
        }

    def test_read_config(self):
        config = read_config.read_configuration(_CONFIG_FILE)
        small_dict = {
            'model': {
                'spatial_dim': config['model']['spatial_dim'],
                'input_channel': config['model']['input_channel'],
                'output_channel': config['model']['output_channel'],
                'dropout': config['model']['dropout'],
                'architecture': {
                    'use_conv_final': config['model']['architecture']['use_conv_final'],
                    'blocks_down': config['model']['architecture']['blocks_down']}
            },
            'loss': {
                'name': config['loss']['name'],
                'include_background': config['loss']['include_background'],
                'sigmoid': config['loss']['sigmoid'],
                'softmax': config['loss']['softmax'],
                'to_onehot_y': config['loss']['to_onehot_y']
            },
            'metric': {
                'name': config['metric']['name'],
                'include_background': config['metric']['include_background'],
                'num_classes': config['metric']['num_classes'],
                'reduction': config['metric']['reduction'],
            }
        }
        self.assertDictEqual(small_dict, self.gt_dict)
        self.assertIsInstance(small_dict['loss']['sigmoid'], bool)
