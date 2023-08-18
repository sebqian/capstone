"""Unit Test for model_module_test.py"""
import unittest
import numpy as np
import torch
from pathlib import Path

import codebase.terminology as term
from codebase.dataloader.images import data_module
import codebase.codebase_settings as cbs
from codebase.projects.hecktor2022 import read_config

_CONFIG_FILE = cbs.CODEBASE_PATH / 'projects' / 'hecktor2022' / 'unittest_config.yml'
_DATA_FOLDER = '/workspace/codebase/dataloader/images/test_data'
_DATA_ID = 'CHUP-052_108'
_DATA_FILE = Path(_DATA_FOLDER) / 'train' / 'images' / (_DATA_ID + '__input.npy')
_LABEL_FILE = Path(_DATA_FOLDER) / 'train' / 'labels' / (_DATA_ID + '__label.npy')


# Define the unit test class
class MedicalImageDataModuleTestCase(unittest.TestCase):

    def setUp(self):
        config = read_config.read_experiment_config(_CONFIG_FILE)
        config['experiment']['data_path'] = _DATA_FOLDER
        self.assertEqual(config['metric']['num_classes'], 3,
                         f"Unexpected classes: {config['metric']['num_classes']}")
        self.plm_data = data_module.MedicalImageDataModule(
            task_type=term.ProblemType.SEGMENTATION,
            config=config,
            transform_dict=data_module._TRANSFORM_DICT
        )
        self.plm_data.prepare_data()
        self.gt_image = torch.Tensor(np.load(_DATA_FILE))
        self.gt_label = torch.Tensor(np.load(_LABEL_FILE))

    def test_prepare_data(self):
        self.assertEqual(self.plm_data.valid_ids, [_DATA_ID])

    def test_val_dataloader(self):
        """Test dataloader without transforms."""
        self.plm_data.setup()
        test_loader = self.plm_data.val_dataloader()
        test_data = [x for x in test_loader]
        self.assertEqual(len(test_data), 1, f'Unexpected data length: {len(test_data)}')
        input_data = test_data[0]['input']
        input_label = test_data[0]['label']
        self.assertEqual(input_data.shape, (1, 2, 256, 256, 32),
                         f'Unexpected input shape {input_data.shape}')
        self.assertEqual(input_label.shape, (1, 3, 256, 256, 32),
                         f'Unexpected label shape {input_label.shape}')
        original_1 = torch.sum(torch.eq(self.gt_label, 1))
        one_hot_1 = torch.sum(torch.gt(input_label[0, 1, ...], 0.55))
        self.assertEqual(original_1, one_hot_1,
                         f'Channel 1: original count {original_1} != data count {one_hot_1}')
        original_2 = torch.sum(torch.eq(self.gt_label, 2))
        one_hot_2 = torch.sum(torch.gt(input_label[:, 2, ...], 0.55))
        self.assertEqual(original_2, one_hot_2,
                         f'Channel 2: original count {original_2} != data count {one_hot_2}')

    def test_train_dataloader(self):
        """Test dataloader without transforms."""
        self.plm_data.setup()
        test_loader = self.plm_data.train_dataloader()
        test_data = [x for x in test_loader]
        input_data = test_data[0]['input']
        input_label = test_data[0]['label']
        np.save(_DATA_FOLDER + '/temp__input.npy', input_data.squeeze(0).numpy())
        np.save(_DATA_FOLDER + '/temp__label.npy', input_label.squeeze(0).numpy())
        self.assertEqual(input_data.shape, (1, 2, 256, 256, 32),
                         f'Unexpected input shape {input_data.shape}')
        self.assertEqual(input_label.shape, (1, 3, 256, 256, 32),
                         f'Unexpected label shape {input_label.shape}')
        original_1 = torch.sum(torch.eq(torch.round(self.gt_label), 1))
        one_hot_1 = torch.sum(torch.gt(input_label[0, 1, ...], 0.55))
        self.assertEqual(original_1, one_hot_1,
                         f'Channel 1: original count {original_1} != data count {one_hot_1}')
        original_2 = torch.sum(torch.eq(torch.round(self.gt_label), 2))
        one_hot_2 = torch.sum(torch.gt(input_label[:, 2, ...], 0.55))
        self.assertEqual(original_2, one_hot_2,
                         f'Channel 2: original count {original_2} != data count {one_hot_2}')
