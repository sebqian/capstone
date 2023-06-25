""" Data module based on pytorch lightning.
This module assuems that data has been preprocessed and different modalities have been stacked.
If preprocessing is necessary, please add preprocessing functions.
Assume the following folder structure for images files:
The data folder must follow the following structure:
    |-- root_dir
        |-- train
            |-- images
                |-- patientID_idx__input.pt (Note double _ here)
                ...
            |-- labels
                |-- patientID_idx__label.pt (Note double _ here)
        |-- valid
        |-- test
    All the images must have the same dimension and are already normalized.
"""
from typing import Any, Dict
from etils import epath

from torch.utils.data import DataLoader
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    RandFlipd,
    RandRotated,
    Compose,
    AsDiscreted,
    LoadImaged,
    EnsureTyped,
)
import pytorch_lightning as pl

import codebase.terminology as term

_TRANSFORM_DICT = {'flip': {'p': 0.5, 'axes': ('LR', 'AP')},
                   # ration range has to consider whether the channel exist or not
                   # because the transform assues no channels
                   'rotate': {'radians': [0, 0.5, 0.5], 'p': 0.8},
                   'affine': {'p': 0.5, 'degrees': 0.5, 'translation': 0.3}}


class MedicalImageDataModule(pl.LightningDataModule):
    """Image Data Module"""
    def __init__(self, task_type: term.ProblemType, root_dir: epath.Path,
                 experiment_config: Dict[str, Any], train_config: Dict[str, Any],
                 valid_config: Dict[str, Any], test_config: Dict[str, Any],
                 transform_dict: Dict[str, Any] = _TRANSFORM_DICT):
        super().__init__()
        self.task_type = task_type
        self.task = experiment_config['name']
        self.train_batch_size = train_config['batch_size']
        self.valid_batch_size = valid_config['batch_size']
        self.train_num_workers = train_config['num_workers']
        self.valid_num_workers = valid_config['num_workers']
        self.include_test = test_config['include']
        self.base_dir = epath.Path(experiment_config['data_path'])
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []
        self.transform_dict = transform_dict
        self.train_transform = None
        self.valid_transform = None
        self.train_set: Dataset
        self.val_set: Dataset
        self.test_set: Dataset

    def get_data_list(self):
        """Gets the lists of image ids for train and validation."""
        file_names = (self.base_dir / 'train' / 'images').glob('*.npy')
        train_ids = [file_name.stem.split('__')[0] for file_name in file_names]
        file_names = (self.base_dir / 'valid' / 'images').glob('*.npy')
        valid_ids = [file_name.stem.split('__')[0] for file_name in file_names]
        print(f'Locating data in {self.base_dir}: {len(train_ids)} for train'
              f' and {len(valid_ids)} for validation')
        return train_ids, valid_ids

    def get_test_data_list(self):
        """Gets the lists of image ids for test."""
        file_names = (self.base_dir / 'test' / 'images').glob('*.npy')
        test_ids = [file_name.stem.split('__')[0] for file_name in file_names]
        return test_ids

    def prepare_data(self):
        """Loads image ids."""
        self.train_ids, self.valid_ids = self.get_data_list()
        if self.include_test:
            self.test_ids = self.get_test_data_list()

    def setup(self, stage=None):
        """Sets up data."""
        self.train_transform, self.valid_transform = self.get_augmentation_transform(self.transform_dict)

        train_files = [{'input': str(self.base_dir / 'train' / 'images' / (id + '__input.npy')),
                       'label': str(self.base_dir / 'train' / 'labels' / (id + '__label.npy'))
                        } for id in self.train_ids]
        self.train_set = Dataset(data=train_files, transform=self.train_transform)

        valid_files = [{'input': str(self.base_dir / 'valid' / 'images' / (id + '__input.npy')),
                       'label': str(self.base_dir / 'valid' / 'labels' / (id + '__label.npy'))
                        } for id in self.valid_ids]
        self.val_set = Dataset(data=valid_files, transform=self.valid_transform)

        if self.include_test:
            self.test_ids = self.get_test_data_list()
            test_files = [{'input': str(self.base_dir / 'test' / 'images' / (id + '__input.npy')),
                          'label': str(self.base_dir / 'test' / 'labels' / (id + '__label.npy'))
                           } for id in self.test_ids]
            self.test_set = Dataset(data=test_files, transform=self.valid_transform)

    def get_augmentation_transform(self, transform_dict: Dict[str, Any]):
        """Gets augumentation transforms."""
        train_augmentation = Compose(
            [
                LoadImaged(keys=['input', 'label']),
                RandFlipd(keys=['input', 'label'], prob=transform_dict['flip']['p']),
                # RandAffined(keys=['input', 'label'], prob=transform_dict['affine']['p'],
                #             rotate_range=transform_dict['affine']['degrees'],),
                RandRotated(keys=['input', 'label'], prob=transform_dict['rotate']['p'],
                            range_x=transform_dict['rotate']['radians'][0],
                            range_y=transform_dict['rotate']['radians'][1],
                            range_z=transform_dict['rotate']['radians'][2],
                            padding_mode='border'),
                EnsureTyped(keys=['input', 'label']),
                AsDiscreted(keys=['label'], threshold=0.5)  # keep one-hot format
            ]
        )

        valid_augmentation = Compose(
            [
                LoadImaged(keys=['input', 'label']),
                EnsureTyped(keys=['input', 'label']),
                AsDiscreted(keys=['label'], threshold=0.5)  # keep one-hot format
            ]
        )
        return train_augmentation, valid_augmentation

    def train_dataloader(self):
        return DataLoader(self.train_set, self.train_batch_size, num_workers=self.train_num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.valid_batch_size, num_workers=self.valid_num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.valid_batch_size, num_workers=self.valid_num_workers)
