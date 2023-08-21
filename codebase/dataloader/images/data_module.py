""" Data module based on pytorch lightning.
This module assuems that data has been preprocessed and different modalities have been stacked.
If preprocessing is necessary, please add preprocessing functions.
Assume the following folder structure for images files:
The data folder must follow the following structure:
    |-- root_dir
        |-- train
            |-- images
                |-- patientID_idx__CT.nii.gz (Note double _ here)
                |-- patientID_idx__PT.nii.gz (Note double _ here)
                ...
            |-- labels
                |-- patientID_idx.nii.gz (Note double _ here)
        |-- valid
        |-- test
    All the images must have the same dimension and are already normalized.
"""
from typing import Any, Dict
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import (
    ConcatItemsd,
    RandAffined,
    RandFlipd,
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureChannelFirstd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandStdShiftIntensityd,
)
import pytorch_lightning as pl

import codebase.terminology as term

_TRANSFORM_DICT = {'flip': {'p': 0.5, 'axes': (0, 1)},
                   # ration range has to consider whether the channel exist or not
                   # because the transform assues no channels
                   'rotate': {'radians': [0.5, 0.5, 0.0], 'p': 0.8},
                   'affine': {'p': 0.5, 'degrees': 0.5, 'translation': 0.3}}


class MedicalImageDataModule(pl.LightningDataModule):
    """Image Data Module"""
    def __init__(self, task_type: term.ProblemType,
                 config: Dict[str, Any],
                 transform_dict: Dict[str, Any] = _TRANSFORM_DICT):
        super().__init__()
        self.task_type = task_type
        self.configs = config
        self.task = self.configs['experiment']['name']
        self.spatial_size = self.configs['model']['spatial_size']
        self.train_batch_size = self.configs['train']['batch_size']
        self.valid_batch_size = self.configs['valid']['batch_size']
        self.test_batch_size = self.configs['test']['batch_size']
        self.train_num_workers = self.configs['train']['num_workers']
        self.valid_num_workers = self.configs['valid']['num_workers']
        self.include_test = self.configs['test']['include']
        self.base_dir = Path(self.configs['experiment']['data_path'])
        self.train_ids = []
        self.valid_ids = []
        self.test_ids = []
        self.transform_dict = transform_dict
        self.train_transform = None
        self.valid_transform = None
        self.train_set: Dataset
        self.val_set: Dataset
        self.test_set: Dataset

    def get_data_list(self):
        """Gets the lists of image ids for train and validation."""
        file_names = (self.base_dir / 'train' / 'images').glob('*__CT.nii.gz')
        train_ids = [file_name.stem.split('__')[0] for file_name in file_names]
        file_names = (self.base_dir / 'valid' / 'images').glob('*__CT.nii.gz')
        valid_ids = [file_name.stem.split('__')[0] for file_name in file_names]
        print(f'Locating data in {self.base_dir}: {len(train_ids)} for train'
              f' and {len(valid_ids)} for validation')
        return train_ids, valid_ids

    def get_test_data_list(self):
        """Gets the lists of image ids for test."""
        file_names = (self.base_dir / 'test' / 'images').glob('*__CT.nii.gz')
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

        train_files = [{'CT': str(self.base_dir / 'train' / 'images' / (id + '__CT.nii.gz')),
                        'PT': str(self.base_dir / 'train' / 'images' / (id + '__PT.nii.gz')),
                        'label': str(self.base_dir / 'train' / 'labels' / (id + '.nii.gz'))
                        } for id in self.train_ids]
        self.train_set = Dataset(data=train_files, transform=self.train_transform)

        valid_files = [{'CT': str(self.base_dir / 'valid' / 'images' / (id + '__CT.nii.gz')),
                        'PT': str(self.base_dir / 'valid' / 'images' / (id + '__PT.nii.gz')),
                        'label': str(self.base_dir / 'valid' / 'labels' / (id + '.nii.gz'))
                        } for id in self.valid_ids]
        self.val_set = Dataset(data=valid_files, transform=self.valid_transform)

        if self.include_test:
            self.test_ids = self.get_test_data_list()
            test_files = [{'CT': str(self.base_dir / 'test' / 'images' / (id + '__CT.nii.gz')),
                           'PT': str(self.base_dir / 'test' / 'images' / (id + '__PT.nii.gz')),
                           'label': str(self.base_dir / 'test' / 'labels' / (id + '.nii.gz'))
                           } for id in self.test_ids]
            self.test_set = Dataset(data=test_files, transform=self.valid_transform)

    def get_augmentation_transform(self, transform_dict: Dict[str, Any]):
        """Gets augumentation transforms."""
        train_augmentation = Compose(
            [
                LoadImaged(keys=['CT', 'PT', 'label'], image_only=False),
                EnsureChannelFirstd(keys=['CT', 'PT', 'label']),
                RandGaussianNoised(keys=['CT']),
                RandStdShiftIntensityd(keys=['CT'], factors=0.2),
                RandScaleIntensityd(keys=['CT'], factors=0.1),
                RandFlipd(keys=['CT', 'PT', 'label'], prob=transform_dict['flip']['p'],
                          spatial_axis=transform_dict['flip']['axes']),
                # RandAffined(keys=['input', 'label'], prob=transform_dict['affine']['p'],
                #             rotate_range=transform_dict['affine']['degrees'],),
                RandAffined(keys=['CT', 'PT', 'label'],
                            mode=('bilinear', 'bilinear', 'nearest'),
                            prob=1.0,
                            spatial_size=tuple(self.configs['model']['spatial_size']),
                            translate_range=(10, 10, 2),
                            rotate_range=(np.pi / 18, np.pi / 18, np.pi / 2),
                            scale_range=(0.15, 0.15, 0.15),
                            padding_mode='border'),
                EnsureTyped(keys=['CT', 'PT', 'label']),  # Note: label not in one-hot form
                # AsDiscreted(keys=['label'], to_onehot=self.configs['metric']['num_classes'])
                ConcatItemsd(keys=['CT', 'PT'], name="input", dim=0)
            ]
        )

        valid_augmentation = Compose(
            [
                LoadImaged(keys=['CT', 'PT', 'label'], image_only=False),
                EnsureChannelFirstd(keys=['CT', 'PT', 'label']),
                # AsDiscreted(keys=['label'], to_onehot=self.configs['metric']['num_classes'])
                ConcatItemsd(keys=['CT', 'PT'], name="input", dim=0)
            ]
        )
        return train_augmentation, valid_augmentation

    def train_dataloader(self):
        # p_dataset = PatchDataset(self.train_set, patch_func=lambda x: x,  # type: ignore
        #                          samples_per_image=self.configs['train']['samples_per_volume'])
        dataloader = DataLoader(self.train_set, batch_size=self.train_batch_size,
                                num_workers=self.train_num_workers, shuffle=True)
        print(f'Train dataloader length: {len(dataloader)}')
        if len(dataloader) == 0:
            raise ValueError('No train data batch available.')
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_set, batch_size=self.valid_batch_size,
                                num_workers=self.valid_num_workers, shuffle=False)
        print(f'Validation dataloader length: {len(dataloader)}')
        if len(dataloader) == 0:
            raise ValueError('No validation data batch available.')
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_set, batch_size=self.test_batch_size,
                                num_workers=self.valid_num_workers, shuffle=False)
        print(f'Test dataloader length: {len(dataloader)}')
        if len(dataloader) == 0:
            raise ValueError('No test data batch available.')
        return dataloader
