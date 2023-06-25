"""subvolume tensors dataloader for ALREADY preprocessed images."""

from typing import Any, Dict
from etils import epath
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Transform,
    RandFlipd,
    RandRotated,
    RandAffined,
    Compose,
    AsDiscreted,
    LoadImaged,
    EnsureTyped,
)

import codebase.terminology as term

_TRANSFORM_DICT = {'flip': {'p': 0.5, 'axes': ('LR', 'AP')},
                   # ration range has to consider whether the channel exist or not
                   # because the transform assues no channels
                   'rotate': {'radians': [0, 0.5, 0.5], 'p': 0.8},
                   'affine': {'p': 0.5, 'degrees': 0.5, 'translation': 0.3}}


def _create_subvolume_transforms(phase: term.Phase,
                                 transform_dict: Dict[str, Any] = _TRANSFORM_DICT) -> Transform:
    """Create transforms."""
    if phase == term.Phase.TRAIN:
        transformation = Compose(
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
    else:
        transformation = Compose(
            [
                LoadImaged(keys=['input', 'label']),
                EnsureTyped(keys=['input', 'label']),
                AsDiscreted(keys=['label'], threshold=0.5)  # keep one-hot format
            ]
        )
    return transformation


class ProcessedSubVolumeDataLoader():
    """This class is used to load processed subvolume data saved as tensors.
    The data folder must follow the following structure:
    |-- subvolume_x
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

    def __init__(self, data_folder: epath.Path,
                 phase: term.Phase,
                 batch_size: int,
                 transform_dict: Dict[str, Any],
                 num_workders: int
                 ) -> None:
        self.data_path = data_folder / phase.value
        self.phase = phase
        self.batch_size = batch_size
        self.num_workers = num_workders
        file_names = (self.data_path / 'images').glob('*.npy')
        self.ids = [file_name.stem.split('__')[0] for file_name in file_names]
        print(f'Find {len(self.ids)} examples for {phase.value}')
        self.transforms = _create_subvolume_transforms(self.phase, transform_dict)

    def get_dataloader(self) -> DataLoader:
        """Get a dataloader."""
        all_files = [{'input': str(self.data_path / 'images' / (id + '__input.npy')),
                      'label': str(self.data_path / 'labels' / (id + '__label.npy'))
                      } for id in self.ids]
        dataset = CacheDataset(data=all_files, transform=self.transforms)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
