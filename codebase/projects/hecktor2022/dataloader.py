"""HECKTOR Dataloaders."""
from typing import Tuple, Dict, Any, List
from etils import epath

from torch.utils.data import DataLoader
import torchio as tio

import codebase.terminology as term
from codebase.dataloader.images import multi_modal_dataloader
from codebase.dataloader.images import subvolume_dataloader


_MODALITIES = [term.Modality.CT, term.Modality.PET]
_TRANSFORM_DICT = {'flip': {'p': 0.5, 'axes': ('LR', 'AP')},
                   # ration range has to consider whether the channel exist or not
                   # because the transform assues no channels
                   'rotate': {'radians': [0, 0.2, 0.2], 'p': 0.7},
                   'affine': {'p': 0.5, 'degrees': 0.5, 'translation': 0.3}}
_PET_WEIGHT_THRESHOLD = 0.6


def get_loader(datafolder: epath.Path,
               phase: term.Phase,
               spatial_size: Tuple[int, int, int],
               batch_size: int = 1,
               max_queue_length: int = 1,
               samples_per_volume: int = 1,
               num_workers: int = 0,
               problem_type: term.ProblemType = term.ProblemType.SEGMENTATION
               ) -> DataLoader:
    """Gets dataloader."""

    # create preprocessing module
    hecktor_loader = multi_modal_dataloader.MultiModalDataLoader(
        data_folder=datafolder,
        phase=phase,
        modalities=_MODALITIES,
        problem_type=problem_type
    )

    # create dataset
    transformation = hecktor_loader.create_augmentation(transform_keys=_TRANSFORM_DICT)
    subjects = hecktor_loader.create_subject_list()
    subject_dataset = hecktor_loader.create_subject_dataset(
        subjects=subjects, augmentation=transformation)

    if phase == term.Phase.TRAIN:
        sampler = tio.data.WeightedSampler(patch_size=spatial_size, probability_map='CENTER')
        # sampler = tio.data.UniformSampler(patch_size=spatial_size)
        # sampler = tio.data.LabelSampler(patch_size=spatial_size,
        #                                 label_probabilities={0: 0, 1: 1, 2: 0})
    elif phase == term.Phase.VALID:  # need update
        sampler = tio.data.WeightedSampler(patch_size=spatial_size, probability_map='CENTER')
        # sampler = tio.data.UniformSampler(patch_size=spatial_size)
        # sampler = tio.data.LabelSampler(patch_size=spatial_size,
        #                                 label_probabilities={0: 0, 1: 1, 2: 0})
    else:  # Definitely need update
        sampler = tio.data.WeightedSampler(patch_size=spatial_size, probability_map='WEIGHT')

    dataloader = hecktor_loader.create_patch_dataloader(
        subject_dataset=subject_dataset,
        max_queue_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return dataloader


def get_subvolume_loader(datafolder: epath.Path,
                         phase: term.Phase,
                         transform_dict: Dict[str, Any],
                         batch_size: int = 1,
                         num_workers: int = 0):
    """Gets a data loader."""
    loader_processor = subvolume_dataloader.ProcessedSubVolumeDataLoader(
        data_folder=datafolder, phase=phase,
        batch_size=batch_size, transform_dict=transform_dict,
        num_workders=num_workers
    )
    return loader_processor.get_dataloader()


def get_train_valid_dataloaders(datafolder: epath.Path,
                                data_type: str,
                                spatial_size: Tuple[int, int, int],
                                batch_size: Tuple[int, int],
                                transform_dict: Dict[str, Any] = _TRANSFORM_DICT,
                                max_queue_length: int = 1,
                                samples_per_volume: int = 1,
                                num_workers: Tuple[int, int] = (1, 1),
                                problem_type: term.ProblemType = term.ProblemType.SEGMENTATION
                                ):
    """Get both train and valid dataloader"""
    if data_type == 'subvolume':
        print('Data type is subvolume.')
        train_loader = get_subvolume_loader(
            datafolder=datafolder,
            phase=term.Phase.TRAIN,
            transform_dict=transform_dict,
            batch_size=batch_size[0],
            num_workers=num_workers[0])

        valid_loader = get_subvolume_loader(
            datafolder=datafolder,
            phase=term.Phase.VALID,
            transform_dict=transform_dict,
            batch_size=batch_size[1],
            num_workers=num_workers[1])

    else:
        train_loader = get_loader(
            problem_type=problem_type,
            datafolder=datafolder,
            phase=term.Phase.TRAIN,
            spatial_size=spatial_size,
            batch_size=batch_size[0],
            max_queue_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            num_workers=num_workers[0]
        )

        valid_loader = get_loader(
            problem_type=problem_type,
            datafolder=datafolder,
            phase=term.Phase.VALID,
            spatial_size=spatial_size,
            batch_size=batch_size[1],
            max_queue_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            num_workers=num_workers[1]
        )

    return train_loader, valid_loader
