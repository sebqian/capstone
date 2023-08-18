"""Multi-Modal image dataloader for ALREADY preprocessed images."""

from typing import Any, Dict, List
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchio as tio

import codebase.terminology as term

_PROBLEM = term.ProblemType.SEGMENTATION
_N_LABLES = 3


class MultiModalDataLoader():
    """This class is used to load multi-modal images.
    The data folder must follow the following structure:
    data_folder
    |-- train
        |-- images
            |-- patientID__CT.nii.gz (Note double _ here)
            |-- patientID__PT.nii.gz
            |-- patientID__DWI.nii.gz
                ...
        |-- labels
            |-- patientID.nii.gz (Note only one label file each ID)
    |-- valid
    |-- test
    |-- subvolume_x
        |-- train
            |-- images
                |-- patientID__input_idx.pt (Note double _ here)
                ...
            |-- labels
                |-- patientID__label_idx.pt (Note double _ here)
        |-- valid
        |-- test

    The files must have modality at the end of the filenames,
    for example: patient_01_PT.nii.gz
    All the images must have the same dimension and are already normalized.
    """

    def __init__(self, data_folder: Path,
                 phase: term.Phase,
                 modalities: List[term.Modality],
                 problem_type: term.ProblemType
                 ) -> None:
        self.data_path = data_folder / phase.value
        self.phase = phase
        self.modalities = modalities
        self.problem_type = problem_type

    def get_patient_lists(self) -> List[str]:
        """Find the patients in a specific csv file with corresponding phase."""
        label_path = self.data_path / 'labels'
        all_files = label_path.glob('*.nii.gz')
        patients = [a_file.stem.split('.')[0] for a_file in all_files]
        print(f'Find {len(patients)} patients for Phase: {self.phase.value} in {label_path}')
        return patients

    def create_subject(self, patient: str, with_weight: bool = True) -> tio.Subject:
        """Create subject for a patient."""
        subject_dict = {'ID': patient}
        for modality in self.modalities:
            img_file = self.data_path / 'images' / (patient + f'__{modality.value}.nii.gz')
            img = tio.ScalarImage(img_file)
            subject_dict[modality.value] = img  # type: ignore

        label_file = self.data_path / 'labels' / (patient + '.nii.gz')
        if self.problem_type == term.ProblemType.REGRESSION:
            label = tio.ScalarImage(label_file)
            subject_dict['LABEL'] = label  # type: ignore
        else:
            label = tio.LabelMap(label_file)
            subject_dict['LABEL'] = label  # type: ignore

        if with_weight:
            weight_file = self.data_path / 'images' / (patient + '__WEIGHT.nii.gz')
            img = tio.ScalarImage(weight_file)
            subject_dict['WEIGHT'] = img  # type: ignore
            tensor_shape = img.shape
            center = torch.zeros(tensor_shape)
            center[:, int(tensor_shape[1]/2), int(tensor_shape[2]/2), :] = 1.0
            center_line_img = tio.ScalarImage(tensor=center, affine=img.affine)
            subject_dict['CENTER'] = center_line_img

        subject = tio.Subject(subject_dict)
        return subject

    def create_subject_list(self) -> List[tio.Subject]:
        """Create subject list for all patients with all transforms."""
        patients = self.get_patient_lists()
        subjects = []
        for patient in patients:
            subjects.append(self.create_subject(patient))
        return subjects

    def create_augmentation(self, transform_keys: Dict[str, Dict[str, Any]],
                            ) -> tio.transforms.Transform:
        """Create transformation for data preprocessing and augmentation."""
        transform_collection = []
        if self.phase == term.Phase.TRAIN:
            for key, value in transform_keys.items():
                if 'affine' == key:
                    degrees = value['degrees']
                    translation = value['translation']
                    transform_collection.append(
                        tio.transforms.RandomAffine(degrees=degrees, translation=translation))
                elif 'flip' == key:
                    p = value['p']
                    axes = value['axes']
                    transform_collection.append(tio.transforms.RandomFlip(axes=axes, p=p))
                # elif 'blur' == key:
                #     transform_collection.append(tio.transforms.GaussianBlur((0.01, 0.5)))
                else:
                    raise ValueError(f'Unsupported transformation: {key}')
        transform_comp = tio.transforms.Compose(transform_collection)
        return transform_comp

    def create_subject_dataset(self, subjects: List[tio.Subject],
                               augmentation: tio.transforms.Transform
                               ) -> tio.SubjectsDataset:
        """Creates subject dataset.
        Args:
            subjects: list of subjects
            augmentation: augment transform applied to subjects
        Return:
            subject dataset
        """
        return tio.SubjectsDataset(subjects=subjects, transform=augmentation)

    def create_patch_dataloader(
            self, subject_dataset: tio.SubjectsDataset,
            max_queue_length: int, samples_per_volume: int,
            sampler: tio.data.PatchSampler, batch_size: int, num_workers: int
            ) -> DataLoader:
        """Create dataloader.
        Args:
            subject_dataset:
            max_queue_length: max number of patches stored in queue
            samples_per_volume: number of patches to extrat from each volume
            sampler: sampler operator
            num_workers: only for queue. dataloader must use 0
        Return:
        """
        patches_queue = tio.data.Queue(
            subjects_dataset=subject_dataset,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=num_workers,
        )
        return DataLoader(patches_queue, batch_size=batch_size, num_workers=0)
