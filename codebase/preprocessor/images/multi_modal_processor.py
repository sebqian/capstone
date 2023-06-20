"""Multi-Modal image preprocessor."""
import csv
import random
from typing import Dict, List, Tuple
from etils import epath
import numpy as np
import pandas as pd
import torch
import torchio as tio

import codebase.terminology as term

_MAX_HU = 1024
_MIN_HU = -1024
_N_LABLES = 3
_EDGE_SLICE = 16
_SUBVOLUME_INTERVAL = 4
_SELECTED_EMPTY_LABEL = 3


def _find_subvolume_index(
        label: torch.Tensor,
        valid_channel: List[int],
        border: int) -> Tuple[List[int], List[int]]:
    """Find the index can be used for subvolumes.
    Args:
        label: [C, H, W, D]
        valid: the list of channel numbers to be considered in label
    Returns:
        good_index for nonzero volume, bad_index for all zero volume
            borders are completely excluded.
    """
    slice_idx = range(border, label.shape[-1]-border)
    good_index = []
    bad_index = []
    for idx in slice_idx:
        a_slice = label[valid_channel, :, :, idx]
        if a_slice.sum() > 0:
            good_index.append(idx)
        else:
            bad_index.append(idx)
    return good_index, bad_index


class MultiModalProcessor():
    """This class is used to preprocess multi-modal images.
    The data folder must follow the following structure:
    data_folder
    |-- *_in_train.csv: list of ids in train
    |-- *_in_valid.csv: list of ids in validation
    |-- *_in_test.csv: list of ids in test
    |-- images
        |-- patientID__CT.nii.gz (Note double _ here)
            ...
    |-- labels
        |-- patientID.nii.gz

    The files must have modality at the end of the filenames,
    for example: patient_01_PT.nii.gz
    """

    def __init__(self, data_folder: epath.Path,
                 phase: term.Phase,
                 modalities: List[term.Modality],
                 reference: term.Modality,
                 problem_type: term.ProblemType
                 ) -> None:
        self.data_path = data_folder
        self.phase = phase
        self.modalities = modalities
        self.reference = reference
        self.problem_type = problem_type

    def get_patient_lists(self) -> List[str]:
        """Find the patients in a specific csv file with corresponding phase."""
        csv_files = self.data_path.glob(f'*_in_{self.phase.value}.csv')
        csv_file = [x for x in csv_files]
        if len(csv_file) > 1:
            raise ValueError(f'More than 1 file is found: {csv_file}')
        if len(csv_file) == 0:
            raise ValueError(f'No file is found here {csv_files}')
        print(f'Find patient list file: {csv_file}')
        csv_file = csv_file[0]
        with open(csv_file, 'r') as f:
            patients = list(csv.reader(f, delimiter=','))
        patients = patients[0]
        patients = [patient for patient in patients]
        print(f'Find {len(patients)} patients for Phase: {self.phase.value}')
        return patients

    def create_subject(self, patient: str) -> tio.Subject:
        """Create subject for a patient."""
        subject_dict = {'ID': patient}
        for modality in self.modalities:
            img_file = self.data_path / 'images' / (patient + f'__{modality.value}.nii.gz')
            img = tio.ScalarImage(img_file)
            subject_dict[modality.value] = img  # type: ignore
        label_file = self.data_path / 'labels' / (patient + '.nii.gz')
        if self.problem_type == term.ProblemType.REGRESSION:
            label = tio.ScalarImage(label_file)
        elif self.problem_type == term.ProblemType.SEGMENTATION:
            label = tio.LabelMap(label_file)
            if _N_LABLES > 1:
                onehot_transform = tio.transforms.OneHot(num_classes=_N_LABLES)
                label = onehot_transform(label)
        subject_dict['LABEL'] = label  # type: ignore
        subject = tio.Subject(subject_dict)
        return subject

    def create_subject_list(self) -> List[tio.Subject]:
        """Create subject list for all patients with all transforms."""
        patients = self.get_patient_lists()
        subjects = []
        for patient in patients:
            subjects.append(self.create_subject(patient))
        return subjects

    def check_resolution(self, subjects: List[tio.Subject]) -> None:
        """Print out the data size and resolution for patients."""
        for subject in subjects:
            print(f'Patient: {subject.ID}')  # type: ignore
            for modality in self.modalities:
                scalar_img = subject[modality.value]
                print(f'\t {modality.value}: Shape - {scalar_img.shape}; Spacing - {scalar_img.spacing}.')

    def resample_to_reference(self, subject: tio.Subject, xy_size: Tuple[int, int]) -> tio.Subject:
        """Assume all the volumes are already co-registered."""
        ref_img = subject[self.reference.value]
        image_size = (xy_size[0], xy_size[1], ref_img.shape[-1])
        resize_for_ref = tio.transforms.Resize(image_size)
        ref_img = resize_for_ref(ref_img)
        resample_transform = tio.transforms.Resample(target=ref_img)  # type: ignore
        return tio.Subject(resample_transform(subject))

    def train_histogram_standardization(
            self, modality: term.Modality) -> np.ndarray:
        """Train histogram standardization for normalization purpose.
        Only need to run once for each dataset.
        Args:
            modality: one modality
            output: path to save landmarks
        Return:
        """
        output_file = self.data_path / f'{modality.value}_landmarks.npy'
        if output_file.exists():  # no need to retrain
            return np.load(output_file)
        img_path = self.data_path / 'images'
        img_files = img_path.glob(f'*__{modality.value}.nii.gz')
        img_files = [f for f in img_files]
        landmarks = tio.HistogramStandardization.train(
            img_files, output_path=output_file)  # type: ignore
        return landmarks

    def create_landmark_dict(self) -> Dict[str, np.ndarray]:
        """Create landmarks for all modalities except for CT."""
        landmarks_dict = {}
        for modality in self.modalities:
            if modality != term.Modality.CT:
                landmarks = self.train_histogram_standardization(modality)
                landmarks_dict[modality.value] = landmarks
        return landmarks_dict

    def create_normalization(self) -> tio.transforms.Transform:
        """Create transformation for data preprocessing and augmentation."""
        transform_collection = []
        # First image normalization for all phases
        landmarks_dict = self.create_landmark_dict()
        for modality in self.modalities:
            if modality == term.Modality.CT:  # CT only
                transform_collection.append(
                    tio.transforms.Clamp(out_min=_MIN_HU, out_max=_MAX_HU,
                                         include=[term.Modality.CT.value]),
                )
                transform_collection.append(
                    tio.transforms.RescaleIntensity(out_min_max=(-1, 1),
                                                    include=[term.Modality.CT.value]),
                )
            else:
                exclude_types = [term.Modality.CT.value, 'LABEL', 'WEIGHT']
                transform_collection.append(tio.transforms.HistogramStandardization(
                    landmarks_dict, exclude=exclude_types))  # type: ignore
                transform_collection.append(tio.transforms.ZNormalization(
                    masking_method=tio.ZNormalization.mean, exclude=exclude_types))

        transform_comp = tio.transforms.Compose(transform_collection)
        return transform_comp

    def preprocess_and_save(self,
                            xy_size: Tuple[int, int],
                            weight_modality: term.Modality,
                            weight_threshold: float) -> int:
        """Preprocess data and save to disk."""
        output_folder = self.data_path / f'processed_{xy_size[0]}x{xy_size[1]}' / self.phase.value
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)
            (output_folder / 'images').mkdir()
            (output_folder / 'labels').mkdir()
        normalizator = self.create_normalization()
        # read raw data
        print('Read raw data ...')
        subjects = self.create_subject_list()
        # process subject one by one:
        n = 0
        for subject in subjects:
            patient_id = subject['ID']
            print(f'Starting preprocessing for {patient_id}')
            # apply resampling
            print('\t Resampling images ...')
            subject = self.resample_to_reference(subject, xy_size)
            # apply normalization
            print('\t Normalizing images ...')
            subject = normalizator(subject)
            # generate weight maps
            print('\t Generating weight maps ...')
            subject = self.create_weight(subject, weight_modality,  # type: ignore
                                         weight_threshold, edge=_EDGE_SLICE)
            # save data
            print('\t Saving data ...')
            for modality in self.modalities:
                filename = output_folder / 'images' / f'{patient_id}__{modality.value}.nii.gz'
                subject[modality.value].save(filename)
            filename = output_folder / 'images' / f'{patient_id}__WEIGHT.nii.gz'
            subject['WEIGHT'].save(filename)
            filename = output_folder / 'labels' / f'{patient_id}.nii.gz'
            subject['LABEL'].save(filename)
            n += 1
        return n

    def create_weight(self, subject: tio.Subject, weight_modality: term.Modality,
                      weight_threshold: float, edge: int) -> tio.Subject:
        """Create weight for a subject."""
        ref = subject[weight_modality.value]
        shape = ref.data.shape
        # move all edges
        weight_tensor = torch.zeros(shape)
        weight_tensor[..., edge:shape[-3]-edge, edge:shape[-2]-edge, edge:shape[-1]-edge] = 1
        weight_tensor = torch.mul(weight_tensor, ref.data)
        weight = tio.LabelMap(tensor=weight_tensor > weight_threshold, affine=ref.affine)
        weight_tensor = weight.data
        weight_tensor[..., 0:edge] = 0
        weight_tensor[..., -edge:-1] = 0
        weight.set_data(weight_tensor)
        subject.add_image(weight, 'WEIGHT')
        return subject

    def create_and_save_subvolumes(self, data_path: epath.Path,
                                   valid_channel: List[int],
                                   subvolume_intervel: int,
                                   subvolume_size: int):
        """This function is used to created subvolume data.
        It should be applied to the already proprocessed data.
        Subvolumes will be saved as tensor directly.
        """
        output_path = data_path / ('subvolume_' + str(subvolume_size))
        out_img_path = output_path / self.phase.value / 'images'
        if not out_img_path.exists():
            out_img_path.mkdir(parents=True, exist_ok=True)
        out_label_path = output_path / self.phase.value / 'labels'
        if not out_label_path.exists():
            out_label_path.mkdir(parents=True, exist_ok=True)
        label_path = data_path / self.phase.value / 'labels'
        all_files = label_path.glob('*.nii.gz')
        patients = [a_file.stem.split('.')[0] for a_file in all_files]
        print(f'Find {len(patients)} patients for Phase: {self.phase.value} in {label_path}')
        for patient in patients:
            print(f'Processing {patient}')
            images = []
            for modality in self.modalities:
                img_file = data_path / self.phase.value / 'images' / (patient + f'__{modality.value}.nii.gz')
                img = tio.ScalarImage(img_file)
                images.append(img.data[0])
            image = torch.stack(images, dim=0)
            label_file = data_path / self.phase.value / 'labels' / (patient + '.nii.gz')
            label = tio.ScalarImage(label_file).data
            half_volume_size = int(subvolume_size / 2) + 1
            good_index, bad_index = _find_subvolume_index(label, valid_channel, half_volume_size)
            previous_idx = 0
            for idx in good_index:
                if idx > previous_idx + subvolume_intervel:
                    slice_lower = idx - half_volume_size
                    slice_upper = idx + subvolume_size - half_volume_size
                    subvolume_img = image[:, :, :, slice_lower: slice_upper]
                    filename = patient + '_' + str(idx) + '__input' + '.npy'
                    np.save(arr=subvolume_img.numpy(), file=(out_img_path / filename))
                    subvolume_label = label[:, :, :, slice_lower: slice_upper]
                    filename = patient + '_' + str(idx) + '__label' + '.npy'
                    np.save(arr=subvolume_label.numpy(), file=(out_label_path / filename))
                    previous_idx = idx
            selected_bad_index = random.sample(bad_index, _SELECTED_EMPTY_LABEL)
            for idx in list(selected_bad_index):
                slice_lower = idx - half_volume_size
                slice_upper = idx + subvolume_size - half_volume_size
                subvolume_img = image[:, :, :, slice_lower: slice_upper]
                filename = patient + '_' + str(idx) + '__input' + '.npy'
                np.save(arr=subvolume_img.numpy(), file=(out_img_path / filename))
                subvolume_label = label[:, :, :, slice_lower: slice_upper]
                filename = patient + '_' + str(idx) + '__label' + '.npy'
                np.save(arr=subvolume_label.numpy(), file=(out_label_path / filename))

    def calculate_volumes(self, data_path: epath.Path, output_file: str,
                          channels: List[int], column_names: List[str]):
        """Caculates the label volume on already processed data."""
        output_filename = data_path / self.phase.value / output_file
        output = []
        label_path = data_path / self.phase.value / 'labels'
        all_files = label_path.glob('*.nii.gz')
        patients = [a_file.stem.split('.')[0] for a_file in all_files]
        print(f'Find {len(patients)} patients for Phase: {self.phase.value} in {label_path}')
        for patient in patients:
            label_file = data_path / self.phase.value / 'labels' / (patient + '.nii.gz')
            label = tio.LabelMap(label_file)
            output_line = [patient]
            spacing = label.spacing
            print(f'Label shape: {label.shape}')
            unit_volume = spacing[0] * spacing[1] * spacing[2]
            print(f'Image spacing: {spacing} and unit volume: {unit_volume}')
            for channel in channels:
                volume = label.data[channel, :, :, :].sum().item() * unit_volume
                output_line.append(str(volume))
            output.append(output_line)

        df = pd.DataFrame(output, columns=column_names)
        df.to_csv(output_filename)
