"""Multi-Modal image preprocessor.
This module utilizes MONAI data for data preprocessing. The transforms for augmentation is not included here.
"""
import csv
import random
from typing import Dict, List, Tuple
from etils import epath
import numpy as np
import pandas as pd
from scipy import ndimage
import torch
import torchio as tio

import codebase.terminology as term

_MAX_HU = 1024
_MIN_HU = -1024
_MIN_SUV_RATIO = 0.05
_BOX_RANGE = 310.0  # in mm
BODY_THRESHOLD = {term.Modality.CT: (-300, 1000), term.Modality.PET: (0.5, 9999)}
IMG_XYSIZE_BEFORE_CROPPING = (512, 512)
_MIN_PIXELS_BRAIN_SLICE = 10000  # minimum # of pixels when we call it's start of brain


def _pet_remove_background(x: torch.Tensor) -> torch.Tensor:
    """Mask the patient body in PET image."""
    print(x.max())
    print(torch.quantile(x, 0.9).item())
    print(torch.quantile(x, 0.7).item())
    max_value = torch.quantile(x, 0.9).item()
    threshold = max_value * 0.05
    print(f'Threshold: {threshold}')
    return x > threshold


def _pet_normalize_to_brain(x: torch.Tensor) -> torch.Tensor:
    """Normalize PET images by the brain uptake.
    The image must be cropped again, with the last slice already in brain.
    In other words, this method onlys applies to the image in the bounding box.
    """
    # assume image in canoical direction, last 3 slices are brain.
    # use PET threshold 0.5
    selected_elements = x[..., -3:-1][x[..., -3:-1] > 0.5]
    mean_value = torch.mean(selected_elements)
    # assert mean_value > 1, f'Mean value of brain SUV is too small: {mean_value}'
    print(f'\t\t Brain mean SUV estimation {mean_value}')
    x /= mean_value  # normalization
    clamped_x = torch.where(x < _MIN_SUV_RATIO, torch.tensor(0.0), x)
    return clamped_x


def _find_subvolume_index(
        label: torch.Tensor,
        valid_channel: List[int],
        border: int,
        index_interval: int) -> Tuple[List[int], List[int]]:
    """Find the index can be used for subvolumes.
    Args:
        label: [C, H, W, D]
        valid_channel: the list of channel numbers to be considered in label
        border: the number of slices at ends not considered for indexing
        index_interval: not consider number of slices next to selected ones
    Returns:
        good_index for nonzero volume, bad_index for all zero volume
            borders are completely excluded.
    """
    slice_idx = range(border, label.shape[-1]-border)
    good_index = []
    bad_index = []
    previous_good = 0
    previous_bad = 0
    for idx in slice_idx:
        a_slice = label[valid_channel, :, :, idx]
        if a_slice.sum() > 0:
            if idx > previous_good + index_interval:
                good_index.append(idx)
                previous_good = idx
        else:
            if idx > previous_bad + index_interval:
                bad_index.append(idx)
                previous_bad = idx
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
        patients = [item for sublist in patients for item in sublist]
        print(f'Find {len(patients)} patients for Phase: {self.phase.value}')
        return patients

    def create_subject(self, patient: str) -> tio.Subject:
        """Create raw subject for a patient."""
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
            # if _N_LABLES > 1:
            #     onehot_transform = tio.transforms.OneHot(num_classes=_N_LABLES)
            #     label = onehot_transform(label)
        subject_dict['LABEL'] = label  # type: ignore
        subject = tio.Subject(subject_dict)
        return subject

    def create_subject_list(self) -> List[tio.Subject]:
        """Create raw subject list for all patients with all transforms."""
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

    def resample_to_reference(self, subject: tio.Subject,
                              xy_size: Tuple[int, int] = IMG_XYSIZE_BEFORE_CROPPING) -> tio.Subject:
        """Assume all the volumes are already co-registered."""
        ref_img = subject[self.reference.value]
        image_size = ref_img.spatial_shape
        resize_for_ref = tio.transforms.Resize((xy_size[0], xy_size[1], image_size[-1]))
        ref_img = resize_for_ref(ref_img)
        resample_transform = tio.transforms.Resample(target=ref_img)  # type: ignore
        return tio.Subject(resample_transform(subject))

    def create_body_mask(self, subject: tio.Subject,
                         thresholds: Dict[term.Modality, Tuple[float, float]]) -> tio.Subject:
        """Create the body contour from images.
        All the image in the subject must have the same spacing and shape.
        """
        subject.check_consistent_attribute('spatial_shape')
        subject.check_consistent_attribute('spacing')
        img = subject[self.modalities[0].value]  # just get an image
        img_shape = img.shape
        af = img.affine
        body_mask_tensor = torch.ones(img_shape)
        for modality, thres in thresholds.items():
            img = subject[modality.value].tensor
            mask1 = img > thres[0]
            mask2 = img < thres[1]
            mask = torch.logical_and(mask1, mask2)
            body_mask_tensor = torch.logical_and(body_mask_tensor, mask)
        subject['BODY'] = tio.LabelMap(tensor=body_mask_tensor, affine=af)
        return subject

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
        print(img_files)
        landmarks = tio.HistogramStandardization.train(
            img_files, masking_function=_pet_remove_background, output_path=output_file)  # type: ignore
        return landmarks

    def create_landmark_dict(self) -> Dict[str, np.ndarray]:
        """Create landmarks for all modalities except for CT."""
        landmarks_dict = {}
        for modality in self.modalities:
            if modality != term.Modality.CT:
                landmarks = self.train_histogram_standardization(modality)
                landmarks_dict[modality.value] = landmarks
        return landmarks_dict

    def create_prior_normalization(self) -> tio.transforms.Transform:
        """Create transformation for data preprocessing before resampling.
        Currently implementation only include histogram standardization, which
            may work the best for MRI images.
        """
        transform_collection = []
        # First image normalization for all phases
        landmarks_dict = self.create_landmark_dict()
        exclude_types = [term.Modality.CT.value, 'LABEL', 'WEIGHT']
        transform_collection.append(tio.transforms.HistogramStandardization(
            landmarks_dict, exclude=exclude_types))  # type: ignore

        transform_comp = tio.transforms.Compose(transform_collection)
        return transform_comp

    def create_post_normalization(self) -> tio.transforms.Transform:
        """Create transformation for data preprocessing after resampling."""
        transform_collection = []
        # CT only
        transform_collection.append(
            tio.transforms.Clamp(out_min=_MIN_HU, out_max=_MAX_HU,
                                 include=[term.Modality.CT.value]),
        )
        transform_collection.append(
            tio.transforms.RescaleIntensity(out_min_max=(0, 1),
                                            in_min_max=(_MIN_HU, _MAX_HU),
                                            include=[term.Modality.CT.value]),
        )
        # PET only
        transform_collection.append(
            tio.transforms.Lambda(function=_pet_normalize_to_brain,
                                  include=[term.Modality.PET.value])
        )
        transform_collection.append(tio.transforms.ZNormalization(
             masking_method=tio.ZNormalization.mean,
             include=[term.Modality.PET.value]))
        sigmoid_transform = tio.transforms.Lambda(
            function=lambda x: torch.sigmoid(x),
            include=[term.Modality.PET.value])
        transform_collection.append(sigmoid_transform)

        # exclude_types = [term.Modality.CT.value, term.Modality.PET.value, 'LABEL', 'BODY', 'WEIGHT']
        # transform_collection.append(tio.transforms.ZNormalization(
        #     masking_method=tio.ZNormalization.mean, exclude=exclude_types))
        transform_comp = tio.transforms.Compose(transform_collection)
        return transform_comp

    def find_top_and_bottom(self,
                            img: tio.ScalarImage) -> Tuple[int, int, np.ndarray]:
        """Determine the slice range which contains the ROIs."""
        img_shape = np.asarray(img.spatial_shape)
        voxel_size = img.spacing
        n_slices = img_shape[-1]  # assume the last dimension is slice
        bottom = n_slices - 1
        center = np.array([0, 0]).astype(int)
        for i in range(n_slices, 0, -1):
            # TODO: better brain localization
            # now just hope brain is in the middle half the image
            box_min = img_shape[0:2] // 4
            box_max = img_shape[0:2] - box_min
            a_slice = img.data[0, ..., i-1].numpy()
            a_slice_mask = np.zeros_like(a_slice)
            a_slice_mask[box_min[0]:box_max[0], box_min[1]:box_max[1]] = 1
            a_slice = np.multiply(a_slice, a_slice_mask)
            assert a_slice.ndim == 2
            # Ideally should use brain contour here. But now it's a single threshold for simplicity
            if a_slice.sum() > _MIN_PIXELS_BRAIN_SLICE:
                bottom = i - 1
                center = np.asarray(ndimage.measurements.center_of_mass(a_slice)).astype(int)
                # center = np.array([img_shape[]])
                break
        top = max(0, int(bottom - _BOX_RANGE / voxel_size[-1]))
        return top, bottom, center

    def find_bounding_box_and_crop(self, subject: tio.Subject,
                                   desired_xy_size: Tuple[int, int]) -> tio.Subject:
        """Determine the bounding box and crop it from the images.
        All the image in the subject must have the same spacing and shape. The BODY mask
            should be ready.
        """
        subject = tio.transforms.ToCanonical()(subject)  # type: ignore
        # now the slices are from inferior to superior
        xy_size = np.asarray(desired_xy_size)
        id = subject['ID']
        img = subject['BODY']
        img_shape = np.asarray(img.spatial_shape)
        voxel_size = img.spacing
        half_xy = (xy_size / 2).astype(int)
        top, bottom, center = self.find_top_and_bottom(img=img)
        assert all(center >= half_xy), f'{center} of {id} is too close to edge.'
        assert all(np.asarray(img_shape[0:2]) - center >= half_xy), f'{center} of {id} is too close to edge.'
        # make sure center is sufficiently away from edges
        # center = center + np.abs(np.minimum(center - half_xy, 0))
        # center = center - np.abs(np.minimum(np.asarray(img_shape[1:3]) - center - half_xy, 0))
        # if not (all(center >= half_xy) and all(np.asarray(img_shape[0:2]) - center >= half_xy)):
        #     print(f'{center} of {id} is too close to edge. Reset xy center')
        #     center = (img_shape[0:2] / 2).astype(int)
        cropping = (center[0] - half_xy[0], img_shape[0] - center[0] - half_xy[0],
                    center[1] - half_xy[1], img_shape[1] - center[1] - half_xy[1],
                    top, img_shape[-1] - bottom)
        print(f'\t Head top center: {center}, slice thickness {voxel_size[-1]}, '
              f'starting at slice {top}, ending at slice {bottom}')
        crop_transform = tio.transforms.Crop(cropping=cropping)
        new_subject = crop_transform(subject)
        return new_subject  # type: ignore

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
        normalizator = self.create_post_normalization()
        # read raw data
        print('Read raw data ...')
        subjects = self.create_subject_list()
        # process subject one by one:
        n = 0
        for subject in subjects:
            # The order of processing matters, do not swap without firm understanding
            patient_id = subject['ID']
            print(f'Starting preprocessing for {patient_id}')
            # apply resampling
            print('\t Resampling images ...')
            subject = self.resample_to_reference(subject)
            # create body mask
            subject = self.create_body_mask(subject, thresholds=BODY_THRESHOLD)
            # cropping image
            print('\t cropping images ...')
            subject = self.find_bounding_box_and_crop(subject, desired_xy_size=xy_size)
            # apply normalization
            print('\t Normalizing images ...')
            subject = normalizator(subject)
            # save data
            print('\t Saving data ...')
            for modality in self.modalities:
                filename = output_folder / 'images' / f'{patient_id}__{modality.value}.nii.gz'
                subject[modality.value].save(filename)  # type: ignore
            # filename = output_folder / 'images' / f'{patient_id}__WEIGHT.nii.gz'
            # subject['WEIGHT'].save(filename)
            filename = output_folder / 'labels' / f'{patient_id}.nii.gz'
            subject['LABEL'].save(filename)  # type: ignore
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
            max_slice = label.shape[-1]
            half_volume_size = int(subvolume_size / 2)
            good_index, bad_index = _find_subvolume_index(
                label, valid_channel, half_volume_size, subvolume_intervel)
            n_label_volumes = 0
            for idx in good_index:
                if idx + subvolume_size < max_slice:
                    slice_lower = idx - random.randrange(half_volume_size)
                else:
                    slice_lower = idx - half_volume_size
                slice_upper = slice_lower + subvolume_size
                subvolume_img = image[:, :, :, slice_lower: slice_upper]
                filename = patient + '_' + str(idx) + '__input' + '.npy'
                np.save(arr=subvolume_img.numpy(), file=(out_img_path / filename))
                subvolume_label = label[:, :, :, slice_lower: slice_upper]
                filename = patient + '_' + str(idx) + '__label' + '.npy'
                np.save(arr=subvolume_label.numpy(), file=(out_label_path / filename))
                n_label_volumes += 1
            n_nolabel_volumes = min(len(bad_index), n_label_volumes)
            selected_bad_index = random.sample(bad_index, n_nolabel_volumes)
            for idx in list(selected_bad_index):
                slice_lower = idx - half_volume_size
                slice_upper = slice_lower + subvolume_size
                subvolume_img = image[:, :, :, slice_lower: slice_upper]
                filename = patient + '_' + str(idx) + '__input' + '.npy'
                np.save(arr=subvolume_img.numpy(), file=(out_img_path / filename))
                subvolume_label = label[:, :, :, slice_lower: slice_upper]
                filename = patient + '_' + str(idx) + '__label' + '.npy'
                np.save(arr=subvolume_label.numpy(), file=(out_label_path / filename))
            print(f'\t generate {n_label_volumes} labelled volumes and {n_nolabel_volumes} not labelled volumes.')

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
