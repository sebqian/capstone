"""HECKTOR 2022 preprocessor."""
import csv
from typing import List, Tuple, Union, Dict
from etils import epath
import torch
import torchio as tio

from projects.hecktor2022 import terminology as term

_ORIGINAL_DATA_FOLDER = 'original'
_SUBVOLUME_SIZE = (128, 128, 32)
_MAX_HU = 1024
_MIN_HU = -1024
# Exclude CHUS because its PETs are way longer than CTs
_VALID_HOSPITALS = ('MDA', 'HGJ', 'HMR', 'CHUM', 'CHUP', 'CHUV', 'CHB', 'USZ')


class HecktorProcessor():
    """This class is used to preprocess the data and saved them into subvolumes:
        Assume images files are in a subfolder called images, labels files are in a
        subfolder called labels; The csv files are available in the parent folder to
        list the patients belong to train, valid, test
    """

    def __init__(self, data_folder: epath.Path,
                 reference: term.Modality = term.Modality.PET,
                 subvolume_size: Tuple[int, int, int] = _SUBVOLUME_SIZE
                 ) -> None:
        self.data_path = data_folder
        self.original_data_path = data_folder / _ORIGINAL_DATA_FOLDER
        x, y, z = subvolume_size
        self.output_path = data_folder / f'subvolumesx{x}x{y}x{z}'
        # Create subfolders
        for phase in list(term.Phase):
            for data_type in ['images', 'labels']:
                subfolder = self.output_path / phase.value / data_type
                subfolder.mkdir(parents=True, exist_ok=True)
        self.reference = reference
        self.subvolume_size = subvolume_size
        self.ct_normalization = [tio.transforms.Clamp(out_min=_MIN_HU, out_max=_MAX_HU),
                                 tio.transforms.RescaleIntensity(out_min_max=(-1, 1))]
        self.pt_normalization = [tio.transforms.ZNormalization(
            masking_method=lambda x: x > 0.1)]

    def get_patient_lists(self, phase: term.Phase) -> List[str]:
        """Find the patients in a specific csv file with corresponding phase."""
        csv_files = self.original_data_path.glob(f'*_in_{phase.value}.csv')
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
        patients = [patient for patient in patients if patient.split('-')[0] in _VALID_HOSPITALS]
        print(f'Find {len(patients)} patients for Phase: {phase.value}')
        return patients

    def check_resolution(self, phase: term.Phase) -> None:
        """Print out the data size and resolution for patients."""
        patients = self.get_patient_lists(phase)
        for patient in patients:
            print(f'Patient: {patient}')
            ct_file = self.original_data_path / 'images' / (patient + '__CT.nii.gz')
            pt_file = self.original_data_path / 'images' / (patient + '__PT.nii.gz')
            ct = tio.ScalarImage(ct_file)
            pet = tio.ScalarImage(pt_file)
            print(f'\t Shape: ct - {ct.shape} and pet - {pet.shape}.')
            print(f'\t Spacing: ct - {ct.spacing} and pet - {pet.spacing}.')

    def resample_normalization(
            self, phase: term.Phase, patient: str
            ) -> Dict[str, Union[str, torch.Tensor]]:
        """Process a patient's data and convert into tensor.
            Assume all the volumes are already co-registered.
        Args:
            phase: pipeline phase
            patient: patient ID
        Returns:
            dict of {'id': patient id,
                'input': input_data,
                'label': label_data if not prediction phase},
                padding applied to make sure D is a muliptle of subvolume d.
        """
        print(f'Resample and Normalize data for {patient}')
        ct_file = self.original_data_path / 'images' / (patient + '__CT.nii.gz')
        pt_file = self.original_data_path / 'images' / (patient + '__PT.nii.gz')
        ct = tio.ScalarImage(ct_file)
        pet = tio.ScalarImage(pt_file)
        print(f'\t Shape: ct - {ct.shape} and pet - {pet.shape}.')
        print(f'\t Spacing: ct - {ct.spacing} and pet - {pet.spacing}.')
        ref_path = ''
        if self.reference == term.Modality.CT:
            ref_path = ct_file
        elif self.reference == term.Modality.PET:
            ref_path = pt_file

        ref_img = tio.ScalarImage(ref_path)
        n_slice = ref_img.shape[-1]
        image_size = (_SUBVOLUME_SIZE[0], _SUBVOLUME_SIZE[1], n_slice)
        resize_for_ref = tio.transforms.Resize(image_size)
        ref_img = resize_for_ref(ref_img)
        resample_transform = tio.transforms.Resample(target=ref_img)

        n_subvolumes = n_slice // self.subvolume_size[-1] + 1
        total_slices = n_subvolumes * self.subvolume_size[-1]
        extra_slices = total_slices - n_slice

        ct_padding = tio.transforms.Pad(padding=(0, 0, 0, 0, 0, extra_slices), padding_mode=-1000)
        pt_padding = tio.transforms.Pad(padding=(0, 0, 0, 0, 0, extra_slices), padding_mode=0)
        ct_transforms = tio.Compose([resample_transform] + self.ct_normalization + [ct_padding])
        pt_transforms = tio.Compose([resample_transform] + self.pt_normalization + [pt_padding])
        processed_ct = ct_transforms(ct)
        processed_pet = pt_transforms(pet)
        input_data = tio.Subject(
            ct=tio.ScalarImage(processed_ct),
            pet=tio.ScalarImage(processed_pet),
        )

        volumes = {'id': patient, 'input': input_data}
        if phase != term.Phase.PRED:
            label_file = self.original_data_path / 'labels' / (patient + '.nii.gz')
            label = tio.LabelMap(label_file)
            processed_label = resample_transform(label).data
            volumes['label'] = processed_label

        return volumes

    def subvolume_creation(self, volumes: Dict[str, Union[str, torch.Tensor]],
                           ) -> List[Dict[str, Union[str, torch.Tensor]]]:
        """Create subvolumes from a tensor volume
        Args:
            volumes: for input [C, H, W, D], for label [H, W, D],
                D must be a multiple of subvolume slice d
            subvolume_slice: number of slices for the subvolume
        Returns:
            a list of dict, each element is a subvolume
        """
        patient = volumes['id']
        input = torch.Tensor(volumes['input'])
        label = torch.Tensor(volumes['label'])
        d = self.subvolume_size[-1]
        assert input.shape[-1] % d == 0, f'\t Incompatible tensor shape: {input.shape}'
        n_subvolumes = input.shape[-1] // d
        subvolumes = []
        for i in range(n_subvolumes):
            start = i * d
            end = start + d
            subvolume = {'id': f'{patient}_{str(i)}',
                         'input': input[..., start:end]}
            if label is not None:
                subvolume['label'] = label[..., start:end]
            subvolumes.append(subvolume)
        return subvolumes

    def save_sub_volumes(
            self, subvolumes: List[Dict[str, Union[str, torch.Tensor]]],
            out_dir: epath.Path) -> int:
        """Save the subvolumes into files.
        Args:
            subvolumes: list of dict {'input': input_data, 'label': label_data}
            out_dir: directory to save
            file_stem: the basic file name stem
        Returns:
            1 means success
        """
        for subvolume in subvolumes:
            id = subvolume['id']
            input_tensor = subvolume['input']
            label_tensor = subvolume['label']
            file_name = out_dir / 'images' / f'{id}_input.pt'
            torch.save(input_tensor, str(file_name))
            file_name = out_dir / 'labels' / f'{id}_label.pt'
            torch.save(label_tensor, str(file_name))
        return 1

    def batch_process_patients(self, phase: term.Phase) -> int:
        """ Batch process patients into subvolumes and save
        Args:
            phase: train, valid or test for patient cohort
        Returns:
            1 is success
        """
        patients = self.get_patient_lists(phase)
        out_dir = self.output_path / phase.value
        for patient in patients:
            volumes = self.resample_normalization(phase, patient)
            print(f'Create subvolumes for {patient}')
            subvolumes = self.subvolume_creation(volumes)
            print(f'Save subvolumes to files for {patient}')
            flag = self.save_sub_volumes(subvolumes, out_dir)
            if flag != 1:
                raise ValueError(f'Fail to save subvolumes for {patient}')
        return 1
