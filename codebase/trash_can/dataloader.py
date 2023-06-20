"""HECKTOR2022 Dataloader. Tensorflow version."""
from typing import Dict, List, Tuple
from etils import epath
# import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from projects.hecktor2022 import terminology as term

_TENSOR_FILE_FORMAT = '.pt'


def read_tensors(input_folder: epath.Path, id: str) -> Dict[str, torch.Tensor]:
    """Load tensors into dictionary."""
    input_file = input_folder / 'images' / (id + '_input' + _TENSOR_FILE_FORMAT)
    label_file = input_folder / 'labels' / (id + '_label' + _TENSOR_FILE_FORMAT)
    input_data = torch.load(str(input_file))  # [C, H, W, D]
    input_data = torch.swapaxes(input_data, 1, 3)  # [C, D, W, H]
    label_data = torch.load(str(label_file))  # [C, H, W, D]
    label_data = torch.swapaxes(label_data, 1, 3)  # [C, D, W, H]
    return {'input': input_data, 'label': label_data}


def _create_transform(transform_types: List[str]) -> transforms.Compose:
    """Creates a transform for different phases."""
    transform_collection = []
    if 'affine' in transform_types:
        transform_collection.append(transforms.RandomAffine(180, 0.2))
    if 'h_flip' in transform_types:
        transform_collection.append(transforms.RandomHorizontalFlip(p=0.5))
    if 'v_flip' in transform_types:
        transform_collection.append(transforms.RandomVerticalFlip(p=0.5))
    if 'blur' in transform_types:
        transform_collection.append(transforms.GaussianBlur((0.01, 0.5)))
    transform_comp = transforms.Compose(transform_collection)
    return transform_comp


# class HECKTORDataset(tf.keras.utils.Sequence):
class HECKTORDataset(Dataset):
    """Defines dataset."""
    def __init__(
            self, data_folder: epath.Path,
            batch_size: int,
            num_samples_per_epoch: int, phase: term.Phase,
            transform_types: List[str] = []) -> None:
        self.num_samples_per_epoch = num_samples_per_epoch
        self.datafolder = data_folder / phase.value
        self.phase = phase
        self.batch_size = batch_size
        example_files = (self.datafolder / 'labels').glob('*' + _TENSOR_FILE_FORMAT)
        self.samples = [(example.stem)[:-6] for example in example_files]
        self.nsamples = len(self.samples)

        if phase == term.Phase.TRAIN:
            self.transform = _create_transform(transform_types)
        else:
            self.transform = transforms.Compose([])

        print(f'Available data size for {self.phase.value}: {self.nsamples}')

    def __getitem__(self, index_: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index_ <= self.nsamples - 1:
            case_id = self.samples[index_]
        else:
            new_index_ = index_ - (index_ // self.nsamples) * self.nsamples
            case_id = self.samples[new_index_]

        dict_images = read_tensors(self.datafolder, case_id)
        input_tensor = self.transform(dict_images['input'])
        label_tensor = self.transform(dict_images['label'])
        return input_tensor, label_tensor

    def __len__(self) -> int:
        return self.num_samples_per_epoch


def get_loader(datafolder: epath.Path,
               phase: term.Phase,
               batch_size: int = 1,
               num_samples_per_epoch: int = 1,
               transform_types: List[str] = [],
               num_works: int = 0) -> DataLoader:
    """Gets dataset."""
    dataset = HECKTORDataset(
        datafolder, batch_size,
        num_samples_per_epoch, phase=phase,
        transform_types=transform_types)

    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_works, pin_memory=False)

    return data_loader
