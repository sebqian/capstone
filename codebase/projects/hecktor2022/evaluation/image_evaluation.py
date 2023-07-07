"""This module is for whole image evaluation. """
from typing import List, Tuple
import csv
import numpy as np
import matplotlib.pyplot as plt
from etils import epath

import torch
import torchio as tio
import pytorch_lightning as pl

from monai.inferers.utils import sliding_window_inference
from monai.metrics import meandice

from lightning_module import seg_model_module
from codebase.preprocessor.images import multi_modal_processor
from codebase import terminology as term
from codebase.projects.hecktor2022 import read_config


class ImageEvaluationModule():

    def __init__(self, checkpoint_path: epath.Path,
                 exp_config: epath.Path,
                 data_path: epath.Path,
                 phase: term.Phase,
                 subvolume_size: Tuple[int, int, int],
                 modalities: List[term.Modality],
                 reference_modality: term.Modality,
                 key_word: str) -> None:
        """
        Args:
            checkpoint_path: for checkpoint
            exp_config: config yaml file path
            data_path: for evaluation
            phase: either test or infer
            subvolume_size: [H, W, D]
            modalities: image modalities for preprocessing
            key_word: used to identify patient's ID
        """
        if not checkpoint_path.exists():
            raise ValueError(f'State_dict is not found at {checkpoint_path}')
        if not data_path.exists():
            raise ValueError(f'Data not found at {data_path}')
        if not exp_config.exists():
            raise ValueError(f'Data not found at {exp_config}')

        self.phase = phase
        self.configs = read_config.read_experiment_config(exp_config)
        self.model = seg_model_module.SegmentationModelModule.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            hparams=self.configs,
            optimizer_class=torch.optim.AdamW
        )
        self.trainer = pl.Trainer()

        self.subvolume_size = subvolume_size
        self.modalities = modalities
        self.ref_modality = reference_modality

        # idntify all the ids in the folder
        self.data_path = data_path
        self.phase = phase
        self.ids = [a_file.stem.split('__')[0] for a_file in (
            data_path / 'images').glob('*' + key_word)]
        assert len(self.ids) > 0

        self.preprocessor = multi_modal_processor.MultiModalProcessor(
            data_folder=data_path,
            phase=self.phase,
            modalities=self.modalities,
            reference=self.ref_modality,
            problem_type=term.ProblemType.SEGMENTATION
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []

        # create the normalizer for image processing
        self.normalizer = self.preprocessor.create_post_normalization()

        # for image evaluation
        self.one_hot_transform = tio.transforms.OneHot(
            num_classes=self.configs['metric']['num_classes'])
        self.test_metrics = meandice.DiceMetric(
            include_background=False,
            reduction='none',
            get_not_nans=False,
            ignore_empty=False,
            num_classes=self.configs['metric']['num_classes']
        )

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

    def create_subject(self, id: str) -> tio.Subject:
        """Creates an object without label."""
        print(f"Loading data for {id}")
        subject_dict = {'ID': id}
        for modality in self.modalities:
            img_file = self.data_path / 'images' / (id + f'__{modality.value}.nii.gz')
            img = tio.ScalarImage(img_file)
            subject_dict[modality.value] = img  # type: ignore
        label_file = self.data_path / 'labels' / (id + '.nii.gz')
        if label_file.exists():
            label = tio.LabelMap(label_file)
            subject_dict['LABEL'] = label  # type: ignore
        subject = tio.Subject(subject_dict)
        return subject

    def case_preprocessing(self,
                           subject: tio.Subject) -> Tuple[tio.Subject, int, int, np.ndarray]:
        """Individual case preprocessing."""
        print(f"Starting preprocessing for {subject['ID']}")
        # apply resampling
        print('\t Resampling images ...')
        subject = self.preprocessor.resample_to_reference(subject)
        # create body mask
        subject = self.preprocessor.create_body_mask(
            subject, thresholds=multi_modal_processor.BODY_THRESHOLD)
        # determine the slice range for prediction
        top, bottom, center = self.preprocessor.find_top_and_bottom(subject['BODY'])
        print(f"\t original image size: {subject['BODY'].spatial_shape}")
        # cropping image
        print('\t cropping images ...')
        subject = self.preprocessor.find_bounding_box_and_crop(
            subject, desired_xy_size=self.subvolume_size[0:2])
        # apply normalization
        print('\t Normalizing images ...')
        subject = self.normalizer(subject)  # type: ignore
        return subject, top, bottom, center

    def case_predict(self, id: str):
        """Predict for a single case."""
        subject = self.create_subject(id)
        # ensure the orientation
        ref_image = subject[self.ref_modality.value]
        original_orientation = ref_image.orientation
        affine = ref_image.affine
        print(f"\t Original orientation: {original_orientation}")
        subject = tio.transforms.ToCanonical()(subject)
        # record the original data shape and affine
        ref_reorient_image = subject[self.ref_modality.value]  # type: ignore
        predict_tensor = torch.zeros(ref_reorient_image.spatial_shape)  # type: ignore# type: ignore
        # preprocessing
        processed_subject, top, bottom, center = self.case_preprocessing(subject)  # type: ignore
        images = [processed_subject[modality.value].data[0] for modality in self.modalities]
        image_tensor = torch.stack(images, dim=0)  # add channel
        image_tensor = image_tensor[None, ...]  # add batch
        one_hot_prediction = sliding_window_inference(
            image_tensor.to(self.device), roi_size=list(self.subvolume_size), sw_batch_size=4, predictor=self.model,
            progress=True)
        roi_prediction = torch.argmax(one_hot_prediction, dim=1).squeeze(0)  # type: ignore
        xstart = center[0] - self.subvolume_size[0] // 2
        xstop = center[0] + self.subvolume_size[0] // 2
        ystart = center[1] - self.subvolume_size[1] // 2
        ystop = center[1] + self.subvolume_size[1] // 2
        predict_tensor[xstart:xstop, ystart:ystop, top:bottom] = roi_prediction
        # orientation correction
        dims = []
        if original_orientation[0] == 'L':
            dims.append(0)
        if original_orientation[1] == 'P':
            dims.append(1)
        if original_orientation[2] == 'I':
            dims.append(2)
        if len(dims) > 0:
            predict_tensor = torch.flip(predict_tensor, dims=tuple(dims))
        final_prediction = tio.LabelMap(tensor=predict_tensor[None, ...], affine=affine)
        subject['PREDICT'] = final_prediction  # type: ignore
        return subject

    def cohort_predict(self, ids: List[str]):
        """Predict and save for a cohort."""
        output_folder = self.data_path / 'preds'
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)
        for id in ids:
            subject = self.case_predict(id)
            filename = output_folder / (id + '__predict.nii.gz')
            subject['PREDICT'].save(filename, squeeze=True)  # type: ignore

    def calculate_dice(self, ids: List[str]) -> np.ndarray:
        """Evaluates dice between the prediction and label for the whole image of an id."""
        # load prediction and label
        predictions = []
        labels = []
        for id in ids:
            subject = self.get_prediction_label_pair(id)
            predictions.append(self.one_hot_transform(subject['PREDICT']).tensor)  # type: ignore
            labels.append(self.one_hot_transform(subject['LABEL']).tensor)  # type: ignore

        # metrics input should be a list of channel-first tensors
        dice = self.test_metrics(predictions, labels).cpu().numpy()  # type: ignore
        print(f'Average dice {dice[0]}')
        # print(f'{id}: GTVp dice {dice[0]}; GTVn dice {dice[1]}')
        return dice[0]

    def get_prediction_label_pair(self, id: str, print_nonzeros: bool = False
                                  ) -> tio.Subject:
        pred_file = self.data_path / 'preds' / (id + '__predict.nii.gz')
        prediction = tio.LabelMap(pred_file)
        label_file = self.data_path / 'labels' / (id + '.nii.gz')
        label = tio.LabelMap(label_file)
        if print_nonzeros:
            label_sum = torch.sum(label.data, dim=(0, 1, 2))
            print(f'Nonzero label slices: {torch.nonzero(label_sum).flatten().cpu().numpy()}')
            predict_sum = torch.sum(prediction.data, dim=(0, 1, 2))
            print(f'Nonzero prediction slices: {torch.nonzero(predict_sum).flatten().cpu().numpy()}')
        img_file = self.data_path / 'images' / (id + '__CT.nii.gz')
        ct = tio.ScalarImage(img_file)
        img_file = self.data_path / 'images' / (id + '__PT.nii.gz')
        pet = tio.ScalarImage(img_file)
        resize_for_ref = tio.transforms.Resize(ct.spatial_shape)
        pet = resize_for_ref(pet)
        subject = tio.Subject({'LABEL': label, 'PREDICT': prediction,
                               'CT': ct, 'PT': pet})
        return subject

    def comparison_plot(self, subject: tio.Subject, nslice: int):
        """The images should be in [BHWD] format."""
        vmin = 0
        vmax = 2
        fig, axes = plt.subplots(1, 4, num=1, clear=True, figsize=(12, 3))
        ct = subject['CT'].data
        axes[0].imshow(ct[0, ..., nslice].cpu().numpy())
        axes[0].set_title('CT', pad=10, fontsize=12, ha='center')  # Set the title and adjust the spacing
        pet = subject['PT'].data  # original data does not have the same size as CT
        axes[1].imshow(pet[0, ..., nslice].cpu().numpy())
        axes[1].set_title('PET', pad=10, verticalalignment='bottom')
        label_data = subject['LABEL'].data
        axes[2].imshow(label_data[0, ..., nslice].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[2].set_title('Label', pad=10, verticalalignment='bottom')
        prediction = subject['PREDICTION'].data
        axes[3].imshow(prediction[0, ..., nslice].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[3].set_title('Prediction', pad=10, verticalalignment='bottom')
        plt.tight_layout()
        plt.show()
