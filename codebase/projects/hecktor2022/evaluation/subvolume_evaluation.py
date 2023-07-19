"""Evaluation Module for inference and for metrics calculation.
    It's for subvolume evaluation. Assume all the test data are already in subvolumes.
    For whole image evaluation, please use image_evaluation.py.
"""

from etils import epath
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from monai.metrics import DiceMetric, SurfaceDiceMetric
from monai.networks import utils

from codebase import terminology as term
from codebase.projects.hecktor2022 import read_config
from codebase.lightning_module import seg_model_module
from codebase.dataloader.images import data_module


class SubVolumeEvaluationModule():
    """Evaluation Module
    Results for different metrics are stored in a list of dataframes.
    """

    def __init__(self, checkpoint_path: epath.Path,
                 exp_config: epath.Path,
                 phase: term.Phase) -> None:
        if not checkpoint_path.exists():
            raise ValueError(f'State_dict is not found at {checkpoint_path}')
        if not exp_config.exists():
            raise ValueError(f'Data not found at {exp_config}')

        self.phase = phase
        self.configs = read_config.read_experiment_config(exp_config)
        self.data_path = epath.Path(self.configs['experiment']['data_path']) / self.phase.value
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = seg_model_module.SegmentationModelModule.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            hparams=self.configs,
            optimizer_class=torch.optim.AdamW
        )
        self.model.freeze()
        self.trainer = pl.Trainer()

        # create test dataset
        self.configs['test']['include'] = True
        mdata = data_module.MedicalImageDataModule(
            task_type=term.ProblemType.SEGMENTATION,
            config=self.configs,
        )
        mdata.prepare_data()
        mdata.setup()
        self.test_dataloader = mdata.test_dataloader()

        self.test_metrics = DiceMetric(
            include_background=False,
            reduction='none',
            get_not_nans=False,
            ignore_empty=True,
            num_classes=self.configs['metric']['num_classes']
        )
        # self.test_metrics = SurfaceDiceMetric(
        #     class_thresholds=[2, 2],
        #     include_background=False,
        #     reduction='none'
        # )

    def run_cohort_test(self):
        """Run through the test subvolumes with existing labels."""
        self.trainer.test(self.model, dataloaders=self.test_dataloader)
        return self.model.all_test_metrics.cpu().numpy()

    def get_all_ids(self) -> List[str]:
        # idntify all the patients in test
        data_path = epath.Path(self.configs['experiment']['data_path']) / self.phase.value
        ids = [a_file.stem.split('__')[0] for a_file in (data_path / 'images').glob('*.npy')]
        assert len(ids) > 0
        return ids

    def get_features_and_label(self, id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get file paths by ID."""
        input_file = self.data_path / 'images' / (id + '__input.npy')
        label_file = self.data_path / 'labels' / (id + '__label.npy')
        input_data = torch.Tensor(np.load(input_file)).to(self.device)
        input_data = input_data[None, ...]  # create batch dimension
        label_data = torch.Tensor(np.load(label_file)).to(self.device)
        label_data = label_data[None, ...]  # create batch dimension
        label_data = utils.one_hot(
            labels=label_data, num_classes=self.configs['metric']['num_classes'],
            dim=1)
        return input_data, label_data

    def evaluate_an_example(self, id: str, print_nonzeros: bool = False
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Calculates the metrics of a single field.
        Args:
            id: subvolume example id

        Returns:
            a single metric value

        Excpetions:
            ValueError: required files can't be found
        """
        example_input, example_label = self.get_features_and_label(id)
        # print(example_input.shape, example_label.shape)
        with torch.no_grad():
            prediction = self.model(example_input)
        # print(prediction.shape)
        result = self.test_metrics(prediction, example_label).cpu().numpy()  # type: ignore
        print(f'{id}: {result}')
        if print_nonzeros:
            label_sum = torch.sum(example_label, dim=(0, 2, 3))
            predict_sum = torch.sum(prediction, dim=(0, 2, 3))
            print(f'Nonzero label slices: {torch.nonzero(label_sum[1:, :]).cpu().numpy()}')
            print(f'Nonzero prediction slices: {torch.nonzero(predict_sum[1:, :]).cpu().numpy()}')
        return example_input, prediction, example_label, result[0]

    def comparison_plot(self, input_data: torch.Tensor, label_data: torch.Tensor,
                        prediction: torch.Tensor, channel: int, nslice: int):
        """Tensors are in [BCHWD] format."""
        vmin = 0
        vmax = 2
        fig, axes = plt.subplots(1, 4, num=1, clear=True, figsize=(12, 3))
        axes[0].imshow(input_data[0, 0, ..., nslice].cpu().numpy())
        axes[0].set_title('CT', pad=10, fontsize=12, ha='center')  # Set the title and adjust the spacing
        axes[1].imshow(input_data[0, 1, ..., nslice].cpu().numpy())
        axes[1].set_title('PET', pad=10, verticalalignment='bottom')
        axes[2].imshow(label_data[0, channel, ..., nslice].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[2].set_title(f'Label-ch{channel}', pad=10, verticalalignment='bottom')
        axes[3].imshow(prediction[0, channel, ..., nslice].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[3].set_title(f'Prediction-ch{channel}', pad=10, verticalalignment='bottom')
        plt.tight_layout()
        plt.show()
