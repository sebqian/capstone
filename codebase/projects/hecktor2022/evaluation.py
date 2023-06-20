"""Evaluation Module for inference and for metrics calculation"""

from etils import epath
from typing import Tuple
import numpy as np
import torch

from codebase.custom_metrics import monai_metrics
_METRICS_CONFIG = {'name': 'dice', 'include_background': False, 'reduction': 'none',
                   'get_not_nans': True, 'num_classes': 2}


class EvaluationModule():
    """Evaluation Module
    Results for different metrics are stored in a list of dataframes.
    """

    def __init__(self, checkpoint_path: epath.Path,
                 data_path: epath.Path) -> None:
        if not checkpoint_path.exists():
            raise ValueError(f'State_dict is not found at {checkpoint_path}')
        if not data_path.exists():
            raise ValueError(f'Data not found at {data_path}')

        checkpoint = torch.load(str(checkpoint_path))
        self.model = checkpoint['model']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # idntify all the patients in test
        self.data_path = data_path
        self.ids = [a_file.stem.split('__')[0] for a_file in (data_path / 'labels').glob('*.npy')]
        assert len(self.ids) > 0

        self.metrics = monai_metrics.get_segmentation_metrics(_METRICS_CONFIG)
        self.results = []

    def _get_features_and_label(self, id: str) -> Tuple[epath.Path, epath.Path]:
        """Get file paths by ID."""
        input_file = self.data_path / 'images' / (id + '__input.npy')
        label_file = self.data_path / 'labels' / (id + '__label.npy')
        return input_file, label_file

    def _prepare_input(self, filename: epath.Path) -> torch.Tensor:
        """Convert input files into input tensor."""
        image = torch.Tensor(np.load(filename)).to(self.device)
        image = image[None, ...]  # create batch dimension
        return image

    def _prepare_label(self, filename: epath.Path) -> torch.Tensor:
        """Convert label file into label tensor."""
        label = torch.Tensor(np.load(filename))
        label = torch.sum(label[1:3, ...], dim=0)
        label = label[None, None, ...].to(self.device)
        return label

    def evaluate_an_example(self, id: str) -> float:
        """Calculates the metrics of a single field.
        Args:
            id: example id

        Returns:
            a single metric value

        Excpetions:
            ValueError: required files can't be found
        """
        example_input, example_label = self._get_features_and_label(id)
        features = self._prepare_input(example_input)
        label = self._prepare_label(example_label)
        output = self.model(features)
        prediction = torch.sigmoid(output)
        prediction = (prediction >= 0.5).int()
        result = self.metrics([prediction], [label]).item()
        print(f'{id}: {result}')
        return result

    def evaluate_cohort(self) -> np.ndarray:
        print(f'Find {len(self.ids)} examples.')
        for id in self.ids:
            self.results.append(self.evaluate_an_example(id))
        results = np.array(self.results)
        print(f'Mean evaluation score: {np.nanmean(results)}')
        return results
