"""Metrics from Manai package."""
from typing import Any, Dict
from monai.metrics import meandice, generalized_dice


def get_segmentation_metrics(config: Dict[str, Any]):
    """Return a segmentation loss."""
    metric = meandice.DiceMetric(
        include_background=config['include_background'],
        reduction=config['reduction'],
        get_not_nans=config['get_not_nans'],
        num_classes=config['num_classes'])  # default

    metric_name = config['name']
    if metric_name == 'generalized_dice':
        print('Use generalized dice metrics.')
        metric = generalized_dice.GeneralizedDiceScore(
            include_background=config['include_background'],
            reduction=config['reduction'],
            )  # default
    return metric
