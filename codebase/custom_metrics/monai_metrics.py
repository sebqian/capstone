"""Metrics from Manai package."""
from typing import Any, Callable, Dict
from monai import metrics


def get_segmentation_metrics(config: Dict[str, Any]) -> Callable:
    """Return a segmentation loss."""
    reduction = config['reduction']
    # Default to dice
    metric = metrics.DiceMetric(
        include_background=config['include_background'],
        ignore_empty=config['ignore_empty'],
        reduction=reduction,
        get_not_nans=config['get_not_nans'],
        num_classes=config['num_classes'])  # default

    metric_name = config['name']
    if metric_name == 'generalized_dice':
        print('Use generalized dice metrics.')
        metric = metrics.GeneralizedDiceScore(
            include_background=config['include_background'],
            reduction=reduction,
            )  # default
    return metric
