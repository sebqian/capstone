"""Metrics from Manai package."""
from typing import Any, Dict
from monai.metrics import meandice, generalized_dice


def get_segmentation_metrics(config: Dict[str, Any]):
    """Return a segmentation loss."""
    # Default to dice
    metric = meandice.DiceHelper(
        include_background=config['include_background'],
        sigmoid=config['sigmoid'],
        softmax=config['softmax'],
        activate=config['activate'],
        ignore_empty=config['ignore_empty'],
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
