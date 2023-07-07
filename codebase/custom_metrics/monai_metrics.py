"""Metrics from Manai package."""
from typing import Any, Callable, Dict
from monai.metrics import meandice, generalized_dice


def get_segmentation_metrics(config: Dict[str, Any],
                             is_test: bool = False) -> Callable:
    """Return a segmentation loss."""
    reduction = config['reduction']
    sigmoid = config['sigmoid']
    softmax = config['softmax']
    activate = config['activate']
    if is_test:
        reduction = 'none'
        softmax = False
        sigmoid = False
        activate = False
    # Default to dice
    metric = meandice.DiceHelper(
        include_background=config['include_background'],
        sigmoid=sigmoid,
        softmax=softmax,
        activate=activate,
        ignore_empty=config['ignore_empty'],
        reduction=reduction,
        get_not_nans=config['get_not_nans'],
        num_classes=config['num_classes'])  # default

    metric_name = config['name']
    if metric_name == 'generalized_dice':
        print('Use generalized dice metrics.')
        metric = generalized_dice.GeneralizedDiceScore(
            include_background=config['include_background'],
            reduction=reduction,
            )  # default
    return metric
