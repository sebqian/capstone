"""Losses from Manai package."""
from typing import Any, Callable, Dict
from monai.losses import (
    DiceLoss,
    DiceCELoss,
    DiceFocalLoss,
    MaskedDiceLoss,
    GeneralizedDiceFocalLoss,
    TverskyLoss,
    FocalLoss)


def get_segmentation_loss(config: Dict[str, Any]) -> Callable:
    """Return a segmentation loss."""
    name = config['name']
    loss = DiceLoss(
        include_background=config['include_background'],
        to_onehot_y=config['to_onehot_y'],
        sigmoid=config['sigmoid'],
        softmax=config['softmax']
        )  # default
    if name == 'dicefocal':
        print('Use dice focal loss')
        loss = DiceFocalLoss(
            include_background=config['include_background'],
            to_onehot_y=config['to_onehot_y'],
            sigmoid=config['sigmoid'],
            softmax=config['softmax']
        )
    if name == 'masked_dice':
        print('Use masked dice loss')
        loss = MaskedDiceLoss(
            include_background=config['include_background'],
            sigmoid=config['sigmoid'],
            softmax=config['softmax']
        )
    if name == 'generalized_dice_focal_loss':
        print('Use generalized_dice_focal_loss')
        loss = GeneralizedDiceFocalLoss(
            include_background=config['include_background'],
            to_onehot_y=config['to_onehot_y'],
            sigmoid=config['sigmoid'],
            softmax=config['softmax']
        )
    if name == 'tversky_loss':
        print('Use tversky_loss')
        loss = TverskyLoss(
            include_background=config['include_background'],
            to_onehot_y=config['to_onehot_y'],
            sigmoid=config['sigmoid'],
            softmax=config['softmax'],
            alpha=0.3,
            beta=0.7
        )
    if name == 'focalloss':
        print('Use focal_loss')
        loss = FocalLoss(
            include_background=config['include_background'],
            to_onehot_y=config['to_onehot_y'],
            use_softmax=config['softmax']
        )
    if name == 'dice_ce':
        print('Use Dice CrossEntropy loss')
        loss = DiceCELoss(
            include_background=config['include_background'],
            to_onehot_y=config['to_onehot_y'],
            sigmoid=config['sigmoid'],
            softmax=config['softmax'],
            squared_pred=True,
            lambda_ce=0.5,
        )
    return loss
