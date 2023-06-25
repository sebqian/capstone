"""Losses from Manai package."""
from typing import Any, Dict
from monai.losses import dice, tversky, focal_loss


def get_segmentation_loss(config: Dict[str, Any]):
    """Return a segmentation loss."""
    name = config['name']
    loss = dice.DiceLoss(
        include_background=config['include_background'],
        to_onehot_y=config['to_onehot_y'],
        sigmoid=config['sigmoid'],
        softmax=config['softmax']
        )  # default
    if name == 'dicefocal':
        print('Use dice focal loss')
        loss = dice.DiceFocalLoss(
            include_background=config['include_background'],
            to_onehot_y=config['to_onehot_y'],
            sigmoid=config['sigmoid'],
            softmax=config['softmax']
        )
    if name == 'masked_dice':
        print('Use masked dice loss')
        loss = dice.MaskedDiceLoss(
            include_background=config['include_background'],
            sigmoid=config['sigmoid'],
            softmax=config['softmax']
        )
    if name == 'generalized_dice_focal_loss':
        print('Use generalized_dice_focal_loss')
        loss = dice.GeneralizedDiceFocalLoss(
            include_background=config['include_background'],
            to_onehot_y=config['to_onehot_y'],
            sigmoid=config['sigmoid'],
            softmax=config['softmax']
        )
    if name == 'tversky_loss':
        print('Use tversky_loss')
        loss = tversky.TverskyLoss(
            include_background=config['include_background'],
            to_onehot_y=config['to_onehot_y'],
            sigmoid=config['sigmoid'],
            softmax=config['softmax'],
            alpha=0.3,
            beta=0.7
        )
    if name == 'focalloss':
        print('Use focal_loss')
        loss = focal_loss.FocalLoss(
            include_background=config['include_background'],
            to_onehot_y=config['to_onehot_y'],
            use_softmax=config['softmax']
        )
    if name == 'dice_ce':
        print('Use Dice CrossEntryo loss')
        loss = dice.DiceCELoss(
            include_background=config['include_background'],
            to_onehot_y=config['to_onehot_y'],
            sigmoid=config['sigmoid'],
            softmax=config['softmax'],
        )
    return loss
