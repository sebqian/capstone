"""Segmentation Loss module."""

import torch
import torch.nn as nn
from abc import abstractmethod


class SegmentationLoss(nn.Module):
    """Base class for segmentation losses."""
    def __init__(self):
        super(SegmentationLoss, self).__init__()

    @abstractmethod
    def forward(self, predictions, targets):
        """Computes the loss."""
        pass


class DiceLoss(SegmentationLoss):
    """Dice loss for 3D segmentation."""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # original dimension: [B, C, H, W, D]
        # Reshape predictions and targets to (batch_size, num_classes, -1)
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        # Calculate intersection and union
        intersection = torch.sum(predictions * targets, dim=2)
        union = torch.sum(predictions, dim=2) + torch.sum(targets, dim=2)

        # Calculate Dice scores
        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Calculate Dice loss
        dice_loss = 1.0 - dice_scores

        # Average the loss across batch, classes, and spatial dimensions
        loss = torch.mean(dice_loss)

        return loss


class CrossEntropyLoss(SegmentationLoss):
    """Cross entropy loss for 3D segmentation."""
    def __init__(self, weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean')

    def forward(self, predictions, targets):
        # Reshape predictions and targets to (batch_size, num_classes, -1)
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        # Calculate the cross-entropy loss
        loss = self.criterion(predictions, targets)

        return loss
