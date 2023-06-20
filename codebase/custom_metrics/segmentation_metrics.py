"""Custom Segmenation Metrics. """

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SegmentationMetrics(ABC, nn.Module):
    def __init__(self):
        super(SegmentationMetrics, self).__init__()

    @abstractmethod
    def forward(self, predictions, targets):
        pass


class DiceCoefficient(SegmentationMetrics):
    def __init__(self, smooth=1e-5):
        super(DiceCoefficient, self).__init__()
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

        # batch averaged mean dice, but not classes averaged.
        return torch.mean(dice_scores, dim=0)
