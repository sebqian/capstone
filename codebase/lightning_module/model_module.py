"""Pytorch lightning model module."""
from typing import Tuple
import torch
from torch.nn.modules import loss
import pytorch_lightning as pl
from monai.networks.utils import one_hot


class SegmentationModelModule(pl.LightningModule):
    def __init__(self, net: torch.nn.Module, criterion: loss._Loss,
                 batch_size: Tuple[int, int],
                 num_classes: int,
                 metrics, optimizer_class,
                 learning_rate: float):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.metrics = metrics
        self.num_classes = num_classes
        self.optimizer_class = optimizer_class
        self.train_batch_size = batch_size[0]
        self.valid_batch_size = batch_size[1]

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        return batch['input'], batch['label']

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        label = one_hot(y, self.num_classes)
        metric_values = self.metrics(y_hat, label)
        # metric_value = metric_values.nanmean()
        values = {'train_loss': loss, 'train_metrics': metric_values}
        self.log_dict(values, prog_bar=True, batch_size=self.train_batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        label = one_hot(y, self.num_classes)
        metric_values = self.metrics(y_hat, label)
        values = {'val_loss': loss, 'valid_metrics': metric_values}
        self.log_dict(values, prog_bar=True, batch_size=self.valid_batch_size)
        return loss
