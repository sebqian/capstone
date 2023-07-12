"""Pytorch lightning model module."""
from typing import Any, Callable, Dict, Tuple
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl

from codebase.models import monai_models
from codebase.custom_losses import monai_losses
from codebase.custom_metrics import monai_metrics


def _count_ones_and_zeros(tag: str, x: torch.Tensor):
    """Counts zeros and ones in a tensor"""
    # Count total elements
    x = x[:, 1:, ...]  # skip background
    total_elements = x.numel()

    # Count zeros and ones
    zeros = torch.sum(x == 0, dim=(2, 3, 4))
    ones = torch.sum(x == 1, dim=(2, 3, 4))
    print(f'{tag}: {zeros} zeros and {ones} ones among total {total_elements}')


class SegmentationModelModule(pl.LightningModule):

    def __init__(self, hparams: Dict[str, Any], optimizer_class: Callable):
        super().__init__()
        self.save_hyperparameters(hparams)
        # self.lr needs to be defined if autotune lr
        self.lr = self.hparams['train']['lr']
        self.net = monai_models.get_model(
            model_name=self.hparams['model']['name'],
            output_type=self.hparams['model']['output_type'],
            spatial_dims=self.hparams['model']['spatial_dim'],
            in_channels=self.hparams['model']['input_channel'],
            out_channels=self.hparams['model']['output_channel'],
            dropout_prob=self.hparams['model']['dropout'],
            network_config=self.hparams['model']['architecture']
        )
        self.criterion = monai_losses.get_segmentation_loss(self.hparams['loss'])
        self.metric = monai_metrics.get_segmentation_metrics(self.hparams['metric'])
        self.optimizer_class = optimizer_class
        self.test_step_outputs = []

    # def _one_hot(self, x: torch.Tensor) -> torch.Tensor:
    #     """One hot conversion function.
    #         monai's one hot reduces the number of nonzeros significantly.
    #         I don't know why.
    #     """
    #     x = F.one_hot(x.long(), num_classes=self.hparams['metric']['num_classes'])
    #     x = torch.swapaxes(x, 1, -1).squeeze(-1)
    #     return x

    def logits_to_onehot(self, x: torch.Tensor):
        """Converts a tensor [BCHWD] into onehot format."""
        probabilities = F.softmax(x, dim=1)
        index_tensor = torch.argmax(probabilities, dim=1)
        one_hot_tensor = F.one_hot(index_tensor[:, None, ...],
                                   num_classes=probabilities.shape[1])
        one_hot_tensor = torch.swapaxes(one_hot_tensor, 1, -1).squeeze(-1)
        return one_hot_tensor.float()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        #                                            factor=0.1, patience=2, verbose=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams['train']['epochs'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # Select the metric to monitor for scheduling
        }
        # return optimizer

    def prepare_batch(self,
                      batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = batch['input']
        images[:, 1, ...] = images[:, 1, ...] / 5.0  # TODO: move normalization into preprocessing
        return images, batch['label']

    def forward(self, x: torch.Tensor):
        """Forward used only for inference.
            It returns prediction, not just the logits.
        """
        logits = self.net(x)
        return self.logits_to_onehot(logits)

    def training_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)  # y is in one_hot form
        logits = self.net(x)
        loss = self.criterion(logits, y)
        one_hot_pred = self.logits_to_onehot(logits)
        metric_values = self.metric(one_hot_pred, y)
        # metric_value = metric_values.nanmean()
        values = {'train_loss': loss, 'train_metrics': metric_values.nanmean()}
        self.log_dict(values, prog_bar=True, batch_size=self.hparams['train']['batch_size'])
        return loss

    def on_train_epoch_end(self):
        # aggregated_metric = self.metric.aggregate().nanmean().item()
        # self.log(name='train_epoch_metric', value=aggregated_metric)
        self.metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        logits = self.net(x)
        loss = self.criterion(logits, y)
        one_hot_pred = self.logits_to_onehot(logits)
        metric_values = self.metric(one_hot_pred, y)
        values = {'val_loss': loss, 'valid_metrics': metric_values.nanmean()}
        self.log_dict(values, prog_bar=True, batch_size=self.hparams['valid']['batch_size'])
        return loss

    def on_validation_epoch_end(self):
        # aggregated_metric = self.metric.aggregate().nanmean().item()
        # self.log(name='valid_epoch_metric', value=aggregated_metric)
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        prediction = self.forward(x)  # get one_hot prediction
        metric_values = self.metric(prediction, y)
        values = {'test_metrics': metric_values.nanmean()}
        self.test_step_outputs.append(metric_values)
        self.log_dict(values, prog_bar=True, batch_size=self.hparams['test']['batch_size'])

    def on_test_epoch_end(self):
        """Gets metrics for each example."""
        self.all_test_metrics = torch.cat(self.test_step_outputs, dim=0)
        self.test_step_outputs.clear()

    def predict_step(self, x: torch.Tensor, batch_idx: int) -> Any:
        y_hat = self.forward(x)
        return y_hat
