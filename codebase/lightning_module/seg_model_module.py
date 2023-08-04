"""Pytorch lightning model module."""
from typing import Any, Callable, Dict, Tuple
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl

from codebase.models import monai_models
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data.utils import decollate_batch
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
        self.spatial_size = tuple(self.hparams['model']['spatial_size'])
        self.net = monai_models.get_model(
            model_name=self.hparams['model']['name'],
            img_size=self.spatial_size,
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
        self.best_val_metric = 0
        self.best_val_epoch = 0
        self.metric_values = []
        self.epoch_loss_values = []
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.pred_onehot = AsDiscrete(argmax=True,
                                      to_onehot=self.hparams['model']['output_channel'])
        self.label_onehot = AsDiscrete(to_onehot=self.hparams['model']['output_channel'])

    def logits_to_onehot(self, x: torch.Tensor):
        """Converts a tensor [BCHWD] into onehot format."""
        probabilities = F.softmax(x, dim=1)
        x = torch.argmax(probabilities, dim=1)
        print(x.shape)
        print(type(x))
        one_hot_tensor = F.one_hot(x[:, None, ...],
                                   num_classes=x.shape[1])
        print('there')
        one_hot_tensor = torch.swapaxes(one_hot_tensor, 1, -1).squeeze(-1)
        return one_hot_tensor.float()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr,
                                         weight_decay=self.hparams['train']['weight_decay'])
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        #                                            factor=0.5, patience=2)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams['train']['epochs'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  # Select the metric to monitor for scheduling
        }
        # return optimizer

    def prepare_batch(self,
                      batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepares batch data."""
        images = batch['input']
        return images, batch['label']

    def forward(self, x: torch.Tensor):
        """Forward used only for inference.
            It returns prediction, not just the logits.
        """
        logits = self.net(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)  # y is in one_hot form
        logits = self.net(x)
        loss = self.criterion(logits, y)
        logs = {"train_loss": loss}
        self.train_step_outputs.append(logs)
        self.log_dict(logs, prog_bar=True, batch_size=self.hparams['train']['batch_size'])
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["train_loss"] for x in self.train_step_outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())
        self.train_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        x, y = self.prepare_batch(batch)
        sw_batch_size = 4
        logits = sliding_window_inference(x, self.spatial_size, sw_batch_size, self.forward)
        loss = self.criterion(logits, y)
        # must convert MetaTensor to Tensor here otherwise wouldn't work. Don't know why.
        outputs = [self.pred_onehot(i) for i in decollate_batch(logits.as_tensor())]  # type: ignore
        labels = [self.label_onehot(i) for i in decollate_batch(y.as_tensor())]  # type: ignore
        self.metric(outputs, labels)
        values = {'val_num': len(outputs), 'val_loss': loss}
        self.validation_step_outputs.append(values)
        self.log_dict({'val_loss': loss}, prog_bar=True, batch_size=self.hparams['valid']['batch_size'])
        return values

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_num"]
        mean_val_metric = self.metric.aggregate().item()
        self.metric.reset()
        # mean_val_loss = torch.tensor(val_loss / num_items)
        logs = {
            "val_mean_dice": mean_val_metric,
        }
        if mean_val_metric > self.best_val_metric:
            self.best_val_metric = mean_val_metric
            self.best_val_epoch = self.current_epoch
        self.metric_values.append(mean_val_metric)
        self.validation_step_outputs.clear()  # free memory
        self.log_dict(logs, prog_bar=True)
        return logs

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
