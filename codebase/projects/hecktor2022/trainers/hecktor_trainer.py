"""HECKTOR Trainer Module."""
from typing import Any, Dict, Optional, Tuple, Sequence
import numpy as np

from etils import epath
import torch
# from torch.utils.data import DataLoader
from monai.data import DataLoader
import torchio as tio
import yaml
import torchsummary

import codebase.terminology as term
from codebase.projects.hecktor2022 import dataloader
from codebase.projects.hecktor2022.models import monai_models
from codebase.custom_losses import segmentation_loss, monai_losses
from codebase.custom_metrics import segmentation_metrics, monai_metrics

_PRINT_MODEL_SUMMARY = True
_PROBLEM_TYPE = term.ProblemType.SEGMENTATION
_NUM_CLASSES = 2            # Number of classes. (= number of output channel)
input_channels = 2          # Number of input channel
_SEGRESNET_CONFIG = {'init_filters': 32, 'use_conv_final': True, 'blocks_down': (1, 2, 2, 4), 'blocks_up': (1, 1, 1)}
_UNET_CONFIG = {'channels': (16, 32, 64, 128, 256), 'strides': (2, 2, 2, 2), 'num_res_units': 2}


def _read_experiment_config(config_file: str) -> Dict[str, Any]:
    """Read experiment configurations."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        return config


class Trainer(object):
    def __init__(self, experiment_config: str):

        np.set_printoptions(precision=3, suppress=True)
        self.config = _read_experiment_config(experiment_config)
        self.datafolder = epath.Path(self.config['experiment']['data_path'])
        self.data_type = self.config['experiment']['data_type']
        self.experiment_name = self.config['experiment']['name']
        self.runsfolder = self.datafolder / 'experiments' / self.experiment_name
        if not self.runsfolder.exists():
            self.runsfolder.mkdir(parents=True, exist_ok=True)

        self.best_pred = 0.0

        spatial_size = (self.config['model']['x'],
                        self.config['model']['y'],
                        self.config['model']['z'])
        input_size = (self.config['model']['input_channel'],
                      self.config['model']['x'],
                      self.config['model']['y'],
                      self.config['model']['z'])
        print(f'Input spatial size: {spatial_size}')

        batch_size = (self.config['train']['batch_size'], self.config['valid']['batch_size'])
        num_workers = (self.config['train']['num_workers'], self.config['valid']['num_workers'])
        max_queue_length = self.config['train']['max_queue_length']
        samples_per_volume = self.config['train']['samples_per_volume']
        self.train_loader, self.valid_loader = dataloader.get_train_valid_dataloaders(
            datafolder=self.datafolder,
            data_type=self.data_type,
            spatial_size=spatial_size,
            batch_size=batch_size,
            max_queue_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            num_workers=num_workers,
            problem_type=_PROBLEM_TYPE
        )

        self.num_steps_train = len(self.train_loader) * self.config['train']['epochs']
        print(f'Total training step: {self.num_steps_train}')

        # Define network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model, self.optimizer = plain_unet.get_model_and_optimizer(
        #     in_channels=2,
        #     out_classes=3,
        #     dimensions=3,
        #     num_encoding_blocks=3,
        #     out_channels_first_layer=8,
        #     normalization='batch',
        #     upsampling_type='linear',
        #     padding=True,
        #     activation='PReLU',
        #     device=self.device)
        model_name = self.config['model']['name']
        model_config = _UNET_CONFIG
        if model_name == 'segresnet':
            model_config = _SEGRESNET_CONFIG

        monai_model = monai_models.get_model(
            model_name=model_name,
            output_type=self.config['model']['output_type'],
            device=self.device,
            spatial_dims=3,
            in_channels=self.config['model']['input_channel'],
            out_channels=self.config['model']['output_channel'],
            dropout_prob=0.3,
            network_config=model_config
        )
        # self.model = torch.nn.Sequential(monai_model, torch.nn.Sigmoid())
        self.model = monai_model

        if _PRINT_MODEL_SUMMARY:
            print(torchsummary.summary(self.model, input_size=input_size))

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config['train']['lr'])
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=num_steps_train, eta_min=0)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        # Define Loss and Criterion
        # self.loss = segmentation_loss.CrossEntropyLoss()  # default
        # if self.config['train']['loss_type'] == 'dice':
        #     self.loss = segmentation_loss.DiceLoss()
        # self.metrics = segmentation_metrics.DiceCoefficient()
        self.loss = monai_losses.get_segmentation_loss(config=self.config['loss'])
        self.metrics = monai_metrics.get_segmentation_metrics(
            config=self.config['metric'])

    def get_model(self) -> torch.nn.Module:
        return self.model

    def prepare_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create features and label for each batch."""
        features = torch.cat([batch[term.Modality.CT.value][tio.DATA], batch['PT'][tio.DATA]], dim=1).to(self.device)
        # label = batch['LABEL'][tio.DATA][:, 1:, ...].to(self.device)  # skip background channel
        label = batch['LABEL'][tio.DATA].to(self.device)
        return features, label

    # def prepare_subvolume_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Create features and label for each batch."""
    #     features = batch['input'].to(self.device)
    #     label = batch['label'].to(self.device)
    #     return features, label

    def prepare_subvolume_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create features and label for each batch."""
        # features = batch['input'][:, 1, ...]
        # features = features[:, None, ...].to(self.device)
        features = batch['input'].to(self.device)
        mask = torch.sum(batch['label'][:, 1:3, ...], dim=1)
        mask = mask[:, None, ...].to(self.device)
        return features, mask

    def train_one_epoch(self, epoch_idx: int, loader: DataLoader
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Training one epoch."""
        # np.set_printoptions(precision=3, suppress=True)
        logging_freq = self.config['train']['logging_frequency_steps']
        running_loss = 0.0
        last_loss = 0.0
        epoch_losses = []
        epoch_metrics = []
        for batch_idx, batch in enumerate(loader):
            if self.data_type == 'subvolume':
                inputs, targets = self.prepare_subvolume_batch(batch)
            else:
                inputs, targets = self.prepare_batch(batch)
            self.model.train(True)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            batch_loss = self.loss(outputs, targets)
            batch_loss.backward()
            self.optimizer.step()
            running_loss += batch_loss.item()
            epoch_losses.append(batch_loss.item())
            prediction = (torch.sigmoid(outputs) > 0.5).int()
            batch_metrics = self.metrics(prediction, targets)
            batch_metrics = batch_metrics.cpu().nanmean()
            epoch_metrics.append(batch_metrics.item())
            if batch_idx % logging_freq == logging_freq - 1:
                last_loss = running_loss/logging_freq
                message = f'\t Epoch {epoch_idx} Step {batch_idx}: '
                message += f'train loss {last_loss:0.3f} '
                print(message)
                running_loss = 0
        return np.array(epoch_losses), np.array(epoch_metrics)

    def valid_one_epoch(self, epoch_idx: int, loader: DataLoader,
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Validating one epoch."""
        epoch_losses = []
        epoch_metrics = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if self.data_type == 'subvolume':
                    inputs, targets = self.prepare_subvolume_batch(batch)
                else:
                    inputs, targets = self.prepare_batch(batch)
                self.model.eval()
                outputs = self.model(inputs)
                batch_loss = self.loss(outputs, targets)
                prediction = (torch.sigmoid(outputs) > 0.5).int()
                batch_metrics = self.metrics(prediction, targets)
                batch_metrics = batch_metrics.cpu().nanmean()
                epoch_losses.append(batch_loss.item())
                epoch_metrics.append(batch_metrics.item())
        return np.array(epoch_losses), np.array(epoch_metrics)

    def train_by_epochs(self, checkpoint_path: Optional[epath.Path] = None):
        history = {'train': {'loss': [], 'dice': []}, 'valid': {'loss': [], 'dice': []}}
        num_epochs = self.config['train']['epochs']
        best_loss = 99999
        counter = 0
        patience = 3

        # load checkpoint
        if checkpoint_path is not None:
            assert checkpoint_path.exists(), f'checkpoint cannot be found: {checkpoint_path}'
            print(f'Use checkpoint: {checkpoint_path}')
            checkpoint = torch.load(str(checkpoint_path))
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = checkpoint['model']
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint['best_loss']

        # val_losses.append(self.valid_epoch(0, self.valid_loader))
        for epoch_idx in range(1, num_epochs + 1):
            epoch_train_loss, epoch_train_metrics = self.train_one_epoch(
                epoch_idx, self.train_loader)
            epoch_valid_loss, epoch_valid_metrics = self.valid_one_epoch(
                epoch_idx, self.valid_loader)
            self.scheduler.step()
            history['train']['loss'].append(epoch_train_loss)
            history['valid']['loss'].append(epoch_valid_loss)
            history['train']['dice'].append(epoch_train_metrics)
            history['valid']['dice'].append(epoch_valid_metrics)
            current_validation_loss = epoch_valid_loss.mean()
            message = f'Epoch {epoch_idx}:  train loss {epoch_train_loss.mean():0.3f} '
            message += f'and dice {epoch_train_metrics.mean():0.3f} ||  '
            message += f'validation loss {current_validation_loss:0.3f} '
            message += f'and dice {epoch_valid_metrics.mean():0.3f}'
            print(message)

            # Model save and early stopping
            if current_validation_loss < best_loss:
                torch.save({
                    # 'model_state_dict': self.model.state_dict(),
                    # 'optimizer_state_dict': self.optimizer.state_dict(),
                    "model": self.model,
                    "optimizer": self.optimizer,
                    'best_loss': current_validation_loss
                }, str(self.runsfolder / 'best_model.pth'))
                best_loss = current_validation_loss
                counter = 0
            else:
                counter += 1

            # stop_training = False
            # if counter >= patience:
            #     stop_training = True
            #     print("Early stopping triggered!")

            # # Stop training if early stopping condition is met
            # if stop_training:
            #     return history
        return history

    def train_by_steps(self):
        history = {'train_losses': [], 'train_dice': []}
        logging_freq = int(self.config['train']['logging_frequency_steps'])
        print('Starting training ...')
        for step_idx in range(self.num_steps_train):
            print(f'step {step_idx} --0')
            batch = next(iter(self.train_loader))
            if self.data_type == 'subvolume':
                inputs, targets = self.prepare_subvolume_batch(batch)
            else:
                inputs, targets = self.prepare_batch(batch)
            # self.optimizer.zero_grad()
            print(f'step {step_idx} --1')
            with torch.set_grad_enabled(True):
                probabilities = self.model(inputs)
                batch_loss = self.loss(probabilities, targets)
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            print(f'step {step_idx} --2')
            history['train_losses'].append(batch_loss.item())
            batch_metrics = self.metrics(probabilities, targets)
            batch_metrics = torch.mean(batch_metrics, dim=0).detach().cpu().numpy()
            history['train_dice'].append(batch_metrics)
            if step_idx % logging_freq == 0:
                message = f'Step {step_idx}: '
                message += f'train loss {batch_loss.item():0.3f} '
                message += f'and dice {batch_metrics}'
                print(message)
            print(f'step {step_idx} --3')
