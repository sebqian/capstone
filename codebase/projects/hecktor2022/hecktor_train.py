"""Train hecktor model in pytorch ligntning style."""
from typing import Any, Dict, List
from etils import epath
import yaml
import argparse
import torch
import torchsummary
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

from codebase import terminology as term
from codebase.lightning_module import model_module
from codebase.models import monai_models
from codebase.custom_losses import monai_losses
from codebase.custom_metrics import monai_metrics
from codebase.dataloader.images import data_module

_PRINT_MODEL_SUMMARY = False
_SEGRESNET_CONFIG = {'init_filters': 32, 'use_conv_final': True,
                     'blocks_down': (1, 2, 2, 4, 4), 'blocks_up': (1, 1, 1, 1)}
# _SEGRESNET_CONFIG = {'init_filters': 32, 'use_conv_final': True,
#                      'blocks_down': (1, 2, 2, 4), 'blocks_up': (1, 1, 1)}
_UNET_CONFIG = {'channels': (16, 32, 64, 128, 256), 'strides': (2, 2, 2, 2), 'num_res_units': 2}


def _read_experiment_config(config_file: epath.Path) -> Dict[str, Any]:
    """Read experiment configurations."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        return config


def get_model_module(config: Dict[str, Any],
                     network_architect: Dict[str, Any]) -> pl.LightningModule:
    """Generates the model module."""
    spatial_size = (config['model']['x'],
                    config['model']['y'],
                    config['model']['z'])
    input_size = (config['model']['input_channel'],
                  config['model']['x'],
                  config['model']['y'],
                  config['model']['z'])
    print(f'Input spatial size: {spatial_size}')

    monai_model = monai_models.get_model(
            model_name=config['model']['name'],
            output_type=config['model']['output_type'],
            spatial_dims=3,
            in_channels=config['model']['input_channel'],
            out_channels=config['model']['output_channel'],
            dropout_prob=0.3,
            network_config=network_architect
        )
    if _PRINT_MODEL_SUMMARY:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torchsummary.summary(monai_model.to(device), input_size=input_size))

    loss = monai_losses.get_segmentation_loss(config['loss'])
    metric = monai_metrics.get_segmentation_metrics(config['metric'])
    hecktor_model = model_module.SegmentationModelModule(
        net=monai_model, criterion=loss,
        metrics=metric,
        num_classes=config['model']['output_channel'],
        batch_size=(config['train']['batch_size'], config['valid']['batch_size']),
        learning_rate=config['train']['lr'],
        optimizer_class=torch.optim.AdamW,
    )

    return hecktor_model


def get_data_module(config: Dict[str, Any]) -> pl.LightningDataModule:
    """Generates data module for training."""
    base_dir = epath.Path(config['experiment']['data_path'])
    mdata = data_module.MedicalImageDataModule(
        task_type=term.ProblemType.SEGMENTATION, root_dir=base_dir,
        experiment_config=config['experiment'], train_config=config['train'],
        valid_config=config['valid'], test_config=config['test'],
    )
    return mdata


def get_callbacks(save_top_k: int) -> List[Callback]:
    """Generates callbacks for trainer."""
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=2, save_top_k=save_top_k, monitor="val_loss",
        mode='min', filename='checkpoint-{epoch:02d}-{val_loss:.2f}',)
    return [checkpoint_callback]


# def main(hparams: argparse.Namespace):
def main(config_file: epath.Path):
    config = _read_experiment_config(config_file)
    base_dir = epath.Path(config['experiment']['data_path'])
    experiment_name = config['experiment']['name']
    runsfolder = base_dir / 'experiments'
    max_epochs = config['train']['epochs']
    print(f'Max epochs: {max_epochs}')
    if not runsfolder.exists():
        runsfolder.mkdir(parents=True, exist_ok=True)

    mdata = get_data_module(config)
    model = get_model_module(config, network_architect=_SEGRESNET_CONFIG)
    callbacks = get_callbacks(3)
    logger = TensorBoardLogger(save_dir=runsfolder, version=1, name=experiment_name)
    trainer = pl.Trainer(accelerator="gpu", logger=logger,
                         max_epochs=max_epochs, check_val_every_n_epoch=1,
                         enable_model_summary=True,
                         enable_progress_bar=True,
                         log_every_n_steps=config['train']['logging_frequency_steps'],
                         callbacks=callbacks,
                         )
    # tuner = Tuner(trainer)
    # tuner.lr_find(model)
    trainer.fit(model, datamodule=mdata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config_file = epath.Path(args.config)

    main(config_file)
