"""Train hecktor model in pytorch ligntning style."""
from typing import Any, Dict, List, Optional
from etils import epath

import argparse
import torch
import torchsummary
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from codebase import terminology as term
from codebase.lightning_module import seg_model_module
from codebase.models import monai_models
from codebase.dataloader.images import data_module
from codebase.projects.hecktor2022 import read_config

_PRINT_MODEL_SUMMARY = False
_SEGRESNET_CONFIG = {'init_filters': 32, 'use_conv_final': True,
                     'blocks_down': (1, 2, 2, 4, 4), 'blocks_up': (1, 1, 1, 1)}
# _SEGRESNET_CONFIG = {'init_filters': 32, 'use_conv_final': True,
#                      'blocks_down': (1, 2, 2, 4), 'blocks_up': (1, 1, 1)}
_UNET_CONFIG = {'channels': (16, 32, 64, 128, 256), 'strides': (2, 2, 2, 2), 'num_res_units': 2}


def get_model_module(config: Dict[str, Any]) -> pl.LightningModule:
    """Generates the model module."""

    if _PRINT_MODEL_SUMMARY:
        spatial_size = config['model']['spatial_size']
        input_size = config['model']['input_channel'] + config['model']['spatial_size']
        print(f'Input spatial size: {spatial_size}')

        monai_model = monai_models.get_model(
                model_name=config['model']['name'],
                img_size=spatial_size,
                output_type=config['model']['output_type'],
                spatial_dims=config['model']['spatial_dims'],
                in_channels=config['model']['input_channel'],
                out_channels=config['model']['output_channel'],
                dropout_prob=config['model']['dropout'],
                network_config=config['model']['architecture']
            )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torchsummary.summary(monai_model.to(device), input_size=input_size))

    hecktor_model = seg_model_module.SegmentationModelModule(
        hparams=config,
        optimizer_class=torch.optim.AdamW,
    )

    return hecktor_model


def get_data_module(config: Dict[str, Any]) -> pl.LightningDataModule:
    """Generates data module for training."""
    mdata = data_module.MedicalImageDataModule(
        task_type=term.ProblemType.SEGMENTATION,
        config=config,
    )
    return mdata


def get_callbacks(save_top_k: int) -> List[Callback]:
    """Generates callbacks for trainer."""
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=2, save_top_k=save_top_k, monitor="val_loss",
        mode='min', filename='checkpoint-{epoch:02d}-{val_loss:.2f}',)
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    return [checkpoint_callback, lr_monitor_callback]


# def main(hparams: argparse.Namespace):
def main(config_file: epath.Path, checkpoint_path: Optional[str]):
    config = read_config.read_experiment_config(config_file)
    base_dir = epath.Path(config['experiment']['data_path'])
    experiment_name = config['experiment']['name']
    runsfolder = base_dir / 'experiments'
    max_epochs = config['train']['epochs']
    print(f'Max epochs: {max_epochs}')
    if not runsfolder.exists():
        runsfolder.mkdir(parents=True, exist_ok=True)

    mdata = get_data_module(config)
    if checkpoint_path:
        model = seg_model_module.SegmentationModelModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            optimizer_class=torch.optim.AdamW
        )
        model.lr = config['train']['lr']
        model.configure_optimizers()
    else:
        model = get_model_module(config)
    print(f'Starting learning rate: {model.lr}')
    callbacks = get_callbacks(5)
    logger = TensorBoardLogger(save_dir=runsfolder, version=1, name=experiment_name)  # type: ignore
    trainer = pl.Trainer(accelerator="gpu", logger=logger,
                         max_epochs=max_epochs, check_val_every_n_epoch=1,
                         precision=16,
                         enable_model_summary=True,
                         enable_progress_bar=True,
                         log_every_n_steps=config['train']['logging_frequency_steps'],
                         callbacks=callbacks,
                         )
    trainer.logger._default_hp_metric = False
    # tuner = Tuner(trainer)
    # tuner.lr_find(model)
    # trainer.fit(model, datamodule=mdata, ckpt_path=checkpoint_path)
    trainer.fit(model, datamodule=mdata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    config_file = epath.Path(args.config)

    main(config_file, args.checkpoint)
