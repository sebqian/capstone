from typing import Any, Dict, Tuple

import torch
from monai.networks.nets import segresnet, unet, swin_unetr


def get_model(
        model_name: str,
        img_size: Tuple[int, int, int],
        output_type: str,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        network_config: Dict[str, Any],
        dropout_prob: float = 0.3,
        ) -> torch.nn.Module:
    """Instantize a model."""

    monai_model = torch.nn.Module()
    if model_name == 'segresnet':
        monai_model = segresnet.SegResNet(
            spatial_dims=spatial_dims,
            init_filters=network_config['segresnet']['init_filters'],
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
            use_conv_final=network_config['segresnet']['use_conv_final'],
            blocks_down=tuple(network_config['segresnet']['blocks_down']),
            blocks_up=tuple(network_config['segresnet']['blocks_up']),
        )
    elif model_name == 'unet':
        monai_model = unet.UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout_prob,
            channels=network_config['channels'],
            strides=network_config['strides'],
            num_res_units=network_config['num_res_units']
        )
    elif model_name == 'swinUnetr':
        monai_model = swin_unetr.SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=network_config['swinUnetr']['feature_size'],
            depths=network_config['swinUnetr']['depths'],
            num_heads=network_config['swinUnetr']['num_heads'],
            drop_rate=network_config['swinUnetr']['drop_rate'],
            attn_drop_rate=network_config['swinUnetr']['attn_drop_rate'],
            use_v2=network_config['swinUnetr']['use_v2']
        )
    if output_type == 'sigmoid':
        model = torch.nn.Sequential(
            monai_model,
            torch.nn.Sigmoid()
        )
        return model
    return monai_model
