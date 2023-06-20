"""Simple dataloader for CSV data."""
from pathlib import Path
from typing import Tuple
import tensorflow as tf

from preprocessor.text import read_text_dataset

_BATCH_SIZE = 20


def get_dataset_train_valid(
        data_path: Path,
        batch_size: int = _BATCH_SIZE,
        label_col: str = 'Label') -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Get datasets for train and validation and/or evaluation"""

    train_processor = read_text_dataset.TextFromCSVProcessor(
        data_dir=data_path,
        pattern='TrainBreastSet*.csv',
        batch_size=batch_size,
        shuffle=True,
        phase='train',
        label_col=label_col
    )

    valid_processor = read_text_dataset.TextFromCSVProcessor(
        data_dir=data_path,
        pattern='ValidBreastSet*.csv',
        batch_size=batch_size,
        shuffle=False,
        phase='valid',
        label_col=label_col
    )

    train_ds = train_processor.get_dataset()
    valid_ds = valid_processor.get_dataset()

    return train_ds, valid_ds
