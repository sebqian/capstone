""" Loading text data from csv and create TF dataset """
from typing import Optional, OrderedDict, Tuple, List
import tensorflow as tf
# from tensorflow.keras import layers
# import pandas as pd
from pathlib import Path

_COL_NAMES = ['Note', 'Label']
_COL_DEFAULTS = ["", ""]
AUTOTUNE = tf.data.AUTOTUNE


# Define a conversion function to convert 'yes' and 'no' to 1 and 0
def _label_conversion(label):
    return tf.cond(tf.equal(label, 'yes'), lambda: tf.constant(1), lambda: tf.constant(0))


def _string_process_train(note: OrderedDict[str, tf.Tensor], label: tf.Tensor
                    ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Convert the output from dataset into tensors."""
    note_tensor = note[_COL_NAMES[0]]
    note_tensor = tf.strings.lower(note_tensor)
    label = tf.strings.lower(label)
    label_tensor = tf.map_fn(_label_conversion, label, dtype=tf.int32)
    return note_tensor, label_tensor


def _string_process_pred(note: OrderedDict[str, tf.Tensor]) -> tf.Tensor:
    """Convert the output from dataset into tensors."""
    note_tensor = note[_COL_NAMES[0]]
    note_tensor = tf.strings.lower(note_tensor)
    return note_tensor


class TextFromCSVProcessor():
    """Text Dataset for NLP."""

    def __init__(
            self, data_dir: Path, pattern: str, batch_size: int,
            shuffle: bool = False, phase: str = 'train',
            col_names: Optional[List[str]] = _COL_NAMES,
            col_defaults: Optional[List] = _COL_DEFAULTS,
            selected_col: Optional[List[str]] = None,
            label_col: Optional[str] = None):

        print(f'Loading data from {data_dir} with pattern {pattern}.')
        self.phase = phase
        self.dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=str(data_dir / pattern), batch_size=batch_size,
            header=True, shuffle=shuffle, column_names=col_names,
            column_defaults=col_defaults, select_columns=selected_col,
            label_name=label_col
            )

        if phase in ['train', 'valid', 'eval']:
            self.dataset = self.dataset.map(_string_process_train)
        elif phase == 'pred':
            self.dataset = self.dataset.map(_string_process_pred)
        else:
            raise ValueError(f'{phase} is not a supported phase [train, valid, eval, pred]')

        self.dataset_status = {'tokenization': False,
                               'embedding': False,
                               'model_preprocessing': False}

    def get_dataset(self) -> tf.data.Dataset:
        if self.phase == 'train':
            return self.dataset.repeat().cache().prefetch(buffer_size=AUTOTUNE)
        return self.dataset.cache().prefetch(buffer_size=AUTOTUNE)
