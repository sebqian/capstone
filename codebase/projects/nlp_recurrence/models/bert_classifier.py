"""This module defines a BERT classifier model."""
from typing import Tuple
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub

from nlp_recurrence.models import bert_variant

_BERT_MODEL_NAME = 'experts_pubmed'


def _create_foundation_model(bert_model_name: str,
                             trainable=False) -> Tuple[hub.KerasLayer, hub.KerasLayer]:
    """Create bert model and preprocessor model"""
    tfhub_handle_encoder = bert_variant.MAP_NAME_TO_HANDLE[bert_model_name]
    tfhub_handle_preprocess = bert_variant.MAP_MODEL_TO_PROCESS[bert_model_name]
    print(f'BERT model selected           : {tfhub_handle_encoder}')
    print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    bert_model = hub.KerasLayer(tfhub_handle_encoder, trainable=trainable, name='BERT_encoder')
    return bert_model, bert_preprocess_model


def build_classifier_model(bert_model_name: str = _BERT_MODEL_NAME) -> tf.keras.Model:
    """Build classifier based on foundation model."""
    encoder, preprocessing_layer = _create_foundation_model(bert_model_name, trainable=True)
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    encoder_inputs = preprocessing_layer(text_input)
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)
