"""NLP BERT trainer module. Tensorflow based."""
from absl import flags
from absl import app

import tensorflow as tf
from official.nlp import optimization

import codebase_settings as cbs
from nlp_recurrence.models import bert_classifier
from nlp_recurrence.models import dataloader

_DATA_PATH = cbs.DATA_PATH / 'breast_recurrence'


FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 10,
                     help='batch size for training (default: 1)')
flags.DEFINE_list('list_GPU_ids', [0, 1, 2, 3],
                  help='list_GPU_ids for training (default: [0])')
flags.DEFINE_integer('epochs', 10,
                     help='training epochs')
flags.DEFINE_integer('exp_id', 0, help='Experiment ID')
flags.DEFINE_string('exp_name', 'Breast_Recurrence', help='Experiment name')


def main(args):
    """Defines main function."""

    if len(args) > 1:
        raise Exception("No parameter is allowed for this function.")

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # load dataset
    train_ds, valid_ds = dataloader.get_dataset_train_valid(
        data_path=_DATA_PATH
    )

    #  Start training
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    classifier_model = bert_classifier.build_classifier_model()

    init_lr = 3e-4
    steps_per_epoch = int(300 / FLAGS.batch_size)
    num_train_steps = steps_per_epoch * FLAGS.epochs
    num_warmup_steps = int(0.1*num_train_steps)
    print(f'Steps per epoch to be used: {steps_per_epoch}')
    print(f'Warmup steps: {num_warmup_steps}')

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(
        init_lr=init_lr, num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type='adamw')
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = classifier_model.fit(x=train_ds,
                                   validation_data=valid_ds,
                                   batch_size=FLAGS.batch_size,            
                                   epochs=FLAGS.epochs,
                                   steps_per_epoch=steps_per_epoch,
                                   validation_batch_size=8,
                                   validation_steps=3)
    print('# Training is done !\n')
    return history


if __name__ == '__main__':
    app.run(main)
