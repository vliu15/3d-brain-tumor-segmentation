import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.arg_parser import train_parser
from utils.losses import focal_loss
from utils.optimizer import scheduler
from utils.constants import *
from utils.metrics import dice_coefficient, segmentation_accuracy
from model.volumetric_cnn import EncDecCNN


def prepare_dataset(path, batch_size, data_format='channels_last'):
    """Returns a BatchDataset object containing loaded data."""
    def parse_example(example_proto):
        """Mapping function to parse a single example."""
        parsed = tf.io.parse_single_example(example_proto, example_desc)
        if data_format == 'channels_last':
            X = tf.reshape(parsed['X'], CHANNELS_LAST_X_SHAPE)
            y = tf.reshape(parsed['y'], CHANNELS_LAST_Y_SHAPE)
            y = tf.cast(y, tf.int32)
            y = tf.squeeze(y, axis=-1)
            y = tf.one_hot(y, depth=OUT_CH, axis=-1)
        elif data_format == 'channels_first':
            X = tf.reshape(parsed['X'], CHANNELS_FIRST_X_SHAPE)
            y = tf.reshape(parsed['y'], CHANNELS_FIRST_Y_SHAPE)
            y = tf.cast(y, tf.int32)
            y = tf.squeeze(y, axis=1)
            y = tf.one_hot(tf.cast(y, tf.int32), depth=OUT_CH, axis=1)
        return (X, y)

    def get_dataset_len(tf_dataset):
        """Returns length of dataset until tf.data.experimental.cardinality is fixed."""
        # return tf.data.experimental.cardinality(tf_dataset)
        return sum(1 for _ in tf_dataset)

    example_desc = {
        'X': tf.io.FixedLenFeature([H * W * D * IN_CH], tf.float32),
        'y': tf.io.FixedLenFeature([H * W * D * 1], tf.float32)
    }

    dataset = tf.data.TFRecordDataset(path)
    dataset = (dataset.map(parse_example)
                      .shuffle(10000, reshuffle_each_iteration=True)
                      .batch(batch_size))

    dataset.__len__ = get_dataset_len(dataset)
    return dataset


def main(args):
    # Load data.
    train_data = prepare_dataset(args.train_loc, args.batch_size)
    val_data = prepare_dataset(args.val_loc, args.batch_size)
    print('{} training examples.'.format(train_data.__len__))
    print('{} validation examples.'.format(val_data.__len__))

    with tf.device(args.device):
        # Initialize model and optimizer.
        model = EncDecCNN(
                        data_format=args.data_format,
                        kernel_size=args.conv_kernel_size,
                        groups=args.gn_groups,
                        kernel_regularizer=tf.keras.regularizers.l2(l=args.l2_scale))

        model.compile(optimizer=tf.keras.optimizers.Adam(
                                    learning_rate=args.lr,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-7,
                                    amsgrad=False),
                      loss=focal_loss(gamma=2,
                                    alpha=0.25,
                                    data_format=args.data_format),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(),
                               tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

            model.fit(train_data,
                      epochs=args.n_epochs,
                      shuffle=True,
                      callbacks=[tf.keras.callbacks.LearningRateScheduler(
                                        scheduler(args.n_epochs, args.lr)),
                                 tf.keras.callbacks.ModelCheckpoint(
                                        args.save_file,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        save_weights_only=False),
                                 tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss',
                                        min_delta=1e-2,
                                        patience=args.patience),
                                 tf.keras.callbacks.CSVLogger(
                                        args.log_file)],
                      validation_data=val_data,
                      validation_freq=1)


if __name__ == '__main__':
    args = train_parser()
    main(args)
