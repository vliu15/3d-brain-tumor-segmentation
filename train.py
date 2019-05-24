import tensorflow as tf
import numpy as np

from utils.arg_parser import train_parser
from utils.losses import FocalLoss
from utils.optimizer import Scheduler
from utils.constants import *
from model.volumetric_cnn import EncDecCNN


def prepare_dataset(path, batch_size, buffer_size, data_format='channels_last'):
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
    dataset_len = get_dataset_len(dataset)

    dataset = (dataset.map(parse_example)
                      .repeat()
                      .shuffle(buffer_size)
                      .batch(batch_size))

    return dataset, dataset_len


def train(args):
    # Load data.
    train_data, n_train = prepare_dataset(
                            args.train_loc, args.batch_size, 500, data_format=args.data_format)
    val_data, n_val = prepare_dataset(
                            args.val_loc, args.batch_size, 50, data_format=args.data_format)
    print('{} training examples.'.format(n_train))
    print('{} validation examples.'.format(n_val))
    train_steps_per_epoch = np.ceil(float(n_train) / args.batch_size).astype(np.int32)
    val_steps_per_epoch = np.ceil(float(n_val) / args.batch_size).astype(np.int32)

    with tf.device(args.device):
        if args.load_file:
            model = tf.keras.models.load_model(args.load_file)
        else:
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
                      loss=FocalLoss(
                                    gamma=2,
                                    alpha=0.25,
                                    data_format=args.data_format),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(),
                               tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

        history = model.fit(train_data,
                            epochs=args.n_epochs,
                            shuffle=True,
                            callbacks=[tf.keras.callbacks.LearningRateScheduler(
                                            Scheduler(args.n_epochs, args.lr)),
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
                            steps_per_epoch=train_steps_per_epoch,
                            validation_steps=val_steps_per_epoch,
                            validation_data=val_data,
                            validation_freq=1)

        return history


if __name__ == '__main__':
    args = train_parser()
    history = train(args)
