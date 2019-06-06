"""Contains util functions for working with model inputs and outputs."""
import tensorflow as tf
import numpy as np

from utils.constants import *


def prepare_batch(X, y, prob=0.5, data_format='channels_last'):
    """Performs augmentation and cropping in training."""
    # Data augmentation.
    if np.random.uniform() < prob:
        axes = (1, 2, 3) if data_format == 'chanenls_last' else (2, 3, 4)
        X = tf.reverse(X, axis=axes)
        y = tf.reverse(y, axis=axes)

    # Sample corner points.
    h = np.random.randint(H, RAW_H)
    w = np.random.randint(W, RAW_W)
    d = np.random.randint(D, RAW_D)

    if data_format == 'channels_last':
        X = X[:, h-H:h, w-W:w, d-D:d, :]
        y = y[:, h-H:h, w-W:w, d-D:d, :]
    else:
        X = X[:, :, h-H:h, w-W:w, d-D:d]
        y = y[:, :, h-H:h, w-W:w, d-D:d]
        
    return X, y


def prepare_val_set(dataset, n_sets=2, prob=0.5, data_format='channels_last'):
    """Prepares validation sets (with cropping and flipping)."""
    def parse_example(X, y):
        return prepare_batch(X, y, prob=prob, data_format=data_format)
    
    for i in range(n_sets):
        if i == 0: val_set = dataset.map(parse_example)
        else: val_set = val_set.concatenate(dataset.map(parse_example))

    return val_set


def prepare_dataset(path, batch_size, buffer_size=1000, data_format='channels_last', repeat=False):
    """Returns a BatchDataset object containing loaded data."""
    def parse_example(example_proto):
        parsed = tf.io.parse_single_example(example_proto, example_desc)
        if data_format == 'channels_last':
            X = tf.reshape(parsed['X'], (RAW_H, RAW_W, RAW_D, IN_CH))
            y = tf.reshape(parsed['y'], (RAW_H, RAW_W, RAW_D, 1))
            y = tf.cast(y, tf.int32)
            y = tf.squeeze(y, axis=-1)
            y = tf.one_hot(y, OUT_CH, axis=-1, dtype=tf.float32)
            y = y[:, :, :, 1:]
        elif data_format == 'channels_first':
            X = tf.reshape(parsed['X'], (IN_CH, RAW_H, RAW_W, RAW_D))
            y = tf.reshape(parsed['y'], (1, RAW_H, RAW_W, RAW_D))
            y = tf.cast(y, tf.int32)
            y = tf.squeeze(y, axis=0)
            y = tf.one_hot(y, OUT_CH, axis=0, dtype=tf.float32)
            y = y[1:, :, :, :]

        return (X, y)

    def get_dataset_len(tf_dataset):
        """Returns length of dataset until tf.data.experimental.cardinality is fixed."""
        # return tf.data.experimental.cardinality(tf_dataset)
        return sum(1 for _ in tf_dataset)

    example_desc = {
        'X': tf.io.FixedLenFeature([RAW_H * RAW_W * RAW_D * IN_CH], tf.float32),
        'y': tf.io.FixedLenFeature([RAW_H * RAW_W * RAW_D * 1], tf.float32)
    }

    dataset = tf.data.TFRecordDataset(path)
    dataset_len = get_dataset_len(dataset)

    if repeat:
        dataset = (dataset.map(parse_example)
                      .repeat()
                      .shuffle(buffer_size)
                      .batch(batch_size))
    else:
        dataset = (dataset.map(parse_example)
                      .shuffle(buffer_size)
                      .batch(batch_size))

    return dataset, dataset_len


def pred_to_one_hot(pred, data_format):
    """Converts output of predicted probabilites to one-hot encodings."""
    axis = -1 if data_format == 'channels_last' else 1
    pred = tf.argmax(pred, axis=axis, output_type=tf.int32)
    pred = tf.one_hot(pred, OUT_CH, axis=axis, dtype=tf.float32)

    return pred
