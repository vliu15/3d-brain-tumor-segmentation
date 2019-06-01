"""Contains util functions for working with model inputs and outputs."""
import tensorflow as tf
import numpy as np

from utils.constants import *


def prepare_dataset(path, batch_size, buffer_size=1000, data_format='channels_last'):
    """Returns a BatchDataset object containing loaded data."""
    def parse_example(example_proto):
        """Mapping function to parse a single example."""
        parsed = tf.io.parse_single_example(example_proto, example_desc)
        if data_format == 'channels_last':
            X = tf.reshape(parsed['X'], CHANNELS_LAST_X_SHAPE)
            y = tf.reshape(parsed['y'], CHANNELS_LAST_Y_SHAPE)
            y = tf.cast(y, tf.int32)
            y = tf.squeeze(y, axis=-1)
            y = tf.one_hot(y, OUT_CH, axis=-1, dtype=tf.float32)
            y = y[:, :, :, 1:]
        elif data_format == 'channels_first':
            X = tf.reshape(parsed['X'], CHANNELS_FIRST_X_SHAPE)
            y = tf.reshape(parsed['y'], CHANNELS_FIRST_Y_SHAPE)
            y = tf.cast(y, tf.int32)
            y = tf.squeeze(y, axis=1)
            y = tf.one_hot(y, OUT_CH, axis=1, dtype=tf.float32)
            y = y[1:, :, :, :]

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
                      .shuffle(buffer_size)
                      .batch(batch_size))

    return dataset, dataset_len


def pred_to_one_hot(pred, data_format):
    """Converts output of predicted probabilites to one-hot encodings."""
    axis = -1 if data_format == 'channels_last' else 1
    pred = tf.argmax(pred, axis=axis, output_type=tf.int32)
    pred = tf.one_hot(pred, OUT_CH, axis=axis, dtype=tf.float32)

    return pred
