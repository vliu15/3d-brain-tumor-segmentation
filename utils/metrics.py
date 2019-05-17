import tensorflow as tf

from utils.constants import *


def dice_coefficient(y_pred, y_true, eps=1e-8, data_format='channels_last'):
    """Returns dice coefficient between predicted and true outputs.

        Args:
            y_pred: decoder output holding probabilities of each voxel
                is a tumor, with one tumor per channel.
            y_pred: true segmentation label with 0 at non-tumor voxels
                and the label number of a voxel with a corresponding tumor.
            eps: optional smoothing value added to the numerator and
                denominator.
            data_format: whether data is in the format `channels_last`
                or `channels_first`.

        Returns:
            dice_coeff: average dice coefficient across all channels.
    """
    axis = (0, 1, 2, 3) if data_format == 'channels_last' else (0, 2, 3, 4)
    shape = (1, 1, 1, 1, -1) if data_format == 'channels_last' else (1, -1, 1, 1, 1)

    # Create binary mask for each label and corresponding channel.
    labels = tf.reshape(tf.convert_to_tensor(LABELS, dtype=tf.float32), shape=shape)
    y_true = 1.0 - tf.dtypes.cast(tf.dtypes.cast(y_true - labels, tf.bool), tf.float32)

    # Round probabilities >0.5 to 1 and <0.5 to 0.
    y_pred = tf.dtypes.cast(y_pred > 0.5, tf.float32)

    numer = 2.0 * tf.math.reduce_sum(y_true * y_pred, axis=axis) + eps
    denom = tf.math.reduce_sum(y_true ** 2, axis=axis) + \
            tf.math.reduce_sum(y_pred ** 2, axis=axis) + eps

    return tf.reduce_mean(numer / denom)


def segmentation_accuracy(x, y_pred, y_true, data_format='channels_last'):
    """Returns voxel-wise accuracy of the prediction, excluding non-brain voxels.

        Args:
            y_pred: decoder output holding probabilities of each voxel
                is a tumor, with one tumor per channel.
            y_pred: true segmentation label with 0 at non-tumor voxels
                and the label number of a voxel with a corresponding tumor.
            data_format: whether data is in the format `channels_last`
                or `channels_first`.

        Returns:
            Brain voxel accuracy: average voxel-wise accuracy across brain voxels.
            Net voxel accuracy: average voxel-wise accuracy across all voxels.
    """
    n_brain_voxels = tf.reduce_sum(tf.dtypes.cast(x > 0, tf.float32))
    n_non_brain_voxels = tf.reduce_sum(1.0 - tf.dtypes.cast(tf.dtypes.cast(x, tf.bool), tf.float32))
    shape = (1, 1, 1, 1, -1) if data_format == 'channels_last' else (1, -1, 1, 1, 1)

    # Create binary mask for each label and corresponding channel.
    labels = tf.reshape(tf.convert_to_tensor(LABELS, dtype=tf.float32), shape=shape)
    labels = tf.broadcast_to(labels, y_true.shape)
    y_true = 1.0 - tf.dtypes.cast(tf.dtypes.cast(y_true - labels, tf.bool), tf.float32)

    # Round probabilities >0.5 to 1 and <0.5 to 0.
    y_pred = tf.dtypes.cast(y_pred > 0.5, tf.float32)

    # Find where true and pred match, but remove all voxels outside of the brain.
    n_correct = 1.0 - tf.dtypes.cast(tf.dtypes.cast(y_true - y_pred, tf.bool), tf.float32)
    return (n_correct - n_non_brain_voxels) / n_brain_voxels, n_correct / n_brain_voxels
