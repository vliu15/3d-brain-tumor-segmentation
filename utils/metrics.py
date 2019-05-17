import tensorflow as tf

from utils.constants import *


def dice_coefficient(y_pred, y_true, eps=1e-8,
                     data_format='channels_last', include_non_brain=False):
    """Returns dice coefficient between predicted and true outputs.

        Args:
            y_pred: decoder output holding probabilities of each voxel
                is a tumor, with one tumor per channel.
            y_true: true segmentation labels of values [0, 3].
            eps: optional smoothing value added to the numerator and
                denominator.
            data_format: whether data is in the format `channels_last`
                or `channels_first`.

        Returns:
            dice_coeff: average dice coefficient across all channels.
    """
    # Extract predictions at each voxel.
    axis = -1 if data_format == 'channels_last' else 1
    y_pred = tf.argmax(y_pred, axis=axis, output_type=tf.int32)

    # Turn into one-hot encodings per voxel.
    y_pred = tf.one_hot(y_pred, len(LABELS), axis=axis, dtype=tf.float32)

    # Correct predictions will have 1, else 0.
    intersection = y_pred * y_true

    # Sum up per channel.
    axis = (0, 1, 2, 3) if data_format == 'channels_last' else (0, 2, 3, 4)
    intersection = tf.reduce_sum(intersection, axis=axis)

    # Count total number of each label
    true_voxels = tf.reduce_sum(y_true, axis=axis)
    pred_voxels = tf.reduce_sum(y_pred)

    # Dice coefficients per channel
    dice_coeff = (2.0 * intersection + eps) / (true_voxels + pred_voxels + eps)

    if include_non_brain:
        return (tf.reduce_mean(dice_coeff), tf.reduce_mean(dice_coeff[1:]))
    else:
        return tf.reduce_mean(dice_coeff)


def segmentation_accuracy(y_pred, y_true, data_format='channels_last'):
    """Returns voxel-wise accuracy of the prediction, excluding non-brain voxels.

        Args:
            y_pred: decoder output holding probabilities of each voxel
                is a tumor, with one tumor per channel.
            y_pred: true segmentation label with 0 at non-tumor voxels
                and the label number of a voxel with a corresponding tumor.
            data_format: whether data is in the format `channels_last`
                or `channels_first`.

        Returns:
             Voxel accuracy: average voxel-wise accuracy across all voxels.
    """
    axis = -1 if data_format == 'channels_last' else 1
    total_voxels = tf.cast(tf.reduce_prod(y_true.shape), tf.float32)

    # Extract predictions at each voxel.
    y_pred = tf.argmax(y_pred, axis=axis)
    y_pred = tf.expand_dims(y_pred, axis=axis)

    # Correct predictions will have 0, else 1.
    correct = tf.cast(tf.cast(y_pred - y_true, tf.bool), tf.float32)

    # Sum up all correct predictions.
    correct = tf.reduce_sum(1.0 - correct)

    return correct / total_voxels
