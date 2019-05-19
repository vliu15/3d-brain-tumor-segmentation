import tensorflow as tf

from utils.constants import *


def dice_coefficient(y_true, y_pred, eps=1e-8,
                     data_format='channels_last', include_brain_only=False):
    """Returns dice coefficient between predicted and true outputs.

        Args:
            y_true: true segmentation labels per voxel, represented by
                one-hot vectors of positions [0, 3].
            y_pred: decoder output holding probabilities of each voxel
                is a tumor, with one tumor per channel.
            eps: optional smoothing value added to the numerator and
                denominator.
            data_format: whether data is in the format `channels_last`
                or `channels_first`.
            include_brain_only: whether to return dice coefficient
                averaged across channels corresponding to tumors.

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

    # Count total number of each label.
    true_voxels = tf.reduce_sum(y_true, axis=axis)
    pred_voxels = tf.reduce_sum(y_pred, axis=axis)

    # Dice coefficients per channel.
    dice_coeff = (2.0 * intersection + eps) / (true_voxels + pred_voxels + eps)

    if include_brain_only:
        return (tf.reduce_mean(dice_coeff), tf.reduce_mean(dice_coeff[1:]))
    else:
        return tf.reduce_mean(dice_coeff)


def segmentation_accuracy(y_true, y_pred, data_format='channels_last'):
    """Returns voxel-wise accuracy of the prediction, excluding non-brain voxels.

        Args:
            y_true: true segmentation labels per voxel, represented by
                one-hot vectors of positions [0, 3].
            y_pred: decoder output holding probabilities of each voxel
                is a tumor, with one tumor per channel.
            data_format: whether data is in the format `channels_last`
                or `channels_first`.

        Returns:
             Voxel accuracy: average voxel-wise accuracy across all voxels.
    """
    axis = -1 if data_format == 'channels_last' else 1

    # Compute shape, divide by number of channels.
    total_voxels = tf.reduce_prod(y_true.shape) / len(LABELS)
    total_voxels = tf.cast(total_voxes, tf.float32)

    # Extract predictions at each voxel.
    y_pred = tf.argmax(y_pred, axis=axis, output_type=tf.int32)
    y_pred = tf.one_hot(y_pred, len(LABELS), axis=axis, dtype=tf.float32)

    # Correct predictions will have 1, else 0.
    correct = tf.reduce_sum(y_pred * y_true)

    return correct / total_voxels


def sensitivity(y_true, y_pred, data_format='channels_last'):
    # Extract predictions at each voxel.
    axis = -1 if data_format == 'channels_last' else 1
    y_pred = tf.argmax(y_pred, axis=axis, output_type=tf.int32)

    # Turn into one-hot encodings per voxel.
    y_pred = tf.one_hot(y_pred, len(LABELS), axis=axis, dtype=tf.float32)

    # Calculate sensitivity.
    true_positives = tf.reduce_sum(y_pred * y_true)
    num_positives = tf.reduce_sum(y_true)

    return true_positives / num_positives


def specificity(y_true, y_pred, data_format='channels_last'):
    # Extract predictions at each voxel.
    axis = -1 if data_format == 'channels_last' else 1
    y_pred = tf.argmax(y_pred, axis=axis, output_type=tf.int32)

    # Turn into one-hot encodings per voxel.
    y_pred = tf.one_hot(y_pred, len(LABELS), axis=axis, dtype=tf.float32)

    # Calculate specificity.
    y_pred = 1.0 - y_pred
    y_true = 1.0 - y_true

    true_negatives = tf.reduce_sum(y_pred * y_true)
    num_negatives = tf.reduce_sum(y_true)

    return true_negatives / num_negatives
