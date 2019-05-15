import tensorflow as tf

from utils.constants import *


def dice_coefficient(y_pred, y_true, eps=1e-8, log=False):
    """Returns dice coefficient between predicted and true outputs.

        Args:
            y_pred: decoder output holding probabilities of each voxel
                is a tumor, with one tumor per channel.
            y_pred: true segmentation label with 0 at non-tumor voxels
                and the label number of a voxel with a corresponding tumor.
            eps: optional smoothing value added to the numerator and
                denominator.
            log: whether to output dictionary of dice coefficients per channel.

        Returns:
            dice_coeff: average dice coefficient across all channels.
                if log: dice coefficients per channel as well.
    """
    y_true = tf.reshape(y_true, [-1])
    y_pred = y_pred * tf.dtypes.cast(y_pred > 0.5, tf.float32)

    dice_coeff = {}

    for l, channel in zip(LABELS, range(OUT_CH)):
        y_pred_ch = tf.reshape(y_pred[..., channel], [-1])
        y_true_ch = tf.dtypes.cast(y_true == l, tf.float32)

        numer = 2.0 * tf.math.reduce_sum(y_true_ch * y_pred_ch) + eps
        denom = tf.math.reduce_sum(y_true_ch ** 2) + tf.math.reduce_sum(y_pred_ch ** 2) + eps
        dice_coeff[l] = numer / denom

    if log:
        return sum(dice_coeff.values()) / len(dice_coeff), dice_coeff
        
    return sum(dice_coeff.values()) / len(dice_coeff)


def voxel_accuracy(y_pred, y_true, log=False):
    """Returns voxel-wise accuracy of the prediction, excluding non-brain voxels.

        Args:
            y_pred: decoder output holding probabilities of each voxel
                is a tumor, with one tumor per channel.
            y_pred: true segmentation label with 0 at non-tumor voxels
                and the label number of a voxel with a corresponding tumor.
            log: whether to output dictionary of voxel accuracies per channel.

        Returns:
        voxel_accuracy: average voxel-wise accuracy across all channels.
            if log: voxel-wise accuracies per channel as well.
    """
    y_pred = y_pred * tf.dtypes.cast(y_pred > 0.5, tf.float32)
    n_brain_voxels = tf.reduce_sum(tf.dtypes.cast(y_true > 0, tf.float32))

    voxel_accuracy = {}

    for l, channel in zip(LABELS, range(OUT_CH)):
        y_pred_ch = y_pred[..., channel]
        y_true_ch = tf.dtypes.cast(y_true == l, tf.float32)

        n_correct = tf.reduce_sum(tf.dtypes.cast((y_pred_ch - y_true_ch) == 0.0, tf.float32))
        voxel_accuracy[l] = n_correct / n_brain_voxels
    
    if log:
        return sum(voxel_accuracy.values()) / len(voxel_accuracy), voxel_accuracy

    return sum(voxel_accuracy.values()) / len(voxel_accuracy)
