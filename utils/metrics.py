"""Contains custom metrics for evaluation."""
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff

from utils.constants import *


def pred_to_one_hot(pred, data_format):
    """Converts output of predicted probabilites to one-hot encodings."""
    axis = -1 if data_format == 'channels_last' else 1

    # Mask out values that correspond to values < 0.5.
    mask = tf.reduce_max(pred, axis=axis, keepdims=True)
    mask = tf.cast(mask > 0.5, tf.float32)

    pred = tf.argmax(pred, axis=axis, output_type=tf.int32)
    pred = tf.one_hot(pred, OUT_CH+1, axis=axis, dtype=tf.float32)
    pred *= mask

    return pred


class HausdorffDistance(tf.keras.metrics.Mean):
    def __init__(self,
                 name='hausdorff_distance',
                 data_format='channels_last'):
        super(HausdorffDistance, self).__init__(name=name)
        self.data_format = data_format

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Updates running mean of Hausdorff distances."""
        y_pred = pred_to_one_hot(y_pred, self.data_format)

        y_true = tf.reshape(y_true, shape=(1, -1))
        y_pred = tf.reshape(y_pred, shape=(1, -1))
        haus_dist = tf.maximum(
                        directed_hausdorff(y_true.numpy(), y_pred.numpy())[0],
                        directed_hausdorff(y_pred.numpy(), y_true.numpy())[0])
        return super(HausdorffDistance, self).update_state(
                            haus_dist, sample_weight=sample_weight)


class DiceCoefficient(tf.keras.metrics.Mean):
    """Implements dice coefficient for binary classification."""
    def __init__(self,
                 name='dice_coefficient',
                 data_format='channels_last',
                 eps=1.0):
        super(DiceCoefficient, self).__init__(name=name)
        self.data_format = data_format
        self.eps = eps

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Updates running mean of dice coefficient scores."""
        axis = (0, 1, 2, 3) if self.data_format == 'channels_last' else (0, 2, 3, 4)
        
        # Correct predictions will have 1, else 0.
        y_pred = pred_to_one_hot(y_pred, self.data_format)
        intersection = tf.reduce_sum(y_pred * y_true, axis=axis)

        pred = tf.reduce_sum(y_pred, axis=axis)
        true = tf.reduce_sum(y_true, axis=axis)

        dice_coeff = tf.reduce_mean((2.0 * intersection + self.eps) / (pred + true + self.eps))

        return super(DiceCoefficient, self).update_state(
                            dice_coeff, sample_weight=sample_weight)
