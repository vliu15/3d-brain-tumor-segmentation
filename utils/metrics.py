import tensorflow as tf

from utils.constants import *


class DiceCoefficient(tf.keras.metrics.Mean):
    def __init__(self,
                 name='dice_coefficient',
                 eps=1e-8,
                 data_format='channels_last'):
        super(DiceCoefficient, self).__init__(name=name)
        self.data_format = data_format
        self.eps = eps

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract predictions at each voxel.
        axis = -1 if self.data_format == 'channels_last' else 1
        y_pred = tf.argmax(y_pred, axis=axis, output_type=tf.int32)

        # Turn into one-hot encodings per voxel.
        y_pred = tf.one_hot(y_pred, len(LABELS), axis=axis, dtype=tf.float32)

        # Correct predictions will have 1, else 0.
        intersection = y_pred * y_true

        # Sum up per channel.
        axis = (0, 1, 2, 3) if self.data_format == 'channels_last' else (0, 2, 3, 4)
        intersection = tf.reduce_sum(intersection, axis=axis)

        # Count total number of each label.
        true_voxels = tf.reduce_sum(y_true, axis=axis)
        pred_voxels = tf.reduce_sum(y_pred, axis=axis)

        # Dice coefficients per channel.
        dice_coeff = (2.0 * intersection + self.eps) / (true_voxels + pred_voxels + self.eps)
        dice_coeff = tf.reduce_mean(dice_coeff)

        return super(DiceCoefficient, self).update_state(
                            dice_coeff, sample_weight=sample_weight)


class TumorVoxelAccuracy(tf.keras.metrics.CategoricalAccuracy):
    def __init__(self,
                 name='tumor_voxel_categorical_accuracy',
                 dtype=tf.float32,
                 data_format='channels_last'):
        super(TumorVoxelAccuracy, self).__init__(
                                            name=name,
                                            dtype=dtype)
        self.data_format = data_format

    def update_state(self,  y_true, y_pred, sample_weight=None):
        if self.data_format == 'channels_last':
            y_true = y_true[:, :, :, :, 1:]
            y_pred = y_pred[:, :, :, :, 1:]
        elif self.data_format == 'channels_first':
            y_true = y_true[:, 1:, :, :, :]
            y_pred = y_pred[:, 1:, :, :, :]

        return super(TumorVoxelAccuracy, self).update_state(
                        y_true, y_pred, sample_weight=sample_weight)


class TumorVoxelPrecision(tf.keras.metrics.Precision):
    def __init__(self,
                 thresholds=0.5,
                 name='tumor_voxel_precision',
                 dtype=tf.float32,
                 data_format='channels_last'):
        super(TumorVoxelPrecision, self).__init__(
                                            thresholds=thresholds,
                                            name=name,
                                            dtype=dtype)
        self.data_format = data_format

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.data_format == 'channels_last':
            y_true = y_true[:, :, :, :, 1:]
            y_pred = y_pred[:, :, :, :, 1:]
        elif self.data_format == 'channels_first':
            y_true = y_true[:, 1:, :, :, :]
            y_pred = y_pred[:, 1:, :, :, :]

        return super(TumorVoxelPrecision, self).update_state(
                        y_true, y_pred, sample_weight=sample_weight)


class TumorVoxelRecall(tf.keras.metrics.Recall):
    def __init__(self,
                 thresholds=0.5,
                 name='tumor_voxel_recall',
                 dtype=tf.float32,
                 data_format='channels_last'):
        super(TumorVoxelRecall, self).__init__(
                                            thresholds=thresholds,
                                            name=name,
                                            dtype=dtype)
        self.data_format = data_format

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.data_format == 'channels_last':
            y_true = y_true[:, :, :, :, 1:]
            y_pred = y_pred[:, :, :, :, 1:]
        elif self.data_format == 'channels_first':
            y_true = y_true[:, 1:, :, :, :]
            y_pred = y_pred[:, 1:, :, :, :]

        return super(TumorVoxelRecall, self).update_state(
                        y_true, y_pred, sample_weight=sample_weight)
