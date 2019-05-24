import tensorflow as tf

from utils.constants import *


class L2Loss(tf.keras.losses.Loss):
    def __init__(self, name='l2_loss'):
        super(L2Loss, self).__init__(name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return tf.reduce_sum((y_true - y_pred) ** 2)


class KullbackLeiblerLoss(tf.keras.losses.Loss):
    def __init__(self, name='kl_loss'):
        super(KullbackLeiblerLoss, self).__init__(name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return tf.keras.losses.kld(y_true, y_pred)


class CrossEntropyLoss(tf.keras.losses.CategoricalCrossentropy):
    def __init__(self,
                 smoothing=0.01,
                 name='cross_entropy_loss',
                 data_format='channels_last'):
        super(CrossEntropyLoss, self).__init__(name=name)
        self.smoothing = smoothing
        self.axis = -1 if data_format == 'channels_last' else 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        # Apply label smoothing if necessary.
        if self.smoothing:
            y_true -= self.smoothing * tf.cast(tf.cast(y_true > 0, tf.bool), tf.float32)
            self.smoothing /= (OUT_CH - 1)
            y_true += self.smoothing * tf.cast(tf.cast(y_true < 1, tf.bool), tf.float32)

        ce_loss = super(CrossEntropyLoss, self).__call__(y_true, y_pred, axis=self.axis)
        return tf.reduce_sum(ce_loss)


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self,
                 smoothing=0.01,
                 gamma=2,
                 alpha=0.25,
                 name='focal_loss',
                 data_format='channels_last'):
        super(FocalLoss, self).__init__(name=name)
        self.smoothing = smoothing
        self.gamma = gamma
        self.alpha = alpha
        self.axis = -1 if data_format == 'channels_last' else 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        # Apply label smoothing if necessary.
        if self.smoothing:
            y_true -= self.smoothing * tf.cast(tf.cast(y_true > 0, tf.bool), tf.float32)
            self.smoothing /= (OUT_CH - 1)
            y_true += self.smoothing * tf.cast(tf.cast(y_true < 1, tf.bool), tf.float32)

        # Normalize probabilities.
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=self.axis, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        focus = (1 - y_pred) ** self.gamma

        return -self.alpha * tf.reduce_sum(focus * y_true * tf.math.log(y_pred))


class GeneralizedDiceLoss(tf.keras.losses.Loss):
    def __init__(self, name='generalized_dice_loss', data_format='channels_last', eps=1e-8):
        super(GeneralizedDiceLoss, self).__init__(name=name)
        self.axis = (0, 1, 2, 3) if data_format == 'channels_last' else (0, 2, 3, 4)
        self.eps = eps

    def __call__(self, y_true, y_pred, sample_weight=None):
        # Compute weight invariance.
        w = 1.0 / (tf.reduce_sum(y_true, axis=self.axis) ** 2 + self.eps)

        # Compute generalized dice loss.
        numer = tf.reduce_sum(w * tf.reduce_sum(y_true * y_pred, axis=self.axis)) + self.eps
        denom = tf.reduce_sum(w * tf.reduce_sum(y_true + y_pred, axis=self.axis)) + self.eps

        return 1.0 - 2.0 * numer / denom


class SensitivitySpecificityLoss(tf.keras.losses.Loss):
    def __init__(self,
                 lamb=0.05,
                 name='sensitivity_specificity_loss',
                 eps=1e-8,
                 data_format='channels_last'):
        super(SensitivitySpecificityLoss, self).__init__(name=name)
        self.lamb = lamb
        self.axis = (0, 1, 2, 3) if data_format == 'channels_last' else (0, 2, 3, 4)

    def __call__(self, y_true, y_pred, sample_weight=None):
        # Compute sensitivity.
        sensitivity_numer = tf.reduce_sum(y_true * (y_true - y_pred) ** 2, axis=self.axis) + self.eps
        sensitivity_denom = tf.reduce_sum(y_true, axis=self.axis) + self.eps
        sensivitity = tf.reduce_mean(sensitivity_numer / sensitivity_denom)

        # Compute specificity.
        specificity_numer = tf.reduce_sum((1 - y_true) * (y_true - y_pred) ** 2, axis=self.axis) + self.eps
        specificity_denom = tf.reduce_sum((1 - y_true), axis=self.axis) + self.eps
        specificity = tf.reduce_mean(specificity_numer / specificity_denom)

        return lamb * sensitivity + (1 - lamb) * specificity


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self,
                 name='custom_loss',
                 data_format='channels_last'):
        super(CustomLoss, self).__init__(name=name)
        self.focal_loss = FocalLoss(data_format=data_format)
        self.generalized_dice_loss = GeneralizedDiceLoss(data_format=data_format)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return FL_WEIGHT * self.focal_loss(y_true, y_pred) + \
                GDL_WEIGHT * self.generalized_dice_loss(y_true, y_pred)
