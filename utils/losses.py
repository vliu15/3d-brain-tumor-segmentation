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
        self.data_format = data_format

    def __call__(self, y_true, y_pred, sample_weight=None):
        axis = -1 if self.data_format == 'channels_last' else 1

        # Apply label smoothing if necessary.
        if self.smoothing:
            y_true = y_true * (1.0 - self.smoothing) + (1 - y_true) * self.smoothing / (OUT_CH - 1)

        return super(CrossEntropyLoss, self).__call__(y_true, y_pred, axis=axis)


class FocalLoss(tf.keras.losses.Loss):
    """Implementation of focal loss: https://arxiv.org/pdf/1708.02002.pdf."""
    def __init__(self,
                 smoothing=0.01,
                 gamma=2,
                 alpha=0.25,
                 name='focal_loss',
                 data_format='channels_last'):
        super(FocalLoss, self).__init__(name=name)
        assert alpha <= 1 and alpha >= 0
        self.smoothing = smoothing
        self.gamma = gamma
        self.alpha = alpha
        self.data_format = data_format

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Computes focal loss between predicted and true probabilities.
        
            Args:
                y_true: one-hot vector indicating true labels.
                y_pred: softmaxed predicted probabilities.
        """
        axis = -1 if self.data_format == 'channels_last' else 1

        # Apply label smoothing if necessary.
        if self.smoothing:
            y_true = y_true * (1.0 - self.smoothing) + (1 - y_true) * self.smoothing / (OUT_CH - 1)

        # Prepare probabilities (may be unnecessary, but Keras backend does this).
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=axis, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Apply p to y=1 and (1-p) to (y!=1) positions.
        y_pred = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # Apply self.alpha for class rebalancing.
        alpha = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)

        focus = (1 - y_pred) ** self.gamma
        return -tf.reduce_sum(alpha * focus * y_true * tf.math.log(y_pred))


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
