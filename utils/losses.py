"""Contains custom loss classes."""
import tensorflow as tf

from utils.constants import *


class L2Loss(tf.keras.losses.Loss):
    def __init__(self, name='l2_loss', **kwargs):
        super(L2Loss, self).__init__(**kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return tf.reduce_mean((y_true - y_pred) ** 2)


class KullbackLeiblerLoss(tf.keras.losses.Loss):
    def __init__(self, name='kl_loss', **kwargs):
        super(KullbackLeiblerLoss, self).__init__(**kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return tf.keras.losses.kld(y_true, y_pred)


class DiceLoss(tf.keras.losses.Loss):
    """Implementation of binary dice loss per channel."""
    def __init__(self,
                 name='dice_loss',
                 eps=1.0,
                 data_format='channels_last',
                 **kwargs):
        """Initializes DiceLoss class and sets attributes needed to calculate loss.

            Args:
                name: str, optional
                    name of this loss class (for tf.Keras.losses.Loss).
                eps: float, optional
                    small number to prevent division by / with 0's.
                data_format: str, optional
                    format of the data, for determining the axis to compute loss over.
        """
        super(DiceLoss, self).__init__(**kwargs)
        assert eps <= 1 and eps > 0, '`eps` needs to be in the range (0, 1].'
        self.data_format = data_format
        self.eps = eps

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Computes dice loss.

            Args:
                y_true: one-hot vector indicating true labels.
                y_pred: predicted probabilities (softmax or sigmoid).
            Returns:
                dice_loss: dice loss, shape=()
        """
        axis = (0, 1, 2, 3) if self.data_format == 'channels_last' else (0, 2, 3, 4)
        
        # Correct predictions will have 1, else 0.
        intersection = tf.reduce_sum(y_pred * y_true, axis=axis)

        pred = tf.reduce_sum(y_pred ** 2, axis=axis)
        true = tf.reduce_sum(y_true ** 2, axis=axis)

        dice_loss = tf.reduce_sum(1.0 - (2.0 * intersection + self.eps) / (pred + true + self.eps))
        return dice_loss


class CrossEntropyLoss(tf.keras.losses.Loss):
    """Implementation of balanced cross entropy loss."""
    def __init__(self,
                 smoothing=1e-4,
                 alpha=0.0,
                 name='cross_entropy_loss',
                 data_format='channels_last',
                 **kwargs):
        """Initializes CrossEntropyLoss class and sets attributes needed to calculate loss.
        
            Args:
                smoothing: float, optional
                    amount of label smoothing to apply. Set to 0 for no smoothing.
                alpha: float, optional
                    amount of balance to apply (as in balanced cross entropy). Set
                    to 0 for regular (unbalanced) cross entropy.
                name: str, optional
                    name of this loss class (for tf.Keras.losses.Loss).
                data_format: str, optional
                    format of the data, for determining the axis to compute loss over.
        """
        super(CrossEntropyLoss, self).__init__(**kwargs)
        assert smoothing <= 1 and smoothing >= 0, '`smoothing` needs to be in the range [0, 1].'
        assert alpha <= 1 and alpha >= 0, '`alpha` needs to be in the range [0, 1].'
        self.smoothing = smoothing
        self.alpha = alpha
        self.data_format = data_format

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Computes alpha-balanced cross entropy loss.

            Args:
                y_true: one-hot vector indicating true labels.
                y_pred: predicted probabilities (softmax or sigmoid).
            Returns:
                ce_loss: cross entropy loss.
        """
        axis = -1 if self.data_format == 'channels_last' else 1

        # Apply label smoothing if necessary.
        if self.smoothing:
            y_true = y_true * (1.0 - self.smoothing) + (1 - y_true) * self.smoothing

        # Apply p to y=1 and (1-p) to (y!=1) positions.
        y_pred = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # Apply self.alpha for class rebalancing.
        alpha = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        ce_loss = -tf.reduce_sum(alpha * tf.math.log(y_pred))

        return ce_loss


class FocalLoss(tf.keras.losses.Loss):
    """Implementation of focal loss: https://arxiv.org/pdf/1708.02002.pdf."""
    def __init__(self,
                 smoothing=1e-4,
                 gamma=2,
                 alpha=0.25,
                 name='focal_loss',
                 data_format='channels_last',
                 **kwargs):
        """Initializes FocalLoss class and sets attributes needed in loss calculation.

            Args:
                smoothing: float, optional
                    amount of label smoothing to apply. Set to 0 for no smoothing.
                gamma: int, optional
                    amount of focal smoothing to apply. Set to 0 for regular
                    balanced cross entropy.
                alpha: float, optional
                    amount of balance to apply (as in balanced cross entropy). Set
                    to 0 for regular (unbalanced) cross entropy.
                name: str, optional
                    name of this loss class (for tf.Keras.losses.Loss).
                data_format: str, optional
                    format of the data, for determining the axis to compute loss over.
        """
        super(FocalLoss, self).__init__(**kwargs)
        assert smoothing <= 1 and smoothing >= 0, '`smoothing` needs to be in the range [0, 1].'
        assert alpha <= 1 and alpha >= 0, '`alpha` needs to be in the range [0, 1].'
        assert gamma >= 0, '`gamma` needs to be a non-negative integer.'
        self.smoothing = smoothing
        self.gamma = gamma
        self.alpha = alpha
        self.data_format = data_format

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Computes focal loss between predicted and true probabilities.
        
            Args:
                y_true: one-hot vector indicating true labels.
                y_pred: predicted probabilities (softmax or sigmoid).
            Returns:
                f_loss: focal loss.
        """
        axis = -1 if self.data_format == 'channels_last' else 1

        # Apply label smoothing if necessary.
        if self.smoothing > 0:
            y_true = y_true * (1.0 - self.smoothing) + (1.0 - y_true) * self.smoothing

        # Clip values to avoid NaN calculations.
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Apply p to y=1 and (1-p) to (y!=1) positions.
        y_pred = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # Apply self.alpha for class rebalancing.
        alpha = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)

        focus = (1 - y_pred) ** self.gamma

        loss = -tf.reduce_sum(alpha * focus * tf.math.log(y_pred), axis=axis)
        return tf.reduce_mean(loss)


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self,
                 name='custom_loss',
                 eps=1.0,
                 data_format='channels_last',
                 **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.l2_loss = L2Loss(name='l2', **kwargs)
        self.kl_loss = KullbackLeiblerLoss(name='kl', **kwargs)
        self.dec_loss = DiceLoss(name='dice', **kwargs)

    def __call__(self, x, y_true, y_pred, y_vae, z_mean, z_logvar, sample_weight=None):
        n = tf.cast(tf.reduce_prod(x.shape), tf.float32)
        return self.dec_loss(y_true, y_pred) + \
                0.1 * self.l2_loss(x, y_vae) + \
                0.1 / n * self.kl_distribution_loss(z_mean, z_logvar)

    @staticmethod
    def kl_distribution_loss(mean, logvar):
        return tf.reduce_sum(mean ** 2 + tf.math.exp(logvar) - logvar - 1)
