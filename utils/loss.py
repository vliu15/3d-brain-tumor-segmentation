import tensorflow as tf

from utils.constants import *


def l2_loss(x, y_vae):
    """Returns the reconstruction (l2) loss between autoencoded and original image."""
    return tf.math.reduce_sum((x - y_vae) ** 2)


def kullbach_liebler_loss(z_mean, z_logvar, n_voxels):
    """Returns KL divergence loss between learned and standard normal distributions."""
    return 1.0 / n_voxels * tf.math.reduce_sum(
                z_mean ** 2 + tf.math.exp(z_logvar) - z_logvar - 1.0)


def generalized_dice_loss(y_true, y_pred, data_format='channels_last', eps=1e-8):
    """Returns generalized dice loss between target and prediction.
    
        Args:
            y_true: segmentation containing class labels at each voxel.
                if channels_last: shape=(1, H, W, D, 1)
                if channels_first: shape=(1, 1, H ,W, D)
            y_pred: model output containing probabilities for each class
                per channel (each class is a channel).
                if channels_last: shape=(1, H, W, D, 3)
                if channels_first: shape=(1, 3, H, W, D)
            data_format: whether data is in the format `channels_last`
                or `channels_first`.

        Returns:
            generalized dice loss.
    """
    # Expand channels to correspond to per class.
    y_true = tf.broadcast_to(y_true, shape=y_pred.shape)

    if data_format == 'channels_last':
        # Create binary mask for each label and corresponding channel.
        labels = tf.reshape(tf.convert_to_tensor(LABELS, dtype=tf.float32), shape=(1, 1, 1, 1, -1))
        labels = tf.broadcast_to(labels, y_true.shape)
        y_true = 1.0 - tf.dtypes.cast(tf.dtypes.cast(y_true - labels, tf.bool), tf.float32)

        # Compute weight invariance.
        w = 1.0 / (tf.reduce_sum(y_true, axis=(0, 1, 2, 3)) ** 2 + eps)

        # Compute generalized dice loss.
        numer = tf.reduce_sum(w * tf.reduce_sum(y_true * y_pred, axis=(0, 1, 2, 3))) + eps
        denom = tf.reduce_sum(w * tf.reduce_sum(y_true + y_pred, axis=(0, 1, 2, 3))) + eps

    elif data_format == 'channels_first':
        # Create binary mask for each label and corresponding channel.
        labels = tf.reshape(tf.convert_to_tensor(LABELS, dtype=tf.float32), shape=(1, -1, 1, 1, 1))
        labels = tf.broadcast_to(labels, y_true.shape)
        y_true = 1.0 - tf.dtypes.cast(tf.dtypes.cast(y_true - labels, tf.bool), tf.float32)
    
        # Compute weight invariance.
        w = 1.0 / (tf.reduce_sum(y_true, axis=(0, 2, 3, 4)) ** 2 + eps)

        # Compute generalized dice loss.
        numer = tf.reduce_sum(w * tf.reduce_sum(y_true * y_pred, axis=(0, 2, 3, 4))) + eps
        denom = tf.reduce_sum(w * tf.reduce_sum(y_true + y_pred, axis=(0, 2, 3, 4))) + eps

    return 1.0 - 2.0 * numer / denom


def sensitivity_specificity_loss(y_true, y_pred, lamb=0.05,
                                 data_format='channels_last', eps=1e-8):
    """Returns sensitivity-specificity loss between target and prediction.
    
        Args:
            y_true: segmentation containing class labels at each voxel.
                if channels_last: shape=(1, H, W, D, 1)
                if channels_first: shape=(1, 1, H ,W, D)
            y_pred: model output containing probabilities for each class
                per channel (each class is a channel).
                if channels_last: shape=(1, H, W, D, OUT_CH)
                if channels_first: shape=(1, OUT_CH, H, W, D)
            lamb: weight of sensitivity, (1 - lamb) is weight of specificity.
            data_format: whether data is in the format `channels_last`
                or `channels_first`.
            eps: small float to avoid division by 0.

        Returns:
            weighted sensitivity-specificity loss.
    """
    # Expand channels to correspond to per class.
    y_true = tf.broadcast_to(y_true, shape=y_pred.shape)

    if data_format == 'channels_last':
        # Create binary mask for each label and corresponding channel.
        labels = tf.reshape(tf.convert_to_tensor(LABELS, dtype=tf.float32), shape=(1, 1, 1, 1, -1))
        labels = tf.broadcast_to(labels, y_true.shape)
        y_true = 1.0 - tf.dtypes.cast(tf.dtypes.cast(y_true - labels, tf.bool), tf.float32)

        # Compute sensitivity.
        sensitivity_numer = tf.reduce_sum(y_true * (y_true - y_pred) ** 2, axis=(0, 1, 2, 3))
        sensitivity_demon = tf.reduce_sum(y_true, axis=(0, 1, 2, 3)) + eps

        # Compute specificity.
        specificity_numer = tf.reduce_sum((1 - y_true) * (y_true - y_pred) ** 2, axis=(0, 1, 2, 3))
        specificity_denom = tf.reduce_sum((1 - y_true), axis=(0, 1, 2, 3)) + eps

    elif data_format == 'channels_first':
        # Create binary mask for each label and corresponding channel.
        labels = tf.reshape(tf.convert_to_tensor(LABELS, dtype=tf.float32), shape=(1, -1, 1, 1, 1))
        labels = tf.broadcast_to(labels, y_true.shape)
        y_true = 1.0 - tf.dtypes.cast(tf.dtypes.cast(y_true - labels, tf.bool), tf.float32)

        # Compute sensitivity.
        sensitivity_numer = tf.reduce_sum(y_true * (y_true - y_pred) ** 2, axis=(0, 2, 3, 4))
        sensitivity_demon = tf.reduce_sum(y_true, axis=(0, 2, 3, 4)) + eps

        # Compute specificity.
        specificity_numer = tf.reduce_sum((1 - y_true) * (y_true - y_pred) ** 2, axis=(0, 2, 3, 4))
        specificity_denom = tf.reduce_sum((1 - y_true), axis=(0, 2, 3, 4)) + eps

    sensivitity = tf.reduce_sum(sensitivity_numer / sensitivity_demon)
    specificity = tf.reduce_sum(sensitivity_numer / sensitivity_demon)

    return lamb * sensitivity + (1 - lamb) * specificity


def compute_myrnenko_loss(x, y_true, y_pred, y_vae, z_mean, z_logvar,
                          eps=1e-8, data_format='channels_last'):
    """Computes and returns Myrnenko loss.
    
        Args:
            x: input to the model.
            y_true: labeled output.
            y_pred: predicted model output.
            y_vae: output from VAE branch.
            z_mean: mean of sampling distribution.
            z_logvar: log variance of sampilng distribution.
            eps: small float to avoid division by 0.

        Returns:
            Myrnenko loss.
    """
    n_voxels = tf.cast(tf.math.reduce_prod(x.shape), tf.float32)

    return 0.1 * l2_loss(x, y_vae) + \
           0.1 * kullbach_liebler_loss(z_mean, z_logvar, n_voxels) + \
           generalized_dice_loss(y_true, y_pred, data_format=data_format)
