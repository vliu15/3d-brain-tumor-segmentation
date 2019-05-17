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
            y_true: segmentation containing one-hot labels at each voxel.
                if channels_last: shape=(1, H, W, D, 4)
                if channels_first: shape=(1, 4, H ,W, D)
            y_pred: model output containing probabilities for each class
                per channel (each class is a channel).
                if channels_last: shape=(1, H, W, D, 4)
                if channels_first: shape=(1, 4, H, W, D)
            data_format: whether data is in the format `channels_last`
                or `channels_first`.

        Returns:
            Generalized dice loss.
    """
    axis = (0, 1, 2, 3) if data_format == 'channels_last' else (0, 2, 3, 4)

    # Compute weight invariance.
    w = 1.0 / (tf.reduce_sum(y_true, axis=axis) ** 2 + eps)

    # Compute generalized dice loss.
    numer = tf.reduce_sum(w * tf.reduce_sum(y_true * y_pred, axis=axis)) + eps
    denom = tf.reduce_sum(w * tf.reduce_sum(y_true + y_pred, axis=axis)) + eps

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
            Weighted sensitivity-specificity (SS) loss.
    """
    axis = (0, 1, 2, 3) if data_format == 'channels_last' else (0, 2, 3, 4)

    # Compute sensitivity.
    sensitivity_numer = tf.reduce_sum(y_true * (y_true - y_pred) ** 2, axis=axis) + eps
    sensitivity_denom = tf.reduce_sum(y_true, axis=axis) + eps
    sensivitity = tf.reduce_mean(sensitivity_numer / sensitivity_denom)

    # Compute specificity.
    specificity_numer = tf.reduce_sum((1 - y_true) * (y_true - y_pred) ** 2, axis=axis) + eps
    specificity_denom = tf.reduce_sum((1 - y_true), axis=axis) + eps
    specificity = tf.reduce_mean(specificity_numer / specificity_denom)

    return lamb * sensitivity + (1 - lamb) * specificity


def myrnenko_loss(x, y_true, y_pred, y_vae, z_mean, z_logvar,
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
            data_format: whether data is in the format `channels_last`
                or `channels_first`.

        Returns:
            Myrnenko loss.
    """
    n_voxels = tf.cast(tf.reduce_prod(x.shape), tf.float32)

    return 0.1 * l2_loss(x, y_vae) + \
           0.1 * kullbach_liebler_loss(z_mean, z_logvar, n_voxels) + \
           generalized_dice_loss(y_true, y_pred, data_format=data_format)


def tunable_loss(x, y_true, y_pred, y_vae, z_mean, z_logvar,
                        eps=1e-8, lamb=0.05, data_format='channels_last'):
    """Computes and returns a custom weighted loss function.
    
        Args:
            x: input to the model.
            y_true: labeled output.
            y_pred: predicted model output.
            y_vae: output from VAE branch.
            z_mean: mean of sampling distribution.
            z_logvar: log variance of sampilng distribution.
            lamb: weight of sensitivity in SS loss.
            eps: small float to avoid division by 0.
            data_format: whether data is in the format `channels_last`
                or `channels_first`.

        Returns:
            Custom weighted loss.
    """
    n_voxels = tf.cast(tf.math.reduce_prod(x.shape), tf.float32)

    return GDL_WEIGHT * generalized_dice_loss(
                            y_true, y_pred, data_format=data_format) + \
           SS_WEIGHT * sensitivity_specificity_loss(
                            y_true, y_pred, lamb=lamb, data_format=data_format) + \
           KL_WEIGHT * kullbach_liebler_loss(z_mean, z_logvar, n_voxels) + \
           L2_WEIGHT * l2_loss(x, y_vae)
