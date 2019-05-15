import tensorflow as tf


def get_myrnenko_loss_fn(x, y_vae, z_mean, z_var,
                         eps=1e-8, data_format='channels_last'):
    """Returns a Keras-compatible function that computes Myrnenko loss.
    
        Args:
            x: input to the model.
            y_vae: output from VAE branch.
            z_mean: mean of sampling distribution.
            z_var: variance of sampilng distribution.
            eps: small float to avoid division by 0.

        Returns:
            myrnenko_loss_fn: callable that returns full Myrnenko loss.
    """
    def myrnenko_loss_fn(y_true, y_pred):
        """Computes and returns Myrnenko loss.
        
            Args:
                y_true: labeled output.
                y_pred: predicted model output.

            Returns:
                Myrnenko loss.
        """
        axis = -1 if data_format == 'channels_last' else 2
        y_true = tf.reshape(y_true, [-1])
        n_channels = y_pred.shape[axis]
        loss_dice = 0.0

        for channel in range(n_channels):
            y_pred_ch = tf.reshape(y_pred[..., channel], [-1])

            numer = 2.0 * tf.math.reduce_sum(tf.math.abs(y_true * y_pred_ch))
            denom = tf.math.reduce_sum(y_true ** 2) + tf.math.reduce_sum(y_pred_ch ** 2) + eps
            loss_dice += numer / denom

        return loss_dice + 0.1 * loss_l2 + 0.1 * loss_kl

    N = tf.cast(tf.math.reduce_prod(x.shape), tf.float32)

    loss_l2 = tf.math.reduce_sum((x - y_vae) ** 2)
    loss_kl = tf.math.reduce_sum(
                z_mean ** 2 + z_var - tf.math.log(z_var) - 1.0) / N
    
    return myrnenko_loss_fn


def compute_myrnenko_loss(x, y_true, y_pred, y_vae, z_mean, z_var,
                          eps=1e-8, data_format='channels_last'):
    """Computes and returns Myrnenko loss.
    
        Args:
            x: input to the model.
            y_true: labeled output.
            y_pred: predicted model output.
            y_vae: output from VAE branch.
            z_mean: mean of sampling distribution.
            z_var: variance of sampilng distribution.
            eps: small float to avoid division by 0.

        Returns:
            Myrnenko loss.
    """
    N = tf.cast(tf.math.reduce_prod(x.shape), tf.float32)

    loss_l2 = tf.math.reduce_sum((x - y_vae) ** 2)
    loss_kl = tf.math.reduce_sum(
                z_mean ** 2 + z_var - tf.math.log(z_var) - 1.0) / N

    axis = -1 if data_format == 'channels_last' else 2
    y_true = tf.reshape(y_true, [-1])
    n_channels = y_pred.shape[axis]
    loss_dice = 0.0

    for channel in range(n_channels):
        y_pred_ch = tf.reshape(y_pred[..., channel], [-1])

        numer = 2.0 * tf.math.reduce_sum(tf.math.abs(y_true * y_pred_ch))
        denom = tf.math.reduce_sum(y_true ** 2) + tf.math.reduce_sum(y_pred_ch ** 2) + eps
        loss_dice += numer / denom

    return loss_dice + 0.1 * loss_l2 + 0.1 * loss_kl