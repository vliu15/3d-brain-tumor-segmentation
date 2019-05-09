import tensorflow as tf


def get_myrnenko_loss_fn(x, y_vae, z_mean, z_var, eps=1e-8):
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
        y_true = tf.contrib.layers.flatten(y_true)
        y_pred = tf.contrib.layers.flatten(y_pred)

        numer = 2.0 * tf.math.reduce_sum(tf.math.abs(y * y_pred))
        denom = tf.math.reduce_sum(y ** 2) + tf.math.reduce_sum(y_pred ** 2) + eps
        loss_dice = numer / denom

        return loss_dice + 0.1 * loss_L2 + 0.1 * loss_KL

    N = tf.math.reduce_prod(x.shape)

    loss_l2 = tf.keras.losses.mse(x, y_vae)
    loss_kl = tf.math.reduce_sum(
                z_mean * z_mean + z_var - tf.math.log(z_var) - 1) / N
    
    return myrnenko_loss_fn


def compute_myrnenko_loss(x, y_true, y_pred, y_vae, z_mean, z_var, eps=1e-8):
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
    N = tf.math.reduce_prod(x.shape)

    loss_l2 = tf.keras.losses.mse(x, y_vae)
    loss_kl = tf.math.reduce_sum(
                z_mean * z_mean + z_var - tf.math.log(z_var) - 1) / N

    y_true = tf.contrib.layers.flatten(y_true)
    y_pred = tf.contrib.layers.flatten(y_pred)

    numer = 2.0 * tf.math.reduce_sum(tf.math.abs(y * y_pred))
    denom = tf.math.reduce_sum(y ** 2) + tf.math.reduce_sum(y_pred ** 2) + eps
    loss_dice = numer / denom

    return loss_dice + 0.1 * loss_L2 + 0.1 * loss_KL