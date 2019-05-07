import tensorflow as tf

def get_loss_func(x, y_vae, z_mean, z_var, eps=1e-8):

    N = tf.math.reduce_prod(x.shape)

    loss_L2 = tf.math.reduce_sum(tf.math.square(x * y_vae))
    loss_KL = tf.math.reduce_sum(z_mean * z_mean + z_var - tf.math.log(z_var) - 1) / N

    def loss_func(y, y_hat):
        y = tf.contrib.layers.flatten(y)
        y_hat = tf.contrib.layers.flatten(y_hat)

        numer = 2.0 * tf.math.reduce_sum(tf.math.abs(y * y_hat))
        denom = tf.math.reduce_sum(y * y) + tf.math.reduce_sum(y_hat * y_hat) + eps
        loss_dice = numer / denom

        return loss_dice + 0.1 * loss_L2 + 0.1 * loss_KL
