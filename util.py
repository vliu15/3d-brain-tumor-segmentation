"""Contains custom loss, dice coefficient, and optimizer classes."""
import tensorflow as tf


class DiceVAELoss(object):
    """Implements custom dice-VAE loss."""
    def __init__(self,
                 name='custom_loss',
                 data_format='channels_last',
                 **kwargs):
        self.axis = (0, 1, 2, 3) if data_format == 'channels_last' else (0, 2, 3, 4)

    def __call__(self, x, y, y_pred, y_vae, z_mean, z_logvar, sample_weight=None):
        l2_loss = tf.reduce_mean((x - y_vae) ** 2)
        kld_loss = tf.reduce_mean(z_mean ** 2 + tf.math.exp(z_logvar) - z_logvar - 1.0)

        # Calculate dice loss.
        intersection = tf.reduce_sum(y_pred * y, axis=self.axis)
        pred = tf.reduce_sum(y_pred ** 2, axis=self.axis)
        true = tf.reduce_sum(y ** 2, axis=self.axis)

        dice_loss = tf.reduce_mean(1.0 - (2.0 * intersection + 1.0) / (pred + true + 1.0))

        return dice_loss + 0.1*l2_loss + 0.1*kld_loss


class DiceCoefficient(object):
    """Implements dice coefficient for binary classification."""
    def __init__(self,
                 name='dice_coefficient',
                 data_format='channels_last'):
        self.name = name
        self.data_format = data_format

    def __call__(self, y_true, y_pred):
        dice_axes = (0, 1, 2) if self.data_format == 'channels_last' else (0, 2, 3, 4)
        onehot_axis = -1 if self.data_format == 'channels_last' else 1

        # Mask out values that correspond to values < 0.5.
        mask = tf.reduce_max(y_pred, axis=onehot_axis, keepdims=True)
        mask = tf.cast(mask > 0.5, tf.float32)

        # Create one-hot encoding of predictions.
        out_ch = y_pred.shape[onehot_axis]
        y_pred = tf.argmax(y_pred, axis=onehot_axis, output_type=tf.int32)
        y_pred = tf.one_hot(y_pred, out_ch, axis=onehot_axis, dtype=tf.float32)
        y_pred *= mask

        # Compute dice score.
        intersection = tf.reduce_sum(y_pred * y_true, axis=dice_axes)
        pred = tf.reduce_sum(y_pred, axis=dice_axes)
        true = tf.reduce_sum(y_true, axis=dice_axes)

        macroavg = tf.reduce_mean((2.0 * intersection + 1.0) / (pred + true + 1.0))
        microavg = tf.reduce_sum(y_pred * y_true) / (tf.reduce_sum(y_pred) + tf.reduce_sum(y_true))

        return macroavg, microavg


class ScheduledOptim(tf.keras.optimizers.Adam):
    """Adam optimizer that allows for scheduling every epoch."""
    def __init__(self,
                 learning_rate=1e-4,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 name='Adam',
                 n_epochs=300,
                 **kwargs):
        super(ScheduledOptim, self).__init__(
                                        learning_rate=learning_rate,
                                        beta_1=beta_1,
                                        beta_2=beta_2,
                                        epsilon=epsilon,
                                        amsgrad=amsgrad,
                                        name=name,
                                        **kwargs)
        self.init_lr = tf.constant(learning_rate, dtype=tf.float32)
        self.n_epochs = float(n_epochs)

    def __call__(self, epoch):
        new_lr = self.init_lr * ((1.0 - epoch / self.n_epochs) ** 0.9)
        self._set_hyper('learning_rate', new_lr)
