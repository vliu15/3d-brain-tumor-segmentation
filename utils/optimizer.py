"""Contains learning rate schedulers."""
import tensorflow as tf


class ScheduledAdam(tf.keras.optimizers.Adam):
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
        super(ScheduledAdam, self).__init__(
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
        """Allows external manual scheduling per epoch."""
        new_lr = self.init_lr * ((1.0 - epoch / self.n_epochs) ** 0.9)
        self._set_hyper('learning_rate', new_lr)


class Scheduler(object):
    """Scheduler compatible with tf.keras.callbacks.LearningRateScheduler."""
    def __init__(self,
                 n_epochs=300,
                 learning_rate=1e-4):
        self.n_epochs = n_epochs
        self.init_lr = learning_rate

    def __call__(self, epoch, lr):
        """Function automatically called after initialization."""
        return self.init_lr * ((1.0 - epoch / self.n_epochs) ** 0.9)
