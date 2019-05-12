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
        self.init_lr = learning_rate
        self.n_epochs = float(n_epochs)
    
    def update_lr(self, epoch_num):
        """Allows external scheduling per epoch."""
        new_lr = self.init_lr * ((1.0 - epoch_num / self.n_epochs) ** 0.9)
        self._set_hyper('learning_rate', new_lr)
