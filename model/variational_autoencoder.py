import tensorflow as tf

# TODO: implement this class
class VariationalAutoEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 groups=8,
                 kernel_size=3,
                 kernel_regularizer=None):
        super(VariationalAutoEncoder, self).__init__()

    def call(self, x):
        y_vae = None
        z_mean = None
        z_var = None
        return y_vae, z_mean, z_var
        