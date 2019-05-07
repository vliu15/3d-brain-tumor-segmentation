import tensorflow as tf

# TODO: implement this class
class VariationalAutoEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 groups=8,
                 kernel_size=3,
                 kernel_regularizer=None):
        super(VariationalAutoEncoder, self).__init__()
        pass

    def call(self, x):
        pass