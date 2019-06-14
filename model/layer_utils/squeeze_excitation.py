"""Contains squeeze-excitation enhancement layer class."""
import tensorflow as tf


class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self,
                 reduction=2,
                 data_format='channels_last',
                 l2_scale=1e-5):
        """ Initializes one squeeze-and-excitation layer. Applies
            concurrent spatial and channel squeeze-and-excitation.
        
            References:
                - [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
                - [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
        """
        super(SqueezeExcitation, self).__init__()
        self.data_format = data_format
        self.reduction = reduction
        self.l2_scale = l2_scale

        self.squeeze = tf.keras.layers.GlobalAveragePooling3D(
                                data_format=data_format)
        self.spatial = tf.keras.layers.Conv3D(
                                filters=1,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_initializer='glorot_normal',
                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                                activation='sigmoid')

        self.scale_ch = tf.keras.layers.Multiply()
        self.scale_sp = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        channels = input_shape[-1] if self.data_format == 'channels_last' else input_shape[1]
        if channels % self.reduction != 0:
            raise ValueError(
                'Reduction ratio, {}, must be a factor of number of channels, {}.'
                .format(self.reduction, channels))

        self.dense_relu = tf.keras.layers.Dense(
                                units=channels // self.reduction,
                                kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_scale),
                                kernel_initializer='he_normal',
                                use_bias=False,
                                activation='relu')
        self.dense_sigmoid = tf.keras.layers.Dense(
                                units=channels,
                                kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_scale),
                                kernel_initializer='glorot_normal',
                                use_bias=False,
                                activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape(
                                (1, 1, 1, -1) if self.data_format == 'channels_last'
                                              else (-1, 1, 1, 1))

    def call(self, inputs, training=None):
        # Channel squeeze & excitation
        chse = self.squeeze(inputs)
        chse = self.dense_relu(chse)
        chse = self.dense_sigmoid(chse)
        chse = self.reshape(chse)
        chse = self.scale_ch([chse, inputs])

        # Spatial squeeze & excitation
        spse = self.spatial(inputs)
        spse = self.scale_sp([spse, inputs])

        return self.add([spse, chse])

    def get_config(self):
        config = super(SqueezeExcitation, self).get_config()
        config.update({'reduction': self.reduction,
                       'data_format': self.data_format,
                       'l2_scale': self.l2_scale})
        return config
