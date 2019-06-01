"""Contains squeeze-excitation enhancement layer class."""
import tensorflow as tf


class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self,
                 reduction=2,
                 data_format='channels_last',
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                 kernel_initializer='he_normal'):
        """Initializes one squeeze-excitation layer.
        
            See https://arxiv.org/pdf/1709.01507.pdf for details.
            This layer is meant to be applied to each residual at
            the beginning of each ConvBlock (which are ResNet blocks).

            Args:
                reduction: int, optional
                    Reduction ratio for excitation size.
                data_format: str, optional
                    The format of the input data. Must be either 'channels_last'
                    or 'channels_first'. Defaults to `channels_last` for CPU
                    development. 'channels_first is used in the paper.
        """
        super(SqueezeExcitation, self).__init__()
        self.data_format = data_format
        self.reduction = reduction
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

        self.global_pool = tf.keras.layers.GlobalAveragePooling3D(
                                data_format=data_format)

        # Initialize reduction and excitation in self.build().

    def build(self, input_shape):
        """Builds dense layers at call-time."""
        channels = input_shape[-1] if self.data_format == 'channels_last' else input_shape[1]
        if channels % self.reduction != 0:
            raise ValueError(
                'Reduction ratio, {}, must be a factor of number of channels, {}.'
                .format(self.reduction, channels))

        self.dense_relu = tf.keras.layers.Dense(channels / self.reduction,
                                                kernel_regularizer=self.kernel_regularizer,
                                                kernel_initializer=self.kernel_initializer,
                                                activation='relu')
        self.dense_sigmoid = tf.keras.layers.Dense(channels,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   kernel_initializer='glorot_normal',
                                                   activation='sigmoid')

    def call(self, inputs, training=False):
        """Returns the forward pass of one SqueezeExcitationLayer.
        
            { GlobalAveragePool -> Dense+ReLu -> Dense+Sigmoid }
        """
        inputs = self.global_pool(inputs)
        inputs = self.dense_relu(inputs)
        inputs = self.dense_sigmoid(inputs)
        return inputs

    def get_config(self):
        config = super(SqueezeExcitation, self).get_config()
        config.update({'data_format': self.data_format,
                       'reduction': self.reduction,
                       'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
                       'kernel_initializer': self.kernel_initializer})
        return config
