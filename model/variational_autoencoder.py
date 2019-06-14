"""Contains custom variational autoencoder class."""
import tensorflow as tf

from model.layer_utils.group_norm import GroupNormalization
from model.layer_utils.downsample import get_downsampling
from model.layer_utils.upsample import get_upsampling
from model.resnet_block import ConvBlock, ConvLayer
from utils.constants import *


def sample(inputs):
    """Samples from the Gaussian given by mean and variance."""
    z_mean, z_logvar = inputs
    eps = tf.random.normal(shape=z_mean.shape, dtype=tf.float32)
    return z_mean + tf.math.exp(0.5 * z_logvar) * eps


class VariationalAutoencoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 groups=8,
                 reduction=2,
                 l2_scale=1e-5,
                 downsampling='conv',
                 upsampling='conv',
                 base_filters=16,
                 depth=4):
        """ Initializes the variational autoencoder: consists of sampling
            then an alternating series of SENet blocks and upsampling.

            References:
                - [3D MRI brain tumor segmentation using autoencoder regularization](https://arxiv.org/pdf/1810.11654.pdf)
        """
        super(VariationalAutoencoder, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.data_format = data_format
        self.l2_scale = l2_scale
        self.config = super(VariationalAutoencoder, self).get_config()
        self.config.update({'groups': groups,
                            'reduction': reduction,
                            'downsampling': downsampling,
                            'upsampling': upsampling,
                            'base_filters': base_filters,
                            'depth': depth})

        # Retrieve downsampling method.
        Downsample = get_downsampling(downsampling)

        # Retrieve upsampling method.
        Upsample = get_upsampling(upsampling)

        # Extra downsampling layer to reduce parameters.
        self.downsample = Downsample(
                            filters=base_filters//2,
                            groups=groups,
                            data_format=data_format,
                            kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale))

        # Build sampling layers.
        self.flatten = tf.keras.layers.Flatten(data_format)
        self.proj = tf.keras.layers.Dense(
                            units=base_filters*(2**(depth-1)),
                            kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                            kernel_initializer='he_normal')
        self.latent_size = base_filters*(2**(depth-2))
        self.sample = tf.keras.layers.Lambda(sample)

        # Extra upsampling layer to counter extra downsampling layer.
        self.upsample = Upsample(
                            filters=base_filters*(2**(depth-1)),
                            groups=groups,
                            data_format=data_format,
                            l2_scale=l2_scale)

        # Build layers at all spatial levels.
        self.levels = []
        for i in range(depth-2, -1, -1):
            upsample = Upsample(
                        filters=base_filters*(2**i),
                        groups=groups,
                        data_format=data_format,
                        l2_scale=l2_scale)
            conv = ConvBlock(
                        filters=base_filters*(2**i),
                        groups=groups,
                        reduction=reduction,
                        data_format=data_format,
                        l2_scale=l2_scale)
            self.levels.append([upsample, conv])

        # Output layer convolution.
        self.out_conv = tf.keras.layers.Conv3D(
                            filters=IN_CH,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            data_format=data_format,
                            kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                            kernel_initializer='he_normal')

    def build(self, input_shape):
        h, w, d = input_shape[1:-1] if self.data_format == 'channels_last' else input_shape[2:]

        # Build reshaping layers after sampling.
        self.unproj = tf.keras.layers.Dense(
                                units=h*w*d*1//8,
                                kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_scale),
                                kernel_initializer='he_normal',
                                activation='relu')
        self.unflatten = tf.keras.layers.Reshape(
                                (h//2, w//2, d//2, 1) if self.data_format == 'channels_last' else (1, h//2, w//2, d//2))


    def call(self, inputs, training=None):
        # Downsample.
        inputs = self.downsample(inputs)

        # Flatten and project
        inputs = self.flatten(inputs)
        inputs = self.proj(inputs)

        # Sample.
        z_mean = inputs[:, :self.latent_size]
        z_logvar = inputs[:, self.latent_size:]
        inputs = self.sample([z_mean, z_logvar])

        # Restored projection and reshape
        inputs = self.unproj(inputs)
        inputs = self.unflatten(inputs)

        # Upsample.
        inputs = self.upsample(inputs)

        # Iterate through spatial levels.
        for level in self.levels:
            upsample, conv = level
            inputs = upsample(inputs, training=training)
            inputs = conv(inputs, training=training)

        # Map convolution to number of original input channels.
        inputs = self.out_conv(inputs)

        return inputs, z_mean, z_logvar

    def get_config(self):
        self.config.update({'data_format': self.data_format,
                            'l2_scale': self.l2_scale})
        return self.config
