"""Contains convolutional sublayer and ResNet block classes."""
import tensorflow as tf

from model.layer_utils.squeeze_excitation import SqueezeExcitation
from model.layer_utils.group_norm import GroupNormalization


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 data_format='channels_last',
                 groups=8,
                 l2_scale=1e-5):
        """ Initializes one convolutional layer. Each layer
            is comprised of [Conv3D, GroupNorm, ReLU].
        """
        super(ConvLayer, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(ConvLayer, self).get_config()
        self.config.update({'filters': filters,
                            'data_format': data_format,
                            'groups': groups,
                            'l2_scale': l2_scale})

        self.conv3d = tf.keras.layers.Conv3D(
                            filters=filters,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            data_format=data_format,
                            kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                            kernel_initializer='he_normal')
        self.norm = GroupNormalization(
                        groups=groups,
                        axis=-1 if data_format == 'channels_last' else 1)
        self.relu = tf.keras.layers.Activation('relu')

    def call(self, inputs, training=None):
        inputs = self.conv3d(inputs)
        inputs = self.norm(inputs, training=training)
        inputs = self.relu(inputs)
        return inputs

    def get_config(self):
        return self.config


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 data_format='channels_last',
                 groups=8,
                 reduction=2,
                 l2_scale=1e-5):
        """ Initializes one SENet block. Builds on basic ResNet block
            structure, but applies squeeze-and-excitation to the residual.

            References:
                - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
        """
        super(ConvBlock, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(ConvBlock, self).get_config()
        self.config.update({'filters': filters,
                            'data_format': data_format,
                            'groups': groups,
                            'reduction': reduction,
                            'l2_scale': l2_scale})

        self.conv3d_ptwise = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                                kernel_initializer='he_normal')
        self.se_layer = SqueezeExcitation(
                                reduction=reduction,
                                data_format=data_format)
        self.conv_layer1 = ConvLayer(
                                filters=filters,
                                data_format=data_format,
                                groups=groups,
                                l2_scale=l2_scale)
        self.conv_layer2 = ConvLayer(
                                filters=filters,
                                data_format=data_format,
                                groups=groups,
                                l2_scale=l2_scale)
        self.residual = tf.keras.layers.Add()

    def call(self, inputs, training=None):
        inputs = self.conv3d_ptwise(inputs)
        res = self.se_layer(inputs, training=training)
        inputs = self.conv_layer1(inputs, training=training)
        inputs = self.conv_layer2(inputs, training=training)
        inputs = self.residual([res, inputs])
        return inputs

    def get_config(self):
        return self.config
