"""Contains custom downsampling classes."""
import tensorflow as tf

from model.layer_utils.group_norm import GroupNormalization


def get_downsampling(downsampling):
    if downsampling == 'max':
        return MaxDownsample
    elif downsampling == 'conv':
        return ConvDownsample


class ConvDownsample(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 data_format='channels_last',
                 groups=8,
                 l2_scale=1e-5,
                 **kwargs):
        super(ConvDownsample, self).__init__()
        self.config = super(ConvDownsample, self).get_config()
        self.config.update({'filters': filters,
                            'data_format': data_format,
                            'groups': groups,
                            'l2_scale': l2_scale})

        self.conv = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=3,
                                strides=2,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                                kernel_initializer='he_normal')
        self.norm = GroupNormalization(
                        groups=groups,
                        axis=-1 if data_format == 'channels_last' else 1)
        self.relu = tf.keras.layers.Activation('relu')

    def __call__(self, inputs, training=None):
        inputs = self.conv(inputs)
        inputs = self.norm(inputs, training=training)
        inputs = self.relu(inputs)
        return inputs

    def get_config(self):
        return self.config


class MaxDownsample(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 **kwargs):
        super(MaxDownsample, self).__init__()
        self.config = super(MaxDownsample, self).get_config()
        self.config.update({'daat_format': data_format})

        self.maxpool = tf.keras.layers.MaxPooling3D(
                            pool_size=2,
                            strides=2,
                            padding='same',
                            data_format=data_format)

    def __call__(self, inputs, training=None):
        inputs = self.maxpool(inputs)
        return inputs

    def get_config(self):
        return self.config
