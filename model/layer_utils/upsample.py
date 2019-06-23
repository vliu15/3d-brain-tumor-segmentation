"""Contains custom upsampling classes."""
import tensorflow as tf

from model.layer_utils.group_norm import GroupNormalization


def get_upsampling(upsampling):
    if upsampling == 'linear':
        return LinearUpsample
    elif upsampling == 'conv':
        return ConvUpsample


class ConvUpsample(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 groups=8,
                 data_format='channels_last',
                 l2_scale=1e-5,
                 **kwargs):
        super(ConvUpsample, self).__init__()
        self.config = super(ConvUpsample, self).get_config()
        self.config.update({'filters': filters,
                            'data_format': data_format,
                            'groups': groups,
                            'l2_scale': l2_scale})

        self.conv = tf.keras.layers.Conv3DTranspose(
                            filters=filters,
                            kernel_size=3,
                            strides=2,
                            padding='same',
                            data_format=data_format)
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


class LinearUpsample(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 groups=8,
                 data_format='channels_last',
                 l2_scale=1e-5,
                 **kwargs):
        super(LinearUpsample, self).__init__()
        self.config = super(LinearUpsample, self).get_config()
        self.config.update({'filters': filters,
                            'groups': groups,
                            'data_format': data_format,
                            'l2_scale': l2_scale})

        self.ptwise = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                                kernel_initializer='he_normal')
        self.linear = tf.keras.layers.UpSampling3D(
                                size=2,
                                data_format=data_format)
        self.norm = GroupNormalization(
                            groups=groups,
                            axis=-1 if data_format == 'channels_last' else 1)
        self.relu = tf.keras.layers.Activation('relu')

    def __call__(self, inputs, training=None):
        inputs = self.ptwise(inputs)
        inputs = self.linear(inputs)
        inputs = self.norm(inputs)
        inputs = self.relu(inputs)
        return inputs

    def get_config(self):
        return self.config
